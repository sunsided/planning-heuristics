//! Which variables are black, and the checks that keep red-black plan repair simple.

use alloc::vec::Vec;
use core::fmt;

use sas_plus::{CausalGraph, State, Task, VarId};

use crate::Relaxation;

/// A choice of black variables for [`crate::RedBlack`]; every other variable is red.
///
/// Black variables must be state variables that no axiom and no conditional effect mentions
/// (as condition or as effect), and no two may be linked in the causal graph. So an operator
/// changes at most one black variable, and an operator that changes a black variable has no
/// precondition on another: moving one black variable never disturbs another, which is what
/// lets the repair insert a path in one variable's domain transition graph at a time. An
/// operator that changes only red variables may still require several black values; the
/// repair moves each in turn.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Painting {
    /// The black variables, ascending.
    black: Vec<VarId>,
}

/// Which end of the causal graph [`Painting::causal_graph_level`] paints first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CausalLevel {
    /// Variables that others depend on first: the sources of the causal graph, such as the
    /// position of a vehicle that loading and unloading operators need.
    Top,
    /// Variables that depend on others first: the sinks, such as the positions of packages.
    Bottom,
}

/// Why a set of variables cannot be painted black; see [`Painting::new`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PaintingError {
    /// The variable does not exist in the task.
    UnknownVariable(VarId),
    /// The variable is derived; derived variables stay red.
    Derived(VarId),
    /// An axiom has a condition on the variable.
    InAxiom(VarId),
    /// A conditional effect sets the variable or has a condition on it.
    ConditionalEffect(VarId),
    /// The two black variables are linked in the causal graph.
    Dependent(VarId, VarId),
}

impl fmt::Display for PaintingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::UnknownVariable(v) => write!(f, "variable {v} does not exist"),
            Self::Derived(v) => write!(f, "variable {v} is derived"),
            Self::InAxiom(v) => write!(f, "variable {v} appears in an axiom"),
            Self::ConditionalEffect(v) => {
                write!(f, "variable {v} appears in a conditional effect")
            }
            Self::Dependent(u, v) => {
                write!(f, "variables {u} and {v} are linked in the causal graph")
            }
        }
    }
}

impl core::error::Error for PaintingError {}

impl Painting {
    /// All variables red: red-black planning degenerates to the delete relaxation, and
    /// [`crate::RedBlack`] equals `h_FF`.
    #[must_use]
    pub fn all_red() -> Self {
        Self::default()
    }

    /// Paints exactly `black` black, after checking each variable is eligible and no two are
    /// linked in the causal graph.
    ///
    /// # Errors
    ///
    /// - [`PaintingError::UnknownVariable`] if a variable is not in `task`;
    /// - [`PaintingError::Derived`], [`PaintingError::InAxiom`] or
    ///   [`PaintingError::ConditionalEffect`] if a variable is not eligible;
    /// - [`PaintingError::Dependent`] for the first two linked variables.
    pub fn new(task: &Task, mut black: Vec<VarId>) -> Result<Self, PaintingError> {
        black.sort_unstable();
        black.dedup();
        let excluded = ineligible(task);
        for &v in &black {
            if v >= task.num_variables() {
                return Err(PaintingError::UnknownVariable(v));
            }

            if let Some(reason) = excluded[v] {
                return Err(reason);
            }
        }

        let graph = CausalGraph::new(task);
        for (i, &u) in black.iter().enumerate() {
            if let Some(&v) = black[i + 1..].iter().find(|&&v| linked(&graph, u, v)) {
                return Err(PaintingError::Dependent(u, v));
            }
        }

        Ok(Self { black })
    }

    /// A greedy painting: among the eligible variables whose domain transition graph is
    /// strongly connected (every value can reach every other, ignoring the conditions on
    /// other variables), paint black in order of ascending domain size, then variable id,
    /// each one not linked in the causal graph to one painted before.
    ///
    /// Strongly connected domain transition graphs make it likely that the repair can move a
    /// black variable back to a value it left.
    #[must_use]
    pub fn greedy(task: &Task) -> Self {
        let mut candidates = repairable(task);
        candidates.sort_by_key(|&v| (task.variables[v].domain_size, v));

        Self::independent(&CausalGraph::new(task), candidates)
    }

    /// A painting by causal graph level (after Katz, Hoffmann & Domshlak 2013): among the
    /// eligible variables with a strongly connected domain transition graph, paint black from
    /// the `start` end of the causal graph, each one not linked to one painted before.
    ///
    /// The level of a variable is the length of the longest path to it from a source of the
    /// condensed causal graph (its strongly connected components); ties go to the lower
    /// variable id. Variables high in the graph tend to be the ones whose real semantics the
    /// relaxation loses most (a truck that "is everywhere at once").
    #[must_use]
    pub fn causal_graph_level(task: &Task, start: CausalLevel) -> Self {
        let graph = CausalGraph::new(task);
        let level = levels(task.num_variables(), |v| graph.successors(v));
        let mut candidates = repairable(task);
        match start {
            CausalLevel::Top => candidates.sort_by_key(|&v| (level[v], v)),
            CausalLevel::Bottom => candidates.sort_by_key(|&v| (core::cmp::Reverse(level[v]), v)),
        }

        Self::independent(&graph, candidates)
    }

    /// A conflict-directed painting (after Domshlak, Hoffmann & Katz 2015): paint black the
    /// variables whose real semantics the relaxed plan of `state` violates most often.
    ///
    /// The `h_FF` relaxed plan of `state` is replayed with each variable's real value; a
    /// *conflict* of a variable is an operator of the plan whose precondition on it differs
    /// from its current value (the value is then assumed moved), or a goal on it that differs
    /// at the end. Among the eligible variables with a strongly connected domain transition
    /// graph and at least one conflict, paint black by descending conflicts (then variable
    /// id), each one not linked in the causal graph to one painted before. All red if `state`
    /// is a dead end or nothing conflicts.
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    #[must_use]
    pub fn conflict_directed(task: &Task, state: &State) -> Self {
        let Some(plan) = Relaxation::new(task).relaxed_plan(state) else {
            return Self::all_red();
        };

        let conflicts = conflicts(task, state, plan.ops());
        let mut candidates: Vec<VarId> = repairable(task)
            .into_iter()
            .filter(|&v| conflicts[v] > 0)
            .collect();
        candidates.sort_by_key(|&v| (core::cmp::Reverse(conflicts[v]), v));

        Self::independent(&CausalGraph::new(task), candidates)
    }

    /// Paints `candidates` black in order, skipping each one linked in `graph` to one painted
    /// before.
    fn independent(graph: &CausalGraph, candidates: Vec<VarId>) -> Self {
        let mut black: Vec<VarId> = Vec::new();
        for v in candidates {
            if !black.iter().any(|&u| linked(graph, u, v)) {
                black.push(v);
            }
        }

        black.sort_unstable();
        Self { black }
    }

    /// The black variables, ascending.
    #[must_use]
    pub fn black(&self) -> &[VarId] {
        &self.black
    }
}

/// For each variable, why it cannot be black; `None` if it can.
fn ineligible(task: &Task) -> Vec<Option<PaintingError>> {
    let mut reason: Vec<Option<PaintingError>> = (0..task.num_variables())
        .map(|v| {
            task.variables[v]
                .is_derived()
                .then_some(PaintingError::Derived(v))
        })
        .collect();
    let mut exclude = |v: VarId, why: PaintingError| {
        reason[v].get_or_insert(why);
    };
    for axiom in &task.axioms {
        for c in &axiom.conditions {
            exclude(c.var, PaintingError::InAxiom(c.var));
        }
    }

    for effect in task.operators.iter().flat_map(|op| &op.eff) {
        if !effect.conditions.is_empty() {
            let v = effect.fact.var;
            exclude(v, PaintingError::ConditionalEffect(v));
            for c in &effect.conditions {
                exclude(c.var, PaintingError::ConditionalEffect(c.var));
            }
        }
    }

    reason
}

/// The eligible variables whose domain transition graph is strongly connected, ascending.
fn repairable(task: &Task) -> Vec<VarId> {
    let excluded = ineligible(task);
    (0..task.num_variables())
        .filter(|&v| excluded[v].is_none() && strongly_connected(task, v))
        .collect()
}

/// For each of the `n` nodes of a graph with `successors`, the length of the longest path to
/// it from a source of the graph's condensation (each strongly connected component counts as
/// one node).
fn levels<'g>(n: usize, successors: impl Fn(usize) -> &'g [usize]) -> Vec<usize> {
    let component = components(n, &successors);
    let count = component.iter().max().map_or(0, |&c| c + 1);
    let mut arcs: Vec<(usize, usize)> = (0..n)
        .flat_map(|u| successors(u).iter().map(move |&v| (u, v)))
        .map(|(u, v)| (component[u], component[v]))
        .filter(|(a, b)| a != b)
        .collect();
    arcs.sort_unstable();
    arcs.dedup();

    // Longest path by relaxation in topological order; `components` numbers the
    // components in reverse topological order, so sources have the highest numbers.
    let mut level = alloc::vec![0; count];
    for c in (0..count).rev() {
        for &(a, b) in arcs.iter().filter(|(a, _)| *a == c) {
            level[b] = level[b].max(level[a] + 1);
        }
    }

    component.iter().map(|&c| level[c]).collect()
}

/// The strongly connected component of each of the `n` nodes (Tarjan), numbered in reverse
/// topological order of the condensation: an arc between components goes from a higher to a
/// lower number.
fn components<'g>(n: usize, successors: &impl Fn(usize) -> &'g [usize]) -> Vec<usize> {
    /// Tarjan's bookkeeping.
    struct Tarjan {
        /// Discovery index per node; `usize::MAX` if unvisited.
        index: Vec<usize>,
        /// Lowest reachable discovery index on the stack per node.
        low: Vec<usize>,
        /// Whether each node is on the stack.
        on_stack: Vec<bool>,
        /// The Tarjan stack.
        stack: Vec<usize>,
        /// Component per node; `usize::MAX` until assigned.
        component: Vec<usize>,
        /// Next discovery index.
        next: usize,
        /// Next component number.
        count: usize,
    }

    let mut t = Tarjan {
        index: alloc::vec![usize::MAX; n],
        low: alloc::vec![0; n],
        on_stack: alloc::vec![false; n],
        stack: Vec::new(),
        component: alloc::vec![usize::MAX; n],
        next: 0,
        count: 0,
    };
    for root in 0..n {
        if t.index[root] != usize::MAX {
            continue;
        }

        // Iterative DFS: (node, next successor position).
        let mut work = alloc::vec![(root, 0)];
        t.index[root] = t.next;
        t.low[root] = t.next;
        t.next += 1;
        t.stack.push(root);
        t.on_stack[root] = true;
        while let Some(&mut (u, ref mut pos)) = work.last_mut() {
            if let Some(&v) = successors(u).get(*pos) {
                *pos += 1;
                if t.index[v] == usize::MAX {
                    t.index[v] = t.next;
                    t.low[v] = t.next;
                    t.next += 1;
                    t.stack.push(v);
                    t.on_stack[v] = true;
                    work.push((v, 0));
                } else if t.on_stack[v] {
                    t.low[u] = t.low[u].min(t.index[v]);
                }

                continue;
            }

            work.pop();
            if let Some(&(parent, _)) = work.last() {
                t.low[parent] = t.low[parent].min(t.low[u]);
            }

            if t.low[u] == t.index[u] {
                while let Some(w) = t.stack.pop() {
                    t.on_stack[w] = false;
                    t.component[w] = t.count;
                    if w == u {
                        break;
                    }
                }

                t.count += 1;
            }
        }
    }

    t.component
}

/// Per variable, the conflicts of the relaxed `plan` from `state`; see
/// [`Painting::conflict_directed`].
fn conflicts(task: &Task, state: &State, plan: &[sas_plus::OpId]) -> Vec<usize> {
    let mut value = state.0.clone();
    let mut conflicts = alloc::vec![0; task.num_variables()];
    for &op in plan {
        let operator = &task.operators[op];
        for f in &operator.pre {
            if value[f.var] != f.value {
                conflicts[f.var] += 1;
                value[f.var] = f.value;
            }
        }

        for e in operator.eff.iter().filter(|e| e.conditions.is_empty()) {
            value[e.fact.var] = e.fact.value;
        }
    }

    for g in &task.goal {
        conflicts[g.var] += usize::from(value[g.var] != g.value);
    }

    conflicts
}

/// Whether `u` and `v` are linked by an arc of the causal graph in either direction.
fn linked(graph: &CausalGraph, u: VarId, v: VarId) -> bool {
    graph.successors(u).binary_search(&v).is_ok() || graph.successors(v).binary_search(&u).is_ok()
}

/// Whether every value of `var` can reach every other through operator effects on `var`,
/// ignoring all conditions on other variables.
fn strongly_connected(task: &Task, var: VarId) -> bool {
    let n = task.variables[var].domain_size as usize;
    let mut arcs: Vec<(usize, usize)> = Vec::new();
    for op in &task.operators {
        let pre = op
            .pre
            .iter()
            .find(|f| f.var == var)
            .map(|f| f.value as usize);
        for e in op.eff.iter().filter(|e| e.fact.var == var) {
            let to = e.fact.value as usize;
            match pre {
                Some(from) => arcs.push((from, to)),
                None => arcs.extend((0..n).map(|from| (from, to))),
            }
        }
    }

    let forward = reach(n, &arcs, false);
    let backward = reach(n, &arcs, true);

    forward.iter().chain(&backward).all(|&r| r)
}

/// Which of the `n` nodes value 0 reaches along `arcs` (`reverse`: against them).
fn reach(n: usize, arcs: &[(usize, usize)], reverse: bool) -> Vec<bool> {
    let mut seen = alloc::vec![false; n];
    let mut stack = alloc::vec![0];
    seen[0] = true;
    while let Some(x) = stack.pop() {
        for &(a, b) in arcs {
            let (from, to) = if reverse { (b, a) } else { (a, b) };
            if from == x && !seen[to] {
                seen[to] = true;
                stack.push(to);
            }
        }
    }

    seen
}

#[cfg(test)]
mod tests {
    //! The causal graph levels on small graphs.

    use alloc::vec::Vec;

    use super::levels;

    /// Levels of a graph given as successor lists.
    fn of(successors: &[&[usize]]) -> Vec<usize> {
        levels(successors.len(), |v| successors[v])
    }

    /// A chain has levels 0, 1, 2; a cycle collapses to one level; isolated nodes are sources.
    #[test]
    fn levels_of_chains_and_cycles() {
        assert_eq!(of(&[&[1], &[2], &[]]), [0, 1, 2]);
        assert_eq!(of(&[&[1], &[2], &[1], &[]]), [0, 1, 1, 0]);
        assert_eq!(of(&[&[2], &[2], &[3], &[]]), [0, 0, 1, 2]);
        assert_eq!(of(&[&[1, 2], &[3], &[], &[]]), [0, 1, 1, 2]);
    }
}
