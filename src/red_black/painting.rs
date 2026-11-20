//! Which variables are black, and the checks that keep red-black plan repair simple.

mod invertible;
mod strategy;

use alloc::vec::Vec;
use core::fmt;

use sas_plus::{CausalGraph, State, Task, VarId};

use invertible::is_rse_invertible;
pub use strategy::Strategy;

/// A choice of black variables for [`crate::RedBlack`]; every other variable is red.
///
/// Black variables must be state variables that no axiom and no conditional effect mentions
/// (as condition or as effect), and, for [`Painting::new`], no two may be linked in the
/// causal graph ([`Painting::dag`] allows an acyclic causal graph on them). So an operator
/// changes at most one black variable, and an operator that changes a black variable has no
/// precondition on another: moving one black variable never disturbs another, which is what
/// lets the repair insert a path in one variable's domain transition graph at a time. An
/// operator that changes only red variables may still require several black values; the
/// repair moves each in turn.
///
/// The repair is guaranteed to succeed (never to fall back to `h_FF`) when every black
/// variable is also RSE-invertible (see [`Painting::greedy`]): this is the tractable fragment
/// of Katz, Hoffmann & Domshlak (AAAI 2013, Corollary 2 and Theorem 2) and Domshlak, Hoffmann
/// & Katz (AIJ 2015, Theorem 11). [`Painting::new`] and [`Painting::dag`] do not check it; every
/// strategy constructor paints only RSE-invertible variables.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Painting {
    /// The black variables, ascending.
    black: Vec<VarId>,
}

/// Which end of the causal graph [`Painting::causal_graph_level`] keeps black.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CausalLevel {
    /// Keep the variables that others depend on black, such as the position of a vehicle
    /// that loading and unloading operators need: paint red from the leaves up (strategy L of
    /// Domshlak, Hoffmann & Katz 2015).
    Top,
    /// Keep the variables near the leaves black: paint red from the roots down (their inverse
    /// strategy, a sanity check). The leaves themselves are always red.
    Bottom,
}

/// The tractable fragment a painting strategy targets: the shape of the causal graph induced
/// on the black variables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fragment {
    /// No arc between two black variables, as [`Painting::new`] requires (Katz, Hoffmann &
    /// Domshlak, AAAI 2013, Corollary 2; configuration E of Domshlak, Hoffmann & Katz 2015).
    Arcless,
    /// An acyclic black causal graph, as [`Painting::dag`] requires (AAAI 2013, Theorem 2;
    /// AIJ 2015, Theorem 11; configuration D).
    Dag,
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
    /// The causal graph on the black variables has a cycle through these two (one arc of it);
    /// see [`Painting::dag`].
    Cyclic(VarId, VarId),
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
            Self::Cyclic(u, v) => {
                write!(
                    f,
                    "variables {u} and {v} lie on a cycle of the black causal graph"
                )
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

    /// Paints exactly `black` black, allowing causal-graph arcs between black variables as long
    /// as the causal graph on them is acyclic (the DAG fragment of Katz, Hoffmann & Domshlak
    /// 2013). An operator then changes at most one black variable (two would link them both
    /// ways), and the repair moves a black variable's causal-graph ancestors on demand before
    /// each transition that needs them, descendants first, so a move never disturbs a value
    /// already in place below it (Figure 7 of Domshlak, Hoffmann & Katz 2015). With
    /// RSE-invertible black variables this always succeeds, but the plan may be exponentially
    /// long in the number of black variables (their Proposition 2); the repair has an operator
    /// budget and falls back to `h_FF` when it runs out, or when a variable is not invertible
    /// and cannot move back.
    ///
    /// # Errors
    ///
    /// - [`PaintingError::UnknownVariable`], [`PaintingError::Derived`],
    ///   [`PaintingError::InAxiom`] or [`PaintingError::ConditionalEffect`] as for
    ///   [`Painting::new`];
    /// - [`PaintingError::Cyclic`] for an arc on a cycle of the black causal graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use planning_heuristics::{Painting, PaintingError};
    /// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
    ///
    /// // Moving `b` needs `a = 1`: an arc a → b, fine for a DAG painting.
    /// let task = Task {
    ///     variables: vec![Variable::new("a", 2), Variable::new("b", 2)],
    ///     operators: vec![
    ///         Operator {
    ///             name: "a".into(),
    ///             pre: vec![],
    ///             eff: vec![Effect::new(Fact::new(0, 1))],
    ///             cost: 1,
    ///         },
    ///         Operator {
    ///             name: "b".into(),
    ///             pre: vec![Fact::new(0, 1)],
    ///             eff: vec![Effect::new(Fact::new(1, 1))],
    ///             cost: 1,
    ///         },
    ///     ],
    ///     initial: State(vec![0, 0]),
    ///     goal: vec![Fact::new(1, 1)],
    ///     axioms: vec![],
    /// };
    ///
    /// assert_eq!(Painting::new(&task, vec![0, 1]), Err(PaintingError::Dependent(0, 1)));
    /// assert_eq!(Painting::dag(&task, vec![0, 1]).unwrap().black(), [0, 1]);
    /// ```
    pub fn dag(task: &Task, mut black: Vec<VarId>) -> Result<Self, PaintingError> {
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

        match cycle_arc(&CausalGraph::new(task), &black) {
            Some((u, v)) => Err(PaintingError::Cyclic(u, v)),
            None => Ok(Self { black }),
        }
    }

    /// A greedy DAG painting of this crate (not one of the papers'): the candidates and order
    /// of [`Painting::greedy`], each painted black unless it closes a cycle in the causal graph
    /// on the variables painted before. Valid for [`Painting::dag`].
    #[must_use]
    pub fn greedy_dag(task: &Task) -> Self {
        let mut candidates = repairable(task);
        candidates.sort_by_key(|&v| (task.variables[v].domain_size, v));
        let graph = CausalGraph::new(task);
        let mut black: Vec<VarId> = Vec::new();
        for v in candidates {
            let pos = black.binary_search(&v).unwrap_err();
            black.insert(pos, v);
            if cycle_arc(&graph, &black).is_some() {
                black.remove(pos);
            }
        }

        Self { black }
    }

    /// A greedy painting of this crate (not one of the papers'): among the eligible
    /// RSE-invertible variables, paint black in order of ascending domain size, then variable
    /// id, each one not linked in the causal graph to one painted before. The repair of such
    /// a painting never falls back to `h_FF`.
    ///
    /// A variable is RSE-invertible (relaxed side effects invertible; Katz, Hoffmann &
    /// Domshlak 2013) if every arc `d → d'` of its domain transition graph has an inverse arc
    /// `d' → d` whose conditions on other variables are among the conditions and effects of
    /// the original arc. Moving a black variable back is then always possible once it moved.
    #[must_use]
    pub fn greedy(task: &Task) -> Self {
        let mut candidates = repairable(task);
        candidates.sort_by_key(|&v| (task.variables[v].domain_size, v));

        Self::independent(&CausalGraph::new(task), candidates)
    }

    /// The causal graph level strategy (L, or its inverse) of Domshlak, Hoffmann & Katz
    /// (2015, Section 6.1), for `fragment`.
    ///
    /// Follows their Figure 8: the variables that are ineligible, not RSE-invertible (see
    /// [`Painting::greedy`]) or leaves of the causal graph start red; while the black causal
    /// graph is not in `fragment`, the black variable with the highest causal graph level
    /// (closest to the leaves) is painted red, or with [`CausalLevel::Bottom`] the one with the
    /// lowest; finally the variables painted red are tried black again, last first, and kept
    /// black if the graph stays in `fragment`. The result is maximal.
    ///
    /// The level of a variable is the length of the longest path to it from a source of the
    /// condensed causal graph (its strongly connected components); ties go to the higher
    /// variable id for "highest" and the lower for "lowest", which follows the causal graph
    /// order a Fast Downward translation numbers its variables in.
    ///
    /// # Examples
    ///
    /// ```
    /// use planning_heuristics::{CausalLevel, Fragment, Painting};
    /// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
    ///
    /// // A truck (variable 0) carries a package (variable 1: at a, at b, in the truck) from a
    /// // to b and must end at a again.
    /// let op = |name: &str, pre: Vec<Fact>, eff: Fact| Operator {
    ///     name: name.into(),
    ///     pre,
    ///     eff: vec![Effect::new(eff)],
    ///     cost: 1,
    /// };
    /// let (t, p) = (|v| Fact::new(0, v), |v| Fact::new(1, v));
    /// let task = Task {
    ///     variables: vec![Variable::new("truck", 2), Variable::new("package", 3)],
    ///     operators: vec![
    ///         op("drive-a-b", vec![t(0)], t(1)),
    ///         op("drive-b-a", vec![t(1)], t(0)),
    ///         op("load-a", vec![t(0), p(0)], p(2)),
    ///         op("load-b", vec![t(1), p(1)], p(2)),
    ///         op("unload-a", vec![t(0), p(2)], p(0)),
    ///         op("unload-b", vec![t(1), p(2)], p(1)),
    ///     ],
    ///     initial: State(vec![0, 0]),
    ///     goal: vec![p(1), t(0)],
    ///     axioms: vec![],
    /// };
    ///
    /// // The package is a leaf of the causal graph, so it is red under every strategy.
    /// let top = Painting::causal_graph_level(&task, CausalLevel::Top, Fragment::Arcless);
    /// assert_eq!(top.black(), [0]);
    /// ```
    #[must_use]
    pub fn causal_graph_level(task: &Task, keep: CausalLevel, fragment: Fragment) -> Self {
        Self::with_strategy(task, &task.initial_state(), Strategy::Level(keep), fragment)
    }

    /// The incident arcs strategy (A) of Domshlak, Hoffmann & Katz (2015, Section 6.1; also
    /// Katz, Hoffmann & Domshlak, AAAI 2013), for `fragment`: Figure 8 as in
    /// [`Painting::causal_graph_level`], painting red the black variable with the most causal
    /// graph arcs to other black variables (each direction counts), ties to the smaller
    /// domain, then to strategy L. This aims to paint few variables red.
    ///
    /// # Examples
    ///
    /// ```
    /// use planning_heuristics::{Fragment, Painting};
    /// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
    ///
    /// // A hub (variable 0) both needs and is needed by spokes 1 and 2; the hub has the most
    /// // arcs (four) and turns red, the spokes are not linked and stay black.
    /// let flip = |var: usize, from: u32, need: Vec<Fact>| {
    ///     let mut pre = vec![Fact::new(var, from)];
    ///     pre.extend(need);
    ///     Operator {
    ///         name: format!("flip{var}-{from}"),
    ///         pre,
    ///         eff: vec![Effect::new(Fact::new(var, 1 - from))],
    ///         cost: 1,
    ///     }
    /// };
    /// let mut operators = Vec::new();
    /// for from in 0..2 {
    ///     operators.push(flip(0, from, vec![Fact::new(1, 0), Fact::new(2, 0)]));
    ///     operators.push(flip(1, from, vec![Fact::new(0, 0)]));
    ///     operators.push(flip(2, from, vec![Fact::new(0, 0)]));
    /// }
    ///
    /// let task = Task {
    ///     variables: (0..3).map(|i| Variable::new(format!("v{i}"), 2)).collect(),
    ///     operators,
    ///     initial: State(vec![0, 0, 0]),
    ///     goal: vec![Fact::new(1, 1)],
    ///     axioms: vec![],
    /// };
    ///
    /// assert_eq!(Painting::incident_arcs(&task, Fragment::Arcless).black(), [1, 2]);
    /// ```
    #[must_use]
    pub fn incident_arcs(task: &Task, fragment: Fragment) -> Self {
        Self::with_strategy(task, &task.initial_state(), Strategy::MostArcs, fragment)
    }

    /// The conflict-directed strategy (C) of Domshlak, Hoffmann & Katz (2015, Section 6.1;
    /// also Katz, Hoffmann & Domshlak, AAAI 2013), for `fragment`: Figure 8 as in
    /// [`Painting::causal_graph_level`], painting red the black variable with the fewest
    /// conflicts, ties to strategy L, so the most critical variables stay black.
    ///
    /// A *conflict* of a variable is an action of the `h_FF` relaxed plan of `state` (the
    /// papers use the initial state) whose precondition on it is violated when the plan is
    /// executed with only that variable black; the value is then assumed moved to the
    /// precondition. If `state` is a dead end, every variable has no conflicts and the
    /// strategy is L.
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    ///
    /// # Examples
    ///
    /// ```
    /// use planning_heuristics::{Fragment, HFf, Painting, RedBlack};
    /// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
    ///
    /// // A truck (variable 0) carries a package (variable 1: at a, at b, in the truck) from a
    /// // to b and must end at a again.
    /// let op = |name: &str, pre: Vec<Fact>, eff: Fact| Operator {
    ///     name: name.into(),
    ///     pre,
    ///     eff: vec![Effect::new(eff)],
    ///     cost: 1,
    /// };
    /// let (t, p) = (|v| Fact::new(0, v), |v| Fact::new(1, v));
    /// let task = Task {
    ///     variables: vec![Variable::new("truck", 2), Variable::new("package", 3)],
    ///     operators: vec![
    ///         op("drive-a-b", vec![t(0)], t(1)),
    ///         op("drive-b-a", vec![t(1)], t(0)),
    ///         op("load-a", vec![t(0), p(0)], p(2)),
    ///         op("load-b", vec![t(1), p(1)], p(2)),
    ///         op("unload-a", vec![t(0), p(2)], p(0)),
    ///         op("unload-b", vec![t(1), p(2)], p(1)),
    ///     ],
    ///     initial: State(vec![0, 0]),
    ///     goal: vec![p(1), t(0)],
    ///     axioms: vec![],
    /// };
    /// let state = task.initial_state();
    ///
    /// // The truck stays black and must really drive back to a.
    /// let painting = Painting::conflict_directed(&task, &state, Fragment::Arcless);
    /// assert_eq!(painting.black(), [0]);
    /// assert_eq!(HFf::new(&task).value(&state), Some(3));
    /// assert_eq!(RedBlack::with_painting(&task, &painting).value(&state), Some(4));
    /// ```
    #[must_use]
    pub fn conflict_directed(task: &Task, state: &State, fragment: Fragment) -> Self {
        Self::with_strategy(task, state, Strategy::FewestConflicts, fragment)
    }

    /// The painting of `task` for `fragment` by `strategy`, following Figure 8 of Domshlak,
    /// Hoffmann & Katz (2015) as described at [`Painting::causal_graph_level`]; the conflict
    /// strategies use the relaxed plan of `state` (the paper: the initial state).
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables, or
    /// [`Strategy::Mixed`] has a weight above 100.
    ///
    /// # Examples
    ///
    /// ```
    /// use planning_heuristics::{Fragment, Painting, Strategy};
    /// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
    ///
    /// // Two switches, each needing the other off to turn on: a causal graph cycle. Every
    /// // strategy paints exactly one of them red for a DAG.
    /// let flip = |var: usize, from: u32, other: usize| Operator {
    ///     name: format!("flip{var}"),
    ///     pre: vec![Fact::new(var, from), Fact::new(other, 0)],
    ///     eff: vec![Effect::new(Fact::new(var, 1 - from))],
    ///     cost: 1,
    /// };
    /// let task = Task {
    ///     variables: vec![Variable::new("a", 2), Variable::new("b", 2)],
    ///     operators: vec![flip(0, 0, 1), flip(0, 1, 1), flip(1, 0, 0), flip(1, 1, 0)],
    ///     initial: State(vec![0, 0]),
    ///     goal: vec![Fact::new(0, 1)],
    ///     axioms: vec![],
    /// };
    /// let state = task.initial_state();
    ///
    /// for strategy in [Strategy::FewestArcs, Strategy::Random { seed: 7 }] {
    ///     let painting = Painting::with_strategy(&task, &state, strategy, Fragment::Dag);
    ///     assert_eq!(painting.black().len(), 1);
    /// }
    /// ```
    #[must_use]
    pub fn with_strategy(
        task: &Task,
        state: &State,
        strategy: Strategy,
        fragment: Fragment,
    ) -> Self {
        Self {
            black: strategy::paint_with(task, state, strategy, fragment),
        }
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

/// The eligible RSE-invertible variables, ascending.
fn repairable(task: &Task) -> Vec<VarId> {
    let excluded = ineligible(task);
    (0..task.num_variables())
        .filter(|&v| excluded[v].is_none() && is_rse_invertible(task, v))
        .collect()
}

/// The variable of `candidates` (not empty) with the largest `key`.
fn max_by_key<K: Ord>(candidates: &[VarId], key: impl Fn(VarId) -> K) -> VarId {
    candidates
        .iter()
        .copied()
        .max_by_key(|&v| key(v))
        .expect("there are candidates")
}

/// For each of the `n` nodes of a graph with `successors`, the length of the longest path to
/// it from a source of the graph's condensation (each strongly connected component counts as
/// one node).
pub(super) fn levels<'g>(n: usize, successors: impl Fn(usize) -> &'g [usize]) -> Vec<usize> {
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

/// An arc `(u, v)` of the causal graph between variables of `black` (ascending) that lies on
/// a cycle of the graph induced on them; `None` if that graph is acyclic.
fn cycle_arc(graph: &CausalGraph, black: &[VarId]) -> Option<(VarId, VarId)> {
    let successors: Vec<Vec<usize>> = black
        .iter()
        .map(|&v| {
            graph
                .successors(v)
                .iter()
                .filter_map(|w| black.binary_search(w).ok())
                .collect()
        })
        .collect();
    let component = components(black.len(), &|b| successors[b].as_slice());

    (0..black.len())
        .flat_map(|u| successors[u].iter().map(move |&v| (u, v)))
        .find(|&(u, v)| component[u] == component[v])
        .map(|(u, v)| (black[u], black[v]))
}

/// Whether `u` and `v` are linked by an arc of the causal graph in either direction.
fn linked(graph: &CausalGraph, u: VarId, v: VarId) -> bool {
    graph.successors(u).binary_search(&v).is_ok() || graph.successors(v).binary_search(&u).is_ok()
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
