//! Which variables are black, and the checks that keep red-black plan repair simple.

use alloc::vec::Vec;
use core::fmt;

use sas_plus::{CausalGraph, Task, VarId};

/// A choice of black variables for [`crate::RedBlack`]; every other variable is red.
///
/// Black variables must be state variables that no axiom and no conditional effect mentions
/// (as condition or as effect), and no two may be linked in the causal graph. So an operator
/// changes at most one black variable, and only its own black variable appears in its
/// preconditions: moving one black variable never disturbs another, which is what lets the
/// repair insert a path in one variable's domain transition graph at a time.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Painting {
    /// The black variables, ascending.
    black: Vec<VarId>,
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
        let excluded = ineligible(task);
        let mut candidates: Vec<VarId> = (0..task.num_variables())
            .filter(|&v| excluded[v].is_none() && strongly_connected(task, v))
            .collect();
        candidates.sort_by_key(|&v| (task.variables[v].domain_size, v));

        let graph = CausalGraph::new(task);
        let mut black: Vec<VarId> = Vec::new();
        for v in candidates {
            if !black.iter().any(|&u| linked(&graph, u, v)) {
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
