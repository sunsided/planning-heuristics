//! Abstraction heuristics: goal distances of an explicit abstraction, looked up per state.
//!
//! [`AbstractionHeuristic`] covers any [`sas_plus::Abstraction`], such as a
//! [`sas_plus::Projection`] (a pattern database). With the `merge-and-shrink` and
//! `cost-partition` features, the heuristics of those crates implement [`Heuristic`] directly.

use alloc::vec::Vec;

use sas_plus::{Abstraction, Projection, State, Task, VarId};

use crate::Heuristic;

/// The goal distances of an abstraction as a heuristic: admissible, since every transition of
/// the task maps to a path of at most its cost in the abstraction.
///
/// `None` means the abstract state of the state is a dead end, or that the abstraction
/// pruned it. Projections never prune. An abstraction that prunes unreachable states (such
/// as merge-and-shrink with `prune: true`) is only sound for states reachable from the
/// initial state, which is all a forward search ever evaluates.
///
/// # Examples
///
/// ```
/// use planning_heuristics::{AbstractionHeuristic, Heuristic};
/// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
///
/// // Two independent switches, each set by an operator of cost 2 and 3.
/// let set = |var: usize, cost: u32| Operator {
///     name: format!("set-{var}"),
///     pre: vec![],
///     eff: vec![Effect::new(Fact::new(var, 1))],
///     cost,
/// };
/// let task = Task {
///     variables: vec![Variable::new("a", 2), Variable::new("b", 2)],
///     operators: vec![set(0, 2), set(1, 3)],
///     initial: State(vec![0, 0]),
///     goal: vec![Fact::new(0, 1), Fact::new(1, 1)],
///     axioms: vec![],
/// };
/// let state = task.initial_state();
///
/// // The pattern database of `b` alone sees only the cost of setting `b`.
/// assert_eq!(AbstractionHeuristic::projection(&task, vec![1]).estimate(&state), Some(3));
/// // The full pattern is exact.
/// assert_eq!(AbstractionHeuristic::projection(&task, vec![0, 1]).estimate(&state), Some(5));
/// ```
#[derive(Clone, Debug)]
pub struct AbstractionHeuristic<A> {
    /// The abstraction that maps states to abstract states.
    abstraction: A,
    /// Goal distance of each abstract state; `None` for abstract dead ends.
    distances: Vec<Option<u64>>,
}

impl<A: Abstraction> AbstractionHeuristic<A> {
    /// Computes the goal distances of `abstraction` under the operator `costs` (one per
    /// label, as from [`Task::costs`]).
    ///
    /// Pass a cost partition's share instead of the full costs to sum several abstraction
    /// heuristics admissibly.
    ///
    /// # Panics
    ///
    /// Panics if `costs` has fewer entries than the abstraction has labels.
    #[must_use]
    pub fn new(abstraction: A, costs: &[u64]) -> Self {
        let distances = abstraction.lts().goal_distances(costs);

        Self {
            abstraction,
            distances,
        }
    }

    /// The abstraction.
    #[must_use]
    pub fn abstraction(&self) -> &A {
        &self.abstraction
    }

    /// The goal distance of the abstract state of `state`; `None` if it is an abstract dead
    /// end or pruned.
    #[must_use]
    pub fn value(&self, state: &State) -> Option<u64> {
        self.abstraction
            .abstract_state(state)
            .and_then(|s| self.distances[s])
    }
}

impl AbstractionHeuristic<Projection> {
    /// The pattern database of `pattern` over `task` with the task's operator costs.
    /// Derived variables in `pattern` are ignored (see [`Projection`]).
    ///
    /// # Panics
    ///
    /// Panics if `pattern` names a variable `task` does not have.
    #[must_use]
    pub fn projection(task: &Task, pattern: Vec<VarId>) -> Self {
        Self::new(Projection::new(task, pattern), &task.costs())
    }
}

impl<A: Abstraction> Heuristic<State> for AbstractionHeuristic<A> {
    fn estimate(&mut self, state: &State) -> Option<u32> {
        self.value(state).map(saturate)
    }
}

/// Narrows an estimate to `u32`, saturating at [`u32::MAX`].
fn saturate(value: u64) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// The final merge-and-shrink abstraction's goal distances as a [`Heuristic`].
///
/// Admissible. `None` means a dead end or, when built with `prune: true`, a state the
/// construction found unreachable from the initial state; build with `prune: false` to
/// evaluate arbitrary states.
#[cfg(feature = "merge-and-shrink")]
impl Heuristic<State> for merge_and_shrink::MergeAndShrinkHeuristic {
    fn estimate(&mut self, state: &State) -> Option<u32> {
        self.value(state).map(saturate)
    }
}

/// A saturated cost partitioning over abstractions as a [`Heuristic`]: the sum of their goal
/// distances under their cost shares. Admissible.
#[cfg(feature = "cost-partition")]
impl<A: Abstraction> Heuristic<State> for cost_partition::CostPartitionedHeuristic<'_, A> {
    fn estimate(&mut self, state: &State) -> Option<u32> {
        self.value(state).map(saturate)
    }
}

/// The maximum over several saturated cost partitionings as a [`Heuristic`]. Admissible.
#[cfg(feature = "cost-partition")]
impl<A: Abstraction> Heuristic<State> for cost_partition::MaxOverOrders<'_, A> {
    fn estimate(&mut self, state: &State) -> Option<u32> {
        self.value(state).map(saturate)
    }
}
