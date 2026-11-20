//! Additive `h_max`: `h_max` summed over a partition of the operator costs (Haslum, Bonet &
//! Geffner 2005, with `m = 1`).

use alloc::vec::Vec;

use sas_plus::{OpId, State, Task};

use crate::{Heuristic, LmCut, Relaxation};

/// Additive `h_max` over a partition of the operators: for each part `i`, `h_max` under the
/// costs of the operators in part `i` (all others cost 0), summed over the parts.
///
/// Every operator's cost lies in exactly one part, so any plan's cost is the sum of its
/// costs under the parts, each at least that part's `h_max`: admissible. One part gives
/// `h_max`; finer partitions can be higher or lower. One `h_max` Dijkstra per part.
///
/// # Examples
///
/// ```
/// use planning_heuristics::{AdditiveHMax, HMax, Heuristic};
/// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
///
/// // Two independent goals of cost 2 and 3: h_max = 3, additive over the effect variables 5.
/// let set = |name: &str, var: usize, cost: u32| Operator {
///     name: name.into(),
///     pre: vec![],
///     eff: vec![Effect::new(Fact::new(var, 1))],
///     cost,
/// };
/// let task = Task {
///     variables: vec![Variable::new("a", 2), Variable::new("b", 2)],
///     operators: vec![set("set-a", 0, 2), set("set-b", 1, 3)],
///     initial: State(vec![0, 0]),
///     goal: vec![Fact::new(0, 1), Fact::new(1, 1)],
///     axioms: vec![],
/// };
/// let state = task.initial_state();
///
/// assert_eq!(HMax::new(&task).estimate(&state), Some(3));
/// assert_eq!(AdditiveHMax::new(&task).estimate(&state), Some(5));
/// ```
#[derive(Clone, Debug)]
pub struct AdditiveHMax {
    /// The `h_max` machinery over the task's relaxation (no context splitting).
    h_max: LmCut<'static>,
    /// Per part, the cost of every operator under that part.
    part_costs: Vec<Vec<u64>>,
}

impl AdditiveHMax {
    /// Additive `h_max` with the operators partitioned by the lowest variable they have an
    /// effect on (operators without effects in part 0). A simple default, not the goal-based
    /// partitioning of the paper.
    #[must_use]
    pub fn new(task: &Task) -> Self {
        let parts: Vec<usize> = task
            .operators
            .iter()
            .map(|op| op.eff.iter().map(|e| e.fact.var).min().unwrap_or(0))
            .collect();

        Self::with_partition(task, &parts)
    }

    /// Additive `h_max` with operator `op` in part `parts[op]`.
    ///
    /// # Panics
    ///
    /// Panics if `parts` does not have one entry per operator.
    #[must_use]
    pub fn with_partition(task: &Task, parts: &[usize]) -> Self {
        assert_eq!(parts.len(), task.operators.len(), "one part per operator");
        let h_max = LmCut::unsplit(Relaxation::new(task));
        let mut used: Vec<usize> = parts.to_vec();
        used.sort_unstable();
        used.dedup();
        let part_costs = used
            .iter()
            .map(|&part| {
                (0..parts.len())
                    .map(|op: OpId| {
                        if parts[op] == part {
                            h_max.costs()[op]
                        } else {
                            0
                        }
                    })
                    .collect()
            })
            .collect();

        Self { h_max, part_costs }
    }

    /// The additive `h_max` estimate of `state`; `None` if the goal is relaxed-unreachable.
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    #[must_use]
    pub fn value(&mut self, state: &State) -> Option<u64> {
        if self.part_costs.is_empty() {
            return self.h_max.h_max_under(state, &[]);
        }

        let mut total = 0_u64;
        for costs in &self.part_costs {
            total = total.saturating_add(self.h_max.h_max_under(state, costs)?);
        }

        Some(total)
    }
}

impl Heuristic<State> for AdditiveHMax {
    fn estimate(&mut self, state: &State) -> Option<u32> {
        self.value(state)
            .map(|v| u32::try_from(v).unwrap_or(u32::MAX))
    }
}
