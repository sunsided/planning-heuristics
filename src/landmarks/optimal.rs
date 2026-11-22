//! Optimal cost partitioning over the fact landmarks, by linear programming (feature `lp`).

use alloc::vec::Vec;

use microlp::{ComparisonOp, OptimizationDirection, Problem, SolveOutcome, Variable};
use sas_plus::{OpId, State, Task};

use super::heuristics::{round_up_admissible, saturate, to_f64};
use super::ordered::OrderedLandmarks;
use super::tracking::AcceptedLandmarks;
use crate::{
    Heuristic, LandmarkTracker, PathHeuristic, Relaxation, TrackingMode, UniformLandmarkCost,
};

/// An admissible landmark heuristic: the optimal cost partitioning over the fact landmarks
/// (Karpas & Domshlak 2009), solved as a linear program. Feature `lp`.
///
/// For the pending landmarks `L` of the state (as for [`UniformLandmarkCost`]) with at least
/// one operator achiever, solve
///
/// ```text
/// maximize   Σ_{l ∈ L} c_l
/// subject to Σ_{l ∈ L achieved by o} c_l ≤ cost(o)   for every operator o
///            c_l ≥ 0
/// ```
///
/// Every plan achieves every pending landmark with a first achiever, and each operator pays
/// at most its cost in total, so the optimum never exceeds the plan cost. Landmarks that only
/// axioms achieve are worth 0. The uniform partitioning is a feasible point, so the optimum is
/// at least [`UniformLandmarkCost`].
///
/// The solver works in `f64` with tolerances. To stay admissible regardless, the solution is
/// clamped to `c_l ≥ 0` and scaled down until every operator constraint holds in exact
/// arithmetic up to the `f64` sums, then rounded up like the uniform estimate; the result is
/// the maximum of that and the uniform estimate. A solver failure falls back to the uniform
/// estimate.
///
/// One LP per state: `O(landmarks × operators)` to build, plus the simplex. Meant for
/// comparisons and small tasks; LM-cut ([`crate::LmCut`]) is far cheaper per state.
///
/// # Examples
///
/// ```
/// use planning_heuristics::{Heuristic, OptimalLandmarkCost, UniformLandmarkCost};
/// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
///
/// // `x` achieves a and b, `y` achieves b and c, `z` only c; all cost 2. Uniform: a gets 1
/// // (half of `x`), b gets 1, c gets 1, sum 3. Optimal: a and c get 2 each, b gets 0: 4 = h*.
/// let op = |name: &str, vars: &[usize]| Operator {
///     name: name.into(),
///     pre: vec![],
///     eff: vars.iter().map(|&v| Effect::new(Fact::new(v, 1))).collect(),
///     cost: 2,
/// };
/// let task = Task {
///     variables: vec![Variable::new("a", 2), Variable::new("b", 2), Variable::new("c", 2)],
///     operators: vec![op("x", &[0, 1]), op("y", &[1, 2]), op("z", &[2])],
///     initial: State(vec![0, 0, 0]),
///     goal: vec![Fact::new(0, 1), Fact::new(1, 1), Fact::new(2, 1)],
///     axioms: vec![],
/// };
/// let state = task.initial_state();
///
/// assert_eq!(UniformLandmarkCost::new(&task).estimate(&state), Some(3));
/// assert_eq!(OptimalLandmarkCost::new(&task).estimate(&state), Some(4));
/// ```
#[derive(Clone, Debug)]
pub struct OptimalLandmarkCost<'r> {
    /// The uniform partitioning: its landmarks, achiever index and costs are reused, and its
    /// estimate is the floor of this one.
    uniform: UniformLandmarkCost<'r>,
}

impl OptimalLandmarkCost<'static> {
    /// Builds the heuristic for `task`, owning its relaxation.
    #[must_use]
    pub fn new(task: &Task) -> Self {
        Self {
            uniform: UniformLandmarkCost::new(task),
        }
    }
}

impl<'r> OptimalLandmarkCost<'r> {
    /// Builds the heuristic over a shared `relaxation`.
    #[must_use]
    pub fn with_relaxation(relaxation: &'r Relaxation) -> Self {
        Self {
            uniform: UniformLandmarkCost::with_relaxation(relaxation),
        }
    }

    /// The optimal cost partitioning estimate of `state`; `None` for a relaxed dead end.
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    #[must_use]
    pub fn value(&mut self, state: &State) -> Option<u64> {
        let (uniform, optimum) = self.both(state)?;

        Some(optimum.unwrap_or(0).max(uniform))
    }

    /// The uniform estimate of `state` and the rounded LP optimum (`None` if the solver
    /// failed); `None` for a relaxed dead end.
    fn both(&mut self, state: &State) -> Option<(u64, Option<u64>)> {
        let uniform = self.uniform.value(state)?;
        let landmarks: Vec<usize> = self
            .uniform
            .labels
            .pending_landmarks()
            .filter(|&l| !self.uniform.achievers[l].is_empty())
            .collect();
        let rows = operator_rows(
            &self.uniform.achievers,
            self.uniform.costs.len(),
            &landmarks,
        );
        let optimum = solve(&landmarks, &rows, &self.uniform.costs).map(round_up_admissible);

        Some((uniform, optimum))
    }
}

impl Heuristic<State> for OptimalLandmarkCost<'_> {
    fn estimate(&mut self, state: &State) -> Option<u32> {
        self.value(state).map(saturate)
    }
}

/// For each of the `num_ops` operators that achieves one of `landmarks` (indices into
/// `achievers`), the indices into `landmarks` it achieves, ascending.
fn operator_rows(
    achievers: &[Vec<OpId>],
    num_ops: usize,
    landmarks: &[usize],
) -> Vec<(OpId, Vec<usize>)> {
    let mut rows: Vec<(OpId, Vec<usize>)> = Vec::new();
    let mut row_of = alloc::vec![usize::MAX; num_ops];
    for (i, &l) in landmarks.iter().enumerate() {
        for &op in &achievers[l] {
            if row_of[op] == usize::MAX {
                row_of[op] = rows.len();
                rows.push((op, Vec::new()));
            }

            rows[row_of[op]].1.push(i);
        }
    }

    rows
}

/// The optimal cost partitioning over the landmarks a search path still has to achieve
/// (Karpas & Domshlak 2009), as a [`PathHeuristic`]. Feature `lp`.
///
/// Tracks landmarks like [`LandmarkTracker`] in [`TrackingMode::UniformCost`]: same
/// acceptance, same required-again rule, same merge by intersection, same dead ends. Instead
/// of the uniform partitioning it solves the LP of [`OptimalLandmarkCost`] over the landmarks
/// still to achieve. Each of them is false in the state and must be made true by a future
/// operator, so the optimum never exceeds the cost of the rest of any plan. The result is
/// made feasible and rounded like [`OptimalLandmarkCost`], then maxed with the uniform
/// estimate, so it is at least [`LandmarkTracker`]'s.
///
/// # Examples
///
/// ```
/// use planning_heuristics::{
///     LandmarkTracker, OptimalLandmarkTracker, PathHeuristic, TrackingMode,
/// };
/// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
///
/// // As for `OptimalLandmarkCost`: uniform gives 3, the optimal partitioning 4 = h*.
/// let op = |name: &str, vars: &[usize]| Operator {
///     name: name.into(),
///     pre: vec![],
///     eff: vars.iter().map(|&v| Effect::new(Fact::new(v, 1))).collect(),
///     cost: 2,
/// };
/// let task = Task {
///     variables: vec![Variable::new("a", 2), Variable::new("b", 2), Variable::new("c", 2)],
///     operators: vec![op("x", &[0, 1]), op("y", &[1, 2]), op("z", &[2])],
///     initial: State(vec![0, 0, 0]),
///     goal: vec![Fact::new(0, 1), Fact::new(1, 1), Fact::new(2, 1)],
///     axioms: vec![],
/// };
/// let start = task.initial_state();
///
/// let mut uniform = LandmarkTracker::for_task(&task, TrackingMode::UniformCost).unwrap();
/// let root = uniform.root(&start);
/// assert_eq!(uniform.estimate(&start, &root), Some(3));
/// let mut optimal = OptimalLandmarkTracker::for_task(&task).unwrap();
/// let root = optimal.root(&start);
/// assert_eq!(optimal.estimate(&start, &root), Some(4));
/// ```
#[derive(Clone, Debug)]
pub struct OptimalLandmarkTracker<'r> {
    /// The tracker in [`TrackingMode::UniformCost`]: its acceptance and pending landmarks are
    /// reused, and its estimate is the floor of this one.
    uniform: LandmarkTracker<'r>,
}

impl OptimalLandmarkTracker<'static> {
    /// A tracker for searches from the initial state of `task`, with its RHW landmarks plus
    /// reasonable orderings; `None` if the initial state is a dead end.
    #[must_use]
    pub fn for_task(task: &Task) -> Option<Self> {
        Some(Self {
            uniform: LandmarkTracker::for_task(task, TrackingMode::UniformCost)?,
        })
    }
}

impl<'r> OptimalLandmarkTracker<'r> {
    /// A tracker over a shared `relaxation` with the landmark `graph` of the search's start
    /// state.
    ///
    /// # Panics
    ///
    /// Panics if a landmark names a fact the relaxation does not have.
    #[must_use]
    pub fn new(relaxation: &'r Relaxation, graph: &OrderedLandmarks) -> Self {
        Self {
            uniform: LandmarkTracker::new(relaxation, graph, TrackingMode::UniformCost),
        }
    }
}

impl PathHeuristic<State, OpId, u32> for OptimalLandmarkTracker<'_> {
    type Info = AcceptedLandmarks;

    fn root(&mut self, state: &State) -> AcceptedLandmarks {
        self.uniform.root(state)
    }

    fn on_expand(
        &mut self,
        parent: &State,
        parent_info: &AcceptedLandmarks,
        op: OpId,
        child: &State,
    ) -> AcceptedLandmarks {
        self.uniform.on_expand(parent, parent_info, op, child)
    }

    fn merge(&mut self, kept: &mut AcceptedLandmarks, other: &AcceptedLandmarks) {
        self.uniform.merge(kept, other);
    }

    fn estimate(&mut self, state: &State, info: &AcceptedLandmarks) -> Option<u32> {
        // Fills the pending landmarks and proves the dead ends.
        let uniform = self.uniform.estimate(state, info)?;
        let t = &self.uniform;
        // Landmarks an axiom can add may come for free: worth 0, as in the uniform estimate.
        let landmarks: Vec<usize> = t
            .pending
            .iter()
            .copied()
            .filter(|&l| !t.by_axiom[l] && !t.achievers[l].is_empty())
            .collect();
        let rows = operator_rows(&t.achievers, t.costs.len(), &landmarks);
        let optimum = solve(&landmarks, &rows, &t.costs).map_or(0, round_up_admissible);

        Some(saturate(optimum).max(uniform))
    }
}

/// Solves the partitioning LP; returns a sum of landmark costs that satisfies every operator
/// constraint, or `None` if the solver fails.
fn solve(landmarks: &[usize], rows: &[(OpId, Vec<usize>)], costs: &[u64]) -> Option<f64> {
    if landmarks.is_empty() {
        return Some(0.0);
    }

    let mut problem = Problem::new(OptimizationDirection::Maximize);
    let vars: Vec<Variable> = landmarks
        .iter()
        .map(|_| problem.add_var(1.0, (0.0, f64::INFINITY)))
        .collect();
    for (op, row) in rows {
        let terms: Vec<(Variable, f64)> = row.iter().map(|&i| (vars[i], 1.0)).collect();
        problem.add_constraint(terms.as_slice(), ComparisonOp::Le, to_f64(costs[*op]));
    }

    let SolveOutcome::Solution(solution) = problem.solve().ok()? else {
        return None;
    };
    let values: Vec<f64> = vars
        .iter()
        .map(|&v| solution.var_value(v).max(0.0))
        .collect();

    Some(feasible_sum(&values, rows, costs))
}

/// The sum of `values` after scaling them down so that every operator row stays within its
/// cost.
fn feasible_sum(values: &[f64], rows: &[(OpId, Vec<usize>)], costs: &[u64]) -> f64 {
    let mut scale: f64 = 1.0;
    for (op, row) in rows {
        let load: f64 = row.iter().map(|&i| values[i]).sum();
        if load > 0.0 {
            scale = scale.min(to_f64(costs[*op]) / load);
        }
    }

    values.iter().sum::<f64>() * scale
}

#[cfg(test)]
mod tests {
    //! The raw LP optimum, before it is maxed with the uniform estimate.

    use alloc::collections::{BTreeMap, BinaryHeap};
    use alloc::format;
    use alloc::vec::Vec;
    use core::cmp::Reverse;

    use sas_plus::random::{RandomTaskParams, random_task};
    use sas_plus::{State, Task};

    use super::OptimalLandmarkCost;

    /// The cost of a cheapest plan from `start`, by Dijkstra; `None` if there is none.
    fn h_star(task: &Task, start: &State) -> Option<u64> {
        let mut best: BTreeMap<Vec<u32>, u64> = BTreeMap::from([(start.0.clone(), 0)]);
        let mut open = BinaryHeap::from([Reverse((0_u64, start.0.clone()))]);
        while let Some(Reverse((g, values))) = open.pop() {
            let state = State(values);
            if best[&state.0] < g {
                continue;
            }

            if task.is_goal(&state) {
                return Some(g);
            }

            for op in task.applicable_operators(&state) {
                let next = task.apply(op, &state);
                let cost = g + u64::from(task.operators[op].cost);
                if best.get(&next.0).is_none_or(|&old| cost < old) {
                    best.insert(next.0.clone(), cost);
                    open.push(Reverse((cost, next.0)));
                }
            }
        }

        None
    }

    /// On every reachable state of random tasks, the solver succeeds and its rounded optimum
    /// lies between the uniform estimate and `h*`; somewhere it beats the uniform one.
    #[test]
    fn raw_optimum_between_uniform_and_h_star() {
        let mut better = 0;
        for seed in 0..64 {
            let params = RandomTaskParams {
                max_conditions: 1,
                ..RandomTaskParams::default()
            };
            let task = random_task(seed, params);
            let mut h = OptimalLandmarkCost::new(&task);
            let mut stack = alloc::vec![task.initial_state()];
            let mut seen = alloc::collections::BTreeSet::from([task.initial_state().0]);
            while let Some(state) = stack.pop() {
                let ctx = || format!("seed {seed}, state {:?}", state.0);
                if let Some((uniform, optimum)) = h.both(&state) {
                    let optimum = optimum.unwrap_or_else(|| panic!("solver failed, {}", ctx()));
                    assert!(
                        uniform <= optimum,
                        "uniform {uniform} > LP {optimum}, {}",
                        ctx()
                    );
                    if let Some(h_star) = h_star(&task, &state) {
                        assert!(optimum <= h_star, "LP {optimum} > h* {h_star}, {}", ctx());
                    }

                    better += usize::from(optimum > uniform);
                }

                for op in task.applicable_operators(&state) {
                    let next = task.apply(op, &state);
                    if seen.insert(next.0.clone()) {
                        stack.push(next);
                    }
                }
            }
        }

        assert!(better > 0, "the LP never beats the uniform partitioning");
    }
}
