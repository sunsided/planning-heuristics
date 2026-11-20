//! Optimal cost partitioning over the fact landmarks, by linear programming (feature `lp`).

use alloc::vec::Vec;

use microlp::{ComparisonOp, OptimizationDirection, Problem, SolveOutcome, Variable};
use sas_plus::{OpId, State, Task};

use super::heuristics::{round_up_admissible, saturate, to_f64};
use crate::{Heuristic, Relaxation, UniformLandmarkCost};

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
        let uniform = self.uniform.value(state)?;
        let landmarks: Vec<usize> = self
            .uniform
            .labels
            .pending_landmarks()
            .filter(|&l| !self.uniform.achievers[l].is_empty())
            .collect();
        let rows = operator_rows(&self.uniform, &landmarks);
        let optimum = solve(&landmarks, &rows, &self.uniform.costs).map_or(0, round_up_admissible);

        Some(optimum.max(uniform))
    }
}

impl Heuristic<State> for OptimalLandmarkCost<'_> {
    fn estimate(&mut self, state: &State) -> Option<u32> {
        self.value(state).map(saturate)
    }
}

/// For each operator that achieves one of `landmarks`, the indices into `landmarks` it
/// achieves, ascending.
fn operator_rows(
    uniform: &UniformLandmarkCost<'_>,
    landmarks: &[usize],
) -> Vec<(OpId, Vec<usize>)> {
    let mut rows: Vec<(OpId, Vec<usize>)> = Vec::new();
    let mut row_of = alloc::vec![usize::MAX; uniform.costs.len()];
    for (i, &l) in landmarks.iter().enumerate() {
        for &op in &uniform.achievers[l] {
            if row_of[op] == usize::MAX {
                row_of[op] = rows.len();
                rows.push((op, Vec::new()));
            }

            rows[row_of[op]].1.push(i);
        }
    }

    rows
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
