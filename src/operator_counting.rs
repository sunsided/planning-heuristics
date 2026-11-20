//! Operator-counting heuristics by linear programming (feature `lp`): the state equation
//! (Bonet 2013) and LM-cut landmarks as constraints over operator counts (Pommerening,
//! Röger, Helmert & Bonet 2014).

use alloc::vec::Vec;

use microlp::{ComparisonOp, Error, OptimizationDirection, Problem, SolveOutcome, Variable};
use sas_plus::{Fact, OpId, State, Task};

use crate::{Heuristic, LmCut};

/// Which constraints an [`OperatorCounting`] heuristic uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CountingConstraints {
    /// The state equation: per fact, the operators that may add it minus those that always
    /// delete it count at least its required net change.
    pub state_equation: bool,
    /// The LM-cut landmarks of the state: each cut's operators count at least 1 in total.
    pub lm_cut: bool,
}

impl CountingConstraints {
    /// Both kinds of constraints.
    pub const ALL: Self = Self {
        state_equation: true,
        lm_cut: true,
    };
}

/// The state-equation constraint of one fact of a state variable.
#[derive(Clone, Debug)]
struct FactBalance {
    /// The fact.
    fact: Fact,
    /// Whether it is a goal fact.
    goal: bool,
    /// The operators with an effect that sets it and no precondition on it already.
    producers: Vec<OpId>,
    /// The operators that require it and always set its variable to another value.
    consumers: Vec<OpId>,
}

/// An operator-counting heuristic (Pommerening, Röger, Helmert & Bonet 2014): minimize the
/// cost `Σ cost(o) · Y_o` of operator counts `Y_o ≥ 0` that satisfy constraints every plan's
/// counts satisfy. Admissible. Feature `lp`.
///
/// - **State equation** (Bonet 2013): for each fact `f` of a state variable, the number of
///   times a plan makes `f` true minus the number of times it makes `f` false equals its end
///   value minus its value in the state. The first is at most the count of operators that may
///   add `f`, the second at least the count of operators that require `f` and always change
///   its variable; the end value is 1 for a goal fact and at least 0 otherwise. Derived
///   variables get no constraint. Conditional effects only count as possible producers.
/// - **LM-cut**: every cut of [`LmCut::landmarks`] is an action landmark, so its operators'
///   counts sum to at least 1. The LP is then at least the LM-cut estimate.
///
/// An infeasible LP or an LM-cut dead end proves a dead end. The LP optimum is rounded up
/// after subtracting a relative tolerance of `1e-6` (true costs are integers); the result is
/// at least the LM-cut estimate when LM-cut constraints are on, and falls back to it if the
/// solver fails. One LP per state, `O(facts × operators)` to build: for comparisons and small
/// tasks.
///
/// # Examples
///
/// ```
/// use planning_heuristics::{CountingConstraints, Heuristic, LmCut, OperatorCounting};
/// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
///
/// // A switch must be on for `a` and off for `b`, then off at the end: the state equation
/// // sees that `on` and `off` are both needed.
/// let op = |name: &str, pre: Vec<Fact>, eff: Fact| Operator {
///     name: name.into(),
///     pre,
///     eff: vec![Effect::new(eff)],
///     cost: 1,
/// };
/// let task = Task {
///     variables: vec![Variable::new("switch", 2), Variable::new("a", 2), Variable::new("b", 2)],
///     operators: vec![
///         op("on", vec![Fact::new(0, 0)], Fact::new(0, 1)),
///         op("off", vec![Fact::new(0, 1)], Fact::new(0, 0)),
///         op("a", vec![Fact::new(0, 1)], Fact::new(1, 1)),
///         op("b", vec![Fact::new(0, 0)], Fact::new(2, 1)),
///     ],
///     initial: State(vec![0, 0, 0]),
///     goal: vec![Fact::new(1, 1), Fact::new(2, 1), Fact::new(0, 0)],
///     axioms: vec![],
/// };
/// let state = task.initial_state();
///
/// assert_eq!(LmCut::new(&task).estimate(&state), Some(3));
/// let mut h = OperatorCounting::new(&task, CountingConstraints::ALL);
/// assert_eq!(h.estimate(&state), Some(4));
/// ```
#[derive(Clone, Debug)]
pub struct OperatorCounting {
    /// LM-cut, if its landmarks are constraints.
    lm_cut: Option<LmCut<'static>>,
    /// The state-equation constraints, empty if not used.
    balances: Vec<FactBalance>,
    /// The cost of each operator.
    costs: Vec<u64>,
}

impl OperatorCounting {
    /// Builds the heuristic for `task` with the `constraints`.
    #[must_use]
    pub fn new(task: &Task, constraints: CountingConstraints) -> Self {
        let balances = if constraints.state_equation {
            balances(task)
        } else {
            Vec::new()
        };

        Self {
            lm_cut: constraints.lm_cut.then(|| LmCut::new(task)),
            balances,
            costs: task.operators.iter().map(|o| u64::from(o.cost)).collect(),
        }
    }

    /// The operator-counting estimate of `state`; `None` for a proven dead end.
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    #[must_use]
    pub fn value(&mut self, state: &State) -> Option<u64> {
        let (landmarks, floor) = match &mut self.lm_cut {
            Some(lm_cut) => (lm_cut.landmarks(state)?, lm_cut.value(state)?),
            None => (Vec::new(), 0),
        };

        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let counts: Vec<Variable> = self
            .costs
            .iter()
            .map(|&c| problem.add_var(to_f64(c), (0.0, f64::INFINITY)))
            .collect();
        for balance in &self.balances {
            let current = state.0[balance.fact.var] == balance.fact.value;
            let rhs = f64::from(u8::from(balance.goal)) - f64::from(u8::from(current));
            if rhs <= 0.0 && balance.consumers.is_empty() {
                continue;
            }

            let mut terms: Vec<(Variable, f64)> = balance
                .producers
                .iter()
                .map(|&o| (counts[o], 1.0))
                .collect();
            terms.extend(balance.consumers.iter().map(|&o| (counts[o], -1.0)));
            problem.add_constraint(terms.as_slice(), ComparisonOp::Ge, rhs);
        }

        for landmark in &landmarks {
            let terms: Vec<(Variable, f64)> =
                landmark.ops().iter().map(|&o| (counts[o], 1.0)).collect();
            problem.add_constraint(terms.as_slice(), ComparisonOp::Ge, 1.0);
        }

        match problem.solve() {
            Ok(SolveOutcome::Solution(solution)) => Some(round_up(solution.objective()).max(floor)),
            Err(Error::Infeasible) => None,
            _ => Some(floor),
        }
    }
}

impl Heuristic<State> for OperatorCounting {
    fn estimate(&mut self, state: &State) -> Option<u32> {
        self.value(state)
            .map(|v| u32::try_from(v).unwrap_or(u32::MAX))
    }
}

/// The state-equation constraints of every fact of a state variable of `task`.
fn balances(task: &Task) -> Vec<FactBalance> {
    let mut all = Vec::new();
    for (var, v) in task.variables.iter().enumerate() {
        if v.is_derived() {
            continue;
        }

        for value in 0..v.domain_size {
            let fact = Fact::new(var, value);
            let producers = (0..task.operators.len())
                .filter(|&o| may_produce(task, o, fact))
                .collect();
            let consumers = (0..task.operators.len())
                .filter(|&o| always_consumes(task, o, fact))
                .collect();
            all.push(FactBalance {
                fact,
                goal: task.goal.contains(&fact),
                producers,
                consumers,
            });
        }
    }

    all
}

/// Whether operator `o` may make `fact` true: an effect sets it and the precondition does not
/// already require it.
fn may_produce(task: &Task, o: OpId, fact: Fact) -> bool {
    let op = &task.operators[o];
    op.eff.iter().any(|e| e.fact == fact) && !op.pre.contains(&fact)
}

/// Whether operator `o` always makes `fact` false: it requires `fact`, every effect on its
/// variable sets another value, and one of them is unconditional.
fn always_consumes(task: &Task, o: OpId, fact: Fact) -> bool {
    let op = &task.operators[o];
    let on_var = || op.eff.iter().filter(|e| e.fact.var == fact.var);

    op.pre.contains(&fact)
        && on_var().all(|e| e.fact.value != fact.value)
        && on_var().any(|e| e.conditions.is_empty())
}

/// Rounds an LP optimum up to an integer after subtracting a relative tolerance of `1e-6`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is positive and far below 2^64; `as` truncates toward zero"
)]
fn round_up(value: f64) -> u64 {
    let value = value - 1e-6 * (1.0 + value.abs());
    if value <= 0.0 {
        return 0;
    }

    value.ceil() as u64
}

/// `cost` as `f64`; exact below `2⁵³`.
#[expect(
    clippy::cast_precision_loss,
    reason = "operator costs are u32, exact in f64"
)]
fn to_f64(cost: u64) -> f64 {
    cost as f64
}
