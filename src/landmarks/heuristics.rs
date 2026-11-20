//! Heuristics over the fact landmarks of each evaluated state.
//!
//! Both recompute the landmarks from every state they evaluate, so they need no search-path
//! bookkeeping and fit [`Heuristic`] as it is. LAMA-style landmark counting instead computes
//! the landmarks once and tracks which ones each path accepted; that needs a stateful
//! extension of the trait and is not done here.

use alloc::borrow::Cow;
use alloc::vec::Vec;

use sas_plus::{OpId, State, Task};

use super::Labels;
use crate::{Heuristic, Relaxation};

/// The landmark-count heuristic: the number of fact landmarks of the state that do not hold in
/// it yet. Not admissible (one operator can achieve several landmarks) and blind to costs.
///
/// The landmarks are recomputed from every evaluated state, so no search-path bookkeeping is
/// needed. LAMA instead computes them once and tracks the landmarks each path has accepted,
/// which needs a stateful extension of [`Heuristic`]; that is not done here.
///
/// # Examples
///
/// ```
/// use planning_heuristics::{Heuristic, LandmarkCount};
/// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
///
/// // One operator sets both variables; the two goal facts are two landmarks.
/// let task = Task {
///     variables: vec![Variable::new("a", 2), Variable::new("b", 2)],
///     operators: vec![Operator {
///         name: "both".into(),
///         pre: vec![],
///         eff: vec![Effect::new(Fact::new(0, 1)), Effect::new(Fact::new(1, 1))],
///         cost: 1,
///     }],
///     initial: State(vec![0, 0]),
///     goal: vec![Fact::new(0, 1), Fact::new(1, 1)],
///     axioms: vec![],
/// };
///
/// assert_eq!(LandmarkCount::new(&task).estimate(&task.initial_state()), Some(2));
/// ```
#[derive(Clone, Debug)]
pub struct LandmarkCount<'r> {
    /// The relaxation of the task, owned or shared.
    relaxation: Cow<'r, Relaxation>,
    /// Label buffers, refilled by every evaluation.
    labels: Labels,
}

impl LandmarkCount<'static> {
    /// Builds the landmark-count heuristic for `task`, owning its relaxation.
    #[must_use]
    pub fn new(task: &Task) -> Self {
        Self::from_cow(Cow::Owned(Relaxation::new(task)))
    }
}

impl<'r> LandmarkCount<'r> {
    /// Builds the landmark-count heuristic over a shared `relaxation`.
    #[must_use]
    pub fn with_relaxation(relaxation: &'r Relaxation) -> Self {
        Self::from_cow(Cow::Borrowed(relaxation))
    }

    /// The heuristic over `relaxation` with empty buffers.
    fn from_cow(relaxation: Cow<'r, Relaxation>) -> Self {
        Self {
            relaxation,
            labels: Labels::default(),
        }
    }

    /// The number of landmarks of `state` that do not hold in it; `None` for a relaxed dead
    /// end.
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    #[must_use]
    pub fn value(&mut self, state: &State) -> Option<u64> {
        if !self.labels.compute(&self.relaxation, state) {
            return None;
        }

        Some(self.labels.pending_landmarks().count() as u64)
    }
}

impl Heuristic<State> for LandmarkCount<'_> {
    fn estimate(&mut self, state: &State) -> Option<u32> {
        self.value(state).map(saturate)
    }
}

/// An admissible landmark heuristic: uniform cost partitioning over the fact landmarks
/// (Karpas & Domshlak 2009).
///
/// Each operator's cost is split evenly among the pending landmarks it can achieve (those of
/// the state that do not hold in it, and that some effect of the operator adds); each
/// landmark is worth its cheapest share, and the estimate is the sum, rounded up. Every plan
/// achieves every pending landmark by some first achiever whose shares sum to at most its
/// cost, so the sum never exceeds the plan cost. Landmarks only axioms achieve are worth 0.
///
/// The sum is accumulated in `f64` and rounded up after subtracting a relative tolerance of
/// `1e-9`, far above the rounding error of the additions, so the estimate stays admissible
/// for sums up to `2⁵³`; beyond that it is approximate.
///
/// # Examples
///
/// ```
/// use planning_heuristics::{Heuristic, UniformLandmarkCost};
/// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
///
/// // `both` (cost 4) achieves both goal landmarks: each gets 2, the sum is 4 = h*.
/// let task = Task {
///     variables: vec![Variable::new("a", 2), Variable::new("b", 2)],
///     operators: vec![Operator {
///         name: "both".into(),
///         pre: vec![],
///         eff: vec![Effect::new(Fact::new(0, 1)), Effect::new(Fact::new(1, 1))],
///         cost: 4,
///     }],
///     initial: State(vec![0, 0]),
///     goal: vec![Fact::new(0, 1), Fact::new(1, 1)],
///     axioms: vec![],
/// };
///
/// assert_eq!(UniformLandmarkCost::new(&task).estimate(&task.initial_state()), Some(4));
/// ```
#[derive(Clone, Debug)]
pub struct UniformLandmarkCost<'r> {
    /// The relaxation of the task, owned or shared.
    relaxation: Cow<'r, Relaxation>,
    /// For each fact, the operators with an effect that adds it, ascending without
    /// duplicates.
    achievers: Vec<Vec<OpId>>,
    /// The cost of each operator.
    costs: Vec<u64>,
    /// Label buffers, refilled by every evaluation.
    labels: Labels,
    /// Per operator, the number of pending landmarks it achieves; `0` until computed in the
    /// current evaluation.
    shares: Vec<u32>,
}

impl UniformLandmarkCost<'static> {
    /// Builds the heuristic for `task`, owning its relaxation.
    #[must_use]
    pub fn new(task: &Task) -> Self {
        Self::from_cow(Cow::Owned(Relaxation::new(task)))
    }
}

impl<'r> UniformLandmarkCost<'r> {
    /// Builds the heuristic over a shared `relaxation`; indexes the achievers of every fact.
    #[must_use]
    pub fn with_relaxation(relaxation: &'r Relaxation) -> Self {
        Self::from_cow(Cow::Borrowed(relaxation))
    }

    /// The heuristic over `relaxation`, with the achiever index and operator costs.
    fn from_cow(relaxation: Cow<'r, Relaxation>) -> Self {
        let num_ops = relaxation.num_operators();
        let mut achievers = alloc::vec![Vec::new(); relaxation.num_facts()];
        let mut costs = alloc::vec![0; num_ops];
        for (op, cost) in costs.iter_mut().enumerate() {
            for rule in relaxation.op_rules(op) {
                *cost = rule.cost;
                if achievers[rule.head].last() != Some(&op) {
                    achievers[rule.head].push(op);
                }
            }
        }

        Self {
            relaxation,
            achievers,
            costs,
            labels: Labels::default(),
            shares: Vec::new(),
        }
    }

    /// The uniform cost partitioning estimate of `state`; `None` for a relaxed dead end.
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    #[must_use]
    pub fn value(&mut self, state: &State) -> Option<u64> {
        if !self.labels.compute(&self.relaxation, state) {
            return None;
        }

        self.shares.clear();
        self.shares.resize(self.costs.len(), 0);
        let mut sum = 0.0;
        for landmark in self.labels.pending_landmarks() {
            let achievers = &self.achievers[landmark];
            sum += cheapest_share(
                &self.relaxation,
                &self.labels,
                achievers,
                &self.costs,
                &mut self.shares,
            );
        }

        Some(round_up_admissible(sum))
    }
}

impl Heuristic<State> for UniformLandmarkCost<'_> {
    fn estimate(&mut self, state: &State) -> Option<u32> {
        self.value(state).map(saturate)
    }
}

/// The cheapest share of a pending landmark with `achievers`: `cost / pending landmarks
/// achieved` over the achievers, filling `shares` (per operator, `0` until computed); `0` if
/// only axioms achieve it.
fn cheapest_share(
    relaxation: &Relaxation,
    labels: &Labels,
    achievers: &[OpId],
    costs: &[u64],
    shares: &mut [u32],
) -> f64 {
    let mut best = f64::INFINITY;
    for &op in achievers {
        if shares[op] == 0 {
            shares[op] = pending_heads(relaxation, labels, op);
        }

        best = best.min(to_f64(costs[op]) / f64::from(shares[op]));
    }

    if best.is_finite() { best } else { 0.0 }
}

/// The number of distinct pending landmarks among the effects of `op`; at least 1 when `op`
/// achieves a pending landmark.
fn pending_heads(relaxation: &Relaxation, labels: &Labels, op: OpId) -> u32 {
    let rules = relaxation.op_rules(op);
    let pending = |i: usize| labels.is_pending(rules[i].head);
    let distinct = (0..rules.len())
        .filter(|&i| pending(i) && !rules[..i].iter().any(|r| r.head == rules[i].head))
        .count();

    u32::try_from(distinct).expect("an operator has fewer than 2^32 effects")
}

/// Rounds a non-negative sum of landmark shares up to an integer, after subtracting a
/// tolerance for the rounding error of the `f64` additions.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is positive; `as` truncates toward zero and saturates"
)]
fn round_up_admissible(sum: f64) -> u64 {
    let value = sum - 1e-9 * (1.0 + sum);
    if value <= 0.0 {
        return 0;
    }

    let truncated = value as u64;

    truncated + u64::from(to_f64(truncated) < value)
}

/// `cost` as `f64`; exact below `2⁵³`.
#[expect(
    clippy::cast_precision_loss,
    reason = "operator costs are u32 and sums stay below 2^53, exact in f64"
)]
fn to_f64(cost: u64) -> f64 {
    cost as f64
}

/// Narrows an estimate to `u32`, saturating at [`u32::MAX`].
fn saturate(value: u64) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}
