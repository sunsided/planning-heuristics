//! Goal-intention heuristics from Rosa, Honorio, Lipovetzky & Sardiña, *Planning as Goal
//! Recognition: Deriving Heuristics from Intention Models* (extended version,
//! arXiv:2603.14824, 2026).
//!
//! See [`GoalIntention`] and [`IntentionTracker`].

mod sampling;

use alloc::vec::Vec;

use sas_plus::{Fact, OpId, State, Task};

use crate::math::ln;
use crate::rng::SplitMix64;
use crate::{PathHeuristic, Relaxation};
use sampling::Graph;

/// Parameters of the fact probability estimation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IntentionParams {
    /// The number of sampled relaxed plans per distribution (the paper: 100).
    pub samples: usize,
    /// The seed of the supporter choices.
    pub seed: u64,
}

impl Default for IntentionParams {
    /// 100 samples, seed 0.
    fn default() -> Self {
        Self {
            samples: 100,
            seed: 0,
        }
    }
}

/// Fact observation probabilities for the relaxed plan observation heuristics `ol^rp`
/// (likelihood) and `op^rp` (posterior) of Rosa et al. (2026).
///
/// From one state (the initial state), sample relaxed plans to the goal by backchaining
/// through the relaxed planning graph (their Appendix B, Algorithm 2: for each open fact,
/// among its achievers at or below the current layer take those of minimum `h_add`, then
/// those chosen least often so far, then one at random). An operator's probability is the
/// fraction of samples containing it; a fact's probability is that of observing at least one
/// of its achievers, `P(q | G) = 1 − Π_a (1 − P(a))`, treating achievers as independent, so
/// even a fact supported in every sample can get less than 1. `P(q | ¬G)` comes the same way
/// from samples that support every reachable non-goal fact instead of the goal.
///
/// With facts treated as independent Bernoulli variables, the likelihood of an observed fact
/// set `O` is `P̃(O | G) = Π_{q ∈ O} P(q | G)`, which by their Theorem 4 is `e^(−KL)` of the
/// one-hot observation against that product distribution. The costs here are its negative
/// logarithm ([`GoalIntention::likelihood_cost`]) and the negative log posterior ratio
/// `Σ_{q ∈ O} −ln P(q | G) + ln P(q | ¬G)` ([`GoalIntention::posterior_cost`]); lower means
/// more goal-intended.
///
/// These are ranking keys, not distance estimates: they grow with the observation (Claim 1),
/// are not admissible, and never prove a dead end. The paper uses them as a tie-breaker in
/// BFWS; [`IntentionTracker`] carries them along search paths.
///
/// # Examples
///
/// ```
/// use planning_heuristics::{GoalIntention, IntentionParams};
/// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
///
/// // The goal `g` has two equally cheap achievers; each is sampled half the time, so `g` gets
/// // 1 − 0.5 · 0.5 = 0.75 although every sample supports it. `x` is never needed.
/// let op = |eff: Fact| Operator {
///     name: "op".into(),
///     pre: vec![],
///     eff: vec![Effect::new(eff)],
///     cost: 1,
/// };
/// let (g, x) = (Fact::new(0, 1), Fact::new(1, 1));
/// let task = Task {
///     variables: vec![Variable::new("g", 2), Variable::new("x", 2)],
///     operators: vec![op(g), op(g), op(x)],
///     initial: State(vec![0, 0]),
///     goal: vec![g],
///     axioms: vec![],
/// };
/// let intention = GoalIntention::new(&task, &IntentionParams::default());
///
/// assert_eq!(intention.probability(g), 0.75);
/// assert_eq!(intention.probability(x), 0.0);
/// assert!((intention.likelihood_cost([g]) - (-(0.75_f64).ln())).abs() < 1e-12);
/// assert_eq!(intention.likelihood_cost([g, x]), f64::INFINITY);
/// ```
#[derive(Clone, Debug)]
pub struct GoalIntention {
    /// The relaxation, for fact ids and operator effects.
    relaxation: Relaxation,
    /// `P(q | G)` per fact.
    goal: Vec<f64>,
    /// `P(q | ¬G)` per fact.
    not_goal: Vec<f64>,
    /// The sampled supporter sets towards the goal, as sources (see [`GoalIntention::samples`]).
    goal_samples: Vec<Vec<usize>>,
    /// The sampled supporter sets towards all reachable non-goal facts.
    not_goal_samples: Vec<Vec<usize>>,
}

impl GoalIntention {
    /// Estimates the fact probabilities of `task` from its initial state.
    #[must_use]
    pub fn new(task: &Task, params: &IntentionParams) -> Self {
        Self::from_state(task, &task.initial_state(), params)
    }

    /// Estimates the fact probabilities of `task` from `state`.
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    #[must_use]
    pub fn from_state(task: &Task, state: &State, params: &IntentionParams) -> Self {
        let relaxation = Relaxation::new(task);
        let graph = Graph::new(&relaxation, state);
        let open = |f: usize| graph.is_reachable(f) && !graph.is_initial(f);
        let goals: Vec<usize> = relaxation
            .goal_facts()
            .iter()
            .copied()
            .filter(|&g| open(g))
            .collect();
        let others: Vec<usize> = (0..relaxation.num_facts())
            .filter(|&f| open(f) && relaxation.goal_facts().binary_search(&f).is_err())
            .collect();
        let reached = relaxation
            .goal_facts()
            .iter()
            .all(|&g| graph.is_reachable(g));
        let mut rng = SplitMix64(params.seed);
        let goal_samples = if reached {
            graph.sample(&goals, params.samples, &mut rng)
        } else {
            Vec::new()
        };
        let not_goal_samples = graph.sample(&others, params.samples, &mut rng);
        let goal = fact_probabilities(&relaxation, &graph, &goal_samples);
        let not_goal = fact_probabilities(&relaxation, &graph, &not_goal_samples);
        let ops = relaxation.num_operators();
        let operators_only = |samples: Vec<Vec<usize>>| -> Vec<Vec<usize>> {
            samples
                .into_iter()
                .map(|s| s.into_iter().filter(|&x| x < ops).collect())
                .collect()
        };

        Self {
            goal,
            not_goal,
            goal_samples: operators_only(goal_samples),
            not_goal_samples: operators_only(not_goal_samples),
            relaxation,
        }
    }

    /// `P(q | G)` of `fact`.
    ///
    /// # Panics
    ///
    /// Panics if `fact` is not a fact of the task.
    #[must_use]
    pub fn probability(&self, fact: Fact) -> f64 {
        self.goal[self.relaxation.fact_id(fact)]
    }

    /// `P(q | ¬G)` of `fact`.
    ///
    /// # Panics
    ///
    /// Panics if `fact` is not a fact of the task.
    #[must_use]
    pub fn not_goal_probability(&self, fact: Fact) -> f64 {
        self.not_goal[self.relaxation.fact_id(fact)]
    }

    /// The sampled relaxed plans towards the goal, as operator sets (axioms left out),
    /// ascending; empty if the goal is relaxed-unreachable.
    #[must_use]
    pub fn samples(&self) -> &[Vec<OpId>] {
        &self.goal_samples
    }

    /// The sampled supporter sets towards every reachable non-goal fact, as operator sets.
    #[must_use]
    pub fn not_goal_samples(&self) -> &[Vec<OpId>] {
        &self.not_goal_samples
    }

    /// `−ln P̃(O | G)` of the observed facts `observed` (each counted as often as given):
    /// `+∞` if one has probability 0 (Claim 2), `0` for no observation.
    pub fn likelihood_cost(&self, observed: impl IntoIterator<Item = Fact>) -> f64 {
        let mut cost = Cost::default();
        for f in observed {
            cost.add(self, self.relaxation.fact_id(f));
        }

        cost.likelihood()
    }

    /// `Σ −ln P(q | G) + ln P(q | ¬G)` over the observed facts `observed`, the negative log
    /// of the posterior ratio of their Eq. 6: `+∞` if a fact has `P(q | G) = 0`, otherwise `−∞`
    /// if one has `P(q | ¬G) = 0` (never NaN).
    pub fn posterior_cost(&self, observed: impl IntoIterator<Item = Fact>) -> f64 {
        let mut cost = Cost::default();
        for f in observed {
            cost.add(self, self.relaxation.fact_id(f));
        }

        cost.posterior()
    }

    /// A path heuristic over these probabilities; see [`IntentionTracker`].
    #[must_use]
    pub fn tracker(&self, kind: IntentionKind, restart: bool) -> IntentionTracker<'_> {
        IntentionTracker {
            intention: self,
            kind,
            restart,
        }
    }

    /// The number of goal facts not true in `state`, the goal count `h_GC`.
    fn goal_count(&self, state: &State) -> usize {
        let goals = self.relaxation.goal_facts();
        goals.len()
            - self
                .relaxation
                .state_facts(state)
                .filter(|&f| goals.binary_search(&(f as usize)).is_ok())
                .count()
    }
}

/// The per-fact probabilities from `samples`: `1 − Π (1 − P(a))` over the sources achieving
/// the fact, with `P(a)` the fraction of samples containing source `a`.
fn fact_probabilities(
    relaxation: &Relaxation,
    graph: &Graph<'_>,
    samples: &[Vec<usize>],
) -> Vec<f64> {
    let mut contained = alloc::vec![0_u32; graph.num_sources()];
    for s in samples {
        for &a in s {
            contained[a] += 1;
        }
    }

    let n = f64::from(
        u32::try_from(samples.len())
            .expect("fewer than 2^32 samples")
            .max(1),
    );
    // Each source counts once per fact it achieves, even with several effects on it.
    let mut achievers: Vec<(usize, usize)> = (0..relaxation.rules().len())
        .map(|r| (relaxation.rules()[r].head, graph.source(r)))
        .collect();
    achievers.sort_unstable();
    achievers.dedup();
    let mut none = alloc::vec![1.0_f64; relaxation.num_facts()];
    for (head, a) in achievers {
        none[head] *= 1.0 - f64::from(contained[a]) / n;
    }

    none.into_iter().map(|p| 1.0 - p).collect()
}

/// A running cost over observed facts: the finite sums and the infinite cases counted, so
/// that adding facts never produces NaN.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Cost {
    /// `Σ −ln P(q | G)` over the facts with positive probability.
    likelihood: f64,
    /// `Σ ln P(q | ¬G)` over the facts with positive probabilities.
    not_goal: f64,
    /// The number of facts with `P(q | G) = 0`.
    impossible: u32,
    /// The number of facts with `P(q | G) > 0` and `P(q | ¬G) = 0`.
    certain: u32,
}

impl Cost {
    /// Adds fact `f` (by fact id).
    fn add(&mut self, intention: &GoalIntention, f: usize) {
        let (p_goal, p_not_goal) = (intention.goal[f], intention.not_goal[f]);
        if p_goal == 0.0 {
            self.impossible += 1;
            return;
        }

        self.likelihood -= ln(p_goal);
        if p_not_goal == 0.0 {
            self.certain += 1;
        } else {
            self.not_goal += ln(p_not_goal);
        }
    }

    /// `−ln P̃(O | G)`.
    fn likelihood(&self) -> f64 {
        if self.impossible > 0 {
            f64::INFINITY
        } else {
            self.likelihood
        }
    }

    /// The negative log posterior ratio.
    fn posterior(&self) -> f64 {
        if self.impossible > 0 {
            f64::INFINITY
        } else if self.certain > 0 {
            f64::NEG_INFINITY
        } else {
            self.likelihood + self.not_goal
        }
    }
}

/// Which of the two heuristics an [`IntentionTracker`] computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntentionKind {
    /// `ol^rp`: the negative log likelihood.
    Likelihood,
    /// `op^rp`: the negative log posterior ratio.
    Posterior,
}

/// The facts observed along a path and their cost, carried by [`IntentionTracker`].
#[derive(Clone, Debug, PartialEq)]
pub struct Observation {
    /// Whether each fact (by fact id) was observed, as a bit set.
    observed: Vec<u64>,
    /// The cost of the observed facts.
    cost: Cost,
    /// The lowest goal count along the path (for restarts).
    best_goal_count: usize,
}

/// The relaxed plan observation heuristics as a path heuristic: the observation of a path is
/// the set of facts its operators added (the effects that fired), its estimate the cost of
/// that set under [`IntentionKind`]. With `restart` (the paper's `ol^rp_r` and `op^rp_r`),
/// the observation restarts empty at every state that lowers the goal count below all
/// earlier states on the path. Estimates are `f64` (lower is better) and never `None`.
///
/// Merging two paths to one state keeps the more goal-intended (lower cost) observation.
#[derive(Clone, Copy, Debug)]
pub struct IntentionTracker<'g> {
    /// The probabilities.
    intention: &'g GoalIntention,
    /// Likelihood or posterior.
    kind: IntentionKind,
    /// Whether to restart at goal count improvements.
    restart: bool,
}

impl IntentionTracker<'_> {
    /// The estimate of `info` under the tracker's kind.
    fn value(&self, info: &Observation) -> f64 {
        match self.kind {
            IntentionKind::Likelihood => info.cost.likelihood(),
            IntentionKind::Posterior => info.cost.posterior(),
        }
    }
}

impl PathHeuristic<State, OpId, f64> for IntentionTracker<'_> {
    type Info = Observation;

    fn root(&mut self, state: &State) -> Observation {
        Observation {
            observed: alloc::vec![0; self.intention.relaxation.num_facts().div_ceil(64)],
            cost: Cost::default(),
            best_goal_count: self.intention.goal_count(state),
        }
    }

    fn on_expand(
        &mut self,
        parent: &State,
        parent_info: &Observation,
        op: OpId,
        child: &State,
    ) -> Observation {
        let relaxation = &self.intention.relaxation;
        let goal_count = self.intention.goal_count(child);
        if self.restart && goal_count < parent_info.best_goal_count {
            let mut info = self.root(child);
            info.best_goal_count = goal_count;
            return info;
        }

        let mut info = parent_info.clone();
        let holds = |f: usize| {
            let fact = relaxation.fact(f);
            parent.0[fact.var] == fact.value
        };
        for rule in relaxation.op_rules(op) {
            let head = rule.head;
            let fired = rule.conditions.iter().all(|&c| holds(c));
            let bit = 1_u64 << (head % 64);
            if fired && info.observed[head / 64] & bit == 0 {
                info.observed[head / 64] |= bit;
                info.cost.add(self.intention, head);
            }
        }

        info
    }

    fn merge(&mut self, kept: &mut Observation, other: &Observation) {
        if self.value(other) < self.value(kept) {
            kept.clone_from(other);
        }
    }

    fn estimate(&mut self, _state: &State, info: &Observation) -> Option<f64> {
        Some(self.value(info))
    }
}
