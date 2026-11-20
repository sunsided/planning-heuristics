//! The goal-intention heuristics of Rosa et al. (2026) against the properties the paper
//! claims, and a tiny enumeration of its plan-library (IRPL) model.

use std::collections::HashMap;

use planning_heuristics::{GoalIntention, IntentionKind, IntentionParams, PathHeuristic};
use sas_plus::random::{RandomTaskParams, random_task};
use sas_plus::{Effect, Fact, OpId, Operator, State, Task, Variable};

use crate::oracle;

/// Random task families: plain, conditional effects, axioms.
fn families() -> [RandomTaskParams; 3] {
    let base = RandomTaskParams::default();
    [
        base,
        RandomTaskParams {
            max_conditions: 2,
            ..base
        },
        RandomTaskParams {
            num_derived: 2,
            ..base
        },
    ]
}

/// All facts of `task`.
fn facts(task: &Task) -> Vec<Fact> {
    task.variables
        .iter()
        .enumerate()
        .flat_map(|(var, v)| (0..v.domain_size).map(move |value| Fact::new(var, value)))
        .collect()
}

/// Every goal sample is a relaxed plan from the initial state; every non-goal sample reaches
/// every reachable non-goal fact; probabilities lie in `[0, 1]`, and open reachable goal facts
/// have positive probability.
#[test]
fn samples_are_relaxed_plans() {
    let mut checked = 0;
    for params in families() {
        for seed in 0..32 {
            let task = random_task(seed, params);
            let state = task.initial_state();
            let intention = GoalIntention::new(&task, &IntentionParams { samples: 20, seed });
            let reachable = oracle::relaxed_costs(&task, &state, planning_heuristics::Combine::Max);
            for sample in intention.samples() {
                assert!(
                    oracle::relaxed_closure_reaches_goal(&task, &state, sample),
                    "seed {seed}"
                );
                checked += 1;
            }

            let mut others = task.clone();
            others.goal = reachable
                .keys()
                .copied()
                .filter(|f| !task.goal.contains(f) && state.0[f.var] != f.value)
                .filter(|f| !task.variables[f.var].is_derived())
                .collect();
            for sample in intention.not_goal_samples() {
                assert!(
                    oracle::relaxed_closure_reaches_goal(&others, &state, sample),
                    "seed {seed}"
                );
            }

            let solvable = task.goal.iter().all(|g| reachable.contains_key(g));
            for f in facts(&task) {
                let p = intention.probability(f);
                assert!((0.0..=1.0).contains(&p), "{f:?}: {p}");
                let open_goal = task.goal.contains(&f)
                    && state.0[f.var] != f.value
                    && !task.variables[f.var].is_derived();
                if open_goal && solvable {
                    assert!(p > 0.0, "goal {f:?}, seed {seed}");
                }
            }
        }
    }

    assert!(checked > 500, "vacuous: {checked}");
}

/// A small deterministic generator.
struct Rng(u64);

impl Rng {
    /// A value in `0..n`.
    fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        usize::try_from((self.0 >> 33) % u64::try_from(n).expect("small")).expect("small")
    }
}

/// Claim 1: extending an observation never raises the likelihood, so the likelihood cost
/// never falls along a path without restarts; the posterior cost is never NaN; and a fact of
/// probability 0 makes the likelihood 0 (Claim 2: infinite cost).
#[test]
fn likelihood_falls_along_paths() {
    let mut rng = Rng(3);
    let mut infinite = 0;
    for params in families() {
        for seed in 0..32 {
            let task = random_task(seed, params);
            let intention = GoalIntention::new(&task, &IntentionParams { samples: 20, seed });
            let mut likelihood = intention.tracker(IntentionKind::Likelihood, false);
            let mut posterior = intention.tracker(IntentionKind::Posterior, false);
            for _ in 0..4 {
                let mut state = task.initial_state();
                let mut info = likelihood.root(&state);
                let mut post = posterior.root(&state);
                let mut last = 0.0;
                for _ in 0..8 {
                    let ops: Vec<OpId> = task.applicable_operators(&state).collect();
                    if ops.is_empty() {
                        break;
                    }

                    let op = ops[rng.below(ops.len())];
                    let child = task.apply(op, &state);
                    info = likelihood.on_expand(&state, &info, op, &child);
                    post = posterior.on_expand(&state, &post, op, &child);
                    let value = likelihood
                        .estimate(&child, &info)
                        .expect("never a dead end");
                    assert!(value >= last, "seed {seed}: {value} < {last}");
                    assert!(!posterior.estimate(&child, &post).expect("some").is_nan());
                    last = value;
                    state = child;
                }

                infinite += usize::from(last.is_infinite());
            }

            for f in facts(&task) {
                if intention.probability(f) == 0.0 {
                    assert_eq!(intention.likelihood_cost([f]), f64::INFINITY);
                    assert_eq!(intention.posterior_cost([f]), f64::INFINITY);
                }
            }
        }
    }

    assert!(infinite > 0, "Claim 2 never exercised");
}

/// Theorem 4 / Eq. 7: the KL divergence between the observation (the observed facts all true,
/// the others updated by Bayes) and the product of the facts' Bernoulli distributions, by
/// enumeration over an intermediate domain with extra facts, equals the negative log
/// likelihood.
#[test]
fn kl_divergence_is_the_negative_log_likelihood() {
    // Four goal facts with two equally cheap achievers each: every probability is 0.75.
    let task = Task {
        variables: (0..4).map(|i| Variable::new(format!("v{i}"), 2)).collect(),
        operators: (0..8)
            .map(|i| Operator {
                name: format!("set{i}"),
                pre: vec![],
                eff: vec![Effect::new(Fact::new(i / 2, 1))],
                cost: 1,
            })
            .collect(),
        initial: State(vec![0; 4]),
        goal: (0..4).map(|v| Fact::new(v, 1)).collect(),
        axioms: vec![],
    };
    let intention = GoalIntention::new(&task, &IntentionParams::default());
    let positive: Vec<Fact> = facts(&task)
        .into_iter()
        .filter(|&f| intention.probability(f) > 0.0 && intention.probability(f) < 1.0)
        .collect();
    assert!(positive.len() >= 4, "{positive:?}");

    let (observed, extra) = positive.split_at(2);
    let p = |f: &Fact| intention.probability(*f);
    let mut kl = 0.0;
    for outcome in 0..(1_u32 << extra.len()) {
        // d_G over Z: the observed facts true (the only outcomes d_O,G supports) times the
        // extra facts' outcome; d_O,G: the extra facts' outcome alone (Bayes, independence).
        let extras: f64 = extra
            .iter()
            .enumerate()
            .map(|(i, f)| {
                if outcome & (1 << i) != 0 {
                    p(f)
                } else {
                    1.0 - p(f)
                }
            })
            .product();
        let d_g = observed.iter().map(p).product::<f64>() * extras;
        kl += extras * (extras / d_g).ln();
    }

    let cost = intention.likelihood_cost(observed.iter().copied());
    assert!((kl - cost).abs() < 1e-9, "{kl} vs {cost}");
}

/// With restarts, the observation empties at every state that lowers the goal count below all
/// earlier ones; without, it keeps every fact observed.
#[test]
fn restarts_at_goal_count_improvements() {
    let op = |pre: Vec<Fact>, eff: Fact| Operator {
        name: "op".into(),
        pre,
        eff: vec![Effect::new(eff)],
        cost: 1,
    };
    let (a, g1, g2) = (Fact::new(0, 1), Fact::new(1, 1), Fact::new(2, 1));
    let task = Task {
        variables: (0..3).map(|i| Variable::new(format!("v{i}"), 2)).collect(),
        operators: vec![op(vec![], a), op(vec![a], g1), op(vec![a], g2)],
        initial: State(vec![0, 0, 0]),
        goal: vec![g1, g2],
        axioms: vec![],
    };
    let intention = GoalIntention::new(&task, &IntentionParams::default());
    let s0 = task.initial_state();
    let s1 = task.apply(0, &s0);
    let s2 = task.apply(1, &s1);
    for (restart, expected) in [(false, intention.likelihood_cost([a, g1])), (true, 0.0)] {
        let mut h = intention.tracker(IntentionKind::Likelihood, restart);
        let i0 = h.root(&s0);
        let i1 = h.on_expand(&s0, &i0, 0, &s1);
        assert_eq!(h.estimate(&s1, &i1), Some(intention.likelihood_cost([a])));
        let i2 = h.on_expand(&s1, &i1, 1, &s2);
        assert_eq!(h.estimate(&s2, &i2), Some(expected), "restart {restart}");
    }
}

/// The plan-library model with uniform weights, enumerated on tiny tasks (a check of the
/// paper's arithmetic, not of the crate): Claim 1 (`P(O | G)` never rises when `O` is
/// extended), Claim 2 (it is 0 without a plan extending `O`) and Lemma 1 (some child has a
/// posterior at least the parent's).
#[test]
fn plan_library_model_claims() {
    let mut checked = 0;
    for seed in 0..32 {
        let task = random_task(
            seed,
            RandomTaskParams {
                num_vars: 3,
                max_domain: 2,
                num_ops: 6,
                ..RandomTaskParams::default()
            },
        );
        let Some(maximal) = maximal_trajectories(&task, 20_000) else {
            continue;
        };

        // Per prefix: (maximal trajectories, plans) extending it.
        let mut counts: HashMap<Vec<OpId>, (usize, usize)> = HashMap::new();
        for (trajectory, is_plan) in &maximal {
            for len in 0..=trajectory.len() {
                let entry = counts.entry(trajectory[..len].to_vec()).or_default();
                entry.0 += 1;
                entry.1 += usize::from(*is_plan);
            }
        }

        let plans = counts[&Vec::new()].1;
        if plans == 0 {
            continue;
        }

        let float = |n: usize| f64::from(u32::try_from(n).expect("few trajectories"));
        let likelihood = |o: &[OpId]| float(counts[o].1) / float(plans);
        let posterior = |o: &[OpId]| float(counts[o].1) / float(counts[o].0);
        for prefix in counts.keys() {
            let children: Vec<&Vec<OpId>> = counts
                .keys()
                .filter(|c| c.len() == prefix.len() + 1 && c.starts_with(prefix))
                .collect();
            for c in &children {
                assert!(likelihood(c) <= likelihood(prefix), "Claim 1, seed {seed}");
            }

            if counts[prefix].1 == 0 {
                assert_eq!(likelihood(prefix), 0.0, "Claim 2");
            }

            if !children.is_empty() {
                let best = children.iter().map(|c| posterior(c)).fold(0.0, f64::max);
                assert!(best >= posterior(prefix) - 1e-12, "Lemma 1, seed {seed}");
            }

            checked += 1;
        }
    }

    assert!(checked > 100, "vacuous: {checked}");
}

/// All maximal acyclic, non-goal-extending action sequences from the initial state, each with
/// whether it ends in a goal state; `None` if there are more than `limit`.
fn maximal_trajectories(task: &Task, limit: usize) -> Option<Vec<(Vec<OpId>, bool)>> {
    let mut out = Vec::new();
    let mut stack = vec![(task.initial_state(), Vec::new(), vec![task.initial_state()])];
    while let Some((state, ops, visited)) = stack.pop() {
        if task.is_goal(&state) {
            out.push((ops, true));
        } else {
            let next: Vec<(OpId, State)> = task
                .applicable_operators(&state)
                .map(|op| (op, task.apply(op, &state)))
                .filter(|(_, s)| !visited.contains(s))
                .collect();
            if next.is_empty() {
                out.push((ops.clone(), false));
            }

            for (op, s) in next {
                let mut ops = ops.clone();
                ops.push(op);
                let mut visited = visited.clone();
                visited.push(s.clone());
                stack.push((s, ops, visited));
            }
        }

        if out.len() > limit {
            return None;
        }
    }

    Some(out)
}
