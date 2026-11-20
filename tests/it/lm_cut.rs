//! LM-cut against `h*`, `h_max` and the fact-landmark heuristics, with every cut checked to be
//! an action landmark by brute force.

use std::collections::{HashSet, VecDeque};

use planning_heuristics::{ActionLandmark, HMax, LmCut, Relaxation, UniformLandmarkCost};
use sas_plus::random::{RandomTaskParams, random_task};
use sas_plus::{Effect, Fact, OpId, Operator, State, Task, Variable};

use crate::{fixtures, oracle};

/// Number of seeds per parameter set.
const SEEDS: u64 = 48;

/// Whether a goal state is reachable from `start` without applying any operator in `banned`.
fn goal_reachable_without(task: &Task, start: &State, banned: &[OpId]) -> bool {
    let mut seen = HashSet::from([start.clone()]);
    let mut queue = VecDeque::from([start.clone()]);
    while let Some(state) = queue.pop_front() {
        if task.is_goal(&state) {
            return true;
        }

        for op in task.applicable_operators(&state) {
            if banned.contains(&op) {
                continue;
            }

            let next = task.apply(op, &state);
            if seen.insert(next.clone()) {
                queue.push_back(next);
            }
        }
    }

    false
}

/// Checks LM-cut on every reachable state of `task`: dead ends as `h_max`, `h_max <= h_LM-cut
/// <= h*`, landmark costs positive and summing to the estimate, and every cut a real action
/// landmark.
///
/// Returns how many states LM-cut is above `h_max`, and above the uniform landmark cost
/// partitioning, so callers can check the test is not vacuous.
fn check_lm_cut(name: &str, task: &Task) -> (usize, usize) {
    let relaxation = Relaxation::new(task);
    let conditional = task
        .operators
        .iter()
        .any(|o| o.eff.iter().any(|e| !e.conditions.is_empty()));
    let mut unsplit = LmCut::with_relaxation(&relaxation);
    assert_eq!(unsplit.is_exact(), !conditional, "{name}");
    let mut split = LmCut::new(task);
    assert!(split.is_exact() || conditional, "{name}");

    let a = check_variant(&format!("{name}, unsplit"), task, &relaxation, &mut unsplit);
    let b = check_variant(&format!("{name}, split"), task, &relaxation, &mut split);

    (a.0 + b.0, a.1 + b.1)
}

/// [`check_lm_cut`] for one LM-cut variant.
fn check_variant(
    name: &str,
    task: &Task,
    relaxation: &Relaxation,
    lm_cut: &mut LmCut<'_>,
) -> (usize, usize) {
    let mut h_max = HMax::with_relaxation(relaxation);
    let mut uniform = UniformLandmarkCost::with_relaxation(relaxation);
    let mut above = (0, 0);

    for (state, h_star) in oracle::perfect(task) {
        let ctx = || format!("{name}, state {:?}", state.0);
        let value = lm_cut.value(&state);
        let max = h_max.value(&state);
        // Context splitting drops effects that contradict their precondition, so the split
        // variant can prove more dead ends than `h_max`; each must be real.
        if max.is_none() || lm_cut.is_exact() && value.is_none() {
            assert!(max.is_none() <= value.is_none(), "dead ends, {}", ctx());
            assert!(
                value.is_some() || h_star.is_none(),
                "false dead end, {}",
                ctx()
            );
        } else {
            assert_eq!(value.is_none(), max.is_none(), "dead ends, {}", ctx());
        }

        let (Some(value), Some(max)) = (value, max) else {
            continue;
        };

        assert!(max <= value, "h_max {max} > h_LM-cut {value}, {}", ctx());
        if let Some(h_star) = h_star {
            assert!(value <= h_star, "h_LM-cut {value} > h* {h_star}, {}", ctx());
        }

        let landmarks = lm_cut.landmarks(&state).expect("not a dead end");
        let sum: u64 = landmarks.iter().map(ActionLandmark::cost).sum();
        assert_eq!(value, sum.max(max), "{}", ctx());
        if lm_cut.is_exact() {
            assert!(sum >= max, "exact, but sum {sum} < h_max {max}, {}", ctx());
        }

        for landmark in &landmarks {
            assert!(
                landmark.cost() > 0 && !landmark.ops().is_empty(),
                "{}",
                ctx()
            );
            if h_star.is_some() {
                assert!(
                    !goal_reachable_without(task, &state, landmark.ops()),
                    "{:?} is no action landmark, {}",
                    landmark.ops(),
                    ctx()
                );
            }
        }

        above.0 += usize::from(value > max);
        above.1 += usize::from(value > uniform.value(&state).expect("not a dead end"));
    }

    above
}

/// Checks [`SEEDS`] random tasks generated with `params`.
///
/// # Panics
///
/// Panics if LM-cut never exceeds `h_max` or the uniform partitioning on any state.
fn check_random(label: &str, params: RandomTaskParams) {
    let mut above = (0, 0);
    for seed in 0..SEEDS {
        let (max, uniform) =
            check_lm_cut(&format!("{label}, seed {seed}"), &random_task(seed, params));
        above = (above.0 + max, above.1 + uniform);
    }

    assert!(above.0 > 0 && above.1 > 0, "{label}: vacuous, {above:?}");
}

/// Plain random tasks.
#[test]
fn lm_cut_on_random_plain_tasks() {
    check_random("plain", RandomTaskParams::default());
}

/// Random tasks with conditional effects.
#[test]
fn lm_cut_on_random_conditional_tasks() {
    check_random(
        "conditional",
        RandomTaskParams {
            max_conditions: 2,
            ..RandomTaskParams::default()
        },
    );
}

/// Random tasks with axioms.
#[test]
fn lm_cut_on_random_axiom_tasks() {
    check_random(
        "axioms",
        RandomTaskParams {
            num_derived: 2,
            ..RandomTaskParams::default()
        },
    );
}

/// Random tasks with conditional effects, axioms and more operators.
#[test]
fn lm_cut_on_random_mixed_tasks() {
    check_random(
        "mixed",
        RandomTaskParams {
            num_vars: 5,
            num_ops: 14,
            max_conditions: 2,
            num_derived: 2,
            ..RandomTaskParams::default()
        },
    );
}

/// Every fixture.
#[test]
fn lm_cut_on_fixtures() {
    for (name, _) in fixtures::FIXTURES {
        check_lm_cut(name, &fixtures::load(name));
    }
}

/// A hand-checked instance: a chain `a → b` (costs 1 and 4) next to an independent goal `c`
/// (cost 2). `h_max = 5`, `h_LM-cut = h* = 7` with one cut per operator.
#[test]
fn lm_cut_hand_checked() {
    let op = |name: &str, pre: Vec<Fact>, eff: Fact, cost: u32| Operator {
        name: name.into(),
        pre,
        eff: vec![Effect::new(eff)],
        cost,
    };
    let task = Task {
        variables: vec![
            Variable::new("a", 2),
            Variable::new("b", 2),
            Variable::new("c", 2),
        ],
        operators: vec![
            op("a", vec![], Fact::new(0, 1), 1),
            op("b", vec![Fact::new(0, 1)], Fact::new(1, 1), 4),
            op("c", vec![], Fact::new(2, 1), 2),
        ],
        initial: State(vec![0, 0, 0]),
        goal: vec![Fact::new(1, 1), Fact::new(2, 1)],
        axioms: vec![],
    };
    let state = task.initial_state();
    let mut lm_cut = LmCut::new(&task);

    assert_eq!(HMax::new(&task).value(&state), Some(5));
    assert_eq!(lm_cut.value(&state), Some(7));
    let mut cuts: Vec<(Vec<OpId>, u64)> = lm_cut
        .landmarks(&state)
        .unwrap()
        .iter()
        .map(|l| (l.ops().to_vec(), l.cost()))
        .collect();
    cuts.sort();
    assert_eq!(cuts, [(vec![0], 1), (vec![1], 4), (vec![2], 2)]);
}

/// The goal holds: no cuts, estimate 0. An empty goal: also 0.
#[test]
fn lm_cut_is_zero_at_the_goal() {
    let task = fixtures::load("gripper-unit");
    let mut goal_state = task.initial_state();
    for g in &task.goal {
        goal_state.0[g.var] = g.value;
    }

    assert_eq!(LmCut::new(&task).landmarks(&goal_state), Some(vec![]));

    let empty = Task {
        goal: vec![],
        ..task.clone()
    };
    assert_eq!(LmCut::new(&empty).value(&empty.initial_state()), Some(0));
}

/// Context splitting makes the cut sum dominate `h_max` on conditional tasks, where the
/// unsplit variant needs the `max(sum, h_max)` fallback on some states.
#[test]
fn context_splitting_removes_the_fallback() {
    let params = RandomTaskParams {
        max_conditions: 2,
        ..RandomTaskParams::default()
    };
    let (mut fallback, mut exact) = (0, 0);
    for seed in 0..SEEDS {
        let task = random_task(seed, params);
        let relaxation = Relaxation::new(&task);
        let mut unsplit = LmCut::with_relaxation(&relaxation);
        let mut split = LmCut::new(&task);
        exact += usize::from(split.is_exact());
        let mut h_max = HMax::with_relaxation(&relaxation);
        for state in oracle::perfect(&task).keys() {
            let Some(max) = h_max.value(state) else {
                continue;
            };

            let sum = |h: &mut LmCut<'_>| {
                h.landmarks(state)
                    .map(|l| l.iter().map(ActionLandmark::cost).sum::<u64>())
            };
            fallback += usize::from(sum(&mut unsplit).is_some_and(|s| s < max));
            if split.is_exact() {
                // `None`: a dead end the split task proves; checked against h* elsewhere.
                let s = sum(&mut split);
                assert!(
                    s.is_none_or(|s| s >= max),
                    "seed {seed}, state {:?}",
                    state.0
                );
            }
        }
    }

    assert!(
        exact > 0 && fallback > 0,
        "vacuous: exact {exact}, fallback {fallback}"
    );
}
