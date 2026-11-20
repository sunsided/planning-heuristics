//! Landmarks and orderings against brute force over the state space, and the landmark
//! heuristics against `h*`.

use std::collections::{HashSet, VecDeque};

use planning_heuristics::{HMax, LandmarkCount, Relaxation, UniformLandmarkCost};
use sas_plus::random::{RandomTaskParams, random_task};
use sas_plus::{Fact, State, Task};

use crate::{fixtures, oracle};

/// Number of seeds per parameter set.
const SEEDS: u64 = 48;

/// Every fact of `task`, indexed by its [`Relaxation::fact_id`].
fn facts_by_id(task: &Task, relaxation: &Relaxation) -> Vec<Fact> {
    let mut facts = vec![Fact::new(0, 0); relaxation.num_facts()];
    for (var, v) in task.variables.iter().enumerate() {
        for value in 0..v.domain_size {
            let fact = Fact::new(var, value);
            facts[relaxation.fact_id(fact)] = fact;
        }
    }

    facts
}

/// Whether `fact` holds in `state`.
fn holds(fact: Fact, state: &State) -> bool {
    state.0[fact.var] == fact.value
}

/// Whether a state satisfying `target` is reachable from `start` through states in which
/// `avoid` does not hold (`start` included).
fn reachable_avoiding(
    task: &Task,
    start: &State,
    avoid: Fact,
    target: impl Fn(&State) -> bool,
) -> bool {
    let mut seen = HashSet::from([start.clone()]);
    let mut queue = VecDeque::from([start.clone()]);
    while let Some(state) = queue.pop_front() {
        if holds(avoid, &state) {
            continue;
        }

        if target(&state) {
            return true;
        }

        for op in task.applicable_operators(&state) {
            let next = task.apply(op, &state);
            if seen.insert(next.clone()) {
                queue.push_back(next);
            }
        }
    }

    false
}

/// Checks the landmarks and orderings of every reachable state of `task` by brute force, and
/// the landmark heuristics against `h*` and `h_max`'s dead ends.
///
/// Returns the number of pending landmarks that are not goal facts and the number of
/// orderings seen, summed over the states, so callers can check the test is not vacuous.
fn check_landmarks(name: &str, task: &Task) -> (usize, usize) {
    let mut seen = (0, 0);
    let relaxation = Relaxation::new(task);
    let facts = facts_by_id(task, &relaxation);
    let mut count = LandmarkCount::with_relaxation(&relaxation);
    let mut uniform = UniformLandmarkCost::with_relaxation(&relaxation);
    let mut h_max = HMax::with_relaxation(&relaxation);

    for (state, h_star) in oracle::perfect(task) {
        let ctx = || format!("{name}, state {:?}", state.0);
        let graph = relaxation.landmarks(&state);
        assert_eq!(
            graph.is_none(),
            h_max.value(&state).is_none(),
            "dead ends, {}",
            ctx()
        );
        assert_eq!(count.value(&state).is_none(), graph.is_none(), "{}", ctx());
        let Some(graph) = graph else {
            assert_eq!(uniform.value(&state), None, "{}", ctx());
            continue;
        };

        for &g in &task.goal {
            assert!(
                graph.landmarks().contains(&relaxation.fact_id(g)),
                "goal, {}",
                ctx()
            );
        }

        if h_star.is_some() {
            check_graph_by_brute_force(task, &state, &facts, &graph, &ctx);
        }

        let ucp = uniform.value(&state).expect("not a relaxed dead end");
        if let Some(h_star) = h_star {
            assert!(ucp <= h_star, "uniform {ucp} > h* {h_star}, {}", ctx());
        }

        let pending = graph
            .landmarks()
            .iter()
            .filter(|&&l| !holds(facts[l], &state))
            .count();
        assert_eq!(count.value(&state), Some(pending as u64), "{}", ctx());
        seen.0 += graph
            .landmarks()
            .iter()
            .filter(|&&l| !holds(facts[l], &state) && !task.goal.contains(&facts[l]))
            .count();
        seen.1 += graph.orderings().len();
    }

    seen
}

/// Every landmark holds on every path to a goal from `state`, and every ordering
/// `(before, after)` holds: `after` never holds before `before` has.
fn check_graph_by_brute_force(
    task: &Task,
    state: &State,
    facts: &[Fact],
    graph: &planning_heuristics::LandmarkGraph,
    ctx: &impl Fn() -> String,
) {
    for &l in graph.landmarks() {
        let goal_without = reachable_avoiding(task, state, facts[l], |s| task.is_goal(s));
        assert!(!goal_without, "{:?} is no landmark, {}", facts[l], ctx());
    }

    for &(before, after) in graph.orderings() {
        let (before, after) = (facts[before], facts[after]);
        let early = reachable_avoiding(task, state, before, |s| holds(after, s));
        assert!(!early, "{after:?} can hold before {before:?}, {}", ctx());
    }
}

/// Checks [`SEEDS`] random tasks generated with `params`.
///
/// # Panics
///
/// Panics if no state of any task has a pending non-goal landmark or an ordering.
fn check_random(label: &str, params: RandomTaskParams) {
    let mut seen = (0, 0);
    for seed in 0..SEEDS {
        let (landmarks, orderings) =
            check_landmarks(&format!("{label}, seed {seed}"), &random_task(seed, params));
        seen = (seen.0 + landmarks, seen.1 + orderings);
    }

    assert!(seen.0 > 0 && seen.1 > 0, "{label}: vacuous, {seen:?}");
}

/// Plain random tasks.
#[test]
fn landmarks_on_random_plain_tasks() {
    check_random("plain", RandomTaskParams::default());
}

/// Random tasks with conditional effects.
#[test]
fn landmarks_on_random_conditional_tasks() {
    check_random(
        "conditional",
        RandomTaskParams {
            max_conditions: 2,
            ..RandomTaskParams::default()
        },
    );
}

/// Random tasks with conditional effects and axioms.
#[test]
fn landmarks_on_random_mixed_tasks() {
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
fn landmarks_on_fixtures() {
    for (name, _) in fixtures::FIXTURES {
        check_landmarks(name, &fixtures::load(name));
    }
}

/// The landmarks are informative: on gripper the uniform partitioning is positive in the
/// initial state and finds landmarks beyond the goal facts.
#[test]
fn landmarks_are_informative_on_gripper() {
    let task = fixtures::load("gripper-unit");
    let relaxation = Relaxation::new(&task);
    let state = task.initial_state();
    let graph = relaxation.landmarks(&state).unwrap();

    assert!(graph.landmarks().len() > task.goal.len());
    assert_ne!(graph.orderings(), []);
    assert!(UniformLandmarkCost::new(&task).value(&state).unwrap() > 0);
}
