//! Abstraction heuristics against `h*`: pattern databases, and with the optional features
//! merge-and-shrink and saturated cost partitioning.

use planning_heuristics::{AbstractionHeuristic, Heuristic};
use sas_plus::random::{RandomTaskParams, random_task};
use sas_plus::{State, Task};

use crate::{fixtures, oracle};

/// Number of seeds per parameter set.
const SEEDS: u64 = 48;

/// The state variables of `task` (derived variables left out).
fn state_variables(task: &Task) -> Vec<usize> {
    (0..task.num_variables())
        .filter(|&v| !task.variables[v].is_derived())
        .collect()
}

/// Checks that `value` never exceeds `h*` and never reports a dead end that is not one on
/// every reachable state of `task`; with `exact`, that it equals `h*`.
fn check_admissible(
    name: &str,
    task: &Task,
    exact: bool,
    mut value: impl FnMut(&State) -> Option<u64>,
) {
    for (state, h_star) in oracle::perfect(task) {
        let h = value(&state);
        let ctx = || format!("{name}, state {:?}: h {h:?}, h* {h_star:?}", state.0);
        if exact {
            assert_eq!(h, h_star, "{}", ctx());
        }

        match (h, h_star) {
            (None, Some(_)) => panic!("false dead end, {}", ctx()),
            (Some(h), Some(h_star)) => assert!(h <= h_star, "{}", ctx()),
            _ => {}
        }
    }
}

/// Pattern databases of every single variable and of all state variables.
fn check_projections(name: &str, task: &Task) {
    let all = state_variables(task);
    let exact = task.axioms.is_empty();
    let mut full = AbstractionHeuristic::projection(task, all.clone());
    check_admissible(name, task, exact, |s| full.value(s));
    assert_eq!(
        full.estimate(&task.initial_state()),
        full.value(&task.initial_state())
            .map(|v| u32::try_from(v).unwrap())
    );

    for &v in &all {
        let pdb = AbstractionHeuristic::projection(task, vec![v]);
        check_admissible(&format!("{name}, pattern {v}"), task, false, |s| {
            pdb.value(s)
        });
    }
}

/// The parameter sets of the random tasks: plain, conditional effects, and axioms.
fn random_params() -> [(&'static str, RandomTaskParams); 3] {
    [
        ("plain", RandomTaskParams::default()),
        (
            "conditional",
            RandomTaskParams {
                max_conditions: 2,
                ..RandomTaskParams::default()
            },
        ),
        (
            "axioms",
            RandomTaskParams {
                num_derived: 2,
                ..RandomTaskParams::default()
            },
        ),
    ]
}

/// Projections on random tasks and the fixtures; the full pattern is exact without axioms.
#[test]
fn projections_are_admissible() {
    for (label, params) in random_params() {
        for seed in 0..SEEDS {
            check_projections(&format!("{label}, seed {seed}"), &random_task(seed, params));
        }
    }

    for (name, _) in fixtures::FIXTURES {
        check_projections(name, &fixtures::load(name));
    }
}

/// Merge-and-shrink without pruning, with a size limit, never exceeds `h*`.
#[cfg(feature = "merge-and-shrink")]
#[test]
fn merge_and_shrink_is_admissible() {
    use merge_and_shrink::{Bisimulation, LinearMerge, MergeAndShrink};

    for (label, params) in random_params() {
        for seed in 0..SEEDS {
            let task = random_task(seed, params);
            let mut builder = MergeAndShrink::new(
                LinearMerge::new((0..task.num_variables()).collect()),
                Bisimulation,
            );
            builder.prune = false;
            builder.max_states = 8;
            let mut h = builder.build(&task);
            check_admissible(&format!("{label}, seed {seed}"), &task, false, |s| {
                h.estimate(s).map(u64::from)
            });
        }
    }
}

/// Saturated cost partitioning over the single-variable projections, alone and maximized
/// over two orders, never exceeds `h*`.
#[cfg(feature = "cost-partition")]
#[test]
fn saturated_cost_partitioning_is_admissible() {
    use cost_partition::{MaxOverOrders, saturated_cost_partitioning};
    use sas_plus::Projection;

    for (label, params) in random_params() {
        for seed in 0..SEEDS {
            let task = random_task(seed, params);
            let vars = state_variables(&task);
            let projections: Vec<Projection> = vars
                .iter()
                .map(|&v| Projection::new(&task, vec![v]))
                .collect();
            let forward: Vec<usize> = (0..projections.len()).collect();
            let backward: Vec<usize> = forward.iter().rev().copied().collect();
            let costs = task.costs();

            let mut scp = saturated_cost_partitioning(&projections, &forward, &costs);
            check_admissible(&format!("{label}, seed {seed}"), &task, false, |s| {
                scp.estimate(s).map(u64::from)
            });
            let mut max = MaxOverOrders::new(&projections, &[forward, backward], &costs);
            check_admissible(&format!("{label}, seed {seed}, max"), &task, false, |s| {
                max.estimate(s).map(u64::from)
            });
        }
    }
}
