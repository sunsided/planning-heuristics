//! `h_add` agrees with width-search's own implementation (`SasProblem::h_add`).
//!
//! Equal values on tasks without axioms and without effect conditions that repeat a
//! precondition fact. Elsewhere the two differ, and ours must be at most theirs wherever both
//! are finite:
//!
//! - with axioms width-search does not treat derived defaults as free, so it can report a dead
//!   end or a higher cost;
//! - width-search does not deduplicate a rule's conditions, so a fact that is both a
//!   precondition and an effect condition counts twice in its sum.

use planning_heuristics::HAdd;
use sas_plus::Task;
use sas_plus::random::{RandomTaskParams, random_task};
use width_search::sas::SasProblem;

use crate::{fixtures, oracle};

/// Compares both `h_add`s on every reachable state of `task`.
fn compare(name: &str, task: &Task) {
    let ours = HAdd::new(task);
    let theirs = SasProblem::new(task);
    let exact = task.axioms.is_empty() && !repeats_precondition(task);

    for state in oracle::perfect(task).keys() {
        let (a, b) = (ours.value(state), theirs.h_add(state));
        if exact {
            assert_eq!(a, b, "{name}, state {:?}", state.0);
        } else if let (Some(a), Some(b)) = (a, b) {
            assert!(a <= b, "{name}, state {:?}: ours {a} > theirs {b}", state.0);
        }
    }
}

/// All fixtures, including those with conditional effects and axioms.
#[test]
fn h_add_matches_width_search_on_fixtures() {
    for (name, _) in fixtures::FIXTURES {
        compare(name, &fixtures::load(name));
    }
}

/// Random tasks with conditional effects and with axioms.
#[test]
fn h_add_matches_width_search_on_random_tasks() {
    let plain = RandomTaskParams::default();
    let params = [
        plain,
        RandomTaskParams {
            max_conditions: 2,
            ..plain
        },
        RandomTaskParams {
            max_conditions: 2,
            num_derived: 2,
            ..plain
        },
    ];
    for (i, p) in params.into_iter().enumerate() {
        for seed in 0..64 {
            compare(&format!("params {i}, seed {seed}"), &random_task(seed, p));
        }
    }
}

/// Whether some effect condition of `task` is also a precondition of its operator.
fn repeats_precondition(task: &Task) -> bool {
    task.operators.iter().any(|op| {
        op.eff
            .iter()
            .any(|e| e.conditions.iter().any(|c| op.pre.contains(c)))
    })
}
