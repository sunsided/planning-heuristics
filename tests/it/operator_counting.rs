//! Operator counting (feature `lp`) against `h*` and LM-cut.

use planning_heuristics::{CountingConstraints, LmCut, OperatorCounting};
use sas_plus::Task;
use sas_plus::random::{RandomTaskParams, random_task};

use crate::{fixtures, oracle};

/// Number of seeds per parameter set.
const SEEDS: u64 = 32;

/// The state equation alone.
const STATE_EQUATION: CountingConstraints = CountingConstraints {
    state_equation: true,
    lm_cut: false,
};

/// LM-cut landmarks alone.
const LANDMARKS: CountingConstraints = CountingConstraints {
    state_equation: false,
    lm_cut: true,
};

/// Checks all three variants on every reachable state of `task`: at most `h*`, only real dead
/// ends, the landmark variants at least LM-cut, and the combination at least each part.
/// Returns how many states the combination beats LM-cut.
fn check_task(name: &str, task: &Task) -> usize {
    let mut lm_cut = LmCut::new(task);
    let mut se = OperatorCounting::new(task, STATE_EQUATION);
    let mut lm = OperatorCounting::new(task, LANDMARKS);
    let mut all = OperatorCounting::new(task, CountingConstraints::ALL);
    let mut above = 0;
    for (state, h_star) in oracle::perfect(task) {
        let ctx = || format!("{name}, state {:?}", state.0);
        let values = [se.value(&state), lm.value(&state), all.value(&state)];
        for v in values {
            match (v, h_star) {
                (None, Some(t)) => panic!("false dead end (h* {t}), {}", ctx()),
                (Some(v), Some(t)) => assert!(v <= t, "{v} > h* {t}, {}", ctx()),
                _ => {}
            }
        }

        let [se, lm, all] = values;
        if let Some(cut) = lm_cut.value(&state) {
            assert!(
                lm.is_none_or(|v| v >= cut) && all.is_none_or(|v| v >= cut),
                "{}",
                ctx()
            );
            above += usize::from(all.is_some_and(|v| v > cut));
        }

        if let (Some(se), Some(lm), Some(all)) = (se, lm, all) {
            assert!(
                all >= se && all >= lm,
                "se {se}, lm {lm}, all {all}, {}",
                ctx()
            );
        }
    }

    above
}

/// Random tasks of every kind and the fixtures; the state equation adds to LM-cut somewhere.
#[test]
fn operator_counting_is_admissible() {
    let kinds = [
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
        (
            "mixed",
            RandomTaskParams {
                num_vars: 5,
                num_ops: 14,
                max_conditions: 2,
                num_derived: 2,
                ..RandomTaskParams::default()
            },
        ),
    ];
    let mut above = 0;
    for (label, params) in kinds {
        for seed in 0..SEEDS {
            above += check_task(&format!("{label}, seed {seed}"), &random_task(seed, params));
        }
    }

    for (name, _) in fixtures::FIXTURES {
        above += check_task(name, &fixtures::load(name));
    }

    assert!(above > 0, "the state equation never adds to LM-cut");
}
