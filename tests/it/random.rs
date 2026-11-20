//! Seeded random tasks from `sas_plus::random`, with and without conditional effects and
//! axioms.

use sas_plus::random::{RandomTaskParams, random_task};

use crate::check::check_task;

/// Number of seeds per parameter set.
const SEEDS: u64 = 64;

/// Checks all properties on [`SEEDS`] random tasks generated with `params`.
fn check_random(label: &str, params: RandomTaskParams) {
    for seed in 0..SEEDS {
        check_task(&format!("{label}, seed {seed}"), &random_task(seed, params));
    }
}

/// Plain SAS+ tasks.
#[test]
fn random_plain() {
    check_random("plain", RandomTaskParams::default());
}

/// Tasks with conditional effects.
#[test]
fn random_conditional_effects() {
    check_random(
        "conditional",
        RandomTaskParams {
            max_conditions: 2,
            ..RandomTaskParams::default()
        },
    );
}

/// Tasks with derived variables and axioms.
#[test]
fn random_axioms() {
    check_random(
        "axioms",
        RandomTaskParams {
            num_derived: 2,
            ..RandomTaskParams::default()
        },
    );
}

/// Tasks with conditional effects, axioms and more operators.
#[test]
fn random_mixed() {
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
