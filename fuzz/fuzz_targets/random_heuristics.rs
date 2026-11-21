//! Fuzzes the heuristics on `sas_plus::random` tasks with fuzzed seeds and parameters,
//! including conditional effects and axioms.

#![no_main]

use libfuzzer_sys::fuzz_target;
use planning_heuristics_fuzz::check_task;
use sas_plus::random::{RandomTaskParams, random_task};

/// Seed, the six [`RandomTaskParams`] bytes (each reduced to a small range), and the walk
/// choices for [`check_task`].
type Input<'a> = (u64, [u8; 6], &'a [u8]);

fuzz_target!(|input: Input<'_>| {
    let (seed, [vars, domain, ops, cost, conditions, derived], choices) = input;
    let params = RandomTaskParams {
        num_vars: 1 + usize::from(vars % 8),
        max_domain: 2 + u32::from(domain % 4),
        num_ops: 1 + usize::from(ops % 24),
        max_cost: 1 + u32::from(cost % 9),
        max_conditions: usize::from(conditions % 3),
        num_derived: usize::from(derived % 3),
    };

    check_task(&random_task(seed, params), choices);
});
