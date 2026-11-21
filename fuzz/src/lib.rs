//! Invariants the fuzz targets check on every task they build.
//!
//! The targets only differ in how they build a [`Task`]: [`check_task`] validates it, bounds
//! its size so one input stays fast, walks a few states from the initial state and checks
//! the delete-relaxation heuristics and LM-cut against each other on every state of the walk.

use planning_heuristics::{HAdd, HFf, HMax, LmCut, Relaxation};
use sas_plus::{State, Task};

/// Most variables a fuzzed task may have; larger tasks are skipped.
const MAX_VARIABLES: usize = 16;

/// Most facts (summed domain sizes) a fuzzed task may have; larger tasks are skipped.
const MAX_FACTS: u64 = 96;

/// Most operators plus axioms a fuzzed task may have; larger tasks are skipped.
const MAX_RULES: usize = 96;

/// Most states checked per input, the initial state included.
const MAX_WALK: usize = 16;

/// Checks the heuristics on `task` along the random walk that `choices` steers.
///
/// Invalid or oversized tasks are skipped. Each byte of `choices` picks one applicable
/// operator (modulo their count); the walk stops early in a state without any.
///
/// # Panics
///
/// Panics when an invariant of [`check_state`] fails, which is the finding.
pub fn check_task(task: &Task, choices: &[u8]) {
    if task.validate().is_err() || is_too_large(task) {
        return;
    }

    let relaxation = Relaxation::new(task);
    let mut heuristics = Heuristics {
        h_max: HMax::with_relaxation(&relaxation),
        h_add: HAdd::with_relaxation(&relaxation),
        h_ff: HFf::with_relaxation(&relaxation),
        lm_cut: LmCut::new(task),
    };

    let mut state = task.initial_state();
    check_state(task, &mut heuristics, &state);
    for &choice in choices.iter().take(MAX_WALK - 1) {
        let Some(next) = successor(task, &state, choice) else {
            break;
        };

        state = next;
        check_state(task, &mut heuristics, &state);
    }
}

/// The heuristics under test, sharing one [`Relaxation`] where they can.
struct Heuristics<'r> {
    /// `h_max` over the shared relaxation.
    h_max: HMax<'r>,
    /// `h_add` over the shared relaxation.
    h_add: HAdd<'r>,
    /// `h_FF` over the shared relaxation.
    h_ff: HFf<'r>,
    /// LM-cut with its own, context-split relaxation.
    lm_cut: LmCut<'r>,
}

/// Whether `task` exceeds the size bounds that keep one fuzz input fast.
fn is_too_large(task: &Task) -> bool {
    let facts: u64 = task
        .variables
        .iter()
        .map(|v| u64::from(v.domain_size))
        .sum();

    task.variables.len() > MAX_VARIABLES
        || facts > MAX_FACTS
        || task.operators.len() + task.axioms.len() > MAX_RULES
}

/// The state reached by the applicable operator that `choice` picks, or `None` if `state`
/// has no applicable operator.
fn successor(task: &Task, state: &State, choice: u8) -> Option<State> {
    let applicable: Vec<_> = task.applicable_operators(state).collect();
    if applicable.is_empty() {
        return None;
    }

    let op = applicable[usize::from(choice) % applicable.len()];

    Some(task.apply(op, state))
}

/// Checks one state: the delete-relaxation heuristics agree on dead ends,
/// `h_max <= h_FF <= h_add`, the relaxed plan costs `h_FF`, goal states are worth 0, and
/// LM-cut never misses a dead end `h_max` proves and never falls below `h_max`.
///
/// # Panics
///
/// Panics with the violated invariant and the state.
fn check_state(task: &Task, heuristics: &mut Heuristics<'_>, state: &State) {
    let max = heuristics.h_max.value(state);
    let add = heuristics.h_add.value(state);
    let ff = heuristics.h_ff.value(state);
    let lm_cut = heuristics.lm_cut.value(state);

    assert_eq!(
        max.is_none(),
        add.is_none(),
        "dead ends, state {:?}",
        state.0
    );
    assert_eq!(
        max.is_none(),
        ff.is_none(),
        "dead ends, state {:?}",
        state.0
    );
    assert!(
        max.is_none() <= lm_cut.is_none(),
        "LM-cut misses an h_max dead end, state {:?}",
        state.0
    );

    if task.is_goal(state) {
        assert_eq!(max, Some(0), "goal state, state {:?}", state.0);
        assert_eq!(add, Some(0), "goal state, state {:?}", state.0);
        assert_eq!(lm_cut, Some(0), "goal state, state {:?}", state.0);
    }

    let (Some(max), Some(add), Some(ff)) = (max, add, ff) else {
        return;
    };

    assert!(
        max <= ff && ff <= add,
        "h_max {max}, h_FF {ff}, h_add {add}, state {:?}",
        state.0
    );
    if let Some(lm_cut) = lm_cut {
        assert!(
            max <= lm_cut,
            "h_max {max} > h_LM-cut {lm_cut}, state {:?}",
            state.0
        );
    }

    check_relaxed_plan(task, heuristics, state, ff);
}

/// Checks that the relaxed plan `h_FF` extracts in `state` costs exactly `ff`, by its own
/// account and summed over its operators.
///
/// # Panics
///
/// Panics if the plan is missing or its cost differs from `ff`.
fn check_relaxed_plan(task: &Task, heuristics: &mut Heuristics<'_>, state: &State, ff: u64) {
    let plan = heuristics
        .h_ff
        .relaxed_plan(state)
        .expect("h_FF is finite, so a relaxed plan exists");
    let summed: u64 = plan
        .ops()
        .iter()
        .map(|&op| u64::from(task.operators[op].cost))
        .sum();

    assert_eq!(plan.cost(), ff, "plan cost, state {:?}", state.0);
    assert_eq!(summed, ff, "summed plan cost, state {:?}", state.0);
}
