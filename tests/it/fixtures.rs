//! The hand-encoded and translated `output.sas` fixtures copied from sas-plus (MIT OR
//! Apache-2.0 only; see `tests/fixtures/README.md`).

use planning_heuristics::HMax;
use sas_plus::Task;
use sas_plus::fdr::SasFile;

use crate::check::check_task;
use crate::oracle;

/// Fixture name and `output.sas` contents.
pub const FIXTURES: &[(&str, &str)] = &[
    (
        "gripper-costs",
        include_str!("../fixtures/gripper-costs/output.sas"),
    ),
    (
        "gripper-unit",
        include_str!("../fixtures/gripper-unit/output.sas"),
    ),
    (
        "conditional-effects",
        include_str!("../fixtures/conditional-effects/output.sas"),
    ),
    (
        "derived-predicates",
        include_str!("../fixtures/derived-predicates/output.sas"),
    ),
    (
        "translated-courier",
        include_str!("../fixtures/translated-courier/output.sas"),
    ),
    (
        "translated-gripper-costs",
        include_str!("../fixtures/translated-gripper-costs/output.sas"),
    ),
    (
        "translated-wiring",
        include_str!("../fixtures/translated-wiring/output.sas"),
    ),
];

/// Parses the fixture `name`.
pub fn load(name: &str) -> Task {
    let (_, text) = FIXTURES
        .iter()
        .find(|(n, _)| *n == name)
        .expect("known fixture");

    SasFile::parse(text).expect("fixture parses").to_task()
}

/// All properties hold on every reachable state of every fixture.
#[test]
fn properties_hold_on_fixtures() {
    for (name, _) in FIXTURES {
        assert!(check_task(name, &load(name)) > 0);
    }
}

/// The oracle reproduces the optimal costs listed in the sas-plus fixture README.
#[test]
fn perfect_heuristic_matches_known_optima() {
    for (name, optimum) in [
        ("gripper-costs", 7),
        ("gripper-unit", 5),
        ("derived-predicates", 2),
        ("translated-courier", 16),
        ("translated-wiring", 4),
    ] {
        let task = load(name);
        let h_star = oracle::perfect(&task)[&task.initial_state()];
        assert_eq!(h_star, Some(optimum), "{name}");
    }
}

/// `h_max` is positive on the initial state of a fixture that needs actions.
#[test]
fn h_max_is_informative_on_gripper() {
    let task = load("gripper-costs");
    let value = HMax::new(&task).value(&task.initial_state());
    assert!(value.is_some_and(|v| v > 0));
}
