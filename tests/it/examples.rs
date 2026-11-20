//! Hand-made tasks with known values for the corner cases of the relaxation.

use planning_heuristics::{HAdd, HFf, HMax, Heuristic};
use sas_plus::{Axiom, Effect, Fact, Operator, State, Task, Variable};

use crate::check::check_task;

/// A task over binary state variables with the given operators and goal.
fn binary_task(num_vars: usize, operators: Vec<Operator>, goal: Vec<Fact>) -> Task {
    Task {
        variables: (0..num_vars)
            .map(|i| Variable::new(format!("v{i}"), 2))
            .collect(),
        operators,
        initial: State(vec![0; num_vars]),
        goal,
        axioms: vec![],
    }
}

/// An operator without precondition.
fn op(eff: Vec<Effect>, cost: u32) -> Operator {
    Operator {
        name: "op".into(),
        pre: vec![],
        eff,
        cost,
    }
}

/// An operator whose conditional effect needs its own unconditional effect is applied twice
/// in the relaxed plan: `h_FF = 2`, not 1, which would undercut `h_max = 2`.
#[test]
fn self_dependent_conditional_effect_repeats_operator() {
    let set_b_if_a = Effect {
        conditions: vec![Fact::new(0, 1)],
        fact: Fact::new(1, 1),
    };
    let task = binary_task(
        2,
        vec![op(vec![Effect::new(Fact::new(0, 1)), set_b_if_a], 1)],
        vec![Fact::new(1, 1)],
    );
    let state = task.initial_state();

    assert_eq!(HMax::new(&task).value(&state), Some(2));
    assert_eq!(HAdd::new(&task).value(&state), Some(2));
    let plan = HFf::new(&task).relaxation().relaxed_plan(&state).unwrap();
    assert_eq!(plan.ops(), [0, 0]);
    assert_eq!(plan.cost(), 2);
    check_task("self-dependent", &task);
}

/// `h_add` counts a shared precondition twice, `h_FF` once, `h_max` takes the longer chain.
#[test]
fn shared_subgoal_separates_the_heuristics() {
    let operators = vec![
        op(vec![Effect::new(Fact::new(0, 1))], 2),
        Operator {
            pre: vec![Fact::new(0, 1)],
            ..op(vec![Effect::new(Fact::new(1, 1))], 1)
        },
        Operator {
            pre: vec![Fact::new(0, 1)],
            ..op(vec![Effect::new(Fact::new(2, 1))], 1)
        },
    ];
    let task = binary_task(3, operators, vec![Fact::new(1, 1), Fact::new(2, 1)]);
    let state = task.initial_state();

    assert_eq!(HMax::new(&task).value(&state), Some(3));
    assert_eq!(HFf::new(&task).value(&state), Some(4));
    assert_eq!(HAdd::new(&task).value(&state), Some(6));
    check_task("shared-subgoal", &task);
}

/// A goal value no rule achieves makes every heuristic report a dead end.
#[test]
fn unreachable_goal_is_a_dead_end() {
    let task = binary_task(
        2,
        vec![op(vec![Effect::new(Fact::new(0, 1))], 1)],
        vec![Fact::new(1, 1)],
    );
    let state = task.initial_state();

    assert_eq!(HMax::new(&task).value(&state), None);
    assert_eq!(HAdd::new(&task).value(&state), None);
    assert_eq!(HFf::new(&task).value(&state), None);
}

/// The default of a derived variable is free: here it holds again only after an operator
/// switches its axiom off, and treating it as unreachable would be a false dead end.
#[test]
fn derived_default_is_free() {
    let mut task = binary_task(
        3,
        vec![
            op(vec![Effect::new(Fact::new(0, 1))], 1),
            Operator {
                pre: vec![Fact::new(1, 0)],
                ..op(vec![Effect::new(Fact::new(2, 1))], 1)
            },
        ],
        vec![Fact::new(2, 1)],
    );
    task.variables[1].axiom_layer = Some(0);
    task.axioms.push(Axiom {
        conditions: vec![Fact::new(0, 0)],
        head: Fact::new(1, 1),
    });
    let state = task.initial_state();
    assert_eq!(state.0, [0, 1, 0]);

    assert_eq!(HMax::new(&task).value(&state), Some(1));
    assert_eq!(HFf::new(&task).value(&state), Some(1));
    check_task("derived-default", &task);
}

/// A satisfied goal costs nothing and needs an empty relaxed plan.
#[test]
fn goal_state_has_empty_plan() {
    let task = binary_task(1, vec![], vec![Fact::new(0, 0)]);
    let state = task.initial_state();

    assert_eq!(HAdd::new(&task).value(&state), Some(0));
    let plan = HFf::new(&task).relaxation().relaxed_plan(&state).unwrap();
    assert_eq!(plan.ops(), []);
}

/// Sums beyond `u32` stay exact in `u64` and saturate, not vanish, in the `u32` estimate.
#[test]
fn u32_estimate_saturates() {
    let task = binary_task(
        2,
        vec![
            op(vec![Effect::new(Fact::new(0, 1))], u32::MAX),
            op(vec![Effect::new(Fact::new(1, 1))], u32::MAX),
        ],
        vec![Fact::new(0, 1), Fact::new(1, 1)],
    );
    let state = task.initial_state();
    let h_add = HAdd::new(&task);

    assert_eq!(h_add.value(&state), Some(2 * u64::from(u32::MAX)));
    assert_eq!(
        Heuristic::<_, u32>::estimate(&h_add, &state),
        Some(u32::MAX)
    );
}
