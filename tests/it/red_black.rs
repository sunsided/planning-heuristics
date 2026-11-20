//! The red-black heuristic against `h_FF`, with its plans checked by an independent red-black
//! validator.

use std::collections::BTreeSet;

use planning_heuristics::{HFf, Painting, PaintingError, RedBlack, Relaxation};
use sas_plus::random::{RandomTaskParams, random_task};
use sas_plus::{Axiom, Effect, Fact, OpId, Operator, State, Task, Variable};

use crate::{fixtures, oracle};

/// Number of seeds per parameter set.
const SEEDS: u64 = 64;

/// A red-black state: the red facts reached so far and the value of every black variable.
struct RedBlackState<'t> {
    /// The task.
    task: &'t Task,
    /// Whether each variable is black.
    black: Vec<bool>,
    /// The red facts reached so far, derived defaults included.
    red: BTreeSet<Fact>,
    /// The state's values; only the black entries are read.
    values: Vec<u32>,
}

impl<'t> RedBlackState<'t> {
    /// The red-black state of `state` with the black variables `black`.
    fn new(task: &'t Task, black: &[usize], state: &State) -> Self {
        let mut is_black = vec![false; task.num_variables()];
        for &v in black {
            is_black[v] = true;
        }

        let mut red: BTreeSet<Fact> = (0..task.num_variables())
            .filter(|&v| !is_black[v])
            .map(|v| Fact::new(v, state.0[v]))
            .collect();
        red.extend(
            (0..task.num_variables())
                .filter(|&v| task.variables[v].is_derived())
                .map(|v| Fact::new(v, task.initial.0[v])),
        );
        let mut s = Self {
            task,
            black: is_black,
            red,
            values: state.0.clone(),
        };
        s.close_axioms();

        s
    }

    /// Whether `fact` holds: by value for a black variable, in the red set otherwise.
    fn holds(&self, fact: Fact) -> bool {
        if self.black[fact.var] {
            self.values[fact.var] == fact.value
        } else {
            self.red.contains(&fact)
        }
    }

    /// Applies `op` if its preconditions hold; returns whether they did.
    fn apply(&mut self, op: OpId) -> bool {
        let operator = &self.task.operators[op];
        if !operator.pre.iter().all(|&f| self.holds(f)) {
            return false;
        }

        let fired: Vec<Fact> = operator
            .eff
            .iter()
            .filter(|e| e.conditions.iter().all(|&c| self.holds(c)))
            .map(|e| e.fact)
            .collect();
        for fact in fired {
            if self.black[fact.var] {
                self.values[fact.var] = fact.value;
            } else {
                self.red.insert(fact);
            }
        }

        self.close_axioms();

        true
    }

    /// Adds axiom heads to the red facts until nothing changes.
    fn close_axioms(&mut self) {
        loop {
            let new: Vec<Fact> = self
                .task
                .axioms
                .iter()
                .filter(|a| !self.red.contains(&a.head))
                .filter(|a| a.conditions.iter().all(|&c| self.holds(c)))
                .map(|a| a.head)
                .collect();
            if new.is_empty() {
                return;
            }

            self.red.extend(new);
        }
    }
}

/// Whether `plan` is applicable from `state` under red-black semantics with the black
/// variables `black` and reaches the goal.
fn is_red_black_plan(task: &Task, black: &[usize], state: &State, plan: &[OpId]) -> bool {
    let mut s = RedBlackState::new(task, black, state);
    plan.iter().all(|&op| s.apply(op)) && task.goal.iter().all(|&g| s.holds(g))
}

/// Checks `h_RB` on every reachable state of `task`; returns the number of states where the
/// repaired plan costs more than `h_FF`.
fn check_red_black(name: &str, task: &Task) -> usize {
    let relaxation = Relaxation::new(task);
    let mut h_ff = HFf::with_relaxation(&relaxation);
    let mut h_rb = RedBlack::with_relaxation(task, &relaxation, &Painting::greedy(task));
    let mut all_red = RedBlack::with_relaxation(task, &relaxation, &Painting::all_red());
    let black = h_rb.black_variables().to_vec();
    let mut above_ff = 0;

    for state in oracle::perfect(task).keys() {
        let ctx = || format!("{name}, state {:?}, black {black:?}", state.0);
        let ff = h_ff.value(state);
        assert_eq!(all_red.value(state), ff, "all red is h_FF, {}", ctx());
        let plan = h_rb.plan(state);
        assert_eq!(plan.is_none(), ff.is_none(), "dead ends, {}", ctx());
        let (Some(plan), Some(ff)) = (plan, ff) else {
            continue;
        };

        assert_eq!(h_rb.value(state), Some(plan.cost()), "{}", ctx());
        assert!(
            plan.cost() >= ff,
            "h_RB {} < h_FF {ff}, {}",
            plan.cost(),
            ctx()
        );
        if plan.is_repaired() {
            assert!(
                is_red_black_plan(task, &black, state, plan.ops()),
                "{}",
                ctx()
            );
            assert_eq!(
                oracle::plan_cost(task, plan.ops()),
                plan.cost(),
                "{}",
                ctx()
            );
            above_ff += usize::from(plan.cost() > ff);
        } else {
            assert_eq!(plan.cost(), ff, "fallback is h_FF, {}", ctx());
        }
    }

    above_ff
}

/// Checks [`SEEDS`] random tasks generated with `params`; returns the number of states where
/// `h_RB > h_FF`.
fn check_random(label: &str, params: RandomTaskParams) -> usize {
    (0..SEEDS)
        .map(|seed| check_red_black(&format!("{label}, seed {seed}"), &random_task(seed, params)))
        .sum()
}

/// Plain random tasks; the black variables make a difference on some states.
#[test]
fn red_black_on_random_plain_tasks() {
    let above = check_random("plain", RandomTaskParams::default());
    assert!(above > 0, "black variables never mattered");
}

/// Random tasks with conditional effects and axioms (their variables stay red).
#[test]
fn red_black_on_random_mixed_tasks() {
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

/// Every fixture; on the gripper and courier fixtures the black variables matter.
#[test]
fn red_black_on_fixtures() {
    let above: usize = fixtures::FIXTURES
        .iter()
        .map(|(name, _)| check_red_black(name, &fixtures::load(name)))
        .sum();
    assert!(above > 0, "black variables never mattered");
}

/// A task over binary variables `v0..vn` with the given operators.
fn binary_task(num_vars: usize, operators: Vec<Operator>) -> Task {
    Task {
        variables: (0..num_vars)
            .map(|i| Variable::new(format!("v{i}"), 2))
            .collect(),
        operators,
        initial: State(vec![0; num_vars]),
        goal: vec![Fact::new(0, 1)],
        axioms: vec![],
    }
}

/// An operator with the given precondition and effects, cost 1.
fn op(pre: Vec<Fact>, eff: Vec<Effect>) -> Operator {
    Operator {
        name: "op".into(),
        pre,
        eff,
        cost: 1,
    }
}

/// Paintings that break the restrictions are rejected with the reason.
#[test]
fn invalid_paintings_are_rejected() {
    let set_both = op(
        vec![],
        vec![Effect::new(Fact::new(0, 1)), Effect::new(Fact::new(1, 1))],
    );
    let conditional = op(
        vec![],
        vec![Effect {
            conditions: vec![Fact::new(2, 1)],
            fact: Fact::new(3, 1),
        }],
    );
    let mut task = binary_task(5, vec![set_both, conditional]);
    task.variables[4].axiom_layer = Some(0);
    task.axioms.push(Axiom {
        conditions: vec![Fact::new(1, 0)],
        head: Fact::new(4, 1),
    });

    let new = |black: Vec<usize>| Painting::new(&task, black);
    assert_eq!(new(vec![0, 1]), Err(PaintingError::InAxiom(1)));
    assert_eq!(new(vec![0, 2]), Err(PaintingError::ConditionalEffect(2)));
    assert_eq!(new(vec![3]), Err(PaintingError::ConditionalEffect(3)));
    assert_eq!(new(vec![4]), Err(PaintingError::Derived(4)));
    assert_eq!(new(vec![7]), Err(PaintingError::UnknownVariable(7)));
    assert!(new(vec![0]).is_ok());

    task.axioms.clear();
    assert_eq!(
        Painting::new(&task, vec![0, 1]),
        Err(PaintingError::Dependent(0, 1))
    );
}

/// The repair falls back to `h_FF` when a black variable cannot move back: here the relaxed
/// plan turns the switch on before fetching `b`, which needs it off, and nothing turns it off.
#[test]
fn failed_repair_falls_back_to_h_ff() {
    let on = op(vec![Fact::new(0, 0)], vec![Effect::new(Fact::new(0, 1))]);
    let fetch_a = op(vec![Fact::new(0, 1)], vec![Effect::new(Fact::new(1, 1))]);
    let fetch_b = Operator {
        cost: 5,
        ..op(vec![Fact::new(0, 0)], vec![Effect::new(Fact::new(2, 1))])
    };
    let mut task = binary_task(3, vec![on, fetch_a, fetch_b]);
    task.goal = vec![Fact::new(1, 1), Fact::new(2, 1)];
    let state = task.initial_state();

    let painting = Painting::new(&task, vec![0]).unwrap();
    let plan = RedBlack::with_painting(&task, &painting)
        .plan(&state)
        .unwrap();

    assert!(!plan.is_repaired());
    assert_eq!(plan.ops(), [0, 1, 2]);
    assert_eq!(plan.cost(), 7);
    assert_eq!(HFf::new(&task).value(&state), Some(7));
}

/// Two black switches: a red operator needs both on, the goal needs the first off again.
/// The relaxed plan (`on0, on1, finish`, cost 3) is repaired by appending `off0`.
#[test]
fn two_black_variables_are_repaired_independently() {
    let set = |var: usize, from: u32, to: u32| {
        op(
            vec![Fact::new(var, from)],
            vec![Effect::new(Fact::new(var, to))],
        )
    };
    let finish = op(
        vec![Fact::new(0, 1), Fact::new(1, 1)],
        vec![Effect::new(Fact::new(2, 1))],
    );
    let mut task = binary_task(3, vec![set(0, 0, 1), set(1, 0, 1), set(0, 1, 0), finish]);
    task.goal = vec![Fact::new(2, 1), Fact::new(0, 0)];
    let state = task.initial_state();

    let painting = Painting::new(&task, vec![0, 1]).unwrap();
    let plan = RedBlack::with_painting(&task, &painting)
        .plan(&state)
        .unwrap();

    assert_eq!(HFf::new(&task).value(&state), Some(3));
    assert!(plan.is_repaired());
    assert_eq!(plan.cost(), 4);
    assert!(is_red_black_plan(&task, &[0, 1], &state, plan.ops()));
}
