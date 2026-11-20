//! The red-black heuristic against `h_FF`, with its plans checked by an independent red-black
//! validator.

use std::collections::BTreeSet;

use planning_heuristics::{
    CausalLevel, Fragment, HFf, HMax, Painting, PaintingError, RedBlack, RedBlackOptions,
    Relaxation, Strategy,
};
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

/// The painting strategies checked, by label; those ending in `-dag` paint an acyclic black
/// causal graph, the others an arcless one.
const STRATEGIES: [&str; 20] = [
    "greedy",
    "greedy-dag",
    "top",
    "top-dag",
    "bottom",
    "bottom-dag",
    "arcs",
    "arcs-dag",
    "fewest-arcs",
    "fewest-arcs-dag",
    "conflicts",
    "conflicts-dag",
    "most-conflicts",
    "most-conflicts-dag",
    "sampled",
    "sampled-dag",
    "mixed",
    "mixed-dag",
    "random",
    "random-dag",
];

/// The painting of `task` by strategy `label` (see [`STRATEGIES`]).
fn paint(task: &Task, label: &str) -> Painting {
    let fragment = if label.ends_with("-dag") {
        Fragment::Dag
    } else {
        Fragment::Arcless
    };
    match label.trim_end_matches("-dag") {
        "greedy" if fragment == Fragment::Dag => Painting::greedy_dag(task),
        "greedy" => Painting::greedy(task),
        "top" => Painting::causal_graph_level(task, CausalLevel::Top, fragment),
        "bottom" => Painting::causal_graph_level(task, CausalLevel::Bottom, fragment),
        "arcs" => Painting::incident_arcs(task, fragment),
        "conflicts" => Painting::conflict_directed(task, &task.initial_state(), fragment),
        other => {
            let strategy = match other {
                "fewest-arcs" => Strategy::FewestArcs,
                "most-conflicts" => Strategy::MostConflicts,
                "sampled" => Strategy::SampledConflicts {
                    samples: 5,
                    seed: 1,
                },
                "mixed" => Strategy::Mixed { arcs_percent: 50 },
                "random" => Strategy::Random { seed: 3 },
                _ => unreachable!("unknown strategy {label}"),
            };
            Painting::with_strategy(task, &task.initial_state(), strategy, fragment)
        }
    }
}

/// Checks `h_RB` with every strategy on every reachable state of `task`; returns, per entry of
/// [`STRATEGIES`], the number of states where the repaired plan costs more than `h_FF`.
///
/// Every strategy paints only RSE-invertible variables, so the repair must always succeed
/// (Domshlak, Hoffmann & Katz 2015, Corollary 1 and Theorem 11).
fn check_red_black(name: &str, task: &Task) -> [usize; STRATEGIES.len()] {
    check_painting(&format!("{name}, any"), task, &any_dag(task), false);

    STRATEGIES.map(|label| {
        let painting = paint(task, label);
        let black = painting.black().to_vec();
        if label.ends_with("-dag") {
            Painting::dag(task, black).expect("a DAG strategy paints a DAG");
        } else {
            Painting::new(task, black).expect("strategies paint independently");
        }

        check_painting(&format!("{name}, {label}"), task, &painting, true)
    })
}

/// A DAG painting that ignores RSE-invertibility: every variable, in id order, that
/// [`Painting::dag`] accepts. Its repairs may fail and fall back to `h_FF`.
fn any_dag(task: &Task) -> Painting {
    let mut black = Vec::new();
    for v in 0..task.num_variables() {
        black.push(v);
        if Painting::dag(task, black.clone()).is_err() {
            black.pop();
        }
    }

    Painting::dag(task, black).expect("built from accepted sets")
}

/// Adds the per-strategy counts `b` to `a`.
fn add(
    mut a: [usize; STRATEGIES.len()],
    b: [usize; STRATEGIES.len()],
) -> [usize; STRATEGIES.len()] {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }

    a
}

/// Asserts that every strategy on its own raised `h_RB` above `h_FF` somewhere.
fn assert_each_matters(above: [usize; STRATEGIES.len()]) {
    for (label, count) in STRATEGIES.iter().zip(above) {
        assert!(
            count > 0,
            "{label}: black variables never mattered, {above:?}"
        );
    }
}

/// The [`RedBlackOptions`] every painting is checked with, by label.
const VARIANTS: [(&str, RedBlackOptions); 3] = [
    (
        "repair",
        RedBlackOptions {
            forward: false,
            follow: false,
        },
    ),
    (
        "forward",
        RedBlackOptions {
            forward: true,
            follow: false,
        },
    ),
    (
        "follow",
        RedBlackOptions {
            forward: false,
            follow: true,
        },
    ),
];

/// Checks `h_RB` with `painting` under every entry of [`VARIANTS`]; returns the number of
/// (variant, state) pairs where the red-black plan costs more than `h_FF`.
fn check_painting(name: &str, task: &Task, painting: &Painting, in_fragment: bool) -> usize {
    VARIANTS
        .iter()
        .map(|&(label, options)| {
            check_variant(
                &format!("{name}, {label}"),
                task,
                painting,
                in_fragment,
                options,
            )
        })
        .sum()
}

/// Checks `h_RB` with `painting` and `options` on every reachable state of `task`; returns
/// the number of states where the red-black plan costs more than `h_FF`.
///
/// With `in_fragment` every repair must succeed; otherwise a failed repair must cost `h_FF`.
/// A plan that claims to be a real plan (stop search) must be one, checked against `h*`.
fn check_variant(
    name: &str,
    task: &Task,
    painting: &Painting,
    in_fragment: bool,
    options: RedBlackOptions,
) -> usize {
    let relaxation = Relaxation::new(task);
    let mut h_ff = HFf::with_relaxation(&relaxation);
    let mut h_max = HMax::with_relaxation(&relaxation);
    let mut h_rb = RedBlack::with_relaxation(task, &relaxation, painting).with_options(options);
    let mut all_red = RedBlack::with_relaxation(task, &relaxation, &Painting::all_red());
    let black = h_rb.black_variables().to_vec();
    let mut above_ff = 0;

    for (state, h_star) in &oracle::perfect(task) {
        let ctx = || format!("{name}, state {:?}, black {black:?}", state.0);
        let ff = h_ff.value(state);
        assert_eq!(all_red.value(state), ff, "all red is h_FF, {}", ctx());
        let plan = h_rb.plan(state);
        assert_eq!(plan.is_none(), ff.is_none(), "dead ends, {}", ctx());
        let (Some(plan), Some(ff)) = (plan, ff) else {
            continue;
        };

        assert_eq!(h_rb.value(state), Some(plan.cost()), "{}", ctx());
        // Following need not contain the relaxed plan; any red-black plan is a relaxed plan.
        let bound = if options.follow {
            h_max.value(state).expect("not a dead end")
        } else {
            ff
        };
        assert!(
            plan.cost() >= bound,
            "h_RB {} < {bound}, {}",
            plan.cost(),
            ctx()
        );
        if !plan.is_repaired() {
            assert!(!in_fragment, "repair failed in the fragment, {}", ctx());
            assert_eq!(plan.cost(), ff, "fallback is h_FF, {}", ctx());
            continue;
        }

        assert!(
            is_red_black_plan(task, &black, state, plan.ops()),
            "{}",
            ctx()
        );
        if plan.is_plan_for(task, state) {
            assert!(
                h_star.is_some_and(|h| h <= plan.cost()),
                "real plan cheaper than h* {h_star:?}, {}",
                ctx()
            );
        }

        assert_eq!(
            oracle::plan_cost(task, plan.ops()),
            plan.cost(),
            "{}",
            ctx()
        );
        above_ff += usize::from(plan.cost() > ff);
    }

    above_ff
}

/// Checks [`SEEDS`] random tasks generated with `params`; returns, per strategy, the number of
/// states where `h_RB > h_FF`.
fn check_random(label: &str, params: RandomTaskParams) -> [usize; STRATEGIES.len()] {
    (0..SEEDS)
        .map(|seed| check_red_black(&format!("{label}, seed {seed}"), &random_task(seed, params)))
        .fold([0; STRATEGIES.len()], add)
}

/// Plain random tasks; with every strategy, the black variables make a difference on some
/// states.
#[test]
fn red_black_on_random_plain_tasks() {
    let above = check_random("plain", RandomTaskParams::default());
    eprintln!("plain: {above:?}");
    assert_each_matters(above);
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

/// Every fixture; with every strategy, the black variables matter on some states.
#[test]
fn red_black_on_fixtures() {
    let above = fixtures::FIXTURES
        .iter()
        .map(|(name, _)| check_red_black(name, &fixtures::load(name)))
        .fold([0; STRATEGIES.len()], add);
    eprintln!("fixtures: {above:?}");
    assert_each_matters(above);
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

/// A truck and a package on two locations; the truck must end where it started. The truck is
/// at the top of the causal graph, the package at the bottom, and only the truck conflicts with
/// the relaxed plan (it must return), so the conflict-directed painting picks it and `h_RB`
/// counts the drive back.
#[test]
fn painting_strategies_hand_checked() {
    let (a, b, in_truck) = (0, 1, 2);
    let op = |name: &str, pre: Vec<Fact>, eff: Vec<Fact>| Operator {
        name: name.into(),
        pre,
        eff: eff.into_iter().map(Effect::new).collect(),
        cost: 1,
    };
    let task = Task {
        variables: vec![Variable::new("truck", 2), Variable::new("package", 3)],
        operators: vec![
            op("drive-a-b", vec![Fact::new(0, a)], vec![Fact::new(0, b)]),
            op("drive-b-a", vec![Fact::new(0, b)], vec![Fact::new(0, a)]),
            op(
                "load-a",
                vec![Fact::new(0, a), Fact::new(1, a)],
                vec![Fact::new(1, in_truck)],
            ),
            op(
                "load-b",
                vec![Fact::new(0, b), Fact::new(1, b)],
                vec![Fact::new(1, in_truck)],
            ),
            op(
                "unload-a",
                vec![Fact::new(0, a), Fact::new(1, in_truck)],
                vec![Fact::new(1, a)],
            ),
            op(
                "unload-b",
                vec![Fact::new(0, b), Fact::new(1, in_truck)],
                vec![Fact::new(1, b)],
            ),
        ],
        initial: State(vec![a, a]),
        goal: vec![Fact::new(1, b), Fact::new(0, a)],
        axioms: vec![],
    };
    let state = task.initial_state();

    for fragment in [Fragment::Arcless, Fragment::Dag] {
        for keep in [CausalLevel::Top, CausalLevel::Bottom] {
            let painting = Painting::causal_graph_level(&task, keep, fragment);
            assert_eq!(painting.black(), [0], "the package is a leaf, so red");
        }

        assert_eq!(Painting::incident_arcs(&task, fragment).black(), [0]);
    }

    let conflicts = Painting::conflict_directed(&task, &state, Fragment::Arcless);
    assert_eq!(conflicts.black(), [0]);

    assert_eq!(HFf::new(&task).value(&state), Some(3));
    let mut h_rb = RedBlack::with_painting(&task, &conflicts);
    assert_eq!(h_rb.value(&state), Some(4));
    let plan = h_rb.plan(&state).unwrap();
    assert!(
        plan.is_plan_for(&task, &state),
        "stop search fires: {plan:?}"
    );
    let relaxed = RedBlack::with_painting(&task, &Painting::all_red())
        .plan(&state)
        .unwrap();
    assert!(!relaxed.is_plan_for(&task, &state), "not real: {relaxed:?}");
}

/// A DAG painting: `b` can only move while `a = 1`, both black. The repair moves `a` up for
/// `b`'s transition and back down for the goal; the plan is valid red-black and costs 3, and a
/// cycle (two operators linking both ways) is rejected.
#[test]
fn dag_painting_moves_ancestors() {
    let ops = vec![
        op(vec![Fact::new(0, 0)], vec![Effect::new(Fact::new(0, 1))]),
        op(vec![Fact::new(0, 1)], vec![Effect::new(Fact::new(0, 0))]),
        op(
            vec![Fact::new(0, 1), Fact::new(1, 0)],
            vec![Effect::new(Fact::new(1, 1))],
        ),
    ];
    let mut task = binary_task(2, ops);
    task.goal = vec![Fact::new(1, 1), Fact::new(0, 0)];
    let painting = Painting::dag(&task, vec![0, 1]).unwrap();
    let state = task.initial_state();
    let mut h_rb = RedBlack::with_painting(&task, &painting);
    let plan = h_rb.plan(&state).unwrap();

    assert!(plan.is_repaired());
    assert!(is_red_black_plan(&task, &[0, 1], &state, plan.ops()));
    assert_eq!(plan.cost(), 3);

    let mut cyclic = task.clone();
    cyclic.operators.push(op(
        vec![Fact::new(1, 1)],
        vec![Effect::new(Fact::new(0, 0))],
    ));
    assert!(matches!(
        Painting::dag(&cyclic, vec![0, 1]),
        Err(PaintingError::Cyclic(_, _))
    ));
}

/// Regression: in the DAG fragment the repair must only use transitions whose black
/// preconditions were already visited (the projected task `Π^B` of Domshlak, Hoffmann & Katz
/// 2015, Figure 4). Black `a` (0, 1, 2) is the parent of black `b`; `a = 2` needs a red fact
/// nothing achieves. Moving `b` back to 0 for the goal is cheap via `a = 2` and dear via the
/// visited `a = 1`; an unrestricted path search picks the cheap one, cannot move `a` to 2 and
/// falls back to `h_FF`. Both variables are RSE-invertible, so the paper guarantees a repair.
#[test]
fn dag_repair_stays_on_visited_values() {
    let (a, b, r, q) = (0, 1, 2, 3);
    let step = |pre: Vec<Fact>, eff: Fact, cost: u32| Operator {
        cost,
        ..op(pre, vec![Effect::new(eff)])
    };
    let task = Task {
        variables: vec![
            Variable::new("a", 3),
            Variable::new("b", 2),
            Variable::new("r", 2),
            Variable::new("q", 2),
        ],
        operators: vec![
            step(vec![Fact::new(a, 0)], Fact::new(a, 1), 1),
            step(vec![Fact::new(a, 1)], Fact::new(a, 0), 1),
            step(vec![Fact::new(a, 1), Fact::new(r, 1)], Fact::new(a, 2), 1),
            step(vec![Fact::new(a, 2), Fact::new(r, 1)], Fact::new(a, 1), 1),
            step(vec![Fact::new(a, 2), Fact::new(b, 0)], Fact::new(b, 1), 1),
            step(vec![Fact::new(a, 2), Fact::new(b, 1)], Fact::new(b, 0), 1),
            step(vec![Fact::new(a, 1), Fact::new(b, 0)], Fact::new(b, 1), 10),
            step(vec![Fact::new(a, 1), Fact::new(b, 1)], Fact::new(b, 0), 10),
            step(vec![Fact::new(b, 1)], Fact::new(q, 1), 1),
        ],
        initial: State(vec![0, 0, 0, 0]),
        goal: vec![Fact::new(q, 1), Fact::new(b, 0)],
        axioms: vec![],
    };
    let state = task.initial_state();
    let painting = Painting::dag(&task, vec![a, b]).unwrap();
    let plan = RedBlack::with_painting(&task, &painting)
        .plan(&state)
        .unwrap();

    assert_eq!(HFf::new(&task).value(&state), Some(12));
    assert!(plan.is_repaired());
    assert!(is_red_black_plan(&task, &[a, b], &state, plan.ops()));
    assert_eq!(plan.cost(), 22);
}

/// Regression: a conditional side effect does not count towards RSE-invertibility. Turning
/// `v` on may set `u` (only if `c`, which never holds), and turning it off needs `u`; so `v`
/// cannot move back and must not be painted black, or the repair falls back inside the
/// claimed fragment.
#[test]
fn conditional_side_effects_do_not_invert() {
    let (v, u, c, g) = (0, 1, 2, 3);
    let up = op(
        vec![Fact::new(v, 0)],
        vec![
            Effect::new(Fact::new(v, 1)),
            Effect {
                conditions: vec![Fact::new(c, 1)],
                fact: Fact::new(u, 1),
            },
        ],
    );
    let down = op(
        vec![Fact::new(v, 1), Fact::new(u, 1)],
        vec![Effect::new(Fact::new(v, 0))],
    );
    let get_g = op(vec![Fact::new(v, 1)], vec![Effect::new(Fact::new(g, 1))]);
    let mut task = binary_task(4, vec![up, down, get_g]);
    task.goal = vec![Fact::new(g, 1), Fact::new(v, 0)];

    assert_eq!(Painting::greedy(&task).black(), [] as [usize; 0]);
    let explicit = Painting::new(&task, vec![v]).unwrap();
    let plan = RedBlack::with_painting(&task, &explicit)
        .plan(&task.initial_state())
        .unwrap();
    assert!(!plan.is_repaired());
}

/// A small `StarShapeLogistics` (Domshlak, Hoffmann & Katz 2015, Example 7): a truck (0) at the
/// centre (value 0) with spokes 1 and 2, a hand (1, 1 = free) holding one package at a time,
/// and packages 2 and 3 (0 = centre, 1 / 2 = spoke, 3 = in the truck) to bring to spokes 1
/// and 2. With truck and hand black, red facts following finds the optimal real plan (7);
/// the repair of the same relaxed plan over-estimates.
#[test]
fn red_facts_following_on_star_shape_logistics() {
    let step = |pre: Vec<Fact>, eff: Vec<Fact>| op(pre, eff.into_iter().map(Effect::new).collect());
    let (truck, hand) = (0, 1);
    let mut operators = Vec::new();
    for spoke in 1..3 {
        operators.push(step(
            vec![Fact::new(truck, 0)],
            vec![Fact::new(truck, spoke)],
        ));
        operators.push(step(
            vec![Fact::new(truck, spoke)],
            vec![Fact::new(truck, 0)],
        ));
    }

    for package in 2..4 {
        for at in 0..3 {
            let (loc, pos) = (Fact::new(truck, at), Fact::new(package, at));
            let held = Fact::new(package, 3);
            operators.push(step(
                vec![loc, pos, Fact::new(hand, 1)],
                vec![held, Fact::new(hand, 0)],
            ));
            operators.push(step(
                vec![loc, held, Fact::new(hand, 0)],
                vec![pos, Fact::new(hand, 1)],
            ));
        }
    }

    let task = Task {
        variables: vec![
            Variable::new("truck", 3),
            Variable::new("hand", 2),
            Variable::new("A", 4),
            Variable::new("B", 4),
        ],
        operators,
        initial: State(vec![0, 1, 0, 0]),
        goal: vec![Fact::new(2, 1), Fact::new(3, 2)],
        axioms: vec![],
    };
    let state = task.initial_state();
    let painting = Painting::dag(&task, vec![truck, hand]).unwrap();
    let follow = RedBlackOptions {
        follow: true,
        ..RedBlackOptions::default()
    };
    let plan = RedBlack::with_painting(&task, &painting)
        .with_options(follow)
        .plan(&state)
        .unwrap();
    let repaired = RedBlack::with_painting(&task, &painting)
        .value(&state)
        .unwrap();

    assert!(plan.is_repaired());
    assert_eq!(plan.cost(), 7);
    assert!(plan.is_plan_for(&task, &state));
    assert!(
        repaired > 7,
        "the repair follows the relaxed plan's order: {repaired}"
    );
}
