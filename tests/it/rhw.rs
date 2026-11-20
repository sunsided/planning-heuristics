//! Back-chained (RHW) landmarks with disjunctions and reasonable orderings against brute
//! force over the state space, compared with the `h^1` (Zhu–Givan) landmarks.

use std::collections::{HashSet, VecDeque};

use planning_heuristics::{OrderedLandmarks, OrderingKind, Relaxation};
use sas_plus::random::{RandomTaskParams, random_task};
use sas_plus::{Fact, State, Task};

use crate::{fixtures, oracle};

/// Number of seeds per parameter set.
const SEEDS: u64 = 48;

/// Counts over many states, to compare the two extractions and keep the tests honest.
#[derive(Debug, Default)]
struct Tally {
    /// Fact landmarks by brute force, not holding in the state.
    true_facts: usize,
    /// Of those, found by `h^1`.
    h1_facts: usize,
    /// Of those, found by back-chaining.
    rhw_facts: usize,
    /// Found by back-chaining but not by `h^1`.
    rhw_only: usize,
    /// Found by `h^1` but not by back-chaining.
    h1_only: usize,
    /// Disjunctive landmarks found.
    disjunctive: usize,
    /// Greedy-necessary orderings found.
    greedy: usize,
    /// Reasonable orderings found.
    reasonable: usize,
}

/// The states reachable from `start`, `start` included.
fn reachable(task: &Task, start: &State) -> Vec<State> {
    let mut seen = HashSet::from([start.clone()]);
    let mut queue = VecDeque::from([start.clone()]);
    let mut all = Vec::new();
    while let Some(state) = queue.pop_front() {
        for op in task.applicable_operators(&state) {
            let next = task.apply(op, &state);
            if seen.insert(next.clone()) {
                queue.push_back(next);
            }
        }

        all.push(state);
    }

    all
}

/// Whether a state satisfying `target` is reachable from `start` through states that satisfy
/// `keep` (`start` included).
fn reachable_keeping(
    task: &Task,
    start: &State,
    keep: impl Fn(&State) -> bool,
    target: impl Fn(&State) -> bool,
) -> bool {
    let mut seen = HashSet::from([start.clone()]);
    let mut queue = VecDeque::from([start.clone()]);
    while let Some(state) = queue.pop_front() {
        if !keep(&state) {
            continue;
        }

        if target(&state) {
            return true;
        }

        for op in task.applicable_operators(&state) {
            let next = task.apply(op, &state);
            if seen.insert(next.clone()) {
                queue.push_back(next);
            }
        }
    }

    false
}

/// Whether one of `facts` holds in `state`.
fn any_holds(facts: &[Fact], state: &State) -> bool {
    facts.iter().any(|f| state.0[f.var] == f.value)
}

/// Checks every landmark and ordering of `graph` from `state` by brute force.
fn check_graph(
    task: &Task,
    state: &State,
    facts: &dyn Fn(&[usize]) -> Vec<Fact>,
    graph: &OrderedLandmarks,
    ctx: &dyn Fn() -> String,
) {
    for l in graph.landmarks() {
        let l = facts(l);
        let goal = reachable_keeping(task, state, |s| !any_holds(&l, s), |s| task.is_goal(s));
        assert!(!goal, "{l:?} is no landmark, {}", ctx());
    }

    for o in graph.orderings() {
        let before = facts(&graph.landmarks()[o.before]);
        let after = facts(&graph.landmarks()[o.after]);
        match o.kind {
            OrderingKind::Natural => {
                let early = reachable_keeping(
                    task,
                    state,
                    |s| !any_holds(&before, s),
                    |s| any_holds(&after, s),
                );
                assert!(!early, "natural {before:?} < {after:?} broken, {}", ctx());
            }
            OrderingKind::GreedyNecessary => check_greedy(task, state, &before, &after, ctx),
            OrderingKind::Reasonable => check_reasonable(task, state, &before, &after, ctx),
        }
    }
}

/// Greedy-necessary: whenever `after` first holds, `before` held in the state just before.
fn check_greedy(
    task: &Task,
    state: &State,
    before: &[Fact],
    after: &[Fact],
    ctx: &dyn Fn() -> String,
) {
    assert!(
        !any_holds(after, state),
        "ordered into a landmark that holds, {}",
        ctx()
    );
    let mut seen = HashSet::from([state.clone()]);
    let mut queue = VecDeque::from([state.clone()]);
    while let Some(s) = queue.pop_front() {
        for op in task.applicable_operators(&s) {
            let next = task.apply(op, &s);
            if any_holds(after, &next) {
                assert!(
                    any_holds(before, &s),
                    "greedy {before:?} < {after:?} broken in {:?}, {}",
                    s.0,
                    ctx()
                );
            } else if seen.insert(next.clone()) {
                queue.push_back(next);
            }
        }
    }
}

/// Reasonable `q < p`: from every state where `p` holds and `q` does not, `q` cannot be
/// reached while `p` stays true.
fn check_reasonable(task: &Task, state: &State, q: &[Fact], p: &[Fact], ctx: &dyn Fn() -> String) {
    for s in reachable(task, state) {
        if any_holds(p, &s) && !any_holds(q, &s) {
            let reached = reachable_keeping(task, &s, |t| any_holds(p, t), |t| any_holds(q, t));
            assert!(
                !reached,
                "reasonable {q:?} < {p:?}: reached q keeping p from {:?}, {}",
                s.0,
                ctx()
            );
        }
    }
}

/// Checks both extractions on every solvable reachable state of `task` and tallies them.
fn check_task(name: &str, task: &Task, tally: &mut Tally) {
    let relaxation = Relaxation::new(task);
    let facts = |l: &[usize]| l.iter().map(|&f| relaxation.fact(f)).collect::<Vec<_>>();
    for (state, h_star) in oracle::perfect(task) {
        let ctx = || format!("{name}, state {:?}", state.0);
        let Some(_) = h_star else {
            continue;
        };

        let mut rhw = relaxation
            .rhw_landmarks(&state)
            .expect("solvable, so no dead end");
        rhw.add_reasonable_orderings(&relaxation, &state);
        let mut h1 = OrderedLandmarks::from(&relaxation.landmarks(&state).expect("solvable"));
        h1.add_reasonable_orderings(&relaxation, &state);
        check_graph(task, &state, &facts, &rhw, &ctx);
        check_graph(task, &state, &facts, &h1, &ctx);
        tally_state(task, &relaxation, &state, &rhw, &h1, tally);
    }
}

/// Adds the counts of one state to `tally`.
fn tally_state(
    task: &Task,
    relaxation: &Relaxation,
    state: &State,
    rhw: &OrderedLandmarks,
    h1: &OrderedLandmarks,
    tally: &mut Tally,
) {
    for id in 0..relaxation.num_facts() {
        let fact = relaxation.fact(id);
        if any_holds(&[fact], state) || task.variables[fact.var].is_derived() {
            continue;
        }

        if reachable_keeping(task, state, |s| !any_holds(&[fact], s), |s| task.is_goal(s)) {
            continue;
        }

        let (in_h1, in_rhw) = (h1.index_of(&[id]).is_some(), rhw.index_of(&[id]).is_some());
        tally.true_facts += 1;
        tally.h1_facts += usize::from(in_h1);
        tally.rhw_facts += usize::from(in_rhw);
        tally.rhw_only += usize::from(in_rhw && !in_h1);
        tally.h1_only += usize::from(in_h1 && !in_rhw);
    }

    tally.disjunctive += rhw.landmarks().iter().filter(|l| l.len() > 1).count();
    for graph in [rhw, h1] {
        tally.greedy += graph
            .orderings()
            .iter()
            .filter(|o| o.kind == OrderingKind::GreedyNecessary)
            .count();
        tally.reasonable += graph
            .orderings()
            .iter()
            .filter(|o| o.kind == OrderingKind::Reasonable)
            .count();
    }
}

/// Checks [`SEEDS`] random tasks of each kind and the fixtures; returns the tally.
fn check_all() -> Tally {
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
    let mut tally = Tally::default();
    for (label, params) in kinds {
        for seed in 0..SEEDS {
            check_task(
                &format!("{label}, seed {seed}"),
                &random_task(seed, params),
                &mut tally,
            );
        }
    }

    for (name, _) in fixtures::FIXTURES {
        check_task(name, &fixtures::load(name), &mut tally);
    }

    tally
}

/// Every landmark, disjunction and ordering of both extractions is sound, and the tests are
/// not vacuous.
#[test]
fn rhw_and_h1_landmarks_by_brute_force() {
    let tally = check_all();
    eprintln!("{tally:?}");

    assert!(
        tally.disjunctive > 0 && tally.greedy > 0 && tally.reasonable > 0,
        "vacuous: {tally:?}"
    );
    assert!(
        tally.rhw_facts > 0 && tally.h1_facts > 0,
        "vacuous: {tally:?}"
    );
}

/// A hand-checked reasonable ordering: making `b` resets `a`, both are goals, so `b` comes
/// first (`b <r a`), and `a <r b` is not added.
#[test]
fn reasonable_ordering_hand_checked() {
    use sas_plus::{Effect, Operator, Variable};

    let task = Task {
        variables: vec![Variable::new("a", 2), Variable::new("b", 2)],
        operators: vec![
            Operator {
                name: "make-a".into(),
                pre: vec![],
                eff: vec![Effect::new(Fact::new(0, 1))],
                cost: 1,
            },
            Operator {
                name: "make-b".into(),
                pre: vec![],
                eff: vec![Effect::new(Fact::new(1, 1)), Effect::new(Fact::new(0, 0))],
                cost: 1,
            },
        ],
        initial: State(vec![0, 0]),
        goal: vec![Fact::new(0, 1), Fact::new(1, 1)],
        axioms: vec![],
    };
    let relaxation = Relaxation::new(&task);
    let state = task.initial_state();
    let mut graph = relaxation.rhw_landmarks(&state).unwrap();
    graph.add_reasonable_orderings(&relaxation, &state);

    let a = graph
        .index_of(&[relaxation.fact_id(Fact::new(0, 1))])
        .unwrap();
    let b = graph
        .index_of(&[relaxation.fact_id(Fact::new(1, 1))])
        .unwrap();
    let reasonable: Vec<(usize, usize)> = graph
        .orderings()
        .iter()
        .filter(|o| o.kind == OrderingKind::Reasonable)
        .map(|o| (o.before, o.after))
        .collect();
    assert_eq!(reasonable, [(b, a)]);
}
