//! `h²` against a regression-based oracle and `h*`, the set-additive heuristic against
//! `h_max`, `h_add` and a relaxed plan validator, and additive `h_max` against `h*`.

use std::collections::BTreeMap;

use planning_heuristics::{AdditiveHMax, H2, H2Error, HAdd, HMax, HSetAdd, Relaxation};
use sas_plus::random::{RandomTaskParams, random_task};
use sas_plus::{Fact, State, Task};

use crate::{fixtures, oracle};

/// Number of seeds per parameter set.
const SEEDS: u64 = 48;

/// The random task kinds: plain, conditional, axioms, mixed.
fn kinds() -> [(&'static str, RandomTaskParams); 4] {
    [
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
    ]
}

/// All test tasks: [`SEEDS`] of each kind plus the fixtures.
fn tasks() -> Vec<(String, Task)> {
    let mut all: Vec<(String, Task)> = kinds()
        .into_iter()
        .flat_map(|(label, params)| {
            (0..SEEDS).map(move |seed| (format!("{label}, seed {seed}"), random_task(seed, params)))
        })
        .collect();
    all.extend(
        fixtures::FIXTURES
            .iter()
            .map(|(name, _)| ((*name).to_owned(), fixtures::load(name))),
    );

    all
}

/// The final effect facts of operator `op`, one per variable (the last effect wins).
fn final_effects(task: &Task, op: usize) -> Vec<Fact> {
    let mut eff: Vec<Fact> = Vec::new();
    for e in &task.operators[op].eff {
        eff.retain(|f| f.var != e.fact.var);
        eff.push(e.fact);
    }

    eff
}

/// `h²` by its regression definition (Haslum & Geffner 2000), for tasks without conditional
/// effects and axioms: a set of at most two facts costs 0 if it holds in `state`, otherwise
/// the cheapest `cost(o) + h²(regression)` over operators that add one of its facts and
/// delete none; larger sets cost their most expensive pair. Swept to the fixpoint.
fn h2_oracle(task: &Task, state: &State) -> Option<u64> {
    let facts: Vec<Fact> = (0..task.num_variables())
        .flat_map(|v| (0..task.variables[v].domain_size).map(move |x| Fact::new(v, x)))
        .collect();
    let holds = |f: &Fact| state.0[f.var] == f.value;
    let sets = singletons_and_pairs(&facts);

    let mut h: BTreeMap<Vec<Fact>, u64> = sets
        .iter()
        .filter(|s| s.iter().all(holds))
        .map(|s| (s.clone(), 0))
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for s in &sets {
            for op in 0..task.operators.len() {
                let eff = final_effects(task, op);
                let adds = s.iter().any(|f| eff.contains(f));
                let deletes = s
                    .iter()
                    .any(|f| eff.iter().any(|e| e.var == f.var && e != f));
                if !adds || deletes {
                    continue;
                }

                let mut regression: Vec<Fact> =
                    s.iter().copied().filter(|f| !eff.contains(f)).collect();
                regression.extend(&task.operators[op].pre);
                let Some(base) = set_cost(&h, &regression) else {
                    continue;
                };

                let cost = base + u64::from(task.operators[op].cost);
                if h.get(s).is_none_or(|&old| cost < old) {
                    h.insert(s.clone(), cost);
                    changed = true;
                }
            }
        }
    }

    set_cost(&h, &task.goal)
}

/// Every single fact of `facts`, and every pair of facts on different variables.
fn singletons_and_pairs(facts: &[Fact]) -> Vec<Vec<Fact>> {
    let mut sets: Vec<Vec<Fact>> = facts.iter().map(|&f| vec![f]).collect();
    for (i, &p) in facts.iter().enumerate() {
        sets.extend(
            facts[i + 1..]
                .iter()
                .filter(|q| q.var != p.var)
                .map(|&q| vec![p, q]),
        );
    }

    sets
}

/// The `h²` cost of `set` under the table `h`: the most expensive of its facts and pairs;
/// `None` if two of its facts contradict each other or a fact or pair has no cost yet.
fn set_cost(h: &BTreeMap<Vec<Fact>, u64>, set: &[Fact]) -> Option<u64> {
    let mut set = set.to_vec();
    set.sort_unstable();
    set.dedup();
    let mut worst = 0;
    for (i, &p) in set.iter().enumerate() {
        for &q in &set[i..] {
            if p.var == q.var && p != q {
                return None;
            }

            let key = if p == q { vec![p] } else { vec![p, q] };
            worst = worst.max(*h.get(&key)?);
        }
    }

    Some(worst)
}

/// `h²` equals the oracle on tasks it shares with it, and on every task `h_max <= h² <= h*`
/// with only real dead ends; it beats `h_max` somewhere.
#[test]
fn h2_by_oracle_and_h_star() {
    let mut above = 0;
    for (name, task) in tasks() {
        let mut h2 = match H2::new(&task) {
            Ok(h2) => h2,
            Err(e) => {
                assert_eq!(e, H2Error::Axioms, "{name}");
                assert!(!task.axioms.is_empty(), "{name}");
                continue;
            }
        };
        let plain = task
            .operators
            .iter()
            .all(|o| o.eff.iter().all(|e| e.conditions.is_empty()));
        let mut h_max = HMax::new(&task);
        for (state, h_star) in oracle::perfect(&task) {
            let ctx = || format!("{name}, state {:?}", state.0);
            let value = h2.value(&state);
            if plain {
                assert_eq!(value, h2_oracle(&task, &state), "oracle, {}", ctx());
            }

            match (value, h_star) {
                (None, Some(_)) => panic!("false dead end, {}", ctx()),
                (Some(v), Some(t)) => assert!(v <= t, "h2 {v} > h* {t}, {}", ctx()),
                _ => {}
            }

            if let (Some(v), Some(m)) = (value, h_max.value(&state)) {
                assert!(m <= v, "h_max {m} > h2 {v}, {}", ctx());
                above += usize::from(v > m);
            }
        }
    }

    assert!(above > 0, "h2 never beats h_max");
}

/// `h_sa <= h_add`, and `h_max <= h_sa` without conditional effects, with the same dead ends;
/// the operator set reaches the goal under the relaxation and costs `h_sa`; `h_sa` is below
/// `h_add` somewhere.
#[test]
fn set_additive_between_h_max_and_h_add() {
    let mut below_add = 0;
    for (name, task) in tasks() {
        let relaxation = Relaxation::new(&task);
        let mut h_sa = HSetAdd::with_relaxation(&relaxation);
        let conditional = task
            .operators
            .iter()
            .any(|o| o.eff.iter().any(|e| !e.conditions.is_empty()));
        let mut h_max = HMax::with_relaxation(&relaxation);
        let mut h_add = HAdd::with_relaxation(&relaxation);
        for state in oracle::perfect(&task).keys() {
            let ctx = || format!("{name}, state {:?}", state.0);
            let (sa, max, add) = (h_sa.value(state), h_max.value(state), h_add.value(state));
            assert_eq!(sa.is_none(), max.is_none(), "dead ends, {}", ctx());
            let (Some(sa), Some(max), Some(add)) = (sa, max, add) else {
                continue;
            };

            assert!(sa <= add, "h_sa {sa} > h_add {add}, {}", ctx());
            // With conditional effects a relaxed plan may need one operator twice, which a
            // set counts once; only without them is the set's cost at least h_max.
            if !conditional {
                assert!(max <= sa, "h_max {max} > h_sa {sa}, {}", ctx());
            }
            let plan = h_sa.plan(state).unwrap();
            assert_eq!(oracle::plan_cost(&task, &plan), sa, "{}", ctx());
            assert!(
                oracle::relaxed_closure_reaches_goal(&task, state, &plan),
                "{}",
                ctx()
            );
            below_add += usize::from(sa < add);
        }
    }

    assert!(below_add > 0, "h_sa never below h_add");
}

/// Additive `h_max` is admissible with `h_max`'s dead ends, equals `h_max` with one part, and
/// beats it somewhere with the default partition.
#[test]
fn additive_h_max_is_admissible() {
    let mut above = 0;
    for (name, task) in tasks() {
        let relaxation = Relaxation::new(&task);
        let mut h_max = HMax::with_relaxation(&relaxation);
        let mut additive = AdditiveHMax::new(&task);
        let mut one_part = AdditiveHMax::with_partition(&task, &vec![0; task.operators.len()]);
        for (state, h_star) in oracle::perfect(&task) {
            let ctx = || format!("{name}, state {:?}", state.0);
            let (value, max) = (additive.value(&state), h_max.value(&state));
            assert_eq!(value.is_none(), max.is_none(), "dead ends, {}", ctx());
            assert_eq!(one_part.value(&state), max, "one part, {}", ctx());
            if let (Some(v), Some(t)) = (value, h_star) {
                assert!(v <= t, "additive h_max {v} > h* {t}, {}", ctx());
            }

            above += usize::from(value > max);
        }
    }

    assert!(above > 0, "additive h_max never beats h_max");
}
