//! Seeded LM-cut ([`LmCut::value_seeded`]) against [`LmCut::value`] and against a brute-force
//! cheapest seeded completion.

use planning_heuristics::{HMax, LmCut, Relaxation};
use sas_plus::random::{RandomTaskParams, random_task};
use sas_plus::{Fact, State, Task};

use crate::oracle;

/// Number of random tasks per parameter set.
const TASKS: u64 = 48;

/// The parameter sets checked: plain, conditional effects, axioms, both.
fn params() -> [(&'static str, RandomTaskParams); 4] {
    let base = RandomTaskParams::default();
    [
        ("plain", base),
        (
            "conditional",
            RandomTaskParams {
                max_conditions: 2,
                ..base
            },
        ),
        (
            "axioms",
            RandomTaskParams {
                num_derived: 2,
                ..base
            },
        ),
        (
            "mixed",
            RandomTaskParams {
                num_vars: 5,
                num_ops: 14,
                max_conditions: 2,
                num_derived: 2,
                ..base
            },
        ),
    ]
}

/// With the seeds of a state and no (or the task's own) costs, seeded LM-cut is exactly
/// `LmCut::value` (split by contexts or over a shared relaxation), and seeded `h_max` over
/// the shared relaxation is `h_max`, on every reachable state.
#[test]
fn state_seeds_equal_lm_cut() {
    for (label, p) in params() {
        for seed in 0..TASKS {
            let task = random_task(seed, p);
            let costs = task.costs();
            let mut lm_cut = LmCut::new(&task);
            let mut seeded = LmCut::new(&task);
            let relaxation = Relaxation::new(&task);
            let mut h_max = HMax::with_relaxation(&relaxation);
            // Unsplit: context splitting can see dead ends h_max misses.
            let mut shared = LmCut::with_relaxation(&relaxation);
            let mut shared_lm_cut = LmCut::with_relaxation(&relaxation);
            let mut seeds = Vec::new();
            for (state, &h_star) in &oracle::perfect(&task) {
                let ctx = format!("{label}, seed {seed}, state {:?}", state.0);
                seeded.seed_state(state, &mut seeds);
                let expected = lm_cut.value(state);
                assert_eq!(seeded.value_seeded(&seeds, None), expected, "{ctx}");
                assert_eq!(seeded.value_seeded(&seeds, Some(&costs)), expected, "{ctx}");
                assert!(expected.is_some() || h_star.is_none(), "{ctx}");
                assert_eq!(
                    shared.value_seeded(&seeds, None),
                    shared_lm_cut.value(state),
                    "{ctx}"
                );
                assert_eq!(
                    shared.h_max_seeded(&seeds, None),
                    h_max.value(state),
                    "{ctx}"
                );
            }
        }
    }
}

/// A small deterministic generator for seeds and costs.
struct Rng(u64);

impl Rng {
    /// A value in `0..n`.
    fn below(&mut self, n: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % n
    }
}

/// Random seeds over the state variables' facts (`None` with probability 1/3, else 0..4);
/// derived facts stay unseeded.
fn random_seeds(task: &Task, lm_cut: &LmCut, rng: &mut Rng) -> Vec<Option<u64>> {
    let mut seeds = vec![None; lm_cut.relaxation().num_facts()];
    for (var, v) in task.variables.iter().enumerate() {
        if v.is_derived() {
            continue;
        }

        for value in 0..v.domain_size {
            let fact = lm_cut.relaxation().fact_id(Fact::new(var, value));
            seeds[fact] = (rng.below(3) != 0).then(|| rng.below(4));
        }
    }

    seeds
}

/// The cheapest seeded completion: the minimum over states `s` with every state variable's
/// value seeded of `Σ seed(s[v]) + h*(s)`, by enumerating all such states; `None` if none
/// reaches the goal.
fn cheapest_completion(task: &Task, lm_cut: &LmCut, seeds: &[Option<u64>]) -> Option<u64> {
    let perfect = task.perfect_heuristic();
    let vars: Vec<usize> = (0..task.num_variables())
        .filter(|&v| !task.variables[v].is_derived())
        .collect();
    let mut values = task.initial.0.clone();
    let mut best: Option<u64> = None;
    let mut index = vec![0_u32; vars.len()];
    'states: loop {
        let mut paid = 0;
        for (&var, &value) in vars.iter().zip(&index) {
            values[var] = value;
            match seeds[lm_cut.relaxation().fact_id(Fact::new(var, value))] {
                Some(s) => paid += s,
                None => paid = u64::MAX,
            }

            if paid == u64::MAX {
                break;
            }
        }

        if paid != u64::MAX {
            let mut state = State(values.clone());
            task.evaluate_axioms(&mut state);
            if let Some(h) = perfect(&state) {
                best = Some(best.map_or(paid + h, |b| b.min(paid + h)));
            }
        }

        for (i, &var) in vars.iter().enumerate() {
            index[i] += 1;
            if index[i] < task.variables[var].domain_size {
                continue 'states;
            }

            index[i] = 0;
        }

        return best;
    }
}

/// With random seeds and random per-call costs, seeded LM-cut and seeded `h_max` never
/// exceed the cheapest seeded completion, and report a dead end only when there is none.
#[test]
fn random_seeds_stay_admissible() {
    let mut rng = Rng(7);
    let mut informative = 0;
    for (label, p) in params() {
        for seed in 0..TASKS {
            // The heuristic keeps the task's original costs; only the oracle's copy changes.
            let mut task = random_task(seed, p);
            let mut lm_cut = LmCut::new(&task);
            for round in 0..4 {
                let seeds = random_seeds(&task, &lm_cut, &mut rng);
                let costs: Vec<u64> = (0..task.operators.len()).map(|_| rng.below(5)).collect();
                for (op, &c) in task.operators.iter_mut().zip(&costs) {
                    op.cost = u32::try_from(c).expect("small costs");
                }

                let ctx = format!("{label}, seed {seed}, round {round}, seeds {seeds:?}");
                let best = cheapest_completion(&task, &lm_cut, &seeds);
                let value = lm_cut.value_seeded(&seeds, Some(&costs));
                let h_max = lm_cut.h_max_seeded(&seeds, Some(&costs));
                if let Some(best) = best {
                    let value = value.unwrap_or_else(|| panic!("false dead end, {ctx}"));
                    assert!(value <= best, "LM-cut {value} > {best}, {ctx}");
                    assert!(h_max.is_some_and(|h| h <= best), "h_max {h_max:?}, {ctx}");
                    informative += usize::from(value > 0);
                }
            }
        }
    }

    assert!(informative > 100, "vacuous: {informative}");
}
