//! The relevance score `h_Ξ` against a brute-force enumeration of the sampling agent's
//! outcomes, and the properties the paper states or implies.

use std::collections::{BTreeMap, HashMap};

use planning_heuristics::{Heuristic, RelevanceParams, RelevanceScore};
use sas_plus::random::{RandomTaskParams, random_task};
use sas_plus::{Effect, Fact, Operator, State, Task, Variable};

use crate::oracle;

/// Parameters that explore small trees completely.
const FULL: RelevanceParams = RelevanceParams {
    min_nodes: usize::MAX,
    rho: 0.0,
    max_nodes: 20_000,
    seed: 0,
};

/// The distribution of the sets of fact labels the agent samples, as bit masks over the
/// task's facts, computed by enumerating its choices on the tree truncated at `state`
/// (independently of the crate: over the task's operators, not the relaxation's rules).
struct Agent<'t> {
    /// The task, without axioms or conditional effects.
    task: &'t Task,
    /// The bit of each fact.
    bit: BTreeMap<Fact, u32>,
    /// The state the tree is truncated at.
    state: &'t State,
}

/// A distribution over label sets.
type Dist = HashMap<u64, f64>;

impl Agent<'_> {
    /// Whether `fact` is true in the state (truncated away).
    fn holds(&self, fact: Fact) -> bool {
        self.state.0[fact.var] == fact.value
    }

    /// The outcomes below fact node `fact` with the facts `path` above it.
    fn fact(&self, fact: Fact, path: &[Fact]) -> Dist {
        let mut path = path.to_vec();
        path.push(fact);
        let achievers: Vec<&Operator> = self
            .task
            .operators
            .iter()
            .filter(|op| op.eff.iter().any(|e| e.fact == fact))
            .filter(|op| op.pre.iter().all(|p| !path.contains(p)))
            .collect();
        let me = 1_u64 << self.bit[&fact];
        if achievers.is_empty() {
            return HashMap::from([(me, 1.0)]);
        }

        let share = 1.0 / f64::from(u32::try_from(achievers.len()).expect("few achievers"));
        let mut dist = Dist::new();
        for op in achievers {
            for (mask, p) in self.all_of(&op.pre, &path) {
                *dist.entry(mask | me).or_default() += p * share;
            }
        }

        dist
    }

    /// The outcomes of taking every fact of `facts` not true in the state.
    fn all_of(&self, facts: &[Fact], path: &[Fact]) -> Dist {
        let mut dist = HashMap::from([(0, 1.0)]);
        for &f in facts.iter().filter(|&&f| !self.holds(f)) {
            let below = self.fact(f, path);
            let mut next = Dist::new();
            for (&m, &p) in &dist {
                for (&n, &q) in &below {
                    *next.entry(m | n).or_default() += p * q;
                }
            }

            dist = next;
        }

        dist
    }

    /// The score of every fact: the probability that its bit is sampled.
    fn scores(&self) -> BTreeMap<Fact, f64> {
        let dist = self.all_of(&self.task.goal, &[]);
        self.bit
            .iter()
            .map(|(&f, &b)| {
                let p = dist
                    .iter()
                    .filter(|&(&m, _)| m & (1 << b) != 0)
                    .map(|(_, &p)| p)
                    .sum();
                (f, p)
            })
            .filter(|&(_, p)| p > 0.0)
            .collect()
    }
}

/// The scores of `h` as a map.
fn scores_of(h: &mut RelevanceScore, state: &State) -> BTreeMap<Fact, f64> {
    h.scores(state).into_iter().collect()
}

/// Asserts two score maps agree up to rounding.
fn assert_close(a: &BTreeMap<Fact, f64>, b: &BTreeMap<Fact, f64>, ctx: &str) {
    for f in a.keys().chain(b.keys()) {
        let (x, y) = (
            a.get(f).copied().unwrap_or(0.0),
            b.get(f).copied().unwrap_or(0.0),
        );
        assert!((x - y).abs() < 1e-9, "{f:?}: {x} vs {y}, {ctx}");
    }
}

/// On fully explored trees of small random tasks, the scores of every reachable state equal
/// the enumerated sampling probabilities (Eqs. 2 and 3 against the definition), and partial
/// exploration never exceeds them (Section 4.2: a lower bound).
#[test]
fn scores_match_the_enumerated_agent() {
    let mut checked = 0;
    for seed in 0..32 {
        let task = random_task(
            seed,
            RandomTaskParams {
                num_vars: 3,
                num_ops: 7,
                ..RandomTaskParams::default()
            },
        );
        let mut full = RelevanceScore::with_params(&task, &FULL);
        if !full.is_complete() {
            continue;
        }

        let mut partial = RelevanceScore::with_params(
            &task,
            &RelevanceParams {
                min_nodes: 8,
                rho: 0.5,
                ..FULL
            },
        );
        let mut bit = BTreeMap::new();
        for (var, v) in task.variables.iter().enumerate() {
            for value in 0..v.domain_size {
                let next = u32::try_from(bit.len()).expect("few facts");
                bit.insert(Fact::new(var, value), next);
            }
        }

        for state in oracle::perfect(&task).keys() {
            let ctx = format!("seed {seed}, state {:?}", state.0);
            let agent = Agent {
                task: &task,
                bit: bit.clone(),
                state,
            };
            let exact = agent.scores();
            let computed = scores_of(&mut full, state);
            assert_close(&computed, &exact, &ctx);
            for (f, s) in scores_of(&mut partial, state) {
                assert!(
                    s <= exact.get(&f).copied().unwrap_or(0.0) + 1e-9,
                    "{f:?}, {ctx}"
                );
            }

            checked += 1;
        }
    }

    assert!(checked > 50, "vacuous: {checked}");
}

/// An operator with precondition `pre` and the single effect `eff`, cost 1.
fn op(pre: Vec<Fact>, eff: Vec<Fact>) -> Operator {
    Operator {
        name: "op".into(),
        pre,
        eff: eff.into_iter().map(Effect::new).collect(),
        cost: 1,
    }
}

/// A task over `n` binary variables, all 0 initially.
fn binary(n: usize, operators: Vec<Operator>, goal: Vec<Fact>) -> Task {
    Task {
        variables: (0..n).map(|i| Variable::new(format!("v{i}"), 2)).collect(),
        operators,
        initial: State(vec![0; n]),
        goal,
        axioms: vec![],
    }
}

/// Not admissible: one operator achieving three goal facts scores 3 against `h* = 1`. Goal
/// states score 0, and relaxed dead ends are reported.
#[test]
fn inadmissible_zero_at_goals_and_dead_ends() {
    let on = |v| Fact::new(v, 1);
    let task = binary(
        3,
        vec![op(vec![], vec![on(0), on(1), on(2)])],
        vec![on(0), on(1), on(2)],
    );
    let mut h = RelevanceScore::with_params(&task, &FULL);
    let state = task.initial_state();
    assert_eq!(h.value(&state), Some(3.0));
    assert_eq!(oracle::perfect(&task)[&state], Some(1));
    assert_eq!(h.value(&State(vec![1, 1, 1])), Some(0.0));

    let stuck = binary(2, vec![], vec![on(0)]);
    assert_eq!(
        RelevanceScore::with_params(&stuck, &FULL).estimate(&stuck.initial_state()),
        None
    );
}

/// Not every landmark scores 1, and the score is not the fraction of plans using a fact: the
/// goal has two achievers, one needing `p` (achievable), one needing `q` (no achiever). Every
/// plan uses `p`, a landmark, but the agent picks the dead-end achiever half the time, so
/// `Ξ(p) = 1/2`. (The paper's discussion says landmarks score 1 on a full tree; that needs
/// every sampled branch to complete.)
#[test]
fn landmarks_can_score_below_one() {
    let on = |v| Fact::new(v, 1);
    let (g, p, q) = (on(0), on(1), on(2));
    let task = binary(
        3,
        vec![
            op(vec![p], vec![g]),
            op(vec![q], vec![g]),
            op(vec![], vec![p]),
        ],
        vec![g],
    );
    let mut h = RelevanceScore::with_params(&task, &FULL);
    let scores = scores_of(&mut h, &task.initial_state());

    assert_eq!(scores[&g], 1.0);
    assert_eq!(scores[&p], 0.5);
    assert_eq!(scores[&q], 0.5);
}
