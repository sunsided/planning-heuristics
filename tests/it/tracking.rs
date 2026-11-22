//! Path-dependent landmark tracking in reference searches: A* with the admissible tracker
//! finds optimal plans and never overestimates, greedy best-first search with LAMA's count
//! finds valid plans.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use planning_heuristics::{
    HMax, LandmarkTracker, PathHeuristic, Relaxation, Stateless, TrackingMode,
};
use sas_plus::random::{RandomTaskParams, random_task};
use sas_plus::{Effect, Fact, OpId, Operator, State, Task, Variable};

use crate::{fixtures, oracle};

/// Number of seeds per parameter set.
const SEEDS: u64 = 48;

/// A* from the initial state with reopening and path information merged on every new path
/// to a state; calls `seen` with every generated state and its estimate. Returns the cost of
/// the plan found.
fn astar<H: PathHeuristic<State, OpId>>(
    task: &Task,
    h: &mut H,
    mut seen: impl FnMut(&State, Option<u32>),
) -> Option<u64> {
    let start = task.initial_state();
    let mut g: HashMap<State, u64> = HashMap::from([(start.clone(), 0)]);
    let mut info: HashMap<State, H::Info> = HashMap::from([(start.clone(), h.root(&start))]);
    let mut open = BinaryHeap::new();
    let h0 = h.estimate(&start, &info[&start]);
    seen(&start, h0);
    open.push(Reverse((u64::from(h0?), 0, start.0)));

    while let Some(Reverse((_, cost, values))) = open.pop() {
        let state = State(values);
        if cost > g[&state] {
            continue;
        }

        if task.is_goal(&state) {
            return Some(cost);
        }

        for op in task.applicable_operators(&state) {
            let child = task.apply(op, &state);
            let child_info = h.on_expand(&state, &info[&state], op, &child);
            match info.get_mut(&child) {
                Some(kept) => h.merge(kept, &child_info),
                None => {
                    info.insert(child.clone(), child_info);
                }
            }

            let next = cost + u64::from(task.operators[op].cost);
            if g.get(&child).is_some_and(|&old| old <= next) {
                continue;
            }

            g.insert(child.clone(), next);
            let estimate = h.estimate(&child, &info[&child]);
            seen(&child, estimate);
            if let Some(e) = estimate {
                open.push(Reverse((next + u64::from(e), next, child.0)));
            }
        }
    }

    None
}

/// Greedy best-first search from the initial state; returns the plan found.
fn gbfs<H: PathHeuristic<State, OpId>>(task: &Task, h: &mut H) -> Option<Vec<OpId>> {
    let start = task.initial_state();
    let mut parent: HashMap<State, Option<(State, OpId)>> = HashMap::from([(start.clone(), None)]);
    let mut info: HashMap<State, H::Info> = HashMap::from([(start.clone(), h.root(&start))]);
    let mut open = BinaryHeap::new();
    let mut tie = 0_u64;
    open.push(Reverse((
        h.estimate(&start, &info[&start])?,
        tie,
        start.0.clone(),
    )));

    while let Some(Reverse((_, _, values))) = open.pop() {
        let state = State(values);
        if task.is_goal(&state) {
            return Some(plan_to(&parent, state));
        }

        for op in task.applicable_operators(&state) {
            let child = task.apply(op, &state);
            let child_info = h.on_expand(&state, &info[&state], op, &child);
            if let Some(kept) = info.get_mut(&child) {
                h.merge(kept, &child_info);
                continue;
            }

            parent.insert(child.clone(), Some((state.clone(), op)));
            info.insert(child.clone(), child_info);
            if let Some(e) = h.estimate(&child, &info[&child]) {
                tie += 1;
                open.push(Reverse((e, tie, child.0)));
            }
        }
    }

    None
}

/// The operators from the initial state to `state` along `parent`.
fn plan_to(parent: &HashMap<State, Option<(State, OpId)>>, mut state: State) -> Vec<OpId> {
    let mut plan = Vec::new();
    while let Some((prev, op)) = parent[&state].clone() {
        plan.push(op);
        state = prev;
    }

    plan.reverse();
    plan
}

/// Checks both tracking modes on `task`, and with feature `lp` the optimal tracker, adding to
/// `optimal_above` how often it beat the uniform one. Returns whether the admissible tracker
/// was informative somewhere (an estimate above 0).
fn check_task(name: &str, task: &Task, optimal_above: &mut usize) -> bool {
    let perfect = oracle::perfect(task);
    let h_star = perfect[&task.initial_state()];
    let Some(mut admissible) = LandmarkTracker::for_task(task, TrackingMode::UniformCost) else {
        assert_eq!(h_star, None, "{name}: landmarks claim a dead start");
        return false;
    };

    let mut informative = false;
    let found = astar(task, &mut admissible, |state, estimate| {
        let truth = perfect[state];
        match (estimate, truth) {
            (Some(e), Some(t)) => {
                assert!(u64::from(e) <= t, "{name}: h {e} > h* {t} at {:?}", state.0);
                informative |= e > 0;
            }
            (None, Some(t)) => panic!("{name}: false dead end at {:?} (h* {t})", state.0),
            _ => {}
        }
    });
    assert_eq!(found, h_star, "{name}: A* with the admissible tracker");

    #[cfg(feature = "lp")]
    {
        *optimal_above += check_optimal(name, task, &perfect);
    }
    #[cfg(not(feature = "lp"))]
    let _ = optimal_above;

    let mut lama = LandmarkTracker::for_task(task, TrackingMode::Count).expect("same landmarks");
    let plan = gbfs(task, &mut lama);
    assert_eq!(
        plan.is_some(),
        h_star.is_some(),
        "{name}: GBFS with LAMA's count"
    );
    if let Some(plan) = plan {
        assert!(task.is_goal(&replay(task, &plan)), "{name}: invalid plan");
    }

    informative
}

/// The optimal tracker beside the uniform one: every estimate is at least the uniform one on
/// the same path information.
#[cfg(feature = "lp")]
struct Both {
    /// The tracker under test, which drives the search.
    optimal: planning_heuristics::OptimalLandmarkTracker<'static>,
    /// The uniform tracker, fed the same path information.
    uniform: LandmarkTracker<'static>,
    /// How many estimates the optimal tracker was strictly above the uniform one.
    above: usize,
}

#[cfg(feature = "lp")]
impl PathHeuristic<State, OpId> for Both {
    type Info = planning_heuristics::AcceptedLandmarks;

    fn root(&mut self, state: &State) -> Self::Info {
        let info = self.optimal.root(state);
        assert_eq!(info, self.uniform.root(state), "same acceptance");
        info
    }

    fn on_expand(
        &mut self,
        parent: &State,
        info: &Self::Info,
        op: OpId,
        child: &State,
    ) -> Self::Info {
        self.optimal.on_expand(parent, info, op, child)
    }

    fn merge(&mut self, kept: &mut Self::Info, other: &Self::Info) {
        self.optimal.merge(kept, other);
    }

    fn estimate(&mut self, state: &State, info: &Self::Info) -> Option<u32> {
        let optimal = self.optimal.estimate(state, info);
        let uniform = self.uniform.estimate(state, info);
        assert_eq!(
            optimal.is_none(),
            uniform.is_none(),
            "same dead ends at {:?}",
            state.0
        );
        if let (Some(o), Some(u)) = (optimal, uniform) {
            assert!(o >= u, "optimal {o} < uniform {u} at {:?}", state.0);
            self.above += usize::from(o > u);
        }
        optimal
    }
}

/// A* with the optimal tracker finds an optimal plan, never overestimates, and is never below
/// the uniform tracker. Returns how many estimates beat the uniform tracker.
#[cfg(feature = "lp")]
fn check_optimal(name: &str, task: &Task, perfect: &HashMap<State, Option<u64>>) -> usize {
    let h_star = perfect[&task.initial_state()];
    let mut both = Both {
        optimal: planning_heuristics::OptimalLandmarkTracker::for_task(task)
            .expect("same landmarks"),
        uniform: LandmarkTracker::for_task(task, TrackingMode::UniformCost)
            .expect("same landmarks"),
        above: 0,
    };
    let found = astar(task, &mut both, |state, estimate| {
        if let (Some(e), Some(t)) = (estimate, perfect[state]) {
            assert!(
                u64::from(e) <= t,
                "{name}: optimal {e} > h* {t} at {:?}",
                state.0
            );
        }
    });
    assert_eq!(found, h_star, "{name}: A* with the optimal tracker");
    both.above
}

/// The state after applying `plan` from the initial state.
///
/// # Panics
///
/// Panics if an operator is not applicable.
fn replay(task: &Task, plan: &[OpId]) -> State {
    plan.iter().fold(task.initial_state(), |state, &op| {
        assert!(
            task.is_applicable(op, &state),
            "operator {op} not applicable"
        );
        task.apply(op, &state)
    })
}

/// Both modes on random tasks of every kind and on the fixtures.
#[test]
fn tracking_in_reference_searches() {
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
    let mut informative = 0;
    let mut optimal_above = 0;
    for (label, params) in kinds {
        for seed in 0..SEEDS {
            let name = format!("{label}, seed {seed}");
            informative += usize::from(check_task(
                &name,
                &random_task(seed, params),
                &mut optimal_above,
            ));
        }
    }

    for (name, _) in fixtures::FIXTURES {
        informative += usize::from(check_task(name, &fixtures::load(name), &mut optimal_above));
    }

    assert!(informative > 0, "the admissible tracker is never above 0");
    #[cfg(feature = "lp")]
    assert!(
        optimal_above > 0,
        "the optimal tracker never beats the uniform one"
    );
}

/// The reference A* itself is optimal with a stateless admissible heuristic.
#[test]
fn reference_astar_with_stateless_h_max() {
    for seed in 0..SEEDS {
        let task = random_task(seed, RandomTaskParams::default());
        let h_star = oracle::perfect(&task)[&task.initial_state()];
        let mut h = Stateless(HMax::new(&task));
        assert_eq!(astar(&task, &mut h, |_, _| {}), h_star, "seed {seed}");
    }
}

/// Hand-checked acceptance: a goal achieved and lost again is required again, and merging
/// two paths keeps only what both accepted.
#[test]
fn goal_lost_again_and_merge() {
    let task = Task {
        variables: vec![Variable::new("a", 2), Variable::new("b", 2)],
        operators: vec![
            Operator {
                name: "set-a".into(),
                pre: vec![],
                eff: vec![Effect::new(Fact::new(0, 1))],
                cost: 1,
            },
            Operator {
                name: "set-b-reset-a".into(),
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
    let start = task.initial_state();
    let graph = relaxation.rhw_landmarks(&start).unwrap();
    let mut h = LandmarkTracker::new(&relaxation, &graph, TrackingMode::UniformCost);

    let root = h.root(&start);
    assert_eq!(h.estimate(&start, &root), Some(2));
    let s1 = task.apply(0, &start);
    let i1 = h.on_expand(&start, &root, 0, &s1);
    assert_eq!(h.estimate(&s1, &i1), Some(1));
    let s2 = task.apply(1, &s1);
    let i2 = h.on_expand(&s1, &i1, 1, &s2);
    assert_eq!(s2.0, [0, 1]);
    assert_eq!(i2.len(), 2, "both landmarks were achieved on this path");
    assert_eq!(h.estimate(&s2, &i2), Some(1), "a is a goal, required again");

    // The same state reached directly from the start never accepted `a`.
    let direct = h.on_expand(&start, &root, 1, &s2);
    let mut merged = i2.clone();
    h.merge(&mut merged, &direct);
    assert_eq!(merged, direct);
    assert_eq!(h.estimate(&s2, &merged), Some(1));
}
