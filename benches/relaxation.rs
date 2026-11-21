//! Benchmarks for `h_max`, `h_add`, `h_FF`, `h_RB`, LM-cut, the landmark heuristics and landmark tracking, next to width-search's sweep-based `h_add`.
//!
//! Each benchmark evaluates a fixed set of reachable states (breadth-first from the initial
//! state) of one task. The larger tasks (`bw-large-a`, `logistics-e`, GPL-2.0 / Apache-2.0
//! test inputs) are read from the sibling width-search checkout and skipped if it is absent.

use std::collections::{HashSet, VecDeque};
use std::hint::black_box;

use criterion::Criterion;
use planning_heuristics::{
    GoalIntention, HAdd, HFf, HMax, HSetAdd, Heuristic, IntentionKind, IntentionParams,
    LandmarkCount, LandmarkTracker, LmCut, Painting, PathHeuristic, RedBlack, RedBlackOptions,
    RelevanceParams, RelevanceScore, TrackingMode, UniformLandmarkCost,
};
use sas_plus::fdr::SasFile;
use sas_plus::random::{RandomTaskParams, random_task};
use sas_plus::{State, Task};
use width_search::sas::SasProblem;

/// Number of states evaluated per benchmark iteration.
const STATES: usize = 100;

/// Runs every benchmark with criterion's command-line flags.
fn main() {
    let mut criterion = Criterion::default().configure_from_args();
    for (name, task) in tasks() {
        bench_task(&mut criterion, &name, &task);
    }

    criterion.final_summary();
}

/// The benchmark tasks: own fixtures, a large random task and, if present, the large
/// translated fixtures of the width-search checkout.
fn tasks() -> Vec<(String, Task)> {
    let dir = env!("CARGO_MANIFEST_DIR");
    let own = ["translated-courier", "translated-wiring", "gripper-costs"]
        .map(|name| (name, format!("{dir}/tests/fixtures/{name}/output.sas")));
    let sibling = ["bw-large-a", "logistics-e"].map(|name| {
        let path = format!("{dir}/../width-search/tests/fixtures/translated/{name}/output.sas");
        (name, path)
    });

    let mut tasks: Vec<(String, Task)> = own
        .into_iter()
        .chain(sibling)
        .filter_map(|(name, path)| Some((name.to_owned(), load(&path)?)))
        .collect();
    let params = RandomTaskParams {
        num_vars: 40,
        max_domain: 4,
        num_ops: 400,
        max_cost: 5,
        max_conditions: 1,
        num_derived: 0,
    };
    tasks.push(("random-40x400".to_owned(), random_task(7, params)));

    tasks
}

/// Parses the `output.sas` file at `path`; `None` if it does not exist.
///
/// # Panics
///
/// Panics if the file exists but does not parse.
fn load(path: &str) -> Option<Task> {
    let text = std::fs::read_to_string(path).ok()?;
    let file: SasFile = text.parse().expect("fixture parses");

    Some(file.to_task())
}

/// Up to [`STATES`] states reachable from the initial state, breadth-first.
fn sample_states(task: &Task) -> Vec<State> {
    let mut seen = HashSet::new();
    let mut queue = VecDeque::from([task.initial_state()]);
    let mut states = Vec::new();
    while let Some(state) = queue.pop_front() {
        if states.len() == STATES {
            break;
        }

        if seen.insert(state.clone()) {
            queue.extend(
                task.applicable_operators(&state)
                    .map(|op| task.apply(op, &state)),
            );
            states.push(state);
        }
    }

    states
}

/// Benchmarks all heuristics on the sampled states of `task`.
fn bench_task(criterion: &mut Criterion, name: &str, task: &Task) {
    let states = sample_states(task);
    let mut group = criterion.benchmark_group(name);
    bench_heuristic(&mut group, "h_max", HMax::new(task), &states);
    bench_heuristic(&mut group, "h_add", HAdd::new(task), &states);
    bench_heuristic(&mut group, "h_ff", HFf::new(task), &states);
    bench_heuristic(&mut group, "h_sa", HSetAdd::new(task), &states);
    bench_heuristic(&mut group, "h_rb", RedBlack::new(task), &states);
    let follow = RedBlackOptions {
        follow: true,
        ..RedBlackOptions::default()
    };
    let h_rb_follow = RedBlack::with_painting(task, &Painting::greedy(task)).with_options(follow);
    bench_heuristic(&mut group, "h_rb_follow", h_rb_follow, &states);
    bench_heuristic(&mut group, "lm_cut", LmCut::new(task), &states);
    bench_heuristic(&mut group, "lm_count", LandmarkCount::new(task), &states);
    bench_heuristic(
        &mut group,
        "lm_uniform",
        UniformLandmarkCost::new(task),
        &states,
    );

    for (label, mode) in [
        ("lm_track_count", TrackingMode::Count),
        ("lm_track_uniform", TrackingMode::UniformCost),
    ] {
        if let Some(tracker) = LandmarkTracker::for_task(task, mode) {
            bench_tracker(&mut group, label, tracker, &states);
        }
    }

    bench_seeded_lm_cut(&mut group, task, &states);
    // A small explored tree (2 000 nodes), far below the paper's 100 000.
    let relevance = RelevanceScore::with_params(
        task,
        &RelevanceParams {
            min_nodes: 2_000,
            max_nodes: 2_000,
            ..RelevanceParams::default()
        },
    );
    bench_heuristic(&mut group, "relevance_2k", relevance, &states);
    bench_intention(&mut group, task, &states);

    let width = SasProblem::new(task);
    group.bench_function("width_search_h_add", |b| {
        b.iter(|| {
            states
                .iter()
                .map(|s| width.h_add(black_box(s)))
                .collect::<Vec<_>>()
        });
    });
    group.finish();
}

/// Benchmarks LM-cut from seeds with per-call operator costs (here: the state's facts at 0,
/// costs halved) two ways: rebuilding the task and `LmCut` per estimate, as a caller without
/// `LmCut::value_seeded` must, and `value_seeded` on one prebuilt `LmCut`.
fn bench_seeded_lm_cut(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    task: &Task,
    states: &[State],
) {
    let costs: Vec<u64> = task
        .operators
        .iter()
        .map(|op| u64::from(op.cost / 2))
        .collect();
    group.bench_function("lm_cut_rebuild", |b| {
        b.iter(|| {
            states
                .iter()
                .map(|s| {
                    let mut rebuilt = task.clone();
                    rebuilt.initial = s.clone();
                    for (op, &c) in rebuilt.operators.iter_mut().zip(&costs) {
                        op.cost = u32::try_from(c).expect("halved costs fit");
                    }

                    LmCut::new(&rebuilt).value(black_box(s))
                })
                .collect::<Vec<_>>()
        });
    });

    let mut lm_cut = LmCut::new(task);
    let mut seeds = Vec::new();
    group.bench_function("lm_cut_seeded", |b| {
        b.iter(|| {
            states
                .iter()
                .map(|s| {
                    lm_cut.seed_state(black_box(s), &mut seeds);
                    lm_cut.value_seeded(&seeds, Some(&costs))
                })
                .collect::<Vec<_>>()
        });
    });
}

/// Benchmarks the goal-intention likelihood as a search uses it: `on_expand` from the start
/// state to each sampled state along an applicable operator (or estimate the root), then
/// `estimate`. The one-off sampling (100 relaxed plans) is not timed.
fn bench_intention(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    task: &Task,
    states: &[State],
) {
    let intention = GoalIntention::new(task, &IntentionParams::default());
    let mut tracker = intention.tracker(IntentionKind::Likelihood, true);
    let start = &states[0];
    let root = tracker.root(start);
    let steps: Vec<(usize, State)> = task
        .applicable_operators(start)
        .map(|op| (op, task.apply(op, start)))
        .collect();
    group.bench_function("intention_ol_r", |b| {
        b.iter(|| {
            steps
                .iter()
                .map(|(op, child)| {
                    let info = tracker.on_expand(start, &root, *op, black_box(child));
                    tracker.estimate(child, &info)
                })
                .collect::<Vec<_>>()
        });
    });
}

/// Benchmarks `heuristic` on all `states` as one iteration.
fn bench_heuristic(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    label: &str,
    mut heuristic: impl Heuristic<State>,
    states: &[State],
) {
    group.bench_function(label, |b| {
        b.iter(|| {
            states
                .iter()
                .map(|s| heuristic.estimate(black_box(s)))
                .collect::<Vec<_>>()
        });
    });
}

/// Benchmarks a path-dependent heuristic as a search uses it: `on_expand` from the start
/// state's information to each sampled state, then `estimate`, all states as one iteration.
fn bench_tracker(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    label: &str,
    mut tracker: LandmarkTracker<'_>,
    states: &[State],
) {
    let start = &states[0];
    let root = tracker.root(start);
    group.bench_function(label, |b| {
        b.iter(|| {
            states
                .iter()
                .map(|s| {
                    let info = tracker.on_expand(start, &root, 0, black_box(s));
                    tracker.estimate(s, &info)
                })
                .collect::<Vec<_>>()
        });
    });
}
