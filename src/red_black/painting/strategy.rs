//! The painting algorithm of Domshlak, Hoffmann & Katz (AIJ 2015, Figure 8): start with the
//! variables that cannot or need not be black painted red, paint more red one at a time until
//! the black causal graph is in the fragment, then repaint black whatever fits again.

use alloc::vec::Vec;
use core::cmp::Reverse;

use sas_plus::{CausalGraph, State, Task, VarId};

use super::invertible::is_rse_invertible;
use super::{CausalLevel, Fragment, cycle_arc, ineligible, levels, linked, max_by_key};
use crate::Relaxation;
use crate::rng::SplitMix64;

/// The choice point of the painting algorithm of Domshlak, Hoffmann & Katz (2015, Figure 8):
/// which black variable to paint red next. Every strategy stops at a maximal painting for its
/// [`Fragment`]; see [`super::Painting::with_strategy`].
///
/// Ties are broken as in the paper: A, its inverse and `Mixed` by smaller domain, then by L;
/// C, its inverse and `SampledConflicts` by L. L itself breaks ties by variable id (see
/// [`super::Painting::causal_graph_level`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Strategy {
    /// L (with [`CausalLevel::Top`]): paint red the variable with the highest causal graph
    /// level, closest to the leaves; L̄ ([`CausalLevel::Bottom`]): the lowest.
    Level(CausalLevel),
    /// A: the variable with the most causal graph arcs to other black variables, so few
    /// variables end up red.
    MostArcs,
    /// Ā, the inverse of A (a sanity check of the paper): the fewest arcs.
    FewestArcs,
    /// C: the variable with the fewest conflicts with the relaxed plan of the given state, so
    /// the most critical variables stay black.
    FewestConflicts,
    /// C̄, the inverse of C (a sanity check): the most conflicts.
    MostConflicts,
    /// `C[N]`: like C, with the conflicts summed (equivalently, averaged) over the relaxed plans
    /// of `samples` states sampled by random walks from the given state, which see conflicts
    /// that only show later in a plan.
    SampledConflicts {
        /// The number of sampled states, `N`; the given state itself is not counted.
        samples: usize,
        /// The seed of the random walks.
        seed: u64,
    },
    /// `CA[p]`: the variable maximising `p · Â(v) + (1 − p) · (1 − Ĉ(v))`, where `Â` and `Ĉ` are
    /// the arc and conflict counts divided by their maxima over the black variables; `p` is
    /// `arcs_percent / 100`. `arcs_percent = 100` is A; 0 is C with the ties of A.
    Mixed {
        /// The weight `p` of the arcs, in percent (at most 100).
        arcs_percent: u8,
    },
    /// RND: a variable chosen uniformly at random (a sanity check).
    Random {
        /// The seed.
        seed: u64,
    },
}

/// The black variables of `task` painted with `strategy` for `fragment` (ascending); the
/// conflict strategies use the relaxed plan of `state`.
///
/// # Panics
///
/// Panics if `state` has fewer values than the task has variables, or `Mixed` has a weight
/// above 100.
pub(super) fn paint_with(
    task: &Task,
    state: &State,
    strategy: Strategy,
    fragment: Fragment,
) -> Vec<VarId> {
    let graph = CausalGraph::new(task);
    let level = levels(task.num_variables(), |v| graph.successors(v));
    let domain = |v: VarId| task.variables[v].domain_size;
    let conflicts = || match strategy {
        Strategy::SampledConflicts { samples, seed } => {
            sampled_conflicts(task, state, samples, seed)
        }
        _ => state_conflicts(task, state),
    };

    match strategy {
        Strategy::Level(CausalLevel::Top) => paint(task, &graph, fragment, |black| {
            max_by_key(black, |v| (level[v], v))
        }),
        Strategy::Level(CausalLevel::Bottom) => paint(task, &graph, fragment, |black| {
            max_by_key(black, |v| (Reverse(level[v]), Reverse(v)))
        }),
        Strategy::MostArcs => paint(task, &graph, fragment, |black| {
            let arcs = |v| incident_arcs(&graph, black, v);
            max_by_key(black, |v| (arcs(v), Reverse(domain(v)), level[v], v))
        }),
        Strategy::FewestArcs => paint(task, &graph, fragment, |black| {
            let arcs = |v| incident_arcs(&graph, black, v);
            max_by_key(black, |v| {
                (Reverse(arcs(v)), Reverse(domain(v)), level[v], v)
            })
        }),
        Strategy::FewestConflicts | Strategy::SampledConflicts { .. } => {
            let c = conflicts();
            paint(task, &graph, fragment, |black| {
                max_by_key(black, |v| (Reverse(c[v]), level[v], v))
            })
        }
        Strategy::MostConflicts => {
            let c = conflicts();
            paint(task, &graph, fragment, |black| {
                max_by_key(black, |v| (c[v], level[v], v))
            })
        }
        Strategy::Mixed { arcs_percent } => {
            assert!(arcs_percent <= 100, "CA[p] needs p <= 1");
            let c = conflicts();
            paint(task, &graph, fragment, |black| {
                let arcs = |v| incident_arcs(&graph, black, v);
                let weight = mixed_weight(black, arcs, &c, arcs_percent);
                max_by_key(black, |v| (weight(v), Reverse(domain(v)), level[v], v))
            })
        }
        Strategy::Random { seed } => {
            let mut rng = SplitMix64(seed);
            paint(task, &graph, fragment, |black| {
                black[rng.below(black.len())]
            })
        }
    }
}

/// The `CA[p]` score of each variable as an exact integer, `P(v)` scaled by
/// `100 · max(Â) · max(Ĉ)` (each maximum at least 1, which leaves a zero count at zero).
fn mixed_weight<'a>(
    black: &'a [VarId],
    arcs: impl Fn(VarId) -> usize + 'a,
    conflicts: &'a [usize],
    arcs_percent: u8,
) -> impl Fn(VarId) -> u128 + 'a {
    let wide = |x: usize| u128::try_from(x).expect("counts fit u128");
    let max_arcs = wide(black.iter().map(|&v| arcs(v)).max().unwrap_or(0).max(1));
    let max_conflicts = wide(
        black
            .iter()
            .map(|&v| conflicts[v])
            .max()
            .unwrap_or(0)
            .max(1),
    );
    let p = u128::from(arcs_percent);

    move |v| {
        p * wide(arcs(v)) * max_conflicts
            + (100 - p) * (max_conflicts - wide(conflicts[v])) * max_arcs
    }
}

/// The conflicts of the relaxed plan of `state` (all zero if it is a dead end).
fn state_conflicts(task: &Task, state: &State) -> Vec<usize> {
    Relaxation::new(task).relaxed_plan(state).map_or_else(
        || alloc::vec![0; task.num_variables()],
        |plan| conflicts(task, &state.0, plan.ops()),
    )
}

/// The conflicts summed over the relaxed plans of `samples` states, each the end of a random
/// walk from `state` of a length drawn uniformly from `0..=2 · |relaxed plan of state|`,
/// stopping early in a state without applicable operators. Dead-end samples count no
/// conflicts.
///
/// The paper does not say how it samples; random walks of about twice the relaxed plan
/// length mirror how Fast Downward samples states.
fn sampled_conflicts(task: &Task, state: &State, samples: usize, seed: u64) -> Vec<usize> {
    let relaxation = Relaxation::new(task);
    let depth = relaxation
        .relaxed_plan(state)
        .map_or(0, |plan| 2 * plan.ops().len());
    let mut rng = SplitMix64(seed);
    let mut total = alloc::vec![0; task.num_variables()];
    let mut applicable = Vec::new();
    for _ in 0..samples {
        let mut sample = state.clone();
        for _ in 0..rng.below(depth + 1) {
            applicable.clear();
            applicable.extend(task.applicable_operators(&sample));
            if applicable.is_empty() {
                break;
            }

            sample = task.apply(applicable[rng.below(applicable.len())], &sample);
        }

        if let Some(plan) = relaxation.relaxed_plan(&sample) {
            for (t, c) in total.iter_mut().zip(conflicts(task, &sample.0, plan.ops())) {
                *t += c;
            }
        }
    }

    total
}

/// The black variables of the Figure 8 painting of `task` for `fragment`, ascending.
/// `select` is the choice point: given the variables still black (ascending, at least two),
/// it returns the one to paint red next.
///
/// Initially red: the variables that are ineligible (see [`super::Painting`]) or not
/// RSE-invertible, and the leaves of the causal graph (no outgoing arc), which by Theorem 14 of
/// the paper do not change the ideal red-black heuristic. The post-process walks the variables
/// painted red by `select` from last to first and paints each black again if the black causal
/// graph stays in the fragment, so the result is maximal: no red variable outside the initial
/// set can be painted black without leaving the fragment.
pub(super) fn paint(
    task: &Task,
    graph: &CausalGraph,
    fragment: Fragment,
    mut select: impl FnMut(&[VarId]) -> VarId,
) -> Vec<VarId> {
    let excluded = ineligible(task);
    let mut black: Vec<VarId> = (0..task.num_variables())
        .filter(|&v| excluded[v].is_none() && !graph.successors(v).is_empty())
        .filter(|&v| is_rse_invertible(task, v))
        .collect();
    let mut painted_red = Vec::new();
    while !fits(graph, &black, fragment) {
        let v = select(&black);
        black.retain(|&u| u != v);
        painted_red.push(v);
    }

    for &v in painted_red.iter().rev() {
        let pos = black.binary_search(&v).unwrap_err();
        black.insert(pos, v);
        if !fits(graph, &black, fragment) {
            black.remove(pos);
        }
    }

    black
}

/// Whether the causal graph induced on `black` (ascending) is in `fragment`.
pub(super) fn fits(graph: &CausalGraph, black: &[VarId], fragment: Fragment) -> bool {
    match fragment {
        Fragment::Arcless => black
            .iter()
            .enumerate()
            .all(|(i, &u)| black[i + 1..].iter().all(|&v| !linked(graph, u, v))),
        Fragment::Dag => cycle_arc(graph, black).is_none(),
    }
}

/// The number of causal graph arcs between `v` and the other variables of `black`, counting
/// each direction: the `A(v)` of strategy A.
pub(super) fn incident_arcs(graph: &CausalGraph, black: &[VarId], v: VarId) -> usize {
    let count = |arcs: &[VarId]| {
        arcs.iter()
            .filter(|&&u| u != v && black.binary_search(&u).is_ok())
            .count()
    };

    count(graph.successors(v)) + count(graph.predecessors(v))
}

/// Per variable, the actions of the relaxed `plan` from `state` whose precondition on it is
/// violated when the plan is executed with only that variable black: the `C(v)` of strategy
/// C. After a violation the value is assumed moved to the precondition (as the repair would);
/// effects then set it as usual.
pub(super) fn conflicts(task: &Task, state: &[u32], plan: &[sas_plus::OpId]) -> Vec<usize> {
    let mut value = state.to_vec();
    let mut conflicts = alloc::vec![0; task.num_variables()];
    for &op in plan {
        let operator = &task.operators[op];
        for f in &operator.pre {
            if value[f.var] != f.value {
                conflicts[f.var] += 1;
                value[f.var] = f.value;
            }
        }

        for e in operator.eff.iter().filter(|e| e.conditions.is_empty()) {
            value[e.fact.var] = e.fact.value;
        }
    }

    conflicts
}

#[cfg(test)]
mod tests {
    //! The Figure 8 loop and its maximality post-process.

    use alloc::format;
    use alloc::vec;
    use alloc::vec::Vec;

    use sas_plus::{CausalGraph, Effect, Fact, Operator, State, Task, Variable};

    use super::{Fragment, Strategy, paint, paint_with};

    /// A free-hand variable (0) and three packages (1 to 3); loading or unloading a package
    /// changes both, so the hand and each package form a causal graph cycle.
    fn star() -> Task {
        let op = |pre: Vec<Fact>, eff: Vec<Fact>| Operator {
            name: "op".into(),
            pre,
            eff: eff.into_iter().map(Effect::new).collect(),
            cost: 1,
        };
        let operators = (1..4)
            .flat_map(|x| {
                [
                    op(
                        vec![Fact::new(0, 1), Fact::new(x, 0)],
                        vec![Fact::new(0, 0), Fact::new(x, 1)],
                    ),
                    op(
                        vec![Fact::new(0, 0), Fact::new(x, 1)],
                        vec![Fact::new(0, 1), Fact::new(x, 0)],
                    ),
                ]
            })
            .collect();

        Task {
            variables: (0..4).map(|i| Variable::new(format!("v{i}"), 2)).collect(),
            operators,
            initial: State(vec![1, 0, 0, 0]),
            goal: vec![Fact::new(1, 1)],
            axioms: vec![],
        }
    }

    /// Painting packages 1 and 2 red first and the hand last leaves 1 and 2 red for nothing;
    /// the post-process paints them black again (the `StarShapeLogistics` remark of Domshlak,
    /// Hoffmann & Katz 2015, Section 6.1).
    #[test]
    fn post_process_restores_maximality() {
        let task = star();
        let graph = CausalGraph::new(&task);
        let mut order = vec![1, 2, 0].into_iter();
        let black = paint(&task, &graph, Fragment::Dag, |black| {
            let v = order.next().expect("three picks suffice");
            assert!(black.contains(&v));
            v
        });

        assert_eq!(black, [1, 2, 3]);
        assert_eq!(order.next(), None);
    }

    /// CA[1] is A, and a sampled C with no samples sees no conflicts, so it is L; RND is
    /// deterministic per seed and maximal like the rest.
    #[test]
    fn strategy_special_cases() {
        let task = star();
        let state = task.initial_state();
        for fragment in [Fragment::Arcless, Fragment::Dag] {
            let with = |strategy| paint_with(&task, &state, strategy, fragment);
            assert_eq!(
                with(Strategy::Mixed { arcs_percent: 100 }),
                with(Strategy::MostArcs)
            );
            assert_eq!(
                with(Strategy::SampledConflicts {
                    samples: 0,
                    seed: 9
                }),
                with(Strategy::Level(super::CausalLevel::Top))
            );
            let random = with(Strategy::Random { seed: 5 });
            assert_eq!(random, with(Strategy::Random { seed: 5 }));
            assert_ne!(random, [] as [usize; 0]);
        }
    }
}
