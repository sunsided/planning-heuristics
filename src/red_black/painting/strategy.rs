//! The painting algorithm of Domshlak, Hoffmann & Katz (AIJ 2015, Figure 8): start with the
//! variables that cannot or need not be black painted red, paint more red one at a time until
//! the black causal graph is in the fragment, then repaint black whatever fits again.

use alloc::vec::Vec;

use sas_plus::{CausalGraph, Task, VarId};

use super::invertible::is_rse_invertible;
use super::{Fragment, cycle_arc, ineligible, linked};

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

    use super::{Fragment, paint};

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
}
