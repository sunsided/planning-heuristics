//! The per-operator and per-black-variable tables of the red-black plan repair.

use alloc::vec::Vec;

use sas_plus::{CausalGraph, Fact, OpId, Task, VarId};

use super::painting::levels;
use crate::Relaxation;

/// An operator that changes a black variable: one arc of its domain transition graph.
#[derive(Clone, Copy, Debug)]
pub(super) struct Transition {
    /// The value the operator requires; `None` if it has no precondition on the variable.
    pub from: Option<u32>,
    /// The value the operator sets.
    pub to: u32,
    /// The operator.
    pub op: OpId,
}

/// The repair tables of one task under one painting. Black variables are numbered by their
/// position in the painting.
#[derive(Clone, Debug)]
pub(super) struct Tables {
    /// Whether each fact (by [`Relaxation::fact_id`]) belongs to a black variable.
    pub is_black_fact: Vec<bool>,
    /// For each black variable, the fact id of its value 0.
    pub black_offsets: Vec<usize>,
    /// For each operator, its preconditions on black variables as `(black index, value)`,
    /// descendants in the black causal graph first (by descending level, stable).
    pub black_pre: Vec<Vec<(usize, u32)>>,
    /// For each operator, its effects on black variables as `(black index, value)`; all
    /// unconditional, since a painting excludes variables of conditional effects.
    pub black_eff: Vec<Vec<(usize, u32)>>,
    /// For each operator, the fact ids of its preconditions on red variables.
    pub red_pre: Vec<Vec<usize>>,
    /// For each operator, its cost.
    pub cost: Vec<u64>,
    /// For each black variable, the operators that change it.
    pub transitions: Vec<Vec<Transition>>,
    /// For each black variable, its domain size.
    pub domain: Vec<u32>,
    /// The goal on black variables as `(black index, value)`, descendants first.
    pub black_goal: Vec<(usize, u32)>,
    /// Whether two black variables are linked in the causal graph (a DAG painting), so that
    /// moving one may require moving another.
    pub linked: bool,
    /// The fact ids of the goal facts on red variables.
    pub red_goal: Vec<usize>,
    /// For each operator, the fact ids of all its effects, conditional ones included.
    pub adds: Vec<Vec<usize>>,
    /// For each operator, the fact ids of its preconditions and effect conditions, ascending;
    /// `None` if one is on a derived variable (axioms may achieve it, so relaxed plan
    /// forwarding never moves the operator).
    pub needs: Vec<Option<Vec<usize>>>,
    /// The black variables (by index) in topological order of the black causal graph, roots
    /// first (by ascending level, stable).
    pub topological: Vec<usize>,
    /// For each black variable with domain size `n`, the cheapest cost from value `x` to
    /// value `y` at `x * n + y` over its domain transition graph, ignoring all conditions on
    /// other variables; `None` if `y` is unreachable from `x`.
    pub distance: Vec<Vec<Option<u64>>>,
    /// For each black variable, the fact ids of every value of each red variable an operator
    /// moving it sets: the red facts its moves may delete under real semantics.
    pub side_facts: Vec<Vec<usize>>,
}

impl Tables {
    /// Indexes `task` for the repair with the black variables `black` (ascending).
    ///
    /// # Panics
    ///
    /// Panics if a variable of `black` is not a variable of `task`.
    pub fn new(task: &Task, relaxation: &Relaxation, black: &[VarId]) -> Self {
        let (level, linked) = black_levels(task, black);
        let by_level = |facts: &mut Vec<(usize, u32)>| {
            facts.sort_by_key(|&(b, _)| core::cmp::Reverse(level[b]));
        };

        let (mut black_pre, red_pre): (Vec<_>, Vec<_>) = task
            .operators
            .iter()
            .map(|op| split_facts(relaxation, black, &op.pre))
            .unzip();
        black_pre.iter_mut().for_each(by_level);
        let (mut black_goal, red_goal) = split_facts(relaxation, black, &task.goal);
        by_level(&mut black_goal);

        let mut topological: Vec<usize> = (0..black.len()).collect();
        topological.sort_by_key(|&b| level[b]);
        let transitions = transitions(task, black);
        let side_facts = transitions
            .iter()
            .map(|ts| side_facts(task, relaxation, black, ts))
            .collect();

        Self {
            is_black_fact: black_fact_mask(task, relaxation, black),
            black_offsets: black
                .iter()
                .map(|&v| relaxation.fact_id(Fact::new(v, 0)))
                .collect(),
            black_pre,
            black_eff: black_effects(task, black),
            red_pre,
            cost: task.operators.iter().map(|op| u64::from(op.cost)).collect(),
            distance: black
                .iter()
                .zip(&transitions)
                .map(|(&v, ts)| distances(task, v, ts))
                .collect(),
            transitions,
            topological,
            side_facts,
            domain: black
                .iter()
                .map(|&v| task.variables[v].domain_size)
                .collect(),
            black_goal,
            red_goal,
            linked,
            adds: task
                .operators
                .iter()
                .map(|op| op.eff.iter().map(|e| relaxation.fact_id(e.fact)).collect())
                .collect(),
            needs: task
                .operators
                .iter()
                .map(|op| needs(task, relaxation, op))
                .collect(),
        }
    }
}

/// Whether each fact (by [`Relaxation::fact_id`]) belongs to one of the variables `black`.
///
/// # Panics
///
/// Panics if a variable of `black` is not a variable of `task`.
pub(super) fn black_fact_mask(task: &Task, relaxation: &Relaxation, black: &[VarId]) -> Vec<bool> {
    let mut is_black_fact = alloc::vec![false; relaxation.num_facts()];
    for &var in black {
        for value in 0..task.variables[var].domain_size {
            is_black_fact[relaxation.fact_id(Fact::new(var, value))] = true;
        }
    }

    is_black_fact
}

/// Splits `facts` into the facts on the variables `black` (ascending), as
/// `(black index, value)` in the order of `facts`, and the fact ids of the others.
fn split_facts(
    relaxation: &Relaxation,
    black: &[VarId],
    facts: &[Fact],
) -> (Vec<(usize, u32)>, Vec<usize>) {
    let black_part = facts
        .iter()
        .filter_map(|f| black.binary_search(&f.var).ok().map(|b| (b, f.value)))
        .collect();
    let red_part = facts
        .iter()
        .filter(|f| black.binary_search(&f.var).is_err())
        .map(|&f| relaxation.fact_id(f))
        .collect();

    (black_part, red_part)
}

/// For each operator of `task`, its effects on the variables `black` (ascending) as
/// `(black index, value)`.
fn black_effects(task: &Task, black: &[VarId]) -> Vec<Vec<(usize, u32)>> {
    task.operators
        .iter()
        .map(|op| {
            op.eff
                .iter()
                .filter_map(|e| {
                    black
                        .binary_search(&e.fact.var)
                        .ok()
                        .map(|b| (b, e.fact.value))
                })
                .collect()
        })
        .collect()
}

/// The fact ids `op` needs: preconditions and effect conditions, ascending; `None` if one is
/// on a derived variable.
fn needs(task: &Task, relaxation: &Relaxation, op: &sas_plus::Operator) -> Option<Vec<usize>> {
    let facts = op
        .pre
        .iter()
        .chain(op.eff.iter().flat_map(|e| &e.conditions));
    if facts.clone().any(|f| task.variables[f.var].is_derived()) {
        return None;
    }

    let mut ids: Vec<usize> = facts.map(|&f| relaxation.fact_id(f)).collect();
    ids.sort_unstable();
    ids.dedup();

    Some(ids)
}

/// The fact ids of every value of each red variable (not in `black`) that an operator of
/// `transitions` sets.
fn side_facts(
    task: &Task,
    relaxation: &Relaxation,
    black: &[VarId],
    transitions: &[Transition],
) -> Vec<usize> {
    let mut vars: Vec<VarId> = transitions
        .iter()
        .flat_map(|t| &task.operators[t.op].eff)
        .map(|e| e.fact.var)
        .filter(|v| black.binary_search(v).is_err())
        .collect();
    vars.sort_unstable();
    vars.dedup();

    vars.iter()
        .flat_map(|&v| {
            (0..task.variables[v].domain_size)
                .map(move |value| relaxation.fact_id(Fact::new(v, value)))
        })
        .collect()
}

/// All-pairs cheapest costs over the domain transition graph of `var` with `transitions`,
/// ignoring outside conditions (Floyd–Warshall; domains are small); see
/// [`Tables::distance`].
fn distances(task: &Task, var: VarId, transitions: &[Transition]) -> Vec<Option<u64>> {
    let n = task.variables[var].domain_size as usize;
    let mut d: Vec<Option<u64>> = alloc::vec![None; n * n];
    for x in 0..n {
        d[x * n + x] = Some(0);
    }

    for t in transitions {
        let cost = u64::from(task.operators[t.op].cost);
        let to = t.to as usize;
        let sources = t.from.map_or(0..n, |f| f as usize..f as usize + 1);
        for x in sources.filter(|&x| x != to) {
            let slot = &mut d[x * n + to];
            *slot = Some(slot.map_or(cost, |old| old.min(cost)));
        }
    }

    for k in 0..n {
        for x in 0..n {
            let Some(xk) = d[x * n + k] else { continue };
            for y in 0..n {
                if let Some(ky) = d[k * n + y] {
                    let via = xk.saturating_add(ky);
                    let slot = &mut d[x * n + y];
                    *slot = Some(slot.map_or(via, |old| old.min(via)));
                }
            }
        }
    }

    d
}

/// For each black variable, the operators that set it, with the value they require.
fn transitions(task: &Task, black: &[VarId]) -> Vec<Vec<Transition>> {
    black
        .iter()
        .map(|&var| {
            task.operators
                .iter()
                .enumerate()
                .flat_map(|(op, o)| {
                    let from = o.pre.iter().find(|f| f.var == var).map(|f| f.value);
                    o.eff
                        .iter()
                        .filter(move |e| e.fact.var == var)
                        .map(move |e| Transition {
                            from,
                            to: e.fact.value,
                            op,
                        })
                })
                .collect()
        })
        .collect()
}

/// Per black variable (by position in `black`), its level in the causal graph induced on the
/// black variables (longest path from a source), and whether that graph has an arc at all.
fn black_levels(task: &Task, black: &[VarId]) -> (Vec<usize>, bool) {
    let graph = CausalGraph::new(task);
    let successors: Vec<Vec<usize>> = black
        .iter()
        .map(|&v| {
            graph
                .successors(v)
                .iter()
                .filter_map(|w| black.binary_search(w).ok())
                .collect()
        })
        .collect();
    let linked = successors.iter().any(|s| !s.is_empty());

    (levels(black.len(), |b| &successors[b]), linked)
}
