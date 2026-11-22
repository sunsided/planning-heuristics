//! Context splitting: compiling conditional effects away, so that LM-cut's cuts are charged
//! per effect context instead of per operator (after Röger, Pommerening & Helmert 2014; the
//! splitting of operators by the context of their effect conditions also appears in Keyder,
//! Hoffmann & Haslum 2014).

use alloc::format;
use alloc::vec::Vec;

use sas_plus::{Effect, Fact, OpId, Operator, Task, VarId};

/// A task whose operators are the context copies of another task's operators.
#[derive(Clone, Debug)]
pub(crate) struct Split {
    /// The compiled task: same variables, axioms, initial state and goal.
    pub(crate) task: Task,
    /// For each operator of [`Split::task`], the operator of the original task it copies.
    pub(crate) op_of: Vec<OpId>,
}

/// One option for a context variable: a value its effect conditions mention, or none of
/// them (the copy then has no precondition on it and none of the effects that need it).
type Choice = Option<u32>;

/// Splits every operator of `task` with conditional effects into one copy per *context*: an
/// assignment of each variable its effect conditions mention (and its precondition does not
/// fix) to one of the mentioned values or to "any other". A copy has the context as extra
/// precondition and, as unconditional effects, the effects whose conditions the context
/// satisfies. An operator with more than `max_contexts` contexts stays as it is.
///
/// Every application of an operator is an application of the copy for its state's context,
/// with the same effects, so the compiled task has every plan of the original; the
/// "any other" copies only add options. `None` if `task` has no conditional effects.
pub(crate) fn split(task: &Task, max_contexts: usize) -> Option<Split> {
    split_by(task, max_contexts, false)
}

/// As [`split`], but exact: a context assigns each variable one of all its values, with no
/// "any other" option, so exactly one copy of an operator applies in each state and the
/// compiled task has exactly the plans of the original (for red-black planning, where an
/// "any other" copy would let a black variable skip an effect it really triggers).
pub(crate) fn split_exact(task: &Task, max_contexts: usize) -> Option<Split> {
    split_by(task, max_contexts, true)
}

/// [`split`], or with `exact` [`split_exact`].
fn split_by(task: &Task, max_contexts: usize, exact: bool) -> Option<Split> {
    let conditional = |op: &Operator| op.eff.iter().any(|e| !e.conditions.is_empty());
    if !task.operators.iter().any(conditional) {
        return None;
    }

    let mut operators = Vec::new();
    let mut op_of = Vec::new();
    let domains: Option<Vec<u32>> =
        exact.then(|| task.variables.iter().map(|v| v.domain_size).collect());
    for (id, op) in task.operators.iter().enumerate() {
        let copies = split_operator(op, max_contexts, domains.as_deref());
        for copy in copies.unwrap_or_else(|| alloc::vec![op.clone()]) {
            operators.push(copy);
            op_of.push(id);
        }
    }

    Some(Split {
        task: Task {
            operators,
            ..task.clone()
        },
        op_of,
    })
}

/// The context copies of `op`; `None` if it has more than `max_contexts` contexts. With the
/// variables' `domains`, every context variable takes each of its values instead of the
/// mentioned ones plus "any other".
fn split_operator(
    op: &Operator,
    max_contexts: usize,
    domains: Option<&[u32]>,
) -> Option<Vec<Operator>> {
    let effects = simplified_effects(op);
    let mut vars: Vec<(VarId, Vec<Choice>)> = Vec::new();
    for c in effects.iter().flat_map(|e| &e.conditions) {
        match vars.iter_mut().find(|(v, _)| *v == c.var) {
            Some((_, choices)) if !choices.contains(&Some(c.value)) => choices.push(Some(c.value)),
            Some(_) => {}
            None => vars.push((c.var, alloc::vec![None, Some(c.value)])),
        }
    }

    // In exact mode a variable has one choice per value; count before building them, so a
    // huge domain is rejected without being materialized.
    let choices =
        |(var, choices): &(VarId, Vec<Choice>)| domains.map_or(choices.len(), |d| d[*var] as usize);
    let count = vars
        .iter()
        .try_fold(1_usize, |n, v| n.checked_mul(choices(v)))?;
    if count > max_contexts {
        return None;
    }

    if let Some(domains) = domains {
        for (var, choices) in &mut vars {
            *choices = (0..domains[*var]).map(Some).collect();
        }
    }

    let copies = (0..count)
        .map(|k| {
            let context = context(&vars, k);
            copy_for(op, &effects, &context, k)
        })
        .collect();

    Some(copies)
}

/// The effects of `op` with the conditions its precondition already guarantees removed, and
/// without the effects whose conditions contradict it (they never fire).
fn simplified_effects(op: &Operator) -> Vec<Effect> {
    let pre = |var: VarId| op.pre.iter().find(|f| f.var == var).map(|f| f.value);
    op.eff
        .iter()
        .filter(|e| {
            e.conditions
                .iter()
                .all(|c| pre(c.var).is_none_or(|v| v == c.value))
        })
        .map(|e| Effect {
            conditions: e
                .conditions
                .iter()
                .copied()
                .filter(|c| pre(c.var).is_none())
                .collect(),
            fact: e.fact,
        })
        .collect()
}

/// The `k`-th context over `vars`, in mixed radix: one fact per variable assigned a value.
fn context(vars: &[(VarId, Vec<Choice>)], mut k: usize) -> Vec<Fact> {
    let mut facts = Vec::new();
    for (var, choices) in vars {
        if let Some(value) = choices[k % choices.len()] {
            facts.push(Fact::new(*var, value));
        }

        k /= choices.len();
    }

    facts
}

/// The copy of `op` for `context`: its precondition plus the context, and the `effects`
/// whose conditions the context satisfies, unconditional and in their original order.
fn copy_for(op: &Operator, effects: &[Effect], context: &[Fact], k: usize) -> Operator {
    let mut pre = op.pre.clone();
    pre.extend_from_slice(context);
    let eff = effects
        .iter()
        .filter(|e| e.conditions.iter().all(|c| context.contains(c)))
        .map(|e| Effect::new(e.fact))
        .collect();

    Operator {
        name: format!("{}#{k}", op.name),
        pre,
        eff,
        cost: op.cost,
    }
}

#[cfg(test)]
mod tests {
    //! The copies of single operators.

    use alloc::vec::Vec;

    use sas_plus::{Effect, Fact, Operator};

    use super::split_operator;

    /// `set c if a=1`, `set d if a=2 and b=1`, plus an effect contradicting the precondition:
    /// contexts over `a ∈ {any, 1, 2}` and `b ∈ {any, 1}`, six copies, over a limit of five
    /// none.
    #[test]
    fn copies_cover_every_context() {
        let op = Operator {
            name: "op".into(),
            pre: alloc::vec![Fact::new(4, 0)],
            eff: alloc::vec![
                Effect {
                    conditions: alloc::vec![Fact::new(0, 1)],
                    fact: Fact::new(2, 1)
                },
                Effect {
                    conditions: alloc::vec![Fact::new(0, 2), Fact::new(1, 1)],
                    fact: Fact::new(3, 1),
                },
                Effect {
                    conditions: alloc::vec![Fact::new(4, 1)],
                    fact: Fact::new(5, 1)
                },
                Effect {
                    conditions: alloc::vec![Fact::new(4, 0)],
                    fact: Fact::new(6, 1)
                },
            ],
            cost: 3,
        };
        let copies = split_operator(&op, 6, None).unwrap();
        assert_eq!(copies.len(), 6);
        assert!(split_operator(&op, 5, None).is_none());

        let effects_with = |pre: &[Fact]| -> Vec<Fact> {
            let copy = copies.iter().find(|c| c.pre[1..] == *pre).unwrap();
            assert!(copy.eff.iter().all(|e| e.conditions.is_empty()) && copy.cost == 3);
            copy.eff.iter().map(|e| e.fact).collect()
        };
        assert_eq!(effects_with(&[]), [Fact::new(6, 1)]);
        assert_eq!(
            effects_with(&[Fact::new(0, 1)]),
            [Fact::new(2, 1), Fact::new(6, 1)]
        );
        assert_eq!(effects_with(&[Fact::new(0, 2)]), [Fact::new(6, 1)]);
        assert_eq!(
            effects_with(&[Fact::new(0, 2), Fact::new(1, 1)]),
            [Fact::new(3, 1), Fact::new(6, 1)]
        );
    }

    /// In exact mode a context variable takes every value of its domain, and a domain too
    /// large for the limit is rejected before its values are built.
    #[test]
    fn exact_copies_take_every_value() {
        let op = Operator {
            name: "op".into(),
            pre: alloc::vec![],
            eff: alloc::vec![Effect {
                conditions: alloc::vec![Fact::new(0, 1)],
                fact: Fact::new(1, 1)
            }],
            cost: 1,
        };
        let copies = split_operator(&op, 64, Some(&[3, 2])).unwrap();
        assert_eq!(copies.len(), 3);
        assert!(copies.iter().all(|c| c.pre.len() == 1));
        assert!(split_operator(&op, 64, Some(&[u32::MAX, 2])).is_none());
    }
}
