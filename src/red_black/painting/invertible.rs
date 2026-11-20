//! Relaxed side effects invertibility (RSE-invertibility) of a variable's domain transition
//! graph (Katz, Hoffmann & Domshlak, AAAI 2013; Domshlak, Hoffmann & Katz, AIJ 2015,
//! Section 4.3).

use alloc::vec::Vec;

use sas_plus::{Fact, Operator, Task, VarId};

/// Whether `var` is RSE-invertible: every arc `(d, a, d')` of its domain transition graph has
/// an inverse arc `(d', a', d)` whose outside condition lies in the outside condition plus the
/// outside effect of `(d, a, d')`.
///
/// An operator without a precondition on `var` induces an arc from every other value. Only
/// unconditional effects count; a variable that a conditional effect sets is never
/// invertible (no arc of it is guaranteed to fire). Arcs from a value to itself are not arcs.
pub(super) fn is_rse_invertible(task: &Task, var: VarId) -> bool {
    let affecting: Vec<&Operator> = task
        .operators
        .iter()
        .filter(|op| op.eff.iter().any(|e| e.fact.var == var))
        .collect();
    if affecting
        .iter()
        .flat_map(|op| &op.eff)
        .any(|e| e.fact.var == var && !e.conditions.is_empty())
    {
        return false;
    }

    let domain = task.variables[var].domain_size;
    affecting.iter().all(|op| {
        let to = effect_on(op, var).expect("the operator affects `var`");
        sources(op, var, domain)
            .filter(|&from| from != to)
            .all(|from| affecting.iter().any(|inv| inverts(inv, op, var, from, to)))
    })
}

/// Whether `inverse` induces the arc `(to, inverse, from)` of `var`'s domain transition graph
/// with an outside condition inside the outside condition and outside effect of `op`.
fn inverts(inverse: &Operator, op: &Operator, var: VarId, from: u32, to: u32) -> bool {
    let starts_at_to = pre_on(inverse, var).is_none_or(|v| v == to);
    let allowed = |f: &Fact| op.pre.contains(f) || op.eff.iter().any(|e| e.fact == *f);

    effect_on(inverse, var) == Some(from)
        && starts_at_to
        && inverse.pre.iter().filter(|f| f.var != var).all(allowed)
}

/// The values `op` can change `var` from: its precondition on `var`, or every value.
fn sources(op: &Operator, var: VarId, domain: u32) -> impl Iterator<Item = u32> {
    match pre_on(op, var) {
        Some(value) => value..value + 1,
        None => 0..domain,
    }
}

/// The value `op` requires of `var`, if any.
fn pre_on(op: &Operator, var: VarId) -> Option<u32> {
    op.pre.iter().find(|f| f.var == var).map(|f| f.value)
}

/// The value `op` sets `var` to, if any.
fn effect_on(op: &Operator, var: VarId) -> Option<u32> {
    op.eff
        .iter()
        .find(|e| e.fact.var == var)
        .map(|e| e.fact.value)
}

#[cfg(test)]
mod tests {
    //! RSE-invertibility on the examples of the papers.

    use alloc::vec;
    use alloc::vec::Vec;

    use sas_plus::{Effect, Fact, Operator, State, Task, Variable};

    use super::is_rse_invertible;

    /// An operator with precondition `pre` and unconditional effects `eff`.
    fn op(pre: Vec<Fact>, eff: Vec<Fact>) -> Operator {
        Operator {
            name: "op".into(),
            pre,
            eff: eff.into_iter().map(Effect::new).collect(),
            cost: 1,
        }
    }

    /// A task with binary variables `hand` (0) and `key` (1: at 0, at 1, held = 2) plus a
    /// robot position (2), and `ops`.
    fn task(ops: Vec<Operator>) -> Task {
        Task {
            variables: vec![
                Variable::new("hand", 2),
                Variable::new("key", 3),
                Variable::new("robot", 2),
            ],
            operators: ops,
            initial: State(vec![1, 0, 0]),
            goal: vec![],
            axioms: vec![],
        }
    }

    /// The hand variable of the `SimpleGrid` example: `take` frees the hand only via its side
    /// effect on the key, so `drop` (which needs the key held) inverts it only because the
    /// outside effect counts.
    #[test]
    fn side_effects_may_enable_the_inverse() {
        let (hand, key, robot) = (0, 1, 2);
        let take = |x| {
            op(
                vec![Fact::new(hand, 1), Fact::new(robot, x), Fact::new(key, x)],
                vec![Fact::new(hand, 0), Fact::new(key, 2)],
            )
        };
        let drop = |x| {
            op(
                vec![Fact::new(hand, 0), Fact::new(robot, x), Fact::new(key, 2)],
                vec![Fact::new(hand, 1), Fact::new(key, x)],
            )
        };
        let grid = task(vec![take(0), take(1), drop(0), drop(1)]);
        assert!(is_rse_invertible(&grid, hand));

        let only_take = task(vec![take(0), drop(1)]);
        assert!(!is_rse_invertible(&only_take, hand));
    }

    /// An operator without a precondition on the variable needs an inverse from every value;
    /// a variable nothing changes is trivially invertible.
    #[test]
    fn unconditioned_arcs_need_every_inverse() {
        let set = |v| op(vec![], vec![Fact::new(1, v)]);
        assert!(is_rse_invertible(&task(vec![set(0), set(1), set(2)]), 1));
        assert!(!is_rse_invertible(&task(vec![set(0), set(1)]), 1));
        assert!(is_rse_invertible(&task(vec![]), 1));
    }

    /// An inverse needing a condition the original arc neither required nor produced does not
    /// count.
    #[test]
    fn extra_conditions_break_invertibility() {
        let up = op(vec![Fact::new(2, 0)], vec![Fact::new(2, 1)]);
        let down = op(
            vec![Fact::new(2, 1), Fact::new(0, 0)],
            vec![Fact::new(2, 0)],
        );
        let down_free = op(vec![Fact::new(2, 1)], vec![Fact::new(2, 0)]);
        assert!(!is_rse_invertible(&task(vec![up.clone(), down.clone()]), 2));
        assert!(is_rse_invertible(&task(vec![up, down, down_free]), 2));
    }
}
