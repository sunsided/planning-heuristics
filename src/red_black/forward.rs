//! Relaxed plan forwarding, option F of Katz, Hoffmann & Domshlak (AAAI 2013): before the
//! repair, move every operator without black effects as close to the start of the relaxed
//! plan as it can go.

use sas_plus::OpId;

use super::tables::Tables;

/// Reorders `plan` in place: each operator without black effects, in plan order, moves left
/// past its predecessors until one achieves a fact it needs (a precondition or an effect
/// condition), as in the paper, which avoids rechecking the relaxed plan after each move.
/// An operator needing a derived fact never moves (axioms, not operators, achieve it). The
/// result is a relaxed plan with the same operators.
pub(super) fn forward(tables: &Tables, plan: &mut [OpId]) {
    for i in 0..plan.len() {
        let op = plan[i];
        if !tables.black_eff[op].is_empty() {
            continue;
        }

        let Some(needs) = &tables.needs[op] else {
            continue;
        };

        let mut j = i;
        while j > 0 && !achieves(tables, plan[j - 1], needs) {
            plan.swap(j - 1, j);
            j -= 1;
        }
    }
}

/// Whether an effect of `op` is one of the facts `needs` (ascending).
fn achieves(tables: &Tables, op: OpId, needs: &[usize]) -> bool {
    tables.adds[op]
        .iter()
        .any(|f| needs.binary_search(f).is_ok())
}

#[cfg(test)]
mod tests {
    //! Forwarding on a hand-built plan.

    use alloc::vec;
    use alloc::vec::Vec;

    use sas_plus::{Effect, Fact, Operator, State, Task, Variable};

    use super::forward;
    use crate::Relaxation;
    use crate::red_black::tables::Tables;

    /// Black `t` (0) moves 0 → 1 (op 0); red op 1 needs nothing, red op 2 needs `t = 1`, red
    /// op 3 needs op 1's effect. Forwarding yields `1, 3, 0, 2`: op 1 to the front, op 3 right
    /// behind its achiever, op 2 stays behind the move it needs.
    #[test]
    fn moves_red_operators_up_to_their_achievers() {
        let op = |pre: Vec<Fact>, eff: Fact| Operator {
            name: "op".into(),
            pre,
            eff: vec![Effect::new(eff)],
            cost: 1,
        };
        let task = Task {
            variables: (0..4)
                .map(|i| Variable::new(alloc::format!("v{i}"), 2))
                .collect(),
            operators: vec![
                op(vec![Fact::new(0, 0)], Fact::new(0, 1)),
                op(vec![], Fact::new(1, 1)),
                op(vec![Fact::new(0, 1)], Fact::new(2, 1)),
                op(vec![Fact::new(1, 1)], Fact::new(3, 1)),
            ],
            initial: State(vec![0; 4]),
            goal: vec![],
            axioms: vec![],
        };
        let relaxation = Relaxation::new(&task);
        let tables = Tables::new(&task, &relaxation, &[0]);
        let mut plan = vec![0, 2, 1, 3];
        forward(&tables, &mut plan);

        assert_eq!(plan, [1, 3, 0, 2]);
    }
}
