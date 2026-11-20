//! Relaxed plan extraction from `h_add` best supporters (the `h_FF` relaxed plan).
//!
//! The goal facts are traced back through their best supporters, which marks the rules a
//! relaxed plan needs. The marked rules are then replayed in the order their heads were
//! closed, under the relaxation: an operator is appended only if the fact its marked rule
//! achieves is not true yet. Applying an operator fires all its effects whose conditions
//! hold, so without conditional effects every operator appears at most once. With them an
//! operator can appear twice when one of its effects needs another of its effects first;
//! counting it once would undercut `h_max`.
//!
//! So with conditional effects `h_FF` here can be higher than Fast Downward's, which counts
//! every operator of the relaxed plan once. Example: `o` (cost 1) adds `a` and, if `a` holds,
//! `b`; goal `b`. The plan is `[o, o]` with `h_FF = 2 = h_max`; Fast Downward gives `1`.

use alloc::vec;
use alloc::vec::Vec;

use sas_plus::{OpId, State};

use super::{Combine, Exploration, Relaxation};

/// A relaxed plan: operators that reach the goal when deletes are ignored.
///
/// Applying [`RelaxedPlan::ops`] in order from the evaluated state, with every effect only
/// adding its fact and derived facts closed under the axioms after each step, reaches the
/// goal.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RelaxedPlan {
    /// The operators in application order. Without conditional effects no operator repeats,
    /// so this is also the set of operators of the plan. With conditional effects an operator
    /// may repeat (see the module docs), and each occurrence counts towards the cost.
    ops: Vec<OpId>,
    /// Sum of the operator costs, saturating at [`u64::MAX`]; this is `h_FF`.
    cost: u64,
}

impl RelaxedPlan {
    /// The operators in application order.
    #[must_use]
    pub fn ops(&self) -> &[OpId] {
        &self.ops
    }

    /// Sum of the operator costs (`h_FF`).
    #[must_use]
    pub fn cost(&self) -> u64 {
        self.cost
    }
}

impl Relaxation {
    /// Extracts a relaxed plan from `state` from the `h_add` best supporters; `None` if the
    /// goal is unreachable under the relaxation (a dead end).
    ///
    /// Its cost lies between `h_max` and `h_add`. Ties between supporters follow
    /// [`Relaxation::explore`], so the plan is deterministic.
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    #[must_use]
    pub fn relaxed_plan(&self, state: &State) -> Option<RelaxedPlan> {
        let exploration = self.explore(state, Combine::Add);
        exploration.goal_cost()?;
        let rules = self.marked_rules(&exploration);

        Some(self.replay(state, &rules))
    }

    /// The best-supporter rules the goal depends on, in the order their heads were closed.
    fn marked_rules(&self, exploration: &Exploration) -> Vec<usize> {
        let mut marked = vec![false; self.rules.len()];
        let mut seen = vec![false; self.num_facts];
        let mut stack = self.goal.clone();
        while let Some(fact) = stack.pop() {
            if core::mem::replace(&mut seen[fact], true) {
                continue;
            }

            if let Some(r) = exploration.supporter(fact) {
                marked[r] = true;
                stack.extend(&self.rules[r].conditions);
            }
        }

        exploration
            .order()
            .iter()
            .filter_map(|&fact| exploration.supporter(fact))
            .filter(|&r| marked[r])
            .collect()
    }

    /// Replays the marked operator `rules` from `state` under the relaxation, appending an
    /// operator whenever its rule's head is not true yet.
    fn replay(&self, state: &State, rules: &[usize]) -> RelaxedPlan {
        let mut reached = self.relaxed_state(state);
        let mut ops = Vec::new();
        let mut cost = 0_u64;
        for &r in rules {
            let rule = &self.rules[r];
            let Some(op) = rule.op else {
                continue;
            };
            if reached[rule.head] {
                continue;
            }

            self.apply_relaxed(op, &mut reached);
            debug_assert!(
                reached[rule.head],
                "a marked rule fires once its conditions hold"
            );
            ops.push(op);
            cost = cost.saturating_add(rule.cost);
        }

        RelaxedPlan { ops, cost }
    }

    /// The facts of `state` plus the free facts, closed under the axioms.
    fn relaxed_state(&self, state: &State) -> Vec<bool> {
        let mut reached = vec![false; self.num_facts];
        for fact in self.state_facts(state).chain(self.free.iter().copied()) {
            reached[fact] = true;
        }

        self.close_axioms(&mut reached);
        reached
    }

    /// Adds every effect of `op` whose conditions hold in `reached`, then closes under the
    /// axioms. Conditions are evaluated before any effect is added.
    fn apply_relaxed(&self, op: OpId, reached: &mut [bool]) {
        let fired: Vec<usize> = self.rules[self.op_rules[op].clone()]
            .iter()
            .filter(|rule| rule.conditions.iter().all(|&c| reached[c]))
            .map(|rule| rule.head)
            .collect();
        for head in fired {
            reached[head] = true;
        }

        self.close_axioms(reached);
    }

    /// Fires axioms whose conditions hold until none adds a fact.
    ///
    /// shortcut: rescans all axioms per pass, `O(axioms²)` worst case per call; switch to
    /// condition counters once axiom-heavy tasks matter.
    fn close_axioms(&self, reached: &mut [bool]) {
        let mut changed = true;
        while changed {
            changed = false;
            for rule in self.axiom_rules() {
                if !reached[rule.head] && rule.conditions.iter().all(|&c| reached[c]) {
                    reached[rule.head] = true;
                    changed = true;
                }
            }
        }
    }
}
