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

use alloc::vec::Vec;
use core::mem;

use sas_plus::{OpId, State};

use super::workspace::refill;
use super::{Combine, Relaxation, Workspace};

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
    /// A plan of `ops` with total cost `cost`.
    pub(crate) fn new(ops: Vec<OpId>, cost: u64) -> Self {
        Self { ops, cost }
    }

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
    /// [`Relaxation::explore`], so the plan is deterministic. This allocates its working
    /// memory; [`crate::HFf::relaxed_plan`] reuses it across calls.
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    #[must_use]
    pub fn relaxed_plan(&self, state: &State) -> Option<RelaxedPlan> {
        let mut workspace = Workspace::default();
        let cost = self.relaxed_plan_in(state, &mut workspace)?;

        Some(RelaxedPlan::new(workspace.plan, cost))
    }

    /// Extracts a relaxed plan into `workspace.plan` and returns its cost; `None` for a dead
    /// end.
    pub(crate) fn relaxed_plan_in(&self, state: &State, workspace: &mut Workspace) -> Option<u64> {
        self.goal_cost_in(state, Combine::Add, workspace)?;
        self.mark_supporters(workspace);

        Some(self.replay(state, workspace))
    }

    /// Marks the best-supporter rules the goal depends on in `workspace.marked`.
    fn mark_supporters(&self, ws: &mut Workspace) {
        refill(&mut ws.marked, self.rules.len(), false);
        refill(&mut ws.seen, self.num_facts, false);
        ws.stack.clear();
        ws.stack.extend_from_slice(&self.goal);
        while let Some(fact) = ws.stack.pop() {
            if mem::replace(&mut ws.seen[fact], true) {
                continue;
            }

            if let Some(r) = ws.supporter[fact] {
                ws.marked[r] = true;
                ws.stack.extend_from_slice(&self.rules[r].conditions);
            }
        }
    }

    /// Replays the marked operator rules from `state` under the relaxation, in the order their
    /// heads were closed, appending an operator to `workspace.plan` whenever its rule's head is
    /// not true yet. Returns the plan cost.
    fn replay(&self, state: &State, ws: &mut Workspace) -> u64 {
        self.relaxed_state(state, ws);
        ws.plan.clear();
        let mut cost = 0_u64;
        for i in 0..ws.order.len() {
            let Some(r) = ws.supporter[ws.order[i]].filter(|&r| ws.marked[r]) else {
                continue;
            };
            let rule = &self.rules[r];
            let Some(op) = rule.op else {
                continue;
            };
            if ws.reached[rule.head] {
                continue;
            }

            self.apply_relaxed(op, ws);
            debug_assert!(
                ws.reached[rule.head],
                "a marked rule fires once its conditions hold"
            );
            ws.plan.push(op);
            cost = cost.saturating_add(rule.cost);
        }

        cost
    }

    /// Sets `workspace.reached` to the facts of `state` plus the free facts, closed under the
    /// axioms.
    fn relaxed_state(&self, state: &State, ws: &mut Workspace) {
        refill(&mut ws.reached, self.num_facts, false);
        for fact in self.state_facts(state).chain(self.free.iter().copied()) {
            ws.reached[fact] = true;
        }

        self.close_axioms(&mut ws.reached);
    }

    /// Adds every effect of `op` whose conditions hold in `workspace.reached`, then closes
    /// under the axioms. Conditions are evaluated before any effect is added.
    fn apply_relaxed(&self, op: OpId, ws: &mut Workspace) {
        ws.fired.clear();
        ws.fired.extend(
            self.rules[self.op_rules[op].clone()]
                .iter()
                .filter(|rule| rule.conditions.iter().all(|&c| ws.reached[c]))
                .map(|rule| rule.head),
        );
        for &head in &ws.fired {
            ws.reached[head] = true;
        }

        self.close_axioms(&mut ws.reached);
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
