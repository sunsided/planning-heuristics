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

use sas_plus::{OpId, State};

use super::{Combine, Relaxation, Rule, Workspace};

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
        ws.marked.reset(self.rules.len(), ws.epoch);
        ws.seen.reset(self.num_facts, ws.epoch);
        ws.stack.clear();
        ws.stack.extend_from_slice(&self.goal);
        while let Some(fact) = ws.stack.pop() {
            if !ws.seen.insert(fact) {
                continue;
            }

            if let Some(r) = ws.facts[fact].supporter(ws.epoch) {
                ws.marked.insert(r);
                ws.stack.extend_from_slice(&self.rules[r].conditions);
            }
        }
    }

    /// Replays the marked operator rules from `state` under the relaxation, in the order their
    /// heads were closed, appending an operator to `workspace.plan` whenever its rule's head is
    /// not true yet. Returns the plan cost.
    ///
    /// Starts after the start facts in `workspace.order`: none has a supporter, so none can
    /// add an operator.
    fn replay(&self, state: &State, ws: &mut Workspace) -> u64 {
        self.relaxed_state(state, ws);
        let start_facts = ws.start_facts;
        debug_assert!(
            ws.order[..start_facts]
                .iter()
                .all(|&f| ws.facts[f as usize].supporter(ws.epoch).is_none()),
            "the exploration closes the start facts first, without supporters"
        );
        ws.plan.clear();
        let mut cost = 0_u64;
        // Taken out so the loop can borrow it while the replay mutates the workspace.
        let order = core::mem::take(&mut ws.order);
        for &fact in &order[start_facts..] {
            let fact = fact as usize;
            let Some(r) = ws.facts[fact]
                .supporter(ws.epoch)
                .filter(|&r| ws.marked.contains(r))
            else {
                continue;
            };
            let rule = &self.rules[r];
            let Some(op) = rule.op else {
                continue;
            };
            if ws.reached.contains(rule.head) {
                continue;
            }

            self.apply_relaxed(op, ws);
            debug_assert!(
                ws.reached.contains(rule.head),
                "a marked rule fires once its conditions hold"
            );
            ws.plan.push(op);
            cost = cost.saturating_add(rule.cost);
        }

        ws.order = order;

        cost
    }

    /// Sets `workspace.reached` to the facts of `state` plus the free facts, closed under the
    /// axioms.
    ///
    /// One pass over the axioms fires those the start facts enable; [`Relaxation::close_axioms`]
    /// then follows only the facts this adds, so axiom chains cost no further passes.
    fn relaxed_state(&self, state: &State, ws: &mut Workspace) {
        ws.reached.reset(self.num_facts, ws.epoch);
        for fact in self.start_facts(state) {
            ws.reached.insert(fact as usize);
        }

        if !self.has_axioms() {
            return;
        }

        ws.stack.clear();
        for rule in &self.rules[self.first_axiom..] {
            fire_if_enabled(rule, ws);
        }

        self.close_axioms(ws);
    }

    /// Whether the task has axioms; without them the replay skips the axiom closure.
    fn has_axioms(&self) -> bool {
        self.first_axiom < self.rules.len()
    }

    /// Adds every effect of `op` whose conditions hold in `workspace.reached`, then closes
    /// under the axioms. Conditions are evaluated before any effect is added.
    fn apply_relaxed(&self, op: OpId, ws: &mut Workspace) {
        ws.fired.clear();
        ws.fired.extend(
            self.rules[self.op_rules[op].clone()]
                .iter()
                .filter(|rule| rule.conditions.iter().all(|&c| ws.reached.contains(c)))
                .map(|rule| rule.head),
        );
        for &head in &ws.fired {
            ws.reached.insert(head);
        }

        if self.has_axioms() {
            ws.stack.extend_from_slice(&ws.fired);
            self.close_axioms(ws);
        }
    }

    /// Re-evaluates the axioms watching each reached fact on `workspace.stack`, firing
    /// those whose conditions now all hold, until no axiom adds a fact.
    ///
    /// An axiom can only become enabled when its last missing condition is reached, so
    /// checking the watchers of each new fact finds every axiom that fires: a replay step
    /// costs the axioms that depend on the facts it added, not a pass over all axioms.
    fn close_axioms(&self, ws: &mut Workspace) {
        while let Some(fact) = ws.stack.pop() {
            for &a in self.axiom_watchers(fact) {
                fire_if_enabled(&self.rules[self.first_axiom + a as usize], ws);
            }
        }
    }
}

/// Fires the axiom `rule` if its head is not reached yet and all its conditions are: adds
/// the head to `workspace.reached` and queues it on `workspace.stack`.
#[inline]
fn fire_if_enabled(rule: &Rule, ws: &mut Workspace) {
    if !ws.reached.contains(rule.head) && rule.conditions.iter().all(|&c| ws.reached.contains(c)) {
        ws.reached.insert(rule.head);
        ws.stack.push(rule.head);
    }
}
