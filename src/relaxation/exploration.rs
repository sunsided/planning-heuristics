//! The worklist that assigns every fact its `h_max` or `h_add` cost.
//!
//! Knuth's generalization of Dijkstra's algorithm to superior functions (Knuth 1977): facts
//! are closed in order of increasing cost; when the last condition of a rule closes, the
//! rule offers its head the combined condition cost plus its own cost. Both `max` and `+`
//! are monotone and never below their arguments, so a closed fact never improves.
//!
//! ```mermaid
//! graph LR
//!     seed["seed: state facts, free facts, rules without conditions"] --> queue
//!     queue["queue: buckets by cost, heap for large costs"] -->|pop lowest| close["close fact"]
//!     close -->|"decrement counters of watching rules"| rule["rule with all conditions closed"]
//!     rule -->|"offer head: combine(conditions) + rule cost"| queue
//! ```

use alloc::vec::Vec;

use sas_plus::State;

use super::queue::Queue;
use super::workspace::{FactSlot, NO_RULE};
use super::{Relaxation, Workspace};

/// How a rule combines the costs of its conditions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Combine {
    /// The most expensive condition (`h_max`): admissible.
    Max,
    /// The sum of all conditions (`h_add`): not admissible, usually more informed.
    Add,
}

impl Combine {
    /// Combines an accumulated condition cost with the cost of one more condition.
    /// Sums saturate at [`u64::MAX`] instead of overflowing.
    fn apply(self, acc: u64, cost: u64) -> u64 {
        match self {
            Self::Max => acc.max(cost),
            Self::Add => acc.saturating_add(cost),
        }
    }
}

/// The result of [`Relaxation::explore`]: the cost of every fact from one state.
#[derive(Clone, Debug)]
pub struct Exploration {
    /// Relaxed cost of each fact; `None` if the fact is unreachable even without deletes.
    cost: Vec<Option<u64>>,
    /// The combined cost of the goal facts; `None` if one is unreachable.
    goal_cost: Option<u64>,
}

impl Exploration {
    /// The combined cost of the goal facts (their maximum or sum); `None` if a goal fact is
    /// unreachable under the relaxation, which proves the state a dead end.
    #[must_use]
    pub fn goal_cost(&self) -> Option<u64> {
        self.goal_cost
    }

    /// The relaxed cost of the fact with id `fact` (see [`Relaxation::fact_id`]); `None` if it
    /// is unreachable.
    ///
    /// # Panics
    ///
    /// Panics if `fact` is not below [`Relaxation::num_facts`].
    #[must_use]
    pub fn cost(&self, fact: usize) -> Option<u64> {
        self.cost[fact]
    }
}

impl Relaxation {
    /// Computes the relaxed cost of every fact from `state`, combining rule conditions with
    /// `combine`.
    ///
    /// `state` must have its derived variables evaluated (as returned by
    /// [`sas_plus::Task::initial_state`] or [`sas_plus::Task::apply`]).
    ///
    /// Runs in `O(facts + rule conditions)` plus the queue: `O(1)` amortized per fact for costs
    /// below 1024, `O(log facts)` above. Allocates its working memory. Ties are broken in a
    /// fixed queue order and a fact keeps the first rule that offered it its final cost, so
    /// results are deterministic. The heuristics ([`crate::HMax`], [`crate::HAdd`],
    /// [`crate::HFf`]) reuse their memory and stop once the goal is settled; use them when
    /// only the goal cost is needed.
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    #[must_use]
    pub fn explore(&self, state: &State, combine: Combine) -> Exploration {
        let mut workspace = Workspace::default();
        let goal_cost = self.run(state, combine, &mut workspace, false);

        Exploration {
            cost: workspace
                .facts
                .iter()
                .map(|slot| slot.cost(workspace.epoch))
                .collect(),
            goal_cost,
        }
    }

    /// The combined goal cost from `state`, stopping as soon as every goal fact is closed.
    /// Leaves the supporters and closing order of the closed facts in `workspace`.
    pub(crate) fn goal_cost_in(
        &self,
        state: &State,
        combine: Combine,
        workspace: &mut Workspace,
    ) -> Option<u64> {
        self.run(state, combine, workspace, true)
    }

    /// Runs the worklist from `state` in `workspace`; with `stop_at_goal`, only until the last
    /// goal fact is closed. Returns the combined goal cost.
    fn run(
        &self,
        state: &State,
        combine: Combine,
        workspace: &mut Workspace,
        stop_at_goal: bool,
    ) -> Option<u64> {
        workspace.reset(self);
        match combine {
            Combine::Max => Run::<false>::new(self, workspace).complete(state, stop_at_goal),
            Combine::Add => Run::<true>::new(self, workspace).complete(state, stop_at_goal),
        }
    }
}

/// One [`Relaxation::run`] over the buffers of a [`Workspace`]; `ADD` selects
/// [`Combine::Add`] over [`Combine::Max`] at compile time, keeping the inner loop branch-free.
struct Run<'a, const ADD: bool> {
    /// The relaxation being explored.
    relaxation: &'a Relaxation,
    /// The working memory, reset for this run.
    ws: &'a mut Workspace,
    /// Number of goal facts not closed yet.
    open_goals: usize,
}

impl<'a, const ADD: bool> Run<'a, ADD> {
    /// A run over the freshly reset `ws`.
    fn new(relaxation: &'a Relaxation, ws: &'a mut Workspace) -> Self {
        Self {
            relaxation,
            ws,
            open_goals: relaxation.goal.len(),
        }
    }

    /// Seeds and drains; returns the combined goal cost.
    fn complete(mut self, state: &State, stop_at_goal: bool) -> Option<u64> {
        self.seed(state);
        self.drain(stop_at_goal);

        self.goal_cost()
    }

    /// The [`Combine`] this run uses.
    const COMBINE: Combine = if ADD { Combine::Add } else { Combine::Max };

    /// Closes the facts of `state` and the free facts at cost 0, lets them trigger their
    /// watchers, then lets every rule without conditions offer its head.
    ///
    /// Closing the cost-0 start facts directly is what popping them first would do, minus the
    /// queue traffic; on small tasks that traffic dominates an evaluation.
    fn seed(&mut self, state: &State) {
        let relaxation = self.relaxation;
        for fact in relaxation.state_facts(state) {
            self.close_at_zero(fact);
        }

        for &fact in &relaxation.free {
            self.close_at_zero(fact);
        }

        let (facts, epoch) = (&self.ws.facts, self.ws.epoch);
        self.open_goals = relaxation
            .goal
            .iter()
            .filter(|&&g| !facts[g].is_closed(epoch))
            .count();

        // All start facts are closed before any rule fires, so none gets a supporter.
        for i in 0..self.ws.order.len() {
            self.trigger_watchers(self.ws.order[i] as usize, 0);
        }

        // Rules without conditions are never triggered by a closing fact.
        for &r in &relaxation.unconditioned {
            let rule = self.ws.rules[r as usize];
            self.offer(rule.head, u64::from(rule.cost), r);
        }
    }

    /// Closes the start fact `fact` with cost 0 and no supporter, unless it already is.
    #[inline]
    fn close_at_zero(&mut self, fact: u32) {
        let epoch = self.ws.epoch;
        let slot = &mut self.ws.facts[fact as usize];
        if slot.is_closed(epoch) {
            return;
        }

        *slot = FactSlot::open(0, NO_RULE, epoch);
        slot.close(epoch);
        self.ws.order.push(fact);
    }

    /// Closes facts in cost order until the queue is empty or, with `stop_at_goal`, until
    /// every goal fact is closed.
    fn drain(&mut self, stop_at_goal: bool) {
        while !(stop_at_goal && self.open_goals == 0) {
            let Some((cost, fact)) = self.ws.queue.pop() else {
                break;
            };
            let epoch = self.ws.epoch;
            let slot = &mut self.ws.facts[fact as usize];
            if slot.is_closed(epoch) {
                continue;
            }

            slot.close(epoch);
            self.ws.order.push(fact);
            if self.relaxation.is_goal[fact as usize] {
                self.open_goals -= 1;
            }

            self.trigger_watchers(fact as usize, cost);
        }
    }

    /// Records the closed `fact` with `cost` in every rule that waits for it, and fires the
    /// rules whose last condition this was.
    #[inline]
    fn trigger_watchers(&mut self, fact: usize, cost: u64) {
        let Workspace {
            epoch,
            facts,
            rules,
            queue,
            ..
        } = &mut *self.ws;
        for &r in self.relaxation.watchers(fact) {
            let slot = &mut rules[r as usize];
            slot.touch(*epoch);
            slot.accumulated = Self::COMBINE.apply(slot.accumulated, cost);
            slot.unsatisfied -= 1;
            if slot.unsatisfied == 0 {
                let total = slot.accumulated.saturating_add(u64::from(slot.cost));
                offer(facts, queue, *epoch, slot.head, total, r);
            }
        }
    }

    /// Lowers the cost of `fact` to `cost` via rule `supporter` ([`NO_RULE`] for none) if
    /// that is strictly cheaper; see [`offer`].
    fn offer(&mut self, fact: u32, cost: u64, supporter: u32) {
        offer(
            &mut self.ws.facts,
            &mut self.ws.queue,
            self.ws.epoch,
            fact,
            cost,
            supporter,
        );
    }

    /// Combines the goal fact costs; `None` if one is unreachable.
    fn goal_cost(&self) -> Option<u64> {
        let combine = Self::COMBINE;
        self.relaxation.goal.iter().try_fold(0, |acc, &g| {
            self.ws.facts[g]
                .cost(self.ws.epoch)
                .map(|c| combine.apply(acc, c))
        })
    }
}

/// Lowers the cost of `fact` in `facts` to `cost` via rule `supporter` ([`NO_RULE`] for none)
/// if that is strictly cheaper, and queues it. A free function so the caller can keep other
/// workspace fields borrowed.
#[inline]
fn offer(
    facts: &mut [FactSlot],
    queue: &mut Queue,
    epoch: u32,
    fact: u32,
    cost: u64,
    supporter: u32,
) {
    let slot = &mut facts[fact as usize];
    if !slot.is_reached(epoch) || cost < slot.cost {
        *slot = FactSlot::open(cost, supporter, epoch);
        queue.push(cost, fact);
    }
}
