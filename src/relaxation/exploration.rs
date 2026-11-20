//! The worklist that assigns every fact its `h_max` or `h_add` cost.
//!
//! Knuth's generalization of Dijkstra's algorithm to superior functions (Knuth 1977): facts
//! are closed in order of increasing cost; when the last condition of a rule closes, the
//! rule offers its head the combined condition cost plus its own cost. Both `max` and `+`
//! are monotone and never below their arguments, so a closed fact never improves.
//!
//! ```mermaid
//! graph LR
//!     seed["seed: state facts, free facts, rules without conditions"] --> heap
//!     heap["heap ordered by (cost, fact id)"] -->|pop lowest| close["close fact"]
//!     close -->|"decrement counters of watching rules"| rule["rule with all conditions closed"]
//!     rule -->|"offer head: combine(conditions) + rule cost"| heap
//! ```

use alloc::vec::Vec;
use core::cmp::Reverse;
use core::mem;

use sas_plus::State;

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
    /// Runs in `O((facts + rule conditions) · log facts)` and allocates its working memory.
    /// Ties are broken towards the lower fact id, and a fact keeps the first rule that offered
    /// it its final cost, so results are deterministic. The heuristics ([`crate::HMax`],
    /// [`crate::HAdd`], [`crate::HFf`]) reuse their memory and stop once the goal is
    /// settled; use them when only the goal cost is needed.
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    #[must_use]
    pub fn explore(&self, state: &State, combine: Combine) -> Exploration {
        let mut workspace = Workspace::default();
        let goal_cost = self.run(state, combine, &mut workspace, false);

        Exploration {
            cost: mem::take(&mut workspace.cost),
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
        let mut run = Run {
            relaxation: self,
            combine,
            ws: workspace,
        };
        run.seed(state);
        run.drain(stop_at_goal);

        run.goal_cost()
    }
}

/// One [`Relaxation::run`] over the buffers of a [`Workspace`].
struct Run<'a> {
    /// The relaxation being explored.
    relaxation: &'a Relaxation,
    /// How rules combine their condition costs.
    combine: Combine,
    /// The working memory, reset for this run.
    ws: &'a mut Workspace,
}

impl Run<'_> {
    /// Offers cost 0 to the facts of `state` and the free facts, then lets every rule
    /// without conditions offer its head.
    fn seed(&mut self, state: &State) {
        let relaxation = self.relaxation;
        for fact in relaxation
            .state_facts(state)
            .chain(relaxation.free.iter().copied())
        {
            self.offer(fact, 0, None);
        }

        // Rules without conditions are never triggered by a closing fact.
        for (r, rule) in relaxation.rules.iter().enumerate() {
            if rule.conditions.is_empty() {
                self.offer(rule.head, rule.cost, Some(r));
            }
        }
    }

    /// Closes facts in cost order until the heap is empty or, with `stop_at_goal`, until
    /// every goal fact is closed.
    fn drain(&mut self, stop_at_goal: bool) {
        let mut open_goals = self.relaxation.goal.len();
        while !(stop_at_goal && open_goals == 0) {
            let Some(Reverse((cost, fact))) = self.ws.heap.pop() else {
                break;
            };
            if self.ws.closed[fact] {
                continue;
            }

            self.ws.closed[fact] = true;
            self.ws.order.push(fact);
            if self.relaxation.is_goal[fact] {
                open_goals -= 1;
            }

            self.trigger_watchers(fact, cost);
        }
    }

    /// Records the closed `fact` with `cost` in every rule that waits for it, and fires the
    /// rules whose last condition this was.
    fn trigger_watchers(&mut self, fact: usize, cost: u64) {
        let relaxation = self.relaxation;
        for &r in relaxation.watchers(fact) {
            self.ws.accumulated[r] = self.combine.apply(self.ws.accumulated[r], cost);
            self.ws.unsatisfied[r] -= 1;
            if self.ws.unsatisfied[r] == 0 {
                let rule = &relaxation.rules[r];
                let total = self.ws.accumulated[r].saturating_add(rule.cost);
                self.offer(rule.head, total, Some(r));
            }
        }
    }

    /// Lowers the cost of `fact` to `cost` via `supporter` if that is strictly cheaper.
    fn offer(&mut self, fact: usize, cost: u64, supporter: Option<usize>) {
        if self.ws.cost[fact].is_none_or(|old| cost < old) {
            self.ws.cost[fact] = Some(cost);
            self.ws.supporter[fact] = supporter;
            self.ws.heap.push(Reverse((cost, fact)));
        }
    }

    /// Combines the goal fact costs; `None` if one is unreachable.
    fn goal_cost(&self) -> Option<u64> {
        let combine = self.combine;
        self.relaxation
            .goal
            .iter()
            .try_fold(0, |acc, &g| self.ws.cost[g].map(|c| combine.apply(acc, c)))
    }
}
