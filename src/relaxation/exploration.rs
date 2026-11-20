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

use alloc::collections::BinaryHeap;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use sas_plus::State;

use super::Relaxation;

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
    /// The rule that first offered each fact its final cost; `None` for facts true in the
    /// state, free facts and unreachable facts. Ties keep the first offer.
    supporter: Vec<Option<usize>>,
    /// The reachable facts in the order they were closed. Every rule's conditions are
    /// closed before its head when the rule supports that head.
    order: Vec<usize>,
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

    /// The rule that achieves `fact` most cheaply; `None` for facts that need no rule.
    pub(super) fn supporter(&self, fact: usize) -> Option<usize> {
        self.supporter[fact]
    }

    /// The reachable facts in the order they were closed.
    pub(super) fn order(&self) -> &[usize] {
        &self.order
    }
}

impl Relaxation {
    /// Computes the relaxed cost of every fact from `state`, combining rule conditions with
    /// `combine`.
    ///
    /// `state` must have its derived variables evaluated (as returned by
    /// [`sas_plus::Task::initial_state`] or [`sas_plus::Task::apply`]).
    ///
    /// Runs in `O((facts + rule conditions) · log facts)`. Ties are broken towards the lower
    /// fact id, and a fact keeps the first rule that offered it its final cost, so results are
    /// deterministic.
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    #[must_use]
    pub fn explore(&self, state: &State, combine: Combine) -> Exploration {
        let mut run = Run::new(self, combine);
        run.seed(state);
        run.drain();

        run.finish()
    }
}

/// The working state of one [`Relaxation::explore`] call.
struct Run<'a> {
    /// The relaxation being explored.
    relaxation: &'a Relaxation,
    /// How rules combine their condition costs.
    combine: Combine,
    /// Best cost offered to each fact so far; final once the fact is closed.
    cost: Vec<Option<u64>>,
    /// The rule behind each entry of [`Run::cost`].
    supporter: Vec<Option<usize>>,
    /// Whether each fact is closed.
    closed: Vec<bool>,
    /// Number of conditions of each rule that are not closed yet.
    unsatisfied: Vec<usize>,
    /// Combined cost of the closed conditions of each rule.
    accumulated: Vec<u64>,
    /// Open facts by `(cost, fact id)`, lowest first. May hold stale entries.
    heap: BinaryHeap<Reverse<(u64, usize)>>,
    /// Closed facts in closing order.
    order: Vec<usize>,
}

impl<'a> Run<'a> {
    /// A run with every fact unreached and every rule waiting for all its conditions.
    fn new(relaxation: &'a Relaxation, combine: Combine) -> Self {
        let n = relaxation.num_facts;

        Self {
            relaxation,
            combine,
            cost: vec![None; n],
            supporter: vec![None; n],
            closed: vec![false; n],
            unsatisfied: relaxation
                .rules
                .iter()
                .map(|r| r.conditions.len())
                .collect(),
            accumulated: vec![0; relaxation.rules.len()],
            heap: BinaryHeap::new(),
            order: Vec::new(),
        }
    }

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

    /// Closes facts in cost order until the heap is empty.
    ///
    /// shortcut: explores every reachable fact, not only until the goal is closed; stop at
    /// the last goal fact once callers need speed more than full fact costs.
    fn drain(&mut self) {
        while let Some(Reverse((cost, fact))) = self.heap.pop() {
            if self.closed[fact] {
                continue;
            }

            self.closed[fact] = true;
            self.order.push(fact);
            self.trigger_watchers(fact, cost);
        }
    }

    /// Records the closed `fact` with `cost` in every rule that waits for it, and fires the
    /// rules whose last condition this was.
    fn trigger_watchers(&mut self, fact: usize, cost: u64) {
        let relaxation = self.relaxation;
        for &r in &relaxation.watchers[fact] {
            self.accumulated[r] = self.combine.apply(self.accumulated[r], cost);
            self.unsatisfied[r] -= 1;
            if self.unsatisfied[r] == 0 {
                let rule = &relaxation.rules[r];
                let total = self.accumulated[r].saturating_add(rule.cost);
                self.offer(rule.head, total, Some(r));
            }
        }
    }

    /// Lowers the cost of `fact` to `cost` via `supporter` if that is strictly cheaper.
    fn offer(&mut self, fact: usize, cost: u64, supporter: Option<usize>) {
        if self.cost[fact].is_none_or(|old| cost < old) {
            self.cost[fact] = Some(cost);
            self.supporter[fact] = supporter;
            self.heap.push(Reverse((cost, fact)));
        }
    }

    /// Combines the goal fact costs and packs the result.
    fn finish(self) -> Exploration {
        let combine = self.combine;
        let goal_cost = self
            .relaxation
            .goal
            .iter()
            .try_fold(0, |acc, &g| self.cost[g].map(|c| combine.apply(acc, c)));

        Exploration {
            cost: self.cost,
            supporter: self.supporter,
            order: self.order,
            goal_cost,
        }
    }
}
