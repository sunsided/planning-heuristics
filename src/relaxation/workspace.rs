//! Working memory for evaluating states, reused across evaluations.

use alloc::vec::Vec;

use sas_plus::OpId;

use super::Relaxation;
use super::queue::Queue;

/// Marks "no supporting rule" in [`FactSlot::supporter`].
pub(super) const NO_RULE: u32 = u32::MAX;

/// Epochs run from 1 up to this; then the workspace is wiped and starts over at 1. Kept below
/// `2³¹` so that `epoch << 1 | 1` fits a `u32` stamp.
pub(super) const MAX_EPOCH: u32 = (1 << 31) - 1;

/// The worklist state of one fact, valid only in the epoch of its stamp.
///
/// Stamps make resetting the workspace `O(1)`: a slot whose stamp is from an older epoch reads
/// as unreached, so an evaluation that stops early never pays for the facts it did not touch.
#[derive(Clone, Copy, Debug)]
pub(super) struct FactSlot {
    /// Best cost offered so far; final once closed. Meaningless unless reached.
    pub cost: u64,
    /// The rule behind [`FactSlot::cost`]; [`NO_RULE`] for facts that need no rule.
    pub supporter: u32,
    /// `epoch << 1` once reached in `epoch`, `epoch << 1 | 1` once also closed. Tracked apart
    /// from the cost, so a saturated sum is never mistaken for an unreachable fact.
    stamp: u32,
}

impl FactSlot {
    /// A fact nothing was ever offered to.
    pub const UNREACHED: Self = Self {
        cost: 0,
        supporter: NO_RULE,
        stamp: 0,
    };

    /// A fact reached in `epoch` with `cost` via `supporter`, not closed yet.
    pub fn open(cost: u64, supporter: u32, epoch: u32) -> Self {
        Self {
            cost,
            supporter,
            stamp: epoch << 1,
        }
    }

    /// Whether a cost was offered in `epoch`.
    pub fn is_reached(self, epoch: u32) -> bool {
        self.stamp >> 1 == epoch
    }

    /// Whether the fact was closed with its final cost in `epoch`.
    pub fn is_closed(self, epoch: u32) -> bool {
        self.stamp == (epoch << 1 | 1)
    }

    /// Marks the fact closed; it must have been reached in `epoch`.
    pub fn close(&mut self, epoch: u32) {
        debug_assert!(self.is_reached(epoch), "only reached facts close");
        self.stamp = epoch << 1 | 1;
    }

    /// The cost if the fact was reached in `epoch`.
    pub fn cost(self, epoch: u32) -> Option<u64> {
        self.is_reached(epoch).then_some(self.cost)
    }

    /// The supporting rule if the fact was reached in `epoch` through one.
    pub fn supporter(self, epoch: u32) -> Option<usize> {
        (self.is_reached(epoch) && self.supporter != NO_RULE).then_some(self.supporter as usize)
    }
}

/// The worklist state of one rule, with the rule's head and cost packed alongside so that
/// firing a rule touches one slot. The counters are valid only in the epoch of the stamp.
#[derive(Clone, Copy, Debug)]
pub(super) struct RuleSlot {
    /// Number of conditions of the rule.
    pub conditions: u32,
    /// Number of conditions not closed yet in [`RuleSlot::stamp`]'s epoch.
    pub unsatisfied: u32,
    /// The fact the rule makes true.
    pub head: u32,
    /// Cost of applying the rule; operator costs are `u32` in sas-plus, axioms cost `0`.
    pub cost: u32,
    /// The epoch the counters belong to.
    pub stamp: u32,
    /// Combined cost of the closed conditions.
    pub accumulated: u64,
}

impl RuleSlot {
    /// A rule with `conditions` conditions, `head` and `cost`, never touched.
    pub fn new(conditions: u32, head: u32, cost: u32) -> Self {
        Self {
            conditions,
            unsatisfied: conditions,
            head,
            cost,
            stamp: 0,
            accumulated: 0,
        }
    }

    /// Restarts the counters if they are from an older epoch than `epoch`.
    #[inline]
    pub fn touch(&mut self, epoch: u32) {
        if self.stamp != epoch {
            self.stamp = epoch;
            self.unsatisfied = self.conditions;
            self.accumulated = 0;
        }
    }
}

/// A set of indices below a fixed length, cleared in `O(1)` by moving to a new epoch.
#[derive(Clone, Debug, Default)]
pub(super) struct Marks {
    /// `stamps[i] == epoch` means `i` is in the set.
    stamps: Vec<u32>,
    /// The epoch of the current contents; `0` before first use.
    epoch: u32,
}

impl Marks {
    /// Empties the set and sizes it for indices below `len`, for the evaluation in `epoch`.
    /// Wipes the stamps when the size changes or the epoch did not advance (after the
    /// workspace's epoch wrapped around).
    pub fn reset(&mut self, len: usize, epoch: u32) {
        if self.stamps.len() != len || epoch <= self.epoch {
            refill(&mut self.stamps, len, 0);
        }

        self.epoch = epoch;
    }

    /// Whether `i` is in the set.
    #[inline]
    pub fn contains(&self, i: usize) -> bool {
        self.stamps[i] == self.epoch
    }

    /// Adds `i`; returns whether it was new.
    #[inline]
    pub fn insert(&mut self, i: usize) -> bool {
        core::mem::replace(&mut self.stamps[i], self.epoch) != self.epoch
    }
}

/// The buffers of one worklist run and one relaxed plan extraction.
///
/// Sized for one [`Relaxation`]; [`Workspace::reset`] refills the buffers in place, so
/// evaluating a state allocates nothing once the buffers have grown.
#[derive(Clone, Debug, Default)]
pub(crate) struct Workspace {
    /// The current epoch: slots stamped with another one are stale. `0` before first use.
    pub(super) epoch: u32,
    /// The worklist state of each fact.
    pub(super) facts: Vec<FactSlot>,
    /// The worklist state of each rule.
    pub(super) rules: Vec<RuleSlot>,
    /// Open facts by cost. May hold stale entries for facts closed since.
    pub(super) queue: Queue,
    /// Closed facts in closing order.
    pub(super) order: Vec<u32>,
    /// Number of start facts (state facts and free facts) leading [`Workspace::order`]; they
    /// are closed at cost 0 without a supporter before any rule fires.
    pub(super) start_facts: usize,
    /// The best-supporter rules the goal depends on.
    pub(super) marked: Marks,
    /// The facts visited while marking supporters.
    pub(super) seen: Marks,
    /// Facts still to visit while marking supporters; during plan replay, the reached facts
    /// whose axiom watchers are still to be checked.
    pub(super) stack: Vec<usize>,
    /// Facts true in the relaxed state during plan replay.
    pub(super) reached: Marks,
    /// Heads of the effects that fire in one relaxed operator application.
    pub(super) fired: Vec<usize>,
    /// The relaxed plan of the last extraction, in application order.
    pub(crate) plan: Vec<OpId>,
}

impl Workspace {
    /// Starts a new evaluation for `relaxation` in `O(1)` by advancing the epoch; the slots
    /// are copied from the templates only on first use and once every `2³¹` evaluations.
    ///
    /// A workspace serves one relaxation: reusing it for another one with the same number of
    /// facts and rules would keep the other's rule heads.
    pub(super) fn reset(&mut self, relaxation: &Relaxation) {
        let sized = self.facts.len() == relaxation.num_facts
            && self.rules.len() == relaxation.rule_init.len();
        if !sized || self.epoch == MAX_EPOCH {
            self.facts.clear();
            self.facts.extend_from_slice(&relaxation.fact_init);
            self.rules.clear();
            self.rules.extend_from_slice(&relaxation.rule_init);
            self.epoch = 0;
        }

        self.epoch += 1;
        self.queue.clear();
        self.order.clear();
    }
}

/// Makes `buffer` hold `len` copies of `value`, reusing its allocation.
pub(super) fn refill<T: Clone>(buffer: &mut Vec<T>, len: usize, value: T) {
    buffer.clear();
    buffer.resize(len, value);
}

#[cfg(test)]
mod tests {
    //! Epoch stamping across the wrap-around.

    use alloc::vec::Vec;

    use sas_plus::random::{RandomTaskParams, random_task};

    use super::{MAX_EPOCH, Workspace};
    use crate::{Combine, Relaxation};

    /// Values and relaxed plans stay equal to fresh evaluations while the epoch wraps.
    #[test]
    fn evaluations_survive_the_epoch_wrap() {
        let params = RandomTaskParams {
            max_conditions: 2,
            num_derived: 2,
            ..RandomTaskParams::default()
        };
        for seed in 0..16 {
            let task = random_task(seed, params);
            let relaxation = Relaxation::new(&task);
            let initial = task.initial_state();
            let mut states: Vec<_> = task
                .applicable_operators(&initial)
                .map(|op| task.apply(op, &initial))
                .collect();
            states.push(initial);

            let mut reused = Workspace::default();
            reused.reset(&relaxation);
            reused.epoch = MAX_EPOCH - 2;
            for _ in 0..3 {
                for state in &states {
                    for combine in [Combine::Max, Combine::Add] {
                        let expected = relaxation.explore(state, combine).goal_cost();
                        assert_eq!(
                            relaxation.goal_cost_in(state, combine, &mut reused),
                            expected
                        );
                    }

                    let cost = relaxation.relaxed_plan_in(state, &mut reused);
                    let fresh = relaxation.relaxed_plan(state);
                    assert_eq!(cost, fresh.as_ref().map(crate::RelaxedPlan::cost));
                    if let Some(fresh) = fresh {
                        assert_eq!(reused.plan, fresh.ops());
                    }
                }
            }

            assert!(reused.epoch < MAX_EPOCH - 2, "the epoch wrapped");
        }
    }
}
