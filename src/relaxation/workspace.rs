//! Working memory for evaluating states, reused across evaluations.

use alloc::vec::Vec;

use sas_plus::OpId;

use super::Relaxation;
use super::queue::Queue;

/// Marks "no supporting rule" in [`FactSlot::supporter`].
pub(super) const NO_RULE: u32 = u32::MAX;

/// The worklist state of one fact.
#[derive(Clone, Copy, Debug)]
pub(super) struct FactSlot {
    /// Best cost offered so far; final once [`FactSlot::closed`]. Meaningless unless
    /// [`FactSlot::reached`].
    pub cost: u64,
    /// The rule behind [`FactSlot::cost`]; [`NO_RULE`] for facts that need no rule.
    pub supporter: u32,
    /// Whether any cost was offered; tracked apart from the cost so a saturated sum is never
    /// mistaken for an unreachable fact.
    pub reached: bool,
    /// Whether the fact was popped with its final cost.
    pub closed: bool,
}

impl FactSlot {
    /// A fact nothing was offered to yet.
    pub const UNREACHED: Self = Self {
        cost: 0,
        supporter: NO_RULE,
        reached: false,
        closed: false,
    };

    /// The cost if the fact was reached.
    pub fn cost(self) -> Option<u64> {
        self.reached.then_some(self.cost)
    }

    /// The supporting rule, if any.
    pub fn supporter(self) -> Option<usize> {
        (self.supporter != NO_RULE).then_some(self.supporter as usize)
    }
}

/// The worklist state of one rule, with the rule's head and cost packed alongside so that
/// firing a rule touches one slot.
#[derive(Clone, Copy, Debug)]
pub(super) struct RuleSlot {
    /// Number of conditions not closed yet.
    pub unsatisfied: u32,
    /// The fact the rule makes true.
    pub head: u32,
    /// Cost of applying the rule; operator costs are `u32` in sas-plus, axioms cost `0`.
    pub cost: u32,
    /// Combined cost of the closed conditions.
    pub accumulated: u64,
}

/// The buffers of one worklist run and one relaxed plan extraction.
///
/// Sized for one [`Relaxation`]; [`Workspace::reset`] refills the buffers in place, so
/// evaluating a state allocates nothing once the buffers have grown.
#[derive(Clone, Debug, Default)]
pub(crate) struct Workspace {
    /// The worklist state of each fact.
    pub(super) facts: Vec<FactSlot>,
    /// The worklist state of each rule.
    pub(super) rules: Vec<RuleSlot>,
    /// Open facts by cost. May hold stale entries for facts closed since.
    pub(super) queue: Queue,
    /// Closed facts in closing order.
    pub(super) order: Vec<u32>,
    /// Whether each rule is a best supporter the goal depends on.
    pub(super) marked: Vec<bool>,
    /// Whether each fact was visited while marking supporters.
    pub(super) seen: Vec<bool>,
    /// Facts still to visit while marking supporters.
    pub(super) stack: Vec<usize>,
    /// Facts true in the relaxed state during plan replay.
    pub(super) reached: Vec<bool>,
    /// Heads of the effects that fire in one relaxed operator application.
    pub(super) fired: Vec<usize>,
    /// The relaxed plan of the last extraction, in application order.
    pub(crate) plan: Vec<OpId>,
}

impl Workspace {
    /// Clears the worklist buffers and sizes them for `relaxation`.
    pub(super) fn reset(&mut self, relaxation: &Relaxation) {
        // Copying from templates compiles to two memcpys; refilling slot by slot showed up
        // in profiles of small tasks.
        self.facts.clear();
        self.facts.extend_from_slice(&relaxation.fact_init);
        self.rules.clear();
        self.rules.extend_from_slice(&relaxation.rule_init);
        self.queue.clear();
        self.order.clear();
    }
}

/// Makes `buffer` hold `len` copies of `value`, reusing its allocation.
pub(super) fn refill<T: Clone>(buffer: &mut Vec<T>, len: usize, value: T) {
    buffer.clear();
    buffer.resize(len, value);
}
