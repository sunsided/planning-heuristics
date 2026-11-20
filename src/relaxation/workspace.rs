//! Working memory for evaluating states, reused across evaluations.

use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::Reverse;

use sas_plus::OpId;

use super::Relaxation;

/// The buffers of one worklist run and one relaxed plan extraction.
///
/// Sized for one [`Relaxation`]; [`Workspace::reset`] refills the buffers in place, so
/// evaluating a state allocates nothing once the buffers have grown.
#[derive(Clone, Debug, Default)]
pub(crate) struct Workspace {
    /// Best cost offered to each fact so far; final once the fact is closed.
    pub cost: Vec<Option<u64>>,
    /// The rule behind each entry of [`Workspace::cost`]; `None` for facts that need no rule.
    pub supporter: Vec<Option<usize>>,
    /// Whether each fact is closed.
    pub closed: Vec<bool>,
    /// Number of conditions of each rule that are not closed yet.
    pub unsatisfied: Vec<usize>,
    /// Combined cost of the closed conditions of each rule.
    pub accumulated: Vec<u64>,
    /// Open facts by `(cost, fact id)`, lowest first. May hold stale entries.
    pub heap: BinaryHeap<Reverse<(u64, usize)>>,
    /// Closed facts in closing order.
    pub order: Vec<usize>,
    /// Whether each rule is a best supporter the goal depends on.
    pub marked: Vec<bool>,
    /// Whether each fact was visited while marking supporters.
    pub seen: Vec<bool>,
    /// Facts still to visit while marking supporters.
    pub stack: Vec<usize>,
    /// Facts true in the relaxed state during plan replay.
    pub reached: Vec<bool>,
    /// Heads of the effects that fire in one relaxed operator application.
    pub fired: Vec<usize>,
    /// The relaxed plan of the last extraction, in application order.
    pub plan: Vec<OpId>,
}

impl Workspace {
    /// Clears the worklist buffers and sizes them for `relaxation`.
    pub fn reset(&mut self, relaxation: &Relaxation) {
        let n = relaxation.num_facts;
        refill(&mut self.cost, n, None);
        refill(&mut self.supporter, n, None);
        refill(&mut self.closed, n, false);
        refill(&mut self.accumulated, relaxation.rules.len(), 0);
        self.unsatisfied.clear();
        self.unsatisfied
            .extend_from_slice(&relaxation.condition_counts);
        self.heap.clear();
        self.order.clear();
    }
}

/// Makes `buffer` hold `len` copies of `value`, reusing its allocation.
pub(super) fn refill<T: Clone>(buffer: &mut Vec<T>, len: usize, value: T) {
    buffer.clear();
    buffer.resize(len, value);
}
