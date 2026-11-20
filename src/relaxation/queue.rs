//! The worklist's priority queue: buckets for small costs, a binary heap above.
//!
//! Relaxed fact costs are integers that never decrease along the worklist (Knuth's
//! monotonicity), so a bucket queue (Dial 1969) indexed by cost pops in `O(1)` amortized:
//! the scan only moves forward. Costs from [`BUCKETS`] on go to a binary heap, so large or
//! sparse costs never allocate or scan a bucket per cost value.

use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::Reverse;

/// Costs below this get a bucket; larger ones go to the heap. Bounds the empty buckets one
/// evaluation can scan.
const BUCKETS: usize = 1024;

/// A monotone priority queue of fact ids by cost.
///
/// Every pushed cost must be at least the cost last popped. Within one cost, facts pop in
/// reverse push order (each bucket is a stack); deterministic, but not by fact id.
#[derive(Clone, Debug, Default)]
pub(super) struct Queue {
    /// `buckets[c]` holds the facts pushed with cost `c`. Grown on demand, never shrunk, so
    /// the allocations are reused across evaluations.
    buckets: Vec<Vec<u32>>,
    /// The bucket the next pop starts scanning from; never above [`Queue::top`].
    current: usize,
    /// One past the highest bucket pushed to since the last [`Queue::clear`].
    top: usize,
    /// Facts with costs of at least [`BUCKETS`], lowest `(cost, fact id)` first.
    heap: BinaryHeap<Reverse<(u64, u32)>>,
}

impl Queue {
    /// Empties the queue, keeping its allocations.
    pub fn clear(&mut self) {
        for bucket in &mut self.buckets[..self.top] {
            bucket.clear();
        }

        self.current = 0;
        self.top = 0;
        self.heap.clear();
    }

    /// Adds `fact` with `cost`.
    #[inline]
    pub fn push(&mut self, cost: u64, fact: u32) {
        let Some(index) = usize::try_from(cost).ok().filter(|&c| c < BUCKETS) else {
            self.heap.push(Reverse((cost, fact)));
            return;
        };

        debug_assert!(index >= self.current, "costs pushed are monotone");
        if index >= self.buckets.len() {
            self.buckets.resize_with(index + 1, Vec::new);
        }

        self.buckets[index].push(fact);
        self.top = self.top.max(index + 1);
    }

    /// Removes a fact with the lowest cost; `None` if the queue is empty.
    #[inline]
    pub fn pop(&mut self) -> Option<(u64, u32)> {
        while self.current < self.top {
            if let Some(fact) = self.buckets[self.current].pop() {
                return Some((self.current as u64, fact));
            }

            self.current += 1;
        }

        self.heap.pop().map(|Reverse(entry)| entry)
    }
}

#[cfg(test)]
mod tests {
    //! The queue against a sorted reference under monotone pushes.

    use alloc::vec;
    use alloc::vec::Vec;

    use super::{BUCKETS, Queue};

    /// Pops interleaved with monotone pushes (some above [`BUCKETS`]) come out sorted by
    /// cost, and a cleared queue starts over.
    #[test]
    fn pops_in_cost_order() {
        let mut queue = Queue::default();
        for round in 0..2_u64 {
            let mut popped = Vec::new();
            let mut cost = 0_u64;
            for i in 0..2000_u32 {
                cost += u64::from(i % 7 == 0) * (1 + u64::from(i % 5)) * 3 * (1 + round);
                queue.push(cost, i);
                queue.push(cost + 2, i);
                if i % 3 == 0 {
                    popped.extend(queue.pop());
                }
            }

            popped.extend(core::iter::from_fn(|| queue.pop()));
            assert_eq!(popped.len(), 4000);
            assert!(popped.windows(2).all(|w| w[0].0 <= w[1].0));
            assert!(popped.iter().any(|&(c, _)| c >= BUCKETS as u64));
            queue.clear();
            assert_eq!(queue.pop(), None);
        }

        let mut queue = Queue::default();
        queue.push(5, 1);
        queue.clear();
        queue.push(0, 2);
        assert_eq!(vec![queue.pop(), queue.pop()], [Some((0, 2)), None]);
    }
}
