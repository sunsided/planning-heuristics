//! The label fixpoint behind the landmarks: one bit set of facts per fact.

use alloc::vec::Vec;

use sas_plus::State;

use crate::Relaxation;
use crate::relaxation::Rule;

/// Bits per word of a label.
const WORD: usize = 64;

/// Labels of all facts from one state, with the buffers reused across states.
///
/// The label of fact `f` is `bits[f * words..(f + 1) * words]`, valid only once `reached[f]`.
#[derive(Clone, Debug, Default)]
pub(super) struct Labels {
    /// Words per label: `ceil(facts / 64)`.
    words: usize,
    /// The labels of all facts, `words` words each.
    bits: Vec<u64>,
    /// Whether each fact has a label, i.e. is relaxed-reachable from the state.
    reached: Vec<bool>,
    /// The facts of the state, as a bit set of `words` words.
    in_state: Vec<u64>,
    /// Facts whose label changed and whose watching rules are still to be re-evaluated.
    queue: Vec<usize>,
    /// Whether each fact is on [`Labels::queue`].
    queued: Vec<bool>,
    /// The offer of the rule being evaluated, `words` words.
    offer: Vec<u64>,
    /// The goal landmarks, as a bit set of `words` words; filled by [`Labels::compute`].
    goal: Vec<u64>,
}

impl Labels {
    /// Computes the labels of every fact from `state`; returns whether every goal fact is
    /// reachable.
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    pub(super) fn compute(&mut self, relaxation: &Relaxation, state: &State) -> bool {
        self.reset(relaxation.num_facts());
        self.seed(relaxation, state);
        while let Some(fact) = self.queue.pop() {
            self.queued[fact] = false;
            for &r in relaxation.watchers(fact) {
                self.evaluate(&relaxation.rules()[r as usize]);
            }
        }

        self.collect_goal(relaxation)
    }

    /// The facts in the label of the reached fact `fact`, ascending.
    pub(super) fn label(&self, fact: usize) -> impl Iterator<Item = usize> + '_ {
        debug_assert!(self.reached[fact], "only reached facts have labels");
        ones(&self.bits[fact * self.words..(fact + 1) * self.words])
    }

    /// The goal landmarks, ascending: the goal facts and the union of their labels.
    pub(super) fn goal_landmarks(&self) -> impl Iterator<Item = usize> + '_ {
        ones(&self.goal)
    }

    /// The goal landmarks that do not hold in the state, ascending.
    pub(super) fn pending_landmarks(&self) -> impl Iterator<Item = usize> + '_ {
        ones_masked(&self.goal, &self.in_state)
    }

    /// Whether `fact` is a goal landmark that does not hold in the state.
    pub(super) fn is_pending(&self, fact: usize) -> bool {
        bit(&self.goal, fact) && !bit(&self.in_state, fact)
    }

    /// Sizes the buffers for `num_facts` facts and clears them, reusing their allocations.
    fn reset(&mut self, num_facts: usize) {
        self.words = num_facts.div_ceil(WORD);
        refill(&mut self.bits, num_facts * self.words, 0);
        refill(&mut self.reached, num_facts, false);
        refill(&mut self.in_state, self.words, 0);
        refill(&mut self.queued, num_facts, false);
        refill(&mut self.offer, self.words, 0);
        refill(&mut self.goal, self.words, 0);
        self.queue.clear();
    }

    /// Labels the state facts with themselves and the free defaults outside the state with
    /// the empty set, then lets the rules without conditions offer their heads.
    fn seed(&mut self, relaxation: &Relaxation, state: &State) {
        for fact in relaxation.state_facts(state) {
            let fact = fact as usize;
            set(&mut self.in_state, fact);
            if !self.reached[fact] {
                self.reached[fact] = true;
                set(&mut self.bits[fact * self.words..], fact);
                self.enqueue(fact);
            }
        }

        // A default outside the state holds only once the axioms stop firing: not a landmark.
        for &fact in relaxation.free_facts() {
            let fact = fact as usize;
            if !self.reached[fact] {
                self.reached[fact] = true;
                self.enqueue(fact);
            }
        }

        for &r in relaxation.unconditioned() {
            self.evaluate(&relaxation.rules()[r as usize]);
        }
    }

    /// Lets `rule` offer `{head} ∪ ⋃ label(conditions)` to its head if all its conditions are
    /// reached, intersecting the head's label with it; queues the head if its label changed.
    fn evaluate(&mut self, rule: &Rule) {
        if !rule.conditions.iter().all(|&c| self.reached[c]) {
            return;
        }

        self.offer.fill(0);
        for &c in &rule.conditions {
            let label = &self.bits[c * self.words..(c + 1) * self.words];
            for (o, &l) in self.offer.iter_mut().zip(label) {
                *o |= l;
            }
        }

        let head = rule.head;
        set(&mut self.offer, head);
        let label = &mut self.bits[head * self.words..(head + 1) * self.words];
        let changed = if self.reached[head] {
            intersect(label, &self.offer)
        } else {
            self.reached[head] = true;
            label.copy_from_slice(&self.offer);
            true
        };
        if changed {
            self.enqueue(head);
        }
    }

    /// Puts `fact` on the worklist unless it is already there.
    fn enqueue(&mut self, fact: usize) {
        if !core::mem::replace(&mut self.queued[fact], true) {
            self.queue.push(fact);
        }
    }

    /// Fills [`Labels::goal`] with the goal facts and the union of their labels; returns whether every goal
    /// fact is reached.
    fn collect_goal(&mut self, relaxation: &Relaxation) -> bool {
        for &g in relaxation.goal_facts() {
            if !self.reached[g] {
                return false;
            }

            let label = &self.bits[g * self.words..(g + 1) * self.words];
            for (o, &l) in self.goal.iter_mut().zip(label) {
                *o |= l;
            }

            // A free default outside the state has the empty label but must hold at the end.
            set(&mut self.goal, g);
        }

        true
    }
}

/// Sets bit `i` of `bits`.
fn set(bits: &mut [u64], i: usize) {
    bits[i / WORD] |= 1 << (i % WORD);
}

/// Whether bit `i` of `bits` is set.
fn bit(bits: &[u64], i: usize) -> bool {
    bits[i / WORD] & (1 << (i % WORD)) != 0
}

/// Intersects `label` with `offer` in place; returns whether `label` changed.
fn intersect(label: &mut [u64], offer: &[u64]) -> bool {
    let mut changed = false;
    for (l, &o) in label.iter_mut().zip(offer) {
        let next = *l & o;
        changed |= next != *l;
        *l = next;
    }

    changed
}

/// The indices of the set bits of `bits`, ascending.
fn ones(bits: &[u64]) -> impl Iterator<Item = usize> + '_ {
    bits.iter()
        .enumerate()
        .flat_map(|(w, &word)| word_ones(w, word))
}

/// The indices of the bits set in `bits` but not in `mask`, ascending.
fn ones_masked<'a>(bits: &'a [u64], mask: &'a [u64]) -> impl Iterator<Item = usize> + 'a {
    bits.iter()
        .zip(mask)
        .enumerate()
        .flat_map(|(w, (&word, &m))| word_ones(w, word & !m))
}

/// The indices of the set bits of `word`, the `w`-th word of a bit set, ascending.
fn word_ones(w: usize, mut word: u64) -> impl Iterator<Item = usize> {
    core::iter::from_fn(move || {
        if word == 0 {
            return None;
        }

        let i = word.trailing_zeros() as usize;
        word &= word - 1;

        Some(w * WORD + i)
    })
}

/// Makes `buffer` hold `len` copies of `value`, reusing its allocation.
fn refill<T: Clone>(buffer: &mut Vec<T>, len: usize, value: T) {
    buffer.clear();
    buffer.resize(len, value);
}
