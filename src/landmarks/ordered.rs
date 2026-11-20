//! Landmark graphs with disjunctive landmarks and typed orderings, and the reasonable
//! orderings (Hoffmann, Porteous & Sebastia 2004) on top of any of them.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use sas_plus::State;

use super::LandmarkGraph;
use crate::Relaxation;

/// How two landmarks of an [`OrderedLandmarks`] graph are ordered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OrderingKind {
    /// Sound: on every plan, `before` holds in a state at or before the first state in which
    /// `after` holds.
    Natural,
    /// Sound and stronger than [`OrderingKind::Natural`]: on every plan, `before` holds in the
    /// state immediately before the first state in which `after` holds.
    GreedyNecessary,
    /// A heuristic hint, not sound: achieving `after` before `before` is wasted effort, since
    /// `after` would be deleted again on the way to `before`. Here only added when every way
    /// to achieve `before` deletes `after` (see [`OrderedLandmarks::add_reasonable_orderings`]).
    Reasonable,
}

/// An ordering between two landmarks of an [`OrderedLandmarks`] graph, by index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LandmarkOrdering {
    /// Index of the earlier landmark in [`OrderedLandmarks::landmarks`].
    pub before: usize,
    /// Index of the later landmark.
    pub after: usize,
    /// The kind of ordering.
    pub kind: OrderingKind,
}

/// A landmark graph whose landmarks may be disjunctions of facts, with typed orderings.
///
/// A landmark is a non-empty set of fact ids ([`Relaxation::fact_id`]): on every plan from
/// the state, at least one of them holds at some point. A single fact is a fact landmark.
/// Built by [`Relaxation::rhw_landmarks`], or converted from the `h^1` [`LandmarkGraph`]
/// (all fact landmarks, natural orderings) with [`From`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct OrderedLandmarks {
    /// The landmarks, each a set of fact ids sorted ascending without duplicates; the list is
    /// sorted and without duplicates.
    landmarks: Vec<Vec<usize>>,
    /// The orderings, sorted, without duplicates.
    orderings: Vec<LandmarkOrdering>,
}

impl OrderedLandmarks {
    /// The landmarks, each a set of fact ids ascending; sorted.
    #[must_use]
    pub fn landmarks(&self) -> &[Vec<usize>] {
        &self.landmarks
    }

    /// The orderings, sorted.
    #[must_use]
    pub fn orderings(&self) -> &[LandmarkOrdering] {
        &self.orderings
    }

    /// The index of the landmark `facts` (sorted ascending), if it is one.
    #[must_use]
    pub fn index_of(&self, facts: &[usize]) -> Option<usize> {
        self.landmarks
            .binary_search_by(|l| l.as_slice().cmp(facts))
            .ok()
    }

    /// Adds reasonable orderings `q →r p` between fact landmarks (Hoffmann, Porteous &
    /// Sebastia 2004, approximated as in LAMA): for a fact landmark `p` that does not hold in
    /// `state`, and a fact landmark `q` that *interferes* with `p` (achieving `q` always
    /// deletes `p`), when
    ///
    /// - `p` is a goal fact, which must hold at the end, or
    /// - `p` and `q` are both ordered (naturally or greedy-necessarily) before a common
    ///   landmark.
    ///
    /// `q` interferes with `p` if they are values of one variable, or if `q` is a state
    /// fact and every operator effect that adds it comes with an effect of the same operator
    /// that sets `p`'s variable to another value under no further conditions. So in every
    /// state where `p` holds and `q` does not, `q` cannot be reached while `p` stays true;
    /// the tests check exactly that.
    ///
    /// Not sound as an ordering (a plan may achieve `p` early and again later), so heuristics
    /// that must stay admissible ignore [`OrderingKind::Reasonable`].
    ///
    /// # Examples
    ///
    /// ```
    /// use planning_heuristics::{OrderingKind, Relaxation};
    /// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
    ///
    /// // Making `b` resets `a`, and both are goals: achieve `b` first.
    /// let op = |name: &str, eff: Vec<Fact>| Operator {
    ///     name: name.into(),
    ///     pre: vec![],
    ///     eff: eff.into_iter().map(Effect::new).collect(),
    ///     cost: 1,
    /// };
    /// let task = Task {
    ///     variables: vec![Variable::new("a", 2), Variable::new("b", 2)],
    ///     operators: vec![
    ///         op("make-a", vec![Fact::new(0, 1)]),
    ///         op("make-b", vec![Fact::new(1, 1), Fact::new(0, 0)]),
    ///     ],
    ///     initial: State(vec![0, 0]),
    ///     goal: vec![Fact::new(0, 1), Fact::new(1, 1)],
    ///     axioms: vec![],
    /// };
    /// let relaxation = Relaxation::new(&task);
    /// let state = task.initial_state();
    /// let mut graph = relaxation.rhw_landmarks(&state).unwrap();
    /// graph.add_reasonable_orderings(&relaxation, &state);
    ///
    /// let a = graph.index_of(&[relaxation.fact_id(Fact::new(0, 1))]).unwrap();
    /// let b = graph.index_of(&[relaxation.fact_id(Fact::new(1, 1))]).unwrap();
    /// assert!(graph
    ///     .orderings()
    ///     .iter()
    ///     .any(|o| (o.before, o.after, o.kind) == (b, a, OrderingKind::Reasonable)));
    /// ```
    pub fn add_reasonable_orderings(&mut self, relaxation: &Relaxation, state: &State) {
        let single: BTreeMap<usize, usize> = self
            .landmarks
            .iter()
            .enumerate()
            .filter(|(_, l)| l.len() == 1)
            .map(|(i, l)| (l[0], i))
            .collect();
        let in_state: Vec<usize> = relaxation.state_facts(state).map(|f| f as usize).collect();
        let mut added = Vec::new();
        for (&p, &pi) in &single {
            if in_state.contains(&p) {
                continue;
            }

            for qi in self.reasonable_candidates(relaxation, &single, p, pi) {
                if qi != pi && interferes(relaxation, self.landmarks[qi][0], p) {
                    added.push(LandmarkOrdering {
                        before: qi,
                        after: pi,
                        kind: OrderingKind::Reasonable,
                    });
                }
            }
        }

        added.retain(|o| {
            !self
                .orderings
                .iter()
                .any(|e| e.before == o.before && e.after == o.after)
        });
        self.orderings.extend(added);
        self.orderings.sort_unstable();
        self.orderings.dedup();
    }

    /// The fact landmarks to test for a reasonable ordering before the fact landmark `p`
    /// (index `pi`): all of them if `p` is a goal fact, otherwise those ordered (soundly)
    /// before a landmark that `p` is ordered before.
    fn reasonable_candidates(
        &self,
        relaxation: &Relaxation,
        single: &BTreeMap<usize, usize>,
        p: usize,
        pi: usize,
    ) -> Vec<usize> {
        if relaxation.goal_facts().binary_search(&p).is_ok() {
            return single.values().copied().collect();
        }

        let sound = |o: &&LandmarkOrdering| o.kind != OrderingKind::Reasonable;
        let mut candidates: Vec<usize> = self
            .orderings
            .iter()
            .filter(sound)
            .filter(|o| o.before == pi)
            .flat_map(|child| {
                self.orderings
                    .iter()
                    .filter(sound)
                    .filter(move |o| o.after == child.after)
                    .map(|o| o.before)
            })
            .filter(|&q| self.landmarks[q].len() == 1)
            .collect();
        candidates.sort_unstable();
        candidates.dedup();

        candidates
    }
}

impl From<&LandmarkGraph> for OrderedLandmarks {
    /// The `h^1` landmarks as fact landmarks with natural orderings.
    fn from(graph: &LandmarkGraph) -> Self {
        let mut builder = Builder::default();
        for &l in graph.landmarks() {
            builder.add(alloc::vec![l]);
        }

        for &(before, after) in graph.orderings() {
            let (b, a) = (
                builder.add(alloc::vec![before]),
                builder.add(alloc::vec![after]),
            );
            builder.order(b, a, OrderingKind::Natural);
        }

        builder.finish(|_| false)
    }
}

/// Collects landmarks and orderings by insertion index, then sorts them into an
/// [`OrderedLandmarks`].
#[derive(Clone, Debug, Default)]
pub(super) struct Builder {
    /// The landmarks in insertion order.
    landmarks: Vec<Vec<usize>>,
    /// The insertion index of each landmark.
    index: BTreeMap<Vec<usize>, usize>,
    /// Orderings between insertion indices.
    orderings: Vec<(usize, usize, OrderingKind)>,
}

impl Builder {
    /// Adds the landmark `facts` (sorted ascending, non-empty) unless present; returns its
    /// insertion index and whether it is new.
    pub(super) fn insert(&mut self, facts: Vec<usize>) -> (usize, bool) {
        debug_assert!(!facts.is_empty() && facts.is_sorted(), "sorted, non-empty");
        if let Some(&i) = self.index.get(&facts) {
            return (i, false);
        }

        let i = self.landmarks.len();
        self.index.insert(facts.clone(), i);
        self.landmarks.push(facts);

        (i, true)
    }

    /// [`Builder::insert`] without the flag.
    pub(super) fn add(&mut self, facts: Vec<usize>) -> usize {
        self.insert(facts).0
    }

    /// Number of landmarks added so far.
    pub(super) fn len(&self) -> usize {
        self.landmarks.len()
    }

    /// The landmark with insertion index `i`.
    pub(super) fn get(&self, i: usize) -> &[usize] {
        &self.landmarks[i]
    }

    /// Orders landmark `before` before `after` (insertion indices); ignores self-loops.
    pub(super) fn order(&mut self, before: usize, after: usize, kind: OrderingKind) {
        if before != after {
            self.orderings.push((before, after, kind));
        }
    }

    /// Sorts the landmarks and orderings, dropping the landmarks `drop` selects and their
    /// orderings. Of two orderings between the same pair, the strongest kind is kept.
    pub(super) fn finish(self, drop: impl Fn(&[usize]) -> bool) -> OrderedLandmarks {
        let mut order: Vec<usize> = (0..self.landmarks.len())
            .filter(|&i| !drop(&self.landmarks[i]))
            .collect();
        order.sort_unstable_by(|&a, &b| self.landmarks[a].cmp(&self.landmarks[b]));
        let mut position = alloc::vec![usize::MAX; self.landmarks.len()];
        for (p, &i) in order.iter().enumerate() {
            position[i] = p;
        }

        let mut strongest: BTreeMap<(usize, usize), OrderingKind> = BTreeMap::new();
        for (b, a, kind) in self.orderings {
            let (b, a) = (position[b], position[a]);
            if b != usize::MAX && a != usize::MAX {
                let entry = strongest.entry((b, a)).or_insert(kind);
                *entry = strength_max(*entry, kind);
            }
        }

        let orderings = strongest
            .into_iter()
            .map(|((before, after), kind)| LandmarkOrdering {
                before,
                after,
                kind,
            })
            .collect();
        let mut landmarks = self.landmarks;
        let landmarks = order
            .into_iter()
            .map(|i| core::mem::take(&mut landmarks[i]))
            .collect();

        OrderedLandmarks {
            landmarks,
            orderings,
        }
    }
}

/// The stronger of two ordering kinds: greedy-necessary over natural over reasonable.
fn strength_max(a: OrderingKind, b: OrderingKind) -> OrderingKind {
    let rank = |k| match k {
        OrderingKind::Reasonable => 0,
        OrderingKind::Natural => 1,
        OrderingKind::GreedyNecessary => 2,
    };

    if rank(b) > rank(a) { b } else { a }
}

/// Whether achieving fact `q` always deletes fact `p`; see
/// [`OrderedLandmarks::add_reasonable_orderings`].
fn interferes(relaxation: &Relaxation, q: usize, p: usize) -> bool {
    let (fq, fp) = (relaxation.fact(q), relaxation.fact(p));
    if fq.var == fp.var {
        return q != p;
    }

    let mut achievers = relaxation.rules().iter().filter(|r| r.head == q).peekable();
    if achievers.peek().is_none() {
        return false;
    }

    achievers.all(|rq| {
        rq.op.is_some_and(|op| {
            relaxation.op_rules(op).iter().any(|r| {
                let f = relaxation.fact(r.head);
                f.var == fp.var
                    && r.head != p
                    && r.conditions
                        .iter()
                        .all(|c| rq.conditions.binary_search(c).is_ok())
            })
        })
    })
}
