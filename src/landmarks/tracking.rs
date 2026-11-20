//! LAMA-style landmark tracking along the search path (Richter & Westphal 2010; Karpas &
//! Domshlak 2009 for the admissible variant).

use alloc::borrow::Cow;
use alloc::vec::Vec;

use sas_plus::{Fact, OpId, State, Task};

use super::heuristics::{round_up_admissible, to_f64};
use super::ordered::{OrderedLandmarks, OrderingKind};
use crate::{PathHeuristic, Relaxation};

/// Bits per word of an [`AcceptedLandmarks`] set.
const WORD: usize = 64;

/// What a [`LandmarkTracker`] estimates from the landmarks still to achieve.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TrackingMode {
    /// LAMA: the number of landmarks still to achieve. A landmark is accepted once it holds
    /// and all landmarks ordered before it (reasonable orderings included) are accepted.
    /// Inadmissible and blind to costs.
    Count,
    /// Admissible: uniform cost partitioning over the landmarks still to achieve, as in
    /// [`crate::UniformLandmarkCost`]. A landmark is accepted once it holds, whatever the
    /// orderings, and only sound orderings decide what is required again.
    UniformCost,
}

/// The landmarks a search path has accepted: the per-node information of a
/// [`LandmarkTracker`]. One bit per landmark of the tracker's graph.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct AcceptedLandmarks {
    /// Bit `i` is set if landmark `i` is accepted.
    bits: Vec<u64>,
}

impl AcceptedLandmarks {
    /// Whether landmark `i` (an index into [`OrderedLandmarks::landmarks`]) is accepted.
    #[must_use]
    pub fn is_accepted(&self, i: usize) -> bool {
        self.bits
            .get(i / WORD)
            .is_some_and(|w| w & (1 << (i % WORD)) != 0)
    }

    /// The number of accepted landmarks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Whether no landmark is accepted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bits.iter().all(|&w| w == 0)
    }

    /// Marks landmark `i` accepted.
    fn accept(&mut self, i: usize) {
        self.bits[i / WORD] |= 1 << (i % WORD);
    }
}

/// Landmark heuristics that track, along the search path, which landmarks of the start state
/// have been achieved (LAMA; Richter & Westphal 2010), as a [`PathHeuristic`].
///
/// The landmark graph is computed once, for the start state of the search (for example with
/// [`Relaxation::rhw_landmarks`] and [`OrderedLandmarks::add_reasonable_orderings`]). Each node
/// carries its [`AcceptedLandmarks`]; a child accepts the landmarks that hold in it (in
/// [`TrackingMode::Count`], only those whose predecessors in the graph the parent accepted).
/// A landmark is still to achieve if it is not accepted, or if it is accepted, does not hold,
/// and is *required again*: it is a goal fact, or it is ordered greedy-necessarily before a
/// landmark that is not accepted. When two paths reach a state, their accepted sets are
/// intersected (Karpas & Domshlak 2009), which keeps the admissible mode admissible.
///
/// In [`TrackingMode::UniformCost`] every landmark still to achieve is false in the state and
/// must be made true by a future operator, so the uniform cost partitioning over them never
/// exceeds the cost of the rest of any plan; a landmark still to achieve that no operator or
/// axiom adds proves a dead end. [`TrackingMode::Count`] never reports a dead end: its
/// ordered acceptance can leave a landmark unaccepted that was already achieved.
///
/// # Examples
///
/// ```
/// use planning_heuristics::{LandmarkTracker, PathHeuristic, TrackingMode};
/// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
///
/// // `a` then `b`: two landmarks, both pending at the start, one after `set-a`.
/// let op = |name: &str, pre: Vec<Fact>, eff: Fact| Operator {
///     name: name.into(),
///     pre,
///     eff: vec![Effect::new(eff)],
///     cost: 1,
/// };
/// let task = Task {
///     variables: vec![Variable::new("a", 2), Variable::new("b", 2)],
///     operators: vec![
///         op("set-a", vec![], Fact::new(0, 1)),
///         op("set-b", vec![Fact::new(0, 1)], Fact::new(1, 1)),
///     ],
///     initial: State(vec![0, 0]),
///     goal: vec![Fact::new(1, 1)],
///     axioms: vec![],
/// };
/// let start = task.initial_state();
/// let mut lama = LandmarkTracker::for_task(&task, TrackingMode::Count).unwrap();
///
/// let root = lama.root(&start);
/// assert_eq!(lama.estimate(&start, &root), Some(2));
/// let next = task.apply(0, &start);
/// let info = lama.on_expand(&start, &root, 0, &next);
/// assert_eq!(lama.estimate(&next, &info), Some(1));
/// ```
#[derive(Clone, Debug)]
pub struct LandmarkTracker<'r> {
    /// The relaxation of the task, owned or shared.
    relaxation: Cow<'r, Relaxation>,
    /// How the estimate is formed and landmarks accepted.
    mode: TrackingMode,
    /// The landmarks, each a set of fact ids.
    landmarks: Vec<Vec<usize>>,
    /// The landmarks' facts, resolved once for the per-node truth tests.
    facts: Vec<Vec<Fact>>,
    /// Per landmark, the landmarks ordered before it by any ordering (for
    /// [`TrackingMode::Count`] acceptance).
    parents: Vec<Vec<usize>>,
    /// Per landmark, the landmarks it is ordered greedy-necessarily before.
    greedy_children: Vec<Vec<usize>>,
    /// Whether each landmark is a single goal fact.
    goal: Vec<bool>,
    /// Per landmark, whether an axiom can add one of its facts.
    by_axiom: Vec<bool>,
    /// Per landmark, the operators that add one of its facts, ascending.
    achievers: Vec<Vec<OpId>>,
    /// The cost of each operator.
    costs: Vec<u64>,
    /// Per operator, the number of pending landmarks it adds in the current estimate.
    shares: Vec<u32>,
    /// The pending landmarks of the current estimate.
    pending: Vec<usize>,
}

impl LandmarkTracker<'static> {
    /// A tracker for searches from the initial state of `task`, with its RHW landmarks plus
    /// reasonable orderings; `None` if the initial state is a dead end.
    #[must_use]
    pub fn for_task(task: &Task, mode: TrackingMode) -> Option<Self> {
        let relaxation = Relaxation::new(task);
        let state = task.initial_state();
        let mut graph = relaxation.rhw_landmarks(&state)?;
        graph.add_reasonable_orderings(&relaxation, &state);

        Some(Self::build(Cow::Owned(relaxation), &graph, mode))
    }
}

impl<'r> LandmarkTracker<'r> {
    /// A tracker over a shared `relaxation` with the landmark `graph` of the search's start
    /// state.
    ///
    /// # Panics
    ///
    /// Panics if a landmark names a fact the relaxation does not have.
    #[must_use]
    pub fn new(relaxation: &'r Relaxation, graph: &OrderedLandmarks, mode: TrackingMode) -> Self {
        Self::build(Cow::Borrowed(relaxation), graph, mode)
    }

    /// The tracker with the graph indexed for acceptance and costs.
    fn build(
        relaxation: Cow<'r, Relaxation>,
        graph: &OrderedLandmarks,
        mode: TrackingMode,
    ) -> Self {
        let n = graph.landmarks().len();
        let mut parents = alloc::vec![Vec::new(); n];
        let mut greedy_children = alloc::vec![Vec::new(); n];
        for o in graph.orderings() {
            parents[o.after].push(o.before);
            if o.kind == OrderingKind::GreedyNecessary {
                greedy_children[o.before].push(o.after);
            }
        }

        let mut fact_ops = alloc::vec![Vec::new(); relaxation.num_facts()];
        let mut fact_axiom = alloc::vec![false; relaxation.num_facts()];
        for rule in relaxation.rules() {
            match rule.op {
                Some(op) => fact_ops[rule.head].push(op),
                None => fact_axiom[rule.head] = true,
            }
        }

        let landmarks: Vec<Vec<usize>> = graph.landmarks().to_vec();
        let facts = landmarks
            .iter()
            .map(|l| l.iter().map(|&f| relaxation.fact(f)).collect())
            .collect();
        let achievers = landmarks
            .iter()
            .map(|l| {
                let mut ops: Vec<OpId> = l
                    .iter()
                    .flat_map(|&f| fact_ops[f].iter().copied())
                    .collect();
                ops.sort_unstable();
                ops.dedup();
                ops
            })
            .collect();
        let goal = landmarks
            .iter()
            .map(|l| l.len() == 1 && relaxation.goal_facts().binary_search(&l[0]).is_ok())
            .collect();
        let by_axiom = landmarks
            .iter()
            .map(|l| l.iter().any(|&f| fact_axiom[f]))
            .collect();
        let costs = (0..relaxation.num_operators())
            .map(|op| relaxation.op_rules(op).first().map_or(0, |r| r.cost))
            .collect();

        Self {
            relaxation,
            mode,
            landmarks,
            facts,
            parents,
            greedy_children,
            goal,
            by_axiom,
            achievers,
            costs,
            shares: Vec::new(),
            pending: Vec::new(),
        }
    }

    /// Whether landmark `i` holds in `state`.
    fn holds(&self, i: usize, state: &State) -> bool {
        self.facts[i].iter().any(|f| state.0[f.var] == f.value)
    }

    /// An empty accepted set sized for the graph.
    fn empty(&self) -> AcceptedLandmarks {
        AcceptedLandmarks {
            bits: alloc::vec![0; self.landmarks.len().div_ceil(WORD)],
        }
    }

    /// Fills [`LandmarkTracker::pending`] with the landmarks still to achieve in `state`.
    fn collect_pending(&mut self, state: &State, info: &AcceptedLandmarks) {
        let mut pending = core::mem::take(&mut self.pending);
        pending.clear();
        for i in 0..self.landmarks.len() {
            let needed =
                !info.is_accepted(i) || (!self.holds(i, state) && self.required_again(i, info));
            if needed {
                pending.push(i);
            }
        }

        self.pending = pending;
    }

    /// Whether the accepted landmark `i` is needed again: a goal, or greedy-necessarily
    /// before an unaccepted landmark.
    fn required_again(&self, i: usize, info: &AcceptedLandmarks) -> bool {
        self.goal[i]
            || self.greedy_children[i]
                .iter()
                .any(|&c| !info.is_accepted(c))
    }

    /// Whether a pending landmark can never be added again: no operator and no axiom adds any
    /// of its facts, and none is a free derived default.
    fn unachievable(&self, i: usize) -> bool {
        let free = self.relaxation.free_facts();
        self.achievers[i].is_empty()
            && !self.by_axiom[i]
            && !self.landmarks[i]
                .iter()
                .any(|&f| free.iter().any(|&x| x as usize == f))
    }

    /// The uniform cost partitioning over the pending landmarks.
    fn uniform_cost(&mut self) -> u64 {
        self.shares.clear();
        self.shares.resize(self.costs.len(), 0);
        for &i in &self.pending {
            for &op in &self.achievers[i] {
                self.shares[op] += 1;
            }
        }

        // A landmark an axiom can add may come for free: worth 0.
        let sum: f64 = self
            .pending
            .iter()
            .filter(|&&i| !self.by_axiom[i])
            .map(|&i| {
                self.achievers[i]
                    .iter()
                    .map(|&op| to_f64(self.costs[op]) / f64::from(self.shares[op]))
                    .fold(f64::INFINITY, f64::min)
            })
            .filter(|share| share.is_finite())
            .sum();

        round_up_admissible(sum)
    }
}

impl PathHeuristic<State, OpId, u32> for LandmarkTracker<'_> {
    type Info = AcceptedLandmarks;

    fn root(&mut self, state: &State) -> AcceptedLandmarks {
        let mut accepted = self.empty();
        for i in 0..self.landmarks.len() {
            if self.holds(i, state) {
                accepted.accept(i);
            }
        }

        accepted
    }

    fn on_expand(
        &mut self,
        _parent: &State,
        parent_info: &AcceptedLandmarks,
        _op: OpId,
        child: &State,
    ) -> AcceptedLandmarks {
        let mut accepted = parent_info.clone();
        for i in 0..self.landmarks.len() {
            if accepted.is_accepted(i) || !self.holds(i, child) {
                continue;
            }

            let ready = self.mode == TrackingMode::UniformCost
                || self.parents[i].iter().all(|&p| parent_info.is_accepted(p));
            if ready {
                accepted.accept(i);
            }
        }

        accepted
    }

    fn merge(&mut self, kept: &mut AcceptedLandmarks, other: &AcceptedLandmarks) {
        for (k, &o) in kept.bits.iter_mut().zip(&other.bits) {
            *k &= o;
        }
    }

    fn estimate(&mut self, state: &State, info: &AcceptedLandmarks) -> Option<u32> {
        self.collect_pending(state, info);
        let value = match self.mode {
            TrackingMode::Count => self.pending.len() as u64,
            TrackingMode::UniformCost => {
                // Only here is every pending landmark certainly needed again.
                if self.pending.iter().any(|&i| self.unachievable(i)) {
                    return None;
                }

                self.uniform_cost()
            }
        };

        Some(u32::try_from(value).unwrap_or(u32::MAX))
    }
}
