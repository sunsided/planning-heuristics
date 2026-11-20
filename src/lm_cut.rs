//! The landmark-cut heuristic `h_LM-cut` (Helmert & Domshlak 2009).
//!
//! See [`LmCut`].

pub(crate) mod contexts;

use alloc::borrow::Cow;
use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::Reverse;

use sas_plus::{OpId, State, Task};

use crate::{Heuristic, Relaxation};

/// Marks a fact or rule that `h_max` has not reached.
const UNREACHED: u64 = u64::MAX;

/// Marks a rule without conditions: its precondition choice is the artificial start fact.
const START: usize = usize::MAX;

/// A disjunctive action landmark found by [`LmCut::landmarks`]: every plan from the state
/// applies at least one of [`ActionLandmark::ops`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ActionLandmark {
    /// The operators of the cut, ascending without duplicates. Never empty.
    ops: Vec<OpId>,
    /// The cost the landmark was charged: the cheapest remaining cost of its operators when
    /// it was found. Always positive.
    cost: u64,
}

impl ActionLandmark {
    /// The operators of the cut, ascending; every plan applies at least one of them.
    #[must_use]
    pub fn ops(&self) -> &[OpId] {
        &self.ops
    }

    /// The cost charged to the landmark; the landmarks' costs sum to `h_LM-cut`.
    #[must_use]
    pub fn cost(&self) -> u64 {
        self.cost
    }
}

/// The landmark-cut heuristic `h_LM-cut` (Helmert & Domshlak 2009). Admissible, and never
/// below `h_max`.
///
/// Repeats until `h_max` of the goal is `0`, with operator costs that shrink each round:
///
/// 1. compute `h_max` and, for every rule, its *precondition choice*: a condition of maximal
///    `h_max` (the artificial start fact for a rule without conditions);
/// 2. the *goal zone* is the goal fact of maximal `h_max` plus every fact from which it is
///    reached through zero-cost rules of the justification graph (precondition choice to
///    head);
/// 3. the facts reachable from the state in that graph without entering the goal zone
///    give the *cut*: the rules from them into the goal zone. Their operators form a
///    disjunctive action landmark;
/// 4. charge it the cheapest remaining cost `m` of its operators, add `m` to the estimate and
///    subtract `m` from each of them.
///
/// The cut is computed over the rules of the [`Relaxation`]: one per effect, so conditional
/// effects are handled exactly, and the axioms as rules of cost 0, so they never appear in a
/// cut. Costs are tracked per operator, not per rule, so an operator whose effects lie in two
/// cuts is charged at most its cost in total. Every real plan is a relaxed plan of the
/// relaxation (derived defaults are free there), so each cut is a landmark of the task and
/// the sum stays admissible.
///
/// Without conditional effects the sum is at least `h_max`. With them, charging an operator
/// lowers all its effects at once, and `h_max` can drop by more than the charge when a cheapest
/// relaxed path uses two effects of one operator. [`LmCut::new`] therefore first splits each
/// conditional operator by *context* (Röger, Pommerening & Helmert 2014): one copy per
/// assignment of its effect-condition variables to a mentioned value or "any other", with the
/// matching effects made unconditional. The copies have every plan of the original task, so
/// the cuts stay landmarks (reported as original operators), and the sum is again at least
/// `h_max`. An operator with more than 64 contexts stays unsplit; then, and over a shared
/// relaxation ([`LmCut::with_relaxation`], which never splits), the estimate is
/// `max(sum, h_max)`, still admissible ([`LmCut::is_exact`] tells which).
///
/// `h_max` is recomputed from scratch each round (a Dijkstra over the rules), so a state
/// costs `O(rounds × rules × log facts)`.
///
/// ```mermaid
/// graph LR
///     hmax["h_max with current costs"] -->|"goal h_max = 0"| done["estimate"]
///     hmax --> zone["goal zone via zero-cost rules"]
///     zone --> cut["rules from the start side into the zone"]
///     cut --> charge["charge cheapest op, subtract"]
///     charge --> hmax
/// ```
///
/// # Examples
///
/// ```
/// use planning_heuristics::{HMax, Heuristic, LmCut};
/// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
///
/// // Two independent goals of cost 2 and 3: h_max = 3, LM-cut finds both cuts.
/// let set = |name: &str, var: usize, cost: u32| Operator {
///     name: name.into(),
///     pre: vec![],
///     eff: vec![Effect::new(Fact::new(var, 1))],
///     cost,
/// };
/// let task = Task {
///     variables: vec![Variable::new("a", 2), Variable::new("b", 2)],
///     operators: vec![set("set-a", 0, 2), set("set-b", 1, 3)],
///     initial: State(vec![0, 0]),
///     goal: vec![Fact::new(0, 1), Fact::new(1, 1)],
///     axioms: vec![],
/// };
/// let state = task.initial_state();
///
/// assert_eq!(HMax::new(&task).estimate(&state), Some(3));
/// let mut lm_cut = LmCut::new(&task);
/// assert_eq!(lm_cut.estimate(&state), Some(5));
/// let landmarks = lm_cut.landmarks(&state).unwrap();
/// assert_eq!(landmarks.len(), 2);
/// assert!(landmarks.iter().all(|l| l.ops().len() == 1));
/// ```
#[derive(Clone, Debug)]
pub struct LmCut<'r> {
    /// The relaxation of the task, owned or shared.
    relaxation: Cow<'r, Relaxation>,
    /// The cost of each operator of the relaxation.
    costs: Vec<u64>,
    /// With context splitting, the original operator of each operator of the relaxation.
    op_of: Option<Vec<OpId>>,
    /// Whether every operator's effects share one condition set (no conditional effect is
    /// left), so the cut sum is at least `h_max` and is the estimate.
    exact: bool,
    /// The rules with head `f` are `by_head[head_starts[f]..head_starts[f + 1]]`.
    head_starts: Vec<usize>,
    /// The concatenated rule lists per head; see [`LmCut::head_starts`].
    by_head: Vec<u32>,
    /// The number of operators of the original task, the length of per-call costs (more
    /// than the relaxation's operators only with context splitting).
    num_task_ops: usize,
    /// Working memory, reused across states.
    work: Work,
}

/// The per-evaluation buffers of [`LmCut`].
#[derive(Clone, Debug, Default)]
struct Work {
    /// The remaining cost of each operator in the current round.
    remaining: Vec<u64>,
    /// The remaining start cost of each fact: the cost of its artificial seed operator from
    /// the start fact; [`UNREACHED`] if it is not seeded. A state seeds its facts with `0`.
    seed: Vec<u64>,
    /// The seeded facts whose seed operator is in the current cut.
    cut_seeds: Vec<usize>,
    /// `h_max` of each fact; [`UNREACHED`] if unreached.
    hmax: Vec<u64>,
    /// Unfinalized conditions left per rule; the rule fires at `0`.
    missing: Vec<u32>,
    /// Precondition choice per fired rule: its condition of maximal `h_max`, or [`START`].
    /// Only meaningful for rules with `missing == 0`.
    pcf: Vec<usize>,
    /// Whether each fact's `h_max` is final.
    done: Vec<bool>,
    /// Dijkstra queue of `(h_max, fact)`.
    heap: BinaryHeap<Reverse<(u64, usize)>>,
    /// Whether each fact is in the goal zone.
    in_zone: Vec<bool>,
    /// Whether each fact is reached from the state without entering the goal zone.
    in_start: Vec<bool>,
    /// Depth-first stack for the zone and the start side.
    stack: Vec<usize>,
    /// Whether each operator is in the current cut.
    in_cut: Vec<bool>,
    /// The operators of the current cut.
    cut: Vec<OpId>,
}

/// Most context copies per operator for [`LmCut::new`].
const MAX_CONTEXTS: usize = 64;

impl LmCut<'static> {
    /// Builds `h_LM-cut` for `task`, owning its relaxation. Operators with conditional
    /// effects are split by context first (at most 64 copies per operator; see
    /// [`LmCut::is_exact`]).
    #[must_use]
    pub fn new(task: &Task) -> Self {
        match contexts::split(task, MAX_CONTEXTS) {
            Some(split) => {
                let mut h = Self::from_cow(Cow::Owned(Relaxation::new(&split.task)));
                h.op_of = Some(split.op_of);
                h.num_task_ops = task.operators.len();
                h
            }
            None => Self::from_cow(Cow::Owned(Relaxation::new(task))),
        }
    }
}

impl LmCut<'static> {
    /// `h_LM-cut` owning `relaxation`, without context splitting.
    pub(crate) fn unsplit(relaxation: Relaxation) -> Self {
        Self::from_cow(Cow::Owned(relaxation))
    }
}

impl<'r> LmCut<'r> {
    /// Builds `h_LM-cut` over a shared `relaxation`, without context splitting.
    #[must_use]
    pub fn with_relaxation(relaxation: &'r Relaxation) -> Self {
        Self::from_cow(Cow::Borrowed(relaxation))
    }

    /// The heuristic over `relaxation`, with the operator costs and the head index.
    fn from_cow(relaxation: Cow<'r, Relaxation>) -> Self {
        let costs = (0..relaxation.num_operators())
            .map(|op| relaxation.op_rules(op).first().map_or(0, |r| r.cost))
            .collect();
        let (head_starts, by_head) = head_index(&relaxation);
        let exact = (0..relaxation.num_operators()).all(|op| {
            let rules = relaxation.op_rules(op);
            rules.iter().all(|r| r.conditions == rules[0].conditions)
        });

        Self {
            num_task_ops: relaxation.num_operators(),
            relaxation,
            costs,
            op_of: None,
            exact,
            head_starts,
            by_head,
            work: Work::default(),
        }
    }

    /// `h_LM-cut` of `state`; `None` if the goal is relaxed-unreachable (a dead end).
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    #[must_use]
    pub fn value(&mut self, state: &State) -> Option<u64> {
        self.load_state(state);
        self.work.remaining.clone_from(&self.costs);

        self.cut_sum()
    }

    /// `h_LM-cut` from seeded start costs instead of a state, optionally under per-call
    /// operator costs; `None` if a goal fact is unreachable from the seeded facts.
    ///
    /// `seeds` has one entry per fact of [`LmCut::relaxation`] (by [`Relaxation::fact_id`]):
    /// the cost of having that fact at the start, or `None` if it is not available. Each
    /// seeded fact becomes an artificial operator from a start fact with the seed as its
    /// cost, which cuts may charge like any other operator. Derived defaults stay free.
    /// `costs`, if given, replaces the operator costs, one per operator of the task the
    /// heuristic was built from (with context splitting, every copy of an operator gets its
    /// cost).
    ///
    /// Admissibility: the result never exceeds the cheapest *seeded completion*, the minimum
    /// over states `s` with every value seeded of `Σ_v seed(s[v]) + h*(s)`, with `h*` under
    /// `costs`. A seed is paid once per fact used, so seeds must be costs that add up across
    /// facts (for prices that overlap, such as two facts of one leaf state, pass the smaller
    /// per-fact share). With seeds `Some(0)` for the facts of a state and `None` elsewhere
    /// (see [`LmCut::seed_state`]) and no `costs`, this is exactly [`LmCut::value`].
    ///
    /// Allocation-free once the buffers have grown to the task's size.
    ///
    /// # Panics
    ///
    /// Panics if `seeds` does not have one entry per fact, or `costs` not one per operator.
    ///
    /// # Examples
    ///
    /// ```
    /// use planning_heuristics::LmCut;
    /// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
    ///
    /// // `b = 1` needs `a = 1`; setting `a` costs 4, setting `b` costs 1.
    /// let task = Task {
    ///     variables: vec![Variable::new("a", 2), Variable::new("b", 2)],
    ///     operators: vec![
    ///         Operator { name: "a".into(), pre: vec![], eff: vec![Effect::new(Fact::new(0, 1))], cost: 4 },
    ///         Operator {
    ///             name: "b".into(),
    ///             pre: vec![Fact::new(0, 1)],
    ///             eff: vec![Effect::new(Fact::new(1, 1))],
    ///             cost: 1,
    ///         },
    ///     ],
    ///     initial: State(vec![0, 0]),
    ///     goal: vec![Fact::new(1, 1)],
    ///     axioms: vec![],
    /// };
    /// let mut lm_cut = LmCut::new(&task);
    /// let mut seeds = Vec::new();
    /// lm_cut.seed_state(&task.initial_state(), &mut seeds);
    /// assert_eq!(lm_cut.value_seeded(&seeds, None), Some(5));
    ///
    /// // Having `a = 1` at the start for 2 is cheaper than setting it.
    /// seeds[lm_cut.relaxation().fact_id(Fact::new(0, 1))] = Some(2);
    /// assert_eq!(lm_cut.value_seeded(&seeds, None), Some(3));
    ///
    /// // With `a` free to set, only `b` costs.
    /// assert_eq!(lm_cut.value_seeded(&seeds, Some(&[0, 1])), Some(1));
    /// ```
    #[must_use]
    pub fn value_seeded(&mut self, seeds: &[Option<u64>], costs: Option<&[u64]>) -> Option<u64> {
        self.load_seeds(seeds);
        self.load_costs(costs);

        self.cut_sum()
    }

    /// `h_max` from seeded start costs, optionally under per-call operator costs; the
    /// arguments are as for [`LmCut::value_seeded`] (seeds count as in `h_max`: a fact
    /// starts at its seed). `None` if a goal fact is unreachable. Admissible in the same
    /// sense, and never above [`LmCut::value_seeded`] on a task without conditional effects.
    ///
    /// # Panics
    ///
    /// Panics if `seeds` does not have one entry per fact, or `costs` not one per operator.
    #[must_use]
    pub fn h_max_seeded(&mut self, seeds: &[Option<u64>], costs: Option<&[u64]>) -> Option<u64> {
        self.load_seeds(seeds);
        self.load_costs(costs);

        self.goal_h_max()
    }

    /// Fills `seeds` (resized to one entry per fact) with `Some(0)` for the facts of `state`
    /// and `None` elsewhere: the seeds under which [`LmCut::value_seeded`] equals
    /// [`LmCut::value`] of `state`.
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    pub fn seed_state(&self, state: &State, seeds: &mut Vec<Option<u64>>) {
        seeds.clear();
        seeds.resize(self.relaxation.num_facts(), None);
        for fact in self.relaxation.state_facts(state) {
            seeds[fact as usize] = Some(0);
        }
    }

    /// The relaxation the heuristic runs on; its [`Relaxation::fact_id`]s index the seeds of
    /// [`LmCut::value_seeded`]. With context splitting it is the relaxation of the split task,
    /// which has the same facts as the task.
    #[must_use]
    pub fn relaxation(&self) -> &Relaxation {
        &self.relaxation
    }

    /// `h_max` of `state` with `costs` (one per operator of the relaxation) instead of the
    /// operator costs; `None` if a goal fact is unreached. Used by [`crate::AdditiveHMax`].
    pub(crate) fn h_max_under(&mut self, state: &State, costs: &[u64]) -> Option<u64> {
        self.load_state(state);
        self.work.remaining.clear();
        self.work.remaining.extend_from_slice(costs);

        self.goal_h_max()
    }

    /// The cut sum under the loaded seeds and costs, at least `h_max`; `None` for a dead end.
    fn cut_sum(&mut self) -> Option<u64> {
        let mut total = 0_u64;
        let h_max = self.run(|_, cost| total = total.saturating_add(cost))?;
        debug_assert!(!self.exact || total >= h_max, "LM-cut dominates h_max");

        Some(total.max(h_max))
    }

    /// `h_max` of the goal under the loaded seeds and costs; `None` if unreached.
    fn goal_h_max(&mut self) -> Option<u64> {
        if self.relaxation.goal_facts().is_empty() {
            return Some(0);
        }

        let goal = self.hmax()?;

        Some(self.work.hmax[goal])
    }

    /// Seeds the facts of `state` with `0` and nothing else.
    fn load_state(&mut self, state: &State) {
        let seed = &mut self.work.seed;
        seed.clear();
        seed.resize(self.relaxation.num_facts(), UNREACHED);
        for fact in self.relaxation.state_facts(state) {
            seed[fact as usize] = 0;
        }
    }

    /// Loads `seeds` (one per fact) as the start costs.
    fn load_seeds(&mut self, seeds: &[Option<u64>]) {
        assert_eq!(
            seeds.len(),
            self.relaxation.num_facts(),
            "one seed per fact"
        );
        self.work.seed.clear();
        self.work
            .seed
            .extend(seeds.iter().map(|s| s.unwrap_or(UNREACHED)));
    }

    /// Loads `costs` (one per task operator, mapped onto context copies) as the operator
    /// costs, or the task's own costs.
    fn load_costs(&mut self, costs: Option<&[u64]>) {
        let Some(costs) = costs else {
            self.work.remaining.clone_from(&self.costs);
            return;
        };

        assert_eq!(costs.len(), self.num_task_ops, "one cost per operator");
        let remaining = &mut self.work.remaining;
        remaining.clear();
        match &self.op_of {
            Some(op_of) => remaining.extend(op_of.iter().map(|&op| costs[op])),
            None => remaining.extend_from_slice(costs),
        }
    }

    /// The operator costs of the relaxation.
    pub(crate) fn costs(&self) -> &[u64] {
        &self.costs
    }

    /// Whether the estimate is the plain cut sum: no conditional effect is left in the
    /// relaxation, either because the task has none or because context splitting compiled
    /// them all away. Otherwise the estimate is `max(sum, h_max)`.
    #[must_use]
    pub fn is_exact(&self) -> bool {
        self.exact
    }

    /// The action landmarks behind [`LmCut::value`], in the order they were found; their
    /// costs sum to the estimate unless `h_max` is higher (only possible with conditional
    /// effects). `None` for a dead end.
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    #[must_use]
    pub fn landmarks(&mut self, state: &State) -> Option<Vec<ActionLandmark>> {
        let mut found = Vec::new();
        let op_of = self.op_of.take();
        self.load_state(state);
        self.work.remaining.clone_from(&self.costs);
        let done = self.run(|ops, cost| {
            let mut ops: Vec<OpId> = match &op_of {
                Some(op_of) => ops.iter().map(|&op| op_of[op]).collect(),
                None => ops.to_vec(),
            };
            ops.sort_unstable();
            ops.dedup();
            found.push(ActionLandmark { ops, cost });
        });
        self.op_of = op_of;
        done?;

        Some(found)
    }

    /// Runs the cut rounds from the loaded seeds and costs, reporting the operators of every
    /// cut with its cost to `found`; returns `h_max` under the full costs, or `None` for a
    /// dead end.
    fn run(&mut self, mut found: impl FnMut(&[OpId], u64)) -> Option<u64> {
        if self.relaxation.goal_facts().is_empty() {
            return Some(0);
        }

        let mut goal = self.hmax()?;
        let h_max = self.work.hmax[goal];
        while self.work.hmax[goal] > 0 {
            self.goal_zone(goal);
            self.start_side();
            let cost = self.charge_cut();
            found(&self.work.cut, cost);
            goal = self.hmax().expect("costs only shrink");
        }

        Some(h_max)
    }

    /// Computes `h_max` and the precondition choices under the remaining costs and seeds;
    /// returns the goal fact of maximal `h_max`, or `None` if a goal fact is unreached. The
    /// goal must not be empty.
    fn hmax(&mut self) -> Option<usize> {
        let relaxation = &*self.relaxation;
        let work = &mut self.work;
        work.reset(relaxation);
        for &fact in relaxation.free_facts() {
            work.offer(fact as usize, 0);
        }

        for fact in 0..work.seed.len() {
            if work.seed[fact] != UNREACHED {
                work.offer(fact, work.seed[fact]);
            }
        }

        for &r in relaxation.unconditioned() {
            let rule = &relaxation.rules()[r as usize];
            work.pcf[r as usize] = START;
            work.offer(rule.head, work.rule_cost(rule));
        }

        while let Some(Reverse((value, fact))) = work.heap.pop() {
            if core::mem::replace(&mut work.done[fact], true) {
                continue;
            }

            for &r in relaxation.watchers(fact) {
                let r = r as usize;
                work.missing[r] -= 1;
                if work.missing[r] == 0 {
                    // Facts finalize in ascending `h_max`: the last condition is a maximal one.
                    work.pcf[r] = fact;
                    let rule = &relaxation.rules()[r];
                    work.offer(rule.head, value + work.rule_cost(rule));
                }
            }
        }

        argmax_goal(relaxation, &work.hmax)
    }

    /// Fills [`Work::in_zone`]: `goal` and every fact from which it is reached through
    /// zero-cost rules of the justification graph.
    fn goal_zone(&mut self, goal: usize) {
        let relaxation = &*self.relaxation;
        let work = &mut self.work;
        work.in_zone.fill(false);
        work.in_zone[goal] = true;
        work.stack.push(goal);
        while let Some(fact) = work.stack.pop() {
            for &r in &self.by_head[self.head_starts[fact]..self.head_starts[fact + 1]] {
                let r = r as usize;
                let rule = &relaxation.rules()[r];
                if work.missing[r] != 0 || work.rule_cost(rule) != 0 {
                    continue;
                }

                // A zero-cost rule into the zone starts at a fact of at least the goal's
                // positive `h_max`, never at the start fact.
                let pcf = work.pcf[r];
                debug_assert_ne!(pcf, START, "zone facts have positive h_max");
                if !core::mem::replace(&mut work.in_zone[pcf], true) {
                    work.stack.push(pcf);
                }
            }
        }
    }

    /// Fills [`Work::cut`] with the operators of the rules that lead from the start side into
    /// the goal zone, and [`Work::cut_seeds`] with the seeded facts in the zone (their seed
    /// operators lead there from the start fact), growing the start side
    /// ([`Work::in_start`]) from the free and seeded facts.
    fn start_side(&mut self) {
        let relaxation = &*self.relaxation;
        let work = &mut self.work;
        work.in_start.fill(false);
        work.cut.clear();
        work.cut_seeds.clear();
        for &fact in relaxation.free_facts() {
            work.enter_start(fact as usize);
        }

        for fact in 0..work.seed.len() {
            if work.seed[fact] == UNREACHED {
                continue;
            }

            if work.in_zone[fact] {
                work.cut_seeds.push(fact);
            } else {
                work.enter_start(fact);
            }
        }

        for &r in relaxation.unconditioned() {
            work.follow(relaxation, r as usize);
        }

        while let Some(fact) = work.stack.pop() {
            for &r in relaxation.watchers(fact) {
                let r = r as usize;
                if work.missing[r] == 0 && work.pcf[r] == fact {
                    work.follow(relaxation, r);
                }
            }
        }
    }

    /// Charges the cut its cheapest remaining cost over its operators and seed operators,
    /// subtracts it from each of them and returns it.
    fn charge_cut(&mut self) -> u64 {
        let work = &mut self.work;
        let cost = work
            .cut
            .iter()
            .map(|&op| work.remaining[op])
            .chain(work.cut_seeds.iter().map(|&fact| work.seed[fact]))
            .min()
            .expect("a goal of positive h_max has a non-empty cut");
        debug_assert!(cost > 0, "cut rules have positive cost");
        for &op in &work.cut {
            work.remaining[op] -= cost;
            work.in_cut[op] = false;
        }

        for &fact in &work.cut_seeds {
            work.seed[fact] -= cost;
        }

        cost
    }
}

impl Work {
    /// Sizes and clears the buffers for `relaxation`, with the remaining costs and seeds kept.
    fn reset(&mut self, relaxation: &Relaxation) {
        let facts = relaxation.num_facts();
        let rules = relaxation.rules();
        self.hmax.clear();
        self.hmax.resize(facts, UNREACHED);
        self.done.clear();
        self.done.resize(facts, false);
        self.in_zone.resize(facts, false);
        self.in_start.resize(facts, false);
        self.in_cut.resize(relaxation.num_operators(), false);
        self.pcf.resize(rules.len(), START);
        self.missing.clear();
        self.missing
            .extend(rules.iter().map(|r| to_u32(r.conditions.len())));
        self.heap.clear();
        self.stack.clear();
    }

    /// The remaining cost of `rule`: its operator's, `0` for an axiom.
    fn rule_cost(&self, rule: &crate::relaxation::Rule) -> u64 {
        rule.op.map_or(0, |op| self.remaining[op])
    }

    /// Lowers the `h_max` of `fact` to `value` if that improves it.
    fn offer(&mut self, fact: usize, value: u64) {
        if value < self.hmax[fact] {
            self.hmax[fact] = value;
            self.heap.push(Reverse((value, fact)));
        }
    }

    /// Adds `fact` to the start side unless it is there or in the goal zone.
    fn enter_start(&mut self, fact: usize) {
        if !self.in_zone[fact] && !core::mem::replace(&mut self.in_start[fact], true) {
            self.stack.push(fact);
        }
    }

    /// Follows the fired rule `r` from the start side: into the cut if its head is in the goal
    /// zone, otherwise onto the start side.
    fn follow(&mut self, relaxation: &Relaxation, r: usize) {
        let rule = &relaxation.rules()[r];
        if !self.in_zone[rule.head] {
            self.enter_start(rule.head);
            return;
        }

        let op = rule
            .op
            .expect("an axiom into the goal zone has its precondition choice in the zone");
        if !core::mem::replace(&mut self.in_cut[op], true) {
            self.cut.push(op);
        }
    }
}

impl Heuristic<State> for LmCut<'_> {
    fn estimate(&mut self, state: &State) -> Option<u32> {
        self.value(state)
            .map(|v| u32::try_from(v).unwrap_or(u32::MAX))
    }
}

/// The goal fact of maximal `h_max`; `None` if one is unreached or the goal is empty.
fn argmax_goal(relaxation: &Relaxation, hmax: &[u64]) -> Option<usize> {
    let goals = relaxation.goal_facts();
    if goals.iter().any(|&g| hmax[g] == UNREACHED) {
        return None;
    }

    goals.iter().copied().max_by_key(|&g| hmax[g])
}

/// The rules indexed by head: `(starts, rules)` with the rules of head `f` at
/// `rules[starts[f]..starts[f + 1]]`.
fn head_index(relaxation: &Relaxation) -> (Vec<usize>, Vec<u32>) {
    let mut starts = alloc::vec![0; relaxation.num_facts() + 1];
    for rule in relaxation.rules() {
        starts[rule.head + 1] += 1;
    }

    for f in 0..relaxation.num_facts() {
        starts[f + 1] += starts[f];
    }

    let mut next = starts.clone();
    let mut rules = alloc::vec![0; relaxation.rules().len()];
    for (r, rule) in relaxation.rules().iter().enumerate() {
        rules[next[rule.head]] = to_u32(r);
        next[rule.head] += 1;
    }

    (starts, rules)
}

/// `n` as `u32`; the relaxation keeps all counts below `2³²`.
fn to_u32(n: usize) -> u32 {
    u32::try_from(n).expect("fewer than 2^32 rules and conditions")
}
