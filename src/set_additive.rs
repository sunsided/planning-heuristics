//! The set-additive heuristic `h_sa` (Keyder & Geffner 2008).

use alloc::borrow::Cow;
use alloc::vec::Vec;

use sas_plus::{OpId, State, Task};

use crate::{Heuristic, Relaxation};

/// Bits per word of an operator set.
const WORD: usize = 64;

/// The set-additive heuristic `h_sa` (Keyder & Geffner 2008): like `h_add`, but each fact
/// carries a relaxed plan, a *set* of operators, instead of a cost, so an operator shared by
/// two subgoals is counted once.
///
/// A rule `conditions → head` offers the union of its conditions' plans plus its operator;
/// the head keeps the cheapest offer (by summed operator cost). The offers are swept until no
/// plan gets cheaper (Bellman-Ford), and the estimate is the cost of the union of the goal
/// facts' plans. Each offer costs at most `h_add`'s sum, so `h_sa <= h_add`. Without
/// conditional effects the union is a relaxed plan, so `h_max <= h_sa`; with them, a relaxed
/// plan may need one operator twice (before and after an effect condition holds), which the
/// set counts once, so `h_sa` can fall below `h_max`. Not admissible.
///
/// One sweep is `O(rules × operators / 64)`; sweeps repeat until nothing improves.
///
/// # Examples
///
/// ```
/// use planning_heuristics::{HAdd, HSetAdd, Heuristic};
/// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
///
/// // `tool` (cost 3) is needed for both goals: h_add counts it twice, h_sa once.
/// let op = |name: &str, pre: Vec<Fact>, eff: Fact, cost: u32| Operator {
///     name: name.into(),
///     pre,
///     eff: vec![Effect::new(eff)],
///     cost,
/// };
/// let task = Task {
///     variables: vec![Variable::new("tool", 2), Variable::new("a", 2), Variable::new("b", 2)],
///     operators: vec![
///         op("tool", vec![], Fact::new(0, 1), 3),
///         op("a", vec![Fact::new(0, 1)], Fact::new(1, 1), 1),
///         op("b", vec![Fact::new(0, 1)], Fact::new(2, 1), 1),
///     ],
///     initial: State(vec![0, 0, 0]),
///     goal: vec![Fact::new(1, 1), Fact::new(2, 1)],
///     axioms: vec![],
/// };
/// let state = task.initial_state();
///
/// assert_eq!(HAdd::new(&task).estimate(&state), Some(8));
/// let mut h_sa = HSetAdd::new(&task);
/// assert_eq!(h_sa.estimate(&state), Some(5));
/// assert_eq!(h_sa.plan(&state).unwrap(), [0, 1, 2]);
/// ```
#[derive(Clone, Debug)]
pub struct HSetAdd<'r> {
    /// The relaxation of the task, owned or shared.
    relaxation: Cow<'r, Relaxation>,
    /// The cost of each operator.
    costs: Vec<u64>,
    /// Words per operator set.
    words: usize,
    /// Per fact, its plan as an operator bit set; valid only where [`HSetAdd::cost`] is set.
    sets: Vec<u64>,
    /// Per fact, the cost of its plan; `None` if unreached.
    cost: Vec<Option<u64>>,
    /// The offer being built.
    offer: Vec<u64>,
}

impl HSetAdd<'static> {
    /// Builds `h_sa` for `task`, owning its relaxation.
    #[must_use]
    pub fn new(task: &Task) -> Self {
        Self::from_cow(Cow::Owned(Relaxation::new(task)))
    }
}

impl<'r> HSetAdd<'r> {
    /// Builds `h_sa` over a shared `relaxation`.
    #[must_use]
    pub fn with_relaxation(relaxation: &'r Relaxation) -> Self {
        Self::from_cow(Cow::Borrowed(relaxation))
    }

    /// The heuristic over `relaxation` with empty buffers.
    fn from_cow(relaxation: Cow<'r, Relaxation>) -> Self {
        let costs: Vec<u64> = (0..relaxation.num_operators())
            .map(|op| relaxation.op_rules(op).first().map_or(0, |r| r.cost))
            .collect();
        let words = costs.len().div_ceil(WORD);

        Self {
            relaxation,
            costs,
            words,
            sets: Vec::new(),
            cost: Vec::new(),
            offer: Vec::new(),
        }
    }

    /// `h_sa` of `state`; `None` if the goal is relaxed-unreachable.
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    #[must_use]
    pub fn value(&mut self, state: &State) -> Option<u64> {
        self.compute(state)?;

        Some(self.set_cost(&self.offer))
    }

    /// The relaxed plan behind [`HSetAdd::value`], as operators ascending; `None` for a dead
    /// end. Applying them repeatedly under the delete relaxation reaches the goal.
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    #[must_use]
    pub fn plan(&mut self, state: &State) -> Option<Vec<OpId>> {
        self.compute(state)?;

        Some(
            (0..self.costs.len())
                .filter(|&op| self.offer[op / WORD] & (1 << (op % WORD)) != 0)
                .collect(),
        )
    }

    /// Sweeps the rules to the fixpoint and leaves the union of the goal plans in
    /// [`HSetAdd::offer`]; `None` if a goal fact is unreached.
    fn compute(&mut self, state: &State) -> Option<()> {
        let relaxation = &*self.relaxation;
        let facts = relaxation.num_facts();
        self.sets.clear();
        self.sets.resize(facts * self.words, 0);
        self.cost.clear();
        self.cost.resize(facts, None);
        self.offer.clear();
        self.offer.resize(self.words, 0);
        for f in relaxation
            .state_facts(state)
            .chain(relaxation.free_facts().iter().copied())
        {
            self.cost[f as usize] = Some(0);
        }

        let rules = relaxation.rules().len();
        let mut changed = true;
        while changed {
            changed = false;
            for r in 0..rules {
                changed |= self.offer_rule(r);
            }
        }

        self.offer.fill(0);
        for &g in self.relaxation.goal_facts() {
            self.cost[g]?;
            let set = &self.sets[g * self.words..(g + 1) * self.words];
            for (o, &s) in self.offer.iter_mut().zip(set) {
                *o |= s;
            }
        }

        Some(())
    }

    /// Lets rule `r` offer its plan to its head; returns whether the head's plan got cheaper.
    fn offer_rule(&mut self, r: usize) -> bool {
        let rule = &self.relaxation.rules()[r];
        if rule.conditions.iter().any(|&c| self.cost[c].is_none()) {
            return false;
        }

        self.offer.fill(0);
        for &c in &rule.conditions {
            let set = &self.sets[c * self.words..(c + 1) * self.words];
            for (o, &s) in self.offer.iter_mut().zip(set) {
                *o |= s;
            }
        }

        if let Some(op) = rule.op {
            self.offer[op / WORD] |= 1 << (op % WORD);
        }

        let cost = self.set_cost(&self.offer);
        let head = rule.head;
        if self.cost[head].is_some_and(|old| old <= cost) {
            return false;
        }

        self.cost[head] = Some(cost);
        self.sets[head * self.words..(head + 1) * self.words].copy_from_slice(&self.offer);

        true
    }

    /// The summed cost of the operators in `set`.
    fn set_cost(&self, set: &[u64]) -> u64 {
        let mut total = 0_u64;
        for (w, &word) in set.iter().enumerate() {
            let mut word = word;
            while word != 0 {
                let op = w * WORD + word.trailing_zeros() as usize;
                total = total.saturating_add(self.costs[op]);
                word &= word - 1;
            }
        }

        total
    }
}

impl Heuristic<State> for HSetAdd<'_> {
    fn estimate(&mut self, state: &State) -> Option<u32> {
        self.value(state)
            .map(|v| u32::try_from(v).unwrap_or(u32::MAX))
    }
}
