//! The delete relaxation of a [`Task`] as a set of rules `conditions → head`.
//!
//! Every operator effect becomes one rule (the operator's precondition plus the effect's
//! conditions), every axiom becomes one rule of cost 0. Facts are numbered variable by
//! variable: `var = value` has id `offset(var) + value`, the same numbering as
//! `width-search`'s SAS+ adapter.
//!
//! Derived variables get their default value (their value in [`Task::initial`]) for free:
//! a derived fact falls back to its default whenever its axioms stop firing, so under the
//! relaxation the default is always available. This over-approximates and keeps every
//! estimate built on it sound.

mod exploration;
mod plan;
mod queue;
mod workspace;

use alloc::vec::Vec;
use core::ops::Range;

use sas_plus::{Fact, OpId, Task};

pub use exploration::{Combine, Exploration};
pub use plan::RelaxedPlan;
pub(crate) use workspace::Workspace;
use workspace::{FactSlot, RuleSlot};

/// The delete relaxation of a task, indexed for the worklist in [`Relaxation::explore`].
///
/// Built once per task in `O(size of the task)`; evaluating states never touches the
/// [`Task`] again.
#[derive(Clone, Debug)]
pub struct Relaxation {
    /// Fact id of `var = 0` for each variable.
    offsets: Vec<u32>,
    /// Total number of facts, the sum of all domain sizes.
    num_facts: usize,
    /// All rules: first the operator effects grouped by operator (see
    /// [`Relaxation::op_rules`]), then the axioms from [`Relaxation::first_axiom`] on.
    rules: Vec<Rule>,
    /// The rules that have fact `f` as a condition are
    /// `watcher_rules[watcher_starts[f]..watcher_starts[f + 1]]`, in ascending rule order.
    /// One flat array instead of a vector per fact keeps the hot loop cache-friendly.
    watcher_starts: Vec<usize>,
    /// The concatenated watcher lists of all facts; see [`Relaxation::watcher_starts`].
    watcher_rules: Vec<u32>,
    /// The axioms that have fact `f` as a condition are
    /// `axiom_watcher_axioms[axiom_watcher_starts[f]..axiom_watcher_starts[f + 1]]`, as
    /// indices relative to [`Relaxation::first_axiom`]. Indexes the axiom closure of the
    /// relaxed plan replay; kept apart from [`Relaxation::watcher_rules`] so a fact without
    /// axiom watchers costs one range lookup.
    axiom_watcher_starts: Vec<usize>,
    /// The concatenated axiom watcher lists; see [`Relaxation::axiom_watcher_starts`].
    axiom_watcher_axioms: Vec<u32>,
    /// The worklist state of each rule before any condition is closed, with its head and
    /// cost packed alongside the counters for the inner loop; copied into the workspace in
    /// one go per evaluation.
    rule_init: Vec<RuleSlot>,
    /// [`FactSlot::UNREACHED`] once per fact, the template the workspace copies per
    /// evaluation.
    fact_init: Vec<FactSlot>,
    /// The rules without conditions, which the worklist must offer up front.
    unconditioned: Vec<u32>,
    /// For each operator, the range of its rules in [`Relaxation::rules`]. Empty for an
    /// operator without effects.
    op_rules: Vec<Range<usize>>,
    /// Index of the first axiom rule; all rules before it belong to operators.
    first_axiom: usize,
    /// Facts that hold in every relaxed state: the defaults of the derived variables.
    free: Vec<u32>,
    /// The goal facts, sorted ascending without duplicates.
    goal: Vec<usize>,
    /// Whether each fact is a goal fact.
    is_goal: Vec<bool>,
}

/// A delete-free rule `conditions → head`: one operator effect or one axiom.
#[derive(Clone, Debug)]
struct Rule {
    /// Fact ids that must all hold, sorted ascending without duplicates.
    conditions: Vec<usize>,
    /// The fact the rule makes true.
    head: usize,
    /// The operator the effect belongs to; `None` for an axiom.
    op: Option<OpId>,
    /// Cost of applying the rule: the operator cost, `0` for an axiom.
    cost: u64,
}

impl Relaxation {
    /// Builds the relaxation of `task`.
    ///
    /// `task` should pass [`Task::validate`]; for an invalid task the estimates are meaningless.
    ///
    /// # Panics
    ///
    /// Panics if a fact of `task` refers to an unknown variable, or if the task has `2³²` or
    /// more facts or rules (ids are stored as `u32`).
    #[must_use]
    pub fn new(task: &Task) -> Self {
        let offsets = fact_offsets(task);
        let num_facts = task.variables.iter().map(|v| v.domain_size as usize).sum();
        assert!(u32::try_from(num_facts).is_ok(), "fewer than 2^32 facts");
        let id = |f: &Fact| (offsets[f.var] + f.value) as usize;

        let (mut rules, op_rules) = operator_rules(task, &id);
        let first_axiom = rules.len();
        rules.extend(task.axioms.iter().map(|a| Rule {
            conditions: sorted_ids(a.conditions.iter().map(id)),
            head: id(&a.head),
            op: None,
            cost: 0,
        }));

        let free = (0..task.num_variables())
            .filter(|&var| task.variables[var].is_derived())
            .map(|var| offsets[var] + task.initial.0[var])
            .collect();
        let goal = sorted_ids(task.goal.iter().map(id));
        let mut is_goal = alloc::vec![false; num_facts];
        for &g in &goal {
            is_goal[g] = true;
        }

        let (watcher_starts, watcher_rules) = watchers(&rules, num_facts);
        let (axiom_watcher_starts, axiom_watcher_axioms) =
            watchers(&rules[first_axiom..], num_facts);
        let (rule_init, unconditioned) = worklist_tables(&rules);

        Self {
            offsets,
            num_facts,
            rules,
            watcher_starts,
            watcher_rules,
            axiom_watcher_starts,
            axiom_watcher_axioms,
            rule_init,
            fact_init: alloc::vec![FactSlot::UNREACHED; num_facts],
            unconditioned,
            op_rules,
            first_axiom,
            free,
            goal,
            is_goal,
        }
    }

    /// Total number of facts; fact ids are `0..num_facts()`.
    #[must_use]
    pub fn num_facts(&self) -> usize {
        self.num_facts
    }

    /// The id of `fact`: `offset(var) + value`, numbered variable by variable.
    ///
    /// # Panics
    ///
    /// Panics if `fact.var` is not a variable of the task.
    #[must_use]
    pub fn fact_id(&self, fact: Fact) -> usize {
        (self.offsets[fact.var] + fact.value) as usize
    }

    /// The fact ids true in `state`, one per variable, in variable order.
    fn state_facts<'s>(&'s self, state: &'s sas_plus::State) -> impl Iterator<Item = u32> + 's {
        state
            .0
            .iter()
            .zip(&self.offsets)
            .map(|(&value, &offset)| offset + value)
    }

    /// The facts true in every relaxed state from `state`: its own facts and the free facts.
    fn start_facts<'s>(&'s self, state: &'s sas_plus::State) -> impl Iterator<Item = u32> + 's {
        self.state_facts(state).chain(self.free.iter().copied())
    }

    /// The rules that have `fact` as a condition, in ascending rule order.
    fn watchers(&self, fact: usize) -> &[u32] {
        &self.watcher_rules[self.watcher_starts[fact]..self.watcher_starts[fact + 1]]
    }

    /// The axioms that have `fact` as a condition, as indices relative to
    /// [`Relaxation::first_axiom`], ascending.
    fn axiom_watchers(&self, fact: usize) -> &[u32] {
        &self.axiom_watcher_axioms
            [self.axiom_watcher_starts[fact]..self.axiom_watcher_starts[fact + 1]]
    }
}

/// Fact id of `var = 0` for each variable of `task`.
///
/// # Panics
///
/// Panics if the task has `2³²` or more facts.
fn fact_offsets(task: &Task) -> Vec<u32> {
    task.variables
        .iter()
        .scan(0, |next, v| {
            let offset = *next;
            *next += v.domain_size as usize;
            Some(to_u32(offset))
        })
        .collect()
}

/// One rule per operator effect, grouped by operator, with each operator's rule range.
fn operator_rules(task: &Task, id: &impl Fn(&Fact) -> usize) -> (Vec<Rule>, Vec<Range<usize>>) {
    let mut rules = Vec::new();
    let mut ranges = Vec::with_capacity(task.operators.len());
    for (op, operator) in task.operators.iter().enumerate() {
        let start = rules.len();
        rules.extend(operator.eff.iter().map(|e| Rule {
            conditions: sorted_ids(operator.pre.iter().chain(&e.conditions).map(id)),
            head: id(&e.fact),
            op: Some(op),
            cost: u64::from(operator.cost),
        }));
        ranges.push(start..rules.len());
    }

    (rules, ranges)
}

/// The ids sorted ascending without duplicates, so that condition counters count each
/// fact once.
fn sorted_ids(ids: impl Iterator<Item = usize>) -> Vec<usize> {
    let mut ids: Vec<usize> = ids.collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// For each of the `num_facts` facts, the rules that have it as a condition, as offsets
/// (`num_facts + 1` entries) into one flat array of rule indices.
fn watchers(rules: &[Rule], num_facts: usize) -> (Vec<usize>, Vec<u32>) {
    let mut starts = alloc::vec![0; num_facts + 1];
    for rule in rules {
        for &c in &rule.conditions {
            starts[c + 1] += 1;
        }
    }

    for f in 0..num_facts {
        starts[f + 1] += starts[f];
    }

    let mut next = starts.clone();
    let mut flat = alloc::vec![0; starts[num_facts]];
    for (r, rule) in rules.iter().enumerate() {
        for &c in &rule.conditions {
            flat[next[c]] = to_u32(r);
            next[c] += 1;
        }
    }

    (starts, flat)
}

/// The packed per-rule worklist state before any condition is closed, and the rules
/// without conditions.
fn worklist_tables(rules: &[Rule]) -> (Vec<RuleSlot>, Vec<u32>) {
    let rule_init = rules
        .iter()
        .map(|r| {
            let cost = u32::try_from(r.cost).expect("rule costs come from u32 operator costs");
            RuleSlot::new(to_u32(r.conditions.len()), to_u32(r.head), cost)
        })
        .collect();
    let unconditioned = (0..rules.len())
        .filter(|&r| rules[r].conditions.is_empty())
        .map(to_u32)
        .collect();

    (rule_init, unconditioned)
}

/// Converts a fact or rule id to the `u32` the worklist stores.
///
/// # Panics
///
/// Panics if `id` does not fit, i.e. the task has `2³²` or more facts or rules.
fn to_u32(id: usize) -> u32 {
    u32::try_from(id).expect("fewer than 2^32 facts and rules")
}
