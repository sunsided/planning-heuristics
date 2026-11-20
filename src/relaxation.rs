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

use alloc::vec::Vec;
use core::ops::Range;

use sas_plus::{Fact, OpId, Task};

pub use exploration::{Combine, Exploration};
pub use plan::RelaxedPlan;

/// The delete relaxation of a task, indexed for the worklist in [`Relaxation::explore`].
///
/// Built once per task in `O(size of the task)`; evaluating states never touches the
/// [`Task`] again.
#[derive(Clone, Debug)]
pub struct Relaxation {
    /// Fact id of `var = 0` for each variable.
    offsets: Vec<usize>,
    /// Total number of facts, the sum of all domain sizes.
    num_facts: usize,
    /// All rules: first the operator effects grouped by operator (see
    /// [`Relaxation::op_rules`]), then the axioms from [`Relaxation::first_axiom`] on.
    rules: Vec<Rule>,
    /// For each fact, the rules that have it as a condition. Ascending rule order.
    watchers: Vec<Vec<usize>>,
    /// For each operator, the range of its rules in [`Relaxation::rules`]. Empty for an
    /// operator without effects.
    op_rules: Vec<Range<usize>>,
    /// Index of the first axiom rule; all rules before it belong to operators.
    first_axiom: usize,
    /// Facts that hold in every relaxed state: the defaults of the derived variables.
    free: Vec<usize>,
    /// The goal facts, sorted ascending without duplicates.
    goal: Vec<usize>,
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
    /// Panics if a fact of `task` refers to an unknown variable.
    #[must_use]
    pub fn new(task: &Task) -> Self {
        let offsets = fact_offsets(task);
        let num_facts = task
            .variables
            .iter()
            .map(|v| v.domain_size as usize)
            .sum();
        let id = |f: &Fact| offsets[f.var] + f.value as usize;

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
            .map(|var| offsets[var] + task.initial.0[var] as usize)
            .collect();
        let goal = sorted_ids(task.goal.iter().map(id));
        let watchers = watchers(&rules, num_facts);

        Self {
            offsets,
            num_facts,
            rules,
            watchers,
            op_rules,
            first_axiom,
            free,
            goal,
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
        self.offsets[fact.var] + fact.value as usize
    }

    /// The fact ids true in `state`, one per variable, in variable order.
    fn state_facts<'s>(&'s self, state: &'s sas_plus::State) -> impl Iterator<Item = usize> + 's {
        state
            .0
            .iter()
            .zip(&self.offsets)
            .map(|(&value, &offset)| offset + value as usize)
    }

    /// The axiom rules, in task order.
    fn axiom_rules(&self) -> &[Rule] {
        &self.rules[self.first_axiom..]
    }
}

/// Fact id of `var = 0` for each variable of `task`.
fn fact_offsets(task: &Task) -> Vec<usize> {
    task.variables
        .iter()
        .scan(0, |next, v| {
            let offset = *next;
            *next += v.domain_size as usize;
            Some(offset)
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

/// For each of the `num_facts` facts, the rules that have it as a condition.
fn watchers(rules: &[Rule], num_facts: usize) -> Vec<Vec<usize>> {
    let mut watchers = alloc::vec![Vec::new(); num_facts];
    for (r, rule) in rules.iter().enumerate() {
        for &c in &rule.conditions {
            watchers[c].push(r);
        }
    }

    watchers
}
