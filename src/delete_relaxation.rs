//! The delete-relaxation heuristics [`HMax`], [`HAdd`] and [`HFf`] over [`sas_plus::State`].
//!
//! Each wraps a [`Relaxation`] built once per task and a workspace that is reused across
//! evaluations, so evaluating a state allocates nothing once the buffers have grown. The
//! workspace sits in a [`RefCell`], which makes the heuristics `Send` but not `Sync`: share
//! one per thread (they are cheap to [`Clone`]).
//!
//! Each implements [`Heuristic`] for the default cost `u32`, saturating at [`u32::MAX`]; the
//! inherent `value` returns the exact `u64` estimate.

use core::cell::RefCell;

use sas_plus::{State, Task};

use crate::relaxation::Workspace;
use crate::{Combine, Heuristic, Relaxation, RelaxedPlan};

/// A relaxation plus reusable working memory, shared by the three heuristics.
#[derive(Clone, Debug)]
struct Evaluator {
    /// The relaxation of the task.
    relaxation: Relaxation,
    /// Working memory for one evaluation at a time; never borrowed across calls.
    workspace: RefCell<Workspace>,
}

impl Evaluator {
    /// Builds the relaxation of `task` with an empty workspace.
    fn new(task: &Task) -> Self {
        Self {
            relaxation: Relaxation::new(task),
            workspace: RefCell::default(),
        }
    }

    /// The combined goal cost from `state`, stopping once the goal is settled.
    fn goal_cost(&self, state: &State, combine: Combine) -> Option<u64> {
        let mut workspace = self.workspace.borrow_mut();
        self.relaxation.goal_cost_in(state, combine, &mut workspace)
    }
}

/// `h_max` (Bonet & Geffner 2001): the cost of the most expensive goal fact when every rule
/// costs its operator plus its most expensive condition. Admissible.
#[derive(Clone, Debug)]
pub struct HMax {
    /// The relaxation and its workspace.
    inner: Evaluator,
}

/// `h_add` (Bonet & Geffner 2001): the sum of the goal fact costs when every rule costs its
/// operator plus the sum of its conditions. Not admissible; counts shared subgoals twice.
#[derive(Clone, Debug)]
pub struct HAdd {
    /// The relaxation and its workspace.
    inner: Evaluator,
}

/// `h_FF` (Hoffmann & Nebel 2001): the cost of a relaxed plan extracted from `h_add` best
/// supporters (see [`Relaxation::relaxed_plan`]). Not admissible; `h_max <= h_FF <= h_add`.
///
/// With conditional effects this `h_FF` can be higher than Fast Downward's: the relaxed plan
/// repeats an operator when one of its effects needs another of its own effects first, and
/// counts it each time. Fast Downward counts every operator once, which can drop below `h_max`.
#[derive(Clone, Debug)]
pub struct HFf {
    /// The relaxation and its workspace.
    inner: Evaluator,
}

impl HMax {
    /// Builds `h_max` for `task`.
    #[must_use]
    pub fn new(task: &Task) -> Self {
        Self {
            inner: Evaluator::new(task),
        }
    }

    /// The `h_max` value of `state`; `None` if the goal is relaxed-unreachable (a dead end).
    #[must_use]
    pub fn value(&self, state: &State) -> Option<u64> {
        self.inner.goal_cost(state, Combine::Max)
    }

    /// The underlying relaxation, for per-fact costs.
    #[must_use]
    pub fn relaxation(&self) -> &Relaxation {
        &self.inner.relaxation
    }
}

impl HAdd {
    /// Builds `h_add` for `task`.
    #[must_use]
    pub fn new(task: &Task) -> Self {
        Self {
            inner: Evaluator::new(task),
        }
    }

    /// The `h_add` value of `state`; `None` if the goal is relaxed-unreachable (a dead end).
    /// Saturates at [`u64::MAX`].
    #[must_use]
    pub fn value(&self, state: &State) -> Option<u64> {
        self.inner.goal_cost(state, Combine::Add)
    }

    /// The underlying relaxation, for per-fact costs.
    #[must_use]
    pub fn relaxation(&self) -> &Relaxation {
        &self.inner.relaxation
    }
}

impl HFf {
    /// Builds `h_FF` for `task`.
    #[must_use]
    pub fn new(task: &Task) -> Self {
        Self {
            inner: Evaluator::new(task),
        }
    }

    /// The `h_FF` value of `state`; `None` if the goal is relaxed-unreachable (a dead end).
    #[must_use]
    pub fn value(&self, state: &State) -> Option<u64> {
        let mut workspace = self.inner.workspace.borrow_mut();
        self.inner.relaxation.relaxed_plan_in(state, &mut workspace)
    }

    /// The relaxed plan behind [`HFf::value`]; `None` for a dead end. Reuses the working
    /// memory; only the returned operator list is allocated.
    #[must_use]
    pub fn relaxed_plan(&self, state: &State) -> Option<RelaxedPlan> {
        let mut workspace = self.inner.workspace.borrow_mut();
        let cost = self
            .inner
            .relaxation
            .relaxed_plan_in(state, &mut workspace)?;

        Some(RelaxedPlan::new(workspace.plan.clone(), cost))
    }

    /// The underlying relaxation, for per-fact costs.
    #[must_use]
    pub fn relaxation(&self) -> &Relaxation {
        &self.inner.relaxation
    }
}

/// Implements [`Heuristic`] over [`State`] for `u32` from the type's `value`, saturating.
macro_rules! impl_heuristic {
    ($($ty:ty),*) => {
        $(
            impl Heuristic<State, u32> for $ty {
                fn estimate(&self, state: &State) -> Option<u32> {
                    self.value(state).map(|v| u32::try_from(v).unwrap_or(u32::MAX))
                }
            }
        )*
    };
}

impl_heuristic!(HMax, HAdd, HFf);
