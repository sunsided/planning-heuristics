//! The delete-relaxation heuristics [`HMax`], [`HAdd`] and [`HFf`] over [`sas_plus::State`].
//!
//! Each wraps a [`Relaxation`] built once per task. Evaluation allocates its working memory
//! per call. Each implements [`Heuristic`] for the default cost `u32`, saturating at
//! [`u32::MAX`]; the inherent `value` returns the exact `u64` estimate.

use sas_plus::{State, Task};

use crate::{Combine, Heuristic, Relaxation};

/// `h_max` (Bonet & Geffner 2001): the cost of the most expensive goal fact when every rule
/// costs its operator plus its most expensive condition. Admissible.
#[derive(Clone, Debug)]
pub struct HMax {
    /// The relaxation of the task.
    relaxation: Relaxation,
}

/// `h_add` (Bonet & Geffner 2001): the sum of the goal fact costs when every rule costs its
/// operator plus the sum of its conditions. Not admissible; counts shared subgoals twice.
#[derive(Clone, Debug)]
pub struct HAdd {
    /// The relaxation of the task.
    relaxation: Relaxation,
}

/// `h_FF` (Hoffmann & Nebel 2001): the cost of a relaxed plan extracted from `h_add` best
/// supporters (see [`Relaxation::relaxed_plan`]). Not admissible; `h_max <= h_FF <= h_add`.
///
/// With conditional effects this `h_FF` can be higher than Fast Downward's: the relaxed plan
/// repeats an operator when one of its effects needs another of its own effects first, and
/// counts it each time. Fast Downward counts every operator once, which can drop below `h_max`.
#[derive(Clone, Debug)]
pub struct HFf {
    /// The relaxation of the task.
    relaxation: Relaxation,
}

impl HMax {
    /// Builds `h_max` for `task`.
    #[must_use]
    pub fn new(task: &Task) -> Self {
        Self {
            relaxation: Relaxation::new(task),
        }
    }

    /// The `h_max` value of `state`; `None` if the goal is relaxed-unreachable (a dead end).
    #[must_use]
    pub fn value(&self, state: &State) -> Option<u64> {
        self.relaxation.explore(state, Combine::Max).goal_cost()
    }

    /// The underlying relaxation, for per-fact costs.
    #[must_use]
    pub fn relaxation(&self) -> &Relaxation {
        &self.relaxation
    }
}

impl HAdd {
    /// Builds `h_add` for `task`.
    #[must_use]
    pub fn new(task: &Task) -> Self {
        Self {
            relaxation: Relaxation::new(task),
        }
    }

    /// The `h_add` value of `state`; `None` if the goal is relaxed-unreachable (a dead end).
    /// Saturates at [`u64::MAX`].
    #[must_use]
    pub fn value(&self, state: &State) -> Option<u64> {
        self.relaxation.explore(state, Combine::Add).goal_cost()
    }

    /// The underlying relaxation, for per-fact costs.
    #[must_use]
    pub fn relaxation(&self) -> &Relaxation {
        &self.relaxation
    }
}

impl HFf {
    /// Builds `h_FF` for `task`.
    #[must_use]
    pub fn new(task: &Task) -> Self {
        Self {
            relaxation: Relaxation::new(task),
        }
    }

    /// The `h_FF` value of `state`; `None` if the goal is relaxed-unreachable (a dead end).
    #[must_use]
    pub fn value(&self, state: &State) -> Option<u64> {
        self.relaxation.relaxed_plan(state).map(|plan| plan.cost())
    }

    /// The underlying relaxation, for relaxed plans and per-fact costs.
    #[must_use]
    pub fn relaxation(&self) -> &Relaxation {
        &self.relaxation
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
