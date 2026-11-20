//! The delete-relaxation heuristics [`HMax`], [`HAdd`] and [`HFf`] over [`sas_plus::State`].
//!
//! Each combines a [`Relaxation`] (read-only, `Sync`) with its own working memory, reused
//! across evaluations, so evaluating a state allocates nothing once the buffers have grown.
//!
//! - `HMax::new(&task)` builds and owns the relaxation: simplest for one thread.
//! - `HMax::with_relaxation(&relaxation)` borrows a shared one: build the [`Relaxation`]
//!   once (keep it in an `Arc` if needed) and create one cheap heuristic per thread.
//!
//! Each implements [`Heuristic`] for the default cost `u32`, saturating at [`u32::MAX`]; the
//! inherent `value` returns the exact `u64` estimate.

use alloc::borrow::Cow;

use sas_plus::{State, Task};

use crate::relaxation::Workspace;
use crate::{Combine, Heuristic, Relaxation, RelaxedPlan};

/// Defines a heuristic type over a borrowed or owned [`Relaxation`] with its constructors,
/// its accessor and its [`Heuristic`] impls; the type supplies `value` itself.
macro_rules! relaxation_heuristic {
    ($(#[$doc:meta])* $name:ident, $what:literal) => {
        $(#[$doc])*
        #[derive(Clone, Debug)]
        pub struct $name<'r> {
            /// The relaxation of the task, owned or shared.
            relaxation: Cow<'r, Relaxation>,
            /// Working memory, refilled in place by every evaluation.
            workspace: Workspace,
        }

        impl $name<'static> {
            #[doc = concat!("Builds ", $what, " for `task`, owning its relaxation.")]
            #[must_use]
            pub fn new(task: &Task) -> Self {
                Self {
                    relaxation: Cow::Owned(Relaxation::new(task)),
                    workspace: Workspace::default(),
                }
            }
        }

        impl<'r> $name<'r> {
            #[doc = concat!("Builds ", $what, " over a shared `relaxation`. Cheap: only the")]
            /// working memory is per heuristic.
            #[must_use]
            pub fn with_relaxation(relaxation: &'r Relaxation) -> Self {
                Self {
                    relaxation: Cow::Borrowed(relaxation),
                    workspace: Workspace::default(),
                }
            }

            /// The underlying relaxation, for per-fact costs.
            #[must_use]
            pub fn relaxation(&self) -> &Relaxation {
                &self.relaxation
            }
        }

        impl Heuristic<State> for $name<'_> {
            fn estimate(&mut self, state: &State) -> Option<u32> {
                self.value(state).map(|v| u32::try_from(v).unwrap_or(u32::MAX))
            }
        }
    };
}

relaxation_heuristic!(
    /// `h_max` (Bonet & Geffner 2001): the cost of the most expensive goal fact when every rule
    /// costs its operator plus its most expensive condition. Admissible.
    HMax,
    "`h_max`"
);

relaxation_heuristic!(
    /// `h_add` (Bonet & Geffner 2001): the sum of the goal fact costs when every rule costs its
    /// operator plus the sum of its conditions. Not admissible; counts shared subgoals twice.
    HAdd,
    "`h_add`"
);

relaxation_heuristic!(
    /// `h_FF` (Hoffmann & Nebel 2001): the cost of a relaxed plan extracted from `h_add` best
    /// supporters (see [`Relaxation::relaxed_plan`]). Not admissible; `h_max <= h_FF <= h_add`.
    ///
    /// With conditional effects this `h_FF` can be higher than Fast Downward's: the relaxed plan
    /// repeats an operator when one of its effects needs another of its own effects first, and
    /// counts it each time. Fast Downward counts every operator once, which can drop below
    /// `h_max`.
    HFf,
    "`h_FF`"
);

impl HMax<'_> {
    /// The `h_max` value of `state`; `None` if the goal is relaxed-unreachable (a dead end).
    #[must_use]
    pub fn value(&mut self, state: &State) -> Option<u64> {
        self.relaxation
            .goal_cost_in(state, Combine::Max, &mut self.workspace)
    }
}

impl HAdd<'_> {
    /// The `h_add` value of `state`; `None` if the goal is relaxed-unreachable (a dead end).
    /// Saturates at [`u64::MAX`].
    #[must_use]
    pub fn value(&mut self, state: &State) -> Option<u64> {
        self.relaxation
            .goal_cost_in(state, Combine::Add, &mut self.workspace)
    }
}

impl HFf<'_> {
    /// The `h_FF` value of `state`; `None` if the goal is relaxed-unreachable (a dead end).
    #[must_use]
    pub fn value(&mut self, state: &State) -> Option<u64> {
        self.relaxation.relaxed_plan_in(state, &mut self.workspace)
    }

    /// The relaxed plan behind [`HFf::value`]; `None` for a dead end. Reuses the working
    /// memory; only the returned operator list is allocated.
    #[must_use]
    pub fn relaxed_plan(&mut self, state: &State) -> Option<RelaxedPlan> {
        let cost = self
            .relaxation
            .relaxed_plan_in(state, &mut self.workspace)?;

        Some(RelaxedPlan::new(self.workspace.plan.clone(), cost))
    }
}
