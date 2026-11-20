//! # Planning heuristics
//!
//! Goal-distance estimates for classical planning tasks given as [`sas_plus::Task`].
//!
//! ## Implemented
//!
//! - [`HMax`], [`HAdd`] and [`HFf`]: the delete-relaxation heuristics `h_max`, `h_add` and
//!   `h_FF` (Bonet & Geffner 2001, Hoffmann & Nebel 2001), computed with a worklist
//!   (generalized Dijkstra) over [`Relaxation`], which also extracts [`RelaxedPlan`]s.
//!
//! ## Planned
//!
//! - Red-black partial delete relaxation
//! - Landmark heuristics (LM-cut, landmark count)
//! - Adapters for merge-and-shrink and cost-partition abstractions
//!
//! See `docs/design.md` in the repository for the design and the milestones.
//!
//! ## Example
//!
//! ```
//! use planning_heuristics::{HAdd, HFf, HMax, Heuristic, Relaxation};
//! use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
//!
//! // One binary variable, one operator that sets it, goal `v0 = 1`.
//! let task = Task {
//!     variables: vec![Variable::new("v0", 2)],
//!     operators: vec![Operator {
//!         name: "set".into(),
//!         pre: vec![Fact::new(0, 0)],
//!         eff: vec![Effect::new(Fact::new(0, 1))],
//!         cost: 3,
//!     }],
//!     initial: State(vec![0]),
//!     goal: vec![Fact::new(0, 1)],
//!     axioms: vec![],
//! };
//! let state = task.initial_state();
//!
//! // Each heuristic owns its relaxation ...
//! assert_eq!(HMax::new(&task).estimate(&state), Some(3));
//!
//! // ... or borrows a shared one; one heuristic per thread, the relaxation is `Sync`.
//! let relaxation = Relaxation::new(&task);
//! let mut h_add = HAdd::with_relaxation(&relaxation);
//! let mut h_ff = HFf::with_relaxation(&relaxation);
//! assert_eq!(h_add.estimate(&state), Some(3));
//! assert_eq!(h_ff.value(&state), Some(3_u64));
//! ```
//!
//! Repository: <https://github.com/sunsided/planning-heuristics>

#![no_std]

extern crate alloc;

mod delete_relaxation;
mod relaxation;

pub use delete_relaxation::{HAdd, HFf, HMax};
pub use relaxation::{Combine, Exploration, Relaxation, RelaxedPlan};

/// An estimate of the cost from a state of type `S` to the nearest goal, in cost type `C`.
///
/// Same signature and semantics as `heuristic_search::Heuristic`, so that a later shared
/// trait (in `heuristic-search` or `search-core`) can replace this one without changing
/// behaviour. `estimate` takes `&mut self` so a heuristic can reuse working memory without
/// interior mutability; share the read-only part (such as a [`Relaxation`]) by reference and
/// create one heuristic per thread.
///
/// Integer costs are the default. A heuristic computed over a wider integer type saturates
/// when it is narrowed (an estimate never wraps around), and a float estimate over integer
/// costs rounds up (`ceil`), which keeps admissibility.
///
/// Implemented for every `FnMut(&S) -> Option<C>`, and for `&mut H` of each heuristic of this
/// crate. A blanket impl for every `&mut H: Heuristic` cannot coexist with the closure impl
/// (`&mut F` is itself `FnMut`), so wrap other heuristics as `|s| h.estimate(s)`.
pub trait Heuristic<S, C = u32> {
    /// Estimates the remaining cost from `state` to a goal.
    ///
    /// `None` means no goal is reachable from `state` (a proven dead end). Searches drop such
    /// states, so `None` is only returned when that is certain.
    fn estimate(&mut self, state: &S) -> Option<C>;
}

impl<S, C, F: FnMut(&S) -> Option<C>> Heuristic<S, C> for F {
    fn estimate(&mut self, state: &S) -> Option<C> {
        self(state)
    }
}

/// Compile-time check: the shared tables are `Sync`, the per-thread heuristics are `Send`.
const _: () = {
    /// Accepts only `Sync` types.
    const fn sync<T: Sync>() {}
    /// Accepts only `Send` types.
    const fn send<T: Send>() {}
    sync::<Relaxation>();
    send::<HMax<'static>>();
    send::<HAdd<'static>>();
    send::<HFf<'static>>();
};
