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
//! use planning_heuristics::{HAdd, HFf, HMax, Heuristic};
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
//! assert_eq!(HMax::new(&task).estimate(&state), Some(3));
//! assert_eq!(HAdd::new(&task).estimate(&state), Some(3));
//! assert_eq!(HFf::new(&task).estimate(&state), Some(3));
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
/// Same shape and semantics as `heuristic_search::Heuristic` (draft), so that a later shared
/// trait (in `heuristic-search` or `search-core`) can replace this one without changing
/// behaviour.
///
/// Integer costs are the default. A heuristic computed over a wider integer type saturates
/// when it is narrowed (an estimate never wraps around), and a float estimate over integer
/// costs rounds up (`ceil`), which keeps admissibility.
pub trait Heuristic<S, C = u32> {
    /// Estimates the remaining cost from `state` to a goal.
    ///
    /// `None` means no goal is reachable from `state` (a proven dead end). Searches drop such
    /// states, so `None` is only returned when that is certain.
    fn estimate(&self, state: &S) -> Option<C>;
}
