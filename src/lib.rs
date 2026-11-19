//! # Planning heuristics
//!
//! Heuristics for classical planning: red-black, FF, landmark and abstraction heuristic adapters.
//!
//! ## Planned scope
//!
//! - Red-black partial delete relaxation
//! - FF and `h_add` style relaxation heuristics
//! - Landmark heuristics
//! - Adapters for merge-and-shrink and cost-partition abstractions
//!
//! ## Status
//!
//! Pre-release. Version 0.0.0 reserves the name; the API is not started yet and will change.
//!
//! Repository: <https://github.com/sunsided/planning-heuristics>

#![no_std]

/// A goal-distance estimate for states of type `S`.
pub trait Heuristic<S> {
    /// The estimated cost type.
    type Cost;

    /// Estimates the cost to reach a goal from `state`.
    ///
    /// Returns `None` if `state` is recognized as a dead end.
    fn estimate(&self, state: &S) -> Option<Self::Cost>;
}
