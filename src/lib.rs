//! # Planning heuristics
//!
//! Goal-distance estimates for classical planning tasks given as [`sas_plus::Task`].
//!
//! ## Implemented
//!
//! - [`HMax`], [`HAdd`] and [`HFf`]: the delete-relaxation heuristics `h_max`, `h_add` and
//!   `h_FF` (Bonet & Geffner 2001, Hoffmann & Nebel 2001), computed with a worklist
//!   (generalized Dijkstra) over [`Relaxation`], which also extracts [`RelaxedPlan`]s.
//! - [`HSetAdd`]: the set-additive heuristic `h_sa` (Keyder & Geffner 2008); [`AdditiveHMax`]:
//!   `h_max` summed over a partition of the operator costs; [`H2`]: `h²` (Haslum & Geffner
//!   2000), which sees pairs of facts.
//! - [`RedBlack`]: the red-black heuristic `h_RB` by relaxed plan repair (Domshlak, Hoffmann &
//!   Katz 2015), with the black variables chosen by a [`Painting`]: greedy, or by the papers'
//!   strategies (causal graph level [`CausalLevel`], incident arcs, conflict-directed) for an
//!   arcless or acyclic black causal graph ([`Fragment`]).
//! - [`Relaxation::landmarks`]: `h^1` fact landmarks with natural orderings
//!   ([`LandmarkGraph`]), and the heuristics [`LandmarkCount`] (inadmissible) and
//!   [`UniformLandmarkCost`] (admissible, uniform cost partitioning); with feature `lp`,
//!   `OptimalLandmarkCost` (admissible, optimal cost partitioning by linear programming).
//! - With feature `lp`, `OperatorCounting`: the state equation (Bonet 2013) and LM-cut
//!   landmarks as an operator-counting LP (Pommerening, Röger, Helmert & Bonet 2014).
//! - [`Relaxation::rhw_landmarks`]: back-chained landmarks (Richter, Helmert & Westphal 2008)
//!   with disjunctive and domain-transition-graph landmarks, as [`OrderedLandmarks`] with
//!   natural, greedy-necessary and reasonable orderings ([`OrderingKind`]).
//! - [`PathHeuristic`]: path-dependent heuristics; [`LandmarkTracker`] tracks the accepted
//!   landmarks along the search path, as LAMA's landmark count or admissibly with uniform cost
//!   partitioning.
//! - [`LmCut`]: the landmark-cut heuristic `h_LM-cut` (Helmert & Domshlak 2009), admissible,
//!   with its disjunctive [`ActionLandmark`]s.
//! - [`AbstractionHeuristic`]: goal distances of any [`sas_plus::Abstraction`], such as a
//!   pattern database over a [`sas_plus::Projection`]. Features `merge-and-shrink` and
//!   `cost-partition` implement [`GoalDistance`] for those crates' heuristics, which
//!   [`Adapter`] turns into a [`Heuristic`].
//!
//! ## Planned
//!
//! - Preferred operators
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

mod abstraction;
mod additive_max;
mod delete_relaxation;
mod h2;
mod landmarks;
mod lm_cut;
#[cfg(feature = "lp")]
mod operator_counting;
mod path;
mod red_black;
mod relaxation;
mod set_additive;

pub use abstraction::{AbstractionHeuristic, Adapter, GoalDistance};
pub use additive_max::AdditiveHMax;
pub use delete_relaxation::{HAdd, HFf, HMax};
pub use h2::{H2, H2Error};
#[cfg(feature = "lp")]
pub use landmarks::OptimalLandmarkCost;
pub use landmarks::{
    AcceptedLandmarks, LandmarkCount, LandmarkGraph, LandmarkOrdering, LandmarkTracker,
    OrderedLandmarks, OrderingKind, TrackingMode, UniformLandmarkCost,
};
pub use lm_cut::{ActionLandmark, LmCut};
#[cfg(feature = "lp")]
pub use operator_counting::{CountingConstraints, OperatorCounting};
pub use path::{PathHeuristic, Stateless};
pub use red_black::{CausalLevel, Fragment, Painting, PaintingError, RedBlack, RedBlackPlan};
pub use relaxation::{Combine, Exploration, Relaxation, RelaxedPlan};
pub use set_additive::HSetAdd;
// The shared trait of the planning crates: `planning_heuristics::Heuristic` and
// `search_core::Heuristic` are one trait. A doc comment here would be prepended to all three.
pub use search_core::{Heuristic, HeuristicFn, heuristic_fn};

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
    send::<LandmarkCount<'static>>();
    send::<UniformLandmarkCost<'static>>();
    send::<RedBlack<'static>>();
    send::<LmCut<'static>>();
    send::<HSetAdd<'static>>();
    send::<AdditiveHMax>();
    send::<H2>();
    send::<LandmarkTracker<'static>>();
    send::<AbstractionHeuristic<sas_plus::Projection>>();
};
