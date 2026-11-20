//! Path-dependent heuristics: estimates that also depend on how the search reached a state.

use crate::Heuristic;

/// A heuristic whose estimate depends on the path to a state, through per-node information
/// the search carries along (Karpas & Domshlak 2009; LAMA's accepted landmarks).
///
/// The search calls [`PathHeuristic::root`] for the start state, [`PathHeuristic::on_expand`]
/// for every generated child, and [`PathHeuristic::merge`] when a second path reaches a state
/// it already holds information for; it stores the returned [`PathHeuristic::Info`] with the
/// node and passes it to [`PathHeuristic::estimate`].
///
/// An admissible path heuristic must give an estimate no higher than the state's true cost to
/// the goal for the information of *any* path to it, and for any merge of such information.
///
/// ```mermaid
/// graph LR
///     root["root(start)"] --> info["Info per node"]
///     info -->|"on_expand(parent, op, child)"| info
///     info -->|"merge (second path)"| info
///     info -->|"estimate(state, info)"| h["Option<C>"]
/// ```
///
/// Not part of `search_core` yet (see `docs/design.md`); a stateless [`Heuristic`] becomes a
/// path heuristic through [`Stateless`].
pub trait PathHeuristic<S, O, C = u32> {
    /// What the search stores per node.
    type Info: Clone;

    /// The information for the start state of a search.
    fn root(&mut self, state: &S) -> Self::Info;

    /// The information for `child`, generated from `parent` (with `parent_info`) by `op`.
    fn on_expand(&mut self, parent: &S, parent_info: &Self::Info, op: O, child: &S) -> Self::Info;

    /// Folds the information `other` of another path to the same state into `kept`.
    fn merge(&mut self, kept: &mut Self::Info, other: &Self::Info);

    /// Estimates the remaining cost from `state`, reached with `info`; `None` for a proven
    /// dead end.
    fn estimate(&mut self, state: &S, info: &Self::Info) -> Option<C>;
}

/// A stateless [`Heuristic`] as a [`PathHeuristic`] with no per-node information.
///
/// # Examples
///
/// ```
/// use planning_heuristics::{PathHeuristic, Stateless, heuristic_fn};
///
/// let mut h = Stateless(heuristic_fn(|n: &u32| 10_u32.checked_sub(*n)));
/// let info = PathHeuristic::<u32, ()>::root(&mut h, &3);
/// assert_eq!(PathHeuristic::<u32, ()>::estimate(&mut h, &3, &info), Some(7));
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Stateless<H>(pub H);

impl<S, O, C, H: Heuristic<S, C>> PathHeuristic<S, O, C> for Stateless<H> {
    type Info = ();

    fn root(&mut self, _state: &S) -> Self::Info {}

    fn on_expand(&mut self, _parent: &S, _parent_info: &(), _op: O, _child: &S) -> Self::Info {}

    fn merge(&mut self, _kept: &mut (), _other: &()) {}

    fn estimate(&mut self, state: &S, _info: &()) -> Option<C> {
        self.0.estimate(state)
    }
}
