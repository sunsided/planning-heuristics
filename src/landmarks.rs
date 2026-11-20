//! Fact landmarks of the delete relaxation (Zhu & Givan 2003; `h^m` landmarks with `m = 1`,
//! Keyder, Richter & Helmert 2010) and the heuristics built on them.
//!
//! See [`LandmarkGraph`] for the extraction, [`LandmarkCount`] and [`UniformLandmarkCost`] for
//! the heuristics.

mod heuristics;
mod labels;
#[cfg(feature = "lp")]
mod optimal;
mod ordered;
mod rhw;
mod tracking;

use alloc::vec::Vec;

use sas_plus::State;

use crate::Relaxation;

pub use heuristics::{LandmarkCount, UniformLandmarkCost};
use labels::Labels;
#[cfg(feature = "lp")]
pub use optimal::OptimalLandmarkCost;
pub use ordered::{LandmarkOrdering, OrderedLandmarks, OrderingKind};
pub use tracking::{AcceptedLandmarks, LandmarkTracker, TrackingMode};

/// The fact landmarks of one state and their natural orderings.
///
/// A fact landmark of a state is a fact that is true at some point of every plan from it. Each
/// fact `f` gets a label: the facts that every relaxed path to `f` makes true no later than
/// `f`. Facts of the state are labelled with themselves; a rule `conditions → f` offers
/// `{f} ∪ ⋃ label(conditions)`, and the label of `f` is the intersection of all its offers.
/// The greatest fixpoint is computed with a worklist that intersects labels until none
/// shrinks. The landmarks are the goal facts and the union of their labels; each fact `l` in the label of a
/// landmark `f` gives the natural ordering `l → f` (`l` holds in a state at or before the
/// first state in which `f` holds).
///
/// Every real plan is a relaxed plan, so these are landmarks of the task itself. The defaults
/// of derived variables are free in the relaxation, but in a real state a default only holds
/// when the axioms stop firing; a default that is not true in the state therefore gets the
/// empty label and is claimed as a landmark only when it is a goal fact.
///
/// ```mermaid
/// graph LR
///     seed["state facts: label {f}; free defaults: label ∅"] --> work
///     work["worklist of facts whose label changed"] -->|watching rule fires| offer
///     offer["offer {head} ∪ labels of conditions"] -->|"intersect, shrank?"| work
///     work -->|empty| goal["landmarks = goal facts ∪ their labels"]
/// ```
///
/// Fact ids are those of [`Relaxation::fact_id`]. Built by [`Relaxation::landmarks`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LandmarkGraph {
    /// The landmark facts, ascending. Includes the goal facts and the landmarks that already
    /// hold in the state.
    landmarks: Vec<usize>,
    /// Natural orderings `(before, after)` between landmarks, sorted: `before` holds in a
    /// state at or before the first state in which `after` holds, on every plan.
    orderings: Vec<(usize, usize)>,
}

impl LandmarkGraph {
    /// The landmark facts, ascending.
    #[must_use]
    pub fn landmarks(&self) -> &[usize] {
        &self.landmarks
    }

    /// The natural orderings `(before, after)`, sorted: on every plan, `before` holds in a
    /// state at or before the first state in which `after` holds.
    #[must_use]
    pub fn orderings(&self) -> &[(usize, usize)] {
        &self.orderings
    }
}

impl Relaxation {
    /// The fact landmarks of `state` with their natural orderings; `None` if the goal is
    /// unreachable under the relaxation (a dead end).
    ///
    /// Runs the label fixpoint in `O(rules × facts / 64)` words per label change and allocates
    /// `facts² / 8` bytes of labels; the heuristics ([`LandmarkCount`],
    /// [`UniformLandmarkCost`]) reuse that memory across states.
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    ///
    /// # Examples
    ///
    /// ```
    /// use planning_heuristics::Relaxation;
    /// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
    ///
    /// // `a` then `b`: setting `b` needs `a`, so `a = 1` is a landmark ordered before `b = 1`.
    /// let op = |name: &str, pre: Vec<Fact>, eff: Fact| Operator {
    ///     name: name.into(),
    ///     pre,
    ///     eff: vec![Effect::new(eff)],
    ///     cost: 1,
    /// };
    /// let task = Task {
    ///     variables: vec![Variable::new("a", 2), Variable::new("b", 2)],
    ///     operators: vec![
    ///         op("set-a", vec![], Fact::new(0, 1)),
    ///         op("set-b", vec![Fact::new(0, 1)], Fact::new(1, 1)),
    ///     ],
    ///     initial: State(vec![0, 0]),
    ///     goal: vec![Fact::new(1, 1)],
    ///     axioms: vec![],
    /// };
    /// let relaxation = Relaxation::new(&task);
    /// let graph = relaxation.landmarks(&task.initial_state()).unwrap();
    ///
    /// let (a1, b1) = (relaxation.fact_id(Fact::new(0, 1)), relaxation.fact_id(Fact::new(1, 1)));
    /// assert_eq!(graph.landmarks(), [a1, b1]);
    /// assert!(graph.orderings().contains(&(a1, b1)));
    /// ```
    #[must_use]
    pub fn landmarks(&self, state: &State) -> Option<LandmarkGraph> {
        let mut labels = Labels::default();
        if !labels.compute(self, state) {
            return None;
        }

        let landmarks: Vec<usize> = labels.goal_landmarks().collect();
        let mut orderings: Vec<(usize, usize)> = landmarks
            .iter()
            .flat_map(|&after| {
                labels
                    .label(after)
                    .filter(move |&before| before != after)
                    .map(move |before| (before, after))
            })
            .filter(|(before, _)| landmarks.binary_search(before).is_ok())
            .collect();
        orderings.sort_unstable();

        Some(LandmarkGraph {
            landmarks,
            orderings,
        })
    }
}
