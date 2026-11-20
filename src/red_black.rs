//! The red-black heuristic `h_RB` (Katz, Hoffmann & Domshlak, ICAPS 2013 and AAAI 2013;
//! Domshlak, Hoffmann & Katz, AIJ 2015): red variables are delete-relaxed, black variables
//! keep their real semantics.
//!
//! See [`RedBlack`] for the repair and [`Painting`] for the choice of black variables.

mod forward;
mod painting;
mod product;
mod repair;
mod tables;

use alloc::borrow::Cow;
use alloc::vec::Vec;

use sas_plus::{OpId, State, Task, VarId};

pub use painting::{CausalLevel, Fragment, MAX_BOUNDED_PRODUCT, Painting, PaintingError, Strategy};
use product::{Outcome, Product};
use repair::Repair;
use tables::Tables;

use crate::relaxation::Workspace;
use crate::{Heuristic, Relaxation};

/// A plan under red-black semantics, from [`RedBlack::plan`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RedBlackPlan {
    /// The operators in application order.
    ops: Vec<OpId>,
    /// Sum of the operator costs; this is `h_RB`.
    cost: u64,
    /// Whether a red-black plan was found (by repair or by red facts following). If not, `ops`
    /// is the relaxed plan, which in general does not reach the goal under red-black
    /// semantics, and `cost` is `h_FF`.
    repaired: bool,
}

/// Options of [`RedBlack`], set with [`RedBlack::with_options`]; the default is plain relaxed
/// plan repair. Both are ignored with a [`Painting::bounded`] painting, which uses product
/// search.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct RedBlackOptions {
    /// Relaxed plan forwarding (option F of Katz, Hoffmann & Domshlak, AAAI 2013): before the
    /// repair, move every operator without black effects as early in the relaxed plan as it
    /// can go, so its red effects do not wait behind black moves. The repaired plan still
    /// contains the relaxed plan, so `h_RB >= h_FF` still holds. Ignored with
    /// [`RedBlackOptions::follow`].
    pub forward: bool,
    /// Red facts following (`h^RB_follow` of Domshlak, Hoffmann & Katz 2015, Section 5.2)
    /// instead of relaxed plan repair: only the red facts the relaxed plan needs (`R+`) are
    /// targets, achieved in any order by any operators, which avoids much of the repair's
    /// over-estimation. Its plan need not contain the relaxed plan, so `h_RB < h_FF` is
    /// possible; it is a relaxed plan, so `h_RB >= h+` still holds. In the tractable fragment
    /// it always succeeds (their Theorem 13), up to the operator budget of DAG paintings.
    pub follow: bool,
}

impl RedBlackPlan {
    /// Whether the plan is a real plan for `state` (all operators applicable under the
    /// task's real semantics and the goal reached): the test behind option S, stop search,
    /// of Katz, Hoffmann & Domshlak (AAAI 2013). A search that gets `true` can stop and
    /// return its path to `state` followed by [`RedBlackPlan::ops`].
    ///
    /// `state` must have its derived variables evaluated (see [`Task::apply`]).
    ///
    /// # Panics
    ///
    /// Panics if the plan names an operator `task` does not have, or `state` has fewer values
    /// than the task has variables.
    #[must_use]
    pub fn is_plan_for(&self, task: &Task, state: &State) -> bool {
        let mut current = state.clone();
        for &op in &self.ops {
            if !task.is_applicable(op, &current) {
                return false;
            }

            current = task.apply(op, &current);
        }

        task.is_goal(&current)
    }

    /// The operators in application order.
    #[must_use]
    pub fn ops(&self) -> &[OpId] {
        &self.ops
    }

    /// Sum of the operator costs (`h_RB`).
    #[must_use]
    pub fn cost(&self) -> u64 {
        self.cost
    }

    /// Whether the relaxed plan was repaired into a red-black plan; `false` when the estimate
    /// fell back to `h_FF`.
    #[must_use]
    pub fn is_repaired(&self) -> bool {
        self.repaired
    }
}

/// The red-black heuristic `h_RB`, by default by relaxed plan repair.
///
/// A red-black state holds a set of red facts, which only grows, and one value per black
/// variable. The estimate comes from relaxed plan repair: take the `h_FF` relaxed plan, and
/// execute it under red-black semantics; before each operator, move every black variable whose
/// precondition value differs along a cheapest path in its domain transition graph (using only
/// transitions whose red preconditions already hold), and at the end move the black variables
/// to their goal values. The cost of the repaired plan is `h_RB`. If a needed path does not
/// exist, the repair gives up and the estimate falls back to the relaxed plan cost (`h_FF`);
/// with RSE-invertible black variables a path always exists (Domshlak, Hoffmann & Katz 2015,
/// Corollary 1 and Theorem 11), so the fallback only happens outside that fragment or at the
/// operator budget of a DAG painting. The repaired plan contains the relaxed plan, so
/// `h_RB >= h_FF`, with equality when every variable is red, and the dead ends are exactly
/// those of `h_FF` (no relaxed plan means no red-black plan). Not admissible.
///
/// Two other algorithms replace the repair, and for both only `h_RB >= h+` holds (every
/// red-black plan is a relaxed plan), not `h_RB >= h_FF`:
///
/// - with [`RedBlackOptions::follow`], red facts following;
/// - with a [`Painting::bounded`] painting, product search over the black variables, which
///   also reports states whose red-black task is unsolvable as dead ends, even when a relaxed
///   plan exists (all of them real dead ends).
///
/// ```mermaid
/// graph LR
///     ff["h_FF relaxed plan"] --> next["next operator"]
///     next -->|"black precondition differs"| path["DTG path for that black variable"]
///     path --> apply["apply under red-black semantics"]
///     next -->|"black preconditions hold"| apply
///     apply --> next
///     next -->|"plan done"| goal["move black variables to goal values"]
///     path -->|"no path"| fallback["fall back to h_FF"]
/// ```
///
/// [`Painting`] decides which variables are black; see there for the restrictions that make
/// the repair succeed. With an arcless black causal graph ([`Painting::new`]) the repair runs
/// in polynomial time; with an acyclic one ([`Painting::dag`]) in time polynomial in the
/// length of the repaired plan, which can be exponential (Domshlak, Hoffmann & Katz 2015,
/// Proposition 2), so it is capped by an operator budget.
///
/// # Examples
///
/// ```
/// use planning_heuristics::{HFf, Heuristic, RedBlack};
/// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
///
/// // A switch must be on to fetch `a` and off to fetch `b`, and end off. The relaxation keeps
/// // both switch values once seen (h_FF = 3: on, fetch-a, fetch-b); a black switch must
/// // really be turned off again.
/// let op = |name: &str, pre: Vec<Fact>, eff: Fact| Operator {
///     name: name.into(),
///     pre,
///     eff: vec![Effect::new(eff)],
///     cost: 1,
/// };
/// let task = Task {
///     variables: vec![
///         Variable::new("switch", 2),
///         Variable::new("a", 2),
///         Variable::new("b", 2),
///     ],
///     operators: vec![
///         op("on", vec![Fact::new(0, 0)], Fact::new(0, 1)),
///         op("off", vec![Fact::new(0, 1)], Fact::new(0, 0)),
///         op("fetch-a", vec![Fact::new(0, 1)], Fact::new(1, 1)),
///         op("fetch-b", vec![Fact::new(0, 0)], Fact::new(2, 1)),
///     ],
///     initial: State(vec![0, 0, 0]),
///     goal: vec![Fact::new(1, 1), Fact::new(2, 1), Fact::new(0, 0)],
///     axioms: vec![],
/// };
/// let state = task.initial_state();
///
/// let mut h_rb = RedBlack::new(&task);
/// assert_eq!(h_rb.black_variables(), [0]);
/// let plan = h_rb.plan(&state).unwrap();
/// assert!(plan.is_repaired());
/// assert_eq!(HFf::new(&task).value(&state), Some(3));
/// assert_eq!(plan.cost(), 4);
/// ```
#[derive(Clone, Debug)]
pub struct RedBlack<'r> {
    /// The relaxation of the task, owned or shared.
    relaxation: Cow<'r, Relaxation>,
    /// The black variables, ascending.
    black: Vec<VarId>,
    /// Per-operator and per-black-variable tables for the repair.
    tables: Tables,
    /// Working memory of the relaxed plan.
    workspace: Workspace,
    /// Working memory of the repair.
    repair: Repair,
    /// The options.
    options: RedBlackOptions,
    /// The relaxed plan handed to the repair, forwarded if [`RedBlackOptions::forward`].
    relaxed: Vec<OpId>,
    /// The product compilation of a [`Painting::bounded`] painting, which replaces the
    /// repair by product search.
    product: Option<Product>,
}

impl RedBlack<'static> {
    /// Builds `h_RB` for `task` with the [`Painting::greedy`] painting, owning its relaxation.
    #[must_use]
    pub fn new(task: &Task) -> Self {
        Self::with_painting(task, &Painting::greedy(task))
    }

    /// Builds `h_RB` for `task` with `painting`, owning its relaxation.
    ///
    /// # Panics
    ///
    /// Panics if `painting` names a variable `task` does not have.
    #[must_use]
    pub fn with_painting(task: &Task, painting: &Painting) -> Self {
        Self::build(task, Cow::Owned(Relaxation::new(task)), painting)
    }
}

impl<'r> RedBlack<'r> {
    /// Builds `h_RB` over a shared `relaxation` of `task` with `painting`.
    ///
    /// # Panics
    ///
    /// Panics if `painting` names a variable `task` does not have.
    #[must_use]
    pub fn with_relaxation(task: &Task, relaxation: &'r Relaxation, painting: &Painting) -> Self {
        Self::build(task, Cow::Borrowed(relaxation), painting)
    }

    /// The heuristic over `relaxation`, with the repair tables of `task` under `painting`.
    fn build(task: &Task, relaxation: Cow<'r, Relaxation>, painting: &Painting) -> Self {
        let tables = Tables::new(task, &relaxation, painting.black());
        let product = painting
            .is_bounded()
            .then(|| Product::new(task, &relaxation, painting.black()));

        Self {
            relaxation,
            black: painting.black().to_vec(),
            tables,
            workspace: Workspace::default(),
            repair: Repair::default(),
            options: RedBlackOptions::default(),
            relaxed: Vec::new(),
            product,
        }
    }

    /// The same heuristic with `options`.
    ///
    /// # Examples
    ///
    /// ```
    /// use planning_heuristics::{RedBlackOptions, RedBlack};
    /// # let task = sas_plus::Task {
    /// #     variables: vec![], operators: vec![], initial: sas_plus::State(vec![]),
    /// #     goal: vec![], axioms: vec![],
    /// # };
    /// let options = RedBlackOptions {
    ///     forward: true,
    ///     ..RedBlackOptions::default()
    /// };
    /// let h_rb = RedBlack::new(&task).with_options(options);
    /// assert!(h_rb.options().forward);
    /// ```
    #[must_use]
    pub fn with_options(mut self, options: RedBlackOptions) -> Self {
        self.options = options;
        self
    }

    /// The options.
    #[must_use]
    pub fn options(&self) -> RedBlackOptions {
        self.options
    }

    /// The black variables, ascending.
    #[must_use]
    pub fn black_variables(&self) -> &[VarId] {
        &self.black
    }

    /// The `h_RB` value of `state`; `None` for a dead end: the goal is relaxed-unreachable,
    /// or (with a [`Painting::bounded`] painting) red-black-unreachable.
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    #[must_use]
    pub fn value(&mut self, state: &State) -> Option<u64> {
        self.run(state).map(|(cost, _)| cost)
    }

    /// The red-black plan behind [`RedBlack::value`]; `None` for a dead end.
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    #[must_use]
    pub fn plan(&mut self, state: &State) -> Option<RedBlackPlan> {
        let (cost, repaired) = self.run(state)?;
        let ops = if repaired {
            self.repair.plan.clone()
        } else {
            self.workspace.plan.clone()
        };

        Some(RedBlackPlan {
            ops,
            cost,
            repaired,
        })
    }

    /// Extracts the relaxed plan and repairs it; returns the cost and whether the repair
    /// succeeded (otherwise the cost is `h_FF`).
    fn run(&mut self, state: &State) -> Option<(u64, bool)> {
        let ff = self
            .relaxation
            .relaxed_plan_in(state, &mut self.workspace)?;
        if let Some(product) = &self.product {
            return match product.search(&self.relaxation, state) {
                Outcome::Plan(plan, cost) => {
                    self.repair.plan = plan;
                    Some((cost, true))
                }
                Outcome::Unsolvable => None,
                Outcome::Exhausted => Some((ff, false)),
            };
        }

        self.relaxed.clone_from(&self.workspace.plan);
        if self.options.forward && !self.options.follow {
            forward::forward(&self.tables, &mut self.relaxed);
        }

        let repaired = if self.options.follow {
            self.repair
                .follow(&self.relaxation, &self.tables, state, &self.relaxed)
        } else {
            self.repair
                .run(&self.relaxation, &self.tables, state, &self.relaxed)
        };

        Some(repaired.map_or((ff, false), |cost| (cost, true)))
    }
}

impl Heuristic<State> for RedBlack<'_> {
    fn estimate(&mut self, state: &State) -> Option<u32> {
        self.value(state)
            .map(|v| u32::try_from(v).unwrap_or(u32::MAX))
    }
}
