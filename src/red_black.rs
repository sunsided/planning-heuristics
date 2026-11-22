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
    /// Prefer FDR-executable black moves (choice point (ii) of Domshlak, Hoffmann & Katz 2015,
    /// Section 5.3.2), in repair and following alike. Each black move takes a shortest DTG
    /// path whose red preconditions are also true when the plan so far is executed under real
    /// semantics, if one exists, and the plain shortest path otherwise. The plan is then more
    /// often a real plan ([`RedBlackPlan::is_plan_for`], stop search), at the price of longer
    /// moves and so possibly higher estimates. Their configuration `RDLS` uses it, `RDLS-`
    /// does not (Table 1).
    pub prefer_executable: bool,
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
/// transitions whose red preconditions already hold; with
/// [`RedBlackOptions::prefer_executable`] preferring one whose red preconditions also hold
/// under real semantics), and at the end move the black variables to their goal values. The cost of the repaired plan is `h_RB`. If a needed path does not
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
    /// If the heuristic runs on the task with its conditional effects compiled away (a
    /// black variable appears in one), the task's operator behind each compiled one.
    op_of: Option<Vec<OpId>>,
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
    /// If a black variable appears in a conditional effect, the heuristic runs on the task
    /// with its conditional effects compiled away (see [`Painting`]) and builds that task's
    /// relaxation itself; `relaxation` is then unused.
    ///
    /// # Panics
    ///
    /// Panics if `painting` names a variable `task` does not have.
    #[must_use]
    pub fn with_relaxation(task: &Task, relaxation: &'r Relaxation, painting: &Painting) -> Self {
        Self::build(task, Cow::Borrowed(relaxation), painting)
    }

    /// The heuristic over `relaxation`, with the repair tables of `task` under `painting`; on
    /// the compiled task if a black variable appears in a conditional effect.
    fn build(task: &Task, relaxation: Cow<'r, Relaxation>, painting: &Painting) -> Self {
        let split = in_conditional_effect(task, painting.black())
            .then(|| painting::split(task))
            .flatten();
        let (task, relaxation, op_of) = match &split {
            Some(s) => (
                &s.task,
                Cow::Owned(Relaxation::new(&s.task)),
                Some(s.op_of.clone()),
            ),
            None => (task, relaxation, None),
        };
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
            op_of,
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
        let mut ops = if repaired {
            self.repair.plan.clone()
        } else {
            self.workspace.plan.clone()
        };
        if let Some(op_of) = &self.op_of {
            for op in &mut ops {
                *op = op_of[*op];
            }
        }

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

        self.repair.prefer_executable = self.options.prefer_executable;
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

/// Whether one of the variables `black` appears in a conditional effect of `task`, as
/// condition or effect.
fn in_conditional_effect(task: &Task, black: &[VarId]) -> bool {
    task.operators
        .iter()
        .flat_map(|op| &op.eff)
        .filter(|e| !e.conditions.is_empty())
        .any(|e| {
            core::iter::once(&e.fact)
                .chain(&e.conditions)
                .any(|f| black.binary_search(&f.var).is_ok())
        })
}

impl Heuristic<State> for RedBlack<'_> {
    fn estimate(&mut self, state: &State) -> Option<u32> {
        self.value(state)
            .map(|v| u32::try_from(v).unwrap_or(u32::MAX))
    }
}

#[cfg(test)]
mod tests {
    //! Black variables of conditional effects, measured against the compiled task.

    use alloc::vec::Vec;

    use sas_plus::Task;
    use sas_plus::fdr::SasFile;
    use sas_plus::random::{RandomTaskParams, random_task};

    use super::{CausalLevel, Fragment, Painting, RedBlack, in_conditional_effect, painting};
    use crate::HFf;

    /// The test fixtures with conditional effects.
    const FIXTURES: [&str; 3] = [
        include_str!("../tests/fixtures/conditional-effects/output.sas"),
        include_str!("../tests/fixtures/translated-courier/output.sas"),
        include_str!("../tests/fixtures/translated-wiring/output.sas"),
    ];

    /// The strategy paintings of `task`.
    fn paintings(task: &Task) -> Vec<Painting> {
        let mut all = alloc::vec![Painting::greedy(task), Painting::greedy_dag(task)];
        for fragment in [Fragment::Arcless, Fragment::Dag] {
            for level in [CausalLevel::Top, CausalLevel::Bottom] {
                all.push(Painting::causal_graph_level(task, level, fragment));
            }
            all.push(Painting::incident_arcs(task, fragment));
            all.push(Painting::conflict_directed(
                task,
                &task.initial_state(),
                fragment,
            ));
        }

        all
    }

    /// Non-vacuity: the strategies paint variables of conditional effects black, and then
    /// the repaired plan costs more than the relaxed plan of the compiled task it repairs
    /// somewhere, so the black moves matter (not only the compilation).
    #[test]
    fn conditional_effect_variables_matter_when_black() {
        let params = RandomTaskParams {
            max_conditions: 2,
            ..RandomTaskParams::default()
        };
        let tasks = FIXTURES
            .iter()
            .map(|text| SasFile::parse(text).expect("fixture parses").to_task())
            .chain((0..32).map(|seed| random_task(seed, params)));
        let (mut painted, mut above) = (0, 0);
        for task in tasks {
            let Some(split) = painting::split(&task) else {
                continue;
            };
            let mut h_ff = HFf::new(&split.task);
            for painting in paintings(&task) {
                if !in_conditional_effect(&task, painting.black()) {
                    continue;
                }

                painted += 1;
                let mut h_rb = RedBlack::with_painting(&task, &painting);
                let mut stack = alloc::vec![task.initial_state()];
                let mut seen = alloc::collections::BTreeSet::from([task.initial_state().0]);
                while let Some(state) = stack.pop() {
                    if let Some(plan) = h_rb.plan(&state) {
                        let ff = h_ff.value(&state).expect("same dead ends");
                        assert!(plan.cost() >= ff || !plan.is_repaired());
                        above += usize::from(plan.is_repaired() && plan.cost() > ff);
                    }

                    for op in task.applicable_operators(&state) {
                        let next = task.apply(op, &state);
                        if seen.len() < 2048 && seen.insert(next.0.clone()) {
                            stack.push(next);
                        }
                    }
                }
            }
        }

        assert!(
            painted > 0,
            "no strategy paints a conditional effect variable black"
        );
        assert!(
            above > 0,
            "black conditional effect variables never raise h_RB"
        );
    }
}
