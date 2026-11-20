//! The relevance score heuristic `h_Ξ` of Kim & Sridharan, *Relevance Score: A Landmark-Like
//! Heuristic for Planning* (arXiv:2403.07510, 2024).
//!
//! See [`RelevanceScore`].

mod tree;

use alloc::borrow::Cow;
use alloc::vec::Vec;

use sas_plus::{Fact, State, Task};

use crate::{Combine, Heuristic, Relaxation};
pub use tree::RelevanceParams;
use tree::{Label, Tree};

/// The relevance score heuristic `h_Ξ` (Kim & Sridharan 2024): the sum over all facts of the
/// probability that a randomly acting agent samples the fact while backchaining from the goal
/// under the delete relaxation.
///
/// The *backtracking tree* has the artificial action `achieveGoal` at the root, the goal facts
/// below it, below each fact the rules that achieve it (operator effects and axioms; a rule is
/// left out if one of its conditions is a fact on the path above, which would be a cycle), and
/// below each rule its conditions. The non-deterministic agent samples a sub-tree from the
/// root: at a fact it picks one achieving rule uniformly at random, at a rule it takes every
/// condition. The relevance score `Ξ(l)` of a fact is the probability that the sample contains
/// a node labelled `l`. In a state `σ`, the tree is truncated: nodes labelled with facts true
/// in `σ` are removed with everything below them. Then, with `k` the number of children,
///
/// - `Ξ(l, n) = 1` if `n` is labelled `l`;
/// - `Ξ(l, f) = Σ_c Ξ(l, c) / k` at a fact node `f` (one child is picked);
/// - `Ξ(l, a) = 1 − Π_c (1 − Ξ(l, c))` at a rule or the root (all children are taken);
///
/// and `h_Ξ(σ) = Σ_l Ξ_σ(l, root)`, so `h_Ξ = 0` exactly in goal states.
///
/// The full tree is usually far too large, so it is explored once, on construction, by the
/// paper's randomised depth-first dives (Algorithm 2, [`RelevanceParams`]); scores over the
/// explored part are lower bounds of the full scores, as unexplored parts count zero. A
/// state's score then costs one pass over the explored tree.
///
/// Not admissible: a single operator achieving three goal facts gives `h_Ξ = 3` where one step
/// suffices. Unlike the paper, a state whose goal is relaxed-unreachable is reported as a dead
/// end (`None`); the paper gives every state a score.
///
/// ```mermaid
/// graph TD
///     root["achieveGoal"] --> g1["goal fact"]
///     root --> g2["goal fact"]
///     g1 -->|"pick one, 1/k"| a1["achiever"]
///     g1 -->|"pick one, 1/k"| a2["achiever"]
///     a1 -->|"take all"| p1["precondition"]
///     a1 -->|"take all"| p2["precondition"]
/// ```
///
/// # Examples
///
/// ```
/// use planning_heuristics::{Heuristic, RelevanceParams, RelevanceScore};
/// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
///
/// // Figure 1 of the paper: the goal has three achievers; `p1` is a precondition of all of
/// // them (a landmark), `p2` of two of them.
/// let op = |pre: Vec<Fact>, eff: Fact| Operator {
///     name: "op".into(),
///     pre,
///     eff: vec![Effect::new(eff)],
///     cost: 1,
/// };
/// let (g, p1, p2) = (Fact::new(0, 1), Fact::new(1, 1), Fact::new(2, 1));
/// let task = Task {
///     variables: (0..3).map(|i| Variable::new(format!("v{i}"), 2)).collect(),
///     operators: vec![
///         op(vec![p1], g),
///         op(vec![p1, p2], g),
///         op(vec![p1, p2], g),
///         op(vec![], p1),
///         op(vec![], p2),
///     ],
///     initial: State(vec![0, 0, 0]),
///     goal: vec![g],
///     axioms: vec![],
/// };
/// let state = task.initial_state();
///
/// let mut h = RelevanceScore::new(&task);
/// let scores = h.scores(&state);
/// let score = |f: Fact| scores.iter().find(|&&(x, _)| x == f).map_or(0.0, |&(_, s)| s);
/// assert_eq!(score(g), 1.0);
/// assert_eq!(score(p1), 1.0);
/// assert!((score(p2) - 2.0 / 3.0).abs() < 1e-12);
/// assert_eq!(h.estimate(&state), Some(3)); // 2.67 rounded
/// ```
#[derive(Clone, Debug)]
pub struct RelevanceScore<'r> {
    /// The relaxation of the task, owned or shared.
    relaxation: Cow<'r, Relaxation>,
    /// The explored backtracking tree.
    tree: Tree,
    /// Working memory of one evaluation.
    work: Work,
}

/// The buffers of one evaluation.
#[derive(Clone, Debug, Default)]
struct Work {
    /// Whether each fact is true in the state (state facts and free facts).
    holds: Vec<bool>,
    /// The depth-first stack of `(node, next child)`.
    stack: Vec<(u32, u32)>,
    /// The scores of finished nodes awaiting their parent: sparse `(fact, Ξ)` lists, ascending
    /// by fact, with the node they belong to.
    done: Vec<(u32, Vec<(u32, f64)>)>,
    /// Dense accumulator per fact for combining children.
    acc: Vec<f64>,
    /// The facts touched in [`Work::acc`].
    touched: Vec<u32>,
    /// Whether each fact is in [`Work::touched`].
    is_touched: Vec<bool>,
}

impl RelevanceScore<'static> {
    /// `h_Ξ` for `task` with the paper's exploration parameters ([`RelevanceParams`]).
    #[must_use]
    pub fn new(task: &Task) -> Self {
        Self::with_params(task, &RelevanceParams::default())
    }

    /// `h_Ξ` for `task`, exploring the tree with `params`.
    #[must_use]
    pub fn with_params(task: &Task, params: &RelevanceParams) -> Self {
        let relaxation = Relaxation::new(task);
        let tree = Tree::explore(&relaxation, params);

        Self {
            relaxation: Cow::Owned(relaxation),
            tree,
            work: Work::default(),
        }
    }
}

impl<'r> RelevanceScore<'r> {
    /// `h_Ξ` over a shared `relaxation`, exploring the tree with `params`.
    #[must_use]
    pub fn with_relaxation(relaxation: &'r Relaxation, params: &RelevanceParams) -> Self {
        Self {
            tree: Tree::explore(relaxation, params),
            relaxation: Cow::Borrowed(relaxation),
            work: Work::default(),
        }
    }

    /// The number of explored tree nodes.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.tree.nodes.len()
    }

    /// Whether the whole tree was explored, so the scores are exact rather than lower bounds.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.tree.complete
    }

    /// The relevance score `Ξ_σ(l)` of every fact with a positive score in `state`, ascending by
    /// fact id.
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    #[must_use]
    pub fn scores(&mut self, state: &State) -> Vec<(Fact, f64)> {
        self.root_scores(state)
            .into_iter()
            .map(|(f, s)| (self.relaxation.fact(f as usize), s))
            .collect()
    }

    /// `h_Ξ` of `state`: the sum of all relevance scores; `None` if the goal is
    /// relaxed-unreachable from `state` (an addition to the paper).
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    #[must_use]
    pub fn value(&mut self, state: &State) -> Option<f64> {
        self.relaxation.explore(state, Combine::Max).goal_cost()?;

        Some(self.root_scores(state).iter().map(|&(_, s)| s).sum())
    }

    /// The root's sparse scores in the tree truncated at `state`.
    fn root_scores(&mut self, state: &State) -> Vec<(u32, f64)> {
        let relaxation = &*self.relaxation;
        let work = &mut self.work;
        work.holds.clear();
        work.holds.resize(relaxation.num_facts(), false);
        for fact in relaxation
            .state_facts(state)
            .chain(relaxation.free_facts().iter().copied())
        {
            work.holds[fact as usize] = true;
        }

        work.acc.clear();
        work.acc.resize(relaxation.num_facts(), 0.0);
        work.is_touched.clear();
        work.is_touched.resize(relaxation.num_facts(), false);
        evaluate(&self.tree, work)
    }
}

/// Scores every node of `tree` bottom-up by a depth-first traversal (so only the scores of
/// the children of nodes on the current path are alive) and returns the root's.
fn evaluate(tree: &Tree, work: &mut Work) -> Vec<(u32, f64)> {
    work.stack.clear();
    work.done.clear();
    work.stack.push((0, 0));
    while let Some(&mut (n, ref mut next)) = work.stack.last_mut() {
        let node = &tree.nodes[n as usize];
        let present = node.expanded && *next < node.num_children;
        if present {
            let child = node.first_child + *next;
            *next += 1;
            if !is_removed(tree, work, child) {
                work.stack.push((child, 0));
            }

            continue;
        }

        work.stack.pop();
        let scores = combine(tree, work, n);
        work.done.push((n, scores));
    }

    work.done.pop().map(|(_, s)| s).unwrap_or_default()
}

/// Whether node `n` is cut off by the truncation: a fact true in the state.
fn is_removed(tree: &Tree, work: &Work, n: u32) -> bool {
    matches!(tree.nodes[n as usize].label, Label::Fact(f) if work.holds[f as usize])
}

/// The scores of node `n` from those of its present children, which are the entries at the
/// end of [`Work::done`] whose parent is `n` (they are removed): Eq. 2 at a fact, Eq. 3 at a
/// rule or the root, and `1` for a fact's own label.
fn combine(tree: &Tree, work: &mut Work, n: u32) -> Vec<(u32, f64)> {
    let node = tree.nodes[n as usize];
    let split = work
        .done
        .iter()
        .rposition(|&(c, _)| tree.nodes[c as usize].parent != n)
        .map_or(0, |i| i + 1);
    let children: Vec<(u32, Vec<(u32, f64)>)> = work.done.drain(split..).collect();
    let fact_node = matches!(node.label, Label::Fact(_));
    let k = f64::from(node.num_children.max(1));
    for (_, scores) in &children {
        for &(f, s) in scores {
            touch(work, f);
            let a = &mut work.acc[f as usize];
            if fact_node {
                *a += s / k;
            } else {
                // Accumulates `1 − Π (1 − s)` as `acc = 1 − (1 − acc)(1 − s)`.
                *a = 1.0 - (1.0 - *a) * (1.0 - s);
            }
        }
    }

    if let Label::Fact(f) = node.label {
        touch(work, f);
        work.acc[f as usize] = 1.0;
    }

    work.touched.sort_unstable();
    let Work {
        touched,
        is_touched,
        acc,
        ..
    } = work;

    touched
        .drain(..)
        .map(|f| {
            is_touched[f as usize] = false;
            (f, core::mem::take(&mut acc[f as usize]))
        })
        .filter(|&(_, s)| s > 0.0)
        .collect()
}

/// Records that fact `f` has a value in [`Work::acc`].
fn touch(work: &mut Work, f: u32) {
    if !core::mem::replace(&mut work.is_touched[f as usize], true) {
        work.touched.push(f);
    }
}

impl Heuristic<State> for RelevanceScore<'_> {
    /// `h_Ξ` rounded to the nearest integer, as the paper does for LAMA.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a sum of probabilities, non-negative and far below u32::MAX"
    )]
    fn estimate(&mut self, state: &State) -> Option<u32> {
        self.value(state).map(|v| (v + 0.5) as u32)
    }
}
