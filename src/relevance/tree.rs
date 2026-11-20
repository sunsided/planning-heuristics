//! The backtracking tree of Kim & Sridharan (2024, Section 3.4) and its partial exploration
//! (their Algorithm 2).

use alloc::vec::Vec;

use crate::Relaxation;
use crate::rng::SplitMix64;

/// What a node of the tree stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Label {
    /// The artificial `achieveGoal` action, the root; its children are the goal facts.
    Root,
    /// A fact (by [`Relaxation::fact_id`]); its children are the rules achieving it.
    Fact(u32),
    /// A rule of the relaxation (an operator effect or an axiom), an "action" of the paper;
    /// its children are its conditions.
    Rule(u32),
}

/// One node of the explored tree.
#[derive(Clone, Copy, Debug)]
pub(super) struct Node {
    /// The label.
    pub label: Label,
    /// The parent; `u32::MAX` for the root.
    pub parent: u32,
    /// The first child; children are stored contiguously. Meaningless until expanded.
    pub first_child: u32,
    /// The number of children in the full tree; `0` until expanded.
    pub num_children: u32,
    /// Whether the children were added to the explored tree.
    pub expanded: bool,
    /// The probability `ξ(n)` that the agent samples this node (their Section 3.7).
    pub xi: f64,
}

/// Parameters of the exploration (their Algorithm 2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelevanceParams {
    /// Explore at least this many nodes (the paper: 100 000) unless the full tree is smaller.
    pub min_nodes: usize,
    /// Then continue while the frontier holds more than this fraction of the explored `ξ`
    /// mass (the paper's `ρ`, 0.2).
    pub rho: f64,
    /// Stop at this many nodes whatever the frontier holds (not in the paper; it bounds
    /// memory).
    pub max_nodes: usize,
    /// The seed of the random dives.
    pub seed: u64,
}

impl Default for RelevanceParams {
    /// The paper's `min_nodes = 100 000` and `ρ = 0.2`, at most 400 000 nodes, seed 0.
    fn default() -> Self {
        Self {
            min_nodes: 100_000,
            rho: 0.2,
            max_nodes: 400_000,
            seed: 0,
        }
    }
}

/// The explored part of the tree, root at index 0. Children always have larger indices than
/// their parent.
#[derive(Clone, Debug)]
pub(super) struct Tree {
    /// The nodes.
    pub nodes: Vec<Node>,
    /// Whether every node is expanded (the tree is the full tree).
    pub complete: bool,
}

impl Tree {
    /// Explores the tree of `relaxation` with Algorithm 2 under `params`.
    pub fn explore(relaxation: &Relaxation, params: &RelevanceParams) -> Self {
        let by_head = rules_by_head(relaxation);
        let mut tree = Self {
            nodes: alloc::vec![Node {
                label: Label::Root,
                parent: u32::MAX,
                first_child: 0,
                num_children: 0,
                expanded: false,
                xi: 1.0,
            }],
            complete: false,
        };
        let mut frontier = Frontier::default();
        frontier.insert(0, 1.0);
        let mut rng = SplitMix64(params.seed);
        let mut total = 1.0;
        while !frontier.is_empty() && tree.nodes.len() < params.max_nodes {
            if tree.nodes.len() >= params.min_nodes && frontier.mass() <= params.rho * total {
                break;
            }

            let start = frontier.choose(&mut rng);
            let before = tree.nodes.len();
            tree.dive(relaxation, &by_head, &mut frontier, &mut rng, start);
            total += tree.nodes[before..].iter().map(|n| n.xi).sum::<f64>();
        }

        tree.complete = frontier.is_empty();
        tree
    }

    /// One depth-first dive from frontier node `n`: expand it, then continue at one child
    /// chosen with probability proportional to `ξ`, until a node without children. A node
    /// without children leaves the frontier (the paper's loop would pick it forever).
    fn dive(
        &mut self,
        relaxation: &Relaxation,
        by_head: &[Vec<u32>],
        frontier: &mut Frontier,
        rng: &mut SplitMix64,
        mut n: u32,
    ) {
        loop {
            frontier.remove(n);
            let children = self.expand(relaxation, by_head, n);
            if children.is_empty() {
                return;
            }

            let mass: f64 = children.clone().map(|c| self.nodes[c as usize].xi).sum();
            for c in children.clone() {
                frontier.insert(c, self.nodes[c as usize].xi);
            }

            let mut pick = f64::from(u32::try_from(rng.next() >> 40).expect("24 bits"))
                / f64::from(1_u32 << 24)
                * mass;
            n = children.clone().last().expect("not empty");
            for c in children {
                pick -= self.nodes[c as usize].xi;
                if pick < 0.0 {
                    n = c;
                    break;
                }
            }
        }
    }

    /// Adds the children of node `n` to the tree and returns their index range.
    fn expand(
        &mut self,
        relaxation: &Relaxation,
        by_head: &[Vec<u32>],
        n: u32,
    ) -> core::ops::Range<u32> {
        let node = self.nodes[n as usize];
        let labels = self.child_labels(relaxation, by_head, n);
        let first = u32::try_from(self.nodes.len()).expect("fewer than 2^32 nodes");
        let k = u32::try_from(labels.len()).expect("fewer than 2^32 children");
        let xi = match node.label {
            Label::Fact(_) => node.xi / f64::from(k.max(1)),
            Label::Root | Label::Rule(_) => node.xi,
        };
        self.nodes.extend(labels.into_iter().map(|label| Node {
            label,
            parent: n,
            first_child: 0,
            num_children: 0,
            expanded: false,
            xi,
        }));
        let node = &mut self.nodes[n as usize];
        node.expanded = true;
        node.first_child = first;
        node.num_children = k;

        first..first + k
    }

    /// The labels of the children of `n` in the full tree: the goal facts of the root, the
    /// distinct conditions of a rule, and the rules achieving a fact except those with a
    /// condition labelling a fact on the fact's path (the fact included), which would be a
    /// cycle (their Section 3.4).
    fn child_labels(&self, relaxation: &Relaxation, by_head: &[Vec<u32>], n: u32) -> Vec<Label> {
        let to_fact = |f: usize| Label::Fact(u32::try_from(f).expect("fewer than 2^32 facts"));
        match self.nodes[n as usize].label {
            Label::Root => relaxation
                .goal_facts()
                .iter()
                .map(|&g| to_fact(g))
                .collect(),
            Label::Rule(r) => {
                let mut conditions = relaxation.rules()[r as usize].conditions.clone();
                conditions.sort_unstable();
                conditions.dedup();
                conditions.into_iter().map(to_fact).collect()
            }
            Label::Fact(f) => {
                let path = self.path_facts(n);
                by_head[f as usize]
                    .iter()
                    .filter(|&&r| {
                        relaxation.rules()[r as usize]
                            .conditions
                            .iter()
                            .all(|c| path.binary_search(c).is_err())
                    })
                    .map(|&r| Label::Rule(r))
                    .collect()
            }
        }
    }

    /// The facts labelling `n` and its ancestors, ascending.
    fn path_facts(&self, mut n: u32) -> Vec<usize> {
        let mut facts = Vec::new();
        while n != u32::MAX {
            let node = &self.nodes[n as usize];
            if let Label::Fact(f) = node.label {
                facts.push(f as usize);
            }

            n = node.parent;
        }

        facts.sort_unstable();
        facts
    }
}

/// The rules of `relaxation` per head fact.
fn rules_by_head(relaxation: &Relaxation) -> Vec<Vec<u32>> {
    let mut by_head = alloc::vec![Vec::new(); relaxation.num_facts()];
    for (r, rule) in relaxation.rules().iter().enumerate() {
        by_head[rule.head].push(u32::try_from(r).expect("fewer than 2^32 rules"));
    }

    by_head
}

/// The unexpanded nodes with their `ξ`, for sampling proportional to `ξ` in `O(log n)` (a
/// Fenwick tree over node indices, regrown by doubling).
#[derive(Debug, Default)]
struct Frontier {
    /// The Fenwick tree of the weights, 1-based.
    fenwick: Vec<f64>,
    /// The weight of each node index; `0` if not in the frontier.
    weight: Vec<f64>,
    /// The number of nodes in the frontier.
    len: usize,
}

impl Frontier {
    /// Whether the frontier is empty.
    fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The total `ξ` in the frontier (the paper's `sumxi(frontier)`).
    fn mass(&self) -> f64 {
        self.prefix(self.weight.len())
    }

    /// Adds node `n` with weight `w`.
    fn insert(&mut self, n: u32, w: f64) {
        let i = n as usize;
        if i >= self.weight.len() {
            self.grow(i + 1);
        }

        self.add(i, w - self.weight[i]);
        if self.weight[i] == 0.0 {
            self.len += 1;
        }

        self.weight[i] = w;
    }

    /// Removes node `n` if present.
    fn remove(&mut self, n: u32) {
        let i = n as usize;
        if i < self.weight.len() && self.weight[i] != 0.0 {
            self.add(i, -self.weight[i]);
            self.weight[i] = 0.0;
            self.len -= 1;
        }
    }

    /// A frontier node drawn with probability proportional to its weight.
    fn choose(&self, rng: &mut SplitMix64) -> u32 {
        let total = self.prefix(self.weight.len());
        let mut pick = f64::from(u32::try_from(rng.next() >> 40).expect("24 bits"))
            / f64::from(1_u32 << 24)
            * total;
        let mut pos = 0;
        let mut step = self.fenwick.len().next_power_of_two();
        while step > 0 {
            let next = pos + step;
            if next < self.fenwick.len() && self.fenwick[next] <= pick {
                pick -= self.fenwick[next];
                pos = next;
            }

            step /= 2;
        }

        // Rounding may land on a removed node: take the next one present.
        let found = (pos..self.weight.len())
            .chain(0..pos)
            .find(|&i| self.weight[i] != 0.0)
            .expect("the frontier is not empty");

        u32::try_from(found).expect("fewer than 2^32 nodes")
    }

    /// Adds `delta` to the weight at index `i` in the Fenwick tree.
    fn add(&mut self, i: usize, delta: f64) {
        let mut j = i + 1;
        while j < self.fenwick.len() {
            self.fenwick[j] += delta;
            j += j & j.wrapping_neg();
        }
    }

    /// The sum of the weights at indices `0..i`.
    fn prefix(&self, i: usize) -> f64 {
        let mut sum = 0.0;
        let mut j = i;
        while j > 0 {
            sum += self.fenwick[j];
            j -= j & j.wrapping_neg();
        }

        sum
    }

    /// Grows the capacity to at least `len` indices and rebuilds the Fenwick tree.
    fn grow(&mut self, len: usize) {
        let capacity = len.next_power_of_two().max(16);
        self.weight.resize(capacity, 0.0);
        self.fenwick = alloc::vec![0.0; capacity + 1];
        for i in 0..capacity {
            let w = self.weight[i];
            if w != 0.0 {
                self.add(i, w);
            }
        }
    }
}
