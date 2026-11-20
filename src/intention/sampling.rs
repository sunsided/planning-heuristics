//! Supporter sampling (Rosa, Honorio, Lipovetzky & Sardiña 2026, Appendix B, Algorithm 2,
//! after Wilken et al. 2024): random relaxed plans backchained level by level through a
//! relaxed planning graph.

use alloc::vec::Vec;

use sas_plus::State;

use crate::rng::SplitMix64;
use crate::{Combine, Relaxation};

/// The relaxed planning graph data the sampler needs, from one state.
pub(super) struct Graph<'r> {
    /// The relaxation.
    relaxation: &'r Relaxation,
    /// Whether each fact is true in the state (or free).
    initial: Vec<bool>,
    /// The layer each fact first appears in; `None` if unreachable. Every rule, axioms
    /// included, adds one layer.
    level: Vec<Option<u32>>,
    /// The `h_add` value of each rule: its cost plus its conditions' `h_add`; `None` if
    /// unreachable.
    rule_add: Vec<Option<u64>>,
    /// The layer of each rule: the largest layer of its conditions; `None` if unreachable.
    rule_level: Vec<Option<u32>>,
    /// The reachable rules per head fact.
    by_head: Vec<Vec<u32>>,
}

impl<'r> Graph<'r> {
    /// The graph of `relaxation` from `state`.
    pub fn new(relaxation: &'r Relaxation, state: &State) -> Self {
        let mut initial = alloc::vec![false; relaxation.num_facts()];
        for fact in relaxation
            .state_facts(state)
            .chain(relaxation.free_facts().iter().copied())
        {
            initial[fact as usize] = true;
        }

        let level = layers(relaxation, &initial);
        let add = relaxation.explore(state, Combine::Add);
        let rules = relaxation.rules();
        let rule_level: Vec<Option<u32>> = rules
            .iter()
            .map(|r| {
                r.conditions
                    .iter()
                    .try_fold(0, |acc, &c| level[c].map(|l| acc.max(l)))
            })
            .collect();
        let rule_add = rules
            .iter()
            .map(|r| {
                r.conditions
                    .iter()
                    .try_fold(r.cost, |acc, &c| add.cost(c).map(|x| acc.saturating_add(x)))
            })
            .collect();
        let mut by_head = alloc::vec![Vec::new(); relaxation.num_facts()];
        for (i, r) in rules.iter().enumerate() {
            if rule_level[i].is_some() {
                by_head[r.head].push(u32::try_from(i).expect("fewer than 2^32 rules"));
            }
        }

        Self {
            relaxation,
            initial,
            level,
            rule_add,
            rule_level,
            by_head,
        }
    }

    /// Whether `fact` holds in the state.
    pub fn is_initial(&self, fact: usize) -> bool {
        self.initial[fact]
    }

    /// Whether `fact` is relaxed-reachable.
    pub fn is_reachable(&self, fact: usize) -> bool {
        self.level[fact].is_some()
    }

    /// The source a rule stands for: its operator, or `num_operators + rule index` for an
    /// axiom.
    pub fn source(&self, rule: usize) -> usize {
        self.relaxation.rules()[rule]
            .op
            .unwrap_or(self.relaxation.num_operators() + rule)
    }

    /// The number of sources: operators plus rules (axioms are numbered by rule index).
    pub fn num_sources(&self) -> usize {
        self.relaxation.num_operators() + self.relaxation.rules().len()
    }

    /// Samples `n` supporter sets for `targets` (facts not in the state). Each is a set of
    /// sources, ascending.
    pub fn sample(&self, targets: &[usize], n: usize, rng: &mut SplitMix64) -> Vec<Vec<usize>> {
        let levels = targets
            .iter()
            .filter_map(|&t| self.level[t])
            .max()
            .unwrap_or(0);
        let mut count = alloc::vec![0_usize; self.num_sources()];
        let mut sampler = Sample::new(self.relaxation.num_facts());

        (0..n)
            .map(|_| sampler.run(self, targets, levels, &mut count, rng))
            .collect()
    }

    /// The candidate supporters of `p` at layer `t` (Algorithm 2, lines 13–15): the reachable
    /// rules achieving it with a layer of at most `t` (any reachable one if none is that
    /// low), then those of minimum `h_add`, then those whose source was chosen least often.
    fn candidates(&self, p: usize, t: u32, count: &[usize]) -> Vec<u32> {
        let rules = &self.by_head[p];
        let low: Vec<u32> = rules
            .iter()
            .copied()
            .filter(|&r| self.rule_level[r as usize].is_some_and(|l| l <= t))
            .collect();
        let pool = if low.is_empty() { rules.clone() } else { low };
        let best = pool.iter().filter_map(|&r| self.rule_add[r as usize]).min();
        let cheapest: Vec<u32> = pool
            .into_iter()
            .filter(|&r| self.rule_add[r as usize] == best)
            .collect();
        let fewest = cheapest
            .iter()
            .map(|&r| count[self.source(r as usize)])
            .min();

        cheapest
            .into_iter()
            .filter(|&r| Some(count[self.source(r as usize)]) == fewest)
            .collect()
    }

    /// The heads `rule`'s source certainly adds when applied through `rule`: every effect of
    /// its operator whose conditions are among `rule`'s (an axiom adds only its head).
    fn certain_adds(&self, rule: usize) -> Vec<usize> {
        let rules = self.relaxation.rules();
        let chosen = &rules[rule];
        match chosen.op {
            Some(op) => self
                .relaxation
                .op_rules(op)
                .iter()
                .filter(|r| r.conditions.iter().all(|c| chosen.conditions.contains(c)))
                .map(|r| r.head)
                .collect(),
            None => alloc::vec![chosen.head],
        }
    }
}

/// The per-sample sets of Algorithm 2, reused across samples.
struct Sample {
    /// Whether each fact is in `C`, the facts still to support at this layer.
    in_c: Vec<bool>,
    /// The facts of `C` (may hold entries removed from `in_c`).
    c: Vec<usize>,
    /// Whether each fact is in `newC`, the facts for the next lower layer.
    in_new: Vec<bool>,
    /// The facts of `newC`.
    new: Vec<usize>,
    /// Whether each fact was supported (`found`).
    found: Vec<bool>,
}

impl Sample {
    /// Empty sets over `facts` facts.
    fn new(facts: usize) -> Self {
        Self {
            in_c: alloc::vec![false; facts],
            c: Vec::new(),
            in_new: alloc::vec![false; facts],
            new: Vec::new(),
            found: alloc::vec![false; facts],
        }
    }

    /// One run of the outer loop body of Algorithm 2: supports `targets` from layer `levels`
    /// down to 0 and returns the chosen sources, ascending. `count` counts choices over all
    /// samples (line 19).
    fn run(
        &mut self,
        graph: &Graph<'_>,
        targets: &[usize],
        levels: u32,
        count: &mut [usize],
        rng: &mut SplitMix64,
    ) -> Vec<usize> {
        self.found.fill(false);
        for &t in targets {
            self.in_c[t] = true;
            self.c.push(t);
        }

        let mut sups = Vec::new();
        for t in (0..=levels).rev() {
            while let Some(p) = self.c.pop() {
                if !core::mem::replace(&mut self.in_c[p], false) {
                    continue;
                }

                let candidates = graph.candidates(p, t, count);
                if candidates.is_empty() {
                    continue;
                }

                let rule = candidates[rng.below(candidates.len())] as usize;
                self.support(graph, p, rule, &mut sups, count);
            }

            for p in self.new.drain(..) {
                if core::mem::replace(&mut self.in_new[p], false) && !self.in_c[p] {
                    self.in_c[p] = true;
                    self.c.push(p);
                }
            }
        }

        for p in self.c.drain(..) {
            self.in_c[p] = false;
        }

        sups.sort_unstable();
        sups.dedup();
        sups
    }

    /// Lines 16–28: marks `p` found, records `rule`'s source, queues its new conditions for
    /// the next layer and drops the facts it certainly adds from both queues (except its own
    /// conditions).
    fn support(
        &mut self,
        graph: &Graph<'_>,
        p: usize,
        rule: usize,
        sups: &mut Vec<usize>,
        count: &mut [usize],
    ) {
        self.found[p] = true;
        let source = graph.source(rule);
        sups.push(source);
        count[source] += 1;
        for &n in &graph.relaxation.rules()[rule].conditions {
            if !graph.initial[n] && !self.found[n] && !self.in_c[n] && !self.in_new[n] {
                self.in_new[n] = true;
                self.new.push(n);
            }
        }

        // A fact the supporter both needs and adds must hold before it, so it stays queued
        // (the paper's lines 25-28 would drop it).
        let conditions = &graph.relaxation.rules()[rule].conditions;
        for r in graph.certain_adds(rule) {
            if !conditions.contains(&r) {
                self.in_c[r] = false;
                self.in_new[r] = false;
            }
        }
    }
}

/// The layer of each fact from the facts `initial`: a sweep to the fixpoint where every rule
/// puts its head one layer above its highest condition.
fn layers(relaxation: &Relaxation, initial: &[bool]) -> Vec<Option<u32>> {
    let mut level: Vec<Option<u32>> = initial.iter().map(|&i| i.then_some(0)).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for rule in relaxation.rules() {
            let Some(at) = rule
                .conditions
                .iter()
                .try_fold(0, |acc, &c| level[c].map(|l| acc.max(l)))
            else {
                continue;
            };

            if level[rule.head].is_none_or(|l| at + 1 < l) {
                level[rule.head] = Some(at + 1);
                changed = true;
            }
        }
    }

    level
}
