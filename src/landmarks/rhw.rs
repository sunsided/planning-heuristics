//! Landmark extraction by back-chaining from the goal (Richter, Helmert & Westphal 2008),
//! with disjunctive and domain-transition-graph landmarks.

use alloc::vec::Vec;

use sas_plus::State;

use super::ordered::{Builder, OrderedLandmarks, OrderingKind};
use crate::Relaxation;
use crate::relaxation::Rule;

/// Largest disjunctive landmark kept, in facts (as in LAMA).
const MAX_DISJUNCTION: usize = 4;

impl Relaxation {
    /// The landmarks of `state` by back-chaining from the goal (Richter, Helmert & Westphal
    /// 2008); `None` if a landmark turns out unreachable (a dead end).
    ///
    /// Starting from the goal facts, for each landmark `L` (a fact, or a disjunction of facts)
    /// that does not hold in `state`:
    ///
    /// 1. its *possible first achievers* are the rules adding a fact of `L` whose conditions
    ///    are relaxed-reachable from `state` without any fact of `L`. Every plan first
    ///    achieves `L` with one of them;
    /// 2. a condition shared by all of them is a fact landmark, ordered greedy-necessarily
    ///    before `L` (naturally if an axiom is among them, since an axiom's conditions hold in
    ///    the same state as its head);
    /// 3. if every first achiever has a condition on some variable `v` and those conditions
    ///    name 2 to 4 values, the disjunction of these facts is a landmark, ordered the same way;
    /// 4. for a fact landmark `v = d` of a state variable, every value of `v` that each path
    ///    from `v`'s value in `state` to `d` in the domain transition graph passes through
    ///    is a fact landmark, ordered naturally before `L`. The graph only has the transitions
    ///    of rules that are possible before `L`.
    ///
    /// Each new landmark is back-chained in turn. Finally disjunctions that contain a fact
    /// landmark are dropped (they hold whenever it does). Landmarks that hold in `state`,
    /// including the free defaults of derived variables, are kept but not back-chained.
    ///
    /// Unlike [`Relaxation::landmarks`] (`h^1`, complete for the causal landmarks of the
    /// delete relaxation), back-chaining finds disjunctions and landmarks of the domain
    /// transition graphs, which the relaxation does not see, but can miss fact landmarks that
    /// only the `h^1` fixpoint proves. Each relaxed reachability check is `O(rules)`, one per
    /// landmark.
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
    /// // `b` can be set from `a = 1` or from `a = 2`: neither is a fact landmark, their
    /// // disjunction is.
    /// let op = |name: &str, pre: Vec<Fact>, eff: Fact| Operator {
    ///     name: name.into(),
    ///     pre,
    ///     eff: vec![Effect::new(eff)],
    ///     cost: 1,
    /// };
    /// let task = Task {
    ///     variables: vec![Variable::new("a", 3), Variable::new("b", 2)],
    ///     operators: vec![
    ///         op("a1", vec![Fact::new(0, 0)], Fact::new(0, 1)),
    ///         op("a2", vec![Fact::new(0, 0)], Fact::new(0, 2)),
    ///         op("b-from-1", vec![Fact::new(0, 1)], Fact::new(1, 1)),
    ///         op("b-from-2", vec![Fact::new(0, 2)], Fact::new(1, 1)),
    ///     ],
    ///     initial: State(vec![0, 0]),
    ///     goal: vec![Fact::new(1, 1)],
    ///     axioms: vec![],
    /// };
    /// let relaxation = Relaxation::new(&task);
    /// let graph = relaxation.rhw_landmarks(&task.initial_state()).unwrap();
    ///
    /// let id = |var, value| relaxation.fact_id(Fact::new(var, value));
    /// assert!(graph.index_of(&[id(0, 1), id(0, 2)]).is_some());
    /// assert!(graph.index_of(&[id(0, 1)]).is_none());
    /// ```
    #[must_use]
    pub fn rhw_landmarks(&self, state: &State) -> Option<OrderedLandmarks> {
        let mut chain = Chain::new(self, state);
        for &g in self.goal_facts() {
            chain.push(alloc::vec![g]);
        }

        while let Some(i) = chain.queue.pop() {
            chain.back_chain(i)?;
        }

        let facts: Vec<usize> = (0..chain.builder.len())
            .map(|i| chain.builder.get(i))
            .filter(|l| l.len() == 1)
            .map(|l| l[0])
            .collect();

        Some(
            chain
                .builder
                .finish(|l| l.len() > 1 && l.iter().any(|f| facts.contains(f))),
        )
    }
}

/// The state of one back-chaining run.
struct Chain<'a> {
    /// The relaxation.
    relaxation: &'a Relaxation,
    /// The state's value per variable.
    values: &'a [u32],
    /// Whether each fact holds from the start: the state's facts and the free defaults.
    holds: Vec<bool>,
    /// The landmarks found so far.
    builder: Builder,
    /// Landmarks (insertion indices) still to back-chain.
    queue: Vec<usize>,
    /// Relaxed reachability of the current exploration, per fact.
    reached: Vec<bool>,
    /// Unreached conditions per rule in the current exploration.
    missing: Vec<u32>,
    /// Exploration stack.
    stack: Vec<usize>,
}

impl<'a> Chain<'a> {
    /// A run from `state` with no landmarks yet.
    fn new(relaxation: &'a Relaxation, state: &'a State) -> Self {
        let mut holds = alloc::vec![false; relaxation.num_facts()];
        for f in relaxation
            .state_facts(state)
            .chain(relaxation.free_facts().iter().copied())
        {
            holds[f as usize] = true;
        }

        Self {
            relaxation,
            values: &state.0,
            holds,
            builder: Builder::default(),
            queue: Vec::new(),
            reached: Vec::new(),
            missing: Vec::new(),
            stack: Vec::new(),
        }
    }

    /// Adds the landmark `facts`; queues it if new. Returns its insertion index.
    fn push(&mut self, facts: Vec<usize>) -> usize {
        let (i, new) = self.builder.insert(facts);
        if new {
            self.queue.push(i);
        }

        i
    }

    /// Back-chains from landmark `i`; `None` if it has no possible first achiever.
    fn back_chain(&mut self, i: usize) -> Option<()> {
        let landmark = self.builder.get(i).to_vec();
        if landmark.iter().any(|&f| self.holds[f]) {
            return Some(());
        }

        self.explore_without(&landmark);
        let achievers: Vec<&Rule> = self
            .relaxation
            .rules()
            .iter()
            .filter(|r| landmark.binary_search(&r.head).is_ok() && self.enabled(r))
            .collect();
        if achievers.is_empty() {
            return None;
        }

        let kind = if achievers.iter().all(|r| r.op.is_some()) {
            OrderingKind::GreedyNecessary
        } else {
            OrderingKind::Natural
        };
        for c in shared_conditions(&achievers) {
            let j = self.push(alloc::vec![c]);
            self.builder.order(j, i, kind);
        }

        for disjunction in self.disjunctions(&achievers) {
            let j = self.push(disjunction);
            self.builder.order(j, i, kind);
        }

        if let [fact] = landmark[..] {
            for value in self.dtg_landmarks(fact) {
                let j = self.push(alloc::vec![value]);
                self.builder.order(j, i, OrderingKind::Natural);
            }
        }

        Some(())
    }

    /// Relaxed reachability from the start facts with every fact of `excluded` unreachable;
    /// fills [`Chain::reached`] and [`Chain::missing`].
    fn explore_without(&mut self, excluded: &[usize]) {
        let relaxation = self.relaxation;
        self.reached.clear();
        self.reached.resize(relaxation.num_facts(), false);
        self.missing.clear();
        self.missing.extend(
            relaxation
                .rules()
                .iter()
                .map(|r| u32::try_from(r.conditions.len()).expect("fewer than 2^32 conditions")),
        );
        for f in 0..relaxation.num_facts() {
            if self.holds[f] {
                self.reach(f, excluded);
            }
        }

        for &r in relaxation.unconditioned() {
            self.reach(relaxation.rules()[r as usize].head, excluded);
        }

        while let Some(f) = self.stack.pop() {
            for &r in relaxation.watchers(f) {
                let r = r as usize;
                self.missing[r] -= 1;
                if self.missing[r] == 0 {
                    self.reach(relaxation.rules()[r].head, excluded);
                }
            }
        }
    }

    /// Marks `fact` reached unless it is excluded or already reached.
    fn reach(&mut self, fact: usize, excluded: &[usize]) {
        if excluded.binary_search(&fact).is_err()
            && !core::mem::replace(&mut self.reached[fact], true)
        {
            self.stack.push(fact);
        }
    }

    /// Whether every condition of `rule` was reached by the last exploration.
    fn enabled(&self, rule: &Rule) -> bool {
        rule.conditions.iter().all(|&c| self.reached[c])
    }

    /// The disjunctive landmarks among the conditions of `achievers`: per variable on which
    /// every achiever has a condition, the facts they name, if 2 to [`MAX_DISJUNCTION`] and
    /// none holds from the start.
    fn disjunctions(&self, achievers: &[&Rule]) -> Vec<Vec<usize>> {
        let var = |c: usize| self.relaxation.fact(c).var;
        let mut vars: Vec<usize> = achievers[0].conditions.iter().map(|&c| var(c)).collect();
        vars.dedup();

        vars.into_iter()
            .filter_map(|v| {
                let mut facts = Vec::new();
                for rule in achievers {
                    let before = facts.len();
                    facts.extend(rule.conditions.iter().copied().filter(|&c| var(c) == v));
                    if facts.len() == before {
                        return None;
                    }
                }

                facts.sort_unstable();
                facts.dedup();
                let size_ok = (2..=MAX_DISJUNCTION).contains(&facts.len());
                (size_ok && facts.iter().all(|&f| !self.holds[f])).then_some(facts)
            })
            .collect()
    }

    /// The values of `fact`'s variable on every path from its start value to `fact` in the
    /// domain transition graph of the rules enabled in the last exploration (the start value
    /// and `fact` excluded). Empty for a derived variable, or if no path exists.
    fn dtg_landmarks(&self, fact: usize) -> Vec<usize> {
        let relaxation = self.relaxation;
        let target = relaxation.fact(fact);
        let Some(&start) = self.values.get(target.var) else {
            return Vec::new();
        };
        let first = relaxation.fact_id(sas_plus::Fact::new(target.var, 0));
        let arcs: Vec<(Option<usize>, usize)> = relaxation
            .rules()
            .iter()
            .filter(|r| r.op.is_some() && relaxation.fact(r.head).var == target.var)
            .filter(|r| r.head == fact || self.enabled(r))
            .map(|r| {
                let from = r
                    .conditions
                    .iter()
                    .copied()
                    .find(|&c| relaxation.fact(c).var == target.var);
                (from, r.head)
            })
            .collect();
        if arcs.is_empty() {
            return Vec::new();
        }

        let start = first + start as usize;
        let size = arcs
            .iter()
            .map(|&(_, to)| to)
            .chain([start, fact])
            .max()
            .map_or(0, |m| m + 1);
        if !dtg_reaches(&arcs, start, fact, None, size) {
            return Vec::new();
        }

        (first..size)
            .filter(|&v| v != start && v != fact && relaxation.fact(v).var == target.var)
            .filter(|&v| !dtg_reaches(&arcs, start, fact, Some(v), size))
            .collect()
    }
}

/// Whether `to` is reachable from `from` along `arcs` (`None` source: any value) without
/// entering `avoid`. Node ids are fact ids below `size`.
fn dtg_reaches(
    arcs: &[(Option<usize>, usize)],
    from: usize,
    to: usize,
    avoid: Option<usize>,
    size: usize,
) -> bool {
    let mut seen = alloc::vec![false; size];
    let mut stack = alloc::vec![from];
    seen[from] = true;
    while let Some(x) = stack.pop() {
        if x == to {
            return true;
        }

        for &(source, next) in arcs {
            if source.is_none_or(|s| s == x) && Some(next) != avoid && !seen[next] {
                seen[next] = true;
                stack.push(next);
            }
        }
    }

    false
}

/// The conditions every rule of `rules` has, ascending.
fn shared_conditions(rules: &[&Rule]) -> Vec<usize> {
    let mut shared = rules[0].conditions.clone();
    for rule in &rules[1..] {
        shared.retain(|c| rule.conditions.binary_search(c).is_ok());
    }

    shared
}
