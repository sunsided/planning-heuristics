//! Relaxed plan repair under red-black semantics.

use alloc::vec::Vec;

use sas_plus::{OpId, State};

use super::tables::{Tables, Transition};
use crate::Relaxation;

/// The red-black state and plan of one repair, with buffers reused across states.
#[derive(Clone, Debug, Default)]
pub(super) struct Repair {
    /// Whether each fact holds: every red fact reached so far and the current value of each
    /// black variable.
    holds: Vec<bool>,
    /// The current value of each black variable.
    black: Vec<u32>,
    /// Whether each fact was ever true in this repair; read for black facts only (red facts
    /// never become false, so for them this is [`Repair::holds`]). The black part of the
    /// set `F` of Domshlak, Hoffmann & Katz (2015, Figure 4).
    visited: Vec<bool>,
    /// The red-black plan so far.
    pub plan: Vec<OpId>,
    /// Its cost so far.
    cost: u64,
    /// Heads of the red effects that fire in one operator application.
    fired: Vec<usize>,
    /// Cheapest known cost to each value of the black variable being moved.
    dist: Vec<Option<u64>>,
    /// For each value of the black variable being moved, the value and operator it is
    /// reached from on the cheapest path.
    via: Vec<Option<(u32, OpId)>>,
    /// Whether each value's cheapest path is final.
    done: Vec<bool>,
    /// The operators of the path being inserted, goal end first.
    path: Vec<OpId>,
    /// Operators applied in this repair while black variables are linked.
    steps: usize,
    /// Most operators one repair may apply while black variables are linked; moving one may
    /// recursively move its causal-graph ancestors, which can blow up.
    budget: usize,
}

impl Repair {
    /// Repairs `relaxed_plan` from `state`; returns the cost of the red-black plan left in
    /// [`Repair::plan`], or `None` if a black variable cannot be moved where it is needed.
    pub fn run(
        &mut self,
        relaxation: &Relaxation,
        tables: &Tables,
        state: &State,
        relaxed_plan: &[OpId],
    ) -> Option<u64> {
        self.start(relaxation, tables, state);
        self.budget = 1024 + 64 * relaxed_plan.len();
        for &op in relaxed_plan {
            self.satisfy_black_pre(relaxation, tables, op, None)?;
            if !tables.red_pre[op].iter().all(|&f| self.holds[f]) {
                return None;
            }

            self.apply(relaxation, tables, op);
        }

        for &(b, value) in &tables.black_goal {
            self.move_black(relaxation, tables, b, value)?;
        }

        if !tables
            .black_goal
            .iter()
            .all(|&(b, value)| self.black[b] == value)
        {
            return None;
        }

        tables
            .red_goal
            .iter()
            .all(|&f| self.holds[f])
            .then_some(self.cost)
    }

    /// Sets the red-black state to `state` plus the free facts, closed under the axioms, with
    /// an empty plan.
    fn start(&mut self, relaxation: &Relaxation, tables: &Tables, state: &State) {
        self.holds.clear();
        self.holds.resize(relaxation.num_facts(), false);
        for fact in relaxation
            .state_facts(state)
            .chain(relaxation.free_facts().iter().copied())
        {
            self.holds[fact as usize] = true;
        }

        self.black.clear();
        self.black.extend(
            tables
                .black_offsets
                .iter()
                .zip(&tables.domain)
                .map(|(&offset, &n)| value_of(&self.holds, offset, n)),
        );
        self.visited.clone_from(&self.holds);
        self.plan.clear();
        self.cost = 0;
        self.steps = 0;
        close_axioms(relaxation, &mut self.holds);
    }

    /// Moves every black variable `op` requires (except `skip`) to the required value,
    /// descendants in the black causal graph first: moving a variable may move its ancestors,
    /// never its descendants. `None` if one cannot be moved, or if a requirement no longer
    /// holds afterwards.
    fn satisfy_black_pre(
        &mut self,
        relaxation: &Relaxation,
        tables: &Tables,
        op: OpId,
        skip: Option<usize>,
    ) -> Option<()> {
        let required = tables.black_pre[op]
            .iter()
            .filter(|&&(b, _)| Some(b) != skip);
        for &(b, value) in required.clone() {
            self.move_black(relaxation, tables, b, value)?;
        }

        required
            .into_iter()
            .all(|&(b, value)| self.black[b] == value)
            .then_some(())
    }

    /// Applies `op`, whose preconditions hold: its red effects whose conditions hold are added,
    /// its black effects set their variables, then the red facts are closed under the axioms.
    fn apply(&mut self, relaxation: &Relaxation, tables: &Tables, op: OpId) {
        self.fired.clear();
        self.fired.extend(
            relaxation
                .op_rules(op)
                .iter()
                .filter(|rule| !tables.is_black_fact[rule.head])
                .filter(|rule| rule.conditions.iter().all(|&c| self.holds[c]))
                .map(|rule| rule.head),
        );
        for &(b, value) in &tables.black_eff[op] {
            let offset = tables.black_offsets[b];
            self.holds[offset + self.black[b] as usize] = false;
            self.holds[offset + value as usize] = true;
            self.visited[offset + value as usize] = true;
            self.black[b] = value;
        }

        for &head in &self.fired {
            self.holds[head] = true;
        }

        close_axioms(relaxation, &mut self.holds);
        self.plan.push(op);
        self.cost = self.cost.saturating_add(tables.cost[op]);
    }

    /// Moves black variable `b` to `target` along a cheapest path of its domain transition
    /// graph whose operators' red preconditions all hold now, applying the path; `None` if
    /// there is no such path. With linked black variables, the black preconditions of each
    /// transition on `b`'s causal-graph ancestors are first satisfied recursively (the path's
    /// cost ignores them), within [`Repair::budget`].
    fn move_black(
        &mut self,
        relaxation: &Relaxation,
        tables: &Tables,
        b: usize,
        target: u32,
    ) -> Option<()> {
        if self.black[b] == target {
            return Some(());
        }

        self.cheapest_paths(tables, b);
        // Taken out: moving ancestors below reuses the buffer.
        let mut path = core::mem::take(&mut self.path);
        path.clear();
        let mut value = target;
        while value != self.black[b] {
            let (from, op) = self.via[value as usize]?;
            path.push(op);
            value = from;
        }

        for i in (0..path.len()).rev() {
            let op = path[i];
            if tables.linked {
                self.steps += 1;
                if self.steps > self.budget {
                    return None;
                }

                self.satisfy_black_pre(relaxation, tables, op, Some(b))?;
            }

            self.apply(relaxation, tables, op);
        }

        path.clear();
        self.path = path;

        Some(())
    }

    /// Dijkstra over the values of black variable `b` from its current value, through the
    /// transitions whose red preconditions hold; fills [`Repair::dist`] and [`Repair::via`].
    /// Domains are small, so the open value is found by a linear scan.
    ///
    /// With linked black variables only transitions of the projected task `Π^B` count: the
    /// black preconditions and the value reached must have been visited
    /// ([`Repair::usable_in_dag`]). Without links there are no black outside conditions and
    /// any value may be reached, as in Katz, Hoffmann & Domshlak (AAAI 2013).
    fn cheapest_paths(&mut self, tables: &Tables, b: usize) {
        let n = tables.domain[b] as usize;
        refill(&mut self.dist, n, None);
        refill(&mut self.via, n, None);
        refill(&mut self.done, n, false);
        self.dist[self.black[b] as usize] = Some(0);
        while let Some(x) = self.closest_open() {
            self.done[x] = true;
            let d = self.dist[x].expect("open values have a distance");
            for t in &tables.transitions[b] {
                let usable = t.from.is_none_or(|from| from as usize == x)
                    && t.to as usize != x
                    && tables.red_pre[t.op].iter().all(|&f| self.holds[f])
                    && (!tables.linked || self.usable_in_dag(tables, b, t));
                let next = d.saturating_add(tables.cost[t.op]);
                let to = t.to as usize;
                if usable && self.dist[to].is_none_or(|old| next < old) {
                    self.dist[to] = Some(next);
                    self.via[to] = Some((u32::try_from(x).expect("u32 values"), t.op));
                }
            }
        }
    }

    /// Whether transition `t` of black variable `b` belongs to the projected task `Π^B` of
    /// Domshlak, Hoffmann & Katz (2015, Figure 4): its black preconditions and the value it
    /// reaches were visited. With RSE-invertible black variables this keeps every
    /// precondition an ancestor is asked for reachable (their Theorem 11).
    fn usable_in_dag(&self, tables: &Tables, b: usize, t: &Transition) -> bool {
        let fact = |b: usize, value: u32| tables.black_offsets[b] + value as usize;

        self.visited[fact(b, t.to)]
            && tables.black_pre[t.op]
                .iter()
                .all(|&(c, value)| self.visited[fact(c, value)])
    }

    /// The value with the smallest distance that is not final yet.
    fn closest_open(&self) -> Option<usize> {
        (0..self.dist.len())
            .filter(|&x| !self.done[x])
            .filter_map(|x| self.dist[x].map(|d| (d, x)))
            .min()
            .map(|(_, x)| x)
    }
}

/// The value of the variable whose value 0 has fact id `offset` and whose domain has `n`
/// values, read from `holds`, where exactly one of its facts holds.
fn value_of(holds: &[bool], offset: usize, n: u32) -> u32 {
    (0..n)
        .find(|&v| holds[offset + v as usize])
        .expect("a state assigns every variable")
}

/// Fires axioms whose conditions hold until none adds a fact. Black variables never appear in
/// axioms, so only red facts are involved.
///
/// shortcut: rescans all axioms per pass like the old replay closure; index it as in
/// `Relaxation::close_axioms` once red-black runs on axiom-heavy tasks.
fn close_axioms(relaxation: &Relaxation, holds: &mut [bool]) {
    let mut changed = true;
    while changed {
        changed = false;
        for rule in relaxation.axiom_rules() {
            if !holds[rule.head] && rule.conditions.iter().all(|&c| holds[c]) {
                holds[rule.head] = true;
                changed = true;
            }
        }
    }
}

/// Makes `buffer` hold `len` copies of `value`, reusing its allocation.
fn refill<T: Clone>(buffer: &mut Vec<T>, len: usize, value: T) {
    buffer.clear();
    buffer.resize(len, value);
}
