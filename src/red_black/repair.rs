//! Relaxed plan repair under red-black semantics.

use alloc::vec::Vec;

use sas_plus::{OpId, State};

use super::tables::Tables;
use crate::Relaxation;

/// The red-black state and plan of one repair, with buffers reused across states.
#[derive(Clone, Debug, Default)]
pub(super) struct Repair {
    /// Whether each fact holds: every red fact reached so far and the current value of each
    /// black variable.
    holds: Vec<bool>,
    /// The current value of each black variable.
    black: Vec<u32>,
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
        for &op in relaxed_plan {
            for &(b, value) in &tables.black_pre[op] {
                self.move_black(relaxation, tables, b, value)?;
            }

            if !tables.red_pre[op].iter().all(|&f| self.holds[f]) {
                return None;
            }

            self.apply(relaxation, tables, op);
        }

        for &(b, value) in &tables.black_goal {
            self.move_black(relaxation, tables, b, value)?;
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
        self.plan.clear();
        self.cost = 0;
        close_axioms(relaxation, &mut self.holds);
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
    /// there is no such path.
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
        self.path.clear();
        let mut value = target;
        while value != self.black[b] {
            let (from, op) = self.via[value as usize]?;
            self.path.push(op);
            value = from;
        }

        for i in (0..self.path.len()).rev() {
            let op = self.path[i];
            self.apply(relaxation, tables, op);
        }

        Some(())
    }

    /// Dijkstra over the values of black variable `b` from its current value, through the
    /// transitions whose red preconditions hold; fills [`Repair::dist`] and [`Repair::via`].
    /// Domains are small, so the open value is found by a linear scan.
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
                    && tables.red_pre[t.op].iter().all(|&f| self.holds[f]);
                let next = d.saturating_add(tables.cost[t.op]);
                let to = t.to as usize;
                if usable && self.dist[to].is_none_or(|old| next < old) {
                    self.dist[to] = Some(next);
                    self.via[to] = Some((u32::try_from(x).expect("u32 values"), t.op));
                }
            }
        }
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
