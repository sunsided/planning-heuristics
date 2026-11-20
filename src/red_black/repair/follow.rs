//! Red facts following (Domshlak, Hoffmann & Katz, AIJ 2015, Section 5.2, Figure 6): commit
//! only to the red facts a relaxed plan uses, not to its operators or their order.

use alloc::vec::Vec;

use sas_plus::{OpId, State};

use super::{Repair, budget};
use crate::Relaxation;
use crate::red_black::tables::Tables;

/// The per-run state of red facts following, kept for its allocations.
#[derive(Clone, Debug, Default)]
pub(in crate::red_black) struct Buffers {
    /// Whether each fact is in `R+`: a red precondition of the relaxed plan or a red goal.
    in_target: Vec<bool>,
    /// The facts of `R+`, ascending.
    target: Vec<usize>,
    /// The operators with an effect in `R+`: the only ones the main loop picks.
    candidates: Vec<OpId>,
    /// Whether each position of the relaxed plan was applied.
    used: Vec<bool>,
    /// The values of the black variable being explored by [`Repair::update`].
    stack: Vec<u32>,
}

impl Repair {
    /// Red facts following from `state` with the targets of `relaxed_plan`; returns the cost
    /// of the red-black plan left in [`Repair::plan`], or `None` if it gets stuck (outside
    /// the tractable fragment) or exceeds the budget.
    ///
    /// The main loop repeatedly picks an operator whose red preconditions hold, whose black
    /// preconditions are achievable (in `B`), and that adds a missing fact of `R+`; it moves
    /// the black variables there (`ACHIEVE`), applies it and recomputes `B` (`UPDATE`). Once
    /// `R+` holds, the black variables move to the goal. Choice point (i) of the paper's
    /// Section 5.3.2: prefer operators whose black moves cannot delete an achieved `R+` fact
    /// under real semantics, then the fewest estimated black move costs (all-pairs
    /// distances, ignoring outside conditions), then the lowest id.
    ///
    /// With axioms a derived target fact may have no achieving operator; then, a step the
    /// paper does not need, the first unapplied operator of the relaxed plan that is
    /// achievable is applied, which keeps the loop finite and complete.
    pub fn follow(
        &mut self,
        relaxation: &Relaxation,
        tables: &Tables,
        state: &State,
        relaxed_plan: &[OpId],
    ) -> Option<u64> {
        self.start(relaxation, tables, state);
        self.budget = budget(relaxed_plan);
        self.collect_targets(relaxation, tables, relaxed_plan);
        self.update(tables);
        while self.following.target.iter().any(|&f| !self.holds[f]) {
            let op = self
                .best_candidate(relaxation, tables)
                .or_else(|| self.next_relaxed(tables, relaxed_plan))?;
            self.satisfy_black_pre(relaxation, tables, op, None)?;
            if !tables.red_pre[op].iter().all(|&f| self.holds[f]) {
                return None;
            }

            self.apply(relaxation, tables, op);
            if let Some(i) = relaxed_plan.iter().position(|&o| o == op) {
                self.following.used[i] = true;
            }

            self.update(tables);
        }

        self.finish(relaxation, tables)
    }

    /// Fills `R+` and the candidate operators from `relaxed_plan`, and clears the used marks.
    fn collect_targets(&mut self, relaxation: &Relaxation, tables: &Tables, relaxed_plan: &[OpId]) {
        let f = &mut self.following;
        f.in_target.clear();
        f.in_target.resize(relaxation.num_facts(), false);
        for &fact in relaxed_plan
            .iter()
            .flat_map(|&op| &tables.red_pre[op])
            .chain(&tables.red_goal)
        {
            f.in_target[fact] = true;
        }

        f.target.clear();
        f.target
            .extend((0..f.in_target.len()).filter(|&fact| f.in_target[fact]));
        f.candidates.clear();
        f.candidates.extend((0..tables.cost.len()).filter(|&op| {
            relaxation
                .op_rules(op)
                .iter()
                .any(|rule| f.in_target[rule.head])
        }));
        f.used.clear();
        f.used.resize(relaxed_plan.len(), false);
    }

    /// `UPDATE` of Figure 6: adds the current black values to `B`, then, roots of the black
    /// causal graph first, every value reachable through transitions whose red preconditions
    /// hold and whose black preconditions are in `B`. `B` only grows during one run.
    fn update(&mut self, tables: &Tables) {
        let fact = |b: usize, value: u32| tables.black_offsets[b] + value as usize;
        for (b, &value) in self.black.iter().enumerate() {
            self.allowed[fact(b, value)] = true;
        }

        let mut stack = core::mem::take(&mut self.following.stack);
        for &b in &tables.topological {
            stack.clear();
            stack.extend((0..tables.domain[b]).filter(|&x| self.allowed[fact(b, x)]));
            while let Some(x) = stack.pop() {
                for t in &tables.transitions[b] {
                    let enabled = t.from.is_none_or(|from| from == x)
                        && !self.allowed[fact(b, t.to)]
                        && tables.red_pre[t.op].iter().all(|&f| self.holds[f])
                        && tables.black_pre[t.op]
                            .iter()
                            .all(|&(c, value)| c == b || self.allowed[fact(c, value)]);
                    if enabled {
                        self.allowed[fact(b, t.to)] = true;
                        stack.push(t.to);
                    }
                }
            }
        }

        self.following.stack = stack;
    }

    /// The best operator of the set `A'` (achievable given `R ∪ B`, adding a missing `R+`
    /// fact), or `None` if it is empty.
    fn best_candidate(&self, relaxation: &Relaxation, tables: &Tables) -> Option<OpId> {
        self.following
            .candidates
            .iter()
            .copied()
            .filter(|&op| self.achievable(tables, op) && self.progresses(relaxation, tables, op))
            .min_by_key(|&op| {
                (
                    self.may_undo(tables, op),
                    self.move_estimate(tables, op),
                    op,
                )
            })
    }

    /// The first operator of `relaxed_plan` not applied yet whose preconditions are in
    /// `R ∪ B`.
    fn next_relaxed(&self, tables: &Tables, relaxed_plan: &[OpId]) -> Option<OpId> {
        relaxed_plan
            .iter()
            .zip(&self.following.used)
            .find(|&(&op, &used)| !used && self.achievable(tables, op))
            .map(|(&op, _)| op)
    }

    /// Whether the red preconditions of `op` hold and its black ones are in `B`.
    fn achievable(&self, tables: &Tables, op: OpId) -> bool {
        tables.red_pre[op].iter().all(|&f| self.holds[f])
            && tables.black_pre[op]
                .iter()
                .all(|&(b, value)| self.allowed[tables.black_offsets[b] + value as usize])
    }

    /// Whether applying `op` after moving the black variables to its preconditions adds a
    /// missing `R+` fact: a rule of it with such a head whose red conditions hold (its black
    /// conditions are preconditions, which `ACHIEVE` establishes).
    fn progresses(&self, relaxation: &Relaxation, tables: &Tables, op: OpId) -> bool {
        relaxation.op_rules(op).iter().any(|rule| {
            self.following.in_target[rule.head]
                && !self.holds[rule.head]
                && rule
                    .conditions
                    .iter()
                    .all(|&c| tables.is_black_fact[c] || self.holds[c])
        })
    }

    /// The estimated cost of moving the black variables to the preconditions of `op`: the sum
    /// of the all-pairs distances from their current values.
    fn move_estimate(&self, tables: &Tables, op: OpId) -> u64 {
        tables.black_pre[op]
            .iter()
            .map(|&(b, value)| {
                let n = tables.domain[b] as usize;
                let from = self.black[b] as usize;
                tables.distance[b][from * n + value as usize].unwrap_or(u64::MAX)
            })
            .fold(0, u64::saturating_add)
    }

    /// Whether reaching the black preconditions of `op` moves a black variable whose moves may
    /// delete an achieved `R+` fact under real semantics (the FDR-executability preference of
    /// the paper's Section 5.3.2, approximated per black variable as there).
    fn may_undo(&self, tables: &Tables, op: OpId) -> bool {
        tables.black_pre[op]
            .iter()
            .filter(|&&(b, value)| self.black[b] != value)
            .any(|&(b, _)| {
                tables.side_facts[b]
                    .iter()
                    .any(|&f| self.following.in_target[f] && self.holds[f])
            })
    }
}
