//! The `h²` heuristic (Haslum & Geffner 2000): `h^m` with `m = 2`.

use alloc::vec::Vec;
use core::fmt;

use sas_plus::{Fact, State, Task, VarId};

use crate::Heuristic;
use crate::lm_cut::contexts;

/// Marks an unreached fact or pair.
const INF: u64 = u64::MAX;

/// Most context copies per operator when [`H2::new`] compiles conditional effects away.
const MAX_CONTEXTS: usize = 1024;

/// Why [`H2::new`] cannot handle a task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum H2Error {
    /// The task has axioms; `h²` over derived variables is not implemented.
    Axioms,
    /// Operator `0` has more than 1024 effect contexts, too many to compile its conditional
    /// effects away.
    TooManyContexts(usize),
}

impl fmt::Display for H2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Axioms => write!(f, "h2 does not support axioms"),
            Self::TooManyContexts(op) => {
                write!(f, "operator {op} has too many conditional effect contexts")
            }
        }
    }
}

impl core::error::Error for H2Error {}

/// One operator in the form `h²` needs: fact ids, effects unconditional and final per
/// variable.
#[derive(Clone, Debug)]
struct Op {
    /// Precondition fact ids, ascending.
    pre: Vec<usize>,
    /// Effect fact ids, one per variable (the last effect on it wins), ascending.
    eff: Vec<usize>,
    /// The variables the effects change, ascending.
    eff_vars: Vec<VarId>,
    /// The operator cost.
    cost: u64,
}

/// The `h²` heuristic (Haslum & Geffner 2000): the cost of the cheapest regression where every
/// set of more than two facts costs as much as its most expensive pair. Admissible, at least
/// `h_max`, and it sees deletes: two facts that no plan makes true together cost more than
/// either alone.
///
/// Computed forward to the fixpoint over all facts and pairs: an operator with precondition
/// `P` and effects `E` reaches each fact and pair within `E` at cost `c + h²(P)`, and each
/// pair `{p, q}` with `p ∈ E` and `q` untouched by the operator at `c + h²(P ∪ {q})`. Two
/// values of one variable never hold together. Memory is `facts²` words; one sweep is
/// `O(operators × effects × facts × preconditions)`, repeated until nothing improves.
///
/// Conditional effects are compiled away by splitting operators by context (as in
/// [`crate::LmCut::new`]); the copies have every plan of the task, so the estimate stays
/// admissible. Axioms are not supported.
///
/// # Examples
///
/// ```
/// use planning_heuristics::{H2, HMax, Heuristic};
/// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
///
/// // Making `b` resets `a`: h_max = 1, but `a` and `b` together need `b` first, then `a`.
/// let op = |name: &str, eff: Vec<Fact>| Operator {
///     name: name.into(),
///     pre: vec![],
///     eff: eff.into_iter().map(Effect::new).collect(),
///     cost: 1,
/// };
/// let task = Task {
///     variables: vec![Variable::new("a", 2), Variable::new("b", 2)],
///     operators: vec![
///         op("make-a", vec![Fact::new(0, 1)]),
///         op("make-b", vec![Fact::new(1, 1), Fact::new(0, 0)]),
///     ],
///     initial: State(vec![0, 0]),
///     goal: vec![Fact::new(0, 1), Fact::new(1, 1)],
///     axioms: vec![],
/// };
/// let state = task.initial_state();
///
/// assert_eq!(HMax::new(&task).estimate(&state), Some(1));
/// assert_eq!(H2::new(&task).unwrap().estimate(&state), Some(2));
/// ```
#[derive(Clone, Debug)]
pub struct H2 {
    /// Fact id of `var = 0` per variable.
    offsets: Vec<usize>,
    /// The variable of each fact.
    var_of: Vec<VarId>,
    /// The operators, conditional effects compiled away.
    ops: Vec<Op>,
    /// The goal fact ids, ascending.
    goal: Vec<usize>,
    /// `h²` of each pair `(p, q)` at `p * facts + q`, symmetric; `(p, p)` is `h²(p)`.
    table: Vec<u64>,
}

impl H2 {
    /// Builds `h²` for `task`.
    ///
    /// # Errors
    ///
    /// - [`H2Error::Axioms`] if the task has axioms;
    /// - [`H2Error::TooManyContexts`] if an operator's conditional effects have more than
    ///   1024 contexts.
    pub fn new(task: &Task) -> Result<Self, H2Error> {
        if !task.axioms.is_empty() {
            return Err(H2Error::Axioms);
        }

        let split = contexts::split(task, MAX_CONTEXTS);
        let operators = split
            .as_ref()
            .map_or(&task.operators, |s| &s.task.operators);
        if let Some(i) = operators
            .iter()
            .position(|o| o.eff.iter().any(|e| !e.conditions.is_empty()))
        {
            let op = split.as_ref().map_or(i, |s| s.op_of[i]);
            return Err(H2Error::TooManyContexts(op));
        }

        let mut offsets = Vec::with_capacity(task.num_variables());
        let mut var_of = Vec::new();
        for (var, v) in task.variables.iter().enumerate() {
            offsets.push(var_of.len());
            var_of.extend(core::iter::repeat_n(var, v.domain_size as usize));
        }

        let id = |f: &Fact| offsets[f.var] + f.value as usize;
        let ops = operators
            .iter()
            .map(|o| {
                let mut final_eff: Vec<Fact> = Vec::new();
                for e in &o.eff {
                    final_eff.retain(|f| f.var != e.fact.var);
                    final_eff.push(e.fact);
                }

                Op {
                    pre: sorted(o.pre.iter().map(id)),
                    eff: sorted(final_eff.iter().map(id)),
                    eff_vars: sorted(final_eff.iter().map(|f| f.var)),
                    cost: u64::from(o.cost),
                }
            })
            .collect();
        let goal = sorted(task.goal.iter().map(id));

        Ok(Self {
            offsets,
            var_of,
            ops,
            goal,
            table: Vec::new(),
        })
    }

    /// `h²` of `state`; `None` if the goal is unreachable even for `h²` (a dead end).
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    #[must_use]
    pub fn value(&mut self, state: &State) -> Option<u64> {
        self.seed(state);
        while self.sweep() {}

        let value = self.of_set(&self.goal);

        (value != INF).then_some(value)
    }

    /// Resets the table to the facts and pairs of `state` at 0, everything else unreached.
    fn seed(&mut self, state: &State) {
        let n = self.var_of.len();
        self.table.clear();
        self.table.resize(n * n, INF);
        let facts: Vec<usize> = state
            .0
            .iter()
            .zip(&self.offsets)
            .map(|(&value, &offset)| offset + value as usize)
            .collect();
        for &p in &facts {
            for &q in &facts {
                self.table[p * n + q] = 0;
            }
        }
    }

    /// One pass over all operators; returns whether a value improved.
    fn sweep(&mut self) -> bool {
        // Taken out so the table can be updated while the operators are read.
        let ops = core::mem::take(&mut self.ops);
        let mut changed = false;
        for op in &ops {
            let base = self.of_set(&op.pre);
            if base == INF {
                continue;
            }

            let reach = base.saturating_add(op.cost);
            for (k, &p) in op.eff.iter().enumerate() {
                for &q in &op.eff[k..] {
                    changed |= self.lower(p, q, reach);
                }

                changed |= self.extend_untouched(p, op, base);
            }
        }

        self.ops = ops;

        changed
    }

    /// Lowers every pair `{p, q}` with `q` untouched by `op` (its variable not among the
    /// effect variables) to `cost(op) + h²(pre(op) ∪ {q})`, given `base = h²(pre(op))`;
    /// returns whether one improved.
    fn extend_untouched(&mut self, p: usize, op: &Op, base: u64) -> bool {
        let n = self.var_of.len();
        let mut changed = false;
        for q in 0..n {
            if op.eff_vars.binary_search(&self.var_of[q]).is_ok() {
                continue;
            }

            let mut h = base.max(self.table[q * n + q]);
            for &x in &op.pre {
                h = h.max(self.table[x * n + q]);
            }

            if h != INF {
                changed |= self.lower(p, q, h.saturating_add(op.cost));
            }
        }

        changed
    }

    /// Lowers the pair `{p, q}` to `value` if that improves it and the two facts can hold
    /// together (they are one fact, or of different variables).
    fn lower(&mut self, p: usize, q: usize, value: u64) -> bool {
        if p != q && self.var_of[p] == self.var_of[q] {
            return false;
        }

        let n = self.var_of.len();
        if value >= self.table[p * n + q] {
            return false;
        }

        self.table[p * n + q] = value;
        self.table[q * n + p] = value;

        true
    }

    /// `h²` of a set of facts: the maximum over its facts and pairs; [`INF`] if two of them
    /// are values of one variable.
    fn of_set(&self, facts: &[usize]) -> u64 {
        let n = self.var_of.len();
        let mut h = 0;
        for (k, &p) in facts.iter().enumerate() {
            for &q in &facts[k..] {
                h = h.max(self.table[p * n + q]);
            }
        }

        h
    }
}

impl Heuristic<State> for H2 {
    fn estimate(&mut self, state: &State) -> Option<u32> {
        self.value(state)
            .map(|v| u32::try_from(v).unwrap_or(u32::MAX))
    }
}

/// `ids` sorted ascending without duplicates.
fn sorted<T: Ord>(ids: impl Iterator<Item = T>) -> Vec<T> {
    let mut ids: Vec<T> = ids.collect();
    ids.sort_unstable();
    ids.dedup();

    ids
}
