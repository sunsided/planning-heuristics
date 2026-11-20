//! Red-black planning with a fixed number of black variables of fixed-size domains (Katz,
//! Hoffmann & Domshlak, ICAPS 2013, Theorem 1 and its tree search; Domshlak, Hoffmann & Katz,
//! AIJ 2015, Theorem 1).
//!
//! The black variables are compiled into one product variable (the papers' first step). The
//! search keeps the values of the product variable visited along its path as a sequence of
//! *groups*: within the last group the variable can switch freely (every value was reached
//! along a cycle of transitions whose red conditions stay true), so operators prevailed by
//! any of its values are usable. Each node closes the red facts under the usable operators
//! (a relaxed fixpoint, executed for real by moving inside the group), stops if the goal
//! holds, and otherwise branches over every transition leaving the group. A transition back
//! into an earlier group closes a cycle and merges all groups from there on, as the ICAPS
//! proof does with its aggregate values. Along one branch at most `k - 1` groups are added and
//! at most `k - 1` merges happen for `k` product values, so the depth is below `2k`.

use alloc::collections::VecDeque;
use alloc::vec::Vec;

use sas_plus::{Fact, OpId, State, Task, VarId};

use crate::Relaxation;

/// Most search nodes one evaluation may expand before giving up (the search is complete but
/// exponential in the product size).
const NODE_BUDGET: usize = 4096;

/// The outcome of one product search.
pub(super) enum Outcome {
    /// A red-black plan and its cost.
    Plan(Vec<OpId>, u64),
    /// The search space is exhausted: the red-black task, and so the task, is unsolvable.
    Unsolvable,
    /// The node budget ran out.
    Exhausted,
}

/// The product compilation of the black variables of one task.
#[derive(Clone, Debug)]
pub(super) struct Product {
    /// Whether each fact (by [`Relaxation::fact_id`]) belongs to a black variable.
    is_black_fact: Vec<bool>,
    /// For each operator, the fact ids of its red preconditions.
    red_pre: Vec<Vec<usize>>,
    /// For each operator, its cost.
    cost: Vec<u64>,
    /// For each operator, the product values its black preconditions allow, ascending.
    matching: Vec<Vec<u32>>,
    /// For each operator with black effects, the value each entry of `matching` moves to;
    /// empty for an operator without black effects.
    next: Vec<Vec<u32>>,
    /// For each product value, the transitions leaving it as `(operator, value reached)`.
    out: Vec<Vec<(OpId, u32)>>,
    /// The product values the goal allows.
    goal_values: Vec<bool>,
    /// The fact ids of the red goal facts.
    red_goal: Vec<usize>,
    /// The black variables, ascending, with the stride of each in a product value.
    strides: Vec<(VarId, u32)>,
}

/// One node of the search: a red-black state, the plan reaching it and the groups.
#[derive(Clone, Debug)]
struct Node {
    /// Whether each red fact holds (black facts are tracked by `current`).
    holds: Vec<bool>,
    /// The plan so far.
    plan: Vec<OpId>,
    /// Its cost.
    cost: u64,
    /// The groups of visited product values; the last is the one the variable is in.
    groups: Vec<Vec<u32>>,
    /// For each product value, the index of its group, if visited.
    group_of: Vec<Option<usize>>,
    /// The current product value.
    current: u32,
}

impl Product {
    /// Compiles the black variables `black` (ascending) of `task`.
    ///
    /// # Panics
    ///
    /// Panics if a variable of `black` is not a variable of `task`, or the product of their
    /// domain sizes does not fit a `u32`.
    pub fn new(task: &Task, relaxation: &Relaxation, black: &[VarId]) -> Self {
        let mut strides = Vec::with_capacity(black.len());
        let mut size: u32 = 1;
        for &v in black {
            strides.push((v, size));
            size = size
                .checked_mul(task.variables[v].domain_size)
                .expect("the product fits u32");
        }

        let mut is_black_fact = alloc::vec![false; relaxation.num_facts()];
        for &v in black {
            for value in 0..task.variables[v].domain_size {
                is_black_fact[relaxation.fact_id(Fact::new(v, value))] = true;
            }
        }

        let index = |v: VarId| black.binary_search(&v).ok().map(|i| strides[i].1);
        let digit = |x: u32, v: VarId| {
            let i = black.binary_search(&v).expect("a black variable");
            x / strides[i].1 % task.variables[v].domain_size
        };
        let mut matching = Vec::with_capacity(task.operators.len());
        let mut next = Vec::with_capacity(task.operators.len());
        for op in &task.operators {
            let values: Vec<u32> = (0..size)
                .filter(|&x| {
                    op.pre
                        .iter()
                        .all(|f| index(f.var).is_none() || digit(x, f.var) == f.value)
                })
                .collect();
            let black_eff: Vec<&Fact> = op
                .eff
                .iter()
                .map(|e| &e.fact)
                .filter(|f| index(f.var).is_some())
                .collect();
            let moved = if black_eff.is_empty() {
                Vec::new()
            } else {
                values
                    .iter()
                    .map(|&x| {
                        black_eff.iter().fold(x, |y, f| {
                            let stride = index(f.var).expect("a black variable");
                            y - digit(y, f.var) * stride + f.value * stride
                        })
                    })
                    .collect()
            };
            matching.push(values);
            next.push(moved);
        }

        let mut out = alloc::vec![Vec::new(); size as usize];
        for (op, (from, to)) in matching.iter().zip(&next).enumerate() {
            for (&x, &y) in from.iter().zip(to) {
                if x != y {
                    out[x as usize].push((op, y));
                }
            }
        }

        let goal_values = (0..size)
            .map(|x| {
                task.goal
                    .iter()
                    .all(|f| index(f.var).is_none() || digit(x, f.var) == f.value)
            })
            .collect();

        Self {
            red_pre: task
                .operators
                .iter()
                .map(|op| {
                    op.pre
                        .iter()
                        .filter(|f| index(f.var).is_none())
                        .map(|&f| relaxation.fact_id(f))
                        .collect()
                })
                .collect(),
            cost: task.operators.iter().map(|op| u64::from(op.cost)).collect(),
            red_goal: task
                .goal
                .iter()
                .filter(|f| index(f.var).is_none())
                .map(|&f| relaxation.fact_id(f))
                .collect(),
            is_black_fact,
            matching,
            next,
            out,
            goal_values,
            strides,
        }
    }

    /// Searches a red-black plan from `state`.
    ///
    /// # Panics
    ///
    /// Panics if `state` has fewer values than the task has variables.
    pub fn search(&self, relaxation: &Relaxation, state: &State) -> Outcome {
        let mut holds = alloc::vec![false; relaxation.num_facts()];
        for fact in relaxation
            .state_facts(state)
            .chain(relaxation.free_facts().iter().copied())
        {
            holds[fact as usize] = true;
        }

        super::repair::close_axioms(relaxation, &mut holds);
        let current = self
            .strides
            .iter()
            .map(|&(v, stride)| state.0[v] * stride)
            .sum();
        let mut group_of = alloc::vec![None; self.out.len()];
        group_of[current as usize] = Some(0);
        let root = Node {
            holds,
            plan: Vec::new(),
            cost: 0,
            groups: alloc::vec![alloc::vec![current]],
            group_of,
            current,
        };
        let mut expanded = 0;

        match self.expand(relaxation, root, &mut expanded) {
            Some(node) => Outcome::Plan(node.plan, node.cost),
            None if expanded > NODE_BUDGET => Outcome::Exhausted,
            None => Outcome::Unsolvable,
        }
    }

    /// Closes `node` under the operators usable in its group, then returns it if the goal
    /// holds, or else the first plan found by branching over the transitions leaving the
    /// group. `None` if there is none or the budget ran out (then `expanded` exceeds it).
    fn expand(
        &self,
        relaxation: &Relaxation,
        mut node: Node,
        expanded: &mut usize,
    ) -> Option<Node> {
        *expanded += 1;
        if *expanded > NODE_BUDGET {
            return None;
        }

        self.close(relaxation, &mut node);
        if let Some(goal) = self.goal_value(&node) {
            self.move_within(relaxation, &mut node, goal);
            return Some(node);
        }

        let last = node.groups.len() - 1;
        for op in 0..self.matching.len() {
            if self.next[op].is_empty() || !self.red_pre_holds(&node, op) {
                continue;
            }

            for (&x, &y) in self.matching[op].iter().zip(&self.next[op]) {
                if node.group_of[x as usize] != Some(last)
                    || node.group_of[y as usize] == Some(last)
                {
                    continue;
                }

                let mut child = node.clone();
                self.move_within(relaxation, &mut child, x);
                self.apply(relaxation, &mut child, op, Some(y));
                enter(&mut child, y);
                if let Some(found) = self.expand(relaxation, child, expanded) {
                    return Some(found);
                }

                if *expanded > NODE_BUDGET {
                    return None;
                }
            }
        }

        None
    }

    /// Applies, until none adds a red fact, every operator usable in the last group: its red
    /// preconditions hold and it is prevailed by, or moves between, values of the group. The
    /// variable moves inside the group as needed.
    fn close(&self, relaxation: &Relaxation, node: &mut Node) {
        let mut changed = true;
        while changed {
            changed = false;
            for op in 0..self.matching.len() {
                if !self.red_pre_holds(node, op) || !self.adds_red(relaxation, node, op) {
                    continue;
                }

                let Some((from, to)) = self.usable_in_group(node, op) else {
                    continue;
                };

                self.move_within(relaxation, node, from);
                self.apply(relaxation, node, op, to);
                changed = true;
            }
        }
    }

    /// For `op` usable in the last group, the value to apply it from (the current one if
    /// possible) and, for a transition, the value it reaches.
    fn usable_in_group(&self, node: &Node, op: OpId) -> Option<(u32, Option<u32>)> {
        let last = Some(node.groups.len() - 1);
        let in_group = |x: u32| node.group_of[x as usize] == last;
        let candidates = self.matching[op]
            .iter()
            .enumerate()
            .filter(|&(i, &x)| in_group(x) && self.next[op].get(i).is_none_or(|&y| in_group(y)));
        let pick = candidates
            .clone()
            .find(|&(_, &x)| x == node.current)
            .or_else(|| candidates.clone().next())?;

        Some((*pick.1, self.next[op].get(pick.0).copied()))
    }

    /// A value of the last group the goal allows, if the red goal holds.
    fn goal_value(&self, node: &Node) -> Option<u32> {
        if !self.red_goal.iter().all(|&f| node.holds[f]) {
            return None;
        }

        node.groups
            .last()
            .expect("there is a group")
            .iter()
            .copied()
            .find(|&x| self.goal_values[x as usize])
    }

    /// Moves the product variable to `target` in the last group along a shortest path (in
    /// operators) of transitions inside the group whose red preconditions hold. Such a path
    /// exists: every group is strongly connected by transitions applied before.
    fn move_within(&self, relaxation: &Relaxation, node: &mut Node, target: u32) {
        if node.current == target {
            return;
        }

        let last = Some(node.groups.len() - 1);
        let mut via: Vec<Option<(u32, OpId)>> = alloc::vec![None; self.out.len()];
        let mut queue = VecDeque::from([node.current]);
        while let Some(x) = queue.pop_front() {
            if x == target {
                break;
            }

            for &(op, y) in &self.out[x as usize] {
                let fresh = y != node.current && via[y as usize].is_none();
                if fresh && node.group_of[y as usize] == last && self.red_pre_holds(node, op) {
                    via[y as usize] = Some((x, op));
                    queue.push_back(y);
                }
            }
        }

        let mut path = Vec::new();
        let mut value = target;
        while value != node.current {
            let (from, op) = via[value as usize].expect("groups are strongly connected");
            path.push((op, value));
            value = from;
        }

        for &(op, to) in path.iter().rev() {
            self.apply(relaxation, node, op, Some(to));
        }
    }

    /// Applies `op` in `node`: red effects whose red conditions hold are added (black
    /// conditions are preconditions, which hold), the product variable moves to `to`, and the
    /// red facts are closed under the axioms.
    fn apply(&self, relaxation: &Relaxation, node: &mut Node, op: OpId, to: Option<u32>) {
        let fired: Vec<usize> = relaxation
            .op_rules(op)
            .iter()
            .filter(|rule| !self.is_black_fact[rule.head])
            .filter(|rule| self.red_conditions_hold(node, &rule.conditions))
            .map(|rule| rule.head)
            .collect();
        for head in fired {
            node.holds[head] = true;
        }

        if let Some(to) = to {
            node.current = to;
        }

        super::repair::close_axioms(relaxation, &mut node.holds);
        node.plan.push(op);
        node.cost = node.cost.saturating_add(self.cost[op]);
    }

    /// Whether some red effect of `op` would add a fact not yet true.
    fn adds_red(&self, relaxation: &Relaxation, node: &Node, op: OpId) -> bool {
        relaxation.op_rules(op).iter().any(|rule| {
            !self.is_black_fact[rule.head]
                && !node.holds[rule.head]
                && self.red_conditions_hold(node, &rule.conditions)
        })
    }

    /// Whether the red facts of `conditions` hold.
    fn red_conditions_hold(&self, node: &Node, conditions: &[usize]) -> bool {
        conditions
            .iter()
            .all(|&c| self.is_black_fact[c] || node.holds[c])
    }

    /// Whether the red preconditions of `op` hold.
    fn red_pre_holds(&self, node: &Node, op: OpId) -> bool {
        self.red_pre[op].iter().all(|&f| node.holds[f])
    }
}

/// Records that the product variable entered `value` from the last group: a value of an
/// earlier group closes a cycle and merges that group and all later ones; a new value starts
/// a group.
fn enter(node: &mut Node, value: u32) {
    if let Some(i) = node.group_of[value as usize] {
        let merged: Vec<u32> = node.groups.drain(i..).flatten().collect();
        for &x in &merged {
            node.group_of[x as usize] = Some(i);
        }

        node.groups.push(merged);
    } else {
        node.group_of[value as usize] = Some(node.groups.len());
        node.groups.push(alloc::vec![value]);
    }
}
