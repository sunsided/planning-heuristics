//! Reference implementations, written independently of the crate: a naive relaxed fixpoint,
//! a relaxed plan validator and an explicit-state Dijkstra for `h*`.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap, VecDeque};

use planning_heuristics::Combine;
use sas_plus::{Fact, OpId, State, Task};

/// Relaxed cost of every reachable fact from `state`, by sweeping all rules until nothing
/// improves (Bellman-Ford style). Derived defaults are free, as in the crate.
pub fn relaxed_costs(task: &Task, state: &State, combine: Combine) -> BTreeMap<Fact, u64> {
    let mut cost: BTreeMap<Fact, u64> = start_facts(task, state).map(|f| (f, 0)).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for (conditions, head, rule_cost) in rules(task) {
            let Some(pre) = combined(&cost, &conditions, combine) else {
                continue;
            };
            let total = pre + rule_cost;
            if cost.get(&head).is_none_or(|&old| total < old) {
                cost.insert(head, total);
                changed = true;
            }
        }
    }

    cost
}

/// The combined goal cost from `state`; `None` if a goal fact is relaxed-unreachable.
pub fn relaxed_goal_cost(task: &Task, state: &State, combine: Combine) -> Option<u64> {
    combined(&relaxed_costs(task, state, combine), &task.goal, combine)
}

/// Whether applying `plan` in order from `state` under the delete relaxation (effects only
/// add, axioms closed after every step, derived defaults free) reaches the goal.
pub fn is_relaxed_plan(task: &Task, state: &State, plan: &[OpId]) -> bool {
    let mut reached: Vec<Fact> = start_facts(task, state).collect();
    close_axioms(task, &mut reached);
    for &op in plan {
        let operator = &task.operators[op];
        if !operator.pre.iter().all(|f| reached.contains(f)) {
            return false;
        }

        let fired: Vec<Fact> = operator
            .eff
            .iter()
            .filter(|e| e.conditions.iter().all(|f| reached.contains(f)))
            .map(|e| e.fact)
            .collect();
        reached.extend(fired);
        close_axioms(task, &mut reached);
    }

    task.goal.iter().all(|f| reached.contains(f))
}

/// Sum of the operator costs of `plan`.
pub fn plan_cost(task: &Task, plan: &[OpId]) -> u64 {
    plan.iter()
        .map(|&op| u64::from(task.operators[op].cost))
        .sum()
}

/// Every state reachable from the initial state with its goal distance `h*`; `None` for
/// states from which no goal is reachable.
pub fn perfect(task: &Task) -> HashMap<State, Option<u64>> {
    let (states, predecessors) = state_space(task);
    let mut distance: Vec<Option<u64>> = vec![None; states.len()];
    let mut heap = BinaryHeap::new();
    for (i, s) in states.iter().enumerate() {
        if task.is_goal(s) {
            distance[i] = Some(0);
            heap.push(Reverse((0, i)));
        }
    }

    while let Some(Reverse((d, i))) = heap.pop() {
        if distance[i].is_some_and(|best| d > best) {
            continue;
        }

        for &(p, cost) in &predecessors[i] {
            let next = d + cost;
            if distance[p].is_none_or(|old| next < old) {
                distance[p] = Some(next);
                heap.push(Reverse((next, p)));
            }
        }
    }

    states.into_iter().zip(distance).collect()
}

/// The reachable states and, for each, its predecessors with the edge cost.
fn state_space(task: &Task) -> (Vec<State>, Vec<Vec<(usize, u64)>>) {
    let mut index: HashMap<State, usize> = HashMap::new();
    let mut states = vec![task.initial_state()];
    let mut predecessors: Vec<Vec<(usize, u64)>> = vec![Vec::new()];
    index.insert(states[0].clone(), 0);
    let mut queue = VecDeque::from([0]);
    while let Some(i) = queue.pop_front() {
        let state = states[i].clone();
        for op in task.applicable_operators(&state) {
            let next = task.apply(op, &state);
            let j = *index.entry(next.clone()).or_insert_with(|| {
                states.push(next);
                predecessors.push(Vec::new());
                queue.push_back(states.len() - 1);
                states.len() - 1
            });
            predecessors[j].push((i, u64::from(task.operators[op].cost)));
        }
    }

    (states, predecessors)
}

/// The facts of `state` plus the defaults of all derived variables.
fn start_facts<'a>(task: &'a Task, state: &'a State) -> impl Iterator<Item = Fact> + 'a {
    let defaults = (0..task.num_variables())
        .filter(|&v| task.variables[v].is_derived())
        .map(|v| Fact::new(v, task.initial.0[v]));
    let current = state.0.iter().enumerate().map(|(v, &x)| Fact::new(v, x));

    current.chain(defaults)
}

/// All delete-free rules `(conditions, head, cost)`: operator effects and axioms.
fn rules(task: &Task) -> Vec<(Vec<Fact>, Fact, u64)> {
    let effects = task.operators.iter().flat_map(|op| {
        op.eff.iter().map(|e| {
            let conditions = op.pre.iter().chain(&e.conditions).copied().collect();
            (conditions, e.fact, u64::from(op.cost))
        })
    });
    let axioms = task
        .axioms
        .iter()
        .map(|a| (a.conditions.clone(), a.head, 0));
    // A rule whose conditions contradict each other never fires in a real state.
    let consistent = |facts: &[Fact]| {
        facts
            .iter()
            .all(|f| facts.iter().all(|g| g.var != f.var || g.value == f.value))
    };

    effects
        .chain(axioms)
        .filter(|(conditions, _, _)| consistent(conditions))
        .collect()
}

/// Combines the costs of `facts`; `None` if one has no cost. Duplicates count once.
fn combined(cost: &BTreeMap<Fact, u64>, facts: &[Fact], combine: Combine) -> Option<u64> {
    let mut unique = facts.to_vec();
    unique.sort_unstable();
    unique.dedup();

    unique.iter().try_fold(0, |acc, f| {
        let c = *cost.get(f)?;
        Some(match combine {
            Combine::Max => acc.max(c),
            Combine::Add => acc + c,
        })
    })
}

/// Adds the heads of axioms whose conditions hold until nothing changes.
fn close_axioms(task: &Task, reached: &mut Vec<Fact>) {
    let mut changed = true;
    while changed {
        changed = false;
        for a in &task.axioms {
            if !reached.contains(&a.head) && a.conditions.iter().all(|f| reached.contains(f)) {
                reached.push(a.head);
                changed = true;
            }
        }
    }
}

/// Whether applying the operators of `ops`, in any order and as often as they apply, under the
/// delete relaxation from `state` (axioms closed, derived defaults free) reaches the goal.
pub fn relaxed_closure_reaches_goal(task: &Task, state: &State, ops: &[OpId]) -> bool {
    let mut reached: Vec<Fact> = start_facts(task, state).collect();
    close_axioms(task, &mut reached);
    let mut changed = true;
    while changed {
        changed = false;
        for &op in ops {
            let operator = &task.operators[op];
            if !operator.pre.iter().all(|f| reached.contains(f)) {
                continue;
            }

            let fired: Vec<Fact> = operator
                .eff
                .iter()
                .filter(|e| e.conditions.iter().all(|f| reached.contains(f)))
                .map(|e| e.fact)
                .filter(|f| !reached.contains(f))
                .collect();
            changed |= !fired.is_empty();
            reached.extend(fired);
            close_axioms(task, &mut reached);
        }
    }

    task.goal.iter().all(|f| reached.contains(f))
}
