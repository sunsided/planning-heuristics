# planning-heuristics: design

Status: draft, milestone 1 implemented (branch `draft/initial`). Version stays 0.0.0.

## Scope

Goal-distance heuristics for classical planning over `sas_plus::Task` (0.2.1):

- delete relaxation: `h_max`, `h_add`, `h_FF` with relaxed plan extraction (milestone 1);
- red-black partial delete relaxation (`h_RB`);
- landmarks: LM-cut, landmark count;
- adapters (optional features) for `merge-and-shrink` and `cost-partition`.

Out of scope: search algorithms (`heuristic-search`, `width-search`), grounding (`pddl-rs`),
the task model itself (`sas-plus`).

This crate becomes the owner of the h_add / relaxed-plan code that `width-search` keeps as a
private copy in `src/sas.rs` today (see `plan/status.md`).

## Heuristic trait

```rust
pub trait Heuristic<S, C = u32> {
    /// `None` = proven dead end.
    fn estimate(&self, state: &S) -> Option<C>;
}
```

Same shape and semantics as `heuristic_search::Heuristic<S, C>` (draft branch): `Option<C>`,
`None` only when no goal is reachable, `u32` as default cost. Rules for cost types:

- Every heuristic here computes in `u64` internally (operator costs are `u32` in sas-plus,
  sums can exceed `u32`). Sums saturate at `u64::MAX`; reachability is tracked separately, so a
  saturated value is never mistaken for a dead end.
- Each heuristic implements only `Heuristic<State, u32>` (the default cost), saturating at
  `u32::MAX`, which keeps an admissible heuristic admissible. One impl means `h.estimate(&s)`
  needs no cost-type annotation. Callers that need the wider type use the inherent
  `value(&State) -> Option<u64>`.
- A future float estimate over integer costs rounds up (`ceil`), as agreed with heuristic-search.

Plan: define the trait locally for now. Once `heuristic-search` is published (or a `search-core`
crate exists because two crates need the trait), depend on it behind a default feature and
re-export its trait instead of this one. The signatures are identical, so heuristics only change
their `impl` path. Not done now to avoid depending on an unpublished draft.

## Delete relaxation (milestone 1)

### Relaxation

`Relaxation::new(task)` builds, once per task:

- fact ids numbered variable by variable (`offset(var) + value`), identical to width-search's
  `SasProblem` feature ids, so width-search can use them directly;
- one rule per operator effect: conditions = precondition ∪ effect conditions (sorted,
  deduplicated, so condition counters count each fact once), head = effect fact, cost = operator
  cost; rules grouped by operator;
- one rule per axiom: conditions = axiom body, head = axiom head, cost 0;
- for every fact, the list of rules that have it as a condition (watchers);
- the goal facts (deduplicated).

### Worklist (Knuth / generalized Dijkstra)

`Relaxation::explore(state, Combine::{Max, Add})`:

1. Offer cost 0 to every fact of `state` and to every free fact (see axioms), and let every rule
   without conditions offer `rule.cost` to its head. These rules are never triggered otherwise.
2. Pop the open fact with the lowest `(cost, fact id)` from a binary heap; skip it if it is
   already closed. Close it and, for each watching rule, combine the cost into the rule's
   accumulator (`max` or saturating `+`) and decrement its counter. A rule whose counter hits 0
   offers `accumulator + rule.cost` to its head.
3. An offer lowers a fact's cost only if strictly cheaper; the offering rule becomes the fact's
   best supporter.

Both `max` and `+` are monotone and never below their arguments, so a closed fact never
improves (Knuth 1977). Cost: `O((facts + rule conditions) · log facts)` per state, against the
`O(facts · rules)` sweeps of width-search's copy.

Tie-breaking (deterministic): lower cost first, then lower fact id; a fact keeps the first rule
that offered it its final cost. Since rules fire in closing order and watchers are in rule
order, "first" means the rule whose last condition closed first, then the lower rule index.

The exploration runs until the heap is empty, so every fact gets its exact relaxed cost
(`Exploration::cost`). Stopping once all goal facts are closed is a known speed-up, left as a
marked shortcut until a caller needs it.

`h_max` = max of the goal fact costs, `h_add` = sum. `None` if a goal fact is unreachable, which
proves a dead end (the relaxation over-approximates reachability).

### h_FF and relaxed plan extraction

`Relaxation::relaxed_plan(state)` (and `HFf`):

1. Explore with `Combine::Add`. Dead end if the goal is unreachable.
2. Mark rules: from the goal facts, follow best supporters back through their conditions.
   Facts true in the state and free facts have no supporter and end a chain.
3. Replay: take the marked rules in the order their heads were closed. Maintain the relaxed
   state (state facts + free facts, closed under axioms). For each marked operator rule whose
   head is not true yet, append its operator and apply it relaxed: add every effect whose
   conditions hold, then close under axioms. Axiom rules are skipped; the closure covers them.
4. `h_FF` = sum of the costs of the appended operators.

The result is a `RelaxedPlan { ops: Vec<OpId>, cost }` in application order. Without
conditional effects no operator repeats, so `ops` is the set of operators of the plan. With
conditional effects an operator repeats when one of its effects needs another of its own
effects first (`o: a; a ⇒ b`, goal `b`, cost 1: plan `[o, o]`, `h_FF = 2`). Counting it once,
as Fast Downward's FF does, gives `h_FF = 1 < h_max = 2`. Accepted consequence: with
conditional effects our `h_FF` can be higher than Fast Downward's on the same state, because a
repeated operator is counted each time. Without conditional effects both agree up to
tie-breaking. The replay keeps:

- the plan valid as a relaxed *sequence* (checked in tests), hence `h_max <= h+ <= h_FF`;
- every appended operator tied to a distinct marked rule, hence `h_FF <= h_add`.

Preferred operators (the plan's operators applicable in the state) fall out of `ops` and are
not a separate API yet.

### Axioms

Axioms are cost-0 rules. Derived variables need one extra rule: their default value (the value
in `Task::initial`) is free, i.e. true in every relaxed state. In the real semantics a derived
variable falls back to its default whenever its axioms stop firing (negation as failure), so the
default can become true after an operator without any rule achieving it. Without the free
default the relaxation would report false dead ends and `h_max <= h*` would fail (test
`derived_default_is_free`). This over-approximates, like Fast Downward's trivial negated axioms.

States passed to the heuristics must have their derived variables evaluated (as produced by
`Task::initial_state` and `Task::apply`).

Not done: dropping rules whose conditions name two values of one variable (a conditional effect
contradicting its own precondition can never fire). Keeping them is sound, only weaker.

### Conditional effects

Supported, not rejected: each effect is its own rule with precondition ∪ conditions, and the
replay handles repeated operators as above.

## Red-black (outline, milestone 3)

`h_RB` (Katz, Hoffmann & Domshlak 2013; Domshlak, Hoffmann & Katz 2015): some variables
("black") keep their deletes, the others ("red") are relaxed.

- Painting: start with all red; paint black variables of the causal graph chosen by a
  strategy (e.g. invertible / RSE-invertible root variables, smallest domains), keeping the
  black causal graph free of arcs between black variables (needs `sas_plus::CausalGraph` and,
  for some strategies, its SCCs).
- Plan repair: compute the relaxed plan as in `h_FF`, then repair it: for each black variable,
  insert DTG paths that move it to the values the red plan needs, in order (the "red-black plan
  repair" of Domshlak et al. 2015; polynomial for DAG black causal graphs with invertible DTGs).
- Value = cost of the repaired plan; dead end if the relaxed part is a dead end.
- Reuses: `Relaxation`, best supporters, `RelaxedPlan`. Needs: DTGs per variable (build here or
  in sas-plus).
- Restriction for a first version: no conditional effects on black variables, no derived
  black variables (reject with an error at construction).

## Landmarks (outline, milestones 4 and 5)

- LM-cut (Helmert & Domshlak 2009): repeated `h_max` with justification graphs. Needs a
  variant of the worklist on unary operator rules with per-rule mutable costs, recomputed
  incrementally after each cut. Reuses `Relaxation` with a cost vector override
  (`explore_with_costs(state, combine, &[u64])`). Conditional effects are open: splitting each
  effect into a unary operator is not enough (it can undercut `h_max`, as the `o: a; a ⇒ b`
  example shows); it needs context splitting (Röger, Pommerening & Helmert 2014), so reject
  conditional effects and axioms at first, as Fast Downward does for axioms.
- Landmark count (Richter & Westphal 2010, `h_LM`): a landmark graph from RHW or `h^m`
  landmarks, plus per-path accepted/required sets. Needs search-path state (the accepted set
  per node), so it does not fit `Heuristic<S, C>` alone; it will need a stateful extension
  (`fn notify(&mut self, parent, op, child)` or a state wrapper). Admissible variant via
  cost partitioning over landmarks, which links to `cost-partition`.

## Adapters (optional features)

- Feature `merge-and-shrink`: `impl Heuristic<State> for MergeAndShrinkHeuristic` (legal while
  the trait is local), saturating its `value(&State) -> Option<u64>` to `u32`.
  At milestone 7, when the trait comes from heuristic-search, this impl becomes an
  orphan; switch to a wrapper type then (or rely on the shared trait's blanket impl for
  `Fn(&S) -> Option<C>`: `|s| mas.value(s)`). `None` means dead end or pruned; the heuristic must
  be built with `prune: false` for arbitrary states, since a pruned unreachable state is not a
  dead end. Waits for merge-and-shrink 0.3.0.
- Feature `cost-partition`: same pattern over its heuristic type (saturated cost partitioning,
  max-over-orders), plus feeding it landmark and relaxation-based components later.
- A generic adapter over `sas_plus::Abstraction` (abstract goal distances on the LTS) may cover
  both if they keep implementing that trait.

All adapters are dashed edges in `plan/architecture.md`: never required dependencies.

## How width-search would migrate

width-search uses `SasProblem::h_add` (private, sweeps) only for
`RelaxedPlan::relaxed_plan_features`: the add effects of the operators of a relaxed plan
extracted from h_add best supporters (LAPKT's choice). With this crate:

```rust
// in width-search's sas.rs, behind its `sas-plus` feature
let plan = self.relaxation.relaxed_plan(state)?;      // or return early with no features
for &op in plan.ops() {
    out.extend(self.task.operators[op].eff.iter().map(|e| self.feature(&e.fact)));
}
```

- Feature ids are identical (`Relaxation::fact_id` = width-search's `feature`).
- `relaxed_plan_features` loses nothing: it gets the same operator set (modulo tie-breaking;
  repeated operators do not change the feature set) and gains derived-default handling, which
  removes false empty plans on axiom tasks.
- Behaviour change to verify against LAPKT: different tie-breaking can change `#r` counts and
  hence BFWS(R) expansion counts on some fixtures.
- `GoalCount` stays in width-search (it is trivial and not a relaxation heuristic). If several
  crates want a goal-count heuristic, add `GoalCount` here as a `Heuristic`.

## How miniplan would migrate

miniplan's `heuristic::relaxed` has `HAdd`, `HMax`, `HFF` over its own STRIPS-like task (positive
and negative preconditions, `f64` costs, `HValue`, preferred operators, backward fact costs for
regression). It can switch only after it grounds to `sas_plus::Task` (roadmap open question; via
`pddl::sas_plus`). Then:

- wrap this crate's types in miniplan's `Heuristic` trait (name + estimate), converting
  `Option<u64>` to `HValue` (`None` → infinity);
- preferred operators: the `RelaxedPlan::ops` applicable in the state;
- backward relaxed costs for regression stay in miniplan until a second user exists;
- miniplan's `f64` costs must first become fixed-point integers (see `plan/status.md`).

## Cross-crate needs

- **width-search**: expose nothing new; it can switch to this crate once published. To compare
  its h_add with ours in tests it would need a public `h_add` (today private). Not done; see
  Testing.
- **sas-plus**: none required. Nice to have: SCCs on `CausalGraph` and DTGs (red-black),
  indexed axiom evaluation (would replace the `close_axioms` rescans here).
- **heuristic-search**: publish the `Heuristic<S, C>` trait (or extract `search-core`), then this
  crate re-exports it.
- **merge-and-shrink**: 0.3.0 with `MergeAndShrinkHeuristic::value` published before the
  adapter feature lands.

## Testing strategy

Integration tests in `tests/it/` (one binary). For every reachable state of every test task
(state space enumerated by BFS):

- `h_max` and `h_add` equal a naive sweep-until-fixpoint oracle written independently in
  `tests/it/oracle.rs` (same relaxation semantics, `BTreeMap` over `Fact`s);
- dead ends: `h_max`, `h_add`, `h_FF` are `None` on the same states, exactly when the oracle's
  relaxed goal is unreachable, and only on states with `h* = None`;
- `h_max <= h*`, where `h*` comes from an explicit backward Dijkstra over the reachable state
  space;
- `h_max <= h_FF <= h_add`;
- the relaxed plan is valid under an independent relaxed validator and its cost is `h_FF`;
- the `u32` trait impl matches the `u64` value (and saturates beyond `u32::MAX`).

Tasks: the MIT/Apache fixtures copied from sas-plus (`tests/fixtures/`, including conditional
effects and axioms), seeded random tasks from `sas_plus::random` (64 seeds each for plain,
conditional effects, axioms, mixed), and hand-made corner cases (self-dependent conditional
effect, shared subgoal, unreachable goal, free derived default, satisfied goal).

Seeds are looped directly; no proptest, since the generator is already seeded and the state
spaces are small. Keep any later proptest at `PROPTEST_CASES=64`.

Comparison with width-search: skipped. Its h_add is private, and its only public relaxed-plan
output (`relaxed_plan_features`) depends on tie-breaking and on its axiom semantics (no free
derived defaults), so equality would be flaky or wrong. If it is wanted later, add as a
dev-dependency:

```toml
[dev-dependencies]
width-search = { path = "../width-search", version = "0.2.0", features = ["sas-plus"] }
```

`cargo publish` keeps only the `version` of a dev-dependency and drops the `path`, so the
published manifest stays valid as long as that version exists on crates.io; a path-only
dev-dependency is stripped entirely. Compare on tasks without axioms, and compare `h_add`
values only once width-search exposes them.

## Milestones

1. **Done**: `Relaxation`, worklist `h_max` / `h_add`, `h_FF` with relaxed plan extraction,
   axioms and conditional effects, `Heuristic` trait, oracle tests.
2. Reuse and speed: per-call buffer reuse (today each call allocates), early stop at the goal,
   preferred operators, incremental evaluation hooks; criterion benchmark against width-search's
   sweep on the fixtures. Then propose the width-search switch.
3. Red-black `h_RB` (painting strategies, plan repair).
4. LM-cut.
5. Landmark count (needs the stateful extension).
6. Adapters: `merge-and-shrink`, `cost-partition` features.
7. Switch to the shared trait from `heuristic-search` / `search-core`.

## References

- B. Bonet, H. Geffner. Planning as heuristic search. AIJ 129, 2001. (`h_max`, `h_add`)
- J. Hoffmann, B. Nebel. The FF planning system. JAIR 14, 2001. (`h_FF`)
- D. E. Knuth. A generalization of Dijkstra's algorithm. IPL 6(1), 1977.
- E. Keyder, H. Geffner. Heuristics for planning with action costs revisited. ECAI 2008.
  (`h_FF` from `h_add` supporters)
- M. Helmert, C. Domshlak. Landmarks, critical paths and abstractions: what's the difference
  anyway? ICAPS 2009. (LM-cut)
- S. Richter, M. Westphal. The LAMA planner. JAIR 39, 2010. (landmark count)
- M. Katz, J. Hoffmann, C. Domshlak. Who said we need to relax all variables? ICAPS 2013.
- C. Domshlak, J. Hoffmann, M. Katz. Red-black planning: a new systematic approach to partial
  delete relaxation. AIJ 221, 2015.
- G. Röger, F. Pommerening, M. Helmert. Optimal planning in the presence of conditional
  effects: extending LM-Cut with context splitting. ECAI 2014.
- M. Helmert. The Fast Downward planning system. JAIR 26, 2006. (axioms, SAS+ format)
