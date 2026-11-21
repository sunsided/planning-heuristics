# planning-heuristics: design

Status: draft, milestones 1 to 7 implemented (branch `draft/initial`). Version stays 0.0.0.

## Scope

Goal-distance heuristics for classical planning over `sas_plus::Task` (0.2.1):

- delete relaxation: `h_max`, `h_add`, `h_FF` with relaxed plan extraction (milestone 1),
  set-additive `h_sa`, additive `h_max`; `h²`; operator counting (state equation, LP);
- red-black partial delete relaxation (`h_RB`);
- landmarks: LM-cut, `h^1` and RHW landmarks with disjunctions and reasonable orderings,
  landmark count, uniform and optimal (LP) cost partitioning, path-dependent tracking;
- landmark-like relevance scores (`h_Ξ`);
- adapters (optional features) for `merge-and-shrink` and `cost-partition`.

Out of scope: search algorithms (`heuristic-search`, `width-search`), grounding (`pddl-rs`),
the task model itself (`sas-plus`).

This crate becomes the owner of the h_add / relaxed-plan code that `width-search` keeps as a
private copy in `src/sas.rs` today (see `plan/status.md`).

## Heuristic trait

```rust
pub trait Heuristic<S, C = u32> {
    /// `None` = proven dead end.
    fn estimate(&mut self, state: &S) -> Option<C>;
}

impl<S, C, H: Heuristic<S, C> + ?Sized> Heuristic<S, C> for &mut H { /* forwards */ }
impl<S, C, H: Heuristic<S, C> + ?Sized> Heuristic<S, C> for Box<H> { /* forwards */ }

pub fn heuristic_fn<S, C, F: FnMut(&S) -> Option<C>>(f: F) -> HeuristicFn<F>;
```

`search_core::Heuristic` (search-core d7eb2c0), re-exported; it matched heuristic-search
`draft/initial` (60d01f4) when it was local here. `&mut self` lets a heuristic reuse working memory without interior
mutability; searches take the heuristic by value, and `&mut h` keeps it with its workspace.
A blanket impl for closures would overlap with the `&mut H` impl (`&mut F` is itself
`FnMut`, E0119), so closures go through `heuristic_fn`.

Semantics: `Option<C>`, `None` only when no goal is reachable, `u32` as default cost. Rules for cost types:

- Every heuristic here computes in `u64` internally (operator costs are `u32` in sas-plus,
  sums can exceed `u32`). Sums saturate at `u64::MAX`; reachability is tracked separately, so a
  saturated value is never mistaken for a dead end.
- Each heuristic implements only `Heuristic<State, u32>` (the default cost), saturating at
  `u32::MAX`, which keeps an admissible heuristic admissible. One impl means `h.estimate(&s)`
  needs no cost-type annotation. Callers that need the wider type use the inherent
  `value(&State) -> Option<u64>`.
- A future float estimate over integer costs rounds up (`ceil`), as agreed with heuristic-search.

Done: the trait, `heuristic_fn` and `HeuristicFn` are `search-core`'s (a path dependency on
its unreleased 0.0.0), re-exported from the crate root; the local definitions are gone. The
signatures are identical, so the heuristics only changed their `impl` path. The dependency is
unconditional rather than behind a default feature: the crate has no use without the trait.
`search_core::Zero` is not re-exported (no heuristic here needs it).

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

1. Close every fact of `state` and every free fact (see axioms) at cost 0 directly, which is
   what popping them first would do minus the queue traffic, and let them trigger their
   watchers. Then let every rule without conditions offer `rule.cost` to its head; these
   rules are never triggered otherwise.
2. Pop an open fact with the lowest cost from the queue; skip it if it is already closed. Close it and, for each watching rule, combine the cost into the rule's
   accumulator (`max` or saturating `+`) and decrement its counter. A rule whose counter hits 0
   offers `accumulator + rule.cost` to its head.
3. An offer lowers a fact's cost only if strictly cheaper; the offering rule becomes the fact's
   best supporter.

Both `max` and `+` are monotone and never below their arguments, so a closed fact never
improves (Knuth 1977). So pushed costs never drop below the last popped one, and the queue is a
bucket queue (Dial 1969): one bucket per cost below 1024, scanned forward only, with a binary
heap for larger costs, so large or sparse costs never scan or allocate a bucket per value.
Cost: `O(facts + rule conditions)` per state plus `O(log facts)` per fact above cost 1024,
against the `O(facts · rules)` sweeps of width-search's copy.

Tie-breaking (deterministic, but not by fact id): equal costs pop in reverse push order (each
bucket is a stack; the heap part orders by fact id). A fact keeps the first rule that offered
it its final cost.

`Relaxation::explore` runs until the queue is empty, so every fact gets its exact relaxed cost
(`Exploration::cost`). The heuristics stop as soon as the last goal fact is closed: goal costs
are final then, and so are the supporters and closing order of every closed fact, which is all
relaxed plan extraction needs. Tests check that both give the same goal cost.

Memory: read-only tables and per-evaluation state are split. `Relaxation` holds the tables
and is `Sync`. `HMax<'r>`, `HAdd<'r>` and `HFf<'r>` hold a `Cow<'r, Relaxation>` plus an owned
workspace (costs, supporters, counters, queue, closing order, plan buffers). A state costs no
allocation once the buffers have grown (`HFf::relaxed_plan` allocates only the returned
operator list), and starting an evaluation is `O(1)`: fact and rule slots carry the epoch of
the evaluation that wrote them, a stale fact slot reads as unreached, and a stale rule slot
restarts its counters on first touch. So an evaluation that stops early at the goal never pays
for the facts and rules it did not reach. The stamps are wiped once every `2³¹` evaluations
(unit-tested across the wrap). Hot per-fact data (cost, supporter, stamp) and per-rule data
(counters, head, cost, stamp) sit in one slot each, ids are `u32`, and the combine operator is
a const generic so the inner loop has no branch on it. Two constructors:

- `HMax::new(&task) -> HMax<'static>` owns its relaxation: one line for one thread;
- `HMax::with_relaxation(&relaxation) -> HMax<'r>` borrows a shared one: build the
  `Relaxation` once (in an `Arc` if it must outlive a scope) and create one heuristic per
  thread; creating one costs only empty buffers.

A compile-time assertion in `lib.rs` checks `Relaxation: Sync` and the heuristics: `Send`; a
test runs one shared relaxation on four threads.
`Relaxation::explore` and `Relaxation::relaxed_plan` allocate a fresh workspace per call.
The watcher lists are one flat array with per-fact offsets (CSR) instead of a vector per fact.

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

## Red-black (milestone 3, done)

`RedBlack` (`h_RB`): black variables keep their deletes, the others are relaxed. Sources:
Katz, Hoffmann & Domshlak, *Who Said We Need to Relax All Variables?*, ICAPS 2013
(red-black planning, complexity, reversibility); Katz, Hoffmann & Domshlak, *Red-Black
Relaxed Plan Heuristics*, AAAI 2013 (RSE-invertibility, relaxed plan repair, paintings A and
C); Domshlak, Hoffmann & Katz, *Red-Black Planning: A New Systematic Approach to Partial
Delete Relaxation*, AIJ 221, 2015 (the journal version: Figures 4, 7 and 8, paintings L, A,
C). All three were checked against the PDFs.

- Painting: black variables must be state variables outside axioms and conditional effects
  (as condition or effect). `Painting::new` additionally requires them pairwise unlinked in
  `sas_plus::CausalGraph` (the arcless fragment), `Painting::dag` an acyclic black causal
  graph. Either way an operator changes at most one black variable (two black effects link
  them both ways: the causal graph has effect-effect arcs). Neither checks RSE-invertibility;
  every strategy constructor paints only RSE-invertible variables. `Painting::bounded` allows
  any black causal graph over a few variables (product of domain sizes at most 256), for the
  fixed-number fragment.
- Strategies: `Painting::with_strategy` runs Figure 8 of the AIJ paper for a `Fragment`
  (`Arcless`: configuration E, `Dag`: configuration D) with every choice point of Section
  6.1 (`Strategy`): L and L̄, A and Ā, C and C̄, C[N], CA[p], RND.
  `Painting::causal_graph_level`, `Painting::incident_arcs` and `Painting::conflict_directed`
  are shorthands for L, A and C.
  `Painting::greedy` and `Painting::greedy_dag` are this crate's own (smallest domain
  first, RSE-invertible candidates); `Painting::all_red` gives `h_FF`.
- Plan repair (Figure 4): execute the `h_FF` relaxed plan under red-black semantics (red facts
  accumulate, each black variable has one value). Before each operator, move every black
  variable whose precondition value differs along a cheapest DTG path (Dijkstra over its
  values) using transitions whose red preconditions hold; at the end move the black goal
  variables. With a DAG painting each transition's black preconditions on causal-graph
  ancestors are satisfied recursively, descendants first (Figure 7), and only transitions of
  the projected task `Π^B` are used: black preconditions and the reached value visited before.
  Value = cost of the repaired plan. Option F (`RedBlackOptions::forward`, AAAI) first moves
  every operator without black effects as early as it can go.
- Red facts following (`RedBlackOptions::follow`, AIJ Section 5.2, Figure 6): targets are
  the red facts `R+` the relaxed plan needs; the main loop picks an operator whose red
  preconditions hold, whose black ones lie in the achievable values `B`, and that adds a
  missing `R+` fact, moves the black variables there with the same machinery as the repair
  (DAG moves restricted to `B`), applies it and recomputes `B`.
- Stop search (option S, AAAI): `RedBlackPlan::is_plan_for` tells a search the red-black
  plan is a real plan; stopping is the search's business.
- Fixed number of black variables (`Painting::bounded`; ICAPS Theorem 1): tree search over
  the product variable with groups of freely switchable values (`src/red_black/product.rs`);
  a node budget of 4096 falls back to `h_FF`, a proved-unsolvable state is a dead end.
- If a needed path does not exist, or a DAG repair exceeds its operator budget, the estimate
  falls back to `h_FF` (`RedBlackPlan::is_repaired` is false). Inside the fragment
  (RSE-invertible black variables) a path always exists. So for repair `h_FF <= h_RB` always
  (the repaired plan contains the relaxed plan, also when forwarded); dead ends are those of
  `h_FF` (Corollary 1). Following only guarantees `h_RB >= h+` (every red-black plan is a
  relaxed plan); product search may find more dead ends than `h_FF`, all real.
- Complexity: arcless, polynomial (paths are simple DTG paths). DAG: the papers' polynomial
  bound (AAAI Theorem 2, AIJ Theorem 11) holds only under Chen & Giménez's succinct macro
  plans; with explicit plans, as here and in the papers' implementation (AIJ Section 5.3.1),
  Figure 7 is polynomial in the returned plan length, which is worst-case exponential
  (Proposition 2). Hence the budget `1024 + 64 · |relaxed plan|` operators.
- Succinct macro plans, not done. The papers only cite them (Chen & Giménez 2008, Theorem 23,
  with Helmert's Observation 7) and describe them in one sentence ("recursive macros for
  value pairs within DTGs"); the construction itself is not in any of the three papers. The
  AIJ paper also argues against them (Section 5.3.1): macros must be rebuilt for every
  `ACHIEVE` call, they move parent variables back to their initial values between requests
  (longer plans, worse estimates), and its implementation uses explicit plans as we do.
  Note that the heuristic value is the plan length, which can itself be exponential in the
  number of black variables; a succinct plan would bound the runtime, not the value.
- Not done: black variables with conditional effects, black derived variables, choice point
  (ii) of AIJ Section 5.3.2 (prefer DTG paths whose red conditions are true under real
  semantics), and the AIJ enumeration of SCC-changing action sequences for the
  fixed-number fragment (its tighter `2k - 2` bound is a proof device; the ICAPS tree search
  is implemented instead).

### Paper comparison

| Paper rule | Our code | Status |
|---|---|---|
| RB semantics: red values accumulate, black values switch; plan reaches the goal (ICAPS Def.) | `Repair::apply`, `holds` / `black` | matches |
| Arcless black causal graph + RSE-invertible: plan generation polynomial (AAAI Cor. 2) | `Painting::new`, Dijkstra per black variable | matches |
| Acyclic black causal graph + RSE-invertible: plan generation polynomial under succinct plans (AAAI Thm. 2, AIJ Thm. 11) | `Painting::dag`, explicit plans with a budget | deliberate deviation: explicit plans like the papers' implementation; budget guards exponential length |
| Causal graph arcs include effect-effect pairs, so a black-acyclic operator changes one black variable | `sas_plus::CausalGraph` | matches |
| RSE-invertible arc: an inverse arc with `ocon' ⊆ ocon ∪ oeff`; unconditioned operators induce arcs from every value | `painting/invertible.rs` | matches; only unconditional effects count in `oeff` (bug fixed, 4e4c441: conditional side effects were accepted, so a variable that could not move back was painted black; regression test `conditional_side_effects_do_not_invert`) |
| Strong connectivity of each `Π^B` DTG (restricted to `F`) follows from RSE-invertibility (AIJ Thm. 11 (ii), Lemma 1) | no separate test | bug fixed (411526e): candidates used to need a strongly connected whole DTG, ignoring outside conditions, which neither implies nor is implied by the fragment; replaced by the RSE-invertibility test |
| Repair main loop: insert `ACHIEVE(pre(a_i)[V^B])` before each relaxed plan action, `ACHIEVE(G[V^B])` at the end (AIJ Fig. 4) | `Repair::run` | matches |
| `Π^B` actions need `pre(a) ⊆ F` and black effect in `F` (visited facts) (AIJ Fig. 4) | `Repair::usable_in_dag` (DAG paintings) | bug fixed (80f486e): the DAG repair searched the whole DTG, asked ancestors for unvisited values and fell back to `h_FF` inside the fragment; regression test `dag_repair_stays_on_visited_values` |
| Arcless case: Dijkstra over DTG arcs whose red outside conditions hold (AAAI, Implementation) | `Repair::cheapest_paths` without the `F` restriction | matches the AAAI text; deliberate deviation from AIJ Fig. 4's `D(v) ∩ F`: the unrestricted search only finds shorter paths, still valid |
| `F` fixed per `ACHIEVE` call | `F` grows with red side effects of inserted moves | deliberate deviation: a superset of arcs over the same black values, so solvability is kept (AIJ 5.3.1 extends DTGs with red side effects too) |
| `ACYCLIC PLANNING`: leaves to roots, shortest DTG path per required value, black outside conditions left to parents (AIJ Fig. 7) | recursive `move_black`, black preconditions sorted descendants first | matches (same operator order, computed on demand) |
| No relaxed plan: `h = ∞`; otherwise repair succeeds (AIJ Sec. 5) | `None` for dead ends; `h_FF` fallback outside the fragment | matches in the fragment; fallback is our extension for user paintings |
| Painting, Fig. 8: start red = not RSE-invertible or causal graph leaf; paint red until in fragment; post-process back to front | `painting/strategy.rs::paint` | matches; ineligible variables (axioms, conditional effects, derived) also start red, since the repair does not support them black |
| L: paint red the highest causal graph level (closest to leaves) | `causal_graph_level(.., Top, ..)`, longest path from a source in the condensed causal graph, ties to the higher variable id | deliberate deviation: Fast Downward's cg-heuristic level is not available; for translator output the variable id follows that order |
| L̄: lowest level first | `CausalLevel::Bottom` | matches (same level proxy) |
| A: most incident arcs to black variables, ties smaller domain, then L | `incident_arcs` (each direction counts) | matches |
| C: fewest conflicts (relaxed plan actions whose precondition on `v` is violated with only `v` black), ties L | `conflict_directed` | matches; the papers do not say what the value is after a violation: we assume it moved there, as the repair does. Previously approximated (counted goals, painted greedily black): replaced |
| C uses the relaxed plan of the initial state | any `state` argument; dead end gives zero conflicts | deliberate generalisation |
| Theorem 14: red leaves do not change `h^RB*` | applied by every paper strategy | greedy paintings may keep leaves black (harmless) |
| Ā, C̄: inverse selection, same tie-breaking | `Strategy::FewestArcs`, `Strategy::MostConflicts` | matches |
| C[N]: minimal average conflicts over N random sample states | `Strategy::SampledConflicts` (sum = average × N) | deliberate choice: the paper does not say how it samples; random walks from the state of up to twice its relaxed plan length |
| CA[p]: max `p·Â + (1−p)(1−Ĉ)`, normalised by the maxima over black variables, ties of A | `Strategy::Mixed { arcs_percent }`, exact integer arithmetic | matches (p in whole percent; a zero maximum counts as 1, leaving the zero counts at 0) |
| RND: uniform choice | `Strategy::Random { seed }` (SplitMix64) | matches |
| F: move actions without black effects towards the start; stop when the predecessor achieves a precondition (AAAI) | `forward.rs` | matches; effect conditions count as preconditions and an operator needing a derived fact does not move, so the reordered plan stays a relaxed plan |
| S: stop when the red-black plan is a real plan (AAAI) | `RedBlackPlan::is_plan_for` | matches (the check; the search stops) |
| Red facts following: `R+ = G[V^R] ∪ ⋃ pre(a)[V^R]`; loop over `A'`; `UPDATE` in topological order; `ACHIEVE` (AIJ Fig. 6) | `repair/follow.rs` | matches; `ACHIEVE` uses the dynamic red facts instead of the explicit inverse-arc set, as Section 5.3.1 does |
| Choice point (i): fewest estimated steps via all-pairs DTG distances, prefer actions whose black moves delete no achieved `R+` fact (AIJ 5.3.2) | `best_candidate` | matches; distances are operator costs, ties to the lowest id |
| With axioms a derived `R+` fact may have no achieving action | apply the first achievable unapplied relaxed plan action, capped at `|R+| + |relaxed plan|` steps | our extension (the paper has no axioms); bug fixed (8c501c8): a repeated relaxed plan operator was never used up and the loop hung; regression test `following_uses_up_repeated_operators` |
| Fixed number of black variables, fixed domains: polynomial (ICAPS Thm. 1) | `Painting::bounded`, `product.rs` tree search with merged aggregate values | matches the ICAPS search; node budget and `h_FF` fallback are ours |

Tests (for every strategy in both fragments, each with plain repair, forwarding and red facts
following): `h_RB >= h_FF` (following: `>= h_max`), same dead ends, `all_red` equals
`h_FF`, every repair succeeds (and, for an out-of-fragment DAG painting, a failed repair
costs exactly `h_FF`), every repaired plan valid under an independent red-black
simulator with matching cost, and every strategy raises `h_RB` above `h_FF` somewhere. Unit
tests check RSE-invertibility on the SimpleGrid hand variable and the Figure 8 maximality
post-process on a StarShapeLogistics-like task. Every plan claiming to be real costs at least
`h*`. Product search, on every task with as many variables black as fit: finite `h*` implies
a plan, a proved dead end has `h* = ∞`, plans are valid red-black plans of cost at least
`h_max`, and with every variable black they are real plans. A StarShapeLogistics test shows
red facts following find the optimal plan where repair over-estimates.

## Landmarks (milestones 4 and 5, done)

- Done: `Relaxation::landmarks` computes `h^1` fact landmarks (Zhu & Givan 2003; Keyder,
  Richter & Helmert 2010) per state with a label fixpoint over the rules: state facts are
  labelled with themselves, a rule offers its head plus the union of its condition labels, and
  a head's label is the intersection of its offers. Landmarks are the goal facts plus the union
  of their labels; a fact in a landmark's label gives a natural ordering (holds at or before the
  first state where the landmark holds). Derived defaults outside the state get the empty
  label (they only hold once the axioms stop firing). Labels are `facts²` bits.
- Done: `LandmarkCount` (landmarks of the state not true in it; inadmissible) and
  `UniformLandmarkCost` (Karpas & Domshlak 2009: each operator's cost split evenly among the
  pending landmarks it adds, each landmark worth its cheapest share; admissible). Both recompute
  the landmarks per state, so they fit the stateless `Heuristic` trait. The `f64` sum is
  rounded up after a relative tolerance of `1e-9`.
- Tests: every landmark and ordering of every reachable state of random tasks and the fixtures
  is checked by brute force over the state space (no goal, resp. no state with the later
  landmark, is reachable while avoiding the landmark); `UniformLandmarkCost <= h*`.
- Done: `OptimalLandmarkCost` (feature `lp`): the Karpas & Domshlak LP over the pending fact
  landmarks, solved with `microlp` 0.8 (pure Rust, std only, Rust 1.86). Surveyed: `minilp`
  0.2.2 (std only, last released 2020; `microlp` is its maintained fork) and `good_lp`
  (a modelling layer whose pure-Rust backends are `microlp` and `clarabel`, default `coin_cbc`
  is C). None builds `no_std`, hence the feature. The solution is clamped,
  scaled to feasibility and rounded up, then maxed with the uniform estimate, so the value is
  admissible whatever the solver's tolerances. Tests: `uniform <= optimal <= h*`.
- Done: `Relaxation::rhw_landmarks` (Richter, Helmert & Westphal 2008) into
  `OrderedLandmarks` (fact-set landmarks, `OrderingKind::{Natural, GreedyNecessary,
  Reasonable}`): back-chaining over possible first achievers (rules whose conditions are
  relaxed-reachable without the landmark), shared conditions, per-variable disjunctions of 2 to
  4 facts, and DTG landmarks. `From<&LandmarkGraph>` converts the `h^1` graph.
  `add_reasonable_orderings` follows LAMA's approximation with a sound interference test.
  Tests check every landmark and ordering kind by brute force for both extractions; on the
  test tasks `h^1` finds 1329 of 1781 brute-force fact landmarks, RHW 1307 (none beyond
  `h^1`), plus 48 disjunctions.
- Done: `PathHeuristic` (root / on_expand / merge / estimate) and `LandmarkTracker` with
  `TrackingMode::Count` (LAMA) and `TrackingMode::UniformCost` (admissible; acceptance
  without orderings, required-again only from goals and greedy-necessary orderings, merge by
  intersection). Tests run a reference A* and GBFS.
- Done: `OperatorCounting` (feature `lp`): operator-counting LP with the state equation
  (Bonet 2013; may-produce minus always-consume per state-variable fact) and LM-cut landmarks
  as constraints (Pommerening, Röger, Helmert & Bonet 2014); infeasible means dead end, the
  optimum is rounded up after a `1e-6` tolerance and floored at LM-cut. Tests: `<= h*`, only
  real dead ends, at least LM-cut, the combination at least each part.
- Open: optimal partitioning over action landmarks (LM-cut's cuts) or the tracked
  landmarks, `PathHeuristic` in `search-core`.

## Relevance score (done)

`RelevanceScore` (`h_Ξ`): Kim & Sridharan, *Relevance Score: A Landmark-Like Heuristic for
Planning*, arXiv:2403.07510, 2024. Checked against the paper's text. Not admissible (one
operator achieving three goal facts: `h_Ξ = 3`, `h* = 1`); `0` exactly at goal states.

| Paper rule | Our code | Status |
|---|---|---|
| Backtracking tree: root `achieveGoal` with the goals, fact → achieving actions, action → preconditions (Sec. 3.4) | `relevance/tree.rs` over the `Relaxation` rules (operator effects and axioms are the actions; conditions are preconditions plus effect conditions) | matches for STRIPS; conditional effects and axioms are our generalisation |
| An action is excluded below fact `f` if a precondition labels a node on `path(f)` (Sec. 3.4) | `child_labels` (the path includes `f`) | matches |
| NDA: one achiever uniformly at a fact, all preconditions at an action (Alg. 1, Sec. 3.5) | defines the recursion | matches |
| Eq. 2 (fact: average over children), Eq. 3 (action: `1 − Π(1 − Ξ)`), `Ξ(l, n) = 1` at own label | `combine`, bottom-up per state | matches; verified against an independent enumeration of the agent's outcomes |
| Eqs. 5–7 and LCA-based computation (Sec. 4.1, 4.3) | not used | deliberate deviation: Eq. 7 is a rewriting of Eqs. 2–3, so the direct recursion gives the same values on the same explored tree |
| Alg. 2: dives from frontier nodes chosen ∝ `ξ`, children added, one chosen ∝ `ξ`; loop while `|T| < 100 000` or frontier mass / explored mass `> ρ = 0.2` | `Tree::explore`, `RelevanceParams` (defaults 100 000, 0.2) | matches, plus: a childless node leaves the frontier (the printed loop never ends otherwise), stop on an empty frontier, `max_nodes` (400 000) caps memory |
| Partial tree gives lower bounds (Sec. 4.2) | unexplored nodes count 0 | matches; tested (partial ≤ full) |
| `T/σ` removes the nodes labelled with facts of `σ` and their descendants; `h_Ξ(σ) = Σ_l Ξ_σ(l)` (Sec. 4.4) | `is_removed` (free facts count as true) | matches the formula (open question: the prose says the agent "stops at" such nodes, which would keep them; the formula is the one that gives `h = 0` at goals) |
| Rounded to the nearest integer for LAMA (Sec. 4.4) | `Heuristic::estimate` | matches |
| Every state gets a score | `None` for relaxed dead ends | deliberate addition |
| Discussion: landmarks score 1 on a full tree | not true in general | a landmark scores 1/2 when the other achiever of the goal has an unachievable precondition (test `landmarks_can_score_below_one`); the score is a sampling probability, not a fraction of plans |

## Goal intention (done)

`GoalIntention`, `IntentionTracker`: Rosa, Honorio, Lipovetzky & Sardiña, *Planning as Goal
Recognition: Deriving Heuristics from Intention Models* (extended version),
arXiv:2603.14824, 2026. Checked against the paper's text; the sampler comes from its
Appendix B, which adapts Wilken et al. (ECAI 2024), not read here. Ranking keys, not
distances: not admissible, growing with the path, never a dead end.

| Paper rule | Our code | Status |
|---|---|---|
| Fact probabilities from 100 sampled relaxed plans at the initial state | `IntentionParams::samples = 100`, `GoalIntention::new` | matches |
| Alg. 2: all goal facts in `C`; layers from the top; achievers of minimum `h_add` in the layer, then minimum count, then random; preconditions not in `s0`, not found, not in `C` go to `newC`; `add(a)` leaves `C` and `newC` | `intention/sampling.rs` | reconstructed: goal facts already true are not queued; layers count axioms as steps; "achievers at the layer" means achievers whose highest condition is at most that layer (any reachable one if none); `add(a)` means effects certain to fire with the chosen effect; a supporter's own preconditions stay queued (the printed lines drop them, which produced non-plans on random tasks; caught by `samples_are_relaxed_plans`) |
| `P(a)` from the samples; a fact's probability "of not observing any supporter action" | `P(a)` = fraction of samples containing `a`; `P(q) = 1 − Π (1 − P(a))` over its achievers | reconstructed (one sentence in the paper); a fact supported in every sample can get less than 1, as the paper notes (doc example: 0.75) |
| `P(q | ¬G)` from samples achieving every achievable non-goal fact | same sampler with those targets | matches |
| `ol_rp`: `P̃(O_F | G) = Π P(q | G)`, ranked by the sum of log probabilities | `likelihood_cost` = `−Σ ln P` | matches (lower cost = higher likelihood) |
| `op_rp`: `Σ ln P(q | G) − ln P(q | ¬G)` | `posterior_cost` (negated) | matches; zero probabilities give `±∞`, never NaN (our rule) |
| Observations: facts added by the observed actions | `IntentionTracker::on_expand`: heads of the effects that fire | matches |
| `_r` variants: only facts observed since the last `h_GC` improvement | `restart: true` | matches |
| UTP re-weighting of samples (OL-UTP) | not built | underspecified for relaxed plans (which state's applicable actions?) |
| BFWS integration (third tie-breaker) | not built | search is out of scope; the tracker is the hook |
| Claim 1, Claim 2 | tests on the implemented likelihood | hold |
| Theorem 4 / Eq. 7 (`KL = −ln ρ(y* | G)`) | test by enumeration over an intermediate domain | holds |
| Lemma 1, Claims 1–2 for the plan-library model (uniform weights) | test-only enumeration on tiny tasks | hold (checks the paper's arithmetic, not the crate) |
| Theorems 1–3, Lemmas 2–4 (search behaviour) | not tested | need a search |

## Adapters (done)

- `AbstractionHeuristic<A: sas_plus::Abstraction>`: goal distances under given costs (a cost
  partition's share or `Task::costs`), looked up per state; `projection(task, pattern)` builds a
  pattern database. `None` for abstract dead ends and pruned states.
- `GoalDistance` (`value(&self, &State) -> Option<u64>`) and `Adapter<H: GoalDistance>`, which
  implements `Heuristic<State>` by forwarding and saturating at `u32::MAX`. Since milestone 7
  the trait is foreign, so `impl Heuristic<State> for MergeAndShrinkHeuristic` would be an
  orphan (E0117); the local `GoalDistance` impls are legal. A `heuristic_fn` closure would also
  work but has an unnameable type. The clean fix is for merge-and-shrink and cost-partition to
  implement `search_core::Heuristic` themselves; `Adapter` stays useful for any other
  `value`-style heuristic.
- Feature `merge-and-shrink`: `GoalDistance for MergeAndShrinkHeuristic`. Build with
  `prune: false` for arbitrary states.
- Feature `cost-partition`: the same for `CostPartitionedHeuristic` and `MaxOverOrders`.
- Both are path dependencies on unreleased versions (merge-and-shrink 0.3.0, cost-partition
  0.2.0) and must be published before this crate.

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

- **width-search**: none required; 0.3 ships `SasProblem::h_add`, so the comparison test
  runs in CI. Its `h_add` sums duplicated rule conditions twice and lacks free derived
  defaults (see Testing); both go away when it switches to this crate.
- **sas-plus**: none required. Nice to have: DTGs (red-black builds its own),
  indexed axiom evaluation (would replace the `close_axioms` rescans here).
- **search-core**: publish 0.1; this crate re-exports its `Heuristic` and depends on it by
  path until then. Its MIGRATION.md calls this crate's switch 1:1, which misses the orphan
  rule for the feature adapters (see Adapters).
- **merge-and-shrink**, **cost-partition**: implement `search_core::Heuristic` directly, so
  the `Adapter` wrapping becomes optional.
- **merge-and-shrink**: publish 0.3.0 (`MergeAndShrinkHeuristic::value`); the
  `merge-and-shrink` feature uses it as a path dependency until then.
- **cost-partition**: publish 0.2.0 (`MaxOverOrders`) for the `cost-partition` feature; an LP
  solver (its planned optimal baseline) would enable optimal landmark cost partitioning here.

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

Comparison with width-search (`tests/it/width_search.rs`): `h_add` against
`width_search::sas::SasProblem::h_add` on every reachable state of the fixtures and of random
tasks (plain, conditional effects, conditional effects plus axioms). The values are equal on
tasks without axioms and without effect conditions that repeat a precondition fact. Elsewhere
ours is at most theirs wherever both are finite, for two known reasons:

- with axioms, width-search does not treat derived defaults as free, so it can report a dead
  end or a higher cost where the default only holds after an operator;
- width-search does not deduplicate a rule's conditions, so a fact that is both a precondition
  and an effect condition is summed twice (e.g. random task seed 31 with `max_conditions: 2`:
  ours 13, theirs 15). Ours counts it once, which is the standard `h_add`.

The dev-dependency is

```toml
[dev-dependencies]
width-search = { path = "../width-search", version = "0.3", features = ["sas-plus"] }
```

`SasProblem::h_add` is public since width-search 0.3. `cargo publish` keeps only the
`version` of a dev-dependency and drops the `path`, so the published manifest resolves against
crates.io.

## Benchmark

`cargo bench --bench relaxation` evaluates 100 reachable states per task (times per 100
states, criterion means, one noisy shared machine: width-search's unchanged code moved by up
to ±40% between runs, so read only large differences).

Milestone 2a (workspace reuse, early stop), criterion means, before / after:

| Task | h_max | h_add | h_FF | width-search h_add (unchanged code) |
|---|---|---|---|---|
| courier | 41.1 / 27.8 µs | 38.0 / 25.0 µs | 54.9 / 28.5 µs | 23.4 / 19.5 µs |
| wiring | 18.1 / 9.6 µs | 16.6 / 8.7 µs | 18.9 / 8.3 µs | 3.6 / 4.0 µs |
| gripper-costs | 14.1 / 9.1 µs | 16.1 / 8.4 µs | 25.6 / 10.4 µs | 13.1 / 8.2 µs |
| bw-large-a | 1.99 / 1.75 ms | 2.43 / 1.86 ms | 2.37 / 1.74 ms | 4.37 / 2.65 ms |
| logistics-e | 2.22 / 2.08 ms | 1.79 / 2.22 ms | 2.69 / 2.17 ms | 3.16 / 0.89 ms |
| random 40×400 | 1.66 / 0.35 ms | 1.21 / 0.30 ms | 1.91 / 0.31 ms | 0.73 / 0.52 ms |

Milestone 2b (bucket queue, packed slots, direct start facts, epoch stamps). Profile first
(callgrind on logistics-e `h_add`; perf is not permitted on this machine): the binary heap was
about 30% of instructions, the rest bounds-checked indexing and per-fact/per-rule bookkeeping
plus about 4% clearing buffers, so the heap hypothesis held only in part. Wiring showed the
fixed per-evaluation cost dominating (resets, start facts through the queue). Two runs each,
before (`5e2efa5`) and after (`e5714ed`), criterion means:

| Task | h_max | h_add | h_FF | width-search h_add |
|---|---|---|---|---|
| courier | 23.8, 18.6 / 11.4, 12.1 µs | 26.9, 16.9 / 11.8, 14.6 µs | 29.8, 24.8 / 17.2, 52.6* µs | 28.9, 20.6 / 20.3, 28.7 µs |
| wiring | 9.7, 8.3 / 2.5, 2.3 µs | 8.7, 7.4 / 2.7, 2.0 µs | 10.0, 8.0 / 4.0, 4.3 µs | 4.5, 3.0 / 3.0, 2.6 µs |
| gripper-costs | 10.9, 7.8 / 5.6, 4.9 µs | 11.6, 7.5 / 6.1, 4.3 µs | 9.4, 8.5 / 8.9, 5.9 µs | 23.6, 9.3 / 10.0, 7.9 µs |
| bw-large-a | 3.55, 1.66 / 1.41, 1.01 ms | 2.54, 1.60 / 1.55, 1.10 ms | 2.29, 1.62 / 1.66, 1.13 ms | 2.27, 1.92 / 2.31, 1.63 ms |
| logistics-e | 1.32, 1.08 / 0.50, 0.37 ms | 1.49, 1.22 / 0.59, 0.40 ms | 2.70, 1.59 / 0.82, 0.58 ms | 0.95, 0.83 / 0.96, 0.58 ms |
| random 40×400 | 298, 275 / 100, 77 µs | 474, 258 / 96, 75 µs | 283, 275 / 109, 83 µs | 720, 441 / 547, 367 µs |

(* one outlier run.) `h_max` and `h_add` are now at or below width-search's `h_add` on every
task in both runs; on logistics-e by about 1.4–1.6×, on bw-large-a about 1.5×, on random about
5×. Instruction counts (callgrind, per evaluation) agree: logistics-e 92k ours vs 144k theirs.
`h_FF` does more work (plan marking and replay) and is above width-search's `h_add` on wiring
(4.0–4.3 vs 2.6–3.0 µs). The guess at the time, that the axiom closure in the replay rescanning
all axioms was the cause, turned out wrong; see milestone 2c.

Milestone 2c (indexed axiom closure, start facts skipped in the replay). Callgrind first, h_FF on
wiring: of about 2,440 instructions per evaluation, the exploration takes about 1,570 (as much as
`h_add`) and the axiom closure only about 190. Wiring has 6 axioms, 32 reachable states, and on
average 0.34 replayed operators per evaluation, so the "rescan per step" cost barely occurs.
Condition counters per axiom lost to the old short-circuiting scan (their per-evaluation setup
cost more than it saved). What stayed: one pass over the axioms for the start state, then only
the axioms watching a newly reached fact are re-evaluated, so a replay step no longer costs a
pass over all axioms. That alone is neutral on wiring (`d5983ec`); the gain comes from starting
the replay after the start facts, which the exploration closes first (count recorded in its
seed). Instructions per evaluation (callgrind, setup subtracted), before (`528665e`) / after
(`e4e7c09`):

| Task | h_max | h_add | h_FF | width-search h_add |
|---|---|---|---|---|
| courier | 4,800 / 4,801 | 4,822 / 4,822 | 6,613 / 6,477 | 6,698 |
| wiring | 1,563 / 1,564 | 1,565 / 1,565 | 2,444 / 2,244 | 1,400 |
| gripper-costs | 3,900 / 3,901 | 3,798 / 3,798 | 4,986 / 4,914 | 5,617 |
| bw-large-a | 182.5k / 182.5k | 174.0k / 174.0k | 179.1k / 178.8k | |
| logistics-e | 91.7k / 91.7k | 91.4k / 91.4k | 112.4k / 112.1k | |
| random 40×400 | 18.6k / 18.6k | 18.6k / 18.6k | 20.3k / 19.5k | |

Wall-clock runs on the shared machine were too noisy to separate these (width-search's unchanged
`h_add` moved by up to ±47% between interleaved runs). `h_FF` on wiring stays above width-search's
`h_add`: the gap is the exploration itself (`h_add` alone is about 1,565 instructions against
width-search's 1,400 on this task) plus about 680 for marking supporters, scanning the closing
order and building the relaxed state, not the axioms. No bench task has more than 6 axioms, so
the scaling of the indexed closure is not measured here.

## Milestones

1. **Done**: `Relaxation`, worklist `h_max` / `h_add`, `h_FF` with relaxed plan extraction,
   axioms and conditional effects, `Heuristic` trait, oracle tests.
2. Speed, **done**: workspace reuse, early stop at the goal, flat watcher lists, bucket
   queue, packed slots, `O(1)` reset with epoch stamps, criterion bench
   (`benches/relaxation.rs`) against width-search's sweep. Indexed axiom closure in
   the plan replay done. Open: preferred operators, incremental evaluation hooks. Then propose the
   width-search switch.
3. Red-black `h_RB` (greedy and all paper paintings for arcless and DAG fragments, plan
   repair with forwarding, red facts following, stop search test, fixed-number fragment),
   **done**.
4. LM-cut, **done** (`src/lm_cut.rs`): `h_max` from scratch per round, operator-level
   costs over effect rules; `LmCut::new` splits conditional operators by context (at most 64
   copies each, `src/lm_cut/contexts.rs`), so the cut sum dominates `h_max`; the
   `max(sum, h_max)` fallback remains for larger operators and shared relaxations.
   Seeded evaluation (`LmCut::value_seeded`, `LmCut::h_max_seeded`): start costs per fact
   instead of a state, and optional per-call operator costs, on the prebuilt relaxation.
   Each seeded fact is an artificial operator from a start fact with the seed as its cost,
   chargeable by cuts, so seeds add up per fact used; the bound is the cheapest seeded
   completion `min_s Σ_v seed(s[v]) + h*(s)`. State seeds (`LmCut::seed_state`) reproduce
   `LmCut::value` exactly (the state facts' zero-cost seed operators never enter a cut).
   Tests: equality on every reachable state of random tasks, `<=` a brute-force completion
   with random seeds and costs. Bench: 6-9x faster than rebuilding task and `LmCut` per call.
5. Landmarks: `h^1` and RHW landmarks, disjunctive landmarks, reasonable orderings, landmark
   count, uniform and optimal (LP, feature `lp`) cost partitioning, path-dependent tracking
   (`PathHeuristic`, `LandmarkTracker`), **done**.
6. Adapters: abstraction heuristics, `merge-and-shrink`, `cost-partition` features, **done**.
7. Switch to the shared trait from `search-core`, **done** (feature adapters via `Adapter`).

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
- M. Katz, J. Hoffmann, C. Domshlak. Red-black relaxed plan heuristics. AAAI 2013.
- C. Domshlak, J. Hoffmann, M. Katz. Red-black planning: a new systematic approach to partial
  delete relaxation. AIJ 221, 2015.
- E. Karpas, C. Domshlak. Cost-optimal planning with landmarks. IJCAI 2009. (uniform and
  optimal cost partitioning, multi-path acceptance)
- B. Bonet. An admissible heuristic for SAS+ planning obtained from the state equation.
  IJCAI 2013.
- F. Pommerening, G. Röger, M. Helmert, B. Bonet. LP-based heuristics for cost-optimal
  planning. ICAPS 2014. (operator counting)
- P. Haslum, H. Geffner. Admissible heuristics for optimal planning. AIPS 2000. (`h^m`)
- P. Haslum, B. Bonet, H. Geffner. New admissible heuristics for domain-independent planning.
  AAAI 2005. (additive `h^m`)
- E. Keyder, H. Geffner. Heuristics for planning with action costs revisited. ECAI 2008.
  (set-additive)
- S. Richter, M. Helmert, M. Westphal. Landmarks revisited. AAAI 2008. (RHW landmarks)
- J. Hoffmann, J. Porteous, L. Sebastia. Ordered landmarks in planning. JAIR 22, 2004.
  (reasonable orderings)
- O. Kim, M. Sridharan. Relevance score: a landmark-like heuristic for planning.
  arXiv:2403.07510, 2024. (`h_Ξ`)
- G. Rosa, J. Honorio, N. Lipovetzky, S. Sardiña. Planning as goal recognition: deriving
  heuristics from intention models (extended version). arXiv:2603.14824, 2026. (`ol_rp`,
  `op_rp`)
- N. Wilken, L. Cohausz, C. Bartelt, H. Stuckenschmidt. Fact probability vector based goal
  recognition. ECAI 2024. (supporter sampling, via Rosa et al.)
- L. Zhu, R. Givan. Landmark extraction via planning graph propagation. ICAPS doctoral
  consortium 2003. (`h^1` labels)
- G. Röger, F. Pommerening, M. Helmert. Optimal planning in the presence of conditional
  effects: extending LM-Cut with context splitting. ECAI 2014.
- M. Helmert. The Fast Downward planning system. JAIR 26, 2006. (axioms, SAS+ format)
