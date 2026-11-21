# planning-heuristics

[![crates.io](https://img.shields.io/crates/v/planning-heuristics.svg)](https://crates.io/crates/planning-heuristics)
[![docs.rs](https://img.shields.io/docsrs/planning-heuristics)](https://docs.rs/planning-heuristics)
[![CI](https://github.com/sunsided/planning-heuristics/actions/workflows/ci.yml/badge.svg)](https://github.com/sunsided/planning-heuristics/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sunsided/planning-heuristics/graph/badge.svg)](https://codecov.io/gh/sunsided/planning-heuristics)
[![MSRV](https://img.shields.io/crates/msrv/planning-heuristics)](https://crates.io/crates/planning-heuristics)
[![License](https://img.shields.io/crates/l/planning-heuristics)](https://github.com/sunsided/planning-heuristics#license)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

Heuristics for classical planning: red-black, FF, landmark and abstraction heuristic adapters.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Usage](#usage)
- [Scope](#scope)
- [Features](#features)
- [MSRV](#msrv)
- [Breaking changes](#breaking-changes)
- [Status](#status)
- [License](#license)
- [Contributing](#contributing)

<!-- END doctoc -->

## Usage

```rust
use planning_heuristics::{HAdd, HFf, HMax, Heuristic, Relaxation};
use sas_plus::{Effect, Fact, Operator, State, Task, Variable};

// One binary variable, one operator that sets it, goal `v0 = 1`.
let task = Task {
    variables: vec![Variable::new("v0", 2)],
    operators: vec![Operator {
        name: "set".into(),
        pre: vec![Fact::new(0, 0)],
        eff: vec![Effect::new(Fact::new(0, 1))],
        cost: 3,
    }],
    initial: State(vec![0]),
    goal: vec![Fact::new(0, 1)],
    axioms: vec![],
};
let state = task.initial_state();

// Each heuristic owns its relaxation ...
assert_eq!(HMax::new(&task).estimate(&state), Some(3));

// ... or borrows a shared one; one heuristic per thread, the relaxation is `Sync`.
let relaxation = Relaxation::new(&task);
let mut h_add = HAdd::with_relaxation(&relaxation);
let mut h_ff = HFf::with_relaxation(&relaxation);
assert_eq!(h_add.estimate(&state), Some(3));
assert_eq!(h_ff.value(&state), Some(3_u64));
```

## Scope

- `h_max`, `h_add` and `h_FF` with relaxed plan extraction (delete relaxation); the
  set-additive `h_sa`; additive `h_max` over an operator partition; `h²` (no axioms)
- Red-black heuristic `h_RB` (Katz, Hoffmann & Domshlak 2013; Domshlak, Hoffmann & Katz
  2015), checked against the papers: relaxed plan repair (optionally forwarded) or red facts
  following, every painting strategy of the papers (L, A, C, their inverses, C[N], CA[p],
  RND) for an arcless or acyclic black causal graph over RSE-invertible variables (where it
  succeeds, up to an operator budget for DAGs), greedy and explicit paintings, product search
  for a few black variables with any causal graph, a stop-search check; `h_FF` fallback
  outside the fragments
- `h^1` fact landmarks with natural orderings, and RHW back-chained landmarks with disjunctive
  and DTG landmarks, greedy-necessary and reasonable orderings; landmark count (inadmissible)
  and uniform landmark cost partitioning (admissible); optimal landmark cost partitioning by
  LP (feature `lp`)
- LAMA-style landmark tracking along the search path (`PathHeuristic`, `LandmarkTracker`),
  inadmissible count or admissible uniform cost partitioning
- LM-cut (`h_LM-cut`), admissible, with axioms, and conditional effects by context splitting;
  also from seeded start costs per fact and per-call operator costs (`LmCut::value_seeded`)
- The relevance score `h_Ξ` (Kim & Sridharan 2024), landmark-like and inadmissible, over a
  partially explored backtracking tree
- Goal-intention ranking keys `ol_rp` / `op_rp` (Rosa et al. 2026) from sampled relaxed plans,
  with a path tracker (`IntentionTracker`)
- Abstraction heuristics: any `sas_plus::Abstraction`, pattern databases over
  `sas_plus::Projection`, and (optional features) merge-and-shrink and saturated cost
  partitioning

Not yet: preferred operators; `h²` with axioms; succinct macro plans for red-black DAGs and
the other gaps listed in [docs/design.md](https://github.com/sunsided/planning-heuristics/blob/main/docs/design.md).

The `Heuristic` trait, `heuristic_fn` and `HeuristicFn` are
[search-core](https://github.com/sunsided/search-core)'s, re-exported from the crate root, so
these heuristics plug into every search crate built on it.

## Features

- `merge-and-shrink`: `GoalDistance` for `merge_and_shrink::MergeAndShrinkHeuristic`; wrap it
  in `Adapter` to get a `Heuristic`.
- `cost-partition`: the same for `cost_partition::CostPartitionedHeuristic` and
  `MaxOverOrders`.
- `lp`: `OptimalLandmarkCost`, the optimal landmark cost partitioning, and
  `OperatorCounting`, the state equation with LM-cut constraints, solved with the
  pure-Rust `microlp`. It needs `std`.

## MSRV

Rust 1.85 for the library with default features and with `merge-and-shrink`. The features
`cost-partition` and `lp` need Rust 1.86 (cost-partition and microlp require it). Tests and
benches may need a newer Rust.

## Breaking changes

- `PaintingError` has a new variant `TooLarge`; `RedBlack` gained `with_options`.
- `Painting::causal_graph_level` and `Painting::conflict_directed` take a `Fragment` and follow
  the papers' painting algorithm: candidates must be RSE-invertible (also for `greedy` and
  `greedy_dag`), causal graph leaves stay red, `CausalLevel` names the end kept black, and
  conflicts no longer count goals. Black sets differ from before.
- `Heuristic`, `heuristic_fn` and `HeuristicFn` are now `search_core`'s (identical signatures;
  existing `use planning_heuristics::Heuristic` paths keep working).
- The feature-gated heuristics of merge-and-shrink and cost-partition no longer implement
  `Heuristic` directly (the orphan rule forbids it for a foreign trait). Wrap them:
  `Adapter(builder.build(&task)).estimate(&state)`. Values are unchanged.

## Status

Pre-release, version 0.0.0, `no_std` with `alloc`. The API will change; see
[docs/design.md](https://github.com/sunsided/planning-heuristics/blob/main/docs/design.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/sunsided/planning-heuristics/blob/main/LICENSE-APACHE))
- MIT license ([LICENSE-MIT](https://github.com/sunsided/planning-heuristics/blob/main/LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this work by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.

## Contributing

See [CONTRIBUTING.md](https://github.com/sunsided/planning-heuristics/blob/main/CONTRIBUTING.md),
including how to build against local checkouts of the sibling crates.
