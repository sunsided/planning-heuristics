# planning-heuristics

[![crates.io](https://img.shields.io/crates/v/planning-heuristics.svg)](https://crates.io/crates/planning-heuristics)
[![docs.rs](https://img.shields.io/docsrs/planning-heuristics)](https://docs.rs/planning-heuristics)
[![CI](https://github.com/sunsided/planning-heuristics/actions/workflows/ci.yml/badge.svg)](https://github.com/sunsided/planning-heuristics/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sunsided/planning-heuristics/graph/badge.svg)](https://codecov.io/gh/sunsided/planning-heuristics)
[![MSRV](https://img.shields.io/crates/msrv/planning-heuristics)](https://crates.io/crates/planning-heuristics)
[![License](https://img.shields.io/crates/l/planning-heuristics)](https://github.com/sunsided/planning-heuristics#license)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

Goal-distance heuristics for classical planning tasks in [sas-plus](https://github.com/sunsided/sas-plus)'s
finite-domain representation: the delete-relaxation family (`h_max`, `h_add`, `h_FF`, `h_sa`, `h²`), red-black
planning, landmarks with count, uniform and optimal cost partitioning, LM-cut, operator counting and adapters for
abstraction heuristics. The crate is `no_std` with `alloc` (only the LP-based heuristics behind the `lp` feature need
`std`), and every state heuristic implements [search-core](https://github.com/sunsided/search-core)'s `Heuristic`
trait, so it plugs into any search crate built on it.

The red-black implementation follows Katz, Hoffmann & Domshlak (2013) and Domshlak, Hoffmann & Katz (2015) closely
and was checked against both papers rule by rule, including every painting strategy they describe. The admissible
heuristics (`h_max`, LM-cut, landmark cost partitioning, `h²`, operator counting, abstractions) are tested against
exact goal distances computed by brute force. One shared `Relaxation` is `Sync`, so a parallel search can build it
once and hand each thread its own heuristic.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Who this is for](#who-this-is-for)
- [Installation](#installation)
- [Usage](#usage)
- [Scope](#scope)
- [Features](#features)
- [MSRV](#msrv)
- [License](#license)
- [Contributing](#contributing)

<!-- END doctoc -->

## Who this is for

- Planning researchers who want to compare heuristics on the same task representation without porting
  Fast Downward's C++, or who need a reference implementation of red-black planning or RHW landmarks to check
  their own against.
- Anyone building a classical planner in Rust: pick `h_FF` or red-black for satisficing greedy search, LM-cut or
  landmark cost partitioning for optimal A*, and LAMA-style landmark tracking for path-dependent search.
- Embedded and robotics teams that plan onboard: the crate runs on `no_std` targets with an allocator.
- If the problem is a plain graph rather than a factored planning task, an explicit-state search with a
  domain-specific heuristic in [heuristic-search](https://github.com/sunsided/heuristic-search) is the better
  fit.
- For merge-and-shrink abstractions or saturated cost partitioning themselves, use
  [merge-and-shrink](https://github.com/sunsided/merge-and-shrink) and
  [cost-partition](https://github.com/sunsided/cost-partition); this crate only adapts them to the
  `Heuristic` trait.

## Installation

```toml
[dependencies]
planning-heuristics = "0.1"
```

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

- `h_max`, `h_add` and `h_FF` with relaxed plan extraction and preferred operators (delete relaxation); the
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

Not yet implemented: `h²` with axioms; succinct macro plans for red-black
DAGs and the other gaps listed in [docs/design.md](https://github.com/sunsided/planning-heuristics/blob/main/docs/design.md).

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

Rust 1.86 for the library, with every feature. Tests and benches may need a newer Rust.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/sunsided/planning-heuristics/blob/main/LICENSE-APACHE))
- MIT license ([LICENSE-MIT](https://github.com/sunsided/planning-heuristics/blob/main/LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this work by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.

## Contributing

See [CONTRIBUTING.md](https://github.com/sunsided/planning-heuristics/blob/main/CONTRIBUTING.md).
