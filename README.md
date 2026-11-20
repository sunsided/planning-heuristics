# planning-heuristics

Heuristics for classical planning: red-black, FF, landmark and abstraction heuristic adapters.

## Scope

- `h_max`, `h_add` and `h_FF` with relaxed plan extraction (delete relaxation)
- Red-black heuristic `h_RB` by relaxed plan repair, with a greedy or explicit painting
- `h^1` fact landmarks with natural orderings; landmark count (inadmissible) and uniform
  landmark cost partitioning (admissible); optimal landmark cost partitioning by LP
  (feature `lp`)
- LM-cut (`h_LM-cut`), admissible, with conditional effects and axioms
- Abstraction heuristics: any `sas_plus::Abstraction`, pattern databases over
  `sas_plus::Projection`, and (optional features) merge-and-shrink and saturated cost
  partitioning

Not yet: LAMA-style landmark tracking along
the search path, preferred operators.

The `Heuristic` trait, `heuristic_fn` and `HeuristicFn` are
[search-core](https://github.com/sunsided/search-core)'s, re-exported from the crate root, so
these heuristics plug into every search crate built on it.

## Features

- `merge-and-shrink`: `GoalDistance` for `merge_and_shrink::MergeAndShrinkHeuristic`; wrap it
  in `Adapter` to get a `Heuristic`.
- `cost-partition`: the same for `cost_partition::CostPartitionedHeuristic` and
  `MaxOverOrders`.

- `lp`: `OptimalLandmarkCost`, the optimal landmark cost partitioning, solved with the
  pure-Rust `microlp`. It needs `std` and Rust 1.86 (the rest of the crate: 1.85).

The first two depend on unreleased versions of those crates (path dependencies for now), as does
search-core.

## Breaking changes

- `Heuristic`, `heuristic_fn` and `HeuristicFn` are now `search_core`'s (identical signatures;
  existing `use planning_heuristics::Heuristic` paths keep working).
- The feature-gated heuristics of merge-and-shrink and cost-partition no longer implement
  `Heuristic` directly (the orphan rule forbids it for a foreign trait). Wrap them:
  `Adapter(builder.build(&task)).estimate(&state)`. Values are unchanged.

## Status

Pre-release, version 0.0.0, `no_std` with `alloc`. The API will change; see
[docs/design.md](docs/design.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
