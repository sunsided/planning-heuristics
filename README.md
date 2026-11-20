# planning-heuristics

Heuristics for classical planning: red-black, FF, landmark and abstraction heuristic adapters.

## Scope

- `h_max`, `h_add` and `h_FF` with relaxed plan extraction (delete relaxation)
- Red-black heuristic `h_RB` by relaxed plan repair, with a greedy or explicit painting
- `h^1` fact landmarks with natural orderings; landmark count (inadmissible) and uniform
  landmark cost partitioning (admissible)
- Abstraction heuristics: any `sas_plus::Abstraction`, pattern databases over
  `sas_plus::Projection`, and (optional features) merge-and-shrink and saturated cost
  partitioning

Not yet: LM-cut, optimal (LP) landmark cost partitioning, LAMA-style landmark tracking along
the search path, preferred operators.

## Features

- `merge-and-shrink`: `Heuristic` for `merge_and_shrink::MergeAndShrinkHeuristic`.
- `cost-partition`: `Heuristic` for `cost_partition::CostPartitionedHeuristic` and
  `MaxOverOrders`.

Both depend on unreleased versions of those crates (path dependencies for now).

## Status

Pre-release, version 0.0.0, `no_std` with `alloc`. The API will change; see
[docs/design.md](docs/design.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
