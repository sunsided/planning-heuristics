# planning-heuristics

Heuristics for classical planning: red-black, FF, landmark and abstraction heuristic adapters.

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
- Abstraction heuristics: any `sas_plus::Abstraction`, pattern databases over
  `sas_plus::Projection`, and (optional features) merge-and-shrink and saturated cost
  partitioning

Not yet: preferred operators; `h²` with axioms; succinct macro plans for red-black DAGs and
the other gaps listed in [docs/design.md](docs/design.md).

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
  pure-Rust `microlp`. It needs `std` and Rust 1.86 (the rest of the crate: 1.85).

The first two depend on unreleased versions of those crates (path dependencies for now), as does
search-core.

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
[docs/design.md](docs/design.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
