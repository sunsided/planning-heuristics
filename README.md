# planning-heuristics

Heuristics for classical planning: red-black, FF, landmark and abstraction heuristic adapters.

## Planned scope

- Red-black partial delete relaxation
- FF and h_add style relaxation heuristics
- Landmark heuristics
- Adapters for merge-and-shrink and cost-partition abstractions

## Status

Pre-release, version 0.0.0. Implemented: `h_max`, `h_add` and `h_FF` with relaxed plan
extraction over `sas_plus::Task`. The API will change; see [docs/design.md](docs/design.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
