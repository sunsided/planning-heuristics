---
applyTo: "**/*.rs"
---

# Rust code

- Every module (`//!`), type, field, variant, function, constant and macro has a doc
  comment, private and `pub(crate)` items included. Docs state intent, invariants, units and
  error/panic conditions (`# Errors`, `# Panics`, `# Safety`). They do not restate the name.
- No `mod.rs`: modules with children are `foo.rs` plus `foo/`. Integration tests live under
  `tests/it/`.
- No separator comments (`// -----`, `// =====`). If a file needs sections, split it into
  submodules.
- Leave a blank line after the closing `}` of a block statement when another statement
  follows, and between logical phases of a function. No lumped code.
- One level of abstraction per function. Long loops (simulations, game ticks, solvers) are
  short drivers that call documented step functions. If a block needs a "what" comment,
  extract it into a function named after that comment.
- `clippy::pedantic` clean. Suppression only with a scoped
  `#[expect(lint, reason = "...")]`. Never `#[allow]`, and never crate-level suppression.
- No `unsafe` unless the crate opts out of `#![forbid(unsafe_code)]` with a documented
  reason. Each unsafe block has one operation and a `// SAFETY:` comment.
- The crate is `#![no_std]` with `alloc`. Flag any `std` use outside the `lp` feature.
  `OptimalLandmarkCost` and `OperatorCounting` sit behind `lp`, the adapters behind
  `merge-and-shrink` and `cost-partition`; flag a feature-gated item without its `#[cfg]` on
  the re-export in `lib.rs`.
- Heuristics implement `search_core::Heuristic` (re-exported) and keep the inherent
  `value(&State) -> Option<u64>`; `None` means a proven dead end, never "unknown".
  Admissibility claims are tested against `h*` from the backward Dijkstra in `tests/it/`.
- A new heuristic gets a check in `tests/it/` over every reachable state of the fixtures and
  the seeded random tasks: dead ends against relaxed reachability or `h*`, its ordering
  against the existing heuristics (`h_max <= h_FF <= h_add`, `h_max <= h_LM-cut <= h*`), and
  an independent oracle where one exists. The fuzz targets check the orderings too.
- Public API changes go with a matching update to the README scope list, the crate docs in
  `lib.rs` and `docs/design.md`.
- Mermaid diagrams in doc comments must match the code they describe. Update them when the
  structure changes.
