# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1](https://github.com/sunsided/planning-heuristics/compare/v0.2.0...v0.2.1) - 2026-11-22

### Changed

- Link the crate to its page on the planning and search site

## [0.2.0](https://github.com/sunsided/planning-heuristics/compare/v0.1.0...v0.2.0) - 2026-11-22

### Added

- Allow black variables with conditional effects in red-black
- Add an optimal cost partitioning landmark tracker
- Support axioms in h²
- Add preferred operators to h_FF

### Changed

- Check compiled red-black paintings against the compiled task
- Reject oversized exact contexts before building them
- Compile conditional effects exactly for red-black planning
- Read derived effect conditions red-black in the real state
- Make executable black moves an option
- Prefer black moves whose red conditions hold under real semantics
- Keep two values of a derived variable mutex in h²
- Skip effects and axioms that can never fire in the relaxation

### Removed

- Remove H2Error::Axioms

## [0.1.0](https://github.com/sunsided/planning-heuristics/releases/tag/v0.1.0) - 2026-11-22

### Added

- Add tables of contents to the README and the design notes
- Add CONTRIBUTING.md
- Add README badges, a usage example and absolute links
- Add cargo-fuzz targets for the relaxation heuristics and LM-cut
- Add REUSE licensing metadata
- Add cargo-deny, Dependabot and typos configuration
- Add the codecov configuration
- Add a Taskfile and CI workflows that mirror each other
- Add the goal-intention heuristics ol_rp and op_rp
- Add the relevance score heuristic
- Add the fixed-number-of-black-variables red-black fragment
- Add the remaining red-black painting strategies
- Add red facts following to red-black
- Add relaxed plan forwarding and the stop search test to red-black
- Allow a black causal DAG in red-black repair
- Add operator counting with the state equation and LM-cut
- Add h^2, the set-additive heuristic and additive h_max
- Add causal-graph-level and conflict-directed paintings
- Add RHW landmarks, disjunctive landmarks and reasonable orderings
- Add optimal landmark cost partitioning behind the lp feature
- Add the LM-cut heuristic
- Add abstraction heuristic adapters
- Add the red-black heuristic with relaxed plan repair
- Add h^1 fact landmarks and two landmark heuristics
- Add a criterion bench for the relaxation heuristics
- Implement Heuristic only for u32 and document h_FF above FD
- Add the design document
- Add h_max, h_add and h_FF over sas_plus::Task

### Changed

- Set the crate version to 0.1.0
- Describe the release setup as it stands after the first release
- Point the SBOM submission at the generated file
- Report dependencies and workflow findings to GitHub code security
- Describe the released scope and audience in the README
- Raise the MSRV to 1.86 and check every feature on it
- Stop tracking Cargo.lock
- Restore the crates.io lockfile entries of the sibling crates
- Split three long test checks into documented helpers
- Split the red-black product and table constructors into helpers
- Adopt the full lint table, clippy.toml and rustfmt.toml
- Depend on the published sibling crates instead of paths
- Keep repository tooling out of the published crate
- Render mermaid diagrams in the rustdoc output
- Test the no_std ln against std on every positive f64
- Spell out the goal-intention probability locals
- Track Cargo.lock
- Require width-search 0.3 for the h_add comparison
- Require search-core 0.3
- Require search-core 0.2
- Require search-core 0.1
- Accept search-core 0.0 and 0.1 during the 0.1.0 release
- Compare the relevance score and goal intention against their papers
- Bench the relevance score and goal intention
- Bench seeded LM-cut against rebuilding the task per call
- Evaluate LM-cut from seeded fact costs and per-call operator costs
- Record the red facts following hang fix in the design table
- Qualify red-black soundness claims and bench on one painting
- Bench red facts following
- Document the remaining red-black work against the papers
- Record the RSE and strong connectivity fixes in the design table
- Ignore conditional side effects in RSE-invertibility
- Compare red-black against the papers in the design
- Paint red-black variables as the papers do
- Keep DAG red-black repair on visited black values
- Split conditional operators by context in LM-cut
- Check that every painting strategy matters on its own
- Test the raw LP optimum, document examples, bench landmark tracking
- Describe LM-cut, LP partitioning, RHW, tracking and paintings as implemented
- Track landmarks along the search path
- Use search-core's Heuristic trait
- Test red-black repair with two black variables
- Describe red-black, landmarks and the adapters as implemented
- Move the landmark docs onto the public types
- Record the axiom closure profile and numbers in the design
- Take the start fact count from the exploration
- Skip the start facts in the relaxed plan replay
- Index the axiom closure of the relaxed plan replay
- Record the profile and the bucket queue numbers in the design
- Reset the workspace in O(1) with epoch stamps
- Use a bucket queue and packed slots in the worklist
- Match heuristic-search's trait: forwarding impls and heuristic_fn
- Split shared relaxation tables from per-thread heuristics
- Record benchmark numbers in the design
- Reuse working memory and stop the worklist at the goal
- Compare h_add with width-search's implementation
- Test u32 saturation and correct the adapter and LM-cut notes
- Test the relaxation heuristics against independent oracles
- Reserve the crate name with a 0.0.0 placeholder

### Fixed

- Fix a hang in red facts following on repeated operators

### Removed

- Drop the stale notes on unreleased sibling versions
