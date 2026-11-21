---
applyTo: ".github/workflows/**,Taskfile.dist.yaml,**/Taskfile.dist.yaml,.pre-commit-config.yaml"
---

# CI, Taskfile and hooks stay in sync

- The `ci` task in `Taskfile.dist.yaml` mirrors the jobs in `.github/workflows/ci.yml`. A
  change that adds, removes or renames a CI job, or changes its flags, must make the same
  change to `task ci` and the task it calls, and the other way round. Flag any drift in review.
- Local hooks in `.pre-commit-config.yaml` call Taskfile tasks. They must not duplicate
  commands.
- Every `uses:` is pinned to a full 40-character commit SHA, with a `# vX.Y.Z` comment. Reject
  tag or branch refs.
- Local actions keep the `./` form with `# zizmor: ignore[self-repository]`: actionlint 1.7.12
  rejects the `$/` form.
- `dtolnay/rust-toolchain` is pinned to a `master` commit (`# master`) and always gets an
  explicit `toolchain:` input. Its `stable`/`nightly` branches are not tags, and Dependabot
  cannot follow them: it re-pinned the `nightly` use to the `stable` commit.
- The workflow sets `permissions: {}` at the top, and each job sets the least privilege it
  needs. Every `actions/checkout` sets `persist-credentials: false`.
- No `pull_request_target`. No `${{ github.event.* }}` or other untrusted expressions
  directly in `run:`. Pass them through `env:`.
- Do not rename `publish-crates-io.yml` or its `crates-io` environment. crates.io trusted publishing is
  bound to both. Deploy jobs run in their own tag-restricted environment, and `needs:` a
  verification job.
- CI jobs: `branch`, `fmt`, `clippy`, `style`, `doc`, `test` (Linux only), `coverage`,
  `deny`, `msrv`, `semver`, `fuzz`, plus `actionlint` and `zizmor` in `actions-lint.yml`.
- `msrv` checks the library only (`cargo hack check --rust-version --each-feature
  --exclude-features lp,cost-partition`, no `--all-targets`): proptest 1.12 and width-search
  need a newer Rust than the declared 1.85, and the `lp` and `cost-partition` features need
  1.86 (README, MSRV). Do not add `--all-targets` or those features back.
- `fuzz/` is a separate nightly workspace. Its tasks live in `fuzz/Taskfile.dist.yaml`,
  included as `fuzz:`; CI's `fuzz` job mirrors `task fuzz:build` and `task fuzz:lint`.
- `Cargo.lock` and `fuzz/Cargo.lock` are tracked, but no step uses `--locked`: CI builds
  against the default branches of the siblings, so a locked sibling version would fail every
  job after its next release.
- `search-core`, `merge-and-shrink`, `cost-partition` and `width-search` are path
  dependencies. Jobs that resolve dependencies check this repository out into
  `planning-heuristics/`, call `.github/actions/checkout-siblings`, and run from
  `planning-heuristics/` (`defaults.run.working-directory`, plus the path inputs of
  `uses:` steps: rust-cache `workspaces`, `manifest-path`, codecov `working-directory`).
  A new sibling dependency goes into that action.
- `cargo fmt` never gets `--all`: it would also format the path dependencies.
- `scripts/check-branch.sh` is the single definition of allowed branch names. It is used by
  the Taskfile, the hook and the CI `branch` job.
- New jobs that should gate merges must also be added to branch protection. Mention that in
  the PR.
