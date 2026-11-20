# Test fixtures

`output.sas` files (Fast Downward translator format, version 3) copied unchanged from
[sunsided/sas-plus](https://github.com/sunsided/sas-plus) `tests/fixtures/`. Only fixtures
written for sas-plus are included; they fall under its licence (MIT OR Apache-2.0). The
GPL-2.0 VHPOP-derived fixtures of sas-plus are deliberately not copied.

| Fixture | Source in sas-plus | Features | Optimal cost |
|---|---|---|---|
| `gripper-costs` | `gripper-costs` (hand-encoded) | action costs | 7 |
| `gripper-unit` | `gripper-unit` (hand-encoded) | `metric 0` | 5 |
| `conditional-effects` | `conditional-effects` (hand-encoded) | conditional effects | not checked |
| `derived-predicates` | `derived-predicates` (hand-encoded) | one axiom | 2 |
| `translated-courier` | `translated/courier` | costs, conditional effects | 16 |
| `translated-gripper-costs` | `translated/gripper-costs` | action costs | 7 (not checked here) |
| `translated-wiring` | `translated/wiring` | costs, axioms | 4 |
