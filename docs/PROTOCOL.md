# Evaluation Protocol

CivWorldBench uses a structured observation history to evaluate quantity and
event forecasting (L1), hidden traits and world rules (L2), and intervention
prediction (L3). `bench/spec_v2.py` and `bench/tasks_v2.py` define the task schemas;
`bench/parse_v2.py` and `bench/score_v2.py` define parsing and scoring.

## Input Contract

Dataset: **In preparation**. The following describes the file contract; no
benchmark records or sample identifiers are included in this repository.

Native question and response files use gzip-compressed JSON Lines. Corresponding
files share an anchor filename beneath `bench/runs/<run>/questions/` and
`bench/runs/<run>/<raw-name>/`. Join records using the anchor, condition, task, and
intervention identifiers rather than line position.

Question fields include `world`, `t1`, `set`, `cond`, `task`, `prompt`, and `truth`.
Intervention rows also use `op`, `sub`, `arm_kind`, and `control_ref`. Only the
model-visible prompt belongs in inference context. The remaining fields include
evaluation metadata and targets.

| File label | Task | Answer contract |
| --- | --- | --- |
| `1a` | Quantity | Five quantiles per civilization, variable, and horizon |
| `1b` | Event | Event probabilities or a categorical distribution |
| `2` | Trait, rules, and auxiliary attitude probes | Rankings or classes |
| `3` | Intervention | Quantity or event forecasts under the specified intervention |

`bench-basic/v1`, `bench-basic/v2`, and `bench-gen/v1` identify the fully observed,
partially observed, and generalization settings. Their published assignments
must be preserved for a paper comparison; an independently collected corpus is
a separate evaluation.

## Retained Constants

- `bench/v2/norms.json` contains only the scorer's `norm` value for each variable.
  Empirical percentiles, ranges, and sample counts have been omitted.
- `bench/v2/mechanism_cards.json` contains engine-based intervention descriptions,
  not observations, arm outcomes, or world-specific values.
- `bench/measure/out/section5/swfi_v1/spec.json` contains the Sandbox-World
  Forecasting Index (SWFI) weights, reference endpoints, and aggregation policy.
  It is a metric definition, not a model-results table.

The scorer additionally requires matching world-to-ruleset assignments,
climatology and event-reference calibration files. These empirical artifacts
are not in the code release. `bench/measure/v2_cells.py` provides their
construction code for independently collected inputs; recalibration changes the
evaluation population and does not reproduce the paper.

For any new scoring run, use a distinct output `--tag`. A subset evaluation must
also specify its own `--anchors` file and preserve the matching factual/control
answers. Missing or unparseable answers are counted as failures by default.

## Renderer Version

The released renderer includes the interval-boundary correction: unit movement
in one news interval starts at that interval's first snapshot. The paper's
reported scores retain their original frozen evaluation inputs. Future dataset
availability and version information will be documented separately.
