# TRT-LLM MoE Pressure Lab

MoE pressure modeling, admission planning, and runtime scheduling ablations for TensorRT-LLM on a fixed `Qwen/Qwen1.5-MoE-A2.7B-Chat` `INT4 weight-only` engine path.

## Abstract

This repository presents a fuller MoE runtime study than a single scheduler heuristic. It starts from the observation that MoE routing skew changes the shape of runtime cost, not merely its mean. Generic scheduling knobs optimize utilization and memory fit, but they do not explicitly control expert or rank pressure. The project therefore introduces an explicit pressure-aware runtime model, validates a first latency-oriented planner (`v1`), and then extends it to a more architecture-like admission and prefill-control design (`v2`).

The result is a structured study of how MoE pressure should influence runtime decisions under real TensorRT-LLM execution.

## Problem Formulation

For MoE inference, token count alone is not a sufficient proxy for runtime cost. Two requests with similar lengths may induce very different step behavior because of:

- expert concentration
- rank concentration
- uneven communication and synchronization pressure
- prefill/decode interaction under skewed batches

This motivates a batch model with two distinct components:

```text
L(B) = L_compute(T(B)) + L_contention(P(B), S(B))
```

where:

- `T(B)` is token-driven compute cost
- `P(B)` is aggregate pressure
- `S(B)` is skew structure, e.g. expert or rank concentration

The runtime question is therefore not only “does the batch fit?”, but also “is this batch composition pressure-safe?”

## Project Objective

The objective is to move MoE pressure into the runtime decision path through progressively stronger mechanisms:

1. explicit pressure representation
2. explicit runtime resource modeling
3. pressure-aware step planning (`v1`)
4. pressure-aware admission and prefill control (`v2`)

The project keeps the following path fixed:

- model: `Qwen/Qwen1.5-MoE-A2.7B-Chat`
- quantization: TensorRT-LLM `INT4 weight-only`
- hardware: `RTX 4060 Ti 16GB`

## Method

### 1. Pressure Model

Implemented in [`scheduler/moe_pressure.py`](scheduler/moe_pressure.py).

Requests are mapped to pressure classes:

- `balanced`
- `hot_expert`
- `hot_rank`

Each request also carries a scalar score. The score is deliberately simple: it is not intended to recover the full routed-expert graph, but to provide a robust scheduling signal that can be composed at the batch level.

### 2. Runtime Resource Model

Implemented in [`scheduler/resource_model.py`](scheduler/resource_model.py).

The runtime model introduces:

- `RequestProfile`
- `RuntimeBudget`
- `StepPlan`

This turns scheduling into a structured pipeline:

`request metadata -> request profile -> runtime budget -> step plan -> execution`

The change matters because it lets subsequent policies operate on explicit runtime objects rather than hidden implicit assumptions.

### 3. V1: Pressure Dispersion

Implemented in [`scheduler/moe_microbatch_scheduler.py`](scheduler/moe_microbatch_scheduler.py).

`v1` is designed to answer the first question: is MoE pressure worth exposing at all?

Its policy is:

- decode-first
- avoid stacking multiple hot requests in one step
- accept throughput loss if that is necessary to reduce tail risk

This is intentionally conservative. It is expected to improve latency while sacrificing some batching efficiency.

### 4. V2: Capacity Admission and Adaptive Chunking

Implemented in:

- [`scheduler/moe_capacity_scheduler.py`](scheduler/moe_capacity_scheduler.py)
- [`scheduler/adaptive_chunking.py`](scheduler/adaptive_chunking.py)

`v2` extends the project in two ways.

#### Admission / capacity reasoning

Instead of merely dispersing requests, `v2` assigns each pending record a utility-like score:

```text
score(r) = prefix_bonus(r) - alpha * pressure(r) - beta * token_cost(r) - gamma * hot_rank_penalty(r)
```

and selects a batch under:

- max batch size
- token budget
- dynamic pressure budget

This is the main architectural step forward relative to `v1`.

#### Adaptive prefill / chunking

For repeated-prefix and mixed-pressure cases, `v2` also adapts:

- effective microbatch size
- effective scheduler token limit

This prevents chunked prefill from being inserted too aggressively when decode pressure is already high.

### 5. Replay Validation

Implemented in [`scheduler/replay_pressure_provider.py`](scheduler/replay_pressure_provider.py) and [`scripts/generate_pressure_traces.py`](scripts/generate_pressure_traces.py).

Replay mode is used to check whether conclusions remain stable when pressure metadata is attached from an offline trace representation rather than directly from the synthetic workload label.

## Why These Changes Should Be Effective

The theoretical argument is modest but useful.

Assume:

1. contention cost is increasing with aggregate pressure
2. near hotspot regimes, contention cost is approximately convex

Then, for a fixed total amount of hot traffic, concentrating many hot requests into the same step increases the tail more than distributing them across several steps. This is a standard convexity argument:

- concentration raises the maximum cost disproportionately
- dispersion lowers the tail at the expense of some batching efficiency

This explains the observed `v1` behavior.

`v2` addresses the natural weakness of `v1`. If `v1` is too conservative, the batch collapses and throughput suffers. By introducing:

- explicit admission
- dynamic pressure budgets
- repeated-prefix preference
- adaptive chunking

`v2` creates an intermediate regime where pressure is controlled, but batching is not entirely abandoned.

This is not a formal proof of optimality for TensorRT-LLM internals. It is a principled scheduler argument that is consistent with the measured data.

## What Was Modified and Why

### [`scheduler/resource_model.py`](scheduler/resource_model.py)

Why it was changed:

- runtime assumptions needed to become explicit

What it contributes:

- structured request profiling
- explicit budgets
- explicit step plans

### [`scheduler/moe_microbatch_scheduler.py`](scheduler/moe_microbatch_scheduler.py)

Why it was changed:

- the runtime needed a first MoE-aware scheduling policy

What it contributes:

- decode-first pressure dispersion
- hot-request isolation

### [`scheduler/moe_capacity_scheduler.py`](scheduler/moe_capacity_scheduler.py)

Why it was changed:

- `v1` demonstrated that pressure matters, but it did not recover throughput

What it contributes:

- batch admission based on a combined score
- dynamic pressure budgeting
- prefix-sensitive preference

### [`scheduler/adaptive_chunking.py`](scheduler/adaptive_chunking.py)

Why it was changed:

- prefill should not be allowed to interfere with decode in the same way under all pressure states

What it contributes:

- pressure-sensitive chunking control
- repeated-prefix aware batch-shape adjustment

### [`scripts/run_full_matrix.sh`](scripts/run_full_matrix.sh)

Why it was changed:

- the full project needed reproducible matrix execution rather than one-off commands

What it contributes:

- sequential execution of baselines, replay generation, `v1`, `v2`, and end-to-end milestone runs

### [`scripts/collect_metrics.py`](scripts/collect_metrics.py)

Why it was changed:

- the conclusions needed to be supported by consistent summary tables

What it contributes:

- comparable metrics across baselines and scheduler variants
- machine-readable and markdown summary tables

## Experimental Protocol

### Fixed workloads

- `Balanced MoE`
- `Hot-Expert`
- `Hot-Rank`
- `Mixed Burst`
- `Repeated-Prefix under MoE Pressure`

### Baselines

- default batching
- `GUARANTEED_NO_EVICT`
- `MAX_UTILIZATION`
- overlap / chunked prefill baseline

### Candidate variants

- `v1 synthetic`
- `v1 replay`
- `v2 synthetic`
- `v2 replay`

## Results

### Strong baseline conclusion

Generic strong baselines were necessary controls, but they were not sufficient:

- they improved the balanced control slightly
- they did not materially improve hot-workload tails
- they did not address MoE-specific pressure structure

### Selected quantitative results

| Workload | Candidate | Baseline E2E p90 | Candidate E2E p90 | Baseline Throughput | Candidate Throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| Balanced | `MAX_UTILIZATION` | `1.4786s` | `1.4768s` | `280.39` | `282.72` |
| Balanced | `v2 replay` | `1.4786s` | `1.4541s` | `280.39` | `305.78` |
| Hot-Expert | `v1 replay` | `1.8421s` | `1.5698s` | `301.32` | `98.43` |
| Hot-Expert | `v2 replay` | `1.8421s` | `1.7928s` | `301.32` | `169.64` |
| Hot-Rank | `v1 replay` | `1.9107s` | `1.7123s` | `293.97` | `100.07` |
| Hot-Rank | `v2 replay` | `1.9107s` | `1.7186s` | `293.97` | `99.26` |
| Mixed Burst | `v2 replay` | `1.9723s` | `1.5660s` | `263.02` | `184.46` |
| Repeated-Prefix + Pressure | `v2 replay` | `1.7533s` | `1.2848s` | `242.35` | `130.94` |

### Reading the table

- `v1` proves that the pressure signal is useful
- `v2` is the main design result because it recovers part of the batching cost while preserving a meaningful pressure-aware effect
- the most compelling `v2` workloads are `Mixed Burst` and `Repeated-Prefix under MoE Pressure`
- `Hot-Rank` remains the hardest case for throughput recovery

## Main Conclusions

1. MoE pressure should be represented explicitly in the runtime decision path.
2. Generic utilization-oriented knobs are not enough for hotspot-heavy MoE traffic.
3. A pressure-aware planner can reduce tail latency substantially.
4. Moving from pure pressure dispersion to admission plus chunking control produces a more balanced runtime design.
5. The remaining open problem is throughput recovery under `Hot-Rank`.

## Limitations

The main limitation is explicit and important:

- the final quantitative path uses the real TensorRT engine backend
- the planner is externalized into batch composition for measurement
- the result is therefore not a pure in-backend PyTorch quantitative benchmark

That limitation defines the scope of the claim, but does not undermine the real engine data reported here.

## Repository Structure

- [`scheduler/`](scheduler): resource model, pressure model, admission, chunking, telemetry
- [`scripts/`](scripts): matrix execution, replay generation, metrics collection
- [`workloads/`](workloads): fixed MoE-specific workloads
- [`artifacts/moe_traces/`](artifacts/moe_traces): replay traces
- [`results/`](results): raw outputs and compare tables
- [`docs/`](docs): phase-by-phase notes, summary documents, final report

## Reproducibility

Matrix execution:

```bash
bash scripts/run_full_matrix.sh baseline-default
bash scripts/run_full_matrix.sh baseline-strong
bash scripts/run_full_matrix.sh traces
bash scripts/run_full_matrix.sh v1-synthetic
bash scripts/run_full_matrix.sh v1-replay
bash scripts/run_full_matrix.sh v2-ablation
bash scripts/run_full_matrix.sh qwen15-final
```

Summary generation:

```bash
bash scripts/wsl_env.sh python scripts/collect_metrics.py ...
```

Primary reference documents:

- [`docs/final_report.md`](docs/final_report.md)
- [`docs/result_summary.md`](docs/result_summary.md)
- [`results/compare_tables/selected_summary.md`](results/compare_tables/selected_summary.md)
