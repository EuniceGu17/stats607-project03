# Baseline Performance — Unit 2 (Storey 2002)

> Fill measured values from your own machine. Do **not** cite 10.1 s from the Unit 2 README as a new measurement.

## Study and environment

- Original source: `baseline/src/` copied before any Project 3 edits.
- Full configuration: `configs/full.json`: m=1000, 27,000 independent families, 54 scenarios, 1000 replicates/scenario and 108,000 method-level records.
- Platform / CPU / OS: **TODO**
- Python / dependency versions: **TODO**, see `results/performance/environment.json`

## Full-study runtime

| Stage | Time (seconds) | Measurement method |
|---|---:|---|
| Baseline simulation | TODO | `make baseline CONFIG=configs/full.json` |
| Analysis | TODO | timed Unit 2 analysis |
| Figures | TODO | timed Unit 2 figures |
| End-to-end | TODO | complete pipeline excluding installation |

## Profiler evidence

Link to `results/performance/baseline_profile.txt`. Summarize the top 5 functions by *cumulative* time and identify the causes. `cProfile` can attribute compiled NumPy work to Python wrapper calls; report that limitation.

| Function | Cumulative seconds | Interpretation |
|---|---:|---|
| TODO | TODO | TODO |

## Complexity

Let m = tests/family, G = # gamma thresholds, B = # family replicates.
- DGP: O(m) per family.
- Sorting: O(m log m).
- Repeated FDR estimation, threshold masks, BH checks, and metrics: O(Gm).
- Full simulation compute: O(B[m log m + Gm]), plus serialization/writing.

Fit empirical log-log slope using `make complexity` *after* the optimized version is installed. Present results of both implementations and note fixed process startup/I/O costs, especially for small m.

## Numerical behavior of baseline

Document the **baseline** warning/exception counts observed on the profiling/stress configs; do not substitute optimized results. Describe setting, frequency, and any known cause. The Unit 2 code already uses `norm.sf`, `log1p` and `expm1` for improved numerical stability.

## Bottlenecks to address

1. Recompute condition dictionaries in the replicate loop.
2. Multiple scans per gamma and method.
3. Temporary rejection masks and repeated truth comparisons.
4. Independent (mu, pi0) blocks executed serially.

Be explicit which were *actually confirmed* by profiling and which were hypothesized from static inspection.
