# Optimization and Validation — Unit 3

## Experimental protocol

Preserve all primary Unit 2 statistical parameters, labels, RNG streams, scenarios, estimands and raw schema. Compare only uncached computation, with same machine, Python environment, config and controlled number of workers. For every table record median of ≥3 independent runs. See `results/performance/` for artifacts.

## Optimization 1: Reuse shared sorted arrays and eliminate repeated scans

**Problem.** Original `src/methods.py` computed a new fixed rejection mask and rescanned p-values for each gamma, then rescanned truth labels for each method; the simulation driver converted the same condition DataFrame into dictionaries on each replicate.

**Before:** `estimate_fdr(p,gamma,...)`, `reject_fixed(p,gamma)`, `reject_bh(p,alpha,order)`, and `evaluate_rejections(rejected,is_null)` inside both nested loops.

**After:** `src/fast_kernel.py`: one stable sort, shared cumulative null counts, one `W` and null-proportion estimate, `searchsorted` for fixed rejection counts, broadcasting for BH eligibility at all gamma values. `src/simulation.py` prepares condition dictionaries once per block.

**Impact:** TODO measured serial baseline vs serial optimized runtime and speedup.

**Tradeoffs:** More compact but less transparent vectorized code, an O(mG) eligibility matrix, extra memory for sorted arrays and cumulative counts. Full asymptotic complexity still O(m log m + Gm); performance gain is primarily reduced constant factors.

## Optimization 2: Parallelize independent (mu, pi0) blocks

**Problem.** Independent blocks ran serially.

**Before:** Outer nested loops evaluate each block on one core.

**After:** `ProcessPoolExecutor` maps 27 independent blocks to worker processes. Workers return ordered block rows; the main process alone writes CSV, preserving row ordering and the immutable family-specific RNG streams.

**Impact:** TODO measured wall times for 1/2/4/(8 if available) workers and speedup/parallel efficiency.

**Tradeoffs:** Process startup, inter-process serialization of rows, temporary memory pressure; multi-process may be slower on small pilot workloads.

## Results

Attach `results/performance/benchmark.png`, `parallel_speedup.png`, `complexity.png`; include measured table for baseline serial, optimized serial and parallel, plus empirical slopes.

## Correctness

Record `make test` output, `make regression` output and differences. Regression test compares per-row results after sorting by scenario, replicate and method and checks single/multiple worker parity. Note any rounding/tolerance issue explicitly.

## Stability

Document warning/exception/nonfinite counts from baseline *and* optimized runs. `make stability-check` currently measures the optimized kernel; baseline behavior needs separate observation. Mention `norm.sf`, `log1p`, `expm1` already present in Unit 2.

## Lessons learned

- Which optimization improved wall-clock time most? **TODO**
- What was unexpected in the profiler? **TODO**
- Was parallelization worthwhile at small problem size? **TODO**
