"""Check submission artifacts and compare local Unit 2/3 results when present."""
from pathlib import Path
import hashlib
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md", "Makefile", "requirements.txt", "ADEMP.md", "ANALYSIS.md",
    "docs/BASELINE.md", "docs/OPTIMIZATION.md", "src/fast_kernel.py",
    "src/performance.py", "src/stability_check.py", "tests/test_regression.py",
    "results/performance/benchmark.csv", "results/performance/complexity.csv",
    "results/performance/stability.csv", "results/performance/baseline_profile.txt",
    "results/performance/optimized_profile.txt", "results/performance/benchmark.png",
    "results/performance/parallel_speedup.png", "results/performance/complexity.png",
]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda:fh.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    failed=[]
    for item in REQUIRED:
        p=ROOT/item
        if not p.is_file(): failed.append('Missing '+item)
    for name in ('BASELINE.md', 'OPTIMIZATION.md'):
        text=(ROOT/'docs'/name).read_text() if (ROOT/'docs'/name).exists() else ''
        if 'TODO' in text: failed.append(f'{name} still contains TODO')
    bench=ROOT/'results/performance/benchmark.csv'
    if bench.exists():
        x=pd.read_csv(bench)
        if not ({('baseline',1),('optimized',1),('optimized',2),('optimized',4)} <= set(zip(x.engine,x.workers))):
            failed.append('Benchmark is missing required implementation/worker comparisons')
        if not (x.groupby(['engine','workers']).size()>=3).all():
            failed.append('At least one benchmark group has fewer than 3 timings')
    stab=ROOT/'results/performance/stability.csv'
    if stab.exists():
        x=pd.read_csv(stab)
        print('Stability trials:',len(x),'warnings:',x.warnings.sum(),
              'nonfinite:',x.nonfinite.sum(),'exceptions:',x.exception.notna().sum())
    raw_a=ROOT/'results/unit2-baseline-measurement/raw/replicates.csv'
    raw_b=ROOT/'results/project03-full/raw/replicates.csv'
    if raw_a.exists() and raw_b.exists():
        if digest(raw_a)!=digest(raw_b): failed.append('Full raw baseline/optimized SHA-256 mismatch')
        else: print('Full raw files: byte identical')
    else:
        print('NOTE: large local raw files not both present; equivalence should be rerun on a machine with full data')
    sum_a=ROOT/'results/full/summaries/summary.csv'
    sum_b=ROOT/'results/project03-full/summaries/summary.csv'
    if sum_a.exists() and sum_b.exists():
        try:
            pd.testing.assert_frame_equal(pd.read_csv(sum_a),pd.read_csv(sum_b),rtol=1e-12,atol=1e-12)
            print('Full summary files: equivalent')
        except AssertionError as e: failed.append('Full summary mismatch: '+str(e)[:200])
    else:
        print('NOTE: one or both full summary files absent')
    for msg in failed:print('FAIL:',msg)
    if failed: return 1
    print('PASS: artifact checklist and available output comparisons')
    return 0

if __name__=='__main__':sys.exit(main())
