"""Unit 2 baseline vs Unit 3 optimized, plus serial vs parallel invariance."""
import pandas as pd
import numpy as np
import pytest

from baseline.src.simulation import run_simulation as baseline_run
from src.simulation import run_simulation as optimized_run
from src.common import load_config, ROOT
from src.fast_kernel import evaluate_family_fast
from src.methods import estimate_fdr, reject_fixed, reject_bh
from src.metrics import evaluate_rejections

KEY = ["scenario_id", "replicate_id", "method"]


def normalize(path):
    return pd.read_csv(path).sort_values(KEY).reset_index(drop=True)


def small_config():
    conf = load_config(ROOT / "configs/smoke.json")
    return dict(conf, m=100, pi0_values=[.5], mu_values=[2.],
                gamma_values=[.001, .01], replicates=12)


def test_baseline_optimized_and_parallel(tmp_path):
    conf = small_config()
    original = normalize(baseline_run(conf, tmp_path / "baseline", force=True))
    serial = normalize(optimized_run(conf, tmp_path / "serial", force=True, workers=1))
    parallel = normalize(optimized_run(conf, tmp_path / "parallel", force=True, workers=2))
    pd.testing.assert_frame_equal(original, serial, check_exact=False, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(serial, parallel, check_exact=False, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("p,truth", [
    ([.01, .01, .9, .001], [True, False, True, False]),
    ([0., 0., 1., 1.], [True, False, True, False]),
    ([.6, .8, .9, .95], [True, False, True, False]),
])
def test_fast_kernel_matches_unit2_methods(p, truth):
    p = np.array(p)
    truth = np.array(truth)
    gamma_values = [.001, .01, .1]
    fast = evaluate_family_fast(p, truth, gamma_values, .5)
    for gamma, (estimate, fixed, bh, threshold) in zip(gamma_values, fast):
        old_estimate = estimate_fdr(p, gamma, .5)
        for key in estimate:
            assert estimate[key] == pytest.approx(old_estimate[key], abs=1e-12)
        expected_fixed = evaluate_rejections(reject_fixed(p, gamma), truth)
        expected_bh_mask, expected_threshold = reject_bh(p, old_estimate["fdr_hat"])
        expected_bh = evaluate_rejections(expected_bh_mask, truth)
        assert fixed == expected_fixed
        assert bh == expected_bh
        assert threshold == pytest.approx(expected_threshold)
