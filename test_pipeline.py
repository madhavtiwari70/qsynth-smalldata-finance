"""
Test Suite for IQP Finance Synth
==================================
Tests the preprocessor, evaluator, and baselines without requiring iqpopt.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from iqp_finance.preprocessor import FinancialDataPreprocessor
from iqp_finance.evaluator import compute_mmd_squared, SyntheticEvaluator
from iqp_finance.baselines import BernoulliBaseline


# ================================================
# Fixtures
# ================================================

@pytest.fixture
def demo_df():
    """Small demo financial DataFrame."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "amount":    rng.exponential(100, size=n),
        "risk":      rng.beta(2, 5, size=n),
        "age":       rng.integers(18, 80, size=n).astype(float),
        "velocity":  rng.exponential(3, size=n),
    })


@pytest.fixture
def demo_binary():
    """Small binary dataset with two classes."""
    rng = np.random.default_rng(42)
    n, d = 100, 16
    X0 = (rng.random((n // 2, d)) > 0.7).astype(float)
    X1 = (rng.random((n // 2, d)) > 0.3).astype(float)
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


# ================================================
# Preprocessor tests
# ================================================

class TestFinancialDataPreprocessor:

    def test_quantile_binarization_shape(self, demo_df):
        pp = FinancialDataPreprocessor(n_qubits=16, binarize_method="quantile", n_quantiles=4)
        X_bin, feat_map = pp.fit_transform(demo_df)
        assert X_bin.shape == (100, 16), f"Expected (100, 16), got {X_bin.shape}"

    def test_threshold_binarization_binary(self, demo_df):
        pp = FinancialDataPreprocessor(n_qubits=16, binarize_method="threshold")
        X_bin, _ = pp.fit_transform(demo_df)
        assert np.all(np.isin(X_bin, [0.0, 1.0])), "Output must be binary"

    def test_zscore_binarization(self, demo_df):
        pp = FinancialDataPreprocessor(n_qubits=16, binarize_method="zscore")
        X_bin, _ = pp.fit_transform(demo_df)
        assert X_bin.shape[1] == 16

    def test_inverse_transform_shape(self, demo_df):
        pp = FinancialDataPreprocessor(n_qubits=16, binarize_method="quantile", n_quantiles=4)
        X_bin, _ = pp.fit_transform(demo_df)
        X_rec = pp.inverse_transform(X_bin)
        assert X_rec.shape == (100, pp.n_features_used_)

    def test_density_reasonable(self, demo_df):
        pp = FinancialDataPreprocessor(n_qubits=16, binarize_method="threshold")
        X_bin, _ = pp.fit_transform(demo_df)
        densities = pp.get_density_per_feature(X_bin)
        for feat, d in densities.items():
            assert 0.0 <= d <= 1.0, f"Density out of range for {feat}: {d}"

    def test_unfitted_raises(self, demo_df):
        pp = FinancialDataPreprocessor(n_qubits=16)
        with pytest.raises(RuntimeError):
            pp.transform(demo_df)

    def test_feature_map_keys(self, demo_df):
        pp = FinancialDataPreprocessor(n_qubits=16, binarize_method="quantile", n_quantiles=4)
        _, feat_map = pp.fit_transform(demo_df)
        # Every qubit in feat_map should have a feature name
        for qubit_idx, info in feat_map.items():
            assert "feature" in info
            assert "bit_position" in info

    def test_numpy_array_input(self):
        rng = np.random.default_rng(0)
        X = rng.random((50, 4))
        pp = FinancialDataPreprocessor(n_qubits=8, binarize_method="threshold")
        X_bin, _ = pp.fit_transform(X)
        assert X_bin.shape == (50, 8)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            FinancialDataPreprocessor(n_qubits=16, binarize_method="invalid")

    def test_non_power_of_2_quantiles_raises(self):
        with pytest.raises(ValueError):
            FinancialDataPreprocessor(n_qubits=16, binarize_method="quantile", n_quantiles=3)


# ================================================
# MMD² tests
# ================================================

class TestMMD:

    def test_mmd_identical_distributions(self):
        rng = np.random.default_rng(42)
        X = (rng.random((100, 16)) > 0.5).astype(float)
        mmd = compute_mmd_squared(X, X, sigma=3.0)
        assert mmd < 1e-10, f"MMD² of identical distributions should be ~0, got {mmd}"

    def test_mmd_different_distributions(self):
        rng = np.random.default_rng(42)
        X1 = (rng.random((100, 16)) > 0.7).astype(float)  # sparse
        X2 = (rng.random((100, 16)) > 0.3).astype(float)  # dense
        mmd = compute_mmd_squared(X1, X2, sigma=3.0)
        assert mmd > 0, "MMD² of different distributions should be > 0"

    def test_mmd_non_negative(self):
        rng = np.random.default_rng(0)
        X1 = (rng.random((50, 16)) > 0.5).astype(float)
        X2 = (rng.random((50, 16)) > 0.5).astype(float)
        mmd = compute_mmd_squared(X1, X2, sigma=3.0)
        assert mmd >= 0, "MMD² must be non-negative"

    def test_mmd_symmetric(self):
        rng = np.random.default_rng(1)
        X1 = (rng.random((60, 16)) > 0.6).astype(float)
        X2 = (rng.random((60, 16)) > 0.4).astype(float)
        mmd_12 = compute_mmd_squared(X1, X2, sigma=3.0)
        mmd_21 = compute_mmd_squared(X2, X1, sigma=3.0)
        assert abs(mmd_12 - mmd_21) < 1e-10, "MMD² must be symmetric"


# ================================================
# Bernoulli Baseline tests
# ================================================

class TestBernoulliBaseline:

    def test_fit_and_sample(self, demo_binary):
        X, y = demo_binary
        baseline = BernoulliBaseline(random_seed=42)
        baseline.fit(X, y)
        X_synth, y_synth = baseline.sample(n_samples=50)
        assert X_synth.shape == (50, 16)
        assert len(y_synth) == 50
        assert np.all(np.isin(X_synth, [0.0, 1.0]))

    def test_class_probs_range(self, demo_binary):
        X, y = demo_binary
        baseline = BernoulliBaseline()
        baseline.fit(X, y)
        for c, probs in baseline.class_probs_.items():
            assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_single_class_sampling(self, demo_binary):
        X, y = demo_binary
        baseline = BernoulliBaseline()
        baseline.fit(X, y)
        X_synth, y_synth = baseline.sample(n_samples=30, class_label=0)
        assert np.all(y_synth == 0)
        assert len(X_synth) == 30

    def test_unfitted_raises(self, demo_binary):
        baseline = BernoulliBaseline()
        with pytest.raises(RuntimeError):
            baseline.sample(10)


# ================================================
# Evaluator tests
# ================================================

class TestSyntheticEvaluator:

    def test_evaluate_returns_keys(self, demo_binary):
        X, y = demo_binary
        rng = np.random.default_rng(42)
        X_synth = (rng.random(X.shape) > 0.5).astype(float)
        y_synth = rng.choice([0, 1], size=len(y))

        evaluator = SyntheticEvaluator(sigma=3.0)
        results = evaluator.evaluate(X, X_synth, y, y_synth)

        assert "mmd2_overall" in results
        assert "mmd2_per_class" in results
        assert "feature_parity" in results
        assert "correlation_similarity" in results
        assert "tstr" in results
        assert "class_balance" in results

    def test_correlation_similarity_range(self, demo_binary):
        X, y = demo_binary
        evaluator = SyntheticEvaluator()
        sim = evaluator._compute_correlation_similarity(X, X)
        assert 0.99 <= sim <= 1.01, f"Self-similarity should be ~1.0, got {sim}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
