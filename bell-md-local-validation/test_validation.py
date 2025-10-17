# ============================================================================
# test_validation.py
"""Pytest unit tests for validation criteria."""

import pytest
import numpy as np
import pandas as pd
from numpy.random import default_rng
from analysis import (
    correlation_vs_angle_data, chsh_canonical_data,
    mi_violation_data, mi_scan_max_tv, witness_product_data
)

@pytest.fixture
def rng():
    return default_rng(42069)

class TestCorrelation:
    """Test correlation vs angle matches singlet prediction."""
    
    def test_correlation_zero_degrees(self, rng):
        df = correlation_vs_angle_data(np.array([0]), n_samples=10000, rng=rng)
        assert abs(df.iloc[0]['correlation'] - (-1.0)) < 0.01, "E[AB|θ=0] should be -1"
    
    def test_correlation_ninety_degrees(self, rng):
        df = correlation_vs_angle_data(np.array([90]), n_samples=10000, rng=rng)
        assert abs(df.iloc[0]['correlation'] - 0.0) < 0.02, "E[AB|θ=90] should be ~0"
    
    def test_correlation_within_tolerance(self, rng):
        df = correlation_vs_angle_data(np.linspace(0, 180, 13), n_samples=20000, rng=rng)
        # Allow 2% error for statistical variance
        assert (df['abs_error'] < 0.02).all(), "Correlation should match theory within 2%"

class TestCHSH:
    """Test CHSH violation."""
    
    def test_chsh_exceeds_classical(self, rng):
        result = chsh_canonical_data(n_samples=20000, rng=rng)
        assert result['chsh_S'] > 2.0, "CHSH must exceed classical bound of 2"
    
    def test_chsh_matches_quantum(self, rng):
        result = chsh_canonical_data(n_samples=50000, rng=rng)
        quantum = 2 * np.sqrt(2)
        assert abs(result['chsh_S'] - quantum) < 0.05, f"CHSH should be ~{quantum:.4f}"
    
    def test_chsh_within_ci(self, rng):
        result = chsh_canonical_data(n_samples=50000, rng=rng)
        quantum = 2 * np.sqrt(2)
        assert (result['chsh_ci_lower'] <= quantum <= result['chsh_ci_upper']), \
            "Quantum prediction should be within CI"

class TestMIViolation:
    """Test measurement independence violation."""
    
    def test_mi_violation_nonzero(self, rng):
        df = mi_violation_data(np.array([45]), n_samples=20000, rng=rng)
        assert df.iloc[0]['TV_distance'] > 0.05, "MI violation should be substantial"
    
    def test_mi_pairwise_matches_hall(self, rng):
        from analysis import mi_scan_max_tv
        result = mi_scan_max_tv(n_samples=80000, rng=rng)
        max_tv = result['max_TV']
        assert 0.12 <= max_tv <= 0.16, \
            f"Pairwise MI budget should match Hall (2010): 14% ±2%, got {max_tv:.4f}"
    
    def test_p_dependent_not_uniform(self, rng):
        df = mi_violation_data(np.array([30, 60]), n_samples=20000, rng=rng)
        # p_dependent should differ from p_uniform significantly
        assert all(df['MI_violation'] > 0.03), "p_dep should differ from p_uniform"

class TestWitnessProduct:
    """Test witness-product bound (Theorem 2)."""
    
    def test_bound_never_violated(self, rng):
        df = witness_product_data(
            grid_size=64, block_size=4,
            N_star_values=[64, 128, 256],
            n_trials=100, rng=rng
        )
        assert df['violations'].sum() == 0, "Witness-product bound should never be violated"
    
    def test_min_product_exceeds_bound(self, rng):
        df = witness_product_data(
            grid_size=64, block_size=4,
            N_star_values=[128, 256],
            n_trials=100, rng=rng
        )
        assert all(df['min_product'] >= df['bound']), "min(R₁·R₂) must be >= bound"
    
    def test_bound_scales_with_N_star(self, rng):
        df = witness_product_data(
            grid_size=64, block_size=4,
            N_star_values=[64, 128, 256, 512],
            n_trials=50, rng=rng
        )
        # Bound should increase with N_star
        assert all(df['bound'].diff().dropna() > 0), "Bound should increase with N★"
    
    def test_min_entropy_corollary(self, rng):
        df = witness_product_data(
            grid_size=64, block_size=4,
            N_star_values=[128, 256],
            n_trials=100, rng=rng
        )
        # H_∞(Π₁|W) + H_∞(Π₂|W) >= log₂(c)
        assert all(df['entropy_pass']), "Min-entropy sum should exceed log₂(c)"
        assert all(df['min_H_sum'] >= df['bound_bits'] - 0.1), \
            "H₁ + H₂ >= log₂(c) within numerical tolerance"

