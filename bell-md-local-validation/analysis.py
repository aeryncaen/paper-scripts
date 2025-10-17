# ============================================================================
# analysis.py
"""Data generators for validation analysis."""

import pandas as pd
import numpy as np
from numpy.random import Generator
from typing import List, Dict, Any
from bell_mdl import sample_lambda, compute_outcomes, sample_uniform_sphere, sector_flags, unit

def correlation_vs_angle_data(
    angles_deg: np.ndarray,
    n_samples: int,
    rng: Generator
) -> pd.DataFrame:
    """
    Generate correlation E[AB|θ] vs angle data.
    
    Args:
        angles_deg: Array of angles in degrees
        n_samples: Samples per angle
        rng: Random generator
    
    Returns:
        DataFrame with columns: angle_deg, correlation, theory, std_error, ci_lower, ci_upper
    """
    angles_rad = np.deg2rad(angles_deg)
    data = []
    
    for angle_deg, angle_rad in zip(angles_deg, angles_rad):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
        
        lam = sample_lambda(a, b, n_samples, rng)
        A, B = compute_outcomes(a, b, lam)
        corr = float(np.mean(A * B))
        se = float(np.sqrt(max(1e-16, 1.0 - corr**2) / n_samples))
        
        data.append({
            'angle_deg': angle_deg,
            'correlation': corr,
            'theory': -np.cos(angle_rad),
            'std_error': se,
            'ci_lower': corr - 1.96 * se,
            'ci_upper': corr + 1.96 * se,
            'abs_error': abs(corr + np.cos(angle_rad))
        })
    
    return pd.DataFrame(data)

def chsh_canonical_data(n_samples: int, rng: Generator) -> Dict[str, Any]:
    """
    Compute CHSH value for canonical geometry.
    
    Returns:
        Dict with correlations and CHSH value
    """
    a  = np.array([1.0, 0.0, 0.0])
    a_prime = np.array([0.0, 1.0, 0.0])
    b  = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    b_prime = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    
    def compute_corr(x, y):
        lam = sample_lambda(x, y, n_samples, rng)
        A, B = compute_outcomes(x, y, lam)
        return float(np.mean(A * B))
    
    E_ab = compute_corr(a, b)
    E_ab_prime = compute_corr(a, b_prime)
    E_a_prime_b = compute_corr(a_prime, b)
    E_a_prime_b_prime = compute_corr(a_prime, b_prime)
    
    S = abs(E_ab + E_ab_prime + E_a_prime_b - E_a_prime_b_prime)
    se_each = np.sqrt((1 - E_ab**2) / n_samples)
    se_S = np.sqrt(4 * se_each**2)
    
    return {
        'E_ab': E_ab,
        'E_ab_prime': E_ab_prime,
        'E_a_prime_b': E_a_prime_b,
        'E_a_prime_b_prime': E_a_prime_b_prime,
        'chsh_S': S,
        'chsh_se': se_S,
        'chsh_ci_lower': S - 1.96 * se_S,
        'chsh_ci_upper': S + 1.96 * se_S,
        'quantum_prediction': 2 * np.sqrt(2),
        'classical_bound': 2.0
    }

def mi_pairwise_tv(
    a: np.ndarray, b: np.ndarray, 
    a2: np.ndarray, b2: np.ndarray,
    n_samples: int, 
    rng: Generator
) -> float:
    """
    Estimate TV(ρ(·|a,b), ρ(·|a2,b2)) - Hall's actual metric.
    
    Both densities are piecewise-constant on their respective two-sector partitions.
    We evaluate under Uniform(S²) sampling.
    
    Args:
        a, b: First setting pair
        a2, b2: Second setting pair
        n_samples: Monte Carlo samples
        rng: Random generator
    
    Returns:
        Total variation distance between the two conditioned densities
    """
    a = unit(a); b = unit(b); a2 = unit(a2); b2 = unit(b2)
    
    # Sample uniformly from S²
    L = sample_uniform_sphere(n_samples, rng)
    
    # Sector flags for each pair
    s1 = sector_flags(a, b, L)      # True = same-sign sector for (a,b)
    s2 = sector_flags(a2, b2, L)    # True = same-sign sector for (a2,b2)
    
    # Compute densities for (a,b)
    c1 = float(np.dot(a, b))
    theta1 = np.arccos(np.clip(c1, -1, 1))
    q1_plus = 1.0 - theta1/np.pi
    q1_minus = 1.0 - q1_plus
    p1_plus = (1.0 + c1)/2.0
    
    eps = 1e-12
    if q1_plus < eps or q1_minus < eps:
        rho1 = np.ones(n_samples)
    else:
        rho1 = np.where(s1, p1_plus/q1_plus, (1-p1_plus)/q1_minus)
    
    # Compute densities for (a2,b2)
    c2 = float(np.dot(a2, b2))
    theta2 = np.arccos(np.clip(c2, -1, 1))
    q2_plus = 1.0 - theta2/np.pi
    q2_minus = 1.0 - q2_plus
    p2_plus = (1.0 + c2)/2.0
    
    if q2_plus < eps or q2_minus < eps:
        rho2 = np.ones(n_samples)
    else:
        rho2 = np.where(s2, p2_plus/q2_plus, (1-p2_plus)/q2_minus)
    
    # TV = (1/2) E[|ρ1 - ρ2|] under Uniform(S²)
    tv = 0.5 * float(np.mean(np.abs(rho1 - rho2)))
    return tv

def mi_scan_max_tv(n_samples: int, rng: Generator) -> Dict[str, Any]:
    """
    Scan canonical CHSH quadruple for worst-case pairwise TV.
    This is the quantity comparable to Hall's ~0.14 benchmark.
    
    Returns:
        Dict with all pairwise TVs and maximum
    """
    a  = np.array([1.0, 0.0, 0.0])
    a_prime = np.array([0.0, 1.0, 0.0])
    b  = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    b_prime = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    
    pairs = [
        ('(a,b) vs (a\',b\')', a, b, a_prime, b_prime),
        ('(a,b\') vs (a\',b)', a, b_prime, a_prime, b),
        ('(a,b) vs (a,b\')', a, b, a, b_prime),
        ('(a\',b) vs (a\',b\')', a_prime, b, a_prime, b_prime),
    ]
    
    results = []
    for label, x1, y1, x2, y2 in pairs:
        tv = mi_pairwise_tv(x1, y1, x2, y2, n_samples, rng)
        results.append({'pair': label, 'TV': tv})
    
    df = pd.DataFrame(results)
    max_tv = df['TV'].max()
    
    return {'pairs': df, 'max_TV': max_tv}

def mi_violation_data(
    angles_deg: np.ndarray,
    n_samples: int,
    rng: Generator
) -> pd.DataFrame:
    """
    Measure MI violation: compare ρ(λ|a,b) to uniform (for visualization).
    
    Note: This shows MI is violated. For comparison to Hall (2010),
    use mi_scan_max_tv() which computes pairwise TV between setting pairs.
    
    Returns:
        DataFrame with TV distance from uniform
    """
    data = []
    
    for angle_deg in angles_deg:
        angle_rad = np.deg2rad(angle_deg)
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
        a = unit(a); b = unit(b)
        
        c = np.dot(a, b)
        theta = np.arccos(np.clip(c, -1, 1))
        p_theory = (1 + c) / 2
        q_theory = 1 - theta / np.pi
        
        # Sample from ρ(λ|a,b)
        lam_dep = sample_lambda(a, b, n_samples, rng)
        same_dep = sector_flags(a, b, lam_dep)
        p_dep = float(np.mean(same_dep))
        
        # Sample from uniform
        lam_uni = sample_uniform_sphere(n_samples, rng)
        same_uni = sector_flags(a, b, lam_uni)
        p_uni = float(np.mean(same_uni))
        
        # Compute TV analytically
        if q_theory > 1e-12 and (1 - q_theory) > 1e-12:
            rho_plus = p_theory / q_theory
            rho_minus = (1 - p_theory) / (1 - q_theory)
            TV = 0.5 * (abs(rho_plus - 1) * q_theory + abs(rho_minus - 1) * (1 - q_theory))
        else:
            TV = 0.0
        
        data.append({
            'angle_deg': angle_deg,
            'p_dependent': p_dep,
            'p_uniform': p_uni,
            'p_theory': p_theory,
            'q_theory': q_theory,
            'TV_distance': TV,
            'MI_violation': abs(p_dep - p_uni)
        })
    
    return pd.DataFrame(data)

def witness_product_data(
    grid_size: int,
    block_size: int,
    N_star_values: List[int],
    n_trials: int,
    rng: Generator
) -> pd.DataFrame:
    """
    Test witness-product bound: R1(W) * R2(W) >= ceil(N_star / m).
    
    Args:
        grid_size: Total grid dimension (e.g., 64x64)
        block_size: Block size for partitions (e.g., 4x4)
        N_star_values: List of witness capacities to test
        n_trials: Trials per N_star
        rng: Random generator
    
    Returns:
        DataFrame with bound validation results including min-entropy
    """
    m = block_size * block_size
    all_states = np.array([(i, j) for i in range(grid_size) for j in range(grid_size)])
    n_states = len(all_states)
    
    data = []
    
    for N_star in N_star_values:
        bound = int(np.ceil(N_star / m))
        bound_bits = np.log2(bound)
        
        products = []
        H_sums = []
        violations = 0
        
        for _ in range(n_trials):
            # Sample N_star states (finite witness)
            idx = rng.choice(n_states, size=N_star, replace=False)
            witness = all_states[idx]
            
            rows = witness[:, 0]
            cols = witness[:, 1]
            
            # Compute residual multiplicities
            row_blocks = rows // block_size
            col_blocks = cols // block_size
            
            R1 = int(np.max(np.bincount(row_blocks, minlength=grid_size // block_size)))
            R2 = int(np.max(np.bincount(col_blocks, minlength=grid_size // block_size)))
            
            product = R1 * R2
            products.append(product)
            
            # Min-entropy: H_∞(Π|W) = -log₂(R(W)/|S_W|) = log₂(|S_W|) - log₂(R(W))
            H1 = np.log2(N_star) - np.log2(R1)
            H2 = np.log2(N_star) - np.log2(R2)
            H_sum = H1 + H2
            H_sums.append(H_sum)
            
            if product < bound:
                violations += 1
        
        min_H_sum = min(H_sums) if H_sums else 0
        
        data.append({
            'N_star': N_star,
            'overlap_m': m,
            'bound': bound,
            'bound_bits': bound_bits,
            'min_product': min(products),
            'mean_product': np.mean(products),
            'max_product': max(products),
            'min_H_sum': min_H_sum,
            'violations': violations,
            'trials': n_trials,
            'pass': violations == 0,
            'entropy_pass': min_H_sum >= bound_bits - 0.1  # small tolerance for float precision
        })
    
    return pd.DataFrame(data)
