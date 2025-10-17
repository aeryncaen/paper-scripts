# bell_mdl.py
"""Core MD-Local deterministic Bell model implementation."""

import numpy as np
from typing import Tuple
from numpy.random import Generator

def unit(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Cannot normalize zero vector")
    return v / norm

def sector_flags(a: np.ndarray, b: np.ndarray, lam_batch: np.ndarray) -> np.ndarray:
    """
    Return boolean mask: True if sgn(a·λ) == sgn(b·λ), False otherwise.
    
    Args:
        a, b: Unit direction vectors
        lam_batch: (N, 3) array of lambda vectors on S²
    
    Returns:
        (N,) boolean array
    """
    da = lam_batch @ a
    db = lam_batch @ b
    return np.where(da >= 0, 1, -1) == np.where(db >= 0, 1, -1)

def sample_lambda(
    a: np.ndarray, 
    b: np.ndarray, 
    n: int, 
    rng: Generator,
    batch_size: int = 4096
) -> np.ndarray:
    """
    Sample n vectors from ρ(λ|a,b) using acceptance-rejection.
    
    The density is piecewise constant on S²:
        p_+ = (1 + cos θ)/2 on same-sign sector
        p_- = (1 - cos θ)/2 on opposite-sign sector
    where θ = arccos(a·b).
    
    Args:
        a, b: Measurement direction vectors (will be normalized)
        n: Number of samples
        rng: NumPy random generator
        batch_size: Samples per batch for efficiency
    
    Returns:
        (n, 3) array of unit vectors on S²
    """
    a = unit(a)
    b = unit(b)
    c = np.clip(np.dot(a, b), -1.0, 1.0)
    theta = np.arccos(c)
    
    p_plus = (1.0 + c) / 2.0
    q_plus = 1.0 - theta / np.pi
    q_minus = 1.0 - q_plus
    
    out = np.empty((n, 3), dtype=float)
    k = 0
    
    # Handle degenerate cases
    if q_plus < 1e-15:  # Only opposite-sign sector exists
        while k < n:
            U = rng.normal(size=(batch_size, 3))
            U /= np.linalg.norm(U, axis=1, keepdims=True)
            accepted = U[~sector_flags(a, b, U)]
            m = min(n - k, len(accepted))
            out[k:k+m] = accepted[:m]
            k += m
        return out
    
    if q_minus < 1e-15:  # Only same-sign sector exists
        while k < n:
            U = rng.normal(size=(batch_size, 3))
            U /= np.linalg.norm(U, axis=1, keepdims=True)
            accepted = U[sector_flags(a, b, U)]
            m = min(n - k, len(accepted))
            out[k:k+m] = accepted[:m]
            k += m
        return out
    
    # General case: acceptance-rejection with sector-dependent rates
    r_plus = p_plus / q_plus
    r_minus = (1 - p_plus) / q_minus
    M = max(r_plus, r_minus)
    alpha_plus = r_plus / M
    alpha_minus = r_minus / M
    
    while k < n:
        U = rng.normal(size=(batch_size, 3))
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        same = sector_flags(a, b, U)
        probs = np.where(same, alpha_plus, alpha_minus)
        accept = rng.random(len(U)) < probs
        accepted = U[accept]
        m = min(n - k, len(accepted))
        out[k:k+m] = accepted[:m]
        k += m
    
    return out

def compute_outcomes(
    a: np.ndarray, 
    b: np.ndarray, 
    lam_batch: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute deterministic local outcomes A and B.
    
    A = sgn(a·λ), B = -sgn(b·λ)
    
    For measure-zero ties (a·λ = 0), break deterministically via tiny hash
    to avoid biasing marginals.
    
    Args:
        a, b: Measurement directions
        lam_batch: (N, 3) array of lambda vectors
    
    Returns:
        A, B: (N,) arrays with values in {-1, +1}
    """
    da = lam_batch @ a
    db = lam_batch @ b
    
    # Deterministic tie-breaking for exact zeros (measure-zero event)
    eps_A = 1e-12 * np.sign(np.sin(1e6 * da))
    eps_B = 1e-12 * np.sign(np.cos(1e6 * db))
    
    A =  np.where(da + eps_A >= 0, 1.0, -1.0)
    B = -np.where(db + eps_B >= 0, 1.0, -1.0)
    return A, B

def sample_uniform_sphere(n: int, rng: Generator) -> np.ndarray:
    """Sample uniformly from S²."""
    U = rng.normal(size=(n, 3))
    return U / np.linalg.norm(U, axis=1, keepdims=True)
