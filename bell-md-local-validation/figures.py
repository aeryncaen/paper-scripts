# ============================================================================
# figures.py
"""Figure generation for validation results."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def plot_correlation_curve(df: pd.DataFrame, output_path: Path):
    """Plot correlation vs angle with error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(
        df['angle_deg'], df['correlation'],
        yerr=1.96 * df['std_error'],
        fmt='o', capsize=4, markersize=6,
        label='MD-Local Model (95% CI)',
        color='#2E86AB', ecolor='#A23B72'
    )
    
    ax.plot(
        df['angle_deg'], df['theory'],
        'r-', linewidth=2.5, label='Singlet Theory: −cos θ'
    )
    
    ax.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.set_xlabel('Angle θ (degrees)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Correlation E[AB|a,b]', fontsize=13, fontweight='bold')
    ax.set_title('Correlation vs Angle: Model Validation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(alpha=0.25)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_mi_comparison(df: pd.DataFrame, output_path: Path):
    """Plot MI violation: dependent vs uniform distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: sector occupancy
    ax1.plot(df['angle_deg'], df['p_dependent'], 'o-', label='ρ(λ|a,b) [dependent]', linewidth=2)
    ax1.plot(df['angle_deg'], df['p_uniform'], 's-', label='Uniform(S²) [independent]', linewidth=2)
    ax1.plot(df['angle_deg'], df['p_theory'], '--', label='Theory p₊', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Angle θ (degrees)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('P(same-sign sector)', fontsize=12, fontweight='bold')
    ax1.set_title('Measurement Dependence: Sector Occupancy', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.25)
    
    # Right: TV distance
    ax2.plot(df['angle_deg'], df['TV_distance'], 'o-', color='#C73E1D', linewidth=2.5, markersize=7)
    ax2.axhline(0.14, color='k', linestyle='--', linewidth=2, label='Hall (2010): 14%')
    ax2.fill_between(df['angle_deg'], 0.12, 0.16, alpha=0.2, color='gray', label='Hall ±2%')
    ax2.set_xlabel('Angle θ (degrees)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Variation Distance', fontsize=12, fontweight='bold')
    ax2.set_title('MI Violation Magnitude', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.25)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_witness_product(df: pd.DataFrame, output_path: Path):
    """Plot witness-product bound validation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    ax.bar(x - width/2, df['bound'], width, label='Theoretical Bound: ⌈N★/m⌉', color='#2E86AB')
    ax.bar(x + width/2, df['min_product'], width, label='Observed Min(R₁·R₂)', color='#F18F01')
    
    ax.set_xlabel('Witness Capacity N★', fontsize=12, fontweight='bold')
    ax.set_ylabel('Product Value', fontsize=12, fontweight='bold')
    ax.set_title('Witness-Product Bound Validation (Theorem 2)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['N_star'])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.25)
    
    # Annotate pass/fail
    for i, (idx, row) in enumerate(df.iterrows()):
        status = '✓ PASS' if row['pass'] else f'✗ FAIL ({row["violations"]})'
        color = 'green' if row['pass'] else 'red'
        ax.text(i, row['min_product'] + 5, status, ha='center', fontsize=9, 
                color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
