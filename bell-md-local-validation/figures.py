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
    """Plot MI violation: dependent vs uniform distributions (validation suite)."""
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

def plot_mi_pairwise_for_paper(pairs_df: pd.DataFrame, max_tv: float, max_tv_se: float, output_path: Path):
    """
    Simplified MI figure for paper: pairwise TV values with error bars.
    This is what should be compared to Hall (2010).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(pairs_df))
    colors = ['#E74C3C' if tv == max_tv else '#3498DB' for tv in pairs_df['TV']]
    
    # Bar plot with error bars
    bars = ax.bar(x, pairs_df['TV'], color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5,
                   yerr=1.96*pairs_df['TV_SE'], capsize=5, error_kw={'linewidth': 2})
    
    # Highlight the max
    max_idx = pairs_df['TV'].idxmax()
    bars[max_idx].set_edgecolor('red')
    bars[max_idx].set_linewidth(3)
    
    ax.axhline(0.14, color='black', linestyle='--', linewidth=2, label='Hall (2010): 14%', zorder=10)
    ax.fill_between([-0.5, len(pairs_df)-0.5], 0.12, 0.16, alpha=0.15, color='gray', 
                    label='Hall ±2%', zorder=0)
    
    ax.set_xlabel('Setting Pair', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Variation Distance', fontsize=13, fontweight='bold')
    ax.set_title('Measurement Independence Violation:\nPairwise TV Between Setting-Conditioned Densities', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('\\', '') for p in pairs_df['pair']], rotation=15, ha='right')
    ax.set_ylim([0, 0.18])
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.25)
    
    # Annotate max with error
    ax.text(max_idx, max_tv + 1.96*max_tv_se + 0.01, 
            f'Max: {max_tv:.4f}±{1.96*max_tv_se:.4f}', 
            ha='center', fontsize=10, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_mi_pinsker_bound(mi_result: dict, output_path: Path):
    """
    Plot rigorous MI lower bound via Pinsker.
    Shows TV to mixture for each setting and resulting MI bound.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: TV to mixture for each setting
    labels = mi_result['setting_labels']
    tvs = mi_result['tvs_to_mixture']
    tv_ses = mi_result['tv_ses']
    
    x = np.arange(len(labels))
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']
    
    bars = ax1.bar(x, tvs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5,
                   yerr=1.96*np.array(tv_ses), capsize=5, error_kw={'linewidth': 2})
    
    # Mean line
    mean_tv = mi_result['mean_tv']
    ax1.axhline(mean_tv, color='purple', linestyle='--', linewidth=2.5, 
                label=f'Mean: {mean_tv:.4f}', zorder=10)
    
    ax1.set_xlabel('Setting Pair', fontsize=13, fontweight='bold')
    ax1.set_ylabel('TV(ρ(·|x,y), ρ_mix)', fontsize=13, fontweight='bold')
    ax1.set_title('Total Variation to Mixture Density', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylim([0, max(tvs) * 1.2])
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(axis='y', alpha=0.25)
    
    # Right: Pinsker bound calculation
    mi_lower = mi_result['mi_lower_bits']
    mi_se = mi_result['mi_lower_se']
    hall_bench = mi_result['hall_branciard_benchmark']
    
    # Bar showing bound vs optimal
    categories = ['Lower Bound\n(Pinsker)', 'Optimal Cost\n(Hall-Branciard)']
    values = [mi_lower, hall_bench]
    colors_right = ['#27AE60', '#E74C3C']
    
    bars2 = ax2.bar(categories, values, color=colors_right, alpha=0.8, 
                    edgecolor='black', linewidth=2)
    
    # Error bar on lower bound
    ax2.errorbar([0], [mi_lower], yerr=1.96*mi_se, fmt='none', 
                 color='black', capsize=10, linewidth=2.5, capthick=2)
    
    # Annotation
    ax2.text(0, mi_lower + 1.96*mi_se + 0.005, 
             f'{mi_lower:.4f}±{1.96*mi_se:.4f}', 
             ha='center', fontsize=11, fontweight='bold', color='#27AE60')
    ax2.text(1, hall_bench + 0.005, f'{hall_bench:.3f}', 
             ha='center', fontsize=11, fontweight='bold', color='#E74C3C')
    
    ax2.set_ylabel('Mutual Information (bits)', fontsize=13, fontweight='bold')
    ax2.set_title('Rigorous MI Bound: I(λ:X,Y) ≥ (2/ln2)·E[TV²]', 
                  fontsize=14, fontweight='bold')
    ax2.set_ylim([0, hall_bench * 1.3])
    ax2.grid(axis='y', alpha=0.25)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_witness_product(df: pd.DataFrame, output_path: Path):
    """Plot witness-product bound validation - clearer visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Show margin above bound
    x = np.arange(len(df))
    margin = df['min_product'] - df['bound_product']
    colors = ['#27AE60' if p else '#E74C3C' for p in df['pass']]
    
    ax1.bar(x, margin, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.axhline(0, color='red', linestyle='--', linewidth=2, label='Bound Threshold')
    ax1.set_xlabel('Witness Capacity N★', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Min(R₁·R₂) - Bound (margin)', fontsize=12, fontweight='bold')
    ax1.set_title('Witness-Product Bound: Safety Margin', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['N_star'])
    ax1.grid(axis='y', alpha=0.25)
    ax1.legend(fontsize=10)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(df.iterrows()):
        ax1.text(i, margin.iloc[i] + 20, f"+{int(margin.iloc[i])}", 
                ha='center', fontsize=10, fontweight='bold')
    
    # Right plot: Min-entropy validation
    x2 = np.arange(len(df))
    width = 0.35
    
    ax2.bar(x2 - width/2, df['entropy_lower_bits'], width, label='Bound: 2·log₂(2/(1+r))', 
            color='#E74C3C', alpha=0.7, edgecolor='black', linewidth=1)
    ax2.bar(x2 + width/2, df['min_H_sum'], width, label='Observed: H₁ + H₂', 
            color='#27AE60', alpha=0.7, edgecolor='black', linewidth=1)
    
    ax2.set_xlabel('Witness Capacity N★', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Bits', fontsize=12, fontweight='bold')
    ax2.set_title('Min-Entropy Corollary: H_∞(Π₁|W) + H_∞(Π₂|W) ≥ 2·log₂(2/(1+r_max))', 
                  fontsize=13, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(df['N_star'])
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.25)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
