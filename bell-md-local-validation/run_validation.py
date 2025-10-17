# ============================================================================
# run_validation.py
"""Main validation runner with clean output."""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from numpy.random import default_rng
from analysis import (
    correlation_vs_angle_data, chsh_canonical_data,
    mi_violation_data, mi_scan_max_tv, witness_product_data
)
from figures import plot_correlation_curve, plot_mi_comparison, plot_mi_pairwise_for_paper, plot_witness_product, plot_mi_pinsker_bound

def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_section(title: str):
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}")

def main():
    # Setup
    start_time = time.perf_counter()
    rng = default_rng(42069)
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    print_header("MD-LOCAL DETERMINISTIC BELL MODEL: VALIDATION SUITE")
    print(f"Random seed: 42069")
    print(f"Output directory: {output_dir}")
    
    # 1. Correlation vs angle
    print_section("1. Correlation vs Angle")
    corr_df = correlation_vs_angle_data(
        angles_deg=np.linspace(0, 180, 13),
        n_samples=20000,
        rng=rng
    )
    print(corr_df.to_string(index=False, float_format='%.4f'))
    print("\nAcceptance diagnostics: mean acceptance = {:.4f}".format(corr_df['acceptance_rate'].mean()))
    plot_correlation_curve(corr_df, output_dir / "correlation_curve.png")
    print(f"✓ Figure saved: {output_dir / 'correlation_curve.png'}")
    
    # 2. CHSH
    print_section("2. CHSH Canonical Geometry")
    chsh_result = chsh_canonical_data(n_samples=50000, rng=rng)
    
    # Create detailed CHSH table with errors
    chsh_details = pd.DataFrame([
        {'Correlator': 'E(a,b)', 'Value': chsh_result['E_ab'], 'SE(E)': chsh_result['se_ab']},
        {'Correlator': "E(a,b')", 'Value': chsh_result['E_ab_prime'], 'SE(E)': chsh_result['se_ab_prime']},
        {'Correlator': "E(a',b)", 'Value': chsh_result['E_a_prime_b'], 'SE(E)': chsh_result['se_a_prime_b']},
        {'Correlator': "E(a',b')", 'Value': chsh_result['E_a_prime_b_prime'], 'SE(E)': chsh_result['se_a_prime_b_prime']},
    ])
    print(chsh_details.to_string(index=False, float_format='%.4f'))
    
    print(f"\nCHSH S = {chsh_result['chsh_S']:.4f} ± {chsh_result['chsh_se']:.4f}")
    print(f"95% CI: [{chsh_result['chsh_ci_lower']:.4f}, {chsh_result['chsh_ci_upper']:.4f}]")
    print(f"Quantum prediction: {chsh_result['quantum_prediction']:.4f}")
    print(f"Classical bound: {chsh_result['classical_bound']:.4f}")
    
    # 3. MI violation
    print_section("3. Measurement Independence Violation (vs Uniform)")
    mi_df = mi_violation_data(
        angles_deg=np.array([15, 30, 45, 60, 75, 90]),
        n_samples=30000,
        rng=rng
    )
    print(mi_df.to_string(index=False, float_format='%.4f'))
    
    max_TV_uni = mi_df['TV_distance'].max()
    print(f"\nMaximum TV distance from uniform: {max_TV_uni:.4f}")
    print(f"Note: This demonstrates MI violation exists.")
    
    # 3b. MI pairwise (Hall metric)
    print_section("3b. MI Budget: Pairwise TV (Hall 2010 Metric)")
    mi_pair_result = mi_scan_max_tv(n_samples=120000, rng=rng)
    
    # Print with error bars
    pairs_with_ci = mi_pair_result['pairs'].copy()
    pairs_with_ci['CI_lower'] = pairs_with_ci['TV'] - 1.96 * pairs_with_ci['TV_SE']
    pairs_with_ci['CI_upper'] = pairs_with_ci['TV'] + 1.96 * pairs_with_ci['TV_SE']
    print(pairs_with_ci[['pair', 'TV', 'TV_SE', 'CI_lower', 'CI_upper']].to_string(index=False, float_format='%.4f'))
    
    print(f"\nMaximum pairwise TV: {mi_pair_result['max_TV']:.4f} ± {mi_pair_result['max_TV_SE']:.4f}")
    print(f"Hall (2010) benchmark: 0.1400 (14%)")
    match = 0.12 <= mi_pair_result['max_TV'] <= 0.16
    print(f"Match: {'✓ YES' if match else '✗ NO'}")
    
    # 3c. MI lower bound (rigorous Pinsker)
    print_section("3c. MI Lower Bound via Pinsker (Rigorous)")
    from analysis import mi_mutual_information_lower_bound
    mi_bound_result = mi_mutual_information_lower_bound(n_samples=120000, rng=rng)
    
    # Print TV to mixture for each setting
    print("TV to mixture (for each setting pair):")
    for label, tv, tv_se in zip(mi_bound_result['setting_labels'], 
                                 mi_bound_result['tvs_to_mixture'],
                                 mi_bound_result['tv_ses']):
        print(f"  {label:8s}: {tv:.4f} ± {tv_se:.4f}")
    
    print(f"\nMean TV to mixture: {mi_bound_result['mean_tv']:.4f}")
    print(f"Mean TV²: {mi_bound_result['mean_tv_squared']:.6f}")
    print(f"\nRigorous MI lower bound: I(λ:X,Y) ≥ {mi_bound_result['mi_lower_bits']:.4f} ± {mi_bound_result['mi_lower_se']:.4f} bits")
    print(f"Hall & Branciard (2020) benchmark: {mi_bound_result['hall_branciard_benchmark']:.3f} bits")
    
    consistent = mi_bound_result['mi_lower_bits'] <= mi_bound_result['hall_branciard_benchmark']
    print(f"Consistency: {'✓ Lower bound ≤ Hall-Branciard optimal' if consistent else '✗ Exceeds known optimal'}")
    
    # Generate MI Pinsker bound figure
    plot_mi_pinsker_bound(mi_bound_result, output_dir / "mi_pinsker_bound.png")
    print(f"✓ Figure saved: {output_dir / 'mi_pinsker_bound.png'}")
    
    plot_mi_comparison(mi_df, output_dir / "mi_violation.png")
    print(f"✓ Figure saved: {output_dir / 'mi_violation.png'}")
    
    # Generate paper-specific MI figure (simpler, just pairwise TV with errors)
    plot_mi_pairwise_for_paper(mi_pair_result['pairs'], mi_pair_result['max_TV'],
                               mi_pair_result['max_TV_SE'],
                               output_dir / "mi_pairwise_paper.png")
    print(f"✓ Paper figure saved: {output_dir / 'mi_pairwise_paper.png'}")
    
    # 3c. No-signaling diagnostics
    print_section("3d. No-Signaling Diagnostics")
    ns = chsh_result['no_signaling']
    ns_table = pd.DataFrame([
        {"Metric": "max_a |Pr(A=+1|a,·) - 0.5|", "Value": ns['max_A_bias']},
        {"Metric": "max_{a,b,b'} |Pr(A|a,b) - Pr(A|a,b')|", "Value": ns['max_A_signal']},
        {"Metric": "max_b |Pr(B=+1|·,b) - 0.5|", "Value": ns['max_B_bias']},
        {"Metric": "max_{b,a,a'} |Pr(B|a,b) - Pr(B|a',b)|", "Value": ns['max_B_signal']},
    ])
    print(ns_table.to_string(index=False, float_format='%.4f'))

    # 4. Witness-product bound
    print_section("4. Witness-Product Bound (Theorem 2) + Min-Entropy Corollary")
    wp_df = witness_product_data(
        grid_size=64, block_size=4,
        N_star_values=[64, 128, 256, 512],
        n_trials=200,
        rng=rng
    )
    
    # Print main results
    print(wp_df[['N_star', 'overlap_m', 'bound_product', 'min_product', 'mean_product', 'violations', 'pass']].to_string(index=False))
    
    # Print entropy results
    print("\nMin-Entropy Corollary: H_∞(Π₁|W) + H_∞(Π₂|W) ≥ 2·log₂(2/(1+r_max))")
    print("  N★   | r_max  | Bound (bits) | min(H₁+H₂) | entropy_pass")
    print("  " + "-"*60)
    for _, row in wp_df.iterrows():
        print(f"  {row['N_star']:3d}  | {row['r_max']:6.4f} |    {row['entropy_lower_bits']:5.2f}     |   {row['min_H_sum']:6.2f}   | {'✓ PASS' if row['entropy_pass'] else '✗ FAIL'}")
    
    all_pass = wp_df['pass'].all()
    entropy_pass = wp_df['entropy_pass'].all()
    print(f"\nProduct bound: {'✓ ALL TESTS PASS' if all_pass else '✗ VIOLATIONS DETECTED'}")
    print(f"Entropy bound: {'✓ ALL TESTS PASS' if entropy_pass else '✗ VIOLATIONS DETECTED'}")
    
    plot_witness_product(wp_df, output_dir / "witness_product.png")
    print(f"✓ Figure saved: {output_dir / 'witness_product.png'}")
    
    # Summary
    print_header("VALIDATION SUMMARY")
    
    # Correlation test
    corr_pass = (corr_df['abs_error'] < 0.02).all()  # 2% tolerance
    max_corr_err = corr_df['abs_error'].max()
    print(f"{'✓ PASS' if corr_pass else '✗ FAIL'} | Correlation matches singlet (max error: {max_corr_err:.4f}, threshold: 0.02)")
    
    # CHSH classical bound
    chsh_classical_pass = chsh_result['chsh_S'] > 2.0
    print(f"{'✓ PASS' if chsh_classical_pass else '✗ FAIL'} | CHSH exceeds classical bound (S = {chsh_result['chsh_S']:.4f} > 2.0)")
    
    # CHSH quantum match
    chsh_quantum_pass = abs(chsh_result['chsh_S'] - chsh_result['quantum_prediction']) < 0.05
    chsh_err = abs(chsh_result['chsh_S'] - chsh_result['quantum_prediction'])
    print(f"{'✓ PASS' if chsh_quantum_pass else '✗ FAIL'} | CHSH matches quantum (error: {chsh_err:.4f}, threshold: 0.05)")
    
    # MI violation - both metrics
    mi_violated = (mi_df['TV_distance'] > 0.05).any()
    mi_hall_match = 0.12 <= mi_pair_result['max_TV'] <= 0.16
    mi_bound_consistent = mi_bound_result['mi_lower_bits'] <= mi_bound_result['hall_branciard_benchmark']
    print(f"{'✓ PASS' if mi_violated else '✗ FAIL'} | MI violation detected (max TV from uniform: {max_TV_uni:.4f} > 0.05)")
    print(f"{'✓ PASS' if mi_hall_match else '✗ FAIL'} | MI matches Hall benchmark (pairwise TV: {mi_pair_result['max_TV']:.4f}, Hall: 0.14)")
    print(f"{'✓ PASS' if mi_bound_consistent else '✗ FAIL'} | MI lower bound consistent (I ≥ {mi_bound_result['mi_lower_bits']:.4f} bits ≤ 0.080 bits)")
    
    # Witness-product
    entropy_pass = wp_df['entropy_pass'].all()
    print(f"{'✓ PASS' if all_pass else '✗ FAIL'} | Witness-product bound holds (0 violations)")
    print(f"{'✓ PASS' if entropy_pass else '✗ FAIL'} | Min-entropy corollary holds")
    
    # Overall
    all_tests_pass = corr_pass and chsh_classical_pass and chsh_quantum_pass and mi_violated and mi_hall_match and mi_bound_consistent and all_pass and entropy_pass
    print("\n" + "─" * 80)
    print(f"OVERALL: {'✓✓✓ ALL TESTS PASS ✓✓✓' if all_tests_pass else '✗ SOME TESTS FAILED'}")
    print("=" * 80)

    elapsed = time.perf_counter() - start_time
    print(f"NumPy version: {np.__version__}")

    import matplotlib
    print(f"Matplotlib version: {matplotlib.__version__}")
    print(f"Elapsed time: {elapsed:.2f} s")

if __name__ == '__main__':
    main()
