"""
CS 5510 Homework 1 - Reconstruction Attack Experiments
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ps2_starter import (
    load_data, PUB_COLS, SENSITIVE, make_random_predicate,
    execute_subsetsums_exact, execute_subsetsums_round,
    execute_subsetsums_noise, execute_subsetsums_sample,
    reconstruction_attack, rmse, success_rate
)

# Data load 
print("Loading dataset...")
data = load_data("fake_healthcare_dataset_sample100.csv")
n = len(data)
print(f"Dataset: {n} patients")
print(f"Target distribution: {(data[SENSITIVE]==1).sum()} abnormal, {(data[SENSITIVE]==0).sum()} normal")

# Majority baseline computation
p_majority = max((data[SENSITIVE]==0).sum(), (data[SENSITIVE]==1).sum()) / n
print(f"Majority baseline: {p_majority:.3f}\n")

def run_trial(defense_type, param, num_queries=200, seed=None):
    """Run single trial with given defense and parameter."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate predicates
    predicates = [make_random_predicate(seed=seed*1000+i if seed else None) 
                  for i in range(num_queries)]
    
    # Get exact answers
    exact_answers = execute_subsetsums_exact(data, predicates)
    
    # Apply defense
    if defense_type == 'round':
        defended_answers = execute_subsetsums_round(param, data, predicates)
    elif defense_type == 'noise':
        defended_answers = execute_subsetsums_noise(param, data, predicates, seed=seed)
    elif defense_type == 'sample':
        defended_answers = execute_subsetsums_sample(param, data, predicates, seed=seed)
    else:
        raise ValueError(f"Unknown defense: {defense_type}")
    
    # Compute RMSE
    accuracy = rmse(defended_answers, exact_answers)
    
    # Run reconstruction attack
    reconstructed = reconstruction_attack(data[PUB_COLS], predicates, defended_answers, solver='auto')
    success = success_rate(reconstructed, data[SENSITIVE].values)
    
    # Count exact matches for undefended case tracking
    exact_match = int(success == 1.0)
    
    return accuracy, success, exact_match

def run_experiments(defense_type, param_range, num_trials=10):
    """Run experiments across parameter range."""
    results = []
    
    print(f"\nRunning {defense_type} defense experiments...")
    for param in param_range:
        rmse_vals = []
        success_vals = []
        exact_matches = []
        
        for trial in range(num_trials):
            acc, succ, exact = run_trial(defense_type, param, num_queries=200, seed=trial*100+param)
            rmse_vals.append(acc)
            success_vals.append(succ)
            exact_matches.append(exact)
        
        avg_rmse = np.mean(rmse_vals)
        avg_success = np.mean(success_vals)
        frac_exact = np.mean(exact_matches)
        
        results.append({
            'param': param,
            'rmse': avg_rmse,
            'success': avg_success,
            'exact_reconstruction_rate': frac_exact
        })
        
        if param == 1:
            print(f"  param={param:3d}: RMSE={avg_rmse:6.2f}, Success={avg_success:.3f}, Exact={frac_exact:.3f} ({int(frac_exact*num_trials)}/{num_trials} trials)")
        else:
            print(f"  param={param:3d}: RMSE={avg_rmse:6.2f}, Success={avg_success:.3f}")
    
    return pd.DataFrame(results)

# Parameter ranges - full sweep from 1 to n
param_range = list(range(1, n + 1))

# Run experiments for each defense
print("="*70)
print("Running all experiments (10 trials per parameter)...")
print("="*70)

results_round = run_experiments('round', param_range, num_trials=10)
results_noise = run_experiments('noise', param_range, num_trials=10)
results_sample = run_experiments('sample', param_range, num_trials=10)

# Save results to CSV
results_round.to_csv('results_round.csv', index=False)
results_noise.to_csv('results_noise.csv', index=False)
results_sample.to_csv('results_sample.csv', index=False)

print("\n[OK] Saved: results_round.csv, results_noise.csv, results_sample.csv")

# Create summary files
def find_transition(df, p_majority):
    """Find first parameter where success <= majority baseline."""
    for _, row in df.iterrows():
        if row['success'] <= p_majority:
            return int(row['param']), float(row['success'])
    return int(df.iloc[-1]['param']), float(df.iloc[-1]['success'])

R_star, success_at_R = find_transition(results_round, p_majority)
sigma_star, success_at_sigma = find_transition(results_noise, p_majority)
t_star, success_at_t = find_transition(results_sample, p_majority)

# Compact summary
summary_compact = pd.DataFrame([
    {'defense': 'round', 'param_star': R_star, 'success_at_star': success_at_R, 'majority_baseline': p_majority},
    {'defense': 'noise', 'param_star': sigma_star, 'success_at_star': success_at_sigma, 'majority_baseline': p_majority},
    {'defense': 'sample', 'param_star': t_star, 'success_at_star': success_at_t, 'majority_baseline': p_majority}
])
summary_compact.to_csv('reconstruction_defense_summary_compact.csv', index=False)

# Full summary
full_data = []
for defense, df in [('round', results_round), ('noise', results_noise), ('sample', results_sample)]:
    for _, row in df.iterrows():
        full_data.append({
            'defense_type': defense,
            'parameter_value': row['param'],
            'avg_rmse': row['rmse'],
            'avg_success_rate': row['success'],
            'attack_failed': row['success'] <= p_majority
        })
summary_full = pd.DataFrame(full_data)
summary_full.to_csv('reconstruction_defense_summary_full.csv', index=False)

print("[OK] Saved: reconstruction_defense_summary_compact.csv, reconstruction_defense_summary_full.csv")

# Create visualizations
print("\nGenerating plots...")

defense_data = {
    'round': (results_round, R_star, 'Rounding (R)'),
    'noise': (results_noise, sigma_star, 'Gaussian Noise (σ)'),
    'sample': (results_sample, t_star, 'Subsampling (t)')
}

for defense_name, (df, transition_param, label) in defense_data.items():
    # RMSE vs parameter
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df['param'], df['rmse'], 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel(f'Parameter ({label})', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title(f'RMSE vs Parameter ({defense_name.capitalize()} Defense)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'rmse_vs_param_{defense_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Success vs parameter
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df['param'], df['success'], 'r-s', linewidth=2, markersize=8)
    ax.axhline(y=p_majority, color='gray', linestyle='--', linewidth=2, label=f'Majority baseline ({p_majority:.3f})')
    ax.axvline(x=transition_param, color='green', linestyle=':', linewidth=2, label=f'Transition ({transition_param})')
    ax.set_xlabel(f'Parameter ({label})', fontsize=12)
    ax.set_ylabel('Reconstruction Success Rate', fontsize=12)
    ax.set_title(f'Success vs Parameter ({defense_name.capitalize()} Defense)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(f'success_vs_param_{defense_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Combined plot (RMSE + Success)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(df['param'], df['rmse'], 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel(f'Parameter ({label})', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title('Query Answer Accuracy', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(df['param'], df['success'], 'r-s', linewidth=2, markersize=8)
    ax2.axhline(y=p_majority, color='gray', linestyle='--', linewidth=2, label=f'Baseline ({p_majority:.3f})')
    ax2.set_xlabel(f'Parameter ({label})', fontsize=12)
    ax2.set_ylabel('Reconstruction Success', fontsize=12)
    ax2.set_title('Attack Success Rate', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    fig.suptitle(f'{defense_name.capitalize()} Defense Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'rmse_{defense_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Trade-off plot
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df['rmse'], df['success'], c=df['param'], cmap='viridis', 
                        s=150, alpha=0.7, edgecolors='black', linewidth=1.5)
    ax.axhline(y=p_majority, color='gray', linestyle='--', linewidth=2, label=f'Baseline ({p_majority:.3f})')
    ax.set_xlabel('RMSE (Lower = More Accurate)', fontsize=12)
    ax.set_ylabel('Reconstruction Success Rate', fontsize=12)
    ax.set_title(f'Trade-off: Accuracy vs Privacy ({defense_name.capitalize()})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f'Parameter ({label})', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'tradeoff_{defense_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

print(f"[OK] Generated 12 plots (4 per defense)")

print("\n" + "="*70)
print("All experiments completed!")
print("="*70)
print(f"\nTransition points:")
print(f"  Rounding: R* = {R_star}, Success = {success_at_R:.3f}")
print(f"  Noise: sigma* = {sigma_star}, Success = {success_at_sigma:.3f}")
print(f"  Sampling: t* = {t_star}, Success = {success_at_t:.3f}")
print(f"  Majority baseline: {p_majority:.3f}")

