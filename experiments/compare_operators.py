"""
Operator Comparison Experiments for Genetic Algorithm Feature Selection
FINAL VERSION - Fixed, Clean, Professional Output
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
import json
from itertools import product
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Tuple, List

from src.genetic_algorithm import GeneticAlgorithm
from src.config import (
    SELECTION_METHODS, CROSSOVER_METHODS, MUTATION_METHODS,
    N_RUNS, EXPERIMENTS_DIR, BEST_FEATURES_DIR, PLOTS_DIR, REPORTS_DIR,
    WEIGHT_THRESHOLD, RANDOM_SEED
)
from src.utils import (
    load_dataset, plot_convergence, plot_feature_reduction,
    plot_operator_comparison, plot_heatmap, plot_feature_frequency
)


# =============================================================================
# HELPER: Convert numpy objects for JSON saving
# =============================================================================
def convert_numpy(obj):
    """Recursively convert numpy arrays and scalars to JSON-serializable types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    if isinstance(obj, tuple):
        return tuple(convert_numpy(i) for i in obj)
    return obj


# =============================================================================
# Run single experiment
# =============================================================================
def run_single_experiment(X_train, y_train, selection, crossover, mutation, run_id):
    ga = GeneticAlgorithm(
        X_train, y_train,
        selection=selection,
        crossover=crossover,
        mutation=mutation
    )
    best_weights, history = ga.evolve()

    best_mask = (best_weights >= WEIGHT_THRESHOLD).astype(int)
    n_selected = int(np.sum(best_mask))
    reduction = (1 - n_selected / len(best_weights)) * 100

    result = {
        'run_id': run_id,
        'selection': selection,
        'crossover': crossover,
        'mutation': mutation,
        'final_fitness': float(ga.best_fitness),
        'n_features_selected': n_selected,
        'reduction_percentage': float(reduction),
        'best_weights': best_weights,
        'best_mask': best_mask
    }
    return result, history


# =============================================================================
# Run all experiments
# =============================================================================
def run_all_experiments(dataset_path):
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    X, y, feature_names = load_dataset(dataset_path)
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    all_results = []
    all_histories = {}
    best_features_list = []

    configs = list(product(SELECTION_METHODS, CROSSOVER_METHODS, MUTATION_METHODS))
    total_runs = len(configs) * N_RUNS

    print(f"\nSelection: {SELECTION_METHODS}")
    print(f"Crossover: {CROSSOVER_METHODS}")
    print(f"Mutation:  {MUTATION_METHODS}")
    print(f"Total configurations: {len(configs)}, Runs per config: {N_RUNS} → {total_runs} total")

    print("\n" + "="*70)
    print("RUNNING EXPERIMENTS")
    print("="*70)

    with tqdm(total=total_runs, desc="Progress", bar_format='{l_bar}{bar} {n_fmt}/{total_fmt} | {elapsed}<{remaining}') as pbar:
        for selection in SELECTION_METHODS:
            for crossover in CROSSOVER_METHODS:
                for mutation in MUTATION_METHODS:
                    config_name = f"{selection}_{crossover}_{mutation}"
                    for run in range(1, N_RUNS + 1):
                        run_id = f"{config_name}_run{run}"
                        np.random.seed(RANDOM_SEED + run)

                        result, history = run_single_experiment(
                            X, y, selection, crossover, mutation, run_id
                        )

                        all_results.append(result)
                        all_histories[run_id] = history

                        # Save best individual info
                        mask = result['best_mask']
                        selected_names = [feature_names[i] for i, selected in enumerate(mask) if selected]
                        best_features_list.append({
                            'run_id': run_id,
                            'selection': selection,
                            'crossover': crossover,
                            'mutation': mutation,
                            'fitness': result['final_fitness'],
                            'n_features': result['n_features_selected'],
                            'reduction_pct': result['reduction_percentage'],
                            'selected_features': selected_names,
                            'best_weights': result['best_weights'],
                            'best_mask': mask.tolist()
                        })

                        pbar.set_postfix({
                            'Config': config_name,
                            'Run': run,
                            'Fitness': f"{result['final_fitness']:.4f}",
                            'Features': result['n_features_selected']
                        })
                        pbar.update(1)

    return pd.DataFrame(all_results), all_histories, feature_names, best_features_list


# =============================================================================
# Generate visualizations
# =============================================================================
def generate_visualizations(results_df, all_histories, feature_names, timestamp):
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Best convergence
    best_run = results_df.loc[results_df['final_fitness'].idxmax()]
    best_history = all_histories[best_run['run_id']]
    plot_convergence(best_history,
                     title=f"Best: {best_run['selection']}+{best_run['crossover']}+{best_run['mutation']}",
                     save_path=os.path.join(PLOTS_DIR, f"best_convergence_{timestamp}.png"))

    # Operator comparison
    plot_operator_comparison(results_df, metric='final_fitness',
                             save_path=os.path.join(PLOTS_DIR, f"comparison_{timestamp}.png"))

    # Heatmaps
    for sel in SELECTION_METHODS:
        plot_heatmap(results_df, sel,
                     save_path=os.path.join(PLOTS_DIR, f"heatmap_{sel}_{timestamp}.png"))

    # Feature frequency
    masks = [np.array(r['best_mask']) for _, r in results_df.iterrows()]
    plot_feature_frequency(masks, feature_names, top_n=20,
                           save_path=os.path.join(PLOTS_DIR, f"feature_frequency_{timestamp}.png"))

    print(f"All plots saved to: {PLOTS_DIR}")


# =============================================================================
# Print beautiful final report
# =============================================================================
def print_final_report(results_df, best_features_list, feature_names, timestamp):
    print("\n" + "="*80)
    print("TOP 5 BEST CONFIGURATIONS - SELECTED FEATURES")
    print("="*80)

    top5 = results_df.sort_values('final_fitness', ascending=False).head(5)

    report_lines = []
    report_lines.append("GENETIC ALGORITHM FEATURE SELECTION - FINAL REPORT")
    report_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Dataset: cleaned_diabetes_data.csv ({len(feature_names)} features)")
    report_lines.append("="*80 + "\n")

    for idx, row in top5.iterrows():
        config = f"{row['selection']} + {row['crossover']} + {row['mutation']}"
        features = next(f['selected_features'] for f in best_features_list if f['run_id'] == row['run_id'])
        n_feat = len(features)
        reduction = row['reduction_percentage']

        print(f"\n#{idx + 1} | {config}")
        print(f"    Fitness: {row['final_fitness']:.6f}")
        print(f"    Features: {n_feat}/{len(feature_names)} (↓ {reduction:.1f}%)")
        print(f"    Selected Features ({n_feat}):")
        for name in features:
            print(f"      • {name}")
        print("-" * 80)

        report_lines.append(f"TOP {idx + 1} | {config}")
        report_lines.append(f"Fitness: {row['final_fitness']:.6f} | Features: {n_feat} | Reduction: {reduction:.1f}%")
        report_lines.append("Features: " + ", ".join(features))
        report_lines.append("")

    # Save report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, f"final_report_{timestamp}.txt")
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    print(f"\nFinal report saved: {report_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_operators.py <path_to_dataset.csv>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    os.makedirs(BEST_FEATURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "="*70)
    print("GA OPERATOR COMPARISON EXPERIMENTS")
    print("="*70)
    print(f"Dataset: {dataset_path}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run experiments
    results_df, histories, feature_names, best_features_list = run_all_experiments(dataset_path)

    # Save results
    results_path = os.path.join(EXPERIMENTS_DIR, f"all_results_{timestamp}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved: {results_path}")

    # Save histories and best features (with numpy conversion)
    histories_path = os.path.join(EXPERIMENTS_DIR, f"histories_{timestamp}.json")
    with open(histories_path, 'w') as f:
        json.dump(convert_numpy(histories), f, indent=2)

    best_feat_path = os.path.join(BEST_FEATURES_DIR, f"best_features_{timestamp}.json")
    with open(best_feat_path, 'w') as f:
        json.dump(convert_numpy(best_features_list), f, indent=2)

    # Generate plots
    generate_visualizations(results_df, histories, feature_names, timestamp)

    # Final beautiful report
    print_final_report(results_df, best_features_list, feature_names, timestamp)

    print("\n" + "="*70)
    print("ALL DONE! Check results/reports/ for your final report.")
    print("="*70)


if __name__ == "__main__":
    main()