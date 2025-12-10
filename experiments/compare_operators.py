"""
Operator Comparison Experiments for Genetic Algorithm Feature Selection

This module runs comprehensive experiments to compare different GA operators:
- Selection methods (tournament, roulette, rank)
- Crossover methods (single_point, uniform, arithmetic)
- Mutation methods (bit_flip, uniform, adaptive)

Each configuration is run multiple times for statistical significance.
Results are saved as CSV files and JSON for further analysis.

Usage:
    python compare_operators.py <path_to_dataset.csv>

Example:
    python compare_operators.py data/processed/cleaned_diabetes_data.csv

"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
from itertools import product
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Tuple, List

# Import GA components
from src.genetic_algorithm import GeneticAlgorithm
from src.config import (
    SELECTION_METHODS,
    CROSSOVER_METHODS,
    MUTATION_METHODS,
    N_RUNS,
    EXPERIMENTS_DIR,
    BEST_FEATURES_DIR,
    PLOTS_DIR,
    WEIGHT_THRESHOLD,
    TEST_SIZE,
    STRATIFY,
    RANDOM_SEED
)

# For data loading
from sklearn.model_selection import train_test_split


def load_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load dataset from CSV file.
    
    Parameters
    ----------
    dataset_path : str
        Path to CSV file with features and target column.
        
    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    feature_names : list of str
        Names of features.
        
    Notes
    -----
    Assumes last column is the target variable.
    """
    # Load data
    df = pd.read_csv(dataset_path)
    
    # Separate features and target (assuming last column is target)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-1].tolist()
    
    return X, y, feature_names


def split_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
        
    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Training and test splits.
    """
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y if STRATIFY else None,
        random_state=RANDOM_SEED
    )


def convert_weights_to_binary(weights: np.ndarray, threshold: float = WEIGHT_THRESHOLD) -> np.ndarray:
    """
    Convert continuous weights to binary mask.
    
    Parameters
    ----------
    weights : np.ndarray
        Weight vector with values in [0, 1].
    threshold : float
        Threshold for selection (default: from config).
        
    Returns
    -------
    mask : np.ndarray
        Binary mask where 1 = selected (weight >= threshold).
    """
    return (weights >= threshold).astype(int)


def run_single_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    selection: str,
    crossover: str,
    mutation: str,
    run_id: str
) -> Tuple[Dict, Dict]:
    """
    Run a single GA experiment with specific operators.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    selection : str
        Selection method name.
    crossover : str
        Crossover method name.
    mutation : str
        Mutation method name.
    run_id : str
        Unique identifier for this run.
        
    Returns
    -------
    result : dict
        Dictionary with experiment results and metrics.
    history : dict
        Evolution history from GA.
    """
    # Initialize GA with specified operators
    ga = GeneticAlgorithm(
        X_train,
        y_train,
        selection=selection,
        crossover=crossover,
        mutation=mutation
    )
    
    # Run evolution
    best_weights, history = ga.evolve()
    
    # Convert best weights to binary mask for feature counting
    best_mask = convert_weights_to_binary(best_weights, ga.weight_threshold)
    n_features_selected = int(np.sum(best_mask))
    n_total_features = ga.n_features
    reduction_percentage = (1 - n_features_selected / n_total_features) * 100
    
    # Get final fitness (accuracy - penalty)
    final_fitness = ga.best_fitness
    
    # Calculate convergence generation (when fitness stabilizes)
    convergence_generation = calculate_convergence_generation(history['best_fitness'])
    
    # Get final accuracy (last value in history)
    # Note: best_fitness includes penalty, so we extract from history
    final_best_fitness = history['best_fitness'][-1]
    
    # Prepare result dictionary
    result = {
        'run_id': run_id,
        'selection': selection,
        'crossover': crossover,
        'mutation': mutation,
        'final_fitness': float(final_fitness),
        'final_best_fitness': float(final_best_fitness),
        'n_features_selected': n_features_selected,
        'n_total_features': n_total_features,
        'reduction_percentage': float(reduction_percentage),
        'convergence_generation': convergence_generation,
        'best_weights': best_weights.tolist(),
        'best_mask': best_mask.tolist()
    }
    
    return result, history


def calculate_convergence_generation(fitness_history: List[float], window: int = 10, threshold: float = 0.001) -> int:
    """
    Calculate generation where fitness converges (stabilizes).
    
    Convergence is defined as when the standard deviation of fitness
    over a sliding window falls below a threshold.
    
    Parameters
    ----------
    fitness_history : list of float
        Best fitness values per generation.
    window : int
        Size of sliding window for convergence check.
    threshold : float
        Standard deviation threshold for convergence.
        
    Returns
    -------
    convergence_gen : int or None
        Generation number where convergence occurred, or None if no convergence.
    """
    if len(fitness_history) < window:
        return None
    
    for i in range(window, len(fitness_history)):
        window_fitness = fitness_history[i - window:i]
        if np.std(window_fitness) < threshold:
            return i
    
    return None  # No convergence detected


def run_all_experiments(dataset_path: str) -> Tuple[pd.DataFrame, Dict, List[str]]:
    """
    Run all operator combinations multiple times.
    
    Generates all combinations of selection, crossover, and mutation methods,
    runs each configuration N_RUNS times, and collects results.
    
    Parameters
    ----------
    dataset_path : str
        Path to dataset CSV file.
        
    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with all experiment results.
    all_histories : dict
        Dictionary mapping run_id to evolution history.
    feature_names : list of str
        Names of features in the dataset.
    """
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    
    # Load dataset
    X, y, feature_names = load_dataset(dataset_path)
    print(f"✓ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    
    # Generate all operator configurations
    configs = list(product(SELECTION_METHODS, CROSSOVER_METHODS, MUTATION_METHODS))
    total_runs = len(configs) * N_RUNS
    
    print("\n" + "="*70)
    print("EXPERIMENT CONFIGURATION")
    print("="*70)
    print(f"Selection methods: {SELECTION_METHODS}")
    print(f"Crossover methods: {CROSSOVER_METHODS}")
    print(f"Mutation methods: {MUTATION_METHODS}")
    print(f"\nTotal configurations: {len(configs)}")
    print(f"Runs per configuration: {N_RUNS}")
    print(f"Total experiments: {total_runs}")
    
    # Storage for results
    all_results = []
    all_histories = {}
    
    # Progress bar
    print("\n" + "="*70)
    print("RUNNING EXPERIMENTS")
    print("="*70)
    pbar = tqdm(total=total_runs, desc="Progress", unit="run")
    
    # Run all experiments
    for selection, crossover, mutation in configs:
        config_name = f"{selection}_{crossover}_{mutation}"
        
        for run in range(N_RUNS):
            run_id = f"{config_name}_run{run+1}"
            
            try:
                # Run single experiment
                result, history = run_single_experiment(
                    X_train, y_train,
                    selection, crossover, mutation,
                    run_id
                )
                
                # Store results
                all_results.append(result)
                all_histories[run_id] = history
                
            except Exception as e:
                print(f"\n⚠️  Error in {run_id}: {e}")
                # Store failed result
                all_results.append({
                    'run_id': run_id,
                    'selection': selection,
                    'crossover': crossover,
                    'mutation': mutation,
                    'final_fitness': -1.0,
                    'error': str(e)
                })
            
            pbar.update(1)
    
    pbar.close()
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save main results CSV
    results_path = os.path.join(EXPERIMENTS_DIR, f"all_results_{timestamp}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"✓ Results saved to: {results_path}")
    
    # Save histories JSON
    histories_path = os.path.join(EXPERIMENTS_DIR, f"histories_{timestamp}.json")
    with open(histories_path, 'w') as f:
        json.dump(all_histories, f, indent=2)
    print(f"✓ Histories saved to: {histories_path}")
    
    # Save best individuals
    best_features_path = os.path.join(BEST_FEATURES_DIR, f"best_individuals_{timestamp}.json")
    best_individuals = {
        row['run_id']: {
            'weights': row['best_weights'],
            'mask': row['best_mask'],
            'n_selected': row['n_features_selected']
        }
        for _, row in results_df.iterrows()
        if 'best_weights' in row
    }
    with open(best_features_path, 'w') as f:
        json.dump(best_individuals, f, indent=2)
    print(f"✓ Best individuals saved to: {best_features_path}")
    
    return results_df, all_histories, feature_names


def generate_summary_statistics(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics grouped by operator configuration.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with all experiment results.
        
    Returns
    -------
    summary_df : pd.DataFrame
        Summary statistics with mean, std, min, max for each configuration.
    """
    print("\n" + "="*70)
    print("GENERATING SUMMARY STATISTICS")
    print("="*70)
    
    # Group by operator configuration
    summary = results_df.groupby(['selection', 'crossover', 'mutation']).agg({
        'final_fitness': ['mean', 'std', 'min', 'max'],
        'n_features_selected': ['mean', 'std'],
        'reduction_percentage': ['mean', 'std'],
        'convergence_generation': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    # Sort by mean fitness (descending)
    summary = summary.sort_values('final_fitness_mean', ascending=False)
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(EXPERIMENTS_DIR, f"summary_{timestamp}.csv")
    summary.to_csv(summary_path, index=False)
    print(f"✓ Summary saved to: {summary_path}")
    
    # Display top configurations
    print("\n📊 TOP 5 OPERATOR CONFIGURATIONS:")
    print("-" * 70)
    top5 = summary.head()
    for idx, row in top5.iterrows():
        print(f"\n{idx + 1}. {row['selection']} + {row['crossover']} + {row['mutation']}")
        print(f"   Fitness: {row['final_fitness_mean']:.4f} ± {row['final_fitness_std']:.4f}")
        print(f"   Features: {row['n_features_selected_mean']:.1f} ± {row['n_features_selected_std']:.1f}")
        print(f"   Reduction: {row['reduction_percentage_mean']:.1f}% ± {row['reduction_percentage_std']:.1f}%")
    
    return summary


def main():
    """Main function to run all experiments."""
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Error: Dataset path required")
        print("\nUsage: python compare_operators.py <path_to_dataset.csv>")
        print("Example: python compare_operators.py data/processed/cleaned_diabetes_data.csv")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    # Verify dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)
    
    # Create output directories
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    os.makedirs(BEST_FEATURES_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    print("\n" + "="*70)
    print("GA OPERATOR COMPARISON EXPERIMENTS")
    print("="*70)
    print(f"Dataset: {dataset_path}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all experiments
    results_df, histories, feature_names = run_all_experiments(dataset_path)
    
    # Generate summary statistics
    summary_df = generate_summary_statistics(results_df)
    
    # Final summary
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE!")
    print("="*70)
    print(f"✓ Total configurations tested: {len(summary_df)}")
    print(f"✓ Total runs completed: {len(results_df)}")
    print(f"✓ Results saved to: {EXPERIMENTS_DIR}")
    
    print("\n📁 NEXT STEPS:")
    print("1. Analyze results: Check CSV files in results/experiments/")
    print("2. Visualize evolution: Use visualization tools on histories JSON")
    print("3. Compare operators: Review summary statistics CSV")
    print("4. Generate report: Use results for final project report")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()