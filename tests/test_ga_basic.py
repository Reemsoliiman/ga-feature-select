"""
Simple test script to verify GeneticAlgorithm implementation.

This script runs a minimal GA with a small population and few generations
to verify that all components work together correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.genetic_algorithm import GeneticAlgorithm
from src import config

def test_ga_basic():
    """Test basic GA functionality."""
    print("=" * 70)
    print("Testing GeneticAlgorithm Implementation")
    print("=" * 70)
    
    # Dataset path
    dataset_path = config.DEFAULT_DATASET_PATH
    print(f"\nDataset: {dataset_path}")
    
    # Create GA with small parameters for quick test
    print("\nInitializing Genetic Algorithm...")
    ga = GeneticAlgorithm(
        dataset_path=dataset_path,
        population_size=10,
        n_generations=5,
        mutation_rate=0.1,
        crossover_rate=0.8,
        selection_method='tournament',
        crossover_method='arithmetic',
        mutation_method='uniform',
        elitism_rate=0.2,
        random_state=42
    )
    
    print(f"  Population size: {ga.population_size}")
    print(f"  Generations: {ga.n_generations}")
    print(f"  Features: {ga.n_features}")
    print(f"  Samples: {ga.X.shape[0]}")
    print(f"  Elitism rate: {ga.elitism_rate}")
    
    # Run evolution
    print("\n" + "=" * 70)
    print("Running Evolution...")
    print("=" * 70 + "\n")
    
    best_weights, best_fitness, history, selected_features = ga.evolve()
    
    # Print results
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"\nBest Fitness: {best_fitness:.4f}")
    print(f"Number of Selected Features: {len(selected_features)}/{ga.n_features}")
    print(f"\nSelected Features:")
    for i, feature in enumerate(selected_features, 1):
        idx = ga.feature_names.index(feature)
        weight = best_weights[idx]
        print(f"  {i}. {feature} (weight: {weight:.3f})")
    
    print(f"\nConvergence:")
    print(f"  Initial Best Fitness: {history['best_fitness'][0]:.4f}")
    print(f"  Final Best Fitness: {history['best_fitness'][-1]:.4f}")
    print(f"  Improvement: {history['best_fitness'][-1] - history['best_fitness'][0]:.4f}")
    
    print("\n" + "=" * 70)
    print("Test Completed Successfully!")
    print("=" * 70)

if __name__ == "__main__":
    test_ga_basic()
