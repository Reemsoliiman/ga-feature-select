"""
Command-Line Interface for GA Feature Selection
Alternative to GUI for quick experiments.
"""
import argparse
import os
from src.genetic_algorithm import GeneticAlgorithm
from src.utils import load_dataset, split_data, plot_convergence
from src.config import SELECTION_METHODS, CROSSOVER_METHODS, MUTATION_METHODS

def main():
    parser = argparse.ArgumentParser(description='GA Feature Selection')
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset CSV file')
    
    # Operator selection
    parser.add_argument('--selection', type=str, default='tournament',
                       choices=SELECTION_METHODS,
                       help='Selection method')
    parser.add_argument('--crossover', type=str, default='single_point',
                       choices=CROSSOVER_METHODS,
                       help='Crossover method')
    parser.add_argument('--mutation', type=str, default='bit_flip',
                       choices=MUTATION_METHODS,
                       help='Mutation method')
    
    # GA parameters
    parser.add_argument('--pop-size', type=int, default=50,
                       help='Population size')
    parser.add_argument('--generations', type=int, default=100,
                       help='Number of generations')
    parser.add_argument('--mut-rate', type=float, default=0.01,
                       help='Mutation rate')
    parser.add_argument('--cross-rate', type=float, default=0.8,
                       help='Crossover rate')
    
    # Output
    parser.add_argument('--output', type=str, default='results/plots/convergence.png',
                       help='Output path for convergence plot')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading dataset from {args.dataset}...")
    X, y, feature_names = load_dataset(args.dataset)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Training: {X_train.shape[0]}, Testing: {X_test.shape[0]}")
    
    # Initialize GA
    print(f"\nConfiguration:")
    print(f"  Selection: {args.selection}")
    print(f"  Crossover: {args.crossover}")
    print(f"  Mutation: {args.mutation}")
    print(f"  Population: {args.pop_size}")
    print(f"  Generations: {args.generations}")
    
    ga = GeneticAlgorithm(
        X_train, y_train,
        selection=args.selection,
        crossover=args.crossover,
        mutation=args.mutation,
        pop_size=args.pop_size,
        n_generations=args.generations,
        mut_rate=args.mut_rate,
        cross_rate=args.cross_rate
    )
    
    # Run evolution
    print(f"\nRunning GA for {args.generations} generations...")
    best_individual, history = ga.evolve()
    
    # Results
    selected_indices = ga.get_selected_features()
    selected_names = [feature_names[i] for i in selected_indices]
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Final Fitness: {history['best_fitness'][-1]:.4f}")
    print(f"Features Selected: {len(selected_names)}/{len(feature_names)}")
    print(f"Reduction: {(1 - len(selected_names)/len(feature_names)) * 100:.1f}%")
    print(f"\nSelected Features:")
    for i, name in enumerate(selected_names, 1):
        print(f"  {i}. {name}")
    
    # Save plot
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plot_convergence(history, save_path=args.output)
    print(f"\n✓ Convergence plot saved to {args.output}")

if __name__ == "__main__":
    main()