"""
Genetic Algorithm for Feature Selection using Continuous Weight Encoding.

This module implements a Genetic Algorithm (GA) for feature selection in machine learning.
Each individual represents a continuous weight vector where weights in [0, 1] indicate
feature importance. Features with weight >= threshold are considered selected.

The GA orchestrates the evolution process by delegating to operator functions
for selection, crossover, and mutation operations.

Key Features:
    - Continuous weight encoding (not binary)
    - Multiple selection strategies (tournament, roulette, rank)
    - Multiple crossover operators (single-point, uniform, arithmetic)
    - Multiple mutation operators (bit-flip, uniform, adaptive)
    - Elitism to preserve best solutions
    - Comprehensive evolution tracking
"""

import numpy as np
import math
from typing import Tuple, Dict, List, Optional

from . import operators
from .fitness import FitnessEvaluator
from .config import (
    POPULATION_SIZE, N_GENERATIONS, MUTATION_RATE, CROSSOVER_RATE,
    ELITISM_RATE, TOURNAMENT_SIZE, RANDOM_STATE, WEIGHT_THRESHOLD, 
    MAX_ELITISM_FRACTION, VERBOSE, SELECTION_METHODS, CROSSOVER_METHODS,
    MUTATION_METHODS, DEFAULT_SELECTION, DEFAULT_CROSSOVER, DEFAULT_MUTATION,
    EARLY_STOPPING, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_GENS, EARLY_STOPPING_DELTA
)


class GeneticAlgorithm:
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        selection: str = DEFAULT_SELECTION,
        crossover: str = DEFAULT_CROSSOVER,
        mutation: str = DEFAULT_MUTATION,
        **kwargs
    ):
        # Data
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        
        # Hyperparameters
        self.pop_size = kwargs.get('pop_size', POPULATION_SIZE)
        self.n_generations = kwargs.get('n_generations', N_GENERATIONS)
        self.mut_rate = kwargs.get('mut_rate', MUTATION_RATE)
        self.cross_rate = kwargs.get('cross_rate', CROSSOVER_RATE)
        self.tournament_size = kwargs.get('tournament_size', TOURNAMENT_SIZE)
        self.weight_threshold = kwargs.get('weight_threshold', WEIGHT_THRESHOLD)
        
        # Early stopping
        self.early_stopping = EARLY_STOPPING
        self.patience = EARLY_STOPPING_PATIENCE
        self.min_gens = EARLY_STOPPING_MIN_GENS
        self.delta = EARLY_STOPPING_DELTA

        self.gens_no_improve = 0
        self.best_fitness_ever = -np.inf

        # Operator methods
        self.selection_method = selection
        self.crossover_method = crossover
        self.mutation_method = mutation
        
        # Validate operator methods
        self._validate_operators()
        
        # Elitism configuration
        n_elites = kwargs.get('n_elites', None)
        elitism_rate = kwargs.get('elitism_rate', ELITISM_RATE)
        
        if n_elites is not None:
            if n_elites <= 0:
                raise ValueError("n_elites must be > 0 when provided.")
            self.n_elites = n_elites
            self.elitism_rate = None
        else:
            if elitism_rate <= 0:
                raise ValueError("elitism_rate must be > 0.")
            self.elitism_rate = elitism_rate
            self.n_elites = None
        
        # Random state
        random_state = kwargs.get('random_state', RANDOM_STATE)
        self.rng = np.random.RandomState(random_state)
        
        # Fitness evaluator
        self.evaluator = FitnessEvaluator(X, y)
        
        # Population and fitness (initialized during evolution)
        self.population = None
        self.fitness_scores = None
        
        # Best solution tracking
        self.best_individual = None
        self.best_fitness = -np.inf
        
        # Evolution history
        self.history = {
            'best_fitness': [],
            'mean_fitness': [],
            'best_weights': [],
            'n_selected_features': []
        }
    
    def _validate_operators(self):
        """Validate that operator methods are supported."""
        if self.selection_method not in SELECTION_METHODS:
            raise ValueError(
                f"Unknown selection method: {self.selection_method}. "
                f"Choose from {SELECTION_METHODS}"
            )
        if self.crossover_method not in CROSSOVER_METHODS:
            raise ValueError(
                f"Unknown crossover method: {self.crossover_method}. "
                f"Choose from {CROSSOVER_METHODS}"
            )
        if self.mutation_method not in MUTATION_METHODS:
            raise ValueError(
                f"Unknown mutation method: {self.mutation_method}. "
                f"Choose from {MUTATION_METHODS}"
            )
    
    def _compute_elite_count(self) -> int:
        if self.n_elites is not None:
            elite_count = self.n_elites
        else:
            elite_count = math.ceil(self.elitism_rate * self.pop_size)
        
        # Clamp to reasonable bounds
        max_elites = math.floor(MAX_ELITISM_FRACTION * self.pop_size)
        max_elites = max(1, max_elites)
        
        elite_count = max(1, elite_count)
        elite_count = min(elite_count, max_elites)
        elite_count = min(elite_count, self.pop_size - 1)
        
        return elite_count
    
    # =============================================================================
    # INITIALIZATION
    # =============================================================================
    
    def _initialize_population(self) -> np.ndarray:
        """
        Initialize population with random continuous weights in [0, 1].
        
        Each individual is a vector of continuous weights, one per feature.
        Weights are uniformly sampled from [0, 1].
        
        Returns
        -------
        population : np.ndarray
            Population array of shape (pop_size, n_features).
        """
        population = self.rng.uniform(
            low=0.0,
            high=1.0,
            size=(self.pop_size, self.n_features)
        )
        return population
    
    # =============================================================================
    # FITNESS EVALUATION
    # =============================================================================
    
    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """
        Evaluate fitness for all individuals in the population.
        
        Converts each weight vector to a binary mask using the threshold,
        then evaluates fitness using the FitnessEvaluator.
        
        Parameters
        ----------
        population : np.ndarray
            Population array of shape (pop_size, n_features).
            
        Returns
        -------
        fitness_scores : np.ndarray
            Array of fitness scores of shape (pop_size,).
        """
        fitness_scores = []
        
        for weights in population:
            # Convert weights to binary mask
            mask = self._weights_to_mask(weights)
            
            # Evaluate fitness
            fitness = self.evaluator.evaluate_individual(mask)
            fitness_scores.append(fitness)
        
        return np.array(fitness_scores)
    
    def _weights_to_mask(self, weights: np.ndarray) -> np.ndarray:
        return (weights >= self.weight_threshold).astype(int)
    
    # =============================================================================
    # GENETIC OPERATORS
    # =============================================================================
    def _select_parents(self, n_parents: int) -> np.ndarray:
        if self.selection_method == 'tournament':
            return operators.tournament_selection(
                self.population,
                self.fitness_scores,
                n_parents,
                tournament_size=self.tournament_size,
                rng=self.rng
            )
        elif self.selection_method == 'roulette':
            return operators.roulette_selection(
                self.population,
                self.fitness_scores,
                n_parents,
                rng=self.rng
            )
        elif self.selection_method == 'rank':
            return operators.rank_selection(
                self.population,
                self.fitness_scores,
                n_parents,
                rng=self.rng
            )
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    def _crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Apply crossover with probability cross_rate
        if self.rng.rand() >= self.cross_rate:
            return parent1.copy(), parent2.copy()
        
        if self.crossover_method == 'single_point':
            return operators.single_point_crossover(parent1, parent2, rng=self.rng)
        elif self.crossover_method == 'uniform':
            return operators.uniform_crossover(parent1, parent2, rng=self.rng)
        elif self.crossover_method == 'arithmetic':
            return operators.arithmetic_crossover(parent1, parent2, rng=self.rng)
        else:
            raise ValueError(f"Unknown crossover method: {self.crossover_method}")
    
    def _mutate(self, individual: np.ndarray, generation: int = 0) -> np.ndarray:
        if self.mutation_method == 'bit_flip':
            return operators.bit_flip_mutation(individual, self.mut_rate, rng=self.rng)
        elif self.mutation_method == 'uniform':
            return operators.uniform_mutation(individual, self.mut_rate, rng=self.rng)
        elif self.mutation_method == 'adaptive':
            return operators.adaptive_mutation(
                individual,
                self.mut_rate,
                generation=generation,
                max_generations=self.n_generations,
                rng=self.rng
            )
        else:
            raise ValueError(f"Unknown mutation method: {self.mutation_method}")
    
    # =============================================================================
    # EVOLUTION
    # =============================================================================
    
    def evolve(self) -> Tuple[np.ndarray, Dict]:
        """
        Run the genetic algorithm evolution loop with early stopping.
        """
        # Initialize population
        self.population = self._initialize_population()
        
        # Early stopping initialization
        self.gens_no_improve = 0
        self.best_fitness_ever = -np.inf  # Tracks true best across all generations

        # Evolution loop
        for generation in range(self.n_generations):
            # Evaluate fitness
            self.fitness_scores = self._evaluate_population(self.population)
            
            # Find current generation's best
            best_idx = np.argmax(self.fitness_scores)
            gen_best_fitness = self.fitness_scores[best_idx]
            gen_best_weights = self.population[best_idx].copy()

            # === EARLY STOPPING & BEST TRACKING LOGIC ===
            improved = False
            if gen_best_fitness > self.best_fitness_ever + self.delta:
                self.best_fitness_ever = gen_best_fitness
                self.best_individual = gen_best_weights.copy()
                self.best_fitness = gen_best_fitness
                self.gens_no_improve = 0
                improved = True
            else:
                self.gens_no_improve += 1

            # Always update current best (for history)
            if gen_best_fitness > self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_individual = gen_best_weights.copy()

            # Track history (based on current global best)
            mean_fitness = np.mean(self.fitness_scores)
            best_mask = self._weights_to_mask(self.best_individual)
            n_selected = int(np.sum(best_mask))

            self.history['best_fitness'].append(self.best_fitness)
            self.history['mean_fitness'].append(mean_fitness)
            self.history['best_weights'].append(self.best_individual.copy())
            self.history['n_selected_features'].append(n_selected)

            # Verbose output
            if VERBOSE:
                status = " (improved)" if improved else ""
                print(
                    f"Generation {generation + 1}/{self.n_generations} | "
                    f"Best Fitness: {self.best_fitness:.6f}{status} | "
                    f"Mean Fitness: {mean_fitness:.4f} | "
                    f"Features: {n_selected}/{self.n_features} | "
                    f"No improve: {self.gens_no_improve}"
                )

            # === EARLY STOPPING CHECK ===
            if (self.early_stopping 
                and generation >= self.min_gens 
                and self.gens_no_improve >= self.patience):
                
                if VERBOSE:
                    print(f"\nEARLY STOPPING at generation {generation + 1}")
                    print(f"No improvement in fitness for {self.patience} generations.")
                    print(f"Best fitness achieved: {self.best_fitness:.6f} with {n_selected} features\n")
                break

            # === REST OF EVOLUTION (unchanged) ===
            elite_count = self._compute_elite_count()
            sorted_indices = np.argsort(self.fitness_scores)[::-1]
            sorted_population = self.population[sorted_indices]

            next_population = []
            elites = sorted_population[:elite_count].copy()
            next_population.extend(elites)

            while len(next_population) < self.pop_size:
                parents = self._select_parents(n_parents=2)
                parent1, parent2 = parents[0], parents[1]
                offspring1, offspring2 = self._crossover(parent1, parent2)
                offspring1 = self._mutate(offspring1, generation)
                offspring2 = self._mutate(offspring2, generation)
                next_population.append(offspring1)
                if len(next_population) < self.pop_size:
                    next_population.append(offspring2)

            self.population = np.array(next_population[:self.pop_size])

        return self.best_individual, self.history

    def get_selected_features(self, feature_names: Optional[List[str]] = None) -> List:
        if self.best_individual is None:
            raise ValueError("No evolution has been run yet. Call evolve() first.")
        
        mask = self._weights_to_mask(self.best_individual)
        selected_indices = np.where(mask == 1)[0]
        
        if feature_names is not None:
            return [feature_names[i] for i in selected_indices]
        else:
            return selected_indices.tolist()