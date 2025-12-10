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
    MUTATION_METHODS, DEFAULT_SELECTION, DEFAULT_CROSSOVER, DEFAULT_MUTATION
)


class GeneticAlgorithm:
    """
    Genetic Algorithm for feature selection with continuous weight encoding.
    
    Each individual is represented as a continuous weight vector in [0, 1].
    For fitness evaluation, weights are converted to binary masks using a threshold.
    
    The class orchestrates the evolution process, delegating operator implementations
    to the operators module for cleaner separation of concerns.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target labels of shape (n_samples,).
    selection : str, optional
        Selection method: 'tournament', 'roulette', 'rank' (default: from config).
    crossover : str, optional
        Crossover method: 'single_point', 'uniform', 'arithmetic' (default: from config).
    mutation : str, optional
        Mutation method: 'bit_flip', 'uniform', 'adaptive' (default: from config).
    **kwargs : dict
        Override config values:
        - pop_size: Population size
        - n_generations: Number of generations
        - mut_rate: Mutation rate
        - cross_rate: Crossover rate
        - elitism_rate: Fraction of elites to preserve
        - n_elites: Absolute number of elites (overrides elitism_rate)
        - tournament_size: Tournament size for tournament selection
        - weight_threshold: Threshold for binary conversion
        - random_state: Random seed
    
    Attributes
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    n_features : int
        Number of features.
    population : np.ndarray
        Current population of shape (pop_size, n_features).
    fitness_scores : np.ndarray
        Fitness scores for current population.
    best_individual : np.ndarray
        Best weight vector found during evolution.
    best_fitness : float
        Fitness of best individual.
    history : dict
        Evolution history tracking metrics per generation.
    
    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> ga = GeneticAlgorithm(X, y, selection='tournament', crossover='uniform')
    >>> best_weights, history = ga.evolve()
    >>> selected_features = ga.get_selected_features()
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        selection: str = DEFAULT_SELECTION,
        crossover: str = DEFAULT_CROSSOVER,
        mutation: str = DEFAULT_MUTATION,
        **kwargs
    ):
        """Initialize the Genetic Algorithm."""
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
        """
        Compute the number of elites for the current generation.
        
        Returns
        -------
        elite_count : int
            Number of elites, clamped to [1, floor(0.3 * pop_size)]
            and never exceeding pop_size - 1.
        """
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
        """
        Convert continuous weights to binary mask.
        
        Parameters
        ----------
        weights : np.ndarray
            Weight vector in [0, 1].
            
        Returns
        -------
        mask : np.ndarray
            Binary mask where 1 indicates weight >= threshold.
        """
        return (weights >= self.weight_threshold).astype(int)
    
    # =============================================================================
    # GENETIC OPERATORS (Delegation to operators module)
    # =============================================================================
    
    def _select_parents(self, n_parents: int) -> np.ndarray:
        """
        Select parents using configured selection method.
        
        Delegates to the appropriate operator function based on selection_method.
        
        Parameters
        ----------
        n_parents : int
            Number of parents to select.
            
        Returns
        -------
        parents : np.ndarray
            Selected parents of shape (n_parents, n_features).
        """
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
        """
        Apply crossover to two parents using configured method.
        
        Applies crossover with probability cross_rate. If crossover is not
        applied, returns copies of parents.
        
        Parameters
        ----------
        parent1, parent2 : np.ndarray
            Parent weight vectors.
            
        Returns
        -------
        offspring1, offspring2 : np.ndarray
            Two offspring weight vectors.
        """
        # Apply crossover with probability cross_rate
        if self.rng.rand() >= self.cross_rate:
            return parent1.copy(), parent2.copy()
        
        # Delegate to operator function
        if self.crossover_method == 'single_point':
            return operators.single_point_crossover(parent1, parent2, rng=self.rng)
        elif self.crossover_method == 'uniform':
            return operators.uniform_crossover(parent1, parent2, rng=self.rng)
        elif self.crossover_method == 'arithmetic':
            return operators.arithmetic_crossover(parent1, parent2, rng=self.rng)
        else:
            raise ValueError(f"Unknown crossover method: {self.crossover_method}")
    
    def _mutate(self, individual: np.ndarray, generation: int = 0) -> np.ndarray:
        """
        Apply mutation to an individual using configured method.
        
        Delegates to the appropriate mutation operator function.
        
        Parameters
        ----------
        individual : np.ndarray
            Weight vector to mutate.
        generation : int, optional
            Current generation (for adaptive mutation).
            
        Returns
        -------
        mutated : np.ndarray
            Mutated weight vector.
        """
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
        Run the genetic algorithm evolution loop.
        
        The algorithm:
        1. Initialize population with random weights
        2. For each generation:
            a. Evaluate fitness for all individuals
            b. Track best solution and history
            c. Compute elite count
            d. Sort population by fitness
            e. Copy elites to next generation
            f. Generate offspring via selection, crossover, mutation
            g. Update population
        3. Return best solution and history
        
        Returns
        -------
        best_weights : np.ndarray
            Best weight vector found.
        history : dict
            Evolution history with keys:
            - 'best_fitness': list of best fitness per generation
            - 'mean_fitness': list of mean fitness per generation
            - 'best_weights': list of best weight vectors per generation
            - 'n_selected_features': list of feature counts per generation
        """
        # Initialize population
        self.population = self._initialize_population()
        
        # Evolution loop
        for generation in range(self.n_generations):
            # Evaluate fitness
            self.fitness_scores = self._evaluate_population(self.population)
            
            # Track best solution
            best_idx = np.argmax(self.fitness_scores)
            gen_best_fitness = self.fitness_scores[best_idx]
            gen_best_weights = self.population[best_idx].copy()
            
            if gen_best_fitness > self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_individual = gen_best_weights
            
            # Track history
            mean_fitness = np.mean(self.fitness_scores)
            best_mask = self._weights_to_mask(self.best_individual)
            n_selected = np.sum(best_mask)
            
            self.history['best_fitness'].append(self.best_fitness)
            self.history['mean_fitness'].append(mean_fitness)
            self.history['best_weights'].append(self.best_individual.copy())
            self.history['n_selected_features'].append(n_selected)
            
            # Verbose output
            if VERBOSE:
                print(
                    f"Generation {generation + 1}/{self.n_generations} | "
                    f"Best Fitness: {self.best_fitness:.4f} | "
                    f"Mean Fitness: {mean_fitness:.4f} | "
                    f"Features: {n_selected}/{self.n_features}"
                )
            
            # Compute elite count
            elite_count = self._compute_elite_count()
            
            # Sort population by fitness (descending)
            sorted_indices = np.argsort(self.fitness_scores)[::-1]
            sorted_population = self.population[sorted_indices]
            
            # Create next generation
            next_population = []
            
            # Step 1: Copy elites
            elites = sorted_population[:elite_count].copy()
            next_population.extend(elites)
            
            # Step 2: Generate offspring
            while len(next_population) < self.pop_size:
                # Select two parents
                parents = self._select_parents(n_parents=2)
                parent1, parent2 = parents[0], parents[1]
                
                # Crossover
                offspring1, offspring2 = self._crossover(parent1, parent2)
                
                # Mutation
                offspring1 = self._mutate(offspring1, generation)
                offspring2 = self._mutate(offspring2, generation)
                
                # Add to next generation
                next_population.append(offspring1)
                if len(next_population) < self.pop_size:
                    next_population.append(offspring2)
            
            # Update population
            self.population = np.array(next_population[:self.pop_size])
        
        return self.best_individual, self.history
    
    def get_selected_features(self, feature_names: Optional[List[str]] = None) -> List:
        """
        Get selected features from best individual.
        
        Parameters
        ----------
        feature_names : list of str, optional
            Names of features. If provided, returns feature names.
            Otherwise, returns feature indices.
            
        Returns
        -------
        selected : list
            List of selected feature names or indices.
            
        Raises
        ------
        ValueError
            If no evolution has been run yet.
        """
        if self.best_individual is None:
            raise ValueError("No evolution has been run yet. Call evolve() first.")
        
        mask = self._weights_to_mask(self.best_individual)
        selected_indices = np.where(mask == 1)[0]
        
        if feature_names is not None:
            return [feature_names[i] for i in selected_indices]
        else:
            return selected_indices.tolist()