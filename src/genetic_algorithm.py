"""
Genetic Algorithm for Feature Selection using continuous feature weights.

Each individual represents a weight vector where each weight is in [0, 1].
Features with weight >= 0.5 are considered selected for fitness evaluation.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
import math

from . import config
from .utils import load_data, weights_to_mask, get_selected_features
from .fitness import evaluate_fitness
from . import operators


class GeneticAlgorithm:
    """
    Genetic Algorithm for feature selection with continuous weight encoding.
    
    Individuals are represented as continuous weight vectors in [0, 1].
    For fitness evaluation, weights are converted to binary masks using
    a threshold (default: 0.5).
    
    Parameters
    ----------
    dataset_path : str
        Path to the CSV dataset file. Last column is the target.
    population_size : int, optional
        Number of individuals in the population (default: from config).
    n_generations : int, optional
        Number of generations to evolve (default: from config).
    mutation_rate : float, optional
        Probability of mutating each weight (default: from config).
    crossover_rate : float, optional
        Probability of applying crossover (default: from config).
    selection_method : str, optional
        Selection method: 'tournament', 'roulette', 'rank' (default: from config).
    crossover_method : str, optional
        Crossover method: 'single_point', 'uniform', 'arithmetic' (default: from config).
    mutation_method : str, optional
        Mutation method: 'bit_flip', 'uniform', 'adaptive' (default: from config).
    elitism_rate : float, optional
        Fraction of population to preserve as elites (default: 0.1).
        Must be > 0. Overridden by n_elites if provided.
    n_elites : int, optional
        Absolute number of elites to preserve (default: None).
        If provided, overrides elitism_rate. Must be > 0.
    weight_threshold : float, optional
        Threshold for converting weights to binary mask (default: 0.5).
    random_state : int, optional
        Random seed for reproducibility (default: from config).
        
    Attributes
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    feature_names : list of str
        Names of features.
    n_features : int
        Number of features.
    population : np.ndarray
        Current population of shape (population_size, n_features).
    fitness_scores : np.ndarray
        Fitness scores for current population.
    history : dict
        Evolution history tracking best/mean fitness per generation.
    """
    
    def __init__(
        self,
        dataset_path: str,
        population_size: int = config.POPULATION_SIZE,
        n_generations: int = config.N_GENERATIONS,
        mutation_rate: float = config.MUTATION_RATE,
        crossover_rate: float = config.CROSSOVER_RATE,
        selection_method: str = config.DEFAULT_SELECTION,
        crossover_method: str = config.DEFAULT_CROSSOVER,
        mutation_method: str = config.DEFAULT_MUTATION,
        elitism_rate: float = config.ELITISM_RATE,
        n_elites: Optional[int] = None,
        weight_threshold: float = config.WEIGHT_THRESHOLD,
        random_state: int = config.RANDOM_SEED
    ):
        """Initialize the Genetic Algorithm."""
        # Load data
        self.X, self.y, self.feature_names = load_data(dataset_path)
        self.n_features = self.X.shape[1]
        
        # GA hyperparameters
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.weight_threshold = weight_threshold
        
        # Operator methods
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        
        # Validate operator methods
        if selection_method not in config.SELECTION_METHODS:
            raise ValueError(f"Unknown selection method: {selection_method}. "
                           f"Choose from {config.SELECTION_METHODS}")
        if crossover_method not in config.CROSSOVER_METHODS:
            raise ValueError(f"Unknown crossover method: {crossover_method}. "
                           f"Choose from {config.CROSSOVER_METHODS}")
        if mutation_method not in config.MUTATION_METHODS:
            raise ValueError(f"Unknown mutation method: {mutation_method}. "
                           f"Choose from {config.MUTATION_METHODS}")
        
        # Elitism configuration
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
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        # Population and fitness (initialized later)
        self.population = None
        self.fitness_scores = None
        
        # History tracking
        self.history = {
            'best_fitness': [],
            'mean_fitness': [],
            'best_weights': [],
            'n_selected_features': []
        }
        
    def _compute_elite_count(self) -> int:
        """
        Compute the number of elites for the current generation.
        
        Returns
        -------
        elite_count : int
            Number of elites, clamped to [1, floor(0.3 * population_size)]
            and never exceeding population_size - 1.
        """
        if self.n_elites is not None:
            # n_elites overrides elitism_rate
            elite_count = self.n_elites
        else:
            # Compute from elitism_rate
            elite_count = math.ceil(self.elitism_rate * self.population_size)
        
        # Clamp to [1, max_elites]
        max_elites = math.floor(config.MAX_ELITISM_FRACTION * self.population_size)
        max_elites = max(1, max_elites)  # Ensure at least 1 elite is possible
        
        elite_count = max(1, elite_count)  # At least 1 elite
        elite_count = min(elite_count, max_elites)  # No more than 30% of population
        elite_count = min(elite_count, self.population_size - 1)  # Leave room for offspring
        
        return elite_count
    
    def initialize_population(self) -> np.ndarray:
        """
        Initialize population with random feature weights in [0, 1].
        
        Each individual is a vector of continuous weights, one per feature.
        Weights are uniformly sampled from [0, 1].
        
        Returns
        -------
        population : np.ndarray
            Population array of shape (population_size, n_features).
            Each element is a weight in [0, 1].
        """
        # Generate random weights uniformly in [0, 1]
        population = self.rng.uniform(
            low=0.0,
            high=1.0,
            size=(self.population_size, self.n_features)
        )
        
        return population
    
    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """
        Evaluate fitness for all individuals in the population.
        
        Converts each individual's weight vector to a binary mask using
        the threshold, then evaluates fitness using the decision tree
        classifier with cross-validation.
        
        Parameters
        ----------
        population : np.ndarray
            Population array of shape (population_size, n_features).
            
        Returns
        -------
        fitness_scores : np.ndarray
            Array of fitness scores of shape (population_size,).
        """
        fitness_scores = np.zeros(population.shape[0])
        
        for i, weights in enumerate(population):
            # Convert weights to binary mask
            mask = weights_to_mask(weights, self.weight_threshold)
            
            # Evaluate fitness
            fitness_scores[i] = evaluate_fitness(mask, self.X, self.y)
        
        return fitness_scores
    
    def _select_parents(self, n_parents: int) -> np.ndarray:
        """
        Select parents from current population using configured selection method.
        
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
                self.population, self.fitness_scores, n_parents, rng=self.rng
            )
        elif self.selection_method == 'roulette':
            return operators.roulette_selection(
                self.population, self.fitness_scores, n_parents, rng=self.rng
            )
        elif self.selection_method == 'rank':
            return operators.rank_selection(
                self.population, self.fitness_scores, n_parents, rng=self.rng
            )
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply crossover to two parents using configured crossover method.
        
        Parameters
        ----------
        parent1, parent2 : np.ndarray
            Parent weight vectors.
            
        Returns
        -------
        offspring1, offspring2 : np.ndarray
            Two offspring weight vectors.
        """
        # Apply crossover with probability crossover_rate
        if self.rng.rand() < self.crossover_rate:
            if self.crossover_method == 'single_point':
                return operators.single_point_crossover(parent1, parent2, rng=self.rng)
            elif self.crossover_method == 'uniform':
                return operators.uniform_crossover(parent1, parent2, rng=self.rng)
            elif self.crossover_method == 'arithmetic':
                return operators.arithmetic_crossover(parent1, parent2, rng=self.rng)
            else:
                raise ValueError(f"Unknown crossover method: {self.crossover_method}")
        else:
            # No crossover: return copies of parents
            return parent1.copy(), parent2.copy()
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Apply mutation to an individual using configured mutation method.
        
        Parameters
        ----------
        individual : np.ndarray
            Weight vector to mutate.
            
        Returns
        -------
        mutated : np.ndarray
            Mutated weight vector.
        """
        if self.mutation_method == 'bit_flip':
            return operators.bit_flip_mutation(individual, self.mutation_rate, rng=self.rng)
        elif self.mutation_method == 'uniform':
            return operators.uniform_mutation(individual, self.mutation_rate, rng=self.rng)
        elif self.mutation_method == 'adaptive':
            return operators.adaptive_mutation(individual, self.mutation_rate, rng=self.rng)
        else:
            raise ValueError(f"Unknown mutation method: {self.mutation_method}")
    
    def evolve(self) -> Tuple[np.ndarray, float, Dict, List[str]]:
        """
        Run the genetic algorithm evolution loop.
        
        The algorithm:
        1. Initialize population with random weights
        2. For each generation:
           a. Evaluate fitness for all individuals
           b. Compute number of elites
           c. Sort population by fitness
           d. Copy elites to next generation
           e. Generate offspring via selection, crossover, mutation
           f. Update population
           g. Track history
        3. Return best solution and metrics
        
        Returns
        -------
        best_weights : np.ndarray
            Best weight vector found.
        best_fitness : float
            Fitness of best solution.
        history : dict
            Evolution history with keys:
            - 'best_fitness': list of best fitness per generation
            - 'mean_fitness': list of mean fitness per generation
            - 'best_weights': list of best weight vectors per generation
            - 'n_selected_features': list of feature counts per generation
        selected_features : list of str
            Names of selected features (weight >= threshold) in best solution.
        """
        # Initialize population
        self.population = self.initialize_population()
        
        # Evolution loop
        for generation in range(self.n_generations):
            # Evaluate fitness
            self.fitness_scores = self.evaluate_population(self.population)
            
            # Track history
            best_idx = np.argmax(self.fitness_scores)
            best_weights = self.population[best_idx].copy()
            best_fitness = self.fitness_scores[best_idx]
            mean_fitness = np.mean(self.fitness_scores)
            
            best_mask = weights_to_mask(best_weights, self.weight_threshold)
            n_selected = np.sum(best_mask)
            
            self.history['best_fitness'].append(best_fitness)
            self.history['mean_fitness'].append(mean_fitness)
            self.history['best_weights'].append(best_weights.copy())
            self.history['n_selected_features'].append(n_selected)
            
            if config.VERBOSE:
                print(f"Generation {generation + 1}/{self.n_generations} | "
                      f"Best Fitness: {best_fitness:.4f} | "
                      f"Mean Fitness: {mean_fitness:.4f} | "
                      f"Features: {n_selected}/{self.n_features}")
            
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
            
            # Step 2: Generate offspring to fill remaining slots
            n_offspring_needed = self.population_size - elite_count
            
            while len(next_population) < self.population_size:
                # Select two parents
                parents = self._select_parents(n_parents=2)
                parent1, parent2 = parents[0], parents[1]
                
                # Crossover
                offspring1, offspring2 = self._crossover(parent1, parent2)
                
                # Mutation
                offspring1 = self._mutate(offspring1)
                offspring2 = self._mutate(offspring2)
                
                # Add offspring to next generation
                next_population.append(offspring1)
                if len(next_population) < self.population_size:
                    next_population.append(offspring2)
            
            # Update population
            self.population = np.array(next_population[:self.population_size])
        
        # Final evaluation
        self.fitness_scores = self.evaluate_population(self.population)
        best_idx = np.argmax(self.fitness_scores)
        best_weights = self.population[best_idx]
        best_fitness = self.fitness_scores[best_idx]
        
        # Get selected features
        selected_features = get_selected_features(
            best_weights, self.feature_names, self.weight_threshold
        )
        
        return best_weights, best_fitness, self.history, selected_features
