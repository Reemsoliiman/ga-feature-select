"""
Genetic Algorithm for feature selection.
Main GA class with population management and evolution loop.
"""

import numpy as np
from .operators import get_selection_fn, get_crossover_fn, get_mutation_fn
from .fitness import evaluate_population


class GeneticAlgorithm:
    """
    Genetic Algorithm for feature selection using binary chromosome encoding.
    
    Each individual represents a feature subset as a binary mask.
    The algorithm evolves the population to maximize classification accuracy
    while minimizing the number of selected features.
    
    Parameters
    ----------
    n_features : int
        Number of features in the dataset.
    population_size : int
        Number of individuals in the population.
    n_generations : int
        Number of generations to evolve.
    classifier_factory : callable
        Function that returns a new classifier instance (e.g., lambda: DecisionTreeClassifier()).
    selection_method : str, default='tournament'
        Selection method: 'tournament', 'roulette', or 'rank'.
    tournament_size : int, default=3
        Tournament size for tournament selection.
    crossover_method : str, default='uniform'
        Crossover method: 'single_point', 'two_point', or 'uniform'.
    crossover_prob : float, default=0.8
        Probability of applying crossover to a pair of parents.
    mutation_method : str, default='bit_flip'
        Mutation method: currently only 'bit_flip' is supported.
    mutation_prob : float, default=0.01
        Probability of flipping each bit during mutation.
    adaptive_mutation : bool, default=False
        Whether to use adaptive mutation (rate decreases over generations).
    elitism_rate : float, default=0.1
        Fraction of top individuals to preserve unchanged each generation.
    fitness_cfg : dict, optional
        Fitness configuration with keys: accuracy_weight, feature_reduction_weight, penalty_threshold.
    eval_cfg : dict, optional
        Evaluation configuration with keys: cv_folds, random_state, metrics.
    random_state : int, optional
        Random seed for reproducibility.
        
    Attributes
    ----------
    population : np.ndarray
        Current population matrix of shape (population_size, n_features).
    history : dict
        Evolution history containing best_fitness, avg_fitness, and best_individual per generation.
    """
    
    def __init__(
        self,
        n_features,
        population_size,
        n_generations,
        classifier_factory,
        selection_method='tournament',
        tournament_size=3,
        crossover_method='uniform',
        crossover_prob=0.8,
        mutation_method='bit_flip',
        mutation_prob=0.01,
        adaptive_mutation=False,
        elitism_rate=0.1,
        fitness_cfg=None,
        eval_cfg=None,
        random_state=None
    ):
        self.n_features = n_features
        self.population_size = population_size
        self.n_generations = n_generations
        self.classifier_factory = classifier_factory
        
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.crossover_method = crossover_method
        self.crossover_prob = crossover_prob
        self.mutation_method = mutation_method
        self.mutation_prob = mutation_prob
        self.adaptive_mutation = adaptive_mutation
        self.elitism_rate = elitism_rate
        
        # Default fitness config
        self.fitness_cfg = fitness_cfg or {
            'accuracy_weight': 0.7,
            'feature_reduction_weight': 0.3,
            'penalty_threshold': 0.95
        }
        
        # Default evaluation config
        self.eval_cfg = eval_cfg or {
            'cv_folds': 5,
            'random_state': 42,
            'metrics': ['accuracy', 'f1_score']
        }
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Set up operator functions
        self.selection_fn = get_selection_fn(selection_method, tournament_size)
        self.crossover_fn = get_crossover_fn(crossover_method)
        self.mutation_fn = get_mutation_fn(mutation_method, adaptive_mutation)
        
        self.population = None
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_individual': []
        }
    
    def _initialize_population(self):
        """
        Initialize population with random binary chromosomes.
        Ensures each individual has at least one feature selected.
        
        Returns
        -------
        population : np.ndarray
            Population matrix of shape (population_size, n_features).
        """
        # Random binary initialization (50% probability for each bit)
        population = np.random.randint(0, 2, size=(self.population_size, self.n_features))
        
        # Ensure each individual has at least one feature selected
        for i in range(self.population_size):
            if np.sum(population[i]) == 0:
                # Randomly flip one bit to 1
                random_idx = np.random.randint(0, self.n_features)
                population[i, random_idx] = 1
        
        return population
    
    def _apply_elitism(self, old_population, old_fitnesses, new_population):
        """
        Apply elitism: preserve top individuals from old population.
        
        Parameters
        ----------
        old_population : np.ndarray
            Previous generation population.
        old_fitnesses : np.ndarray
            Fitness values of old population.
        new_population : np.ndarray
            New offspring population.
            
        Returns
        -------
        combined_population : np.ndarray
            Population with elite individuals preserved.
        """
        n_elite = int(self.elitism_rate * self.population_size)
        
        if n_elite == 0:
            return new_population
        
        # Get indices of top individuals
        elite_indices = np.argsort(old_fitnesses)[-n_elite:]
        elite_individuals = old_population[elite_indices]
        
        # Replace worst new individuals with elite
        combined_population = new_population.copy()
        combined_population[:n_elite] = elite_individuals
        
        return combined_population
    
    def evolve(self, X, y):
        """
        Run the genetic algorithm to find optimal feature subset.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target labels of shape (n_samples,).
            
        Returns
        -------
        best_mask : np.ndarray
            Binary mask of selected features, shape (n_features,).
        history : dict
            Evolution history with keys: best_fitness, avg_fitness, best_individual.
        """
        # Initialize population
        self.population = self._initialize_population()
        
        # Evolution loop
        for generation in range(self.n_generations):
            # Evaluate fitness of all individuals
            fitnesses = evaluate_population(
                self.population, X, y, self.classifier_factory,
                self.fitness_cfg, self.eval_cfg
            )
            
            # Record statistics
            best_idx = np.argmax(fitnesses)
            self.history['best_fitness'].append(fitnesses[best_idx])
            self.history['avg_fitness'].append(np.mean(fitnesses))
            self.history['best_individual'].append(self.population[best_idx].copy())
            
            # Print progress if verbose
            if generation % 10 == 0 or generation == self.n_generations - 1:
                n_selected = np.sum(self.population[best_idx])
                print(f"Generation {generation + 1}/{self.n_generations}: "
                      f"Best Fitness = {fitnesses[best_idx]:.4f}, "
                      f"Avg Fitness = {np.mean(fitnesses):.4f}, "
                      f"Features = {n_selected}/{self.n_features}")
            
            # Stop if last generation
            if generation == self.n_generations - 1:
                break
            
            # Selection: select parents
            n_parents = self.population_size
            if n_parents % 2 == 1:
                n_parents += 1  # Ensure even number for pairing
            parents = self.selection_fn(self.population, fitnesses, n_parents)
            
            # Crossover and mutation: create new offspring
            offspring = []
            for i in range(0, len(parents), 2):
                parent1 = parents[i]
                parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
                
                # Apply crossover with probability
                if np.random.random() < self.crossover_prob:
                    child1, child2 = self.crossover_fn(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Apply mutation
                if self.adaptive_mutation:
                    child1 = self.mutation_fn(child1, self.mutation_prob, generation, self.n_generations)
                    child2 = self.mutation_fn(child2, self.mutation_prob, generation, self.n_generations)
                else:
                    child1 = self.mutation_fn(child1, self.mutation_prob)
                    child2 = self.mutation_fn(child2, self.mutation_prob)
                
                # Ensure at least one feature selected
                if np.sum(child1) == 0:
                    child1[np.random.randint(0, self.n_features)] = 1
                if np.sum(child2) == 0:
                    child2[np.random.randint(0, self.n_features)] = 1
                
                offspring.extend([child1, child2])
            
            # Convert to array and trim to population size
            offspring = np.array(offspring[:self.population_size])
            
            # Apply elitism
            self.population = self._apply_elitism(self.population, fitnesses, offspring)
        
        # Return best individual from final generation
        final_fitnesses = evaluate_population(
            self.population, X, y, self.classifier_factory,
            self.fitness_cfg, self.eval_cfg
        )
        best_idx = np.argmax(final_fitnesses)
        best_mask = self.population[best_idx]
        
        return best_mask, self.history
