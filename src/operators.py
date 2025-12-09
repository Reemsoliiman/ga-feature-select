"""
Genetic Algorithm operators for selection, crossover, and mutation.

These operators work with continuous weight vectors in [0, 1].
"""

import numpy as np
from typing import Tuple


# =============================================================================
# SELECTION OPERATORS
# =============================================================================

def tournament_selection(
    population: np.ndarray,
    fitness_scores: np.ndarray,
    n_parents: int,
    tournament_size: int = 3,
    rng: np.random.RandomState = None
) -> np.ndarray:
    """
    Tournament selection: randomly sample individuals and select the best.
    
    Parameters
    ----------
    population : np.ndarray
        Population array of shape (population_size, n_features).
    fitness_scores : np.ndarray
        Fitness scores of shape (population_size,).
    n_parents : int
        Number of parents to select.
    tournament_size : int, optional
        Number of individuals in each tournament (default: 3).
    rng : np.random.RandomState, optional
        Random number generator.
        
    Returns
    -------
    parents : np.ndarray
        Selected parents of shape (n_parents, n_features).
    """
    if rng is None:
        rng = np.random.RandomState()
    
    parents = []
    for _ in range(n_parents):
        # Randomly select tournament_size individuals
        tournament_indices = rng.choice(len(population), size=tournament_size, replace=False)
        # Select the best from tournament
        best_idx = tournament_indices[np.argmax(fitness_scores[tournament_indices])]
        parents.append(population[best_idx])
    
    return np.array(parents)


def roulette_selection(
    population: np.ndarray,
    fitness_scores: np.ndarray,
    n_parents: int,
    rng: np.random.RandomState = None
) -> np.ndarray:
    """
    Roulette wheel selection (fitness-proportionate selection).
    
    TODO: Implement roulette wheel selection.
    For now, falls back to tournament selection.
    """
    # Placeholder: use tournament selection
    return tournament_selection(population, fitness_scores, n_parents, rng=rng)


def rank_selection(
    population: np.ndarray,
    fitness_scores: np.ndarray,
    n_parents: int,
    rng: np.random.RandomState = None
) -> np.ndarray:
    """
    Rank-based selection.
    
    TODO: Implement rank-based selection.
    For now, falls back to tournament selection.
    """
    # Placeholder: use tournament selection
    return tournament_selection(population, fitness_scores, n_parents, rng=rng)


# =============================================================================
# CROSSOVER OPERATORS
# =============================================================================

def single_point_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    rng: np.random.RandomState = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single-point crossover for continuous weight vectors.
    
    TODO: Implement single-point crossover.
    For now, returns parents unchanged.
    """
    # Placeholder: return parents unchanged
    return parent1.copy(), parent2.copy()


def uniform_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    rng: np.random.RandomState = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniform crossover for continuous weight vectors.
    
    TODO: Implement uniform crossover.
    For now, returns parents unchanged.
    """
    # Placeholder: return parents unchanged
    return parent1.copy(), parent2.copy()


def arithmetic_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    alpha: float = 0.5,
    rng: np.random.RandomState = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Arithmetic crossover: linear combination of parent weights.
    
    offspring1 = alpha * parent1 + (1 - alpha) * parent2
    offspring2 = alpha * parent2 + (1 - alpha) * parent1
    
    Parameters
    ----------
    parent1, parent2 : np.ndarray
        Parent weight vectors of shape (n_features,).
    alpha : float, optional
        Blend coefficient (default: 0.5 for equal blend).
    rng : np.random.RandomState, optional
        Random number generator (not used, for API consistency).
        
    Returns
    -------
    offspring1, offspring2 : np.ndarray
        Two offspring weight vectors.
    """
    offspring1 = alpha * parent1 + (1 - alpha) * parent2
    offspring2 = alpha * parent2 + (1 - alpha) * parent1
    
    # Ensure weights stay in [0, 1] (should be automatic with alpha in [0,1])
    offspring1 = np.clip(offspring1, 0.0, 1.0)
    offspring2 = np.clip(offspring2, 0.0, 1.0)
    
    return offspring1, offspring2


# =============================================================================
# MUTATION OPERATORS
# =============================================================================

def bit_flip_mutation(
    individual: np.ndarray,
    mutation_rate: float,
    rng: np.random.RandomState = None
) -> np.ndarray:
    """
    Bit-flip mutation for continuous weights (adapted).
    
    TODO: Implement proper bit-flip or alternative for continuous weights.
    For now, falls back to uniform mutation.
    """
    return uniform_mutation(individual, mutation_rate, rng)


def uniform_mutation(
    individual: np.ndarray,
    mutation_rate: float,
    rng: np.random.RandomState = None
) -> np.ndarray:
    """
    Uniform mutation: replace weight with random value in [0, 1].
    
    Parameters
    ----------
    individual : np.ndarray
        Weight vector of shape (n_features,).
    mutation_rate : float
        Probability of mutating each weight.
    rng : np.random.RandomState, optional
        Random number generator.
        
    Returns
    -------
    mutated : np.ndarray
        Mutated weight vector.
    """
    if rng is None:
        rng = np.random.RandomState()
    
    mutated = individual.copy()
    mutation_mask = rng.rand(len(individual)) < mutation_rate
    mutated[mutation_mask] = rng.uniform(0.0, 1.0, size=np.sum(mutation_mask))
    
    return mutated


def adaptive_mutation(
    individual: np.ndarray,
    mutation_rate: float,
    rng: np.random.RandomState = None
) -> np.ndarray:
    """
    Adaptive mutation for continuous weights.
    
    TODO: Implement adaptive mutation strategy.
    For now, falls back to uniform mutation.
    """
    return uniform_mutation(individual, mutation_rate, rng)
