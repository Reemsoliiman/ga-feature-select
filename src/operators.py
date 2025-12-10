"""
Genetic Algorithm Operators Module.

This module contains all selection, crossover, and mutation operators
used by the Genetic Algorithm for feature selection. All operators work
with continuous weight encoding where weights are in [0, 1].

Operators are implemented as pure functions for modularity and testability.

"""

import numpy as np
from typing import Tuple

from .config import TOURNAMENT_SIZE


# =============================================================================
# SELECTION OPERATORS
# =============================================================================

def tournament_selection(
    population: np.ndarray,
    fitness_scores: np.ndarray,
    n_parents: int,
    tournament_size: int = TOURNAMENT_SIZE,
    rng: np.random.RandomState = None
) -> np.ndarray:
    """
    Tournament selection: pick k random individuals, select the best.
    
    In each tournament, randomly select tournament_size individuals
    and choose the one with highest fitness.
    
    Parameters
    ----------
    population : np.ndarray
        Population array of shape (pop_size, n_features).
    fitness_scores : np.ndarray
        Fitness scores of shape (pop_size,).
    n_parents : int
        Number of parents to select.
    tournament_size : int, optional
        Number of individuals per tournament (default: from config).
    rng : np.random.RandomState, optional
        Random number generator for reproducibility.
        
    Returns
    -------
    parents : np.ndarray
        Selected parents of shape (n_parents, n_features).
        
    Examples
    --------
    >>> pop = np.random.rand(10, 5)
    >>> fitness = np.random.rand(10)
    >>> parents = tournament_selection(pop, fitness, n_parents=4)
    >>> parents.shape
    (4, 5)
    """
    if rng is None:
        rng = np.random.RandomState()
    
    pop_size = population.shape[0]
    parents = []
    
    for _ in range(n_parents):
        # Select random tournament participants
        indices = rng.choice(pop_size, tournament_size, replace=False)
        tournament_fitness = fitness_scores[indices]
        
        # Winner has highest fitness
        winner_idx = indices[np.argmax(tournament_fitness)]
        parents.append(population[winner_idx].copy())
    
    return np.array(parents)


def roulette_selection(
    population: np.ndarray,
    fitness_scores: np.ndarray,
    n_parents: int,
    rng: np.random.RandomState = None
) -> np.ndarray:
    """
    Roulette wheel selection: probability proportional to fitness.
    
    Each individual's selection probability is proportional to its fitness.
    Handles negative fitness by shifting all values to be positive.
    
    Parameters
    ----------
    population : np.ndarray
        Population array of shape (pop_size, n_features).
    fitness_scores : np.ndarray
        Fitness scores of shape (pop_size,).
    n_parents : int
        Number of parents to select.
    rng : np.random.RandomState, optional
        Random number generator for reproducibility.
        
    Returns
    -------
    parents : np.ndarray
        Selected parents of shape (n_parents, n_features).
        
    Notes
    -----
    If all fitness scores are equal, selection becomes uniform random.
    """
    if rng is None:
        rng = np.random.RandomState()
    
    # Handle negative fitness by shifting to positive
    min_fitness = np.min(fitness_scores)
    shifted_fitness = fitness_scores - min_fitness + 1e-6
    
    # Compute selection probabilities
    probabilities = shifted_fitness / np.sum(shifted_fitness)
    
    # Select parents
    pop_size = population.shape[0]
    indices = rng.choice(pop_size, size=n_parents, p=probabilities, replace=True)
    
    return population[indices].copy()


def rank_selection(
    population: np.ndarray,
    fitness_scores: np.ndarray,
    n_parents: int,
    rng: np.random.RandomState = None
) -> np.ndarray:
    """
    Rank selection: probability based on rank, not raw fitness.
    
    Individuals are ranked by fitness, and selection probability is
    proportional to rank. This reduces selection pressure compared
    to roulette selection and handles negative fitness naturally.
    
    Parameters
    ----------
    population : np.ndarray
        Population array of shape (pop_size, n_features).
    fitness_scores : np.ndarray
        Fitness scores of shape (pop_size,).
    n_parents : int
        Number of parents to select.
    rng : np.random.RandomState, optional
        Random number generator for reproducibility.
        
    Returns
    -------
    parents : np.ndarray
        Selected parents of shape (n_parents, n_features).
        
    Notes
    -----
    Rank 1 is assigned to the worst individual, rank N to the best.
    This provides more balanced selection than roulette wheel.
    """
    if rng is None:
        rng = np.random.RandomState()
    
    # Compute ranks (1 = worst, N = best)
    ranks = np.argsort(np.argsort(fitness_scores)) + 1
    
    # Compute selection probabilities
    probabilities = ranks / np.sum(ranks)
    
    # Select parents
    pop_size = population.shape[0]
    indices = rng.choice(pop_size, size=n_parents, p=probabilities, replace=True)
    
    return population[indices].copy()


# =============================================================================
# CROSSOVER OPERATORS
# =============================================================================

def single_point_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    rng: np.random.RandomState = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single-point crossover: cut at random point and swap segments.
    
    A random crossover point is chosen. The first child inherits
    genes before the point from parent1 and after from parent2.
    The second child does the opposite.
    
    Parameters
    ----------
    parent1, parent2 : np.ndarray
        Parent weight vectors of shape (n_features,).
    rng : np.random.RandomState, optional
        Random number generator for reproducibility.
        
    Returns
    -------
    child1, child2 : np.ndarray
        Two offspring weight vectors.
        
    Examples
    --------
    >>> p1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> p2 = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    >>> c1, c2 = single_point_crossover(p1, p2)
    # If crossover point is 2: c1 = [0.1, 0.2, 0.7, 0.6, 0.5]
    """
    if rng is None:
        rng = np.random.RandomState()
    
    n_features = len(parent1)
    point = rng.randint(1, n_features)
    
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    
    return child1, child2


def uniform_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    rng: np.random.RandomState = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniform crossover: each gene randomly inherited from either parent.
    
    For each gene position, randomly choose which parent contributes
    to each child. Provides more thorough mixing than single-point.
    
    Parameters
    ----------
    parent1, parent2 : np.ndarray
        Parent weight vectors of shape (n_features,).
    rng : np.random.RandomState, optional
        Random number generator for reproducibility.
        
    Returns
    -------
    child1, child2 : np.ndarray
        Two offspring weight vectors.
        
    Notes
    -----
    Each gene has 50% chance of coming from either parent,
    creating maximum genetic diversity.
    """
    if rng is None:
        rng = np.random.RandomState()
    
    n_features = len(parent1)
    mask = rng.randint(0, 2, n_features, dtype=bool)
    
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    
    return child1, child2


def arithmetic_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    alpha: float = 0.5,
    rng: np.random.RandomState = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Arithmetic crossover: weighted average of parents.
    
    Creates offspring as linear combinations of parent weight vectors.
    Particularly effective for continuous representations.
    
    Parameters
    ----------
    parent1, parent2 : np.ndarray
        Parent weight vectors of shape (n_features,).
    alpha : float, optional
        Blending coefficient in [0, 1] (default: 0.5).
        child1 = alpha * parent1 + (1-alpha) * parent2
    rng : np.random.RandomState, optional
        Random number generator (unused, for API consistency).
        
    Returns
    -------
    child1, child2 : np.ndarray
        Two offspring weight vectors, clipped to [0, 1].
        
    Notes
    -----
    With alpha=0.5, offspring are the average of parents.
    Results are clipped to maintain [0, 1] range for weights.
    """
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = (1 - alpha) * parent1 + alpha * parent2
    
    # Ensure weights stay in [0, 1]
    child1 = np.clip(child1, 0, 1)
    child2 = np.clip(child2, 0, 1)
    
    return child1, child2


# =============================================================================
# MUTATION OPERATORS
# =============================================================================

def bit_flip_mutation(
    individual: np.ndarray,
    mutation_rate: float,
    rng: np.random.RandomState = None
) -> np.ndarray:
    """
    Bit flip mutation: invert weights with probability mutation_rate.
    
    For continuous weights, inversion means w' = 1 - w.
    This flips the "selection state" of a feature.
    
    Parameters
    ----------
    individual : np.ndarray
        Weight vector of shape (n_features,).
    mutation_rate : float
        Probability of mutating each gene.
    rng : np.random.RandomState, optional
        Random number generator for reproducibility.
        
    Returns
    -------
    mutated : np.ndarray
        Mutated weight vector.
        
    Examples
    --------
    >>> weights = np.array([0.2, 0.8, 0.5])
    >>> mutated = bit_flip_mutation(weights, mutation_rate=0.5)
    # Flipped genes: 0.2 -> 0.8, 0.8 -> 0.2, 0.5 -> 0.5
    """
    if rng is None:
        rng = np.random.RandomState()
    
    mutated = individual.copy()
    n_features = len(individual)
    
    # Generate mutation mask
    mutation_mask = rng.rand(n_features) < mutation_rate
    
    # Flip weights: w' = 1 - w
    mutated[mutation_mask] = 1 - mutated[mutation_mask]
    
    return mutated


def uniform_mutation(
    individual: np.ndarray,
    mutation_rate: float,
    rng: np.random.RandomState = None
) -> np.ndarray:
    """
    Uniform mutation: replace weights with random values.
    
    Each gene is mutated with probability mutation_rate.
    Mutated genes are replaced with random values from [0, 1].
    
    Parameters
    ----------
    individual : np.ndarray
        Weight vector of shape (n_features,).
    mutation_rate : float
        Probability of mutating each gene.
    rng : np.random.RandomState, optional
        Random number generator for reproducibility.
        
    Returns
    -------
    mutated : np.ndarray
        Mutated weight vector.
        
    Notes
    -----
    Introduces more diversity than bit_flip since new values
    are completely random rather than deterministic inversions.
    """
    if rng is None:
        rng = np.random.RandomState()
    
    mutated = individual.copy()
    n_features = len(individual)
    
    # Generate mutation mask
    mutation_mask = rng.rand(n_features) < mutation_rate
    n_mutations = np.sum(mutation_mask)
    
    # Replace with random weights
    if n_mutations > 0:
        mutated[mutation_mask] = rng.uniform(0, 1, size=n_mutations)
    
    return mutated


def adaptive_mutation(
    individual: np.ndarray,
    mutation_rate: float,
    generation: int = 0,
    max_generations: int = 100,
    rng: np.random.RandomState = None
) -> np.ndarray:
    """
    Adaptive mutation: mutation rate decreases over generations.
    
    Starts with high mutation for exploration, decreases over time
    for exploitation. Mutation rate adapts as:
    adaptive_rate = mutation_rate * (1 - generation / max_generations)
    
    Parameters
    ----------
    individual : np.ndarray
        Weight vector of shape (n_features,).
    mutation_rate : float
        Base mutation rate at generation 0.
    generation : int, optional
        Current generation number (default: 0).
    max_generations : int, optional
        Total number of generations (default: 100).
    rng : np.random.RandomState, optional
        Random number generator for reproducibility.
        
    Returns
    -------
    mutated : np.ndarray
        Mutated weight vector.
        
    Notes
    -----
    At generation 0: uses full mutation_rate
    At final generation: mutation_rate approaches 0
    This balances exploration (early) and exploitation (late).
    """
    if rng is None:
        rng = np.random.RandomState()
    
    # Calculate adaptive rate
    if max_generations > 0:
        adaptive_rate = mutation_rate * (1 - generation / max_generations)
    else:
        adaptive_rate = mutation_rate
    
    adaptive_rate = max(0, adaptive_rate)  # Ensure non-negative
    
    # Apply bit flip with adaptive rate
    mutated = individual.copy()
    n_features = len(individual)
    
    mutation_mask = rng.rand(n_features) < adaptive_rate
    mutated[mutation_mask] = 1 - mutated[mutation_mask]
    
    return mutated