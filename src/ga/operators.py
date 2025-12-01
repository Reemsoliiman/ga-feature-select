"""
Genetic Algorithm operators for feature selection.
Includes selection, crossover, mutation, and dispatcher functions.
"""

import numpy as np


# ==================== SELECTION OPERATORS ====================

def tournament_selection(population, fitnesses, tournament_size, n_parents):
    """
    Tournament selection: randomly sample tournament_size individuals,
    select the one with best fitness. Repeat n_parents times.
    
    Parameters
    ----------
    population : np.ndarray
        Population matrix of shape (pop_size, n_features).
    fitnesses : np.ndarray
        Fitness values of shape (pop_size,).
    tournament_size : int
        Number of individuals in each tournament.
    n_parents : int
        Number of parents to select.
        
    Returns
    -------
    parents : np.ndarray
        Selected parents of shape (n_parents, n_features).
    """
    pop_size = population.shape[0]
    parents = []
    
    for _ in range(n_parents):
        # Randomly sample tournament_size indices
        tournament_indices = np.random.choice(pop_size, size=tournament_size, replace=False)
        # Find the index with best fitness in tournament
        tournament_fitnesses = fitnesses[tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        parents.append(population[winner_idx].copy())
    
    return np.array(parents)


def roulette_wheel_selection(population, fitnesses, n_parents):
    """
    Roulette wheel (fitness-proportionate) selection.
    
    Parameters
    ----------
    population : np.ndarray
        Population matrix of shape (pop_size, n_features).
    fitnesses : np.ndarray
        Fitness values of shape (pop_size,).
    n_parents : int
        Number of parents to select.
        
    Returns
    -------
    parents : np.ndarray
        Selected parents of shape (n_parents, n_features).
    """
    # Shift fitnesses to be positive if needed
    min_fitness = np.min(fitnesses)
    if min_fitness < 0:
        shifted_fitnesses = fitnesses - min_fitness + 1e-10
    else:
        shifted_fitnesses = fitnesses + 1e-10
    
    # Normalize to probabilities
    probabilities = shifted_fitnesses / np.sum(shifted_fitnesses)
    
    # Sample parent indices
    parent_indices = np.random.choice(
        len(population), size=n_parents, replace=True, p=probabilities
    )
    
    return population[parent_indices].copy()


def rank_selection(population, fitnesses, n_parents):
    """
    Rank-based selection: assign probabilities based on fitness rank.
    
    Parameters
    ----------
    population : np.ndarray
        Population matrix of shape (pop_size, n_features).
    fitnesses : np.ndarray
        Fitness values of shape (pop_size,).
    n_parents : int
        Number of parents to select.
        
    Returns
    -------
    parents : np.ndarray
        Selected parents of shape (n_parents, n_features).
    """
    # Rank individuals (lowest fitness gets rank 1, highest gets rank pop_size)
    ranks = np.argsort(np.argsort(fitnesses)) + 1
    
    # Probability proportional to rank
    probabilities = ranks / np.sum(ranks)
    
    # Sample parent indices
    parent_indices = np.random.choice(
        len(population), size=n_parents, replace=True, p=probabilities
    )
    
    return population[parent_indices].copy()


# ==================== CROSSOVER OPERATORS ====================

def single_point_crossover(parent1, parent2):
    """
    Single-point crossover: choose one cut point, swap segments.
    
    Parameters
    ----------
    parent1 : np.ndarray
        First parent chromosome.
    parent2 : np.ndarray
        Second parent chromosome.
        
    Returns
    -------
    child1, child2 : tuple of np.ndarray
        Two offspring chromosomes.
    """
    n_features = len(parent1)
    if n_features <= 1:
        return parent1.copy(), parent2.copy()
    
    # Choose random cut point in [1, n_features-1]
    cut_point = np.random.randint(1, n_features)
    
    child1 = np.concatenate([parent1[:cut_point], parent2[cut_point:]])
    child2 = np.concatenate([parent2[:cut_point], parent1[cut_point:]])
    
    return child1, child2


def two_point_crossover(parent1, parent2):
    """
    Two-point crossover: choose two cut points, swap middle segment.
    
    Parameters
    ----------
    parent1 : np.ndarray
        First parent chromosome.
    parent2 : np.ndarray
        Second parent chromosome.
        
    Returns
    -------
    child1, child2 : tuple of np.ndarray
        Two offspring chromosomes.
    """
    n_features = len(parent1)
    if n_features <= 2:
        return single_point_crossover(parent1, parent2)
    
    # Choose two random cut points
    cut_points = sorted(np.random.choice(range(1, n_features), size=2, replace=False))
    cut1, cut2 = cut_points
    
    child1 = np.concatenate([parent1[:cut1], parent2[cut1:cut2], parent1[cut2:]])
    child2 = np.concatenate([parent2[:cut1], parent1[cut1:cut2], parent2[cut2:]])
    
    return child1, child2


def uniform_crossover(parent1, parent2, p_swap=0.5):
    """
    Uniform crossover: for each gene, randomly choose which parent to inherit from.
    
    Parameters
    ----------
    parent1 : np.ndarray
        First parent chromosome.
    parent2 : np.ndarray
        Second parent chromosome.
    p_swap : float, default=0.5
        Probability of swapping each gene.
        
    Returns
    -------
    child1, child2 : tuple of np.ndarray
        Two offspring chromosomes.
    """
    n_features = len(parent1)
    
    # Random mask: True means swap that gene
    swap_mask = np.random.random(n_features) < p_swap
    
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    child1[swap_mask] = parent2[swap_mask]
    child2[swap_mask] = parent1[swap_mask]
    
    return child1, child2


# ==================== MUTATION OPERATORS ====================

def bit_flip_mutation(individual, mutation_prob):
    """
    Bit-flip mutation: flip each bit with probability mutation_prob.
    
    Parameters
    ----------
    individual : np.ndarray
        Binary chromosome to mutate.
    mutation_prob : float
        Probability of flipping each bit.
        
    Returns
    -------
    mutated : np.ndarray
        Mutated chromosome.
    """
    mutated = individual.copy()
    mutation_mask = np.random.random(len(individual)) < mutation_prob
    mutated[mutation_mask] = 1 - mutated[mutation_mask]
    return mutated


def adaptive_bit_flip_mutation(individual, base_prob, gen_idx, n_gens):
    """
    Adaptive bit-flip mutation: mutation rate decreases with generation.
    
    Parameters
    ----------
    individual : np.ndarray
        Binary chromosome to mutate.
    base_prob : float
        Base mutation probability.
    gen_idx : int
        Current generation index (0-based).
    n_gens : int
        Total number of generations.
        
    Returns
    -------
    mutated : np.ndarray
        Mutated chromosome.
    """
    # Decrease mutation rate linearly: high at start, low at end
    progress = gen_idx / max(n_gens - 1, 1)
    adaptive_prob = base_prob * (1 - 0.8 * progress)  # Decays to 20% of base
    
    return bit_flip_mutation(individual, adaptive_prob)


# ==================== DISPATCHER FUNCTIONS ====================

def get_selection_fn(method, tournament_size=3):
    """
    Get selection function by name.
    
    Parameters
    ----------
    method : str
        Selection method name: 'tournament', 'roulette', or 'rank'.
    tournament_size : int, default=3
        Tournament size (only used for tournament selection).
        
    Returns
    -------
    selection_fn : callable
        Selection function with signature (population, fitnesses, n_parents).
    """
    if method == "tournament":
        return lambda pop, fit, n: tournament_selection(pop, fit, tournament_size, n)
    elif method == "roulette":
        return roulette_wheel_selection
    elif method == "rank":
        return rank_selection
    else:
        raise ValueError(f"Unknown selection method: {method}")


def get_crossover_fn(method):
    """
    Get crossover function by name.
    
    Parameters
    ----------
    method : str
        Crossover method name: 'single_point', 'two_point', or 'uniform'.
        
    Returns
    -------
    crossover_fn : callable
        Crossover function with signature (parent1, parent2) -> (child1, child2).
    """
    if method == "single_point":
        return single_point_crossover
    elif method == "two_point":
        return two_point_crossover
    elif method == "uniform":
        return uniform_crossover
    else:
        raise ValueError(f"Unknown crossover method: {method}")


def get_mutation_fn(method, adaptive=False):
    """
    Get mutation function by name.
    
    Parameters
    ----------
    method : str
        Mutation method name: currently only 'bit_flip' is supported.
    adaptive : bool, default=False
        Whether to use adaptive mutation rate.
        
    Returns
    -------
    mutation_fn : callable
        If adaptive=False: signature (individual, mutation_prob).
        If adaptive=True: signature (individual, base_prob, gen_idx, n_gens).
    """
    if method == "bit_flip":
        if adaptive:
            return adaptive_bit_flip_mutation
        else:
            return bit_flip_mutation
    else:
        raise ValueError(f"Unknown mutation method: {method}")
