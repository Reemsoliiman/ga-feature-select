# GeneticAlgorithm Class API Reference

## Quick Start

```python
from src.genetic_algorithm import GeneticAlgorithm

# Minimal usage
ga = GeneticAlgorithm(dataset_path="data/processed/cleaned_diabetes_data.csv")
best_weights, best_fitness, history, selected_features = ga.evolve()

# Custom configuration
ga = GeneticAlgorithm(
    dataset_path="data/processed/cleaned_diabetes_data.csv",
    population_size=100,
    n_generations=200,
    mutation_rate=0.05,
    crossover_rate=0.9,
    selection_method='tournament',
    crossover_method='arithmetic',
    mutation_method='uniform',
    elitism_rate=0.15,  # 15% elites
    n_elites=None,      # Or specify exact number
    weight_threshold=0.5,
    random_state=42
)
```

## Class Structure

```
GeneticAlgorithm
├── Attributes
│   ├── X: np.ndarray                    # Feature matrix
│   ├── y: np.ndarray                    # Target vector
│   ├── feature_names: List[str]         # Feature names
│   ├── n_features: int                  # Number of features
│   ├── population: np.ndarray           # Current population
│   ├── fitness_scores: np.ndarray       # Current fitness
│   └── history: dict                    # Evolution history
│
├── Public Methods
│   └── evolve() -> Tuple
│       └── Returns: (best_weights, best_fitness, history, selected_features)
│
├── Internal Methods
│   ├── initialize_population() -> np.ndarray
│   ├── evaluate_population(population) -> np.ndarray
│   ├── _compute_elite_count() -> int
│   ├── _select_parents(n_parents) -> np.ndarray
│   ├── _crossover(parent1, parent2) -> Tuple[np.ndarray, np.ndarray]
│   └── _mutate(individual) -> np.ndarray
│
└── Configuration
    ├── population_size: int (default: 50)
    ├── n_generations: int (default: 100)
    ├── mutation_rate: float (default: 0.01)
    ├── crossover_rate: float (default: 0.8)
    ├── selection_method: str (default: 'tournament')
    ├── crossover_method: str (default: 'arithmetic')
    ├── mutation_method: str (default: 'uniform')
    ├── elitism_rate: float (default: 0.1)
    ├── n_elites: Optional[int] (default: None)
    ├── weight_threshold: float (default: 0.5)
    └── random_state: int (default: 42)
```

## Evolution Algorithm Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION                                           │
│    ↓                                                         │
│    initialize_population()                                  │
│    → Create random weights [0,1] for each feature          │
│    → Shape: (population_size, n_features)                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. MAIN LOOP (for each generation)                         │
└─────────────────────────────────────────────────────────────┘
    │
    ├─► evaluate_population(population)
    │   ├─ Convert weights to binary masks (>= 0.5)
    │   └─ Compute fitness with Decision Tree + CV
    │
    ├─► Track history
    │   ├─ Best fitness
    │   ├─ Mean fitness
    │   ├─ Best weights
    │   └─ Number of selected features
    │
    ├─► _compute_elite_count()
    │   ├─ Use n_elites if provided
    │   ├─ Otherwise: ceil(elitism_rate * population_size)
    │   └─ Clamp to [1, 0.3 * population_size]
    │
    ├─► ELITISM: Sort and copy top elites
    │   └─ sorted_population[:elite_count] → next_generation
    │
    └─► OFFSPRING GENERATION
        │
        ├─► while len(next_generation) < population_size:
        │   │
        │   ├─► _select_parents(2)
        │   │   └─ tournament / roulette / rank
        │   │
        │   ├─► _crossover(parent1, parent2)
        │   │   ├─ Apply with crossover_rate probability
        │   │   └─ arithmetic / single_point / uniform
        │   │
        │   ├─► _mutate(offspring1)
        │   │   └─ uniform / bit_flip / adaptive
        │   │
        │   └─► _mutate(offspring2)
        │
        └─► Update population = elites + offspring
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. RETURN RESULTS                                           │
│    ↓                                                         │
│    best_weights: Weight vector of best solution            │
│    best_fitness: Fitness score                             │
│    history: Per-generation metrics                         │
│    selected_features: Feature names (weight >= 0.5)        │
└─────────────────────────────────────────────────────────────┘
```

## Elitism Rules

```python
# Priority: n_elites overrides elitism_rate
if n_elites is not None:
    elite_count = n_elites
else:
    elite_count = ceil(elitism_rate * population_size)

# Constraints (all must be satisfied):
elite_count >= 1                                    # At least 1 elite
elite_count <= floor(0.3 * population_size)        # At most 30%
elite_count <= population_size - 1                 # Leave room for offspring
```

## Return Values

### best_weights
- Type: `np.ndarray`
- Shape: `(n_features,)`
- Description: Continuous weight vector for each feature
- Example: `[0.82, 0.13, 0.67, ...]`

### best_fitness
- Type: `float`
- Description: Fitness score of best solution
- Formula: `accuracy - lambda * (n_selected / n_total)`
- Example: `0.7543`

### history
- Type: `dict`
- Keys:
  - `'best_fitness'`: List of best fitness per generation
  - `'mean_fitness'`: List of mean fitness per generation
  - `'best_weights'`: List of best weight vectors per generation
  - `'n_selected_features'`: List of feature counts per generation

### selected_features
- Type: `List[str]`
- Description: Names of features with weight >= threshold
- Example: `['age', 'bmi', 'blood_pressure']`

## Example Usage Scenarios

### Scenario 1: Quick Test with Defaults
```python
ga = GeneticAlgorithm(dataset_path="data/processed/cleaned_diabetes_data.csv")
best_weights, best_fitness, history, selected = ga.evolve()
print(f"Selected {len(selected)} features with fitness {best_fitness:.4f}")
```

### Scenario 2: Experiment with Different Operators
```python
for selection in ['tournament', 'roulette', 'rank']:
    for crossover in ['arithmetic', 'single_point', 'uniform']:
        ga = GeneticAlgorithm(
            dataset_path="data/processed/cleaned_diabetes_data.csv",
            selection_method=selection,
            crossover_method=crossover,
            random_state=42
        )
        _, fitness, _, _ = ga.evolve()
        print(f"{selection} + {crossover}: {fitness:.4f}")
```

### Scenario 3: High Elitism for Exploitation
```python
ga = GeneticAlgorithm(
    dataset_path="data/processed/cleaned_diabetes_data.csv",
    population_size=100,
    n_elites=30,  # Keep top 30 individuals
    mutation_rate=0.001  # Low mutation for refinement
)
```

### Scenario 4: High Exploration
```python
ga = GeneticAlgorithm(
    dataset_path="data/processed/cleaned_diabetes_data.csv",
    elitism_rate=0.05,  # Only 5% elites
    mutation_rate=0.1,  # High mutation
    crossover_rate=0.95  # High crossover
)
```

## Operator Dispatch

### Selection Methods
- `'tournament'` → `operators.tournament_selection()` ✅ Implemented
- `'roulette'` → `operators.roulette_selection()` ⚠️ Placeholder
- `'rank'` → `operators.rank_selection()` ⚠️ Placeholder

### Crossover Methods
- `'single_point'` → `operators.single_point_crossover()` ⚠️ Placeholder
- `'uniform'` → `operators.uniform_crossover()` ⚠️ Placeholder
- `'arithmetic'` → `operators.arithmetic_crossover()` ✅ Implemented

### Mutation Methods
- `'bit_flip'` → `operators.bit_flip_mutation()` ⚠️ Placeholder
- `'uniform'` → `operators.uniform_mutation()` ✅ Implemented
- `'adaptive'` → `operators.adaptive_mutation()` ⚠️ Placeholder

## Error Handling

### Raised Exceptions
```python
# Invalid dataset path
FileNotFoundError: "Dataset not found at: <path>"

# Invalid CSV structure
ValueError: "Dataset must have at least 2 columns"

# Invalid operator names
ValueError: "Unknown selection method: <method>"
ValueError: "Unknown crossover method: <method>"
ValueError: "Unknown mutation method: <method>"

# Invalid elitism configuration
ValueError: "n_elites must be > 0 when provided"
ValueError: "elitism_rate must be > 0"
```
