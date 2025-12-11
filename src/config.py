"""
Configuration file for Genetic Algorithm Feature Selection.

This module centralizes all hyperparameters and settings for the GA,
fitness evaluation, data preprocessing, experiments, and output paths.

Project: Feature Selection Using Genetic Algorithms for Decision Trees

"""

import os

# =============================================================================
# PATHS CONFIGURATION
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
EXPERIMENTS_DIR = os.path.join(RESULTS_DIR, 'experiments')
BEST_FEATURES_DIR = os.path.join(RESULTS_DIR, 'best_features')
REPORTS_DIR = os.path.join(RESULTS_DIR, 'reports')  # NEW: For generated reports

# Default dataset path (processed diabetes data)
DEFAULT_DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, 'cleaned_diabetes_data.csv')

# =============================================================================
# GENETIC ALGORITHM PARAMETERS
# =============================================================================
# Population and Evolution
POPULATION_SIZE = 75
N_GENERATIONS = 50
RANDOM_SEED = 42
RANDOM_STATE = 42

# Genetic Operators Rates
MUTATION_RATE = 0.01  # Probability of mutating each gene
CROSSOVER_RATE = 0.8  # Probability of performing crossover
ELITISM_RATE = 0.1    # Fraction of top individuals to preserve (10%)

# Elitism Constraints
MAX_ELITISM_FRACTION = 0.3  # Maximum fraction of population for elites

# Feature Selection Threshold
WEIGHT_THRESHOLD = 0.6  # Weight >= threshold means feature selected
# Note: This is used in fitness_evaluator.py to convert continuous weights to binary

# =============================================================================
# OPERATOR CONFIGURATION
# =============================================================================
# Default operators for GA
DEFAULT_SELECTION = 'tournament'
DEFAULT_CROSSOVER = 'arithmetic'
DEFAULT_MUTATION = 'uniform'

# Available operator methods
SELECTION_METHODS = ['tournament', 'roulette', 'rank']
CROSSOVER_METHODS = ['single_point', 'uniform', 'arithmetic']
MUTATION_METHODS = ['bit_flip', 'uniform', 'adaptive']

# Tournament Selection Parameters
TOURNAMENT_SIZE = 3  # Number of individuals in each tournament

# Adaptive Mutation Parameters (if using adaptive mutation)
ADAPTIVE_MUTATION_MIN = 0.001  # Minimum mutation rate
ADAPTIVE_MUTATION_MAX = 0.1    # Maximum mutation rate

# =============================================================================
# FITNESS EVALUATION PARAMETERS
# =============================================================================
# Cross-Validation Settings
CV_FOLDS = 5  # Number of k-fold cross-validation folds

# Feature Penalty
LAMBDA_PENALTY = 0.01  # Penalty weight for feature count
# Formula: fitness = accuracy - lambda * (n_selected / n_total)
# Higher lambda = stronger penalty for using many features

# Decision Tree Hyperparameters
DT_MAX_DEPTH = 5              # Maximum depth of decision tree
DT_MIN_SAMPLES_SPLIT = 10     # Minimum samples required to split node
DT_MIN_SAMPLES_LEAF = 5       # Minimum samples required in leaf node
DT_RANDOM_STATE = 42          # Random state for reproducibility

# =============================================================================
# EARLY STOPPING CONFIG
# =============================================================================
EARLY_STOPPING = True                  # Enable/disable early stopping
EARLY_STOPPING_PATIENCE = 10           # Stop if no improvement for 10 generations
EARLY_STOPPING_MIN_GENS = 10           # Don't stop before generation 10
EARLY_STOPPING_DELTA = 0.0001          # Minimum improvement to count as "better"

# =============================================================================
# PARALLEL PROCESSING (NEW)
# =============================================================================
USE_PARALLEL = True   # Enable parallel fitness evaluation
N_JOBS = -1           # Number of parallel jobs (-1 = use all cores)

# =============================================================================
# DATA PREPROCESSING PARAMETERS
# =============================================================================
# Missing Value Handling
HANDLE_MISSING = 'mean'  # Options: 'mean', 'median', 'drop', 'knn'

# Feature Scaling
NORMALIZE = True      # Whether to normalize features (recommended: True)
SCALING_METHOD = 'standard'  # Options: 'standard', 'minmax', 'robust'

# Train-Test Split
TEST_SIZE = 0.2       # Fraction of data for testing
STRATIFY = True       # Whether to stratify split by target class

# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================
# Number of independent runs for statistical significance
N_RUNS = 2  # Recommended: 3-10 for robust results

# Experiment configurations to test (examples)
EXPERIMENT_CONFIGS = {
    'exp1_selection': {
        'name': 'Selection Methods Comparison',
        'vary_param': 'selection',
        'values': ['tournament', 'roulette', 'rank']
    },
    'exp2_crossover': {
        'name': 'Crossover Methods Comparison',
        'vary_param': 'crossover',
        'values': ['single_point', 'uniform', 'arithmetic']
    },
    'exp3_mutation': {
        'name': 'Mutation Rate Analysis',
        'vary_param': 'mutation_rate',
        'values': [0.001, 0.01, 0.05, 0.1]
    }
}

# =============================================================================
# OUTPUT AND LOGGING SETTINGS
# =============================================================================
# Save Options
SAVE_PLOTS = True         # Save evolution plots
SAVE_RESULTS = True       # Save numerical results to CSV/JSON
SAVE_BEST_INDIVIDUAL = True  # Save best individual's details

# Plotting Options
PLOT_FORMAT = 'png'       # Options: 'png', 'pdf', 'svg'
PLOT_DPI = 300            # Resolution for saved plots

# Logging
VERBOSE = True            # Print progress during evolution
LOG_INTERVAL = 10         # Print stats every N generations
SAVE_LOGS = True          # Save logs to file

