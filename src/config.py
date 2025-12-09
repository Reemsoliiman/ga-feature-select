"""
Configuration file for Genetic Algorithm Feature Selection.

This module centralizes all hyperparameters and settings for the GA,
fitness evaluation, data preprocessing, experiments, and output paths.
"""

import os

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
EXPERIMENTS_DIR = os.path.join(RESULTS_DIR, 'experiments')
BEST_FEATURES_DIR = os.path.join(RESULTS_DIR, 'best_features')

# Default dataset path (processed diabetes data)
DEFAULT_DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, 'cleaned_diabetes_data.csv')

# =============================================================================
# GENETIC ALGORITHM PARAMETERS
# =============================================================================
POPULATION_SIZE = 50
N_GENERATIONS = 100
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.8
ELITISM_RATE = 0.1  # Default: preserve top 10% of population
RANDOM_SEED = 42

# Maximum elitism fraction (elites won't exceed this fraction of population)
MAX_ELITISM_FRACTION = 0.3

# Weight threshold for feature selection (weight >= threshold means selected)
WEIGHT_THRESHOLD = 0.5

# =============================================================================
# OPERATOR DEFAULTS
# =============================================================================
DEFAULT_SELECTION = 'tournament'
DEFAULT_CROSSOVER = 'arithmetic'
DEFAULT_MUTATION = 'uniform'

# Available operators
SELECTION_METHODS = ['tournament', 'roulette', 'rank']
CROSSOVER_METHODS = ['single_point', 'uniform', 'arithmetic']
MUTATION_METHODS = ['bit_flip', 'uniform', 'adaptive']

# =============================================================================
# FITNESS EVALUATION PARAMETERS
# =============================================================================
CV_FOLDS = 5  # Number of cross-validation folds
LAMBDA_PENALTY = 0.01  # Penalty weight for feature count (higher = more penalty)

# Decision tree parameters
DT_MAX_DEPTH = 5
DT_MIN_SAMPLES_SPLIT = 10
DT_MIN_SAMPLES_LEAF = 5
DT_RANDOM_STATE = 42

# =============================================================================
# DATA PREPROCESSING
# =============================================================================
HANDLE_MISSING = 'mean'  # Options: 'mean', 'median', 'drop'
NORMALIZE = True  # Whether to normalize features

# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================
N_RUNS = 3  # Number of runs per operator configuration

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================
SAVE_PLOTS = True
SAVE_RESULTS = True
VERBOSE = True  # Print progress during evolution
