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
REPORTS_DIR = os.path.join(RESULTS_DIR, 'reports')  

# Default dataset path 
DEFAULT_DATASET_PATH = os.path.join(DATA_DIR, 'diabetes_raw_data.csv')

# =============================================================================
# GENETIC ALGORITHM PARAMETERS
# =============================================================================
# Population and Evolution
POPULATION_SIZE = 75
N_GENERATIONS = 50
RANDOM_SEED = 42
RANDOM_STATE = 42

# Genetic Operators Rates
MUTATION_RATE = 0.01  
CROSSOVER_RATE = 0.8  
ELITISM_RATE = 0.1    

# Elitism Constraints
MAX_ELITISM_FRACTION = 0.3 

# Feature Selection Threshold
WEIGHT_THRESHOLD = 0.6  

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
TOURNAMENT_SIZE = 3  

# Adaptive Mutation Parameters 
ADAPTIVE_MUTATION_MIN = 0.001  
ADAPTIVE_MUTATION_MAX = 0.1    

# =============================================================================
# FITNESS EVALUATION PARAMETERS
# =============================================================================
# Cross-Validation Settings
CV_FOLDS = 5  

# Feature Penalty
LAMBDA_PENALTY = 0.01  

# Decision Tree Hyperparameters
DT_MAX_DEPTH = 5             
DT_MIN_SAMPLES_SPLIT = 10     
DT_MIN_SAMPLES_LEAF = 5       
DT_RANDOM_STATE = 42        

# =============================================================================
# EARLY STOPPING CONFIG
# =============================================================================
EARLY_STOPPING = True                  
EARLY_STOPPING_PATIENCE = 10           
EARLY_STOPPING_MIN_GENS = 10           
EARLY_STOPPING_DELTA = 0.0001          

# =============================================================================
# PARALLEL PROCESSING
# =============================================================================
USE_PARALLEL = True  
N_JOBS = -1           

# =============================================================================
# DATA PREPROCESSING PARAMETERS
# =============================================================================
# Missing Value Handling
HANDLE_MISSING = 'mean'  # Options: 'mean', 'median', 'drop', 'knn'

# Feature Scaling
NORMALIZE = True      
SCALING_METHOD = 'standard'  

# Train-Test Split
TEST_SIZE = 0.2       
STRATIFY = True      

# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================
# Number of independent runs for statistical significance
N_RUNS = 2  

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
SAVE_PLOTS = True         
SAVE_RESULTS = True       
SAVE_BEST_INDIVIDUAL = True  

# Plotting Options
PLOT_FORMAT = 'png'       
PLOT_DPI = 300           

# Logging
VERBOSE = True            
LOG_INTERVAL = 10         
SAVE_LOGS = True        

