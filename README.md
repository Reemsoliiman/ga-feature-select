# Feature Selection using Genetic Algorithms

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/) [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![Code Style](https://img.shields.io/badge/Code%20Style-PEP8-black.svg)](https://www.python.org/dev/peps/pep-0008/)

This project implements a **Genetic Algorithm (GA) for feature selection** in machine learning datasets, with a focus on medical diagnosis applications (e.g., diabetes prediction). The GA optimizes feature subsets to maximize classification accuracy using a decision tree classifier while applying a penalty for using too many features, promoting model interpretability and efficiency.

The core algorithm evolves a population of binary masks (where 1 indicates a selected feature) through selection, crossover, mutation, and elitism. Fitness is evaluated via cross-validation accuracy minus a feature count penalty.

## Key Features
- **GA Operators:** Supports multiple selection (tournament, roulette, rank), crossover (single-point, uniform, arithmetic), and mutation (bit-flip, uniform, adaptive) methods.
- **Fitness Evaluation:** Uses scikit-learn's DecisionTreeClassifier with cross-validation and a configurable penalty for feature reduction.
- **Interfaces:** Command-line (CLI) via `main.py`, graphical user interface (GUI) via `app.py` (Streamlit-based), and automated experiments via `compare_operators.py`.
- **Visualization:** Plots for convergence, feature reduction, operator comparisons, and more.
- **Testing:** Unit tests for operators and a toy problem validation.
- **Outputs:** Selected features, performance metrics, experiment summaries, and visualizations saved to `results/`.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/Reemsoliiman/ga-feature-select.git
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   (Requires Python 3.8+; dependencies include numpy, pandas, scikit-learn, matplotlib, seaborn, streamlit, etc.)

3. Prepare datasets: Place CSV files in `data/raw/` (e.g., `diabetes_raw_data.csv`). The last column should be the target variable.

## Usage

### 1. Command-Line Interface (CLI) - `main.py`
Run the GA on a dataset with configurable parameters.
```
python main.py --dataset data/raw/diabetes_raw_data.csv --pop_size 50 --n_gen 100 --selection tournament --crossover single_point --mutation bit_flip
```
- **Arguments:**
  - `--dataset`: Path to CSV file (required).
  - `--pop_size`: Population size (default: 50).
  - `--n_gen`: Number of generations (default: 100).
  - `--mut_rate`: Mutation rate (default: 0.01).
  - `--cross_rate`: Crossover rate (default: 0.8).
  - `--selection`: Selection method (tournament, roulette, rank).
  - `--crossover`: Crossover method (single_point, uniform, arithmetic).
  - `--mutation`: Mutation method (bit_flip, uniform, adaptive).
- **Output:** Prints summary, saves results to `results/` (CSV, plots, JSON).

### 2. Graphical User Interface (GUI) - `app.py`
Interactive Streamlit app for uploading datasets, configuring parameters, running GA, and visualizing results.
```
streamlit run app.py
```
- Upload a CSV, adjust sliders/selectors, click "Run GA".
- View convergence plots, selected features, and download results (JSON/CSV).

### 3. Operator Comparison Experiments - `compare_operators.py`
Automates runs for all 27 operator combinations (3 selections × 3 crossovers × 3 mutations), with multiple runs per config.
```
python experiments/compare_operators.py data/raw/diabetes_raw_data.csv
```
- **Output:** CSVs/JSONs in `results/experiments/`, summaries, and best features. Use for benchmarking operators.

### 4. Notebooks
- `notebooks/01_data_analysis.ipynb`: Exploratory Data Analysis (EDA) on datasets.
- `notebooks/02_results_analysis.ipynb`: Analyze experiment outputs, visualize comparisons.

### 5. Tests
Run unit and integration tests:
```
python -m unittest discover tests/
```
- `test_operators.py`: Validates GA operators.
- `test_toy_problem.py`: Verifies GA on synthetic data with known informative features.

## Configuration
All hyperparameters are centralized in `src/config.py`:
- GA: `POPULATION_SIZE`, `N_GENERATIONS`, `MUTATION_RATE`, etc.
- Fitness: `LAMBDA_PENALTY` (feature reduction weight), `CV_FOLDS`.
- Data: `HANDLE_MISSING` (mean/median/drop), `NORMALIZE` (True/False).
- Experiments: `N_RUNS` (per config), operator lists.
- Outputs: Directory paths (`RESULTS_DIR`, etc.).

Modify this file to tweak defaults without changing code.

## Directory Structure
```
FeatureSelectionGA/
├── data/
│   ├── raw/                  # Original datasets (.csv)
│   └── processed/            # Cached preprocessed data
├── src/
│   ├── __init__.py
│   ├── genetic_algorithm.py  # GA class: evolve(), operators switcher
│   ├── fitness.py            # evaluate(mask, X, y) -> accuracy - λ×features
│   ├── decision_tree.py      # Sklearn DT wrapper
│   ├── utils.py              # load_data(), plotting functions
│   └── config.py             # All hyperparameters
├── experiments/
│   └── compare_operators.py  # Runs 27 configs (3×3×3), saves results
├── gui/
│   └── app.py                # Tkinter/Streamlit GUI
├── tests/
│   ├── test_operators.py     # Unit tests for crossover/mutation
│   └── test_toy_problem.py   # Known-answer validation
├── notebooks/
│   ├── 01_data_analysis.ipynb # EDA only
│   └── 02_results_analysis.ipynb # Post-experiment analysis
├── results/
│   ├── experiments/          # CSVs from all runs
│   ├── plots/                # Generated graphs
│   └── best_features/        # Selected feature subsets (JSON)
├── main.py                   # CLI alternative to GUI
├── requirements.txt
└── README.md
```

## How It Works (Workflow)
1. **Data Loading & Preprocessing:** Load CSV, handle missing values, encode categoricals, normalize (via `utils.py`).
2. **GA Initialization:** Create random binary population (via `genetic_algorithm.py`).
3. **Evolution Loop:**
   - Evaluate fitness: Cross-validate accuracy with decision tree, subtract feature penalty (via `fitness.py`).
   - Select parents, crossover, mutate, apply elitism.
   - Track history (fitness, features selected).
4. **Output & Analysis:** Save best subset, plots (convergence, feature frequency), and metrics. Explore via GUI or notebooks.

## Contributing
Contributions are welcome! Please fork the repo, create a feature branch, and submit a pull request. Follow PEP8 style.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author
Reem Soliman  
(If you have questions or need more details, open an issue!)
