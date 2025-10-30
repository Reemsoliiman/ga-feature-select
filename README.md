# GA-FeatureSelect: Evolutionary Feature Selection for Medical Diagnosis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A genetic algorithm-based feature selection framework optimized for medical diagnosis datasets. This project implements a multi-objective optimization approach that maximizes classification accuracy while minimizing feature count, addressing the critical challenge of model interpretability and efficiency in healthcare applications.

## 🎯 Project Overview

Feature selection is crucial in medical machine learning where:
- High-dimensional data can lead to overfitting
- Model interpretability is essential for clinical adoption
- Computational efficiency impacts deployment feasibility
- Reducing features can reveal meaningful biomarkers

This project uses evolutionary computation to intelligently search the feature space, finding optimal subsets that balance predictive performance with model simplicity.

## 🔬 Key Features

- **Custom Genetic Algorithm Implementation**: From-scratch GA with configurable operators
- **Multi-Objective Optimization**: Simultaneous optimization of accuracy and feature reduction
- **Comprehensive Operator Analysis**: Systematic comparison of selection, crossover, and mutation strategies
- **Decision Tree Integration**: Fast, interpretable base classifier
- **Robust Evaluation Framework**: Cross-validation, statistical significance testing
- **Visualization Suite**: Convergence plots, feature importance, performance metrics

## 📊 Methodology

### Genetic Algorithm Components
- **Individual Representation**: Binary encoding where each bit represents feature inclusion
- **Fitness Function**: Weighted combination of classification accuracy and feature count
- **Selection Methods**: Tournament, roulette wheel, rank-based
- **Crossover Operators**: Single-point, two-point, uniform
- **Mutation Strategies**: Bit-flip with adaptive rates

### Experimental Design
- Multiple independent runs for statistical validity
- Comparison against baseline methods (all features, random selection)
- Ablation studies on GA hyperparameters
- Feature stability analysis across runs

## 🗂️ Repository Structure

```
ga-feature-select/
├── data/                   # Dataset storage (not tracked)
├── src/
│   ├── ga/
│   │   ├── __init__.py
│   │   ├── genetic_algorithm.py
│   │   ├── operators.py
│   │   └── fitness.py
│   ├── models/
│   │   └── classifier.py
│   ├── utils/
│   │   ├── preprocessing.py
│   │   └── evaluation.py
│   └── visualization/
│       └── plots.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_experiments.ipynb
│   └── 03_ga_optimization.ipynb
├── experiments/
│   └── configs/            # Experiment configuration files
├── results/                # Output metrics, plots, models
├── tests/                  # Unit tests
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip or conda package manager
```

### Installation
```bash
# Clone the repository
git clone https://github.com/Reemsoliiman/ga-feature-select.git
cd ga-feature-select

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start
```python
from src.ga import GeneticAlgorithm
from src.models import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer

# Load data
X, y = load_breast_cancer(return_X_y=True)

# Initialize GA
ga = GeneticAlgorithm(
    n_features=X.shape[1],
    population_size=50,
    n_generations=100,
    classifier=DecisionTreeClassifier()
)

# Run feature selection
best_features, history = ga.evolve(X, y)
print(f"Selected {sum(best_features)} features with accuracy: {history['best_fitness'][-1]:.3f}")
```

## 📈 Expected Outcomes

- **Feature Reduction**: 30-70% reduction in feature count
- **Accuracy Maintenance**: Comparable or improved accuracy vs. all features
- **Operator Insights**: Clear performance differences between GA strategies
- **Computational Efficiency**: Analysis of convergence speed and resource usage

## 🔧 Configuration

Key hyperparameters can be tuned in `experiments/configs/`:
- Population size
- Generation count
- Mutation rate
- Crossover probability
- Selection pressure
- Fitness function weights

## 📚 Datasets

This project includes experiments on:
- **Breast Cancer Wisconsin** (diagnostic)
- **Heart Disease UCI**
- **Diabetes** (Pima Indians)

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:
- Additional selection/crossover operators
- Multi-objective optimization algorithms (NSGA-II)
- Integration with other classifiers (SVM, Neural Networks)
- Parallel GA implementation
- Real-time visualization dashboard


## 🙏 Acknowledgments

- Inspired by research in evolutionary computation for feature selection
- Built for demonstrating ML engineering skills in healthcare AI applications

## 📧 Contact

For questions or collaboration opportunities, please open an issue or reach out via email.

---

**Note**: This is an educational/portfolio project demonstrating feature selection techniques. For production medical applications, consult with domain experts and follow appropriate regulatory guidelines.