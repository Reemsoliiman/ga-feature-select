# Data Directory

This directory contains the datasets used for feature selection experiments.

## Structure

```
data/
├── raw/          # Original, immutable datasets
├── processed/    # Cleaned, preprocessed datasets
└── external/     # Data from third-party sources
```

## Datasets

### 1. Breast Cancer Wisconsin (Diagnostic)
- **Source**: UCI Machine Learning Repository / sklearn
- **Features**: 30 numeric predictive attributes
- **Target**: Binary (Malignant/Benign)
- **Samples**: 569
- **Use**: Primary benchmark for GA performance

### 2. Heart Disease UCI
- **Source**: UCI Machine Learning Repository
- **Features**: 13 clinical attributes
- **Target**: Binary (Presence/Absence of heart disease)
- **Samples**: 303
- **Use**: Cross-validation of feature selection approach

### 3. Diabetes (Pima Indians)
- **Source**: UCI Machine Learning Repository / sklearn
- **Features**: 8 medical predictor variables
- **Target**: Binary (Diabetes onset)
- **Samples**: 768
- **Use**: Small-scale feature set experiment

## Adding Custom Datasets

Place your datasets in `raw/` directory following this structure:
- CSV format with header row
- Last column as target variable
- No missing values (or handle in preprocessing)
- Numeric features (or encode categorical variables)

## Data Privacy

⚠️ **Important**: Do not commit any datasets containing:
- Personal health information (PHI)
- Personally identifiable information (PII)
- Proprietary or sensitive data

All data files are gitignored by default for privacy protection.