# XGBoost Training Report

**Generated on:** 2025-12-26 00:47:39  
**Model ID:** `6c0032e8-6874-4a7b-a7d6-4813108027a0`  
**Model Folder:** `trained_models/10004/6c0032e8-6874-4a7b-a7d6-4813108027a0`

## Executive Summary

This report documents a comprehensive XGBoost training experiment conducted for academic research and reproducibility purposes. The experiment involved hyperparameter optimization and cross-validated model training with detailed performance analysis, data validation, and feature importance evaluation.

### Key Results
### 🎯 关键性能指标

- **准确率 (Accuracy):** 0.913492 (±0.048821)
- **F1分数 (F1 Score):** 0.912742 (±0.049478)
- **精确率 (Precision):** 0.922516 (±0.040910)
- **召回率 (Recall):** 0.913492 (±0.048821)

- **交叉验证折数:** 5
- **数据集规模:** 139 样本, 18 特征

### ⚙️ 最优超参数

- **n_estimators:** 120
- **max_depth:** 10
- **learning_rate:** 0.038858277024010755
- **subsample:** 0.7604786936090321
- **colsample_bytree:** 0.669898450859942
- **colsample_bylevel:** 0.8275349786040751
- **reg_alpha:** 3.5151690388649365e-08
- **reg_lambda:** 0.00830659664849261
- **min_child_weight:** 1
- **gamma:** 0.12351582603601631

- **训练时间:** 20.84 秒

---

## 1. Experimental Setup

### 1.1 Dataset Information

| Parameter | Value |
|-----------|-------|
| Data File | `http://47.99.180.80/file/uploads/data_b_for_Dynasty.xlsx` |
| Data Shape | {'n_samples': 139, 'n_features': 18} |
| Number of Features | 18 |
| Number of Targets | 1 |

### 1.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Task Type | Classification |

### 1.3 Hardware and Software Environment

- **Python Version:** 3.8+
- **Machine Learning Framework:** XGBoost, scikit-learn
- **Data Processing:** pandas, numpy
- **Hyperparameter Optimization:** Optuna
- **Device:** CPU

---

## 2. Data Processing and Validation

### 2.1 Data Loading and Initial Inspection

The training data was loaded from `N/A` and underwent comprehensive preprocessing to ensure model compatibility and optimal performance.

**Input Features (N/A columns):**
`Na2O`, `MgO`, `Al2O3`, `SiO2`, `K2O`, `CaO`, `TiO2`, `Fe2O3`, `As2O3`, `MnO`, `CuO`, `ZnO`, `PbO2`, `Rb2O`, `SrO`, `Y2O3`, `ZrO2`, `P2O5`

**Target Variables (1 column):**
`Dynasty`


### 2.4 Data Quality Assessment

Comprehensive data validation was performed using multiple statistical methods to ensure dataset quality and suitability for machine learning model training. The validation framework employed established statistical techniques for thorough data quality assessment.

#### 2.4.1 Overall Quality Metrics

| Metric | Value | Threshold | Interpretation |
|--------|-------|-----------|----------------|
| Overall Data Quality Score | 0/100 | ≥80 (Excellent), ≥60 (Good) | Poor - Significant issues require resolution |
| Quality Level | Poor | - | Categorical assessment |
| Ready for Training | No | Yes | Model training readiness |
| Critical Issues | 39 | 0 | Data integrity problems |
| Warnings | 0 | <5 | Minor data quality concerns |

#### 2.4.2 Validation Methodology and Results

| Check Name | Method Used | Status | Issues Found | Key Findings |
|------------|-------------|--------|-------------|-------------|
| Feature Names | Statistical Analysis | ✅ PASSED | 0 | No issues |
| Data Dimensions | Statistical Analysis | ✅ PASSED | 0 | No issues |
| Target Variable | Statistical Analysis | ✅ PASSED | 0 | No issues |
| Data Leakage | Statistical Analysis | ❌ FAILED | 7 | 7 issues found |
| Sample Balance | Chi-square, Gini coefficient | ✅ PASSED | 1 | Balanced; ratio=0.144 |
| Feature Correlations | Pearson/Spearman/Kendall | ✅ PASSED | 1 | 4 high correlations |
| Multicollinearity Detection | Variance Inflation Factor (VIF) | ❌ FAILED | 2 | 8 high VIF; avg=188.16 |
| Feature Distributions | Shapiro-Wilk, Jarque-Bera, D'Agostino | ❌ FAILED | 30 | 30 distribution issues |


#### 2.4.2.4 Sample Balance Analysis

**Methodology**: Chi-square goodness-of-fit test and Gini coefficient calculation for class distribution assessment.

**Results**:
- Minority class ratio: 0.1439
- Dataset balance: Balanced
- Number of classes: 4

**Class Distribution**:
| Class | Count | Proportion | Cumulative % |
|-------|-------|------------|-------------|
| Yuan | 0.4028776978417266 | 0.4029 | 40.3% |
| Ming | 0.302158273381295 | 0.3022 | 70.5% |
| Song | 0.1510791366906475 | 0.1511 | 85.6% |
| Qing | 0.14388489208633093 | 0.1439 | 100.0% |


**Methodological Implications**: Balanced class distribution supports unbiased model training and reliable performance metrics.

#### 2.4.2.1 Feature Correlation Analysis

**Methodology**: Pearson, Spearman, and Kendall correlation coefficients were computed for all feature pairs. The correlation threshold was set at |r| ≥ 0.7.

**Results**: 4 feature pairs exceeded the correlation threshold, indicating potential redundancy in the feature space.

**Feature Classification**:
Continuous Features: Na2O, MgO, Al2O3, SiO2, K2O, CaO, TiO2, Fe2O3, MnO, CuO, ZnO, PbO2, Rb2O, SrO, Y2O3, ZrO2, P2O5
Categorical Features: As2O3
Target Feature: Dynasty

**Statistical Findings**:
**Continuous Features vs Continuous Features Correlation Analysis (Pearson Correlation Coefficient)**:

| Feature 1 | Feature 2 | Correlation | Absolute Value |
|-----------|-----------|-------------|----------------|
| Al2O3 | SiO2 | -0.8174 | 0.8174 |
| K2O | SrO | 0.7515 | 0.7515 |
| K2O | Fe2O3 | -0.7234 | 0.7234 |
| K2O | Rb2O | 0.7138 | 0.7138 |
| Rb2O | SrO | 0.6380 | 0.6380 |
| K2O | ZrO2 | -0.5748 | 0.5748 |
| MgO | K2O | -0.5716 | 0.5716 |
| Fe2O3 | ZrO2 | 0.5605 | 0.5605 |
| Fe2O3 | SrO | -0.5091 | 0.5091 |
| MgO | Al2O3 | 0.4970 | 0.4970 |


**Continuous Features vs Continuous Features Correlation Analysis (Spearman's Rank Correlation)**:

| Feature 1 | Feature 2 | Correlation | Absolute Value |
|-----------|-----------|-------------|----------------|
| K2O | Fe2O3 | -0.7787 | 0.7787 |
| K2O | SrO | 0.7723 | 0.7723 |
| Al2O3 | SiO2 | -0.7542 | 0.7542 |
| K2O | Rb2O | 0.6929 | 0.6929 |
| Rb2O | SrO | 0.6443 | 0.6443 |
| K2O | ZrO2 | -0.6280 | 0.6280 |
| Fe2O3 | SrO | -0.5977 | 0.5977 |
| Fe2O3 | ZrO2 | 0.5737 | 0.5737 |
| MgO | K2O | -0.5613 | 0.5613 |
| MgO | Al2O3 | 0.5426 | 0.5426 |


**Continuous Features vs Categorical Features Correlation Analysis (Correlation Ratio)**:

| Categorical Feature | Continuous Feature | Correlation Ratio | Absolute Value | Strength |
|-------------------|-------------------|-------------------|----------------|----------|
| As2O3 | PbO2 | 0.0636 | 0.0636 | Medium effect (Moderate association) |
| As2O3 | ZnO | 0.0385 | 0.0385 | Small effect (Weak association) |
| As2O3 | Y2O3 | 0.0312 | 0.0312 | Small effect (Weak association) |
| As2O3 | SiO2 | 0.0310 | 0.0310 | Small effect (Weak association) |
| As2O3 | Al2O3 | 0.0237 | 0.0237 | Small effect (Weak association) |
| As2O3 | SrO | 0.0229 | 0.0229 | Small effect (Weak association) |
| As2O3 | MnO | 0.0189 | 0.0189 | Small effect (Weak association) |
| As2O3 | Fe2O3 | 0.0137 | 0.0137 | Small effect (Weak association) |
| As2O3 | CuO | 0.0129 | 0.0129 | Small effect (Weak association) |
| As2O3 | MgO | 0.0091 | 0.0091 | Negligible |
| As2O3 | K2O | 0.0077 | 0.0077 | Negligible |
| As2O3 | P2O5 | 0.0074 | 0.0074 | Negligible |
| As2O3 | Na2O | 0.0061 | 0.0061 | Negligible |
| As2O3 | CaO | 0.0050 | 0.0050 | Negligible |
| As2O3 | ZrO2 | 0.0046 | 0.0046 | Negligible |
| As2O3 | Rb2O | 0.0039 | 0.0039 | Negligible |
| As2O3 | TiO2 | 0.0025 | 0.0025 | Negligible |


**Continuous Features vs Target Variable Correlation Analysis**:

| Feature | Correlation | Method | Absolute Value | Strength |
|---------|-------------|--------|----------------|----------|
| K2O | 0.6467 | correlation_ratio | 0.6467 | Moderate |
| Fe2O3 | 0.5321 | correlation_ratio | 0.5321 | Moderate |
| ZrO2 | 0.5006 | correlation_ratio | 0.5006 | Moderate |
| Rb2O | 0.4755 | correlation_ratio | 0.4755 | Weak |
| Al2O3 | 0.4682 | correlation_ratio | 0.4682 | Weak |
| Y2O3 | 0.4473 | correlation_ratio | 0.4473 | Weak |
| SiO2 | 0.4200 | correlation_ratio | 0.4200 | Weak |
| MnO | 0.4071 | correlation_ratio | 0.4071 | Weak |
| SrO | 0.3768 | correlation_ratio | 0.3768 | Weak |
| MgO | 0.3186 | correlation_ratio | 0.3186 | Weak |
| Na2O | 0.2639 | correlation_ratio | 0.2639 | Very Weak |
| ZnO | 0.2298 | correlation_ratio | 0.2298 | Very Weak |
| PbO2 | 0.1656 | correlation_ratio | 0.1656 | Very Weak |
| P2O5 | 0.0883 | correlation_ratio | 0.0883 | Very Weak |
| CuO | 0.0531 | correlation_ratio | 0.0531 | Very Weak |
| TiO2 | 0.0291 | correlation_ratio | 0.0291 | Very Weak |
| CaO | 0.0264 | correlation_ratio | 0.0264 | Very Weak |


**Categorical Features vs Target Variable Correlation Analysis**:

| Categorical Feature | Association | Method | Absolute Value | Strength |
|------------------- |-------------|--------|----------------|----------|
| As2O3 | 0.1327 | cramers_v | 0.1327 | Very Weak |


**Impact Assessment**: High feature correlation may lead to multicollinearity issues and reduced model interpretability.

#### 2.4.2.2 Multicollinearity Detection

**Methodology**: Variance Inflation Factor (VIF) analysis was conducted using linear regression. VIF values ≥ 5.0 indicate problematic multicollinearity.

**Results**: 
- Average VIF: 188.156
- Maximum VIF: 1000.000
- Features with VIF ≥ 5.0: 8

**Statistical Findings**:
**VIF Scores for All Features**:

| Feature | VIF Score | R² | Interpretation | Status |
|---------|-----------|----|--------------|---------|
| Al2O3 | 1000.0000 | 0.9990 | Severe | ⚠️ HIGH |
| SiO2 | 1000.0000 | 0.9990 | Severe | ⚠️ HIGH |
| K2O | 1000.0000 | 0.9990 | Severe | ⚠️ HIGH |
| Na2O | 159.8896 | 0.9937 | Severe | ⚠️ HIGH |
| Fe2O3 | 115.7713 | 0.9914 | Severe | ⚠️ HIGH |
| MgO | 38.1326 | 0.9738 | Severe | ⚠️ HIGH |
| As2O3 | 33.6418 | 0.9703 | Severe | ⚠️ HIGH |
| CaO | 15.6726 | 0.9362 | Severe | ⚠️ HIGH |
| TiO2 | 4.4093 | 0.7732 | Acceptable | ✅ LOW |
| Rb2O | 4.0223 | 0.7514 | Acceptable | ✅ LOW |
| SrO | 3.3204 | 0.6988 | Acceptable | ✅ LOW |
| ZrO2 | 2.3457 | 0.5737 | Acceptable | ✅ LOW |
| ZnO | 1.9081 | 0.4759 | Acceptable | ✅ LOW |
| MnO | 1.8430 | 0.4574 | Acceptable | ✅ LOW |
| Y2O3 | 1.6324 | 0.3874 | Acceptable | ✅ LOW |
| PbO2 | 1.6175 | 0.3818 | Acceptable | ✅ LOW |
| CuO | 1.3498 | 0.2591 | Acceptable | ✅ LOW |
| P2O5 | 1.2511 | 0.2007 | Acceptable | ✅ LOW |


**Methodological Impact**: Elevated VIF scores suggest linear dependencies between predictors, which may compromise model stability and coefficient interpretation.

#### 2.4.2.3 Feature Distribution Analysis

**Methodology**: 
- Continuous features: Shapiro-Wilk test (n≤5000), Jarque-Bera test (n≥50), D'Agostino test (n≥20) for normality
- Skewness assessment using sample skewness coefficient
- Outlier detection via Interquartile Range (IQR) method
- Categorical features: Gini coefficient, entropy, and class imbalance ratio analysis

**Results**: 0 distribution-related issues identified across 7 continuous and 6 categorical features.

**Continuous Features Statistical Summary**:
| Feature | mean | std | min | max | max | median | Skewness | Kurtosis | Normality | Outliers (%) | Issues |
|---------|----------|---------|----------|---------|----------|---------|----------|-----------|-------------|--------|
| Al2O3 | 19.499 | 1.891 | 14.195 | 26.101 | 26.101 | 19.365 | 0.135 | 0.591 | Yes | 1.4% | 0 |
| CaO | 0.185 | 0.084 | 0.098 | 0.679 | 0.679 | 0.163 | 3.046 | 12.076 | No | 7.2% | 3 |
| CuO | 40.072 | 27.174 | 0.000 | 140.000 | 140.000 | 40.000 | 0.523 | 0.536 | No | 0.7% | 1 |
| Fe2O3 | 0.919 | 0.246 | 0.580 | 2.072 | 2.072 | 0.880 | 1.192 | 2.791 | No | 1.4% | 2 |
| K2O | 4.963 | 1.279 | 2.586 | 7.748 | 7.748 | 4.776 | 0.281 | -0.902 | No | 0.0% | 1 |
| MgO | 0.136 | 0.127 | 0.000 | 0.529 | 0.529 | 0.112 | 0.893 | 0.244 | No | 0.7% | 1 |
| MnO | 387.338 | 194.301 | 90.000 | 1190.000 | 1190.000 | 350.000 | 1.215 | 2.061 | No | 2.2% | 2 |
| Na2O | 0.418 | 0.293 | 0.064 | 1.636 | 1.636 | 0.342 | 1.373 | 2.048 | No | 6.5% | 3 |
| P2O5 | 38.345 | 33.850 | 0.000 | 180.000 | 180.000 | 30.000 | 0.994 | 1.173 | No | 0.7% | 1 |
| PbO2 | 84.245 | 50.633 | 0.000 | 280.000 | 280.000 | 80.000 | 1.247 | 2.137 | No | 5.8% | 3 |
| Rb2O | 289.784 | 36.285 | 200.000 | 390.000 | 390.000 | 290.000 | 0.172 | -0.431 | No | 0.0% | 0 |
| SiO2 | 72.817 | 1.908 | 66.670 | 78.233 | 78.233 | 72.632 | 0.156 | 0.581 | Yes | 2.9% | 0 |
| SrO | 69.424 | 34.363 | 0.000 | 150.000 | 150.000 | 70.000 | 0.175 | -0.840 | No | 0.0% | 1 |
| TiO2 | 0.072 | 0.037 | 0.022 | 0.355 | 0.355 | 0.074 | 3.923 | 25.806 | No | 5.0% | 3 |
| Y2O3 | 118.489 | 75.833 | 40.000 | 630.000 | 630.000 | 100.000 | 3.351 | 15.978 | No | 8.6% | 3 |
| ZnO | 138.417 | 55.629 | 60.000 | 360.000 | 360.000 | 120.000 | 1.467 | 2.504 | No | 5.8% | 3 |
| ZrO2 | 182.230 | 46.719 | 100.000 | 390.000 | 390.000 | 170.000 | 1.314 | 2.721 | No | 1.4% | 2 |

**Continuous Feature Distribution Issues:**
- Feature 'Na2O' is highly skewed (1.37)
- Feature 'Na2O' has moderate outlier ratio (6.5%)
- Feature 'Na2O' significantly deviates from normal distribution
- Feature 'MgO' significantly deviates from normal distribution
- Feature 'K2O' significantly deviates from normal distribution
- Feature 'CaO' has extreme skewness (3.05)
- Feature 'CaO' has moderate outlier ratio (7.2%)
- Feature 'CaO' significantly deviates from normal distribution
- Feature 'TiO2' has extreme skewness (3.92)
- Feature 'TiO2' has moderate outlier ratio (5.0%)
- Feature 'TiO2' significantly deviates from normal distribution
- Feature 'Fe2O3' is highly skewed (1.19)
- Feature 'Fe2O3' significantly deviates from normal distribution
- Feature 'MnO' is highly skewed (1.21)
- Feature 'MnO' significantly deviates from normal distribution
- Feature 'CuO' significantly deviates from normal distribution
- Feature 'ZnO' is highly skewed (1.47)
- Feature 'ZnO' has moderate outlier ratio (5.8%)
- Feature 'ZnO' significantly deviates from normal distribution
- Feature 'PbO2' is highly skewed (1.25)
- Feature 'PbO2' has moderate outlier ratio (5.8%)
- Feature 'PbO2' significantly deviates from normal distribution
- Feature 'SrO' significantly deviates from normal distribution
- Feature 'Y2O3' has extreme skewness (3.35)
- Feature 'Y2O3' has moderate outlier ratio (8.6%)
- Feature 'Y2O3' significantly deviates from normal distribution
- Feature 'ZrO2' is highly skewed (1.31)
- Feature 'ZrO2' significantly deviates from normal distribution
- Feature 'P2O5' significantly deviates from normal distribution


**Categorical Features Statistical Summary**:
| Feature | Classes | Gini Coeff | Imbalance Ratio | Entropy | Issues |
|---------|---------|------------|-----------------|---------|--------|
| distribution_statistics | 0 | 0.000 | 1.0:1 | 0.000 | 0 |
| imbalance_analysis | 0 | 0.000 | 1.0:1 | 0.000 | 0 |
| cardinality_analysis | 0 | 0.000 | 1.0:1 | 0.000 | 0 |


**Distribution Quality Impact**: Feature distributions meet statistical assumptions for machine learning applications.

#### 2.4.2.5 Statistical Summary

**Validation Framework Performance**:
- Total validation checks: 8
- Passed checks: 5 (62.5%)
- Failed checks: 3

**Data Quality Confidence**: Based on the comprehensive validation framework, the dataset demonstrates moderate statistical reliability for machine learning applications.

#### 2.4.3 Data Quality Issues and Impact Assessment

**Critical Issues Identified:**

- High suspicion data leakage: 'Na2O' (correlation: 1.000)
- High suspicion data leakage: 'Al2O3' (correlation: 1.000)
- High suspicion data leakage: 'SiO2' (correlation: 1.000)
- High suspicion data leakage: 'K2O' (correlation: 1.000)
- High suspicion data leakage: 'CaO' (correlation: 1.000)
- High suspicion data leakage: 'TiO2' (correlation: 1.000)
- High suspicion data leakage: 'Fe2O3' (correlation: 1.000)
- Detected 8 features with high VIF (>= 5.0)
- Found 4 highly correlated feature pairs
- Feature 'Na2O' is highly skewed (1.37)
- Feature 'Na2O' has moderate outlier ratio (6.5%)
- Feature 'Na2O' significantly deviates from normal distribution

**Data Quality Recommendations:**

1. Address distribution issues through transformation or preprocessing
2. Resolve multicollinearity using VIF-guided feature selection or regularization
3. Apply balancing techniques (SMOTE, undersampling, class weights)
4. Remove or investigate highly correlated features for potential data leakage
5. Investigate high correlations and consider feature selection
6. Consider target transformation for heavily skewed targets


#### 2.4.4 Academic and Methodological Implications

The data validation results indicate that the dataset does not meet the quality standards required for academic machine learning research. Poor data quality may compromise experimental validity. Significant preprocessing and quality improvements are recommended before publication.

**Reproducibility Impact**: Low reproducibility confidence due to data quality issues. Preprocessing standardization required for reliable replication.


### 2.2 Data Preprocessing Pipeline

The data underwent comprehensive preprocessing to optimize model performance and ensure consistent data quality.

#### 2.2.1 Feature Preprocessing

**Preprocessing Method**: StandardScaler (Z-score normalization)

```python
# Feature transformation: X_scaled = (X - μ) / σ
# Where μ = mean, σ = standard deviation
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
```

**Preprocessing Benefits:**
- **Feature Consistency**: Normalizes different scales and units
- **Algorithm Optimization**: Improves convergence for distance-based methods
- **Numerical Stability**: Prevents overflow/underflow in computations
- **Cross-Validation Integrity**: Separate scaling per fold prevents data leakage

### 2.3 Feature Engineering

### 2.3 Feature Selection and Engineering

#### 2.3.1 Feature Selection Strategy

**Approach**: Comprehensive feature utilization

XGBoost inherently performs feature selection during the training of boosted trees. Key mechanisms include:
- **Greedy Search**: At each split, the algorithm selects the feature and split point that maximize the gain.
- **Regularization**: L1 (Lasso) and L2 (Ridge) regularization penalize complex models, effectively shrinking the coefficients of less important features.
- **Feature Importance Calculation**: XGBoost provides multiple metrics (gain, weight, cover) to score feature relevance automatically.

#### 2.3.2 Feature Engineering Pipeline

**Current Features**: All original features retained for maximum information preservation.
**Categorical Encoding**: Best practice is to one-hot encode categorical features for XGBoost.
**Missing Value Strategy**: XGBoost has a built-in, optimized routine to handle missing values by learning a default direction for them at each split.
**Feature Interaction**: Captured implicitly and explicitly through the tree-based structure of the model.


---

## 3. Hyperparameter Optimization

### 3.1 Hyperparameter Search Space

The optimization process systematically explored a comprehensive parameter space designed to balance model complexity and performance:

| Parameter | Range/Options | Description |
|-----------|---------------|-------------|
| n_estimators | 50-150 (step: 10) | Number of boosting rounds (trees) in the ensemble |
| max_depth | 1-10 (step: 1) | Maximum depth of each tree in the ensemble |
| learning_rate | 0.01-0.3 (log scale) | Step size shrinkage to prevent overfitting |
| subsample | 0.6-1.0 (linear scale) | Fraction of samples used for training each tree |
| colsample_bytree | 0.6-1.0 (linear scale) | Fraction of features used for training each tree |
| colsample_bylevel | 0.6-1.0 (linear scale) | Fraction of features used for each level in each tree |
| reg_alpha | 1e-08-10.0 (log scale) | L1 regularization term on weights (Lasso regularization) |
| reg_lambda | 1e-08-10.0 (log scale) | L2 regularization term on weights (Ridge regularization) |
| min_child_weight | 1-10 (step: 1) | Minimum sum of instance weight needed in a child node |
| gamma | 1e-08-10.0 (log scale) | Minimum loss reduction required to make a split |

### 3.2 Optimization Algorithm and Strategy

**Algorithm**: TPE (Tree-structured Parzen Estimator)
**Total Trials**: 50
**Completed Trials**: 50
**Best Score**: 0.912742

**Optimization Strategy:**
- **Initial Exploration**: 10 random trials for space exploration
- **Exploitation-Exploration Balance**: TPE algorithm balances promising regions with unexplored space
- **Cross-Validation**: Each trial evaluated using stratified k-fold cross-validation
- **Early Stopping**: Poor-performing trials terminated early to improve efficiency

### 3.3 Best Parameters Found

```json
{
  "n_estimators": 120,
  "max_depth": 10,
  "learning_rate": 0.038858277024010755,
  "subsample": 0.7604786936090321,
  "colsample_bytree": 0.669898450859942,
  "colsample_bylevel": 0.8275349786040751,
  "reg_alpha": 3.5151690388649365e-08,
  "reg_lambda": 0.00830659664849261,
  "min_child_weight": 1,
  "gamma": 0.12351582603601631
}
```

### 3.4 Optimization Convergence

The optimization process completed **50 trials** with the best configuration achieving a cross-validation score of **0.912742**.

**Key Optimization Insights:**
- **Ensemble Size**: 120 boosting rounds balances performance and computational efficiency
- **Tree Complexity**: Maximum depth of 10 controls model complexity and overfitting
- **Learning Rate**: 0.038858277024010755 provides optimal step size for gradient descent
- **Regularization**: L1=3.52e-08, L2=8.31e-03 prevent overfitting
- **Sampling**: 0.7604786936090321 row sampling and 0.669898450859942 column sampling for robustness

## 4. Final Model Training

### 4.1 Cross-Validation Training

The final model was trained using 5-fold cross-validation with optimized hyperparameters. Training metrics and validation results were recorded comprehensively.

### 4.2 Training Results

| Metric | Value |
|--------|-------|
### Cross-Validation Performance Metrics

| Metric | Mean ± Std | Min | Max |
|--------|------------|-----|-----|
| ACCURACY | 0.913492 ± 0.048821 | 0.857143 | 1.000000 |
| F1 | 0.912742 ± 0.049478 | 0.854867 | 1.000000 |
| PRECISION | 0.922516 ± 0.040910 | 0.893707 | 1.000000 |
| RECALL | 0.913492 ± 0.048821 | 0.857143 | 1.000000 |



#### Fold-wise Results

#### Detailed Fold-wise Performance

| Fold | ACCURACY | F1 | PRECISION | RECALL |
|------|---------|---------|---------|---------|
| 1 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 2 | 0.928571 | 0.928571 | 0.928571 | 0.928571 |
| 3 | 0.857143 | 0.854867 | 0.895238 | 0.857143 |
| 4 | 0.892857 | 0.891615 | 0.893707 | 0.892857 |
| 5 | 0.888889 | 0.888659 | 0.895062 | 0.888889 |

#### Statistical Summary

| Metric | Mean | Std Dev | Min | Max | 95% CI |
|--------|------|---------|-----|-----|--------|
| ACCURACY | 0.913492 | 0.048821 | 0.857143 | 1.000000 | [0.870699, 0.956285] |
| F1 | 0.912742 | 0.049478 | 0.854867 | 1.000000 | [0.869373, 0.956112] |
| PRECISION | 0.922516 | 0.040910 | 0.893707 | 1.000000 | [0.886656, 0.958375] |
| RECALL | 0.913492 | 0.048821 | 0.857143 | 1.000000 | [0.870699, 0.956285] |

### 4.3 Model Performance Visualization

#### Classification Performance Analysis

The cross-validation analysis demonstrates the model's classification performance through ROC curves showing the trade-off between true positive rate and false positive rate.

<div style="text-align: center; margin: 20px 0;">
    <img src="cross_validation_data/cross_validation_roc_curves.png" alt="Cross-Validation ROC Curves" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    <p style="font-style: italic; color: #666; margin-top: 10px;">Cross-Validation ROC Curves</p>
</div>



### 4.4 Feature Importance Analysis

#### Feature Importance Analysis

This analysis employs multiple methodologies to comprehensively evaluate feature importance in the XGBoost model:

**Analysis Methods:**

1. **Built-in Importance (Gain, Cover, Weight)**:
   - **Gain**: The average training loss reduction gained when a feature is used for splitting. It is the most common and relevant metric.
   - **Cover**: The average number of samples affected by splits on this feature.
   - **Weight**: The number of times a feature is used to split the data across all trees.

2. **Permutation Importance**:
   - Model-agnostic method measuring feature contribution to model performance
   - Evaluates performance drop when feature values are randomly shuffled
   - More reliable for correlated features and unbiased feature ranking
   - Computed on out-of-sample data to avoid overfitting

**XGBoost Tree-based Feature Importance:**

| Rank | Feature | Gain | Weight | Cover | Gain % | Weight % |
|------|---------|------|--------|-------|--------|----------|
| 1 | `K2O` | 3.1242 | 260 | 14.63 | 12.6% | 11.8% |
| 2 | `Y2O3` | 2.5802 | 149 | 13.74 | 10.4% | 6.8% |
| 3 | `Al2O3` | 2.4136 | 201 | 13.99 | 9.8% | 9.1% |
| 4 | `ZrO2` | 1.9319 | 166 | 11.54 | 7.8% | 7.5% |
| 5 | `SiO2` | 1.9030 | 112 | 11.05 | 7.7% | 5.1% |
| 6 | `Fe2O3` | 1.8709 | 218 | 10.92 | 7.6% | 9.9% |
| 7 | `Rb2O` | 1.7362 | 141 | 13.66 | 7.0% | 6.4% |
| 8 | `MnO` | 1.3282 | 188 | 8.90 | 5.4% | 8.5% |
| 9 | `SrO` | 1.1309 | 63 | 10.44 | 4.6% | 2.9% |
| 10 | `PbO2` | 0.9751 | 83 | 10.90 | 3.9% | 3.8% |
| 11 | `ZnO` | 0.9646 | 66 | 9.54 | 3.9% | 3.0% |
| 12 | `CuO` | 0.9363 | 81 | 10.37 | 3.8% | 3.7% |
| 13 | `Na2O` | 0.9066 | 137 | 10.58 | 3.7% | 6.2% |
| 14 | `MgO` | 0.8237 | 95 | 7.63 | 3.3% | 4.3% |
| 15 | `TiO2` | 0.6208 | 91 | 7.73 | 2.5% | 4.1% |
| 16 | `P2O5` | 0.6170 | 61 | 9.12 | 2.5% | 2.8% |
| 17 | `CaO` | 0.6063 | 83 | 7.32 | 2.5% | 3.8% |
| 18 | `As2O3` | 0.2526 | 6 | 11.72 | 1.0% | 0.3% |


**Permutation Feature Importance:**

| Rank | Feature | Mean Importance | Std Dev | 95% CI | Reliability |
|------|---------|-----------------|---------|--------|-------------|
| 1 | `K2O` | 0.0820 | 0.0148 | [0.0530, 0.1110] | 🟡 Medium |
| 2 | `Al2O3` | 0.0403 | 0.0073 | [0.0259, 0.0547] | 🟡 Medium |
| 3 | `CuO` | 0.0216 | 0.0000 | [0.0216, 0.0216] | 🟢 High |
| 4 | `ZrO2` | 0.0173 | 0.0073 | [0.0029, 0.0316] | 🔴 Low |
| 5 | `Rb2O` | 0.0072 | 0.0064 | [-0.0054, 0.0198] | 🔴 Low |
| 6 | `Y2O3` | 0.0072 | 0.0046 | [-0.0017, 0.0161] | 🔴 Low |
| 7 | `MnO` | 0.0043 | 0.0035 | [-0.0026, 0.0112] | 🔴 Low |
| 8 | `Na2O` | 0.0029 | 0.0035 | [-0.0040, 0.0098] | 🔴 Low |
| 9 | `MgO` | 0.0000 | 0.0000 | [0.0000, 0.0000] | 🔴 Low |
| 10 | `SiO2` | 0.0000 | 0.0000 | [0.0000, 0.0000] | 🔴 Low |
| 11 | `CaO` | 0.0000 | 0.0000 | [0.0000, 0.0000] | 🔴 Low |
| 12 | `TiO2` | 0.0000 | 0.0000 | [0.0000, 0.0000] | 🔴 Low |
| 13 | `Fe2O3` | 0.0000 | 0.0000 | [0.0000, 0.0000] | 🔴 Low |
| 14 | `As2O3` | 0.0000 | 0.0000 | [0.0000, 0.0000] | 🔴 Low |
| 15 | `ZnO` | 0.0000 | 0.0000 | [0.0000, 0.0000] | 🔴 Low |
| 16 | `PbO2` | 0.0000 | 0.0000 | [0.0000, 0.0000] | 🔴 Low |
| 17 | `SrO` | 0.0000 | 0.0000 | [0.0000, 0.0000] | 🔴 Low |
| 18 | `P2O5` | 0.0000 | 0.0000 | [0.0000, 0.0000] | 🔴 Low |


**Feature Importance Method Comparison:**

| Feature | XGB Gain Rank | Permutation Rank | Rank Difference | Consistency |
|---------|---------------|------------------|-----------------|-------------|
| `Na2O` | 13 | 8 | 5 | 🔴 Poor |
| `MgO` | 14 | 9 | 5 | 🔴 Poor |
| `Al2O3` | 3 | 2 | 1 | 🟢 Excellent |
| `SiO2` | 5 | 10 | 5 | 🔴 Poor |
| `K2O` | 1 | 1 | 0 | 🟢 Excellent |
| `CaO` | 17 | 11 | 6 | 🔴 Poor |
| `TiO2` | 15 | 12 | 3 | 🔴 Poor |
| `Fe2O3` | 6 | 13 | 7 | 🔴 Poor |
| `As2O3` | 18 | 14 | 4 | 🔴 Poor |
| `MnO` | 8 | 7 | 1 | 🟢 Excellent |
| `CuO` | 12 | 3 | 9 | 🔴 Poor |
| `ZnO` | 11 | 15 | 4 | 🔴 Poor |
| `PbO2` | 10 | 16 | 6 | 🔴 Poor |
| `Rb2O` | 7 | 5 | 2 | 🟡 Good |
| `SrO` | 9 | 17 | 8 | 🔴 Poor |
| `Y2O3` | 2 | 6 | 4 | 🔴 Poor |
| `ZrO2` | 4 | 4 | 0 | 🟢 Excellent |
| `P2O5` | 16 | 18 | 2 | 🟡 Good |


**Statistical Summary:**

- **Total Features Analyzed**: 18
- **Gain-based Top Feature**: `K2O` (Gain: 3.1242)
- **Permutation-based Top Feature**: `K2O` (Importance: 0.0820)

**Method Reliability Assessment:**
- **Average Permutation Std**: 0.0026
- **Method Agreement**: High

**Feature Importance Visualizations:**

![Feature Importance Comparison](feature_importance_comparison.png)

**Method Comparison Plot**: `feature_importance_comparison.png`

![Permutation Feature Importance](feature_importance_permutation.png)

**Permutation Importance Plot**: `feature_importance_permutation.png`

![Tree-based Feature Importance](feature_importance_tree.png)

**Tree-based Importance Plot**: `feature_importance_tree.png`

**Feature Importance Data Files:**

- `feature_importance.csv` - Detailed feature importance scores and statistics

**Statistical Interpretation:**

- **Threshold Selection**: Features with importance > 1/n_features are considered significant
- **Cumulative Importance**: Top features typically capture 80-90% of total importance
- **Stability Assessment**: Low standard deviation in permutation importance indicates reliable features
- **Domain Validation**: Feature rankings should align with domain knowledge and expectations

**Technical Implementation Notes:**

- Tree-based importance computed using XGBoost's `feature_importances_` attribute or `get_score()` method.
- Permutation importance calculated with 10 repetitions for statistical robustness
- Random state fixed for reproducible permutation results
- Analysis performed on validation data to avoid overfitting bias


---

## 5. Model Architecture and Configuration

### 5.1 XGBoost Configuration

The final model uses an XGBoost gradient boosting ensemble with the following specifications:

| Component | Configuration |
|-----------|---------------|
| Booster | gbtree (tree-based model) |

### 5.2 Training Parameters

| Parameter | Value |
|-----------|-------|
| Task Type | Classification |

---

## 6. Conclusions and Future Work

### 6.1 Key Findings

2. **Hyperparameter Optimization**: Systematic optimization improved model performance

### 6.2 Reproducibility

This experiment is fully reproducible using the following artifacts:
- **Cross-Validation Data**: `trained_models/10004/6c0032e8-6874-4a7b-a7d6-4813108027a0/cross_validation_data/`
- **Feature Importance**: `trained_models/10004/6c0032e8-6874-4a7b-a7d6-4813108027a0/feature_importance.csv`

### 6.3 Technical Implementation

- **Framework**: XGBoost for gradient boosting implementation, scikit-learn for pipeline integration.
- **Data Processing**: pandas and numpy for data handling.
- **Cross-Validation**: K-fold cross-validation with stratification support for classification.
- **Feature Importance**: Built-in XGBoost feature importance calculation (Gain, Cover, Weight).
- **Serialization**: Joblib or Pickle for model and preprocessor persistence.

---

## Appendix

### A.1 System Information

- **Generation Time**: 2025-12-26 00:47:39
- **Model ID**: `6c0032e8-6874-4a7b-a7d6-4813108027a0`
- **Training System**: XGBoost MCP Tool
- **Report Version**: 2.1 (XGBoost Enhanced)

### A.2 File Structure

```
6c0032e8-6874-4a7b-a7d6-4813108027a0/
├── model.joblib
├── preprocessing_pipeline.pkl
├── evaluation_metrics.csv
├── feature_importance.csv
├── optimization_history.csv
├── raw_data.csv
├── categorical_feature_cardinality.png
├── categorical_feature_distributions.png
├── categorical_vif_details.png
├── continuous_feature_distributions.png
├── continuous_feature_normality.png
├── continuous_feature_outliers.png
├── continuous_feature_violin_plots.png
├── continuous_pearson_correlation.png
├── continuous_spearman_correlation.png
├── encoding_strategy_summary.png
├── feature_importance_comparison.png
├── feature_importance_permutation.png
├── feature_importance_tree.png
├── feature_target_categorical_association.png
├── feature_target_continuous_correlation.png
├── mixed_correlation_ratio.png
├── vif_scores.png
├── vif_threshold_analysis.png
├── cross_validation_results.json
├── data_validation_report.json
├── feature_importance_analysis.json
├── hyperparameter_optimization.json
├── metadata.json
├── preprocessing_info.json
├── training_report.json
├── training_summary.json
├── cross_validation_data/
│   ├── 6c0032e8-6874-4a7b-a7d6-4813108027a0_cv_predictions_original.csv
│   ├── 6c0032e8-6874-4a7b-a7d6-4813108027a0_cv_predictions_processed.csv
│   ├── 6c0032e8-6874-4a7b-a7d6-4813108027a0_original_data.csv
│   ├── 6c0032e8-6874-4a7b-a7d6-4813108027a0_preprocessed_data.csv
│   ├── 6c0032e8-6874-4a7b-a7d6-4813108027a0_roc_curves.png
│   ├── cross_validation_roc_curves.png
│   ├── cross_validation_visualization.png
└── academic_report.md               # This report
```

### A.3 Data Files and JSON Artifacts

The following JSON files contain detailed intermediate data for reproducibility:

- **Feature Importance**: `trained_models/10004/6c0032e8-6874-4a7b-a7d6-4813108027a0/feature_importance.csv`

---

*This report was automatically generated by the Enhanced XGBoost MCP Tool for academic research and reproducibility purposes.*
