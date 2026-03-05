# XGBoost Training Report

**Generated on:** 2025-12-26 01:31:57  
**Model ID:** `916339bd-2069-4d41-b0be-a7eb58f98b1f`  
**Model Folder:** `trained_models/10004/916339bd-2069-4d41-b0be-a7eb58f98b1f`

## Executive Summary

This report documents a comprehensive XGBoost training experiment conducted for academic research and reproducibility purposes. The experiment involved hyperparameter optimization and cross-validated model training with detailed performance analysis, data validation, and feature importance evaluation.

### Key Results
### 🎯 关键性能指标

- **准确率 (Accuracy):** 0.926455 (±0.052389)
- **F1分数 (F1 Score):** 0.925044 (±0.054033)
- **精确率 (Precision):** 0.934512 (±0.050744)
- **召回率 (Recall):** 0.926455 (±0.052389)

- **交叉验证折数:** 5
- **数据集规模:** 136 样本, 18 特征

### ⚙️ 最优超参数

- **n_estimators:** 90
- **max_depth:** 9
- **learning_rate:** 0.1062272193666865
- **subsample:** 0.8763648583333697
- **colsample_bytree:** 0.8900789980930954
- **colsample_bylevel:** 0.6134343539737888
- **reg_alpha:** 0.0004686094248233639
- **reg_lambda:** 1.2167282577786425e-05
- **min_child_weight:** 3
- **gamma:** 0.011789862047242498

- **训练时间:** 18.25 秒

---

## 1. Experimental Setup

### 1.1 Dataset Information

| Parameter | Value |
|-----------|-------|
| Data File | `http://47.99.180.80/file/uploads/data_g_for_Dynasty_1.xlsx` |
| Data Shape | {'n_samples': 136, 'n_features': 18} |
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
| Critical Issues | 38 | 0 | Data integrity problems |
| Warnings | 0 | <5 | Minor data quality concerns |

#### 2.4.2 Validation Methodology and Results

| Check Name | Method Used | Status | Issues Found | Key Findings |
|------------|-------------|--------|-------------|-------------|
| Feature Names | Statistical Analysis | ✅ PASSED | 0 | No issues |
| Data Dimensions | Statistical Analysis | ✅ PASSED | 0 | No issues |
| Target Variable | Statistical Analysis | ✅ PASSED | 0 | No issues |
| Data Leakage | Statistical Analysis | ❌ FAILED | 8 | 8 issues found |
| Sample Balance | Chi-square, Gini coefficient | ✅ PASSED | 0 | Balanced; ratio=0.294 |
| Feature Correlations | Pearson/Spearman/Kendall | ✅ PASSED | 1 | 3 high correlations |
| Multicollinearity Detection | Variance Inflation Factor (VIF) | ❌ FAILED | 2 | 9 high VIF; avg=300.88 |
| Feature Distributions | Shapiro-Wilk, Jarque-Bera, D'Agostino | ❌ FAILED | 28 | 28 distribution issues |


#### 2.4.2.4 Sample Balance Analysis

**Methodology**: Chi-square goodness-of-fit test and Gini coefficient calculation for class distribution assessment.

**Results**:
- Minority class ratio: 0.2941
- Dataset balance: Balanced
- Number of classes: 3

**Class Distribution**:
| Class | Count | Proportion | Cumulative % |
|-------|-------|------------|-------------|
| Yuan | 0.39705882352941174 | 0.3971 | 39.7% |
| Ming | 0.3088235294117647 | 0.3088 | 70.6% |
| Qing | 0.29411764705882354 | 0.2941 | 100.0% |


**Methodological Implications**: Balanced class distribution supports unbiased model training and reliable performance metrics.

#### 2.4.2.1 Feature Correlation Analysis

**Methodology**: Pearson, Spearman, and Kendall correlation coefficients were computed for all feature pairs. The correlation threshold was set at |r| ≥ 0.7.

**Results**: 3 feature pairs exceeded the correlation threshold, indicating potential redundancy in the feature space.

**Feature Classification**:
Continuous Features: Na2O, MgO, Al2O3, SiO2, K2O, CaO, TiO2, Fe2O3, MnO, CuO, ZnO, PbO2, Rb2O, SrO, Y2O3, ZrO2, P2O5
Categorical Features: As2O3
Target Feature: Dynasty

**Statistical Findings**:
**Continuous Features vs Continuous Features Correlation Analysis (Pearson Correlation Coefficient)**:

| Feature 1 | Feature 2 | Correlation | Absolute Value |
|-----------|-----------|-------------|----------------|
| SiO2 | CaO | -0.7914 | 0.7914 |
| K2O | Rb2O | 0.7871 | 0.7871 |
| CaO | Rb2O | -0.7640 | 0.7640 |
| K2O | CaO | -0.6531 | 0.6531 |
| Al2O3 | SiO2 | -0.6362 | 0.6362 |
| SiO2 | P2O5 | -0.6180 | 0.6180 |
| TiO2 | MnO | 0.5450 | 0.5450 |
| SiO2 | Rb2O | 0.5414 | 0.5414 |
| Al2O3 | Y2O3 | 0.5198 | 0.5198 |
| MnO | P2O5 | 0.5142 | 0.5142 |


**Continuous Features vs Continuous Features Correlation Analysis (Spearman's Rank Correlation)**:

| Feature 1 | Feature 2 | Correlation | Absolute Value |
|-----------|-----------|-------------|----------------|
| SiO2 | CaO | -0.8177 | 0.8177 |
| K2O | Rb2O | 0.7952 | 0.7952 |
| CaO | Rb2O | -0.7648 | 0.7648 |
| SiO2 | P2O5 | -0.6740 | 0.6740 |
| K2O | CaO | -0.6567 | 0.6567 |
| Fe2O3 | SrO | -0.6421 | 0.6421 |
| MgO | SiO2 | -0.6338 | 0.6338 |
| MgO | P2O5 | 0.6084 | 0.6084 |
| Al2O3 | SiO2 | -0.6010 | 0.6010 |
| CaO | P2O5 | 0.5937 | 0.5937 |


**Continuous Features vs Categorical Features Correlation Analysis (Correlation Ratio)**:

| Categorical Feature | Continuous Feature | Correlation Ratio | Absolute Value | Strength |
|-------------------|-------------------|-------------------|----------------|----------|
| As2O3 | PbO2 | 0.0908 | 0.0908 | Medium effect (Moderate association) |
| As2O3 | ZnO | 0.0384 | 0.0384 | Small effect (Weak association) |
| As2O3 | ZrO2 | 0.0336 | 0.0336 | Small effect (Weak association) |
| As2O3 | Rb2O | 0.0323 | 0.0323 | Small effect (Weak association) |
| As2O3 | Na2O | 0.0280 | 0.0280 | Small effect (Weak association) |
| As2O3 | SiO2 | 0.0261 | 0.0261 | Small effect (Weak association) |
| As2O3 | TiO2 | 0.0245 | 0.0245 | Small effect (Weak association) |
| As2O3 | CaO | 0.0238 | 0.0238 | Small effect (Weak association) |
| As2O3 | MnO | 0.0202 | 0.0202 | Small effect (Weak association) |
| As2O3 | MgO | 0.0173 | 0.0173 | Small effect (Weak association) |
| As2O3 | Y2O3 | 0.0166 | 0.0166 | Small effect (Weak association) |
| As2O3 | P2O5 | 0.0159 | 0.0159 | Small effect (Weak association) |
| As2O3 | SrO | 0.0136 | 0.0136 | Small effect (Weak association) |
| As2O3 | CuO | 0.0094 | 0.0094 | Negligible |
| As2O3 | Al2O3 | 0.0074 | 0.0074 | Negligible |
| As2O3 | Fe2O3 | 0.0040 | 0.0040 | Negligible |
| As2O3 | K2O | 0.0036 | 0.0036 | Negligible |


**Continuous Features vs Target Variable Correlation Analysis**:

| Feature | Correlation | Method | Absolute Value | Strength |
|---------|-------------|--------|----------------|----------|
| K2O | 0.5451 | correlation_ratio | 0.5451 | Moderate |
| MnO | 0.4689 | correlation_ratio | 0.4689 | Weak |
| CaO | 0.4568 | correlation_ratio | 0.4568 | Weak |
| Rb2O | 0.3929 | correlation_ratio | 0.3929 | Weak |
| SrO | 0.3558 | correlation_ratio | 0.3558 | Weak |
| SiO2 | 0.3235 | correlation_ratio | 0.3235 | Weak |
| Fe2O3 | 0.2796 | correlation_ratio | 0.2796 | Very Weak |
| P2O5 | 0.2754 | correlation_ratio | 0.2754 | Very Weak |
| TiO2 | 0.2446 | correlation_ratio | 0.2446 | Very Weak |
| ZrO2 | 0.1253 | correlation_ratio | 0.1253 | Very Weak |
| PbO2 | 0.1048 | correlation_ratio | 0.1048 | Very Weak |
| Y2O3 | 0.1042 | correlation_ratio | 0.1042 | Very Weak |
| ZnO | 0.0452 | correlation_ratio | 0.0452 | Very Weak |
| Na2O | 0.0359 | correlation_ratio | 0.0359 | Very Weak |
| MgO | 0.0285 | correlation_ratio | 0.0285 | Very Weak |
| Al2O3 | 0.0252 | correlation_ratio | 0.0252 | Very Weak |
| CuO | 0.0106 | correlation_ratio | 0.0106 | Very Weak |


**Categorical Features vs Target Variable Correlation Analysis**:

| Categorical Feature | Association | Method | Absolute Value | Strength |
|------------------- |-------------|--------|----------------|----------|
| As2O3 | 0.1343 | cramers_v | 0.1343 | Very Weak |


**Impact Assessment**: High feature correlation may lead to multicollinearity issues and reduced model interpretability.

#### 2.4.2.2 Multicollinearity Detection

**Methodology**: Variance Inflation Factor (VIF) analysis was conducted using linear regression. VIF values ≥ 5.0 indicate problematic multicollinearity.

**Results**: 
- Average VIF: 300.882
- Maximum VIF: 1000.000
- Features with VIF ≥ 5.0: 9

**Statistical Findings**:
**VIF Scores for All Features**:

| Feature | VIF Score | R² | Interpretation | Status |
|---------|-----------|----|--------------|---------|
| MgO | 1000.0000 | 0.9990 | Severe | ⚠️ HIGH |
| Al2O3 | 1000.0000 | 0.9990 | Severe | ⚠️ HIGH |
| SiO2 | 1000.0000 | 0.9990 | Severe | ⚠️ HIGH |
| K2O | 1000.0000 | 0.9990 | Severe | ⚠️ HIGH |
| CaO | 1000.0000 | 0.9990 | Severe | ⚠️ HIGH |
| Na2O | 227.9260 | 0.9956 | Severe | ⚠️ HIGH |
| Fe2O3 | 127.2795 | 0.9921 | Severe | ⚠️ HIGH |
| As2O3 | 35.8186 | 0.9721 | Severe | ⚠️ HIGH |
| Rb2O | 6.0672 | 0.8352 | Moderate | ⚠️ MODERATE |
| TiO2 | 3.2078 | 0.6883 | Acceptable | ✅ LOW |
| P2O5 | 2.7931 | 0.6420 | Acceptable | ✅ LOW |
| MnO | 2.5999 | 0.6154 | Acceptable | ✅ LOW |
| SrO | 2.4518 | 0.5921 | Acceptable | ✅ LOW |
| Y2O3 | 1.8387 | 0.4561 | Acceptable | ✅ LOW |
| PbO2 | 1.6701 | 0.4012 | Acceptable | ✅ LOW |
| ZrO2 | 1.5521 | 0.3557 | Acceptable | ✅ LOW |
| ZnO | 1.4466 | 0.3087 | Acceptable | ✅ LOW |
| CuO | 1.2323 | 0.1885 | Acceptable | ✅ LOW |


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
| Al2O3 | 14.345 | 2.022 | 9.967 | 20.100 | 20.100 | 14.287 | 0.243 | 0.103 | Yes | 1.5% | 0 |
| CaO | 7.479 | 3.621 | 1.116 | 17.361 | 17.361 | 7.048 | 0.525 | -0.638 | No | 0.0% | 1 |
| CuO | 27.206 | 18.567 | 0.000 | 80.000 | 80.000 | 30.000 | 0.315 | -0.312 | No | 0.0% | 0 |
| Fe2O3 | 0.870 | 0.215 | 0.589 | 1.852 | 1.852 | 0.814 | 1.872 | 4.614 | No | 5.1% | 3 |
| K2O | 5.006 | 1.427 | 2.206 | 8.240 | 8.240 | 4.815 | 0.214 | -0.785 | No | 0.0% | 1 |
| MgO | 0.655 | 0.668 | 0.000 | 7.046 | 7.046 | 0.576 | 6.569 | 59.850 | No | 5.1% | 3 |
| MnO | 2046.765 | 1147.161 | 360.000 | 6870.000 | 6870.000 | 1975.000 | 0.946 | 1.559 | No | 2.2% | 1 |
| Na2O | 0.385 | 0.302 | 0.062 | 2.015 | 2.015 | 0.316 | 2.470 | 8.779 | No | 3.7% | 2 |
| P2O5 | 818.676 | 457.192 | 60.000 | 3010.000 | 3010.000 | 745.000 | 1.297 | 3.404 | No | 3.7% | 2 |
| PbO2 | 57.353 | 33.472 | 0.000 | 160.000 | 160.000 | 50.000 | 0.919 | 0.666 | No | 5.9% | 2 |
| Rb2O | 251.103 | 47.527 | 130.000 | 380.000 | 380.000 | 250.000 | -0.037 | -0.157 | Yes | 0.0% | 0 |
| SiO2 | 70.212 | 3.865 | 62.446 | 80.082 | 80.082 | 69.537 | 0.520 | -0.295 | No | 0.7% | 1 |
| SrO | 201.250 | 77.090 | 40.000 | 450.000 | 450.000 | 195.000 | 0.562 | 0.365 | No | 2.2% | 1 |
| TiO2 | 0.054 | 0.020 | 0.021 | 0.116 | 0.116 | 0.052 | 0.959 | 0.676 | No | 7.4% | 2 |
| Y2O3 | 132.132 | 84.067 | 30.000 | 530.000 | 530.000 | 110.000 | 2.635 | 7.945 | No | 8.1% | 3 |
| ZnO | 150.147 | 102.241 | 60.000 | 760.000 | 760.000 | 130.000 | 3.535 | 15.319 | No | 5.9% | 3 |
| ZrO2 | 151.838 | 48.338 | 80.000 | 370.000 | 370.000 | 140.000 | 2.358 | 7.270 | No | 4.4% | 2 |

**Continuous Feature Distribution Issues:**
- Feature 'Na2O' has extreme skewness (2.47)
- Feature 'Na2O' significantly deviates from normal distribution
- Feature 'MgO' has extreme skewness (6.57)
- Feature 'MgO' has moderate outlier ratio (5.1%)
- Feature 'MgO' significantly deviates from normal distribution
- Feature 'SiO2' significantly deviates from normal distribution
- Feature 'K2O' significantly deviates from normal distribution
- Feature 'CaO' significantly deviates from normal distribution
- Feature 'TiO2' has moderate outlier ratio (7.4%)
- Feature 'TiO2' significantly deviates from normal distribution
- Feature 'Fe2O3' is highly skewed (1.87)
- Feature 'Fe2O3' has moderate outlier ratio (5.1%)
- Feature 'Fe2O3' significantly deviates from normal distribution
- Feature 'MnO' significantly deviates from normal distribution
- Feature 'ZnO' has extreme skewness (3.54)
- Feature 'ZnO' has moderate outlier ratio (5.9%)
- Feature 'ZnO' significantly deviates from normal distribution
- Feature 'PbO2' has moderate outlier ratio (5.9%)
- Feature 'PbO2' significantly deviates from normal distribution
- Feature 'SrO' significantly deviates from normal distribution
- Feature 'Y2O3' has extreme skewness (2.64)
- Feature 'Y2O3' has moderate outlier ratio (8.1%)
- Feature 'Y2O3' significantly deviates from normal distribution
- Feature 'ZrO2' has extreme skewness (2.36)
- Feature 'ZrO2' significantly deviates from normal distribution
- Feature 'P2O5' is highly skewed (1.30)
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
- High suspicion data leakage: 'MgO' (correlation: 1.000)
- High suspicion data leakage: 'Al2O3' (correlation: 1.000)
- High suspicion data leakage: 'SiO2' (correlation: 1.000)
- High suspicion data leakage: 'K2O' (correlation: 1.000)
- High suspicion data leakage: 'CaO' (correlation: 1.000)
- High suspicion data leakage: 'TiO2' (correlation: 1.000)
- High suspicion data leakage: 'Fe2O3' (correlation: 1.000)
- Detected 9 features with high VIF (>= 5.0)
- Found 3 highly correlated feature pairs
- Feature 'Na2O' has extreme skewness (2.47)
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
**Best Score**: 0.925044

**Optimization Strategy:**
- **Initial Exploration**: 10 random trials for space exploration
- **Exploitation-Exploration Balance**: TPE algorithm balances promising regions with unexplored space
- **Cross-Validation**: Each trial evaluated using stratified k-fold cross-validation
- **Early Stopping**: Poor-performing trials terminated early to improve efficiency

### 3.3 Best Parameters Found

```json
{
  "n_estimators": 90,
  "max_depth": 9,
  "learning_rate": 0.1062272193666865,
  "subsample": 0.8763648583333697,
  "colsample_bytree": 0.8900789980930954,
  "colsample_bylevel": 0.6134343539737888,
  "reg_alpha": 0.0004686094248233639,
  "reg_lambda": 1.2167282577786425e-05,
  "min_child_weight": 3,
  "gamma": 0.011789862047242498
}
```

### 3.4 Optimization Convergence

The optimization process completed **50 trials** with the best configuration achieving a cross-validation score of **0.925044**.

**Key Optimization Insights:**
- **Ensemble Size**: 90 boosting rounds balances performance and computational efficiency
- **Tree Complexity**: Maximum depth of 9 controls model complexity and overfitting
- **Learning Rate**: 0.1062272193666865 provides optimal step size for gradient descent
- **Regularization**: L1=4.69e-04, L2=1.22e-05 prevent overfitting
- **Sampling**: 0.8763648583333697 row sampling and 0.8900789980930954 column sampling for robustness

## 4. Final Model Training

### 4.1 Cross-Validation Training

The final model was trained using 5-fold cross-validation with optimized hyperparameters. Training metrics and validation results were recorded comprehensively.

### 4.2 Training Results

| Metric | Value |
|--------|-------|
### Cross-Validation Performance Metrics

| Metric | Mean ± Std | Min | Max |
|--------|------------|-----|-----|
| ACCURACY | 0.926455 ± 0.052389 | 0.851852 | 1.000000 |
| F1 | 0.925044 ± 0.054033 | 0.849237 | 1.000000 |
| PRECISION | 0.934512 ± 0.050744 | 0.850676 | 1.000000 |
| RECALL | 0.926455 ± 0.052389 | 0.851852 | 1.000000 |



#### Fold-wise Results

#### Detailed Fold-wise Performance

| Fold | ACCURACY | F1 | PRECISION | RECALL |
|------|---------|---------|---------|---------|
| 1 | 0.928571 | 0.930639 | 0.942857 | 0.928571 |
| 2 | 0.888889 | 0.882735 | 0.912698 | 0.888889 |
| 3 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 4 | 0.851852 | 0.849237 | 0.850676 | 0.851852 |
| 5 | 0.962963 | 0.962610 | 0.966330 | 0.962963 |

#### Statistical Summary

| Metric | Mean | Std Dev | Min | Max | 95% CI |
|--------|------|---------|-----|-----|--------|
| ACCURACY | 0.926455 | 0.052389 | 0.851852 | 1.000000 | [0.880534, 0.972376] |
| F1 | 0.925044 | 0.054033 | 0.849237 | 1.000000 | [0.877683, 0.972406] |
| PRECISION | 0.934512 | 0.050744 | 0.850676 | 1.000000 | [0.890033, 0.978992] |
| RECALL | 0.926455 | 0.052389 | 0.851852 | 1.000000 | [0.880534, 0.972376] |

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
| 1 | `K2O` | 4.1651 | 62 | 19.03 | 15.1% | 9.4% |
| 2 | `MnO` | 3.6134 | 85 | 17.25 | 13.1% | 12.9% |
| 3 | `Rb2O` | 3.3218 | 32 | 22.38 | 12.0% | 4.8% |
| 4 | `CaO` | 2.8202 | 45 | 20.73 | 10.2% | 6.8% |
| 5 | `SrO` | 1.5419 | 63 | 15.84 | 5.6% | 9.5% |
| 6 | `ZrO2` | 1.5091 | 14 | 14.39 | 5.5% | 2.1% |
| 7 | `P2O5` | 1.4758 | 47 | 13.21 | 5.3% | 7.1% |
| 8 | `MgO` | 1.3097 | 34 | 11.32 | 4.7% | 5.2% |
| 9 | `PbO2` | 1.1612 | 21 | 13.65 | 4.2% | 3.2% |
| 10 | `TiO2` | 1.0696 | 52 | 13.97 | 3.9% | 7.9% |
| 11 | `As2O3` | 0.9447 | 6 | 12.98 | 3.4% | 0.9% |
| 12 | `Y2O3` | 0.9385 | 34 | 12.76 | 3.4% | 5.2% |
| 13 | `Fe2O3` | 0.8884 | 71 | 13.77 | 3.2% | 10.8% |
| 14 | `SiO2` | 0.6608 | 23 | 13.38 | 2.4% | 3.5% |
| 15 | `CuO` | 0.6206 | 7 | 12.91 | 2.2% | 1.1% |
| 16 | `Al2O3` | 0.6092 | 26 | 11.78 | 2.2% | 3.9% |
| 17 | `Na2O` | 0.4895 | 20 | 13.53 | 1.8% | 3.0% |
| 18 | `ZnO` | 0.4674 | 18 | 11.02 | 1.7% | 2.7% |


**Permutation Feature Importance:**

| Rank | Feature | Mean Importance | Std Dev | 95% CI | Reliability |
|------|---------|-----------------|---------|--------|-------------|
| 1 | `K2O` | 0.1471 | 0.0275 | [0.0931, 0.2010] | 🟡 Medium |
| 2 | `MnO` | 0.1103 | 0.0104 | [0.0899, 0.1307] | 🟢 High |
| 3 | `Fe2O3` | 0.0088 | 0.0029 | [0.0031, 0.0146] | 🔴 Low |
| 4 | `TiO2` | 0.0074 | 0.0047 | [-0.0018, 0.0165] | 🔴 Low |
| 5 | `CuO` | 0.0074 | 0.0000 | [0.0074, 0.0074] | 🟢 High |
| 6 | `SrO` | 0.0074 | 0.0000 | [0.0074, 0.0074] | 🟢 High |
| 7 | `MgO` | 0.0059 | 0.0029 | [0.0001, 0.0116] | 🔴 Low |
| 8 | `Al2O3` | 0.0044 | 0.0036 | [-0.0026, 0.0115] | 🔴 Low |
| 9 | `Y2O3` | 0.0029 | 0.0036 | [-0.0041, 0.0100] | 🔴 Low |
| 10 | `P2O5` | 0.0029 | 0.0036 | [-0.0041, 0.0100] | 🔴 Low |
| 11 | `Na2O` | 0.0000 | 0.0000 | [0.0000, 0.0000] | 🔴 Low |
| 12 | `SiO2` | 0.0000 | 0.0000 | [0.0000, 0.0000] | 🔴 Low |
| 13 | `As2O3` | 0.0000 | 0.0000 | [0.0000, 0.0000] | 🔴 Low |
| 14 | `PbO2` | 0.0000 | 0.0047 | [-0.0091, 0.0091] | 🔴 Low |
| 15 | `ZrO2` | 0.0000 | 0.0000 | [0.0000, 0.0000] | 🔴 Low |
| 16 | `ZnO` | -0.0015 | 0.0029 | [-0.0072, 0.0043] | 🔴 Low |
| 17 | `CaO` | -0.0029 | 0.0036 | [-0.0100, 0.0041] | 🔴 Low |
| 18 | `Rb2O` | -0.0029 | 0.0036 | [-0.0100, 0.0041] | 🔴 Low |


**Feature Importance Method Comparison:**

| Feature | XGB Gain Rank | Permutation Rank | Rank Difference | Consistency |
|---------|---------------|------------------|-----------------|-------------|
| `Na2O` | 17 | 11 | 6 | 🔴 Poor |
| `MgO` | 8 | 7 | 1 | 🟢 Excellent |
| `Al2O3` | 16 | 8 | 8 | 🔴 Poor |
| `SiO2` | 14 | 12 | 2 | 🟡 Good |
| `K2O` | 1 | 1 | 0 | 🟢 Excellent |
| `CaO` | 4 | 17 | 13 | 🔴 Poor |
| `TiO2` | 10 | 4 | 6 | 🔴 Poor |
| `Fe2O3` | 13 | 3 | 10 | 🔴 Poor |
| `As2O3` | 11 | 13 | 2 | 🟡 Good |
| `MnO` | 2 | 2 | 0 | 🟢 Excellent |
| `CuO` | 15 | 5 | 10 | 🔴 Poor |
| `ZnO` | 18 | 16 | 2 | 🟡 Good |
| `PbO2` | 9 | 14 | 5 | 🔴 Poor |
| `Rb2O` | 3 | 18 | 15 | 🔴 Poor |
| `SrO` | 5 | 6 | 1 | 🟢 Excellent |
| `Y2O3` | 12 | 9 | 3 | 🔴 Poor |
| `ZrO2` | 6 | 15 | 9 | 🔴 Poor |
| `P2O5` | 7 | 10 | 3 | 🔴 Poor |


**Statistical Summary:**

- **Total Features Analyzed**: 18
- **Gain-based Top Feature**: `K2O` (Gain: 4.1651)
- **Permutation-based Top Feature**: `K2O` (Importance: 0.1471)

**Method Reliability Assessment:**
- **Average Permutation Std**: 0.0041
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
- **Cross-Validation Data**: `trained_models/10004/916339bd-2069-4d41-b0be-a7eb58f98b1f/cross_validation_data/`
- **Feature Importance**: `trained_models/10004/916339bd-2069-4d41-b0be-a7eb58f98b1f/feature_importance.csv`

### 6.3 Technical Implementation

- **Framework**: XGBoost for gradient boosting implementation, scikit-learn for pipeline integration.
- **Data Processing**: pandas and numpy for data handling.
- **Cross-Validation**: K-fold cross-validation with stratification support for classification.
- **Feature Importance**: Built-in XGBoost feature importance calculation (Gain, Cover, Weight).
- **Serialization**: Joblib or Pickle for model and preprocessor persistence.

---

## Appendix

### A.1 System Information

- **Generation Time**: 2025-12-26 01:31:57
- **Model ID**: `916339bd-2069-4d41-b0be-a7eb58f98b1f`
- **Training System**: XGBoost MCP Tool
- **Report Version**: 2.1 (XGBoost Enhanced)

### A.2 File Structure

```
916339bd-2069-4d41-b0be-a7eb58f98b1f/
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
│   ├── 916339bd-2069-4d41-b0be-a7eb58f98b1f_cv_predictions_original.csv
│   ├── 916339bd-2069-4d41-b0be-a7eb58f98b1f_cv_predictions_processed.csv
│   ├── 916339bd-2069-4d41-b0be-a7eb58f98b1f_original_data.csv
│   ├── 916339bd-2069-4d41-b0be-a7eb58f98b1f_preprocessed_data.csv
│   ├── 916339bd-2069-4d41-b0be-a7eb58f98b1f_roc_curves.png
│   ├── cross_validation_roc_curves.png
│   ├── cross_validation_visualization.png
└── academic_report.md               # This report
```

### A.3 Data Files and JSON Artifacts

The following JSON files contain detailed intermediate data for reproducibility:

- **Feature Importance**: `trained_models/10004/916339bd-2069-4d41-b0be-a7eb58f98b1f/feature_importance.csv`

---

*This report was automatically generated by the Enhanced XGBoost MCP Tool for academic research and reproducibility purposes.*
