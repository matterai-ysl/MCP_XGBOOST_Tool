# XGBoost Training Report

**Generated on:** 2025-12-26 13:01:43  
**Model ID:** `6d4097cc-a65c-4c53-9307-aa789e82a318`  
**Model Folder:** `trained_models/10004/6d4097cc-a65c-4c53-9307-aa789e82a318`

## Executive Summary

This report documents a comprehensive XGBoost training experiment conducted for academic research and reproducibility purposes. The experiment involved hyperparameter optimization and cross-validated model training with detailed performance analysis, data validation, and feature importance evaluation.

### Key Results
### 🎯 关键性能指标

- **准确率 (Accuracy):** 0.762963 (±0.046934)
- **F1分数 (F1 Score):** 0.756252 (±0.056473)
- **精确率 (Precision):** 0.784464 (±0.051891)
- **召回率 (Recall):** 0.762963 (±0.046934)

- **交叉验证折数:** 5
- **数据集规模:** 139 样本, 26 特征

### ⚙️ 最优超参数

- **n_estimators:** 110
- **max_depth:** 5
- **learning_rate:** 0.047428446181337715
- **subsample:** 0.9072988320968841
- **colsample_bytree:** 0.9845801489448449
- **colsample_bylevel:** 0.8631513530923792
- **reg_alpha:** 0.003953129596044868
- **reg_lambda:** 0.0030462424843672458
- **min_child_weight:** 9
- **gamma:** 0.08271639849598074

- **训练时间:** 19.84 秒

---

## 1. Experimental Setup

### 1.1 Dataset Information

| Parameter | Value |
|-----------|-------|
| Data File | `http://47.99.180.80/file/uploads/data_b_for_kiln_openfe_1.csv` |
| Data Shape | {'n_samples': 139, 'n_features': 26} |
| Number of Features | 26 |
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
`Na2O`, `MgO`, `Al2O3`, `SiO2`, `K2O`, `Fe2O3`, `As2O3`, `MnO`, `CuO`, `ZnO`, `PbO2`, `Rb2O`, `SrO`, `Y2O3`, `ZrO2`, `P2O5`, `autoFE_f_0`, `autoFE_f_1`, `autoFE_f_2`, `autoFE_f_3`, `autoFE_f_4`, `autoFE_f_5`, `autoFE_f_6`, `autoFE_f_7`, `autoFE_f_8`, `autoFE_f_9`

**Target Variables (1 column):**
`kiln`


### 2.4 Data Quality Assessment

Comprehensive data validation was performed using multiple statistical methods to ensure dataset quality and suitability for machine learning model training. The validation framework employed established statistical techniques for thorough data quality assessment.

#### 2.4.1 Overall Quality Metrics

| Metric | Value | Threshold | Interpretation |
|--------|-------|-----------|----------------|
| Overall Data Quality Score | 0/100 | ≥80 (Excellent), ≥60 (Good) | Poor - Significant issues require resolution |
| Quality Level | Poor | - | Categorical assessment |
| Ready for Training | No | Yes | Model training readiness |
| Critical Issues | 53 | 0 | Data integrity problems |
| Warnings | 0 | <5 | Minor data quality concerns |

#### 2.4.2 Validation Methodology and Results

| Check Name | Method Used | Status | Issues Found | Key Findings |
|------------|-------------|--------|-------------|-------------|
| Feature Names | Statistical Analysis | ✅ PASSED | 0 | No issues |
| Data Dimensions | Statistical Analysis | ✅ PASSED | 0 | No issues |
| Target Variable | Statistical Analysis | ✅ PASSED | 0 | No issues |
| Data Leakage | Statistical Analysis | ❌ FAILED | 9 | 9 issues found |
| Sample Balance | Chi-square, Gini coefficient | ✅ PASSED | 1 | Balanced; ratio=0.129 |
| Feature Correlations | Pearson/Spearman/Kendall | ❌ FAILED | 2 | 17 high correlations |
| Multicollinearity Detection | Variance Inflation Factor (VIF) | ❌ FAILED | 2 | 17 high VIF; avg=307.97 |
| Feature Distributions | Shapiro-Wilk, Jarque-Bera, D'Agostino | ❌ FAILED | 40 | 40 distribution issues |


#### 2.4.2.4 Sample Balance Analysis

**Methodology**: Chi-square goodness-of-fit test and Gini coefficient calculation for class distribution assessment.

**Results**:
- Minority class ratio: 0.1295
- Dataset balance: Balanced
- Number of classes: 5

**Class Distribution**:
| Class | Count | Proportion | Cumulative % |
|-------|-------|------------|-------------|
| ZLG | 0.28776978417266186 | 0.2878 | 28.8% |
| JHS | 0.28776978417266186 | 0.2878 | 57.6% |
| WDP | 0.1510791366906475 | 0.1511 | 72.7% |
| XJ | 0.14388489208633093 | 0.1439 | 87.1% |
| QDG | 0.12949640287769784 | 0.1295 | 100.0% |


**Methodological Implications**: Balanced class distribution supports unbiased model training and reliable performance metrics.

#### 2.4.2.1 Feature Correlation Analysis

**Methodology**: Pearson, Spearman, and Kendall correlation coefficients were computed for all feature pairs. The correlation threshold was set at |r| ≥ 0.7.

**Results**: 17 feature pairs exceeded the correlation threshold, indicating potential redundancy in the feature space.

**Feature Classification**:
Continuous Features: Na2O, MgO, Al2O3, SiO2, K2O, Fe2O3, MnO, CuO, ZnO, PbO2, Rb2O, SrO, Y2O3, ZrO2, P2O5, autoFE_f_0, autoFE_f_1, autoFE_f_2, autoFE_f_3, autoFE_f_4, autoFE_f_5, autoFE_f_6, autoFE_f_7, autoFE_f_8, autoFE_f_9
Categorical Features: As2O3
Target Feature: kiln

**Statistical Findings**:
**Continuous Features vs Continuous Features Correlation Analysis (Pearson Correlation Coefficient)**:

| Feature 1 | Feature 2 | Correlation | Absolute Value |
|-----------|-----------|-------------|----------------|
| Fe2O3 | autoFE_f_0 | -0.9058 | 0.9058 |
| Fe2O3 | autoFE_f_9 | 0.8954 | 0.8954 |
| PbO2 | autoFE_f_8 | 0.8669 | 0.8669 |
| autoFE_f_1 | autoFE_f_7 | 0.8380 | 0.8380 |
| MnO | autoFE_f_1 | -0.8261 | 0.8261 |
| Al2O3 | SiO2 | -0.8174 | 0.8174 |
| autoFE_f_0 | autoFE_f_9 | -0.8164 | 0.8164 |
| ZnO | autoFE_f_5 | 0.7714 | 0.7714 |
| autoFE_f_0 | autoFE_f_2 | -0.7693 | 0.7693 |
| K2O | SrO | 0.7515 | 0.7515 |


**Continuous Features vs Continuous Features Correlation Analysis (Spearman's Rank Correlation)**:

| Feature 1 | Feature 2 | Correlation | Absolute Value |
|-----------|-----------|-------------|----------------|
| MnO | autoFE_f_1 | -0.9832 | 0.9832 |
| Fe2O3 | autoFE_f_0 | -0.9451 | 0.9451 |
| MnO | autoFE_f_7 | -0.8731 | 0.8731 |
| autoFE_f_1 | autoFE_f_7 | 0.8540 | 0.8540 |
| Fe2O3 | autoFE_f_9 | 0.8437 | 0.8437 |
| autoFE_f_0 | autoFE_f_9 | -0.8402 | 0.8402 |
| K2O | autoFE_f_4 | 0.8075 | 0.8075 |
| PbO2 | autoFE_f_8 | 0.7976 | 0.7976 |
| ZnO | autoFE_f_5 | 0.7912 | 0.7912 |
| K2O | Fe2O3 | -0.7787 | 0.7787 |


**Continuous Features vs Categorical Features Correlation Analysis (Correlation Ratio)**:

| Categorical Feature | Continuous Feature | Correlation Ratio | Absolute Value | Strength |
|-------------------|-------------------|-------------------|----------------|----------|
| As2O3 | PbO2 | 0.0636 | 0.0636 | Medium effect (Moderate association) |
| As2O3 | autoFE_f_8 | 0.0449 | 0.0449 | Small effect (Weak association) |
| As2O3 | ZnO | 0.0385 | 0.0385 | Small effect (Weak association) |
| As2O3 | autoFE_f_5 | 0.0359 | 0.0359 | Small effect (Weak association) |
| As2O3 | Y2O3 | 0.0312 | 0.0312 | Small effect (Weak association) |
| As2O3 | SiO2 | 0.0310 | 0.0310 | Small effect (Weak association) |
| As2O3 | autoFE_f_0 | 0.0305 | 0.0305 | Small effect (Weak association) |
| As2O3 | Al2O3 | 0.0237 | 0.0237 | Small effect (Weak association) |
| As2O3 | SrO | 0.0229 | 0.0229 | Small effect (Weak association) |
| As2O3 | autoFE_f_1 | 0.0229 | 0.0229 | Small effect (Weak association) |
| As2O3 | MnO | 0.0189 | 0.0189 | Small effect (Weak association) |
| As2O3 | autoFE_f_9 | 0.0150 | 0.0150 | Small effect (Weak association) |
| As2O3 | Fe2O3 | 0.0137 | 0.0137 | Small effect (Weak association) |
| As2O3 | autoFE_f_4 | 0.0136 | 0.0136 | Small effect (Weak association) |
| As2O3 | CuO | 0.0129 | 0.0129 | Small effect (Weak association) |
| As2O3 | autoFE_f_3 | 0.0117 | 0.0117 | Small effect (Weak association) |
| As2O3 | autoFE_f_6 | 0.0092 | 0.0092 | Negligible |
| As2O3 | MgO | 0.0091 | 0.0091 | Negligible |
| As2O3 | autoFE_f_2 | 0.0081 | 0.0081 | Negligible |
| As2O3 | K2O | 0.0077 | 0.0077 | Negligible |


**Continuous Features vs Target Variable Correlation Analysis**:

| Feature | Correlation | Method | Absolute Value | Strength |
|---------|-------------|--------|----------------|----------|
| autoFE_f_0 | 0.6204 | correlation_ratio | 0.6204 | Moderate |
| autoFE_f_7 | 0.5907 | correlation_ratio | 0.5907 | Moderate |
| Fe2O3 | 0.4973 | correlation_ratio | 0.4973 | Weak |
| ZrO2 | 0.4924 | correlation_ratio | 0.4924 | Weak |
| autoFE_f_5 | 0.4902 | correlation_ratio | 0.4902 | Weak |
| Y2O3 | 0.4468 | correlation_ratio | 0.4468 | Weak |
| autoFE_f_9 | 0.4400 | correlation_ratio | 0.4400 | Weak |
| SiO2 | 0.4221 | correlation_ratio | 0.4221 | Weak |
| autoFE_f_1 | 0.4133 | correlation_ratio | 0.4133 | Weak |
| K2O | 0.3848 | correlation_ratio | 0.3848 | Weak |
| Al2O3 | 0.3739 | correlation_ratio | 0.3739 | Weak |
| autoFE_f_2 | 0.3323 | correlation_ratio | 0.3323 | Weak |
| MnO | 0.3236 | correlation_ratio | 0.3236 | Weak |
| autoFE_f_3 | 0.2784 | correlation_ratio | 0.2784 | Very Weak |
| ZnO | 0.2625 | correlation_ratio | 0.2625 | Very Weak |
| MgO | 0.2511 | correlation_ratio | 0.2511 | Very Weak |
| autoFE_f_4 | 0.2357 | correlation_ratio | 0.2357 | Very Weak |
| SrO | 0.2163 | correlation_ratio | 0.2163 | Very Weak |
| Na2O | 0.2093 | correlation_ratio | 0.2093 | Very Weak |
| Rb2O | 0.1626 | correlation_ratio | 0.1626 | Very Weak |
| autoFE_f_8 | 0.1236 | correlation_ratio | 0.1236 | Very Weak |
| PbO2 | 0.1081 | correlation_ratio | 0.1081 | Very Weak |
| autoFE_f_6 | 0.1068 | correlation_ratio | 0.1068 | Very Weak |
| CuO | 0.0687 | correlation_ratio | 0.0687 | Very Weak |
| P2O5 | 0.0229 | correlation_ratio | 0.0229 | Very Weak |


**Categorical Features vs Target Variable Correlation Analysis**:

| Categorical Feature | Association | Method | Absolute Value | Strength |
|------------------- |-------------|--------|----------------|----------|
| As2O3 | 0.1705 | cramers_v | 0.1705 | Very Weak |


**Impact Assessment**: High feature correlation may lead to multicollinearity issues and reduced model interpretability.

#### 2.4.2.2 Multicollinearity Detection

**Methodology**: Variance Inflation Factor (VIF) analysis was conducted using linear regression. VIF values ≥ 5.0 indicate problematic multicollinearity.

**Results**: 
- Average VIF: 307.967
- Maximum VIF: 1000.000
- Features with VIF ≥ 5.0: 17

**Statistical Findings**:
**VIF Scores for All Features**:

| Feature | VIF Score | R² | Interpretation | Status |
|---------|-----------|----|--------------|---------|
| ZnO | 1000.0000 | 0.9990 | Severe | ⚠️ HIGH |
| PbO2 | 1000.0000 | 0.9990 | Severe | ⚠️ HIGH |
| Rb2O | 1000.0000 | 0.9990 | Severe | ⚠️ HIGH |
| ZrO2 | 1000.0000 | 0.9990 | Severe | ⚠️ HIGH |
| autoFE_f_5 | 1000.0000 | 0.9990 | Severe | ⚠️ HIGH |
| autoFE_f_8 | 1000.0000 | 0.9990 | Severe | ⚠️ HIGH |
| Al2O3 | 590.5064 | 0.9983 | Severe | ⚠️ HIGH |
| SiO2 | 584.6447 | 0.9983 | Severe | ⚠️ HIGH |
| K2O | 277.3733 | 0.9964 | Severe | ⚠️ HIGH |
| Fe2O3 | 213.6970 | 0.9953 | Severe | ⚠️ HIGH |
| autoFE_f_9 | 167.0146 | 0.9940 | Severe | ⚠️ HIGH |
| As2O3 | 34.8448 | 0.9713 | Severe | ⚠️ HIGH |
| autoFE_f_7 | 33.3759 | 0.9700 | Severe | ⚠️ HIGH |
| autoFE_f_0 | 31.9269 | 0.9687 | Severe | ⚠️ HIGH |
| autoFE_f_1 | 25.5940 | 0.9609 | Severe | ⚠️ HIGH |
| Na2O | 17.3092 | 0.9422 | Severe | ⚠️ HIGH |
| MgO | 5.7521 | 0.8261 | Moderate | ⚠️ MODERATE |
| MnO | 4.7338 | 0.7888 | Acceptable | ✅ LOW |
| autoFE_f_2 | 4.3516 | 0.7702 | Acceptable | ✅ LOW |
| SrO | 3.7871 | 0.7359 | Acceptable | ✅ LOW |
| autoFE_f_4 | 2.7903 | 0.6416 | Acceptable | ✅ LOW |
| autoFE_f_3 | 2.4407 | 0.5903 | Acceptable | ✅ LOW |
| CuO | 2.0560 | 0.5136 | Acceptable | ✅ LOW |
| autoFE_f_6 | 1.8518 | 0.4600 | Acceptable | ✅ LOW |
| Y2O3 | 1.7342 | 0.4234 | Acceptable | ✅ LOW |
| P2O5 | 1.3632 | 0.2664 | Acceptable | ✅ LOW |


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
| Y2O3 | 118.489 | 75.833 | 40.000 | 630.000 | 630.000 | 100.000 | 3.351 | 15.978 | No | 8.6% | 3 |
| ZnO | 138.417 | 55.629 | 60.000 | 360.000 | 360.000 | 120.000 | 1.467 | 2.504 | No | 5.8% | 3 |
| ZrO2 | 182.230 | 46.719 | 100.000 | 390.000 | 390.000 | 170.000 | 1.314 | 2.721 | No | 1.4% | 2 |
| autoFE_f_0 | 22.527 | 5.471 | 10.781 | 33.941 | 33.941 | 23.612 | -0.195 | -0.827 | No | 0.0% | 1 |
| autoFE_f_1 | 0.064 | 0.032 | 0.016 | 0.210 | 0.210 | 0.056 | 1.273 | 2.536 | No | 1.4% | 2 |
| autoFE_f_2 | 0.579 | 0.295 | 0.042 | 1.000 | 1.000 | 0.583 | -0.051 | -1.227 | No | 0.0% | 1 |
| autoFE_f_3 | 19.447 | 1.287 | 15.444 | 23.770 | 23.770 | 19.299 | -0.059 | 1.726 | No | 8.6% | 2 |
| autoFE_f_4 | 4.319 | 2.115 | 0.000 | 7.748 | 7.748 | 4.576 | -0.851 | 0.044 | No | 14.4% | 2 |
| autoFE_f_5 | -43.813 | 73.400 | -280.000 | 210.000 | 210.000 | -50.000 | -0.084 | 1.076 | Yes | 2.9% | 0 |
| autoFE_f_6 | 88.129 | 25.752 | 20.000 | 140.000 | 140.000 | 80.000 | 0.571 | -0.183 | No | 0.0% | 1 |
| autoFE_f_7 | 0.003 | 0.002 | 0.001 | 0.011 | 0.011 | 0.002 | 1.976 | 4.651 | No | 10.1% | 3 |
| autoFE_f_8 | -205.540 | 69.960 | -350.000 | 60.000 | 60.000 | -220.000 | 0.909 | 1.213 | No | 3.6% | 1 |
| autoFE_f_9 | 264.156 | 74.516 | 162.264 | 642.332 | 642.332 | 242.593 | 2.046 | 5.744 | No | 7.2% | 3 |

**Continuous Feature Distribution Issues:**
- Feature 'Na2O' is highly skewed (1.37)
- Feature 'Na2O' has moderate outlier ratio (6.5%)
- Feature 'Na2O' significantly deviates from normal distribution
- Feature 'MgO' significantly deviates from normal distribution
- Feature 'K2O' significantly deviates from normal distribution
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
- Feature 'autoFE_f_0' significantly deviates from normal distribution
- Feature 'autoFE_f_1' is highly skewed (1.27)
- Feature 'autoFE_f_1' significantly deviates from normal distribution
- Feature 'autoFE_f_2' significantly deviates from normal distribution
- Feature 'autoFE_f_3' has moderate outlier ratio (8.6%)
- Feature 'autoFE_f_3' significantly deviates from normal distribution
- Feature 'autoFE_f_4' has high outlier ratio (14.4%)
- Feature 'autoFE_f_4' significantly deviates from normal distribution
- Feature 'autoFE_f_6' significantly deviates from normal distribution
- Feature 'autoFE_f_7' is highly skewed (1.98)
- Feature 'autoFE_f_7' has high outlier ratio (10.1%)
- Feature 'autoFE_f_7' significantly deviates from normal distribution
- Feature 'autoFE_f_8' significantly deviates from normal distribution
- Feature 'autoFE_f_9' has extreme skewness (2.05)
- Feature 'autoFE_f_9' has moderate outlier ratio (7.2%)
- Feature 'autoFE_f_9' significantly deviates from normal distribution


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
- Passed checks: 4 (50.0%)
- Failed checks: 4

**Data Quality Confidence**: Based on the comprehensive validation framework, the dataset demonstrates low statistical reliability for machine learning applications.

#### 2.4.3 Data Quality Issues and Impact Assessment

**Critical Issues Identified:**

- High suspicion data leakage: 'Na2O' (correlation: 1.000)
- High suspicion data leakage: 'Al2O3' (correlation: 1.000)
- High suspicion data leakage: 'SiO2' (correlation: 1.000)
- High suspicion data leakage: 'K2O' (correlation: 1.000)
- High suspicion data leakage: 'Fe2O3' (correlation: 1.000)
- High suspicion data leakage: 'autoFE_f_0' (correlation: 1.000)
- High suspicion data leakage: 'autoFE_f_1' (correlation: 1.000)
- High suspicion data leakage: 'autoFE_f_7' (correlation: 1.000)
- High suspicion data leakage: 'autoFE_f_9' (correlation: 1.000)
- Found 17 high feature correlations (threshold: 0.7)
- Excessive multicollinearity detected - consider feature selection
- Detected 17 features with high VIF (>= 5.0)

**Data Quality Recommendations:**

1. Address distribution issues through transformation or preprocessing
2. Resolve multicollinearity using VIF-guided feature selection or regularization
3. Apply balancing techniques (SMOTE, undersampling, class weights)
4. Remove or investigate highly correlated features for potential data leakage
5. Address multicollinearity through feature selection or regularization
6. Investigate high correlations and consider feature selection
7. Consider target transformation for heavily skewed targets


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
**Best Score**: 0.756252

**Optimization Strategy:**
- **Initial Exploration**: 10 random trials for space exploration
- **Exploitation-Exploration Balance**: TPE algorithm balances promising regions with unexplored space
- **Cross-Validation**: Each trial evaluated using stratified k-fold cross-validation
- **Early Stopping**: Poor-performing trials terminated early to improve efficiency

### 3.3 Best Parameters Found

```json
{
  "n_estimators": 110,
  "max_depth": 5,
  "learning_rate": 0.047428446181337715,
  "subsample": 0.9072988320968841,
  "colsample_bytree": 0.9845801489448449,
  "colsample_bylevel": 0.8631513530923792,
  "reg_alpha": 0.003953129596044868,
  "reg_lambda": 0.0030462424843672458,
  "min_child_weight": 9,
  "gamma": 0.08271639849598074
}
```

### 3.4 Optimization Convergence

The optimization process completed **50 trials** with the best configuration achieving a cross-validation score of **0.756252**.

**Key Optimization Insights:**
- **Ensemble Size**: 110 boosting rounds balances performance and computational efficiency
- **Tree Complexity**: Maximum depth of 5 controls model complexity and overfitting
- **Learning Rate**: 0.047428446181337715 provides optimal step size for gradient descent
- **Regularization**: L1=3.95e-03, L2=3.05e-03 prevent overfitting
- **Sampling**: 0.9072988320968841 row sampling and 0.9845801489448449 column sampling for robustness

## 4. Final Model Training

### 4.1 Cross-Validation Training

The final model was trained using 5-fold cross-validation with optimized hyperparameters. Training metrics and validation results were recorded comprehensively.

### 4.2 Training Results

| Metric | Value |
|--------|-------|
### Cross-Validation Performance Metrics

| Metric | Mean ± Std | Min | Max |
|--------|------------|-----|-----|
| ACCURACY | 0.762963 ± 0.046934 | 0.678571 | 0.814815 |
| F1 | 0.756252 ± 0.056473 | 0.649427 | 0.804527 |
| PRECISION | 0.784464 ± 0.051891 | 0.689177 | 0.837037 |
| RECALL | 0.762963 ± 0.046934 | 0.678571 | 0.814815 |



#### Fold-wise Results

#### Detailed Fold-wise Performance

| Fold | ACCURACY | F1 | PRECISION | RECALL |
|------|---------|---------|---------|---------|
| 1 | 0.785714 | 0.785714 | 0.785714 | 0.785714 |
| 2 | 0.750000 | 0.749433 | 0.785714 | 0.750000 |
| 3 | 0.678571 | 0.649427 | 0.689177 | 0.678571 |
| 4 | 0.785714 | 0.792159 | 0.824675 | 0.785714 |
| 5 | 0.814815 | 0.804527 | 0.837037 | 0.814815 |

#### Statistical Summary

| Metric | Mean | Std Dev | Min | Max | 95% CI |
|--------|------|---------|-----|-----|--------|
| ACCURACY | 0.762963 | 0.046934 | 0.678571 | 0.814815 | [0.721824, 0.804102] |
| F1 | 0.756252 | 0.056473 | 0.649427 | 0.804527 | [0.706751, 0.805753] |
| PRECISION | 0.784464 | 0.051891 | 0.689177 | 0.837037 | [0.738979, 0.829948] |
| RECALL | 0.762963 | 0.046934 | 0.678571 | 0.814815 | [0.721824, 0.804102] |

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
| 1 | `Al2O3` | 7.7741 | 47 | 29.18 | 10.7% | 5.8% |
| 2 | `Y2O3` | 6.3078 | 23 | 35.21 | 8.7% | 2.8% |
| 3 | `autoFE_f_5` | 5.7481 | 67 | 29.08 | 7.9% | 8.3% |
| 4 | `autoFE_f_0` | 5.5364 | 76 | 35.11 | 7.6% | 9.4% |
| 5 | `autoFE_f_7` | 5.1545 | 35 | 24.92 | 7.1% | 4.3% |
| 6 | `autoFE_f_1` | 5.0056 | 22 | 26.46 | 6.9% | 2.7% |
| 7 | `autoFE_f_2` | 4.2938 | 63 | 34.23 | 5.9% | 7.8% |
| 8 | `ZrO2` | 3.8141 | 45 | 26.94 | 5.2% | 5.6% |
| 9 | `autoFE_f_9` | 3.0698 | 77 | 33.45 | 4.2% | 9.5% |
| 10 | `autoFE_f_8` | 2.8013 | 56 | 28.91 | 3.9% | 6.9% |
| 11 | `autoFE_f_6` | 2.3861 | 37 | 28.32 | 3.3% | 4.6% |
| 12 | `autoFE_f_3` | 2.3060 | 42 | 29.93 | 3.2% | 5.2% |
| 13 | `SiO2` | 2.0039 | 19 | 26.42 | 2.8% | 2.3% |
| 14 | `Fe2O3` | 1.9316 | 30 | 34.51 | 2.7% | 3.7% |
| 15 | `K2O` | 1.9311 | 25 | 26.53 | 2.7% | 3.1% |
| 16 | `PbO2` | 1.8949 | 28 | 30.12 | 2.6% | 3.5% |
| 17 | `autoFE_f_4` | 1.8058 | 39 | 27.72 | 2.5% | 4.8% |
| 18 | `ZnO` | 1.4633 | 12 | 22.51 | 2.0% | 1.5% |
| 19 | `As2O3` | 1.4368 | 2 | 34.75 | 2.0% | 0.2% |
| 20 | `P2O5` | 1.3586 | 12 | 25.18 | 1.9% | 1.5% |
| 21 | `MgO` | 1.0678 | 25 | 22.44 | 1.5% | 3.1% |
| 22 | `MnO` | 1.0333 | 5 | 21.42 | 1.4% | 0.6% |
| 23 | `Na2O` | 1.0001 | 14 | 27.27 | 1.4% | 1.7% |
| 24 | `SrO` | 0.9237 | 5 | 22.00 | 1.3% | 0.6% |
| 25 | `Rb2O` | 0.7025 | 3 | 23.53 | 1.0% | 0.4% |
| 26 | `CuO` | 0.0000 | 0 | 0.00 | 0.0% | 0.0% |


**Permutation Feature Importance:**

| Rank | Feature | Mean Importance | Std Dev | 95% CI | Reliability |
|------|---------|-----------------|---------|--------|-------------|
| 1 | `autoFE_f_2` | 0.0964 | 0.0283 | [0.0410, 0.1518] | 🟡 Medium |
| 2 | `autoFE_f_0` | 0.0734 | 0.0160 | [0.0420, 0.1048] | 🟡 Medium |
| 3 | `autoFE_f_9` | 0.0590 | 0.0084 | [0.0425, 0.0754] | 🟡 Medium |
| 4 | `autoFE_f_3` | 0.0547 | 0.0125 | [0.0301, 0.0793] | 🟡 Medium |
| 5 | `autoFE_f_8` | 0.0446 | 0.0084 | [0.0282, 0.0610] | 🟡 Medium |
| 6 | `autoFE_f_6` | 0.0432 | 0.0120 | [0.0196, 0.0668] | 🟡 Medium |
| 7 | `autoFE_f_5` | 0.0374 | 0.0132 | [0.0116, 0.0633] | 🔴 Low |
| 8 | `Fe2O3` | 0.0245 | 0.0191 | [-0.0130, 0.0619] | 🔴 Low |
| 9 | `Al2O3` | 0.0201 | 0.0054 | [0.0096, 0.0307] | 🟡 Medium |
| 10 | `autoFE_f_4` | 0.0129 | 0.0054 | [0.0024, 0.0235] | 🔴 Low |
| 11 | `PbO2` | 0.0101 | 0.0035 | [0.0032, 0.0170] | 🔴 Low |
| 12 | `Y2O3` | 0.0072 | 0.0046 | [-0.0017, 0.0161] | 🔴 Low |
| 13 | `autoFE_f_1` | 0.0072 | 0.0046 | [-0.0017, 0.0161] | 🔴 Low |
| 14 | `autoFE_f_7` | 0.0058 | 0.0054 | [-0.0048, 0.0163] | 🔴 Low |
| 15 | `ZrO2` | 0.0043 | 0.0073 | [-0.0101, 0.0187] | 🔴 Low |
| 16 | `Rb2O` | 0.0043 | 0.0035 | [-0.0026, 0.0112] | 🔴 Low |
| 17 | `Na2O` | 0.0029 | 0.0035 | [-0.0040, 0.0098] | 🔴 Low |
| 18 | `SiO2` | 0.0029 | 0.0035 | [-0.0040, 0.0098] | 🔴 Low |
| 19 | `As2O3` | 0.0029 | 0.0035 | [-0.0040, 0.0098] | 🔴 Low |
| 20 | `ZnO` | 0.0029 | 0.0035 | [-0.0040, 0.0098] | 🔴 Low |
| 21 | `P2O5` | 0.0029 | 0.0035 | [-0.0040, 0.0098] | 🔴 Low |
| 22 | `SrO` | 0.0014 | 0.0029 | [-0.0042, 0.0071] | 🔴 Low |
| 23 | `MgO` | 0.0000 | 0.0046 | [-0.0089, 0.0089] | 🔴 Low |
| 24 | `K2O` | 0.0000 | 0.0046 | [-0.0089, 0.0089] | 🔴 Low |
| 25 | `MnO` | 0.0000 | 0.0000 | [0.0000, 0.0000] | 🔴 Low |
| 26 | `CuO` | 0.0000 | 0.0000 | [0.0000, 0.0000] | 🔴 Low |


**Feature Importance Method Comparison:**

| Feature | XGB Gain Rank | Permutation Rank | Rank Difference | Consistency |
|---------|---------------|------------------|-----------------|-------------|
| `Na2O` | 23 | 17 | 6 | 🔴 Poor |
| `MgO` | 21 | 23 | 2 | 🟡 Good |
| `Al2O3` | 1 | 9 | 8 | 🔴 Poor |
| `SiO2` | 13 | 18 | 5 | 🔴 Poor |
| `K2O` | 15 | 24 | 9 | 🔴 Poor |
| `Fe2O3` | 14 | 8 | 6 | 🔴 Poor |
| `As2O3` | 19 | 19 | 0 | 🟢 Excellent |
| `MnO` | 22 | 25 | 3 | 🔴 Poor |
| `CuO` | 26 | 26 | 0 | 🟢 Excellent |
| `ZnO` | 18 | 20 | 2 | 🟡 Good |
| `PbO2` | 16 | 11 | 5 | 🔴 Poor |
| `Rb2O` | 25 | 16 | 9 | 🔴 Poor |
| `SrO` | 24 | 22 | 2 | 🟡 Good |
| `Y2O3` | 2 | 12 | 10 | 🔴 Poor |
| `ZrO2` | 8 | 15 | 7 | 🔴 Poor |
| `P2O5` | 20 | 21 | 1 | 🟢 Excellent |
| `autoFE_f_0` | 4 | 2 | 2 | 🟡 Good |
| `autoFE_f_1` | 6 | 13 | 7 | 🔴 Poor |
| `autoFE_f_2` | 7 | 1 | 6 | 🔴 Poor |
| `autoFE_f_3` | 12 | 4 | 8 | 🔴 Poor |
| `autoFE_f_4` | 17 | 10 | 7 | 🔴 Poor |
| `autoFE_f_5` | 3 | 7 | 4 | 🔴 Poor |
| `autoFE_f_6` | 11 | 6 | 5 | 🔴 Poor |
| `autoFE_f_7` | 5 | 14 | 9 | 🔴 Poor |
| `autoFE_f_8` | 10 | 5 | 5 | 🔴 Poor |
| `autoFE_f_9` | 9 | 3 | 6 | 🔴 Poor |


**Statistical Summary:**

- **Total Features Analyzed**: 26
- **Gain-based Top Feature**: `Al2O3` (Gain: 7.7741)
- **Permutation-based Top Feature**: `autoFE_f_2` (Importance: 0.0964)

**Method Reliability Assessment:**
- **Average Permutation Std**: 0.0072
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
- **Cross-Validation Data**: `trained_models/10004/6d4097cc-a65c-4c53-9307-aa789e82a318/cross_validation_data/`
- **Feature Importance**: `trained_models/10004/6d4097cc-a65c-4c53-9307-aa789e82a318/feature_importance.csv`

### 6.3 Technical Implementation

- **Framework**: XGBoost for gradient boosting implementation, scikit-learn for pipeline integration.
- **Data Processing**: pandas and numpy for data handling.
- **Cross-Validation**: K-fold cross-validation with stratification support for classification.
- **Feature Importance**: Built-in XGBoost feature importance calculation (Gain, Cover, Weight).
- **Serialization**: Joblib or Pickle for model and preprocessor persistence.

---

## Appendix

### A.1 System Information

- **Generation Time**: 2025-12-26 13:01:43
- **Model ID**: `6d4097cc-a65c-4c53-9307-aa789e82a318`
- **Training System**: XGBoost MCP Tool
- **Report Version**: 2.1 (XGBoost Enhanced)

### A.2 File Structure

```
6d4097cc-a65c-4c53-9307-aa789e82a318/
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
│   ├── 6d4097cc-a65c-4c53-9307-aa789e82a318_cv_predictions_original.csv
│   ├── 6d4097cc-a65c-4c53-9307-aa789e82a318_cv_predictions_processed.csv
│   ├── 6d4097cc-a65c-4c53-9307-aa789e82a318_original_data.csv
│   ├── 6d4097cc-a65c-4c53-9307-aa789e82a318_preprocessed_data.csv
│   ├── 6d4097cc-a65c-4c53-9307-aa789e82a318_roc_curves.png
│   ├── cross_validation_roc_curves.png
│   ├── cross_validation_visualization.png
└── academic_report.md               # This report
```

### A.3 Data Files and JSON Artifacts

The following JSON files contain detailed intermediate data for reproducibility:

- **Feature Importance**: `trained_models/10004/6d4097cc-a65c-4c53-9307-aa789e82a318/feature_importance.csv`

---

*This report was automatically generated by the Enhanced XGBoost MCP Tool for academic research and reproducibility purposes.*
