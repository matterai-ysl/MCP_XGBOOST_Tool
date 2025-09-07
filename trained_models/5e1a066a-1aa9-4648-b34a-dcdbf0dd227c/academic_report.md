# XGBoost Training Report

**Generated on:** 2025-09-05 21:15:06  
**Model ID:** `5e1a066a-1aa9-4648-b34a-dcdbf0dd227c`  
**Model Folder:** `trained_models/5e1a066a-1aa9-4648-b34a-dcdbf0dd227c`

## Executive Summary

This report documents a comprehensive XGBoost training experiment conducted for academic research and reproducibility purposes. The experiment involved hyperparameter optimization and cross-validated model training with detailed performance analysis, data validation, and feature importance evaluation.

### Key Results
### 🎯 关键性能指标

- **R²分数 (R² Score):** 0.917772 (±0.042994)
- **平均绝对误差 (Mean Absolute Error):** 0.216153 (±0.069454)
- **均方误差 (Mean Squared Error):** 0.073894 (±0.035337)

- **交叉验证折数:** 5
- **数据集规模:** 53 样本, 4 特征

### ⚙️ 最优超参数

- **n_estimators:** 140
- **max_depth:** 8
- **learning_rate:** 0.05224427292497112
- **subsample:** 0.7770062988994653
- **colsample_bytree:** 0.7975834899659389
- **colsample_bylevel:** 0.6957358462320232
- **reg_alpha:** 0.0016370237686921822
- **reg_lambda:** 0.036880682382636656
- **min_child_weight:** 8
- **gamma:** 9.801959340802334e-06

- **训练时间:** 3.84 秒

---

## 1. Experimental Setup

### 1.1 Dataset Information

| Parameter | Value |
|-----------|-------|
| Data File | `./TS-AF.xls` |
| Data Shape | {'n_samples': 53, 'n_features': 4} |
| Number of Features | 4 |
| Number of Targets | 1 |

### 1.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Task Type | Regression |

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
`SiO2`, `CeO2`, `Lignin`, `2-VN`

**Target Variables (1 column):**
`TS-AF`


### 2.4 Data Quality Assessment

Comprehensive data validation was performed using multiple statistical methods to ensure dataset quality and suitability for machine learning model training. The validation framework employed established statistical techniques for thorough data quality assessment.

#### 2.4.1 Overall Quality Metrics

| Metric | Value | Threshold | Interpretation |
|--------|-------|-----------|----------------|
| Overall Data Quality Score | 87.5/100 | ≥80 (Excellent), ≥60 (Good) | Good - Meets academic standards |
| Quality Level | Good | - | Categorical assessment |
| Ready for Training | Yes | Yes | Model training readiness |
| Critical Issues | 1 | 0 | Data integrity problems |
| Warnings | 0 | <5 | Minor data quality concerns |

#### 2.4.2 Validation Methodology and Results

| Check Name | Method Used | Status | Issues Found | Key Findings |
|------------|-------------|--------|-------------|-------------|
| Feature Names | Statistical Analysis | ❌ FAILED | 1 | 1 issues found |
| Data Dimensions | Statistical Analysis | ✅ PASSED | 0 | No issues |
| Target Variable | Statistical Analysis | ✅ PASSED | 0 | No issues |
| Data Leakage | Statistical Analysis | ✅ PASSED | 0 | No issues |
| Sample Balance | Chi-square, Gini coefficient | ✅ PASSED | 1 | Target distribution checked |
| Feature Correlations | Pearson/Spearman/Kendall | ✅ PASSED | 0 | 0 high correlations |
| Multicollinearity Detection | Variance Inflation Factor (VIF) | ✅ PASSED | 0 | 0 high VIF; avg=1.22 |
| Feature Distributions | Shapiro-Wilk, Jarque-Bera, D'Agostino | ✅ PASSED | 2 | 2 distribution issues |


#### 2.4.2.4 Target Variable Distribution Analysis

**Methodology**: For regression tasks, the distribution of the continuous target variable is analyzed using descriptive statistics to identify its central tendency, dispersion, and shape.

**Results**:
- **Outlier Ratio**: 0.00% of target values were identified as outliers using the IQR method.
- **Distribution Shape**: Skewness of 0.157 and Kurtosis of -1.254.

**Target Variable Statistics**:
| Statistic | Value |
|-----------|-------|
| Mean | 3.3659 |
| Standard Deviation | 1.5110 |
| Minimum | 1.1100 |
| 25th Percentile | 2.2400 |
| Median (50th) | 3.1600 |
| 75th Percentile | 4.6338 |
| Maximum | 6.1416 |
| Skewness | 0.1572 |
| Kurtosis | -1.2539 |

**Methodological Implications**: The distribution of the target variable is crucial for regression model performance. Significant skewness or a high number of outliers may suggest the need for target transformation (e.g., log transformation) to improve model stability and accuracy.

#### 2.4.2.1 Feature Correlation Analysis

**Methodology**: Pearson, Spearman, and Kendall correlation coefficients were computed for all feature pairs. The correlation threshold was set at |r| ≥ 0.7.

**Results**: 0 feature pairs exceeded the correlation threshold, indicating potential redundancy in the feature space.

**Feature Classification**:
Continuous Features: SiO2, CeO2, Lignin, 2_VN
Categorical Features: None
Target Feature: TS_AF

**Statistical Findings**:
**Continuous Features vs Continuous Features Correlation Analysis (Pearson Correlation Coefficient)**:

| Feature 1 | Feature 2 | Correlation | Absolute Value |
|-----------|-----------|-------------|----------------|
| CeO2 | 2_VN | -0.3772 | 0.3772 |
| SiO2 | CeO2 | 0.3555 | 0.3555 |
| Lignin | 2_VN | 0.3094 | 0.3094 |
| SiO2 | Lignin | -0.2485 | 0.2485 |
| CeO2 | Lignin | -0.2014 | 0.2014 |
| SiO2 | 2_VN | -0.1711 | 0.1711 |


**Continuous Features vs Continuous Features Correlation Analysis (Spearman's Rank Correlation)**:

| Feature 1 | Feature 2 | Correlation | Absolute Value |
|-----------|-----------|-------------|----------------|
| SiO2 | CeO2 | 0.2890 | 0.2890 |
| CeO2 | 2_VN | -0.2787 | 0.2787 |
| Lignin | 2_VN | 0.2710 | 0.2710 |
| SiO2 | Lignin | -0.2525 | 0.2525 |
| SiO2 | 2_VN | -0.2024 | 0.2024 |
| CeO2 | Lignin | -0.1642 | 0.1642 |


**Continuous Features vs Target Variable Correlation Analysis**:

| Feature | Correlation | Method | Absolute Value | Strength |
|---------|-------------|--------|----------------|----------|
| SiO2 | 0.9365 | pearson | 0.9365 | Strong |
| Lignin | -0.4835 | pearson | 0.4835 | Weak |
| CeO2 | 0.2754 | pearson | 0.2754 | Very Weak |
| 2_VN | -0.1687 | pearson | 0.1687 | Very Weak |


**Impact Assessment**: Low feature correlation indicates good feature independence, supporting model stability.

#### 2.4.2.2 Multicollinearity Detection

**Methodology**: Variance Inflation Factor (VIF) analysis was conducted using linear regression. VIF values ≥ 5.0 indicate problematic multicollinearity.

**Results**: 
- Average VIF: 1.224
- Maximum VIF: 1.299
- Features with VIF ≥ 5.0: 0

**Statistical Findings**:
**VIF Scores for All Features**:

| Feature | VIF Score | R² | Interpretation | Status |
|---------|-----------|----|--------------|---------|
| CeO2 | 1.2992 | 0.2303 | Acceptable | ✅ LOW |
| 2_VN | 1.2486 | 0.1991 | Acceptable | ✅ LOW |
| SiO2 | 1.1891 | 0.1590 | Acceptable | ✅ LOW |
| Lignin | 1.1574 | 0.1360 | Acceptable | ✅ LOW |


**Methodological Impact**: Low VIF scores support the assumption of feature independence required for many machine learning algorithms.

#### 2.4.2.3 Feature Distribution Analysis

**Methodology**: 
- Continuous features: Shapiro-Wilk test (n≤5000), Jarque-Bera test (n≥50), D'Agostino test (n≥20) for normality
- Skewness assessment using sample skewness coefficient
- Outlier detection via Interquartile Range (IQR) method
- Categorical features: Gini coefficient, entropy, and class imbalance ratio analysis

**Results**: 0 distribution-related issues identified across 7 continuous and 0 categorical features.

**Continuous Features Statistical Summary**:
| Feature | mean | std | min | max | max | median | Skewness | Kurtosis | Normality | Outliers (%) | Issues |
|---------|----------|---------|----------|---------|----------|---------|----------|-----------|-------------|--------|
| 2_VN | 4.592 | 2.722 | 0.000 | 9.800 | 9.800 | 4.300 | 0.171 | -0.802 | Yes | 0.0% | 0 |
| CeO2 | 5.391 | 4.029 | 0.200 | 14.800 | 14.800 | 5.100 | 0.716 | -0.158 | No | 0.0% | 0 |
| Lignin | 4.504 | 3.373 | 0.000 | 9.900 | 9.900 | 3.800 | 0.311 | -1.381 | No | 0.0% | 1 |
| SiO2 | 26.547 | 9.147 | 10.000 | 40.000 | 40.000 | 26.000 | -0.055 | -1.247 | No | 0.0% | 1 |

**Continuous Feature Distribution Issues:**
- Feature 'SiO2' significantly deviates from normal distribution
- Feature 'Lignin' significantly deviates from normal distribution


**Categorical Features Statistical Summary**:
No categorical features analyzed.

**Distribution Quality Impact**: Feature distributions meet statistical assumptions for machine learning applications.

#### 2.4.2.5 Statistical Summary

**Validation Framework Performance**:
- Total validation checks: 8
- Passed checks: 7 (87.5%)
- Failed checks: 1

**Data Quality Confidence**: Based on the comprehensive validation framework, the dataset demonstrates high statistical reliability for machine learning applications.

#### 2.4.3 Data Quality Issues and Impact Assessment

**Critical Issues Identified:**

- Invalid feature name after normalization: '2_VN'



#### 2.4.4 Academic and Methodological Implications

The data validation results indicate that the dataset meets the quality standards required for academic machine learning research. Moderate data quality with some limitations. Results should be interpreted with consideration of identified data quality issues.

**Reproducibility Impact**: High reproducibility confidence with comprehensive data quality documentation supporting experimental replication.


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
**Best Score**: -0.216153

**Optimization Strategy:**
- **Initial Exploration**: 10 random trials for space exploration
- **Exploitation-Exploration Balance**: TPE algorithm balances promising regions with unexplored space
- **Cross-Validation**: Each trial evaluated using stratified k-fold cross-validation
- **Early Stopping**: Poor-performing trials terminated early to improve efficiency

### 3.3 Best Parameters Found

```json
{
  "n_estimators": 140,
  "max_depth": 8,
  "learning_rate": 0.05224427292497112,
  "subsample": 0.7770062988994653,
  "colsample_bytree": 0.7975834899659389,
  "colsample_bylevel": 0.6957358462320232,
  "reg_alpha": 0.0016370237686921822,
  "reg_lambda": 0.036880682382636656,
  "min_child_weight": 8,
  "gamma": 9.801959340802334e-06
}
```

### 3.4 Optimization Convergence

The optimization process completed **50 trials** with the best configuration achieving a cross-validation score of **-0.216153**.

**Key Optimization Insights:**
- **Ensemble Size**: 140 boosting rounds balances performance and computational efficiency
- **Tree Complexity**: Maximum depth of 8 controls model complexity and overfitting
- **Learning Rate**: 0.05224427292497112 provides optimal step size for gradient descent
- **Regularization**: L1=1.64e-03, L2=3.69e-02 prevent overfitting
- **Sampling**: 0.7770062988994653 row sampling and 0.7975834899659389 column sampling for robustness

## 4. Final Model Training

### 4.1 Cross-Validation Training

The final model was trained using 5-fold cross-validation with optimized hyperparameters. Training metrics and validation results were recorded comprehensively.

### 4.2 Training Results

| Metric | Value |
|--------|-------|
### Cross-Validation Performance Metrics

| Metric | Mean ± Std | Min | Max |
|--------|------------|-----|-----|
| MAE | 0.216153 ± 0.069454 | 0.091443 | 0.277225 |
| MSE | 0.073894 ± 0.035337 | 0.014415 | 0.104349 |
| R2 | 0.917772 ± 0.042994 | 0.878985 | 0.978499 |



#### Fold-wise Results

#### Detailed Fold-wise Performance

| Fold | MAE | MSE | R2 |
|------|---------|---------|---------|
| 1 | 0.091443 | 0.014415 | 0.978499 |
| 2 | 0.254448 | 0.100928 | 0.886292 |
| 3 | 0.189775 | 0.052026 | 0.961358 |
| 4 | 0.267873 | 0.097754 | 0.883724 |
| 5 | 0.277225 | 0.104349 | 0.878985 |

#### Statistical Summary

| Metric | Mean | Std Dev | Min | Max | 95% CI |
|--------|------|---------|-----|-----|--------|
| MAE | 0.216153 | 0.069454 | 0.091443 | 0.277225 | [0.155273, 0.277032] |
| MSE | 0.073894 | 0.035337 | 0.014415 | 0.104349 | [0.042920, 0.104869] |
| R2 | 0.917772 | 0.042994 | 0.878985 | 0.978499 | [0.880086, 0.955457] |

### 4.3 Model Performance Visualization

#### Training Performance Analysis

The cross-validation analysis demonstrates the model's predictive performance through scatter plots comparing predicted versus actual values.

<div style="text-align: center; margin: 20px 0;">
    <img src="cross_validation_data/cross_validation_scatter.png" alt="Cross-Validation Scatter Plot" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    <p style="font-style: italic; color: #666; margin-top: 10px;">Cross-Validation: Predicted vs Actual Values</p>
</div>


<div style="text-align: center; margin: 20px 0;">
    <img src="cross_validation_data/cross_validation_scatter_normalized.png" alt="Normalized Cross-Validation Scatter Plot" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    <p style="font-style: italic; color: #666; margin-top: 10px;">Cross-Validation Results on Normalized Data</p>
</div>


<div style="text-align: center; margin: 20px 0;">
    <img src="cross_validation_data/cross_validation_scatter_original.png" alt="Original Scale Cross-Validation Scatter Plot" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    <p style="font-style: italic; color: #666; margin-top: 10px;">Cross-Validation Results on Original Scale</p>
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
| 1 | `SiO2` | 2.1431 | 149 | 30.70 | 67.1% | 34.1% |
| 2 | `Lignin` | 0.6510 | 110 | 31.19 | 20.4% | 25.2% |
| 3 | `2-VN` | 0.2261 | 78 | 29.69 | 7.1% | 17.8% |
| 4 | `CeO2` | 0.1738 | 100 | 28.00 | 5.4% | 22.9% |


**Permutation Feature Importance:**

| Rank | Feature | Mean Importance | Std Dev | 95% CI | Reliability |
|------|---------|-----------------|---------|--------|-------------|
| 1 | `SiO2` | 1.3260 | 0.1448 | [1.0422, 1.6098] | 🟡 Medium |
| 2 | `Lignin` | 0.1732 | 0.0290 | [0.1164, 0.2300] | 🟡 Medium |
| 3 | `CeO2` | 0.0149 | 0.0055 | [0.0042, 0.0256] | 🔴 Low |
| 4 | `2-VN` | 0.0123 | 0.0026 | [0.0073, 0.0173] | 🟡 Medium |


**Feature Importance Method Comparison:**

| Feature | XGB Gain Rank | Permutation Rank | Rank Difference | Consistency |
|---------|---------------|------------------|-----------------|-------------|
| `SiO2` | 1 | 1 | 0 | 🟢 Excellent |
| `CeO2` | 4 | 3 | 1 | 🟢 Excellent |
| `Lignin` | 2 | 2 | 0 | 🟢 Excellent |
| `2-VN` | 3 | 4 | 1 | 🟢 Excellent |


**Statistical Summary:**

- **Total Features Analyzed**: 4
- **Gain-based Top Feature**: `SiO2` (Gain: 2.1431)
- **Permutation-based Top Feature**: `SiO2` (Importance: 1.3260)

**Method Reliability Assessment:**
- **Average Permutation Std**: 0.0454
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
| Task Type | Regression |

---

## 6. Conclusions and Future Work

### 6.1 Key Findings

2. **Hyperparameter Optimization**: Systematic optimization improved model performance

### 6.2 Reproducibility

This experiment is fully reproducible using the following artifacts:
- **Cross-Validation Data**: `trained_models/5e1a066a-1aa9-4648-b34a-dcdbf0dd227c/cross_validation_data/`
- **Feature Importance**: `trained_models/5e1a066a-1aa9-4648-b34a-dcdbf0dd227c/feature_importance.csv`

### 6.3 Technical Implementation

- **Framework**: XGBoost for gradient boosting implementation, scikit-learn for pipeline integration.
- **Data Processing**: pandas and numpy for data handling.
- **Cross-Validation**: K-fold cross-validation with stratification support for classification.
- **Feature Importance**: Built-in XGBoost feature importance calculation (Gain, Cover, Weight).
- **Serialization**: Joblib or Pickle for model and preprocessor persistence.

---

## Appendix

### A.1 System Information

- **Generation Time**: 2025-09-05 21:15:06
- **Model ID**: `5e1a066a-1aa9-4648-b34a-dcdbf0dd227c`
- **Training System**: XGBoost MCP Tool
- **Report Version**: 2.1 (XGBoost Enhanced)

### A.2 File Structure

```
5e1a066a-1aa9-4648-b34a-dcdbf0dd227c/
├── model.joblib
├── preprocessing_pipeline.pkl
├── evaluation_metrics.csv
├── feature_importance.csv
├── optimization_history.csv
├── raw_data.csv
├── continuous_feature_distributions.png
├── continuous_feature_normality.png
├── continuous_feature_outliers.png
├── continuous_feature_violin_plots.png
├── continuous_pearson_correlation.png
├── continuous_spearman_correlation.png
├── feature_importance_comparison.png
├── feature_importance_permutation.png
├── feature_importance_tree.png
├── feature_target_continuous_correlation.png
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
│   ├── 5e1a066a-1aa9-4648-b34a-dcdbf0dd227c_cv_predictions_original.csv
│   ├── 5e1a066a-1aa9-4648-b34a-dcdbf0dd227c_cv_predictions_processed.csv
│   ├── 5e1a066a-1aa9-4648-b34a-dcdbf0dd227c_cv_scatter_plot.png
│   ├── 5e1a066a-1aa9-4648-b34a-dcdbf0dd227c_original_data.csv
│   ├── 5e1a066a-1aa9-4648-b34a-dcdbf0dd227c_preprocessed_data.csv
│   ├── cross_validation_scatter.png
│   ├── cross_validation_scatter_normalized.png
│   ├── cross_validation_scatter_original.png
└── academic_report.md               # This report
```

### A.3 Data Files and JSON Artifacts

The following JSON files contain detailed intermediate data for reproducibility:

- **Feature Importance**: `trained_models/5e1a066a-1aa9-4648-b34a-dcdbf0dd227c/feature_importance.csv`

---

*This report was automatically generated by the Enhanced XGBoost MCP Tool for academic research and reproducibility purposes.*
