# XGBoost Training Report

**Generated on:** 2025-12-01 15:58:51  
**Model ID:** `fe766431-d542-406b-9cfd-8affee0ddcc8`  
**Model Folder:** `trained_models/10667/fe766431-d542-406b-9cfd-8affee0ddcc8`

## Executive Summary

This report documents a comprehensive XGBoost training experiment conducted for academic research and reproducibility purposes. The experiment involved hyperparameter optimization and cross-validated model training with detailed performance analysis, data validation, and feature importance evaluation.

### Key Results
### 🎯 关键性能指标

- **R²分数 (R² Score):** 0.854594 (±0.006834)
- **平均绝对误差 (Mean Absolute Error):** 0.265244 (±0.006278)
- **均方误差 (Mean Squared Error):** 0.145077 (±0.006833)

- **交叉验证折数:** 5
- **数据集规模:** 2703 样本, 4 特征

### ⚙️ 最优超参数

- **n_estimators:** 110
- **max_depth:** 9
- **learning_rate:** 0.15351410544118682
- **subsample:** 0.7171483033388761
- **colsample_bytree:** 0.8653670933226423
- **colsample_bylevel:** 0.9346975594939649
- **reg_alpha:** 1.202643795145194
- **reg_lambda:** 2.876347176040799e-08
- **min_child_weight:** 5
- **gamma:** 2.8452172966599057e-08

- **训练时间:** 9.89 秒

---

## 1. Experimental Setup

### 1.1 Dataset Information

| Parameter | Value |
|-----------|-------|
| Data File | `http://47.99.180.80/file/uploads/SLM_4.xls` |
| Data Shape | {'n_samples': 2703, 'n_features': 4} |
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
`layer_thickness`, `hatch_distance`, `laser_power`, `laser_velocity`

**Target Variables (1 column):**
`relative_density`


### 2.4 Data Quality Assessment

Comprehensive data validation was performed using multiple statistical methods to ensure dataset quality and suitability for machine learning model training. The validation framework employed established statistical techniques for thorough data quality assessment.

#### 2.4.1 Overall Quality Metrics

| Metric | Value | Threshold | Interpretation |
|--------|-------|-----------|----------------|
| Overall Data Quality Score | 100.0/100 | ≥80 (Excellent), ≥60 (Good) | Excellent - Suitable for publication |
| Quality Level | Excellent | - | Categorical assessment |
| Ready for Training | Yes | Yes | Model training readiness |
| Critical Issues | 0 | 0 | Data integrity problems |
| Warnings | 0 | <5 | Minor data quality concerns |

#### 2.4.2 Validation Methodology and Results

| Check Name | Method Used | Status | Issues Found | Key Findings |
|------------|-------------|--------|-------------|-------------|
| Feature Names | Statistical Analysis | ✅ PASSED | 0 | No issues |
| Data Dimensions | Statistical Analysis | ✅ PASSED | 0 | No issues |
| Target Variable | Statistical Analysis | ✅ PASSED | 0 | No issues |
| Data Leakage | Statistical Analysis | ✅ PASSED | 0 | No issues |
| Sample Balance | Chi-square, Gini coefficient | ✅ PASSED | 0 | Target distribution checked |
| Feature Correlations | Pearson/Spearman/Kendall | ✅ PASSED | 0 | 0 high correlations |
| Multicollinearity Detection | Variance Inflation Factor (VIF) | ✅ PASSED | 0 | 0 high VIF; avg=1.00 |
| Feature Distributions | Shapiro-Wilk, Jarque-Bera, D'Agostino | ✅ PASSED | 4 | 4 distribution issues |


#### 2.4.2.4 Target Variable Distribution Analysis

**Methodology**: For regression tasks, the distribution of the continuous target variable is analyzed using descriptive statistics to identify its central tendency, dispersion, and shape.

**Results**:
- **Outlier Ratio**: 1.66% of target values were identified as outliers using the IQR method.
- **Distribution Shape**: Skewness of -1.244 and Kurtosis of 1.633.

**Target Variable Statistics**:
| Statistic | Value |
|-----------|-------|
| Mean | 92.5462 |
| Standard Deviation | 4.6224 |
| Minimum | 75.8000 |
| 25th Percentile | 90.9000 |
| Median (50th) | 93.2000 |
| 75th Percentile | 95.8000 |
| Maximum | 99.5000 |
| Skewness | -1.2442 |
| Kurtosis | 1.6335 |

**Methodological Implications**: The distribution of the target variable is crucial for regression model performance. Significant skewness or a high number of outliers may suggest the need for target transformation (e.g., log transformation) to improve model stability and accuracy.

#### 2.4.2.1 Feature Correlation Analysis

**Methodology**: Pearson, Spearman, and Kendall correlation coefficients were computed for all feature pairs. The correlation threshold was set at |r| ≥ 0.7.

**Results**: 0 feature pairs exceeded the correlation threshold, indicating potential redundancy in the feature space.

**Feature Classification**:
Continuous Features: layer_thickness, hatch_distance, laser_power, laser_velocity
Categorical Features: None
Target Feature: relative_density

**Statistical Findings**:
**Continuous Features vs Continuous Features Correlation Analysis (Pearson Correlation Coefficient)**:

| Feature 1 | Feature 2 | Correlation | Absolute Value |
|-----------|-----------|-------------|----------------|
| layer_thickness | laser_power | 0.0182 | 0.0182 |
| layer_thickness | laser_velocity | -0.0167 | 0.0167 |
| laser_power | laser_velocity | -0.0050 | 0.0050 |
| hatch_distance | laser_velocity | -0.0037 | 0.0037 |
| hatch_distance | laser_power | 0.0026 | 0.0026 |
| layer_thickness | hatch_distance | 0.0021 | 0.0021 |


**Continuous Features vs Continuous Features Correlation Analysis (Spearman's Rank Correlation)**:

| Feature 1 | Feature 2 | Correlation | Absolute Value |
|-----------|-----------|-------------|----------------|
| layer_thickness | laser_power | 0.0180 | 0.0180 |
| layer_thickness | laser_velocity | -0.0172 | 0.0172 |
| laser_power | laser_velocity | -0.0049 | 0.0049 |
| hatch_distance | laser_velocity | -0.0034 | 0.0034 |
| hatch_distance | laser_power | 0.0023 | 0.0023 |
| layer_thickness | hatch_distance | 0.0021 | 0.0021 |


**Continuous Features vs Target Variable Correlation Analysis**:

| Feature | Correlation | Method | Absolute Value | Strength |
|---------|-------------|--------|----------------|----------|
| laser_velocity | -0.4079 | pearson | 0.4079 | Weak |
| laser_power | 0.3220 | pearson | 0.3220 | Weak |
| layer_thickness | -0.1794 | pearson | 0.1794 | Very Weak |
| hatch_distance | -0.1257 | pearson | 0.1257 | Very Weak |


**Impact Assessment**: Low feature correlation indicates good feature independence, supporting model stability.

#### 2.4.2.2 Multicollinearity Detection

**Methodology**: Variance Inflation Factor (VIF) analysis was conducted using linear regression. VIF values ≥ 5.0 indicate problematic multicollinearity.

**Results**: 
- Average VIF: 1.000
- Maximum VIF: 1.001
- Features with VIF ≥ 5.0: 0

**Statistical Findings**:
**VIF Scores for All Features**:

| Feature | VIF Score | R² | Interpretation | Status |
|---------|-----------|----|--------------|---------|
| layer_thickness | 1.0006 | 0.0006 | Acceptable | ✅ LOW |
| laser_power | 1.0004 | 0.0004 | Acceptable | ✅ LOW |
| laser_velocity | 1.0003 | 0.0003 | Acceptable | ✅ LOW |
| hatch_distance | 1.0000 | 0.0000 | Acceptable | ✅ LOW |


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
| hatch_distance | 54.782 | 15.663 | 30.000 | 80.000 | 80.000 | 55.000 | 0.033 | -1.224 | No | 0.0% | 1 |
| laser_power | 129.408 | 30.460 | 80.000 | 180.000 | 180.000 | 130.000 | 0.015 | -1.207 | No | 0.0% | 1 |
| laser_velocity | 1651.720 | 517.399 | 800.000 | 2500.000 | 2500.000 | 1700.000 | -0.031 | -1.194 | No | 0.0% | 1 |
| layer_thickness | 49.724 | 18.587 | 20.000 | 80.000 | 80.000 | 50.000 | 0.021 | -1.216 | No | 0.0% | 1 |

**Continuous Feature Distribution Issues:**
- Feature 'layer_thickness' significantly deviates from normal distribution
- Feature 'hatch_distance' significantly deviates from normal distribution
- Feature 'laser_power' significantly deviates from normal distribution
- Feature 'laser_velocity' significantly deviates from normal distribution


**Categorical Features Statistical Summary**:
No categorical features analyzed.

**Distribution Quality Impact**: Feature distributions meet statistical assumptions for machine learning applications.

#### 2.4.2.5 Statistical Summary

**Validation Framework Performance**:
- Total validation checks: 8
- Passed checks: 8 (100.0%)
- Failed checks: 0

**Data Quality Confidence**: Based on the comprehensive validation framework, the dataset demonstrates high statistical reliability for machine learning applications.

#### 2.4.3 Data Quality Issues and Impact Assessment

**No critical issues detected.**

All validation checks passed successfully, indicating high data quality suitable for academic research with strong statistical foundations.



#### 2.4.4 Academic and Methodological Implications

The data validation results indicate that the dataset meets the quality standards required for academic machine learning research. High data quality supports robust experimental conclusions and enhances reproducibility of results. The dataset meets standards for academic publication.

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
**Best Score**: -0.265244

**Optimization Strategy:**
- **Initial Exploration**: 10 random trials for space exploration
- **Exploitation-Exploration Balance**: TPE algorithm balances promising regions with unexplored space
- **Cross-Validation**: Each trial evaluated using stratified k-fold cross-validation
- **Early Stopping**: Poor-performing trials terminated early to improve efficiency

### 3.3 Best Parameters Found

```json
{
  "n_estimators": 110,
  "max_depth": 9,
  "learning_rate": 0.15351410544118682,
  "subsample": 0.7171483033388761,
  "colsample_bytree": 0.8653670933226423,
  "colsample_bylevel": 0.9346975594939649,
  "reg_alpha": 1.202643795145194,
  "reg_lambda": 2.876347176040799e-08,
  "min_child_weight": 5,
  "gamma": 2.8452172966599057e-08
}
```

### 3.4 Optimization Convergence

The optimization process completed **50 trials** with the best configuration achieving a cross-validation score of **-0.265244**.

**Key Optimization Insights:**
- **Ensemble Size**: 110 boosting rounds balances performance and computational efficiency
- **Tree Complexity**: Maximum depth of 9 controls model complexity and overfitting
- **Learning Rate**: 0.15351410544118682 provides optimal step size for gradient descent
- **Regularization**: L1=1.20e+00, L2=2.88e-08 prevent overfitting
- **Sampling**: 0.7171483033388761 row sampling and 0.8653670933226423 column sampling for robustness

## 4. Final Model Training

### 4.1 Cross-Validation Training

The final model was trained using 5-fold cross-validation with optimized hyperparameters. Training metrics and validation results were recorded comprehensively.

### 4.2 Training Results

| Metric | Value |
|--------|-------|
### Cross-Validation Performance Metrics

| Metric | Mean ± Std | Min | Max |
|--------|------------|-----|-----|
| MAE | 0.265244 ± 0.006278 | 0.257767 | 0.274591 |
| MSE | 0.145077 ± 0.006833 | 0.134038 | 0.152742 |
| R2 | 0.854594 ± 0.006834 | 0.845208 | 0.862616 |



#### Fold-wise Results

#### Detailed Fold-wise Performance

| Fold | MAE | MSE | R2 |
|------|---------|---------|---------|
| 1 | 0.274591 | 0.152742 | 0.861945 |
| 2 | 0.263843 | 0.141257 | 0.849471 |
| 3 | 0.257767 | 0.134038 | 0.862616 |
| 4 | 0.270095 | 0.151159 | 0.853728 |
| 5 | 0.259926 | 0.146187 | 0.845208 |

#### Statistical Summary

| Metric | Mean | Std Dev | Min | Max | 95% CI |
|--------|------|---------|-----|-----|--------|
| MAE | 0.265244 | 0.006278 | 0.257767 | 0.274591 | [0.259742, 0.270747] |
| MSE | 0.145077 | 0.006833 | 0.134038 | 0.152742 | [0.139088, 0.151066] |
| R2 | 0.854594 | 0.006834 | 0.845208 | 0.862616 | [0.848604, 0.860584] |

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
| 1 | `laser_velocity` | 1.0281 | 1645 | 201.37 | 29.0% | 23.2% |
| 2 | `laser_power` | 0.9794 | 1860 | 207.53 | 27.6% | 26.2% |
| 3 | `layer_thickness` | 0.8382 | 1890 | 186.39 | 23.7% | 26.6% |
| 4 | `hatch_distance` | 0.6976 | 1707 | 199.41 | 19.7% | 24.0% |


**Permutation Feature Importance:**

| Rank | Feature | Mean Importance | Std Dev | 95% CI | Reliability |
|------|---------|-----------------|---------|--------|-------------|
| 1 | `laser_power` | 0.7390 | 0.0197 | [0.7004, 0.7776] | 🟢 High |
| 2 | `laser_velocity` | 0.7305 | 0.0070 | [0.7167, 0.7443] | 🟢 High |
| 3 | `layer_thickness` | 0.6420 | 0.0121 | [0.6183, 0.6658] | 🟢 High |
| 4 | `hatch_distance` | 0.5895 | 0.0161 | [0.5579, 0.6211] | 🟢 High |


**Feature Importance Method Comparison:**

| Feature | XGB Gain Rank | Permutation Rank | Rank Difference | Consistency |
|---------|---------------|------------------|-----------------|-------------|
| `layer_thickness` | 3 | 3 | 0 | 🟢 Excellent |
| `hatch_distance` | 4 | 4 | 0 | 🟢 Excellent |
| `laser_power` | 2 | 1 | 1 | 🟢 Excellent |
| `laser_velocity` | 1 | 2 | 1 | 🟢 Excellent |


**Statistical Summary:**

- **Total Features Analyzed**: 4
- **Gain-based Top Feature**: `laser_velocity` (Gain: 1.0281)
- **Permutation-based Top Feature**: `laser_power` (Importance: 0.7390)

**Method Reliability Assessment:**
- **Average Permutation Std**: 0.0137
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
- **Cross-Validation Data**: `trained_models/10667/fe766431-d542-406b-9cfd-8affee0ddcc8/cross_validation_data/`
- **Feature Importance**: `trained_models/10667/fe766431-d542-406b-9cfd-8affee0ddcc8/feature_importance.csv`

### 6.3 Technical Implementation

- **Framework**: XGBoost for gradient boosting implementation, scikit-learn for pipeline integration.
- **Data Processing**: pandas and numpy for data handling.
- **Cross-Validation**: K-fold cross-validation with stratification support for classification.
- **Feature Importance**: Built-in XGBoost feature importance calculation (Gain, Cover, Weight).
- **Serialization**: Joblib or Pickle for model and preprocessor persistence.

---

## Appendix

### A.1 System Information

- **Generation Time**: 2025-12-01 15:58:51
- **Model ID**: `fe766431-d542-406b-9cfd-8affee0ddcc8`
- **Training System**: XGBoost MCP Tool
- **Report Version**: 2.1 (XGBoost Enhanced)

### A.2 File Structure

```
fe766431-d542-406b-9cfd-8affee0ddcc8/
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
│   ├── cross_validation_scatter.png
│   ├── cross_validation_scatter_normalized.png
│   ├── cross_validation_scatter_original.png
│   ├── fe766431-d542-406b-9cfd-8affee0ddcc8_cv_predictions_original.csv
│   ├── fe766431-d542-406b-9cfd-8affee0ddcc8_cv_predictions_processed.csv
│   ├── fe766431-d542-406b-9cfd-8affee0ddcc8_cv_scatter_plot.png
│   ├── fe766431-d542-406b-9cfd-8affee0ddcc8_original_data.csv
│   ├── fe766431-d542-406b-9cfd-8affee0ddcc8_preprocessed_data.csv
└── academic_report.md               # This report
```

### A.3 Data Files and JSON Artifacts

The following JSON files contain detailed intermediate data for reproducibility:

- **Feature Importance**: `trained_models/10667/fe766431-d542-406b-9cfd-8affee0ddcc8/feature_importance.csv`

---

*This report was automatically generated by the Enhanced XGBoost MCP Tool for academic research and reproducibility purposes.*
