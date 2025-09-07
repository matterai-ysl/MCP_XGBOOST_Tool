# Prediction Experiment Report

**Generated on:** 2025-09-05 20:16:08  
**Experiment Name:** 2228777e-7f0e-4ecc-9d2a-c6498c632a57  
**Model ID:** `2bb2bed9-0fc4-445a-ac0e-8e01f24154d0`  
**Output Directory:** `trained_models/2bb2bed9-0fc4-445a-ac0e-8e01f24154d0/predictions/2228777e-7f0e-4ecc-9d2a-c6498c632a57`

## Executive Summary

This report documents a comprehensive machine learning prediction experiment conducted using a pre-trained regression model. The experiment involved preprocessing input data, making predictions, and providing detailed statistical analysis of the prediction results.

### Key Results
- **Number of Predictions:** 1
- **Feature Count:** 4
- **Target Count:** 1
- **Model Type:** Regression
- **Processing Time:** 0.059 seconds

---

## 1. Experiment Setup

### 1.1 Input Data Information

| Parameter | Value |
|-----------|-------|
| Number of Samples | 1 |
| Number of Features | 4 |
| Number of Targets | 1 |
| Data Shape | N/A |
| Data Type | Numerical (floating-point) |
| Preprocessing Applied | No |

### 1.2 Feature Information

**Input Features (4 columns):**
`SiO2`, `CeO2`, `Lignin`, `2-VN`

**Target Variables (1 column):**
`target`

### 1.3 Model Information

| Component | Details |
|-----------|---------|
| **Model Type** | Regression Model |
| **Model ID** | `2bb2bed9-0fc4-445a-ac0e-8e01f24154d0` |
| **Framework** | XGBoost (scikit-learn interface) |
| **Prediction Method** | Ensemble averaging of boosted trees |

---

## 2. Prediction Results

### 2.1 Prediction Statistics


#### Single Target Prediction Statistics

**Target: target**

| Statistic | Value |
|-----------|-------|
| Mean Prediction | -1.563496 |
| Standard Deviation | 0.000000 |
| Minimum Prediction | -1.563496 |
| Maximum Prediction | -1.563496 |
| Prediction Range | 0.000000 |
| Median | -1.563496 |
| 25th Percentile | -1.563496 |
| 75th Percentile | -1.563496 |


---

## 3. Generated Files

| File | Description |
|------|-------------|
| `01_original_data.csv` | Original input data as provided |
| `02_processed_features.csv` | Features after preprocessing/normalization |
| `03_predictions_processed_scale.csv` | Predictions in processed/normalized scale |
| `04_predictions_original_scale.csv` | Predictions transformed back to original scale |
| `05_confidence_scores.csv` | Confidence/uncertainty scores for each prediction |
| `06_combined_results.csv` | All data combined: features + predictions + confidence |
| `prediction_experiment_report.md` | This detailed experiment report |
| `prediction_report.html` | Interactive HTML report |

---

## 4. Detailed Prediction Results

This section provides a comprehensive view of each prediction with corresponding input features and confidence scores.

### Feature Values and Predictions

| Sample | SiO2 | CeO2 | Lignin | 2-VN | Prediction | Confidence |
|--------|--------|--------|--------|--------|-----------|----------|
| **#1** | 14.000 | 1.400 | 9.500 | 2.800 | 28.8027 | 0.999 |

### Interpretation Guide

- **Prediction Values**: Continuous numerical predictions representing the target variable
- **Confidence Scores**: Range from 0.0 (low confidence) to 1.0 (high confidence)
  - **≥0.8**: High confidence - Very reliable predictions
  - **0.5-0.8**: Medium confidence - Moderately reliable
  - **<0.5**: Low confidence - Review recommended


---

## 5. Confidence Analysis

### 5.1 Confidence Calculation Method

**Regression Confidence Calculation:**
- **Primary Method**: Prediction variance across individual trees in XGBoost ensemble
- **Formula**: `confidence = 1 / (1 + variance_across_trees)`
- **Range**: 0-1, where higher values indicate more confident predictions
- **Multi-target Handling**: For models with multiple targets, confidence is averaged across all targets
- **Fallback Method**: For non-ensemble models, uses inverse relationship with prediction magnitude

**Technical Details:**
- Each tree in the XGBoost ensemble makes an independent prediction
- Variance across these individual predictions indicates uncertainty
- Low variance (trees agree) → High confidence
- High variance (trees disagree) → Low confidence

### 5.2 Confidence Statistics

**Mean Confidence:** 0.999  
**Standard Deviation:** 0.000  
**Min Confidence:** 0.999  
**Max Confidence:** 0.999  

### 5.3 Confidence Distribution

| Confidence Level | Count | Percentage | Description |
|------------------|-------|------------|-------------|
| High (≥0.8) | 1 | 100.0% | Very reliable predictions |
| Medium (0.5-0.8) | 0 | 0.0% | Moderately reliable predictions |
| Low (<0.5) | 0 | 0.0% | Uncertain predictions - review recommended |

### 5.4 Confidence Interpretation Guide

**For Regression Models:**
- **High Confidence (≥0.8)**: Trees in ensemble show strong agreement
- **Medium Confidence (0.5-0.8)**: Moderate agreement, reliable for most applications
- **Low Confidence (<0.5)**: High prediction variance, may indicate:
  - Input data outside training distribution
  - Insufficient training data for similar cases
  - High inherent noise in target variable
  - Model complexity mismatch with data complexity



---

*Report generated on 2025-09-05 20:16:08*
*MCP XGBoost Tool - Prediction Experiment Report*
