# XGBOOST MCP Tool

A comprehensive XGBOOST machine learning tool designed for MCP protocol integration, providing high-performance gradient boosting capabilities for AI applications and automated workflows.

## 📋 Overview

This project adapts existing Random Forest MCP tool architecture to implement XGBOOST functionality while maintaining complete feature compatibility and user experience consistency.

## 🚀 Features

- **High Performance**: XGBOOST gradient boosting algorithm with excellent ML competition performance
- **Flexibility**: Support for regression, classification, and multi-target tasks
- **Interpretability**: Rich feature importance analysis capabilities
- **User-Friendly**: Consistent interface design with existing tools

### Core MCP Tool Functions

- `train_xgboost_regressor` - Train XGBOOST regression models with multi-target support
- `train_xgboost_classifier` - Train XGBOOST classification models
- `predict_from_file` - Batch prediction capabilities
- `predict_from_values` - Real-time predictions
- `analyze_global_feature_importance` - Global feature importance analysis
- `analyze_local_feature_importance` - Local feature importance analysis
- Model management functions (list, info, delete)

## 🛠️ Installation

### Prerequisites

- Python >= 3.8
- UV package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd MCP-XGBOOST-Tool
```

2. Create virtual environment:
```bash
uv venv
```

3. Activate virtual environment:
```bash
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

4. Install dependencies:
```bash
uv pip install -r requirements.txt
```

## 📦 Dependencies

- xgboost >= 2.0.0
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- optuna >= 3.4.0
- shap >= 0.42.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- fastapi >= 0.104.0
- mcp >= 1.0.0
- joblib >= 1.3.0
- jinja2 >= 3.1.0

## 🏗️ Project Structure

```
src/mcp_xgboost_tool/
├── mcp_server.py                   # MCP服务器 (核心改造)
├── xgboost_wrapper.py              # XGBOOST包装类 (新建)
├── training.py                     # 训练引擎 (改造)
├── prediction.py                   # 预测引擎 (轻微改造)
├── feature_importance.py           # 特征重要性 (改造)
├── local_feature_importance.py     # 局部特征重要性 (改造)
├── hyperparameter_optimizer.py     # 超参数优化 (改造)
└── model_manager.py               # 模型管理 (轻微改造)
```

## 🎯 Development Roadmap

### Phase 1: Core Algorithm Replacement (1-2 weeks)
- [x] Project setup
- [ ] Create XGBoostWrapper class
- [ ] Update MCP server functions
- [ ] Refactor TrainingEngine

### Phase 2: Hyperparameter Optimization & Feature Analysis (1-2 weeks)
- [ ] Update HyperparameterOptimizer for XGBOOST parameters
- [ ] Enhance FeatureImportanceAnalyzer
- [ ] Update cross-validation strategies
- [ ] Enhance evaluation metrics

### Phase 3: Reports & Visualization (1 week)
- [ ] Update HTML report generator
- [ ] Adjust visualization components
- [ ] Update academic report format

### Phase 4: Testing & Optimization (1 week)
- [ ] Complete functional testing
- [ ] Performance optimization
- [ ] Edge case handling
- [ ] Documentation completion

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📊 Performance Requirements

- Training speed: No slower than Random Forest on most datasets
- Prediction accuracy: Superior to Random Forest on benchmark datasets
- Memory usage: Reasonable memory footprint
- API response time: <500ms for small-scale predictions

## 🔧 Development Status

This project is currently in development. Check the roadmap above for current progress.
