# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an XGBoost MCP Tool - a comprehensive machine learning server that provides XGBoost capabilities through the MCP (Model Context Protocol). The project implements a FastMCP-based server with advanced XGBoost functionality for regression and classification tasks.

## Essential Commands

### Development & Testing
```bash
# Install dependencies
uv pip install -r requirements.txt

# Run the main MCP server (standard MCP protocol)
python -m src.mcp_xgboost_tool

# Run the single port server (HTTP + MCP combined)
python single_port_server.py --host 0.0.0.0 --port 8100

# Run tests
pytest

# Type checking (if mypy is installed)
mypy src/
```

### Server Modes

The project supports two server modes:

1. **Standard MCP Server**: Traditional MCP protocol server (`python -m src.mcp_xgboost_tool`)
2. **Single Port Server**: HTTP + MCP combined server (`python single_port_server.py`)

The single port server provides:
- MCP endpoint: `http://localhost:8100/mcp`
- File download API: `http://localhost:8100/api/download/file/{path}`
- Static file serving: `http://localhost:8100/static/{path}`
- Health checks: `http://localhost:8100/api/health`

## Core Architecture

### Key Components

The codebase follows a modular architecture centered around XGBoost ML capabilities:

- **`mcp_server.py`**: Main MCP server implementation using FastMCP
- **`xgboost_wrapper.py`**: Core XGBoost algorithm wrapper with enhanced functionality
- **`training.py`**: Training engine for model creation and management
- **`prediction.py`**: Prediction engine for batch and real-time predictions  
- **`feature_importance.py`**: Global feature importance analysis
- **`local_feature_importance.py`**: Local/instance-level feature importance (SHAP, LIME)
- **`hyperparameter_optimizer.py`**: Optuna-based hyperparameter optimization
- **`model_manager.py`**: Model persistence and lifecycle management

### Data Processing Pipeline

- **`data_preprocessing.py`**: Data cleaning and feature engineering
- **`data_validator.py`**: Input validation and data quality checks
- **`data_utils.py`**: Common data manipulation utilities
- **`cross_validation.py`**: Cross-validation strategies and evaluation

### Analysis & Reporting

- **`html_report_generator.py`**: Comprehensive HTML report generation
- **`academic_report_generator.py`**: Academic-style analysis reports
- **`visualization_generator.py`**: Chart and plot generation
- **`metrics_evaluator.py`**: Model performance metrics calculation
- **`performance_analysis.py`**: Detailed performance analysis tools

### Error Handling & Optimization

- **`xgboost_error_handler.py`**: XGBoost-specific error handling and validation
- **`xgboost_data_optimizer.py`**: Data optimization for XGBoost performance
- **`training_monitor.py`**: Training progress monitoring and callbacks
- **`performance_monitoring.py`**: Runtime performance monitoring

## MCP Tools Available

The server provides 13 core MCP tools with **unified model_id system**:

### Training Tools (Queue-based)
1. **`train_xgboost_regressor`** - Submit XGBoost regression training task to queue (returns model_id)
2. **`train_xgboost_classifier`** - Submit XGBoost classification training task to queue (returns model_id)

### Prediction Tools
3. **`predict_from_file`** - Batch predictions from CSV files (uses model_id)
4. **`predict_from_values`** - Real-time predictions from input values (uses model_id)

### Analysis Tools
5. **`analyze_global_feature_importance`** - Global feature importance analysis (uses model_id)
6. **`analyze_local_feature_importance`** - Local feature importance (SHAP/LIME) (uses model_id)

### Model Management Tools
7. **`list_models`** - List all trained models with metadata
8. **`get_model_info`** - Detailed model information and statistics (uses model_id)
9. **`delete_model`** - Remove trained models and associated files (uses model_id)

### Queue Management Tools
10. **`get_training_results`** - Get training results and status using model_id (unified tool)
11. **`list_training_tasks`** - List all training tasks with their status
12. **`get_queue_status`** - Get overall training queue status
13. **`cancel_training_task`** - Cancel a training task using model_id

## File Structure

### Models Directory (`trained_models/`)

Each trained model creates a directory structure:
```
trained_models/{model_id}/
├── model.joblib                    # Serialized XGBoost model
├── preprocessing_pipeline.pkl      # Data preprocessing pipeline
├── metadata.json                   # Model metadata and hyperparameters
├── feature_importance.csv          # Feature importance scores
├── evaluation_metrics.csv          # Performance metrics
├── cross_validation_results.json   # CV results
├── feature_analysis/               # Feature analysis outputs
│   ├── global/                     # Global importance plots
│   └── local/                      # Local importance analysis
└── optimization_history.csv        # Hyperparameter optimization history
```

### Configuration

- **`config.py`**: Server configuration (port, URLs, paths)
- Base URL: `http://localhost:8100`
- Default port: `8100`
- Models directory: `./trained_models`

## Development Patterns

### Adding New MCP Tools

When adding new MCP tools, follow this pattern in `mcp_server.py`:

```python
@mcp.tool()
def your_tool_name(param1: str, param2: int = 100) -> Dict[str, Any]:
    """Tool description for MCP clients."""
    try:
        # Implementation here
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error in your_tool_name: {str(e)}")
        return {"status": "error", "message": str(e)}
```

### Error Handling

Use the XGBoost error handler decorator for XGBoost-specific operations:

```python
from .xgboost_error_handler import xgboost_error_handler

@xgboost_error_handler
def your_xgboost_function():
    # XGBoost operations here
    pass
```

### Model ID Generation

Model IDs are generated using UUID4 format. Always use the pattern:
```python
model_id = str(uuid.uuid4())
```

### Data Validation

Always validate input data using the DataValidator:
```python
validator = DataValidator()
validation_result = validator.validate_data(df, target_column)
if not validation_result["is_valid"]:
    raise ValueError(validation_result["message"])
```

## Testing

The project includes comprehensive testing with pytest. Tests should cover:
- MCP tool functionality
- XGBoost wrapper operations  
- Data validation and preprocessing
- Error handling scenarios
- Model persistence and loading

Run tests with:
```bash
pytest tests/ -v
```

## Training Workflow (Queue-Based)

The training tools now use an asynchronous queue-based approach with **unified ID system** for better clarity and concurrency:

### 1. Submit Training Task
```python
# MCP client call
result = await mcp_client.call_tool("train_xgboost_regressor", {
    "data_source": "/path/to/data.csv",
    "target_dimension": 1,
    "optimize_hyperparameters": True,
    "n_trials": 50,
    "cv_folds": 5
})

# Returns immediately with unified model_id
model_id = result["model_id"]  # One ID for both training task and model
print(result["usage_note"])  # Guidance on using the model_id
```

### 2. Monitor Progress and Get Results
```python
# Check training status and get results using model_id (unified tool)
result = await mcp_client.call_tool("get_training_results", {
    "model_id": model_id  # Use the same model_id throughout
})

# For running/queued tasks
if result['training_status'] in ['queued', 'running']:
    print(f"Training status: {result['training_status']}")
    print(f"Model ID: {result['model_id']}")

# For completed tasks - all training data is directly available
if result['training_status'] == 'completed':
    print(f"Model ID: {result['model_id']}")
    print(f"Performance: {result['performance_summary']}")
    print(f"Training time: {result['training_time_seconds']}s")

    # Direct access to all training results without re-packaging
    feature_importance = result['feature_importance']
    cv_results = result['cross_validation_results']
    optimization_results = result['optimization_results']
    model_params = result['model_params']
    metadata = result['metadata']
```

### 3. Queue Management
```python
# Get overall queue status
queue_status = await mcp_client.call_tool("get_queue_status", {})
print(f"Running tasks: {queue_status['queue']['running_tasks']}")

# List all tasks
tasks = await mcp_client.call_tool("list_training_tasks", {})
print(f"Total tasks: {tasks['count']}")

# Cancel a training task using model_id
cancel_result = await mcp_client.call_tool("cancel_training_task", {
    "model_id": model_id  # Use model_id to cancel
})

# Check training status and results with unified tool
training_results = await mcp_client.call_tool("get_training_results", {
    "model_id": model_id  # Same model_id throughout the workflow
})
```

### Key Benefits of Unified ID System
- **Simpler for LLMs**: Only one ID to track instead of separate task_id and model_id
- **Intuitive workflow**: The same model_id is used from submission through prediction
- **Clear lifecycle**: model_id represents the model from training queue to deployment
- **Reduced confusion**: No need to remember which ID to use for which operation

## Concurrency Features

- **Non-blocking Training**: Training requests return immediately with model_id
- **Concurrent Processing**: Multiple users can train models simultaneously
- **Resource Control**: Configurable maximum concurrent tasks (default: 3)
- **Queue Management**: Tasks are queued and processed in order
- **Status Tracking**: Real-time training status and progress monitoring using model_id
- **Task Cancellation**: Cancel long-running tasks using model_id
- **Unified ID System**: Single model_id serves as both task identifier and model identifier

## Dependencies

Key dependencies and their purposes:
- **xgboost>=2.0.0**: Core ML algorithm
- **scikit-learn>=1.3.0**: ML utilities and metrics
- **pandas>=2.0.0**: Data manipulation
- **optuna>=3.4.0**: Hyperparameter optimization (runs asynchronously)
- **shap>=0.47.0**: Model explainability
- **fastapi>=0.104.0**: Web framework
- **mcp>=1.0.0**: MCP protocol implementation
- **matplotlib/seaborn/plotly**: Visualization libraries
- **asyncio**: Async queue management and concurrency control