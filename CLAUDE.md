# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an XGBoost MCP Tool - a comprehensive machine learning server that provides XGBoost capabilities through the MCP (Model Context Protocol). The project implements a FastMCP-based server with advanced XGBoost functionality for regression and classification tasks, supporting multi-user isolation, asynchronous queue-based training, and multiple transport modes (stdio, streamable-http, SSE).

## Essential Commands

### Development & Testing
```bash
# Install dependencies
uv pip install -r requirements.txt

# Run the main MCP server (stdio transport)
python -m src.mcp_xgboost_tool

# Run the single port server (streamable-http + file APIs)
python single_port_server.py --host 0.0.0.0 --port 8100

# Run the FastAPI server (stdio or SSE mode)
python run_mcp_server_fastapi.py --mode stdio
python run_mcp_server_fastapi.py --mode sse --port 8100

# Run tests (note: test suite not yet implemented)
pytest

# Type checking (if mypy is installed)
mypy src/
```

### Server Modes

The project supports three server entry points:

| Entry Point | Command | Transport |
|-------------|---------|-----------|
| Package main | `python -m src.mcp_xgboost_tool` | MCP stdio (`mcp.stdio.stdio_server`) |
| Single-port server | `python single_port_server.py` | Streamable HTTP at `/mcp` + file APIs |
| FastAPI (SSE) | `python run_mcp_server_fastapi.py --mode sse` | SSE at `/sse` |
| FastAPI (stdio) | `python run_mcp_server_fastapi.py --mode stdio` | FastMCP stdio |

**Single-port server endpoints:**

| Path | Purpose |
|------|---------|
| `/mcp` | MCP streamable-http endpoint |
| `/api/health` | Health check |
| `/api/info` | Server info |
| `/api/models/list` | List model files |
| `/api/download/file/{path}` | Download file |
| `/download/file/{path}` | Download file (alias) |
| `/static` | Static files from `trained_models/` |
| `/static/model` | Alias for static models |
| `/static/reports` | Static files from `reports/` |

## Core Architecture

### Key Components

The codebase follows a modular architecture centered around XGBoost ML capabilities (~30k lines of Python):

- **`mcp_server.py`**: Main MCP server implementation using FastMCP; defines all 14 tools
- **`xgboost_wrapper.py`** (964 lines): Core XGBoost algorithm wrapper for regression/classification with CV integration
- **`training.py`** (1,223 lines): Training engine with hyperparameter tuning and report generation
- **`prediction.py`** (2,264 lines): Batch and real-time prediction with report generation
- **`feature_importance.py`** (2,029 lines): Global feature importance (tree, permutation, SHAP)
- **`local_feature_importance.py`** (2,988 lines): Local/instance-level feature importance (SHAP waterfall, force, decision plots)
- **`feature_interaction_analyzer.py`**: Feature interaction decoupling — targeted synergy/antagonism region extraction with Gaussian smoothing (distinct from global SHAP interaction heatmaps)
- **`hyperparameter_optimizer.py`** (436 lines): Optuna-based hyperparameter optimization (TPE/GP)
- **`model_manager.py`** (361 lines): Model persistence, metadata, and lifecycle management
- **`training_queue.py`** (381 lines): Async queue manager for concurrent training tasks

### Data Processing Pipeline

- **`data_preprocessing.py`** (574 lines): Preprocessing pipeline (scaling, imputation, encoding)
- **`data_validator.py`** (2,716 lines): Data validation, integrity checks, leakage detection, quality scoring
- **`data_utils.py`** (585 lines): Data loading, encoding detection, validation utilities
- **`cross_validation.py`** (977 lines): Cross-validation with StratifiedKFold/KFold

### Analysis & Reporting

- **`html_report_generator.py`** (3,878 lines): Comprehensive HTML and JSON report generation
- **`academic_report_generator.py`** (2,920 lines): Academic-style analysis reports (validation, hyperparams, feature importance)
- **`visualization_generator.py`** (2,167 lines): Charts and plots (feature importance, CV, comparisons)
- **`metrics_evaluator.py`** (187 lines): Model performance metrics for regression and classification
- **`performance_analysis.py`** (562 lines): Performance analysis and benchmarking

### Error Handling & Optimization

- **`xgboost_error_handler.py`** (438 lines): XGBoost-specific error handling and validation decorators
- **`xgboost_data_optimizer.py`** (375 lines): DMatrix conversion, memory handling, dtype optimization
- **`training_monitor.py`** (368 lines): Training progress monitoring and callbacks
- **`performance_monitoring.py`** (1,055 lines): Long-term monitoring, degradation detection, retraining suggestions

### Other Modules

- **`config.py`** (11 lines): Server configuration (BASE_URL, MCP_PORT, URL helpers)
- **`allow_uesr.py`** (8 lines): Simple user access control via `AUTHORIZED_USERS` whitelist

## MCP Tools Available

The server provides 14 core MCP tools with **unified model_id system**.

> **Important**: The actual registered tool names use `xgboost` prefix. The table below shows the exact function names as registered with FastMCP.

### Training Tools (Queue-based)
1. **`train_xgboost_regressor`** - Submit XGBoost regression training task (returns model_id)
   - Params: `data_source`, `target_dimension=1`, `optimize_hyperparameters=True`, `n_trials=50`, `cv_folds=5`, `scoring_metric="MAE"`, `validate_data=True`, `save_model=True`, `apply_preprocessing=True`, `scaling_method="standard"`, `enable_gpu=True`, `device="auto"`
2. **`train_xgboost_classifier`** - Submit XGBoost classification training task (returns model_id)
   - Params: `data_source`, `target_dimension=1`, `optimize_hyperparameters=True`, `n_trials=50`, `cv_folds=5`, `scoring_metric="f1_weighted"`, `apply_preprocessing=True`, `scaling_method="standard"`, `validate_data=True`, `save_model=True`, `enable_gpu=True`, `device="auto"`

### Prediction Tools
3. **`predict_from_file_xgbost`** - Batch predictions from CSV files (uses model_id)
   - Params: `model_id`, `data_source`, `output_path=None`, `include_confidence=True`, `generate_report=True`
4. **`predict_from_values_xgboost`** - Real-time predictions from feature values (uses model_id)
   - Params: `model_id`, `feature_values`, `feature_names=None`, `include_confidence=True`, `save_intermediate_files=True`, `generate_report=True`, `output_path=None`

### Analysis Tools
5. **`analyze_xgboost_global_feature_importance`** - Global feature importance analysis (SHAP, basic, permutation)
   - Params: `model_id`, `analysis_types=["shap"]`, `generate_plots=True`, `generate_report=True`
6. **`analyze_xgboost_local_feature_importance`** - Local feature importance (SHAP waterfall, force, decision)
   - Params: `model_id`, `sample_data=None`, `data_source=None`, `plot_types=["waterfall","force","decision"]`, `generate_plots=True`, `generate_report=True`

### Feature Interaction Decoupling
7. **`decouple_xgboost_feature_interaction`** - Targeted decoupling of a specific feature pair into synergy/antagonism regions
   - Params: `model_id`, `feature_1`, `feature_2`, `grid_resolution=30`, `smoothing_sigma=None`, `generate_plots=True`
   - Unlike the raw SHAP interaction heatmaps in global feature importance (qualitative overview of ALL pairs), this tool performs quantitative region analysis on ONE specified pair using Gaussian-smoothed boundary extraction

### Model Management Tools
8. **`list_xgboost_models`** - List all trained models with metadata
9. **`get_xgboost_model_info`** - Detailed model information and statistics (uses model_id)
10. **`delete_xgboost_model`** - Remove trained models and associated files (uses model_id)

### Queue Management Tools
11. **`get_xgboost_training_results`** - Get training results and status using model_id (unified tool)
12. **`list_xgboost_training_tasks`** - List all training tasks with their status (optional `user_id` filter)
13. **`get_xgboost_queue_status`** - Get overall training queue status
14. **`cancel_xgboost_training_task`** - Cancel a training task using model_id

### Multi-User Isolation

Most tools accept `ctx: Context` and extract `user_id` from request headers (`ctx.request_context.request.headers.get("user_id")`). This enables per-user model directories under `trained_models/{user_id}/`.

## File Structure

### Project Root
```
MCP_XGBOOST_Tool/
├── src/mcp_xgboost_tool/          # Main package (all core modules)
├── trained_models/                 # Model storage ({user_id}/{model_id}/ subdirs)
├── queue/                          # Training task queue JSON files
├── prediction_reports/             # Generated prediction reports
├── single_port_server.py           # Single-port HTTP+MCP server
├── run_mcp_server_fastapi.py       # FastAPI-based MCP server (stdio/SSE)
├── main.py                         # Placeholder entry point
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project metadata (openpyxl, xlrd extras)
├── uv.lock                         # uv lockfile
├── CLAUDE.md                       # This file
├── README.md                       # Project overview
├── FEATURE_IMPORTANCE_ENHANCEMENT.md  # SHAP v2.0 enhancements doc
└── LICENSE                         # MIT license
```

### Models Directory (`trained_models/`)

Each trained model creates a directory structure (optionally under a user_id subdirectory):
```
trained_models/[{user_id}/]{model_id}/
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

- **`config.py`**: Server configuration
- Base URL: `https://www.matterai.cn/xgboost` (production); `http://localhost:8100` (local, commented out)
- Default port: `8100`
- Models directory: `./trained_models`
- Helper functions: `get_download_url(path)`, `get_static_url(path)`

## Development Patterns

### Adding New MCP Tools

When adding new MCP tools, follow this pattern in `mcp_server.py`:

```python
@mcp.tool()
async def your_tool_name(param1: str, param2: int = 100, ctx: Context = None) -> str:
    """Tool description for MCP clients."""
    try:
        user_id = None
        if ctx and hasattr(ctx, 'request_context'):
            user_id = ctx.request_context.request.headers.get("user_id")
        # Implementation here
        return json.dumps({"status": "success", "result": result}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error in your_tool_name: {str(e)}")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)
```

### Error Handling

Use the XGBoost error handler decorator for XGBoost-specific operations:

```python
from .xgboost_error_handler import xgboost_error_handler

@xgboost_error_handler
def your_xgboost_function():
    pass
```

### Model ID Generation

Model IDs are generated using UUID4 format:
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

Test infrastructure (pytest, pytest-asyncio, pytest-cov) is included in `requirements.txt`, but **no test files have been created yet**. The `tests/` directory does not exist. Tests should cover:
- MCP tool functionality
- XGBoost wrapper operations
- Data validation and preprocessing
- Error handling scenarios
- Model persistence and loading

## Training Workflow (Queue-Based)

The training tools use an asynchronous queue-based approach with **unified ID system** for better clarity and concurrency:

### 1. Submit Training Task
```python
result = await mcp_client.call_tool("train_xgboost_regressor", {
    "data_source": "/path/to/data.csv",
    "target_dimension": 1,
    "optimize_hyperparameters": True,
    "n_trials": 50,
    "cv_folds": 5
})
model_id = result["model_id"]
```

### 2. Monitor Progress and Get Results
```python
result = await mcp_client.call_tool("get_xgboost_training_results", {
    "model_id": model_id
})

if result['training_status'] == 'completed':
    print(f"Performance: {result['performance_summary']}")
    feature_importance = result['feature_importance']
    cv_results = result['cross_validation_results']
    optimization_results = result['optimization_results']
```

### 3. Queue Management
```python
queue_status = await mcp_client.call_tool("get_xgboost_queue_status", {})
tasks = await mcp_client.call_tool("list_xgboost_training_tasks", {})
cancel_result = await mcp_client.call_tool("cancel_xgboost_training_task", {
    "model_id": model_id
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
- **User Isolation**: Per-user model directories via request header `user_id`

## Dependencies

Key dependencies (from `requirements.txt`), requires Python >= 3.12:

- **xgboost>=2.0.0**: Core ML algorithm
- **scikit-learn>=1.3.0**: ML utilities and metrics
- **pandas>=2.0.0**: Data manipulation
- **numpy>=1.24.0**: Numerical computing
- **optuna>=3.4.0**: Hyperparameter optimization (runs asynchronously)
- **shap>=0.47.0**: Model explainability
- **fastapi>=0.104.0**: Web framework for SSE/HTTP servers
- **fastmcp>=1.0.0**: FastMCP protocol implementation
- **uvicorn>=0.24.0**: ASGI server
- **joblib>=1.3.0**: Model serialization
- **jinja2>=3.1.0**: Template rendering for reports
- **pydantic>=2.4.0**: Data validation
- **psutil>=5.9.0**: System resource monitoring
- **chardet>=5.0.0**: File encoding detection
- **scipy>=1.11.0**: Scientific computing
- **matplotlib>=3.7.0 / seaborn>=0.12.0 / plotly>=5.15.0**: Visualization libraries
- **openpyxl>=3.1.5 / xlrd>=2.0.2**: Excel file support (in pyproject.toml)

## Known Issues

- `allow_uesr.py` filename contains a typo (should be `allow_user.py`)
- `predict_from_file_xgbost` tool name has a typo (missing 'o' in `xgboost`)
- `__main__.py` docstring still references "MCP Neural Network Tool"
- `__init__.py` has all imports commented out with legacy Random Forest references
- No test suite exists despite pytest being in dependencies
