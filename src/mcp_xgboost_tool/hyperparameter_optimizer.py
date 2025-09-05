# -*- coding: utf-8 -*-
"""
Hyperparameter Optimization Module

This module provides hyperparameter optimization functionality using Optuna
for XGBoost models, supporting TPE and GP algorithms.
"""

import logging
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from xgboost import XGBClassifier, XGBRegressor
import optuna
from optuna.samplers import TPESampler
import warnings

from .xgboost_wrapper import XGBoostWrapper

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """
    Hyperparameter optimization for XGBoost models using Optuna.
    Based on simplified and robust design from reference implementations.
    """
    
    def __init__(self, 
                 sampler_type: str = "TPE",
                 n_trials: int = 50,
                 cv_folds: int = 5,
                 random_state: Optional[int] = 42,
                 enable_gpu: bool = True,
                 device: str = "auto"):
        """Initialize HyperparameterOptimizer with GPU support."""
        self.sampler_type = sampler_type.upper()
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.enable_gpu = enable_gpu
        self.device = device        
        # Optimization state
        self.study = None
        self.best_params = None
        self.best_score = None
        self.task_type = None
        
        # Create sampler
        self.sampler = self._create_sampler()
        
    def _create_sampler(self):
        """Create Optuna sampler based on configuration."""
        if self.sampler_type == "TPE":
            return TPESampler(
                n_startup_trials=10, 
                n_ei_candidates=24,
                seed=self.random_state
            )
        elif self.sampler_type == "GP":
            try:
                from optuna.integration import SkoptSampler
                return SkoptSampler(
                    skopt_kwargs={
                        'base_estimator': 'GP',
                        'n_initial_points': 10,
                        'acq_func': 'EI'
                    }
                )
            except ImportError:
                logger.warning("Scikit-optimize not available, falling back to TPE")
                return TPESampler(seed=self.random_state)
        else:
            logger.warning(f"Unknown sampler type {self.sampler_type}, using TPE")
            return TPESampler(seed=self.random_state)
            
    def _suggest_hyperparameters(self, trial) -> Dict[str, Any]:
        """Suggest hyperparameters for XGBoost optimization."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150, step=10),
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),
            'random_state': self.random_state
        }
        return params
        
    def _create_objective_function(self, X: np.ndarray, y: np.ndarray, 
                                 task_type: str, scoring_metric: str) -> Callable:
        """Create the objective function for Optuna optimization."""
            
        # Create cross-validation splitter
        if task_type == "classification":
            cv_splitter = StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )
        else:
            cv_splitter = KFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )
            
        def objective(trial):
            """Optuna objective function."""
            try:
                # Get suggested hyperparameters
                params = self._suggest_hyperparameters(trial)
                
                # Add GPU support parameters
                gpu_params = {}
                if self.enable_gpu and self.device in ["auto", "cuda", "gpu"]:
                    # Try to use GPU if available  
                    try:
                        # Test GPU availability by creating a simple wrapper
                        from .xgboost_wrapper import XGBoostWrapper
                        test_wrapper = XGBoostWrapper(enable_gpu=True, device=self.device)
                        if test_wrapper.gpu_available:
                            gpu_params.update({
                                'tree_method': 'hist',
                                'device': 'cuda'
                            })
                        else:
                            gpu_params.update({
                                'tree_method': 'hist',
                                'device': 'cpu'
                            })
                    except Exception:
                        # Fall back to CPU if GPU setup fails
                        gpu_params.update({
                            'tree_method': 'hist',
                            'device': 'cpu'
                        })
                else:
                    gpu_params.update({
                        'tree_method': 'hist',
                        'device': 'cpu'
                    })
                
                # Create XGBoost model with GPU support
                final_params = {**params, **gpu_params, 'n_jobs': 1, 'verbosity': 0}
                if task_type == "classification":
                    model = XGBClassifier(**final_params)
                else:
                    model = XGBRegressor(**final_params)
                
                # Perform cross-validation
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scores = cross_val_score(
                        model, X, y,
                        cv=cv_splitter,
                        scoring=scoring_metric,
                        n_jobs=1
                    )
                
                mean_score = np.mean(scores)
                
                # Validate the score
                if np.isnan(mean_score) or np.isinf(mean_score):
                    logger.warning(f"Trial {trial.number} produced invalid score: {mean_score}")
                    raise optuna.exceptions.TrialPruned()
                
                # Store additional trial information
                trial.set_user_attr("cv_scores", scores.tolist())
                trial.set_user_attr("cv_std", float(np.std(scores)))
                trial.set_user_attr("params", params)
                
                return mean_score
                
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {str(e)}")
                raise optuna.exceptions.TrialPruned()
        
        return objective

    def optimize(self, 
                X: Union[np.ndarray, pd.DataFrame], 
                y: np.ndarray,
                task_type: Optional[str] = None,
                scoring_metric: Optional[str] = None,
                save_dir: Optional[str] = None) -> Tuple[Dict[str, Any], float, Optional[pd.DataFrame]]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X: Feature matrix
            y: Target variable
            task_type: 'classification' or 'regression' (auto-detected if None)
            scoring_metric: Scoring metric for evaluation
            save_dir: Directory to save optimization history CSV
            
        Returns:
            Tuple of (best_params, best_score, trials_dataframe)
        """
        
        # Input validation
        if X is None or len(X) == 0:
            raise ValueError("X cannot be None or empty")
        if y is None or len(y) == 0:
            raise ValueError("y cannot be None or empty")
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length. Got X: {len(X)}, y: {len(y)}")
            
        # Convert DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Check for NaN values
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values. Please clean your data first.")
        
        # Handle NaN values in target - different approach for categorical vs numeric targets
        try:
            if np.any(np.isnan(y)):
                raise ValueError("y contains NaN values. Please clean your data first.")
        except TypeError:
            # If np.isnan fails (e.g., for categorical/string targets), use pandas isna
            if pd.Series(y).isna().any():
                raise ValueError("y contains NaN values. Please clean your data first.")
            
        # Auto-detect task type if not provided
        if task_type is None:
            xgb_temp = XGBoostWrapper()
            task_type = xgb_temp._detect_task_type(y)
            
        self.task_type = task_type
        
        # Set default scoring metric
        if scoring_metric is None:
            scoring_metric = "f1_weighted" if task_type == "classification" else "r2"
            
        logger.info(f"Starting optimization for {task_type} task")
        logger.info(f"Data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Using {self.sampler_type} algorithm with {self.n_trials} trials")
        logger.info(f"Scoring metric: {scoring_metric}")
        
        # Create study - determine optimization direction based on scoring metric
        # Metrics that should be maximized (higher is better)
        maximize_metrics = [
            "accuracy", "f1", "f1_weighted", "f1_macro", "f1_micro", 
            "r2", "roc_auc", "roc_auc_ovr", "roc_auc_ovo",
            "neg_mean_absolute_error", "neg_mean_squared_error", 
            "neg_root_mean_squared_error", "neg_median_absolute_error"
        ]
        direction = "maximize" if scoring_metric in maximize_metrics else "minimize"
        self.study = optuna.create_study(direction=direction, sampler=self.sampler)
        
        # Create objective function
        objective = self._create_objective_function(X, y, task_type, scoring_metric)
        
        # Run optimization with progress bar (synchronous)
        self.study.optimize(
            objective, 
            n_trials=self.n_trials, 
            show_progress_bar=True
        )
        
        # Check if any trials completed successfully
        completed_trials = [trial for trial in self.study.trials 
                          if trial.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed_trials:
            raise ValueError(
                f"No trials completed successfully out of {len(self.study.trials)} trials. "
                "This may be due to data issues or invalid parameters."
            )
        
        # Store results
        self.best_params = self.study.best_params.copy()
        self.best_score = self.study.best_value
        
        # Get trials dataframe
        trials_df = self.study.trials_dataframe()
        
        # Save trials results if save_dir provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            csv_path = os.path.join(save_dir, 'optimization_history.csv')
            trials_df.to_csv(csv_path, index=False)
            logger.info(f"Optimization history saved to {csv_path}")
        
        logger.info(f"Optimization completed:")
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best score: {self.best_score}")
        
        return self.best_params, self.best_score, trials_df

    async def optimize_async(self, 
                           X: Union[np.ndarray, pd.DataFrame], 
                           y: np.ndarray,
                           task_type: Optional[str] = None,
                           scoring_metric: Optional[str] = None,
                           save_dir: Optional[str] = None) -> Tuple[Dict[str, Any], float, Optional[pd.DataFrame]]:
        """
        Asynchronous version of hyperparameter optimization using Optuna.
        
        This method runs the optimization in a separate thread pool to prevent
        blocking the event loop during long-running optimization tasks.
        
        Args:
            X: Feature matrix
            y: Target variable
            task_type: 'classification' or 'regression' (auto-detected if None)
            scoring_metric: Scoring metric for evaluation
            save_dir: Directory to save optimization history CSV
            
        Returns:
            Tuple of (best_params, best_score, trials_dataframe)
        """
        logger.info("Starting asynchronous hyperparameter optimization...")
        
        # Run the synchronous optimization method in a thread pool
        loop = asyncio.get_event_loop()
        
        # Use a dedicated thread pool for CPU-intensive optimization
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="optuna-") as executor:
            result = await loop.run_in_executor(
                executor,
                self.optimize,
                X, y, task_type, scoring_metric, save_dir
            )
        
        logger.info("Asynchronous hyperparameter optimization completed")
        return result

    def get_optimization_results(self) -> Dict[str, Any]:
        """Get comprehensive optimization results."""
        if self.study is None:
            return {}
            
        best_trial = self.study.best_trial
        
        # Create optimization history
        optimization_history = [
            {
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'datetime': trial.datetime_start.isoformat() if trial.datetime_start else None,
                'duration': (trial.datetime_complete - trial.datetime_start).total_seconds() 
                           if trial.datetime_complete and trial.datetime_start else None
            }
            for trial in self.study.trials
        ]
        
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_trial_number': best_trial.number,
            'n_trials': len(self.study.trials),
            'n_completed_trials': len([t for t in self.study.trials 
                                     if t.state == optuna.trial.TrialState.COMPLETE]),
            'task_type': self.task_type,
            'sampler_type': self.sampler_type,
            'optimization_history': optimization_history
        }
        
        # Add cross-validation details from best trial
        if hasattr(best_trial, 'user_attrs'):
            if 'cv_scores' in best_trial.user_attrs:
                results['best_cv_scores'] = best_trial.user_attrs['cv_scores']
                results['best_cv_std'] = best_trial.user_attrs['cv_std']
                
        return results
        
    def create_optimized_model(self) -> XGBoostWrapper:
        """Create XGBoostWrapper with optimized hyperparameters."""
        if self.best_params is None:
            raise ValueError("No optimization results available. Run optimize() first.")
            
        # Create a new XGBoostWrapper with the best found parameters
        optimized_params = self.study.best_params.copy()
        
        # Add GPU/device settings back in if they were used
        if self.enable_gpu is not None:
            optimized_params['enable_gpu'] = self.enable_gpu
        if self.device is not None:
            optimized_params['device'] = self.device

        xgb_model = XGBoostWrapper(task_type=self.task_type, **optimized_params)
        
        return xgb_model


def optimizer_optuna(n_trials: int, algo: str, data: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    """
    Simplified optimization function following reference Optimizer.py pattern.
    
    Args:
        n_trials: Number of optimization trials
        algo: Algorithm type ("TPE" or "GP")
        data: Dictionary containing X, y, task_type, scoring_metric
        
    Returns:
        Tuple of (best_params, best_score)
    """
    
    # Extract data
    X = data['X']
    y = data['y']
    task_type = data.get('task_type')
    scoring_metric = data.get('scoring_metric')
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        sampler_type=algo,
        n_trials=n_trials
    )
    
    # Run optimization
    best_params, best_score, _ = optimizer.optimize(
        X=X, 
        y=y, 
        task_type=task_type, 
        scoring_metric=scoring_metric
    )
    
    print("\n" * 2)
    print("Best params:", best_params)
    print("\n" * 2)
    print("Best score:", best_score)
    print()
    
    return best_params, best_score