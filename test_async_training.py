#!/usr/bin/env python
"""
Test script for verifying the async training implementation.
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import os

# Disable matplotlib GUI backend for testing
os.environ['MPLBACKEND'] = 'Agg'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from mcp_xgboost_tool.training import TrainingEngine


async def test_async_training():
    """Test the async training implementation."""
    
    # Create a simple test dataset
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    # Create regression data
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.1
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Save test data to CSV
    test_data_path = Path(__file__).parent / 'test_data.csv'
    df.to_csv(test_data_path, index=False)
    
    logger.info(f"Test data created with shape: {df.shape}")
    logger.info(f"Test data saved to: {test_data_path}")
    
    # Initialize training engine
    training_engine = TrainingEngine()
    
    
    # Initialize paths
    test_class_path = None
    
    try:
        # Test 1: Basic async training (regression)
        logger.info("\n=== Test 1: Basic Async Regression Training ===")
        result = await training_engine.train_xgboost(
            data_source=str(test_data_path),
            target_column='target',
            optimize_hyperparameters=False,  # Skip optimization for quick test
            n_trials=10,
            cv_folds=3,
            validate_data=True,
            save_model=True,
            apply_preprocessing=True,
            task_type='regression'
        )
        
        logger.info(f"✅ Regression training completed successfully!")
        logger.info(f"Model ID: {result.get('model_id')}")
        logger.info(f"Training time: {result.get('training_time_seconds', 0):.2f} seconds")
        logger.info(f"Performance summary: {result.get('performance_summary')}")
        
        # Test 2: Classification with async training
        logger.info("\n=== Test 2: Basic Async Classification Training ===")
        
        # Create classification data
        y_class = (y > y.mean()).astype(int)
        df['target_class'] = y_class
        test_class_path = Path(__file__).parent / 'test_class_data.csv'
        df.to_csv(test_class_path, index=False)
        
        result_class = await training_engine.train_xgboost_classification(
            data_source=str(test_class_path),
            target_column='target_class',
            optimize_hyperparameters=False,
            n_trials=10,
            cv_folds=3,
            validate_data=True,
            save_model=True,
            apply_preprocessing=True
        )
        
        logger.info(f"✅ Classification training completed successfully!")
        logger.info(f"Model ID: {result_class.get('model_id')}")
        logger.info(f"Training time: {result_class.get('training_time_seconds', 0):.2f} seconds")
        logger.info(f"Performance summary: {result_class.get('performance_summary')}")
        
        # Test 3: Concurrent training (test async capabilities)
        logger.info("\n=== Test 3: Concurrent Training Test ===")
        
        tasks = []
        for i in range(2):
            task = training_engine.train_xgboost(
                data_source=str(test_data_path),
                target_column='target',
                optimize_hyperparameters=False,
                n_trials=5,
                cv_folds=2,
                validate_data=False,  # Skip validation for speed
                save_model=False,  # Don't save for this test
                apply_preprocessing=True,
                task_type='regression'
            )
            tasks.append(task)
        
        # Run tasks concurrently
        results = await asyncio.gather(*tasks)
        
        logger.info(f"✅ Concurrent training completed!")
        for i, res in enumerate(results):
            logger.info(f"  Task {i+1}: {res.get('performance_summary')}")
        
        logger.info("\n✅ All async training tests passed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test files
        if test_data_path.exists():
            test_data_path.unlink()
        if test_class_path and test_class_path.exists():
            test_class_path.unlink()
            
    return True


async def main():
    """Main test runner."""
    logger.info("Starting async training tests...")
    success = await test_async_training()
    
    if success:
        logger.info("\n🎉 All tests passed! The async training implementation is working correctly.")
    else:
        logger.error("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())