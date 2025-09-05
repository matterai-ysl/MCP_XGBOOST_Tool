#!/usr/bin/env python3
"""
Concurrent Training Test Script

This script tests the concurrent training capabilities of the XGBoost MCP Tool
by submitting multiple training tasks simultaneously and monitoring their progress.
"""

import asyncio
import time
import logging
from typing import List, Dict, Any
from src.mcp_xgboost_tool.training_queue import TrainingQueueManager
from src.mcp_xgboost_tool.training import TrainingEngine
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def create_test_data(filename: str, n_samples: int = 1000, n_features: int = 10, task_type: str = "regression"):
    """Create synthetic test data for training."""
    logger.info(f"Creating test data: {filename} ({task_type})")
    
    # Generate random features
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    if task_type == "regression":
        # Simple linear relationship with noise
        y = X[:, 0] * 2 + X[:, 1] * -1.5 + X[:, 2] * 0.8 + np.random.randn(n_samples) * 0.1
    else:  # classification
        # Binary classification based on feature combinations
        y = ((X[:, 0] + X[:, 1] * 0.5) > 0).astype(int)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Save to CSV
    df.to_csv(filename, index=False)
    logger.info(f"Test data saved to {filename}")
    return filename

async def test_single_training():
    """Test single training task with async engine."""
    logger.info("Testing single async training...")
    
    # Create test data
    test_file = await create_test_data("test_data_single.csv", n_samples=500, task_type="regression")
    
    # Create training engine
    training_engine = TrainingEngine()
    
    # Test async training
    start_time = time.time()
    result = await training_engine.train_xgboost_async(
        data_source=test_file,
        target_column="target",
        task_type="regression",
        optimize_hyperparameters=True,
        n_trials=20,  # Reduce trials for faster testing
        cv_folds=3,
        save_model=False  # Don't save models for testing
    )
    end_time = time.time()
    
    logger.info(f"Single training completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Model performance: {result.get('performance_summary', 'N/A')}")
    
    return result

async def test_concurrent_training_with_queue(num_tasks: int = 3, max_concurrent: int = 2):
    """Test concurrent training using the queue manager."""
    logger.info(f"Testing concurrent training with queue ({num_tasks} tasks, {max_concurrent} concurrent)")
    
    # Create queue manager
    queue_manager = TrainingQueueManager(max_concurrent_tasks=max_concurrent)
    await queue_manager.start()
    
    try:
        # Create different test datasets
        test_files = []
        for i in range(num_tasks):
            task_type = "regression" if i % 2 == 0 else "classification"
            filename = f"test_data_concurrent_{i}_{task_type}.csv"
            test_file = await create_test_data(
                filename, 
                n_samples=300 + i * 100, 
                task_type=task_type
            )
            test_files.append((test_file, task_type))
        
        # Submit tasks
        task_ids = []
        start_time = time.time()
        
        for i, (test_file, task_type) in enumerate(test_files):
            task_id = await queue_manager.submit_task(
                task_type=task_type,
                params={
                    'data_source': test_file,
                    'target_column': 'target',
                    'task_type': task_type,
                    'optimize_hyperparameters': True,
                    'n_trials': 15,  # Reduce for faster testing
                    'cv_folds': 3,
                    'save_model': False
                },
                user_id=f"user_{i}"
            )
            task_ids.append(task_id)
            logger.info(f"Submitted task {i+1}/{num_tasks}: {task_id} ({task_type})")
        
        # Monitor progress
        completed_tasks = 0
        while completed_tasks < num_tasks:
            await asyncio.sleep(2)  # Check every 2 seconds
            
            # Get status of all tasks
            running_count = 0
            for task_id in task_ids:
                status = await queue_manager.get_task_status(task_id)
                if status:
                    if status['status'] == 'completed':
                        if task_id not in [t for t in task_ids if hasattr(t, '_completed')]:
                            logger.info(f"Task {task_id} completed: {status['status']}")
                            completed_tasks += 1
                            setattr(task_id, '_completed', True)
                    elif status['status'] == 'running':
                        running_count += 1
                    elif status['status'] == 'failed':
                        logger.error(f"Task {task_id} failed: {status.get('error_message', 'Unknown error')}")
                        completed_tasks += 1
            
            # Get queue status
            queue_status = await queue_manager.get_queue_status()
            logger.info(f"Queue status: {queue_status['running_tasks']} running, {queue_status['queued_tasks']} queued")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"All {num_tasks} tasks completed in {total_time:.2f} seconds")
        logger.info(f"Average time per task: {total_time/num_tasks:.2f} seconds")
        
        # Get final results
        results = []
        for task_id in task_ids:
            status = await queue_manager.get_task_status(task_id)
            if status and status['status'] == 'completed':
                results.append(status['result'])
        
        logger.info(f"Successfully completed {len(results)} out of {num_tasks} tasks")
        
        return results
        
    finally:
        await queue_manager.stop()

async def test_performance_comparison():
    """Compare sequential vs concurrent training performance."""
    logger.info("Performance comparison: Sequential vs Concurrent")
    
    num_tasks = 4
    
    # Test 1: Sequential training (simulated)
    logger.info("Testing sequential training pattern...")
    sequential_start = time.time()
    
    training_engine = TrainingEngine()
    sequential_results = []
    
    for i in range(num_tasks):
        task_type = "regression" if i % 2 == 0 else "classification"
        test_file = await create_test_data(
            f"test_seq_{i}_{task_type}.csv", 
            n_samples=400, 
            task_type=task_type
        )
        
        result = await training_engine.train_xgboost_async(
            data_source=test_file,
            target_column="target",
            task_type=task_type,
            optimize_hyperparameters=True,
            n_trials=10,
            cv_folds=3,
            save_model=False
        )
        sequential_results.append(result)
        logger.info(f"Sequential task {i+1}/{num_tasks} completed")
    
    sequential_time = time.time() - sequential_start
    
    # Test 2: Concurrent training with queue
    logger.info("Testing concurrent training with queue...")
    concurrent_results = await test_concurrent_training_with_queue(
        num_tasks=num_tasks, 
        max_concurrent=3
    )
    
    # Results
    logger.info("=== PERFORMANCE COMPARISON ===")
    logger.info(f"Sequential training: {sequential_time:.2f} seconds")
    logger.info(f"Concurrent training: {concurrent_time:.2f} seconds")
    logger.info(f"Speedup: {sequential_time/concurrent_time:.2f}x")
    logger.info(f"Sequential tasks completed: {len(sequential_results)}/{num_tasks}")
    logger.info(f"Concurrent tasks completed: {len(concurrent_results)}/{num_tasks}")

async def test_queue_management_features():
    """Test queue management features like cancellation and status tracking."""
    logger.info("Testing queue management features...")
    
    queue_manager = TrainingQueueManager(max_concurrent_tasks=2)
    await queue_manager.start()
    
    try:
        # Create test data
        test_file = await create_test_data("test_queue_mgmt.csv", n_samples=800, task_type="regression")
        
        # Submit multiple tasks
        task_ids = []
        for i in range(5):  # Submit 5 tasks
            task_id = await queue_manager.submit_task(
                task_type="regression",
                params={
                    'data_source': test_file,
                    'target_column': 'target',
                    'task_type': 'regression',
                    'optimize_hyperparameters': True,
                    'n_trials': 25,  # Longer tasks for testing cancellation
                    'cv_folds': 5,
                    'save_model': False
                },
                user_id=f"test_user_{i}"
            )
            task_ids.append(task_id)
        
        logger.info(f"Submitted {len(task_ids)} tasks")
        
        # Wait a bit for some tasks to start
        await asyncio.sleep(3)
        
        # Test cancellation
        if len(task_ids) >= 3:
            cancel_task_id = task_ids[2]  # Cancel the 3rd task
            cancelled = await queue_manager.cancel_task(cancel_task_id)
            logger.info(f"Cancelled task {cancel_task_id}: {cancelled}")
        
        # List tasks for specific user
        user_tasks = await queue_manager.list_tasks(user_id="test_user_1")
        logger.info(f"Tasks for test_user_1: {len(user_tasks)}")
        
        # Get overall queue status
        queue_status = await queue_manager.get_queue_status()
        logger.info(f"Queue status: {queue_status}")
        
        # Wait for completion or timeout
        timeout = 60  # 1 minute timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            queue_status = await queue_manager.get_queue_status()
            if queue_status['running_tasks'] == 0 and queue_status['queued_tasks'] == 0:
                break
            await asyncio.sleep(2)
        
        # Final status check
        final_tasks = await queue_manager.list_tasks()
        status_counts = {}
        for task in final_tasks:
            status = task['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        logger.info(f"Final task status counts: {status_counts}")
        
    finally:
        await queue_manager.stop()

async def main():
    """Main test function."""
    logger.info("Starting XGBoost MCP Tool concurrent training tests...")
    
    try:
        # Test 1: Single async training
        await test_single_training()
        
        # Test 2: Concurrent training with queue
        await test_concurrent_training_with_queue(num_tasks=3, max_concurrent=2)
        
        # Test 3: Queue management features
        await test_queue_management_features()
        
        # Test 4: Performance comparison (commented out for now due to time)
        # await test_performance_comparison()
        
        logger.info("All concurrent training tests completed successfully! 🎉")
        
    except Exception as e:
        logger.error(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())