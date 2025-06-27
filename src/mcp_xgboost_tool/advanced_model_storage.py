"""
Advanced Model Storage Management System

This module provides enhanced model storage capabilities including:
- Multi-format serialization
- Version control
- Performance benchmarking
- Secure file management
- Model integrity verification
"""

import logging
import json
import pickle
import joblib
import hashlib
import gzip
import os
import shutil
import time
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
from datetime import datetime
from pathlib import Path
import uuid
import numpy as np
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SerializationFormat(Enum):
    """Supported serialization formats."""
    JOBLIB = "joblib"
    PICKLE = "pickle"
    JOBLIB_COMPRESSED = "joblib_gz"
    PICKLE_COMPRESSED = "pickle_gz"


@dataclass
class ModelVersion:
    """Model version information."""
    version: str
    created_at: str
    model_path: str
    metadata_path: str
    checksum: str
    size_bytes: int
    description: Optional[str] = None
    parent_version: Optional[str] = None


@dataclass
class ModelPackage:
    """Complete model package including all components."""
    model_object: Any
    preprocessors: Dict[str, Any]
    feature_metadata: Dict[str, Any]
    training_config: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    model_id: str
    model_name: str
    task_type: str
    created_at: str
    version: str


@dataclass
class BenchmarkResult:
    """Performance benchmark result."""
    operation: str
    model_id: str
    version: str
    duration_seconds: float
    memory_used_mb: float
    file_size_bytes: int
    cpu_percent: float
    timestamp: str
    metadata: Dict[str, Any]


class ModelSerializer:
    """Advanced model serialization with multiple format support."""
    
    def __init__(self, compression_level: int = 6):
        """
        Initialize ModelSerializer.
        
        Args:
            compression_level: Compression level for compressed formats (1-9)
        """
        self.compression_level = compression_level
        
    def serialize_model(
        self, 
        model_package: ModelPackage, 
        output_path: Path,
        format_type: SerializationFormat = SerializationFormat.JOBLIB
    ) -> Dict[str, Any]:
        """
        Serialize a complete model package.
        
        Args:
            model_package: Complete model package to serialize
            output_path: Path to save the serialized model
            format_type: Serialization format to use
            
        Returns:
            Serialization result information
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare serialization data
            serialization_data = {
                'model_object': model_package.model_object,
                'preprocessors': model_package.preprocessors,
                'feature_metadata': model_package.feature_metadata,
                'training_config': model_package.training_config,
                'performance_metrics': model_package.performance_metrics,
                'package_metadata': {
                    'model_id': model_package.model_id,
                    'model_name': model_package.model_name,
                    'task_type': model_package.task_type,
                    'created_at': model_package.created_at,
                    'version': model_package.version,
                    'serialization_format': format_type.value,
                    'serialization_timestamp': datetime.now().isoformat()
                }
            }
            
            # Serialize based on format
            if format_type == SerializationFormat.JOBLIB:
                joblib.dump(serialization_data, output_path)
            elif format_type == SerializationFormat.PICKLE:
                with open(output_path, 'wb') as f:
                    pickle.dump(serialization_data, f)
            elif format_type == SerializationFormat.JOBLIB_COMPRESSED:
                with gzip.open(output_path, 'wb', compresslevel=self.compression_level) as f:
                    joblib.dump(serialization_data, f)
            elif format_type == SerializationFormat.PICKLE_COMPRESSED:
                with gzip.open(output_path, 'wb', compresslevel=self.compression_level) as f:
                    pickle.dump(serialization_data, f)
            else:
                raise ValueError(f"Unsupported serialization format: {format_type}")
            
            # Calculate checksum
            checksum = self._calculate_checksum(output_path)
            file_size = output_path.stat().st_size
            
            logger.info(f"Serialized model {model_package.model_id} using {format_type.value}")
            
            return {
                'success': True,
                'output_path': str(output_path),
                'format': format_type.value,
                'checksum': checksum,
                'size_bytes': file_size,
                'compression_level': self.compression_level if 'compressed' in format_type.value else None
            }
            
        except Exception as e:
            logger.error(f"Failed to serialize model: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def deserialize_model(
        self, 
        model_path: Path,
        verify_checksum: Optional[str] = None
    ) -> Tuple[ModelPackage, Dict[str, Any]]:
        """
        Deserialize a model package.
        
        Args:
            model_path: Path to the serialized model
            verify_checksum: Optional checksum to verify file integrity
            
        Returns:
            Tuple of (ModelPackage, deserialization_info)
        """
        try:
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Verify checksum if provided
            if verify_checksum:
                current_checksum = self._calculate_checksum(model_path)
                if current_checksum != verify_checksum:
                    raise ValueError(f"Checksum mismatch. Expected: {verify_checksum}, Got: {current_checksum}")
            
            # Determine format based on file extension and content
            format_type = self._detect_format(model_path)
            
            # Deserialize based on format
            if format_type == SerializationFormat.JOBLIB:
                data = joblib.load(model_path)
            elif format_type == SerializationFormat.PICKLE:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
            elif format_type == SerializationFormat.JOBLIB_COMPRESSED:
                with gzip.open(model_path, 'rb') as f:
                    data = joblib.load(f)
            elif format_type == SerializationFormat.PICKLE_COMPRESSED:
                with gzip.open(model_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                raise ValueError(f"Cannot determine serialization format for: {model_path}")
            
            # Extract model package
            package_metadata = data.get('package_metadata', {})
            
            model_package = ModelPackage(
                model_object=data['model_object'],
                preprocessors=data.get('preprocessors', {}),
                feature_metadata=data.get('feature_metadata', {}),
                training_config=data.get('training_config', {}),
                performance_metrics=data.get('performance_metrics', {}),
                model_id=package_metadata.get('model_id', 'unknown'),
                model_name=package_metadata.get('model_name', 'Unknown Model'),
                task_type=package_metadata.get('task_type', 'unknown'),
                created_at=package_metadata.get('created_at', datetime.now().isoformat()),
                version=package_metadata.get('version', '1.0.0')
            )
            
            deserialization_info = {
                'success': True,
                'format': format_type.value,
                'file_size': model_path.stat().st_size,
                'deserialization_timestamp': datetime.now().isoformat(),
                'original_format': package_metadata.get('serialization_format'),
                'original_timestamp': package_metadata.get('serialization_timestamp')
            }
            
            logger.info(f"Deserialized model {model_package.model_id} from {format_type.value}")
            
            return model_package, deserialization_info
            
        except Exception as e:
            logger.error(f"Failed to deserialize model: {e}")
            raise
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _detect_format(self, file_path: Path) -> SerializationFormat:
        """Detect serialization format based on file extension and content."""
        if file_path.suffix == '.gz':
            # Compressed file - determine based on second-to-last extension
            stem = file_path.stem  # Get filename without .gz
            if '.joblib' in str(file_path):
                return SerializationFormat.JOBLIB_COMPRESSED
            else:
                return SerializationFormat.PICKLE_COMPRESSED
        elif file_path.suffix == '.joblib':
            return SerializationFormat.JOBLIB
        elif file_path.suffix in ['.pkl', '.pickle']:
            return SerializationFormat.PICKLE
        else:
            # Default to joblib for unknown extensions
            return SerializationFormat.JOBLIB


class ModelBenchmarker:
    """Performance benchmarking for model operations."""
    
    def __init__(self):
        """Initialize ModelBenchmarker."""
        self.benchmark_history = []
        
    def benchmark_operation(
        self,
        operation_name: str,
        operation_func: Callable,
        model_id: str = "unknown",
        version: str = "unknown",
        **kwargs
    ) -> Tuple[Any, BenchmarkResult]:
        """
        Benchmark a model operation.
        
        Args:
            operation_name: Name of the operation being benchmarked
            operation_func: Function to benchmark
            model_id: Model identifier
            version: Model version
            **kwargs: Arguments to pass to operation_func
            
        Returns:
            Tuple of (operation_result, benchmark_result)
        """
        # Get initial system metrics
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            start_cpu_times = process.cpu_times()
        else:
            process = None
            initial_memory = 0
            start_cpu_times = None
        
        # Start timing
        start_time = time.time()
        
        try:
            # Execute operation
            result = operation_func(**kwargs)
            
            # End timing
            end_time = time.time()
            
            # Calculate metrics
            duration = end_time - start_time
            
            if PSUTIL_AVAILABLE and process:
                end_cpu_times = process.cpu_times()
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = final_memory - initial_memory
                
                # Calculate CPU usage approximation
                if start_cpu_times:
                    cpu_time_used = (end_cpu_times.user - start_cpu_times.user + 
                                   end_cpu_times.system - start_cpu_times.system)
                    cpu_percent = (cpu_time_used / duration * 100) if duration > 0 else 0
                else:
                    cpu_percent = 0
            else:
                final_memory = 0
                memory_used = 0
                cpu_percent = 0
            
            # Determine file size if applicable
            file_size = 0
            if hasattr(result, 'get') and result.get('size_bytes'):
                file_size = result['size_bytes']
            
            # Create benchmark result
            benchmark_result = BenchmarkResult(
                operation=operation_name,
                model_id=model_id,
                version=version,
                duration_seconds=duration,
                memory_used_mb=memory_used,
                file_size_bytes=file_size,
                cpu_percent=cpu_percent,
                timestamp=datetime.now().isoformat(),
                metadata={
                    'initial_memory_mb': initial_memory,
                    'final_memory_mb': final_memory,
                    'success': True
                }
            )
            
            self.benchmark_history.append(benchmark_result)
            
            logger.info(f"Benchmarked {operation_name}: {duration:.3f}s, {memory_used:.1f}MB")
            
            return result, benchmark_result
            
        except Exception as e:
            # Record failed operation
            end_time = time.time()
            duration = end_time - start_time
            
            benchmark_result = BenchmarkResult(
                operation=operation_name,
                model_id=model_id,
                version=version,
                duration_seconds=duration,
                memory_used_mb=0,
                file_size_bytes=0,
                cpu_percent=0,
                timestamp=datetime.now().isoformat(),
                metadata={
                    'success': False,
                    'error': str(e)
                }
            )
            
            self.benchmark_history.append(benchmark_result)
            
            logger.error(f"Benchmarked {operation_name} failed: {e}")
            raise
    
    def get_benchmark_summary(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Get benchmark summary statistics.
        
        Args:
            operation: Filter by operation name (all if None)
            
        Returns:
            Summary statistics
        """
        if not self.benchmark_history:
            return {}
        
        # Filter by operation if specified
        results = self.benchmark_history
        if operation:
            results = [r for r in results if r.operation == operation]
        
        if not results:
            return {}
        
        # Calculate statistics
        durations = [r.duration_seconds for r in results if r.metadata.get('success', True)]
        memory_usage = [r.memory_used_mb for r in results if r.metadata.get('success', True)]
        file_sizes = [r.file_size_bytes for r in results if r.file_size_bytes > 0]
        
        summary = {
            'total_operations': len(results),
            'successful_operations': len(durations),
            'failed_operations': len(results) - len(durations),
            'duration_stats': {
                'mean_seconds': np.mean(durations) if durations else 0,
                'median_seconds': np.median(durations) if durations else 0,
                'min_seconds': np.min(durations) if durations else 0,
                'max_seconds': np.max(durations) if durations else 0,
                'std_seconds': np.std(durations) if durations else 0
            },
            'memory_stats': {
                'mean_mb': np.mean(memory_usage) if memory_usage else 0,
                'median_mb': np.median(memory_usage) if memory_usage else 0,
                'min_mb': np.min(memory_usage) if memory_usage else 0,
                'max_mb': np.max(memory_usage) if memory_usage else 0,
                'std_mb': np.std(memory_usage) if memory_usage else 0
            },
            'file_size_stats': {
                'mean_bytes': np.mean(file_sizes) if file_sizes else 0,
                'median_bytes': np.median(file_sizes) if file_sizes else 0,
                'total_bytes': np.sum(file_sizes) if file_sizes else 0
            }
        }
        
        return summary


class AdvancedModelStorage:
    """
    Advanced model storage system with version control and performance tracking.
    """
    
    def __init__(self, storage_dir: str = "models_advanced"):
        """
        Initialize AdvancedModelStorage.
        
        Args:
            storage_dir: Directory for advanced model storage
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.models_dir = self.storage_dir / "models"
        self.versions_dir = self.storage_dir / "versions"
        self.metadata_dir = self.storage_dir / "metadata"
        self.benchmarks_dir = self.storage_dir / "benchmarks"
        
        for dir_path in [self.models_dir, self.versions_dir, self.metadata_dir, self.benchmarks_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.serializer = ModelSerializer()
        self.benchmarker = ModelBenchmarker()
        
        # Load storage metadata
        self.storage_metadata_file = self.storage_dir / "storage_metadata.json"
        self.storage_metadata = self._load_storage_metadata()
        
        # Load benchmark history
        self.benchmark_file = self.benchmarks_dir / "benchmark_history.json"
        self._load_benchmark_history()
        
        logger.info(f"Initialized AdvancedModelStorage at: {storage_dir}")
    
    def _load_storage_metadata(self) -> Dict[str, Any]:
        """Load storage metadata."""
        if self.storage_metadata_file.exists():
            try:
                with open(self.storage_metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load storage metadata: {e}")
                return self._create_default_storage_metadata()
        else:
            return self._create_default_storage_metadata()
    
    def _create_default_storage_metadata(self) -> Dict[str, Any]:
        """Create default storage metadata."""
        return {
            'storage_version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'models': {},
            'storage_stats': {
                'total_models': 0,
                'total_versions': 0,
                'total_size_bytes': 0
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def _save_storage_metadata(self):
        """Save storage metadata."""
        try:
            self.storage_metadata['last_updated'] = datetime.now().isoformat()
            with open(self.storage_metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.storage_metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save storage metadata: {e}")
    
    def _load_benchmark_history(self):
        """Load benchmark history from file."""
        if self.benchmark_file.exists():
            try:
                with open(self.benchmark_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                    
                # Convert to BenchmarkResult objects
                for item in history_data:
                    benchmark_result = BenchmarkResult(
                        operation=item['operation'],
                        model_id=item['model_id'],
                        version=item['version'],
                        duration_seconds=item['duration_seconds'],
                        memory_used_mb=item['memory_used_mb'],
                        file_size_bytes=item['file_size_bytes'],
                        cpu_percent=item['cpu_percent'],
                        timestamp=item['timestamp'],
                        metadata=item['metadata']
                    )
                    self.benchmarker.benchmark_history.append(benchmark_result)
                    
            except Exception as e:
                logger.warning(f"Failed to load benchmark history: {e}")
    
    def _save_benchmark_history(self):
        """Save benchmark history to file."""
        try:
            history_data = []
            for result in self.benchmarker.benchmark_history:
                history_data.append({
                    'operation': result.operation,
                    'model_id': result.model_id,
                    'version': result.version,
                    'duration_seconds': result.duration_seconds,
                    'memory_used_mb': result.memory_used_mb,
                    'file_size_bytes': result.file_size_bytes,
                    'cpu_percent': result.cpu_percent,
                    'timestamp': result.timestamp,
                    'metadata': result.metadata
                })
            
            with open(self.benchmark_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save benchmark history: {e}")
    
    def store_model(
        self,
        model_package: ModelPackage,
        format_type: SerializationFormat = SerializationFormat.JOBLIB,
        description: Optional[str] = None,
        parent_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store a model with version control.
        
        Args:
            model_package: Complete model package to store
            format_type: Serialization format
            description: Version description
            parent_version: Parent version for tracking changes
            
        Returns:
            Storage result information
        """
        try:
            model_id = model_package.model_id
            version = model_package.version
            
            # Create model-specific directories
            model_dir = self.models_dir / model_id
            version_dir = self.versions_dir / model_id / version
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine file extension based on format
            ext_map = {
                SerializationFormat.JOBLIB: '.joblib',
                SerializationFormat.PICKLE: '.pkl',
                SerializationFormat.JOBLIB_COMPRESSED: '.joblib.gz',
                SerializationFormat.PICKLE_COMPRESSED: '.pkl.gz'
            }
            extension = ext_map.get(format_type, '.joblib')
            
            # Store model
            model_path = version_dir / f"model{extension}"
            serialization_result = self.serializer.serialize_model(
                model_package, model_path, format_type
            )
            
            if not serialization_result['success']:
                return serialization_result
            
            # Create version metadata
            version_metadata = {
                'model_id': model_id,
                'version': version,
                'model_name': model_package.model_name,
                'task_type': model_package.task_type,
                'created_at': datetime.now().isoformat(),
                'description': description,
                'parent_version': parent_version,
                'model_path': str(model_path),
                'format': format_type.value,
                'checksum': serialization_result['checksum'],
                'size_bytes': serialization_result['size_bytes'],
                'feature_metadata': model_package.feature_metadata,
                'training_config': model_package.training_config,
                'performance_metrics': model_package.performance_metrics
            }
            
            # Save version metadata
            version_metadata_path = version_dir / "version_metadata.json"
            with open(version_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(version_metadata, f, indent=2, ensure_ascii=False)
            
            # Update storage metadata
            if model_id not in self.storage_metadata['models']:
                self.storage_metadata['models'][model_id] = {
                    'model_name': model_package.model_name,
                    'task_type': model_package.task_type,
                    'created_at': datetime.now().isoformat(),
                    'versions': {}
                }
                self.storage_metadata['storage_stats']['total_models'] += 1
            
            # Add version to model metadata
            self.storage_metadata['models'][model_id]['versions'][version] = {
                'created_at': datetime.now().isoformat(),
                'size_bytes': serialization_result['size_bytes'],
                'checksum': serialization_result['checksum'],
                'description': description,
                'parent_version': parent_version
            }
            
            # Update storage stats
            self.storage_metadata['storage_stats']['total_versions'] += 1
            self.storage_metadata['storage_stats']['total_size_bytes'] += serialization_result['size_bytes']
            
            # Save storage metadata
            self._save_storage_metadata()
            
            logger.info(f"Stored model {model_id} version {version}")
            
            return {
                'success': True,
                'model_id': model_id,
                'version': version,
                'model_path': str(model_path),
                'metadata_path': str(version_metadata_path),
                'checksum': serialization_result['checksum'],
                'size_bytes': serialization_result['size_bytes']
            }
            
        except Exception as e:
            logger.error(f"Failed to store model: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def load_model(
        self, 
        model_id: str, 
        version: Optional[str] = None,
        verify_integrity: bool = True
    ) -> Tuple[ModelPackage, Dict[str, Any]]:
        """
        Load a model from storage.
        
        Args:
            model_id: Model identifier
            version: Specific version to load (latest if None)
            verify_integrity: Whether to verify file integrity
            
        Returns:
            Tuple of (ModelPackage, load_info)
        """
        try:
            if model_id not in self.storage_metadata['models']:
                raise ValueError(f"Model {model_id} not found")
            
            model_metadata = self.storage_metadata['models'][model_id]
            
            # Determine version to load
            if version is None:
                # Load latest version
                versions = list(model_metadata['versions'].keys())
                if not versions:
                    raise ValueError(f"No versions found for model {model_id}")
                version = max(versions)  # Assuming semantic versioning
            
            if version not in model_metadata['versions']:
                available_versions = list(model_metadata['versions'].keys())
                raise ValueError(f"Version {version} not found for model {model_id}. Available: {available_versions}")
            
            # Load version metadata
            version_dir = self.versions_dir / model_id / version
            version_metadata_path = version_dir / "version_metadata.json"
            
            if not version_metadata_path.exists():
                raise FileNotFoundError(f"Version metadata not found: {version_metadata_path}")
            
            with open(version_metadata_path, 'r', encoding='utf-8') as f:
                version_metadata = json.load(f)
            
            # Load model
            model_path = Path(version_metadata['model_path'])
            checksum = version_metadata['checksum'] if verify_integrity else None
            
            model_package, deserialization_info = self.serializer.deserialize_model(
                model_path, checksum
            )
            
            load_info = {
                'model_id': model_id,
                'version': version,
                'loaded_at': datetime.now().isoformat(),
                'model_path': str(model_path),
                'version_metadata': version_metadata,
                'deserialization_info': deserialization_info
            }
            
            logger.info(f"Loaded model {model_id} version {version}")
            
            return model_package, load_info
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all models in storage."""
        try:
            models_list = []
            
            for model_id, model_metadata in self.storage_metadata['models'].items():
                versions = model_metadata.get('versions', {})
                latest_version = max(versions.keys()) if versions else None
                
                model_info = {
                    'model_id': model_id,
                    'model_name': model_metadata.get('model_name', 'Unknown'),
                    'task_type': model_metadata.get('task_type', 'unknown'),
                    'created_at': model_metadata.get('created_at'),
                    'total_versions': len(versions),
                    'latest_version': latest_version,
                    'total_size_bytes': sum(v.get('size_bytes', 0) for v in versions.values()),
                    'versions': list(versions.keys())
                }
                
                models_list.append(model_info)
            
            return models_list
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def get_model_versions(self, model_id: str) -> List[Dict[str, Any]]:
        """Get all versions of a specific model."""
        try:
            if model_id not in self.storage_metadata['models']:
                raise ValueError(f"Model {model_id} not found")
            
            model_metadata = self.storage_metadata['models'][model_id]
            versions_info = []
            
            for version, version_data in model_metadata['versions'].items():
                version_info = {
                    'version': version,
                    'created_at': version_data.get('created_at'),
                    'size_bytes': version_data.get('size_bytes', 0),
                    'checksum': version_data.get('checksum'),
                    'description': version_data.get('description'),
                    'parent_version': version_data.get('parent_version')
                }
                versions_info.append(version_info)
            
            # Sort by creation date
            versions_info.sort(key=lambda x: x['created_at'], reverse=True)
            
            return versions_info
            
        except Exception as e:
            logger.error(f"Failed to get model versions: {e}")
            raise
    
    def delete_model_version(
        self, 
        model_id: str, 
        version: str,
        keep_metadata: bool = False
    ) -> Dict[str, Any]:
        """
        Delete a specific model version.
        
        Args:
            model_id: Model identifier
            version: Version to delete
            keep_metadata: Whether to keep metadata (soft delete)
            
        Returns:
            Deletion result
        """
        try:
            if model_id not in self.storage_metadata['models']:
                return {'success': False, 'error': f"Model {model_id} not found"}
            
            model_metadata = self.storage_metadata['models'][model_id]
            
            if version not in model_metadata['versions']:
                return {'success': False, 'error': f"Version {version} not found"}
            
            version_data = model_metadata['versions'][version]
            version_dir = self.versions_dir / model_id / version
            
            # Delete files
            if version_dir.exists() and not keep_metadata:
                shutil.rmtree(version_dir)
            
            # Update metadata
            size_bytes = version_data.get('size_bytes', 0)
            del model_metadata['versions'][version]
            
            # Update storage stats
            self.storage_metadata['storage_stats']['total_versions'] -= 1
            self.storage_metadata['storage_stats']['total_size_bytes'] -= size_bytes
            
            # If no versions left, remove model
            if not model_metadata['versions']:
                del self.storage_metadata['models'][model_id]
                self.storage_metadata['storage_stats']['total_models'] -= 1
            
            self._save_storage_metadata()
            
            logger.info(f"Deleted model {model_id} version {version}")
            
            return {
                'success': True,
                'model_id': model_id,
                'version': version,
                'size_freed_bytes': size_bytes
            }
            
        except Exception as e:
            logger.error(f"Failed to delete model version: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            stats = self.storage_metadata['storage_stats'].copy()
            
            # Calculate directory sizes
            total_actual_size = 0
            for model_dir in self.models_dir.iterdir():
                if model_dir.is_dir():
                    for file_path in model_dir.rglob('*'):
                        if file_path.is_file():
                            total_actual_size += file_path.stat().st_size
            
            stats['actual_storage_size_bytes'] = total_actual_size
            stats['storage_efficiency'] = stats['total_size_bytes'] / max(total_actual_size, 1)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}
    
    def store_model_with_benchmark(
        self,
        model_package: ModelPackage,
        format_type: SerializationFormat = SerializationFormat.JOBLIB,
        description: Optional[str] = None,
        parent_version: Optional[str] = None
    ) -> Tuple[Dict[str, Any], BenchmarkResult]:
        """
        Store a model with performance benchmarking.
        
        Args:
            model_package: Complete model package to store
            format_type: Serialization format
            description: Version description
            parent_version: Parent version for tracking changes
            
        Returns:
            Tuple of (storage_result, benchmark_result)
        """
        def store_operation():
            return self.store_model(model_package, format_type, description, parent_version)
        
        result, benchmark = self.benchmarker.benchmark_operation(
            operation_name="store_model",
            operation_func=store_operation,
            model_id=model_package.model_id,
            version=model_package.version
        )
        
        # Save benchmark history
        self._save_benchmark_history()
        
        return result, benchmark
    
    def load_model_with_benchmark(
        self, 
        model_id: str, 
        version: Optional[str] = None,
        verify_integrity: bool = True
    ) -> Tuple[Tuple[ModelPackage, Dict[str, Any]], BenchmarkResult]:
        """
        Load a model with performance benchmarking.
        
        Args:
            model_id: Model identifier
            version: Specific version to load (latest if None)
            verify_integrity: Whether to verify file integrity
            
        Returns:
            Tuple of ((ModelPackage, load_info), benchmark_result)
        """
        def load_operation():
            return self.load_model(model_id, version, verify_integrity)
        
        result, benchmark = self.benchmarker.benchmark_operation(
            operation_name="load_model",
            operation_func=load_operation,
            model_id=model_id,
            version=version or "latest"
        )
        
        # Save benchmark history
        self._save_benchmark_history()
        
        return result, benchmark
    
    def get_performance_report(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Args:
            model_id: Filter by model ID (all models if None)
            
        Returns:
            Performance report
        """
        # Filter benchmark history by model if specified
        if model_id:
            filtered_history = [r for r in self.benchmarker.benchmark_history 
                              if r.model_id == model_id]
        else:
            filtered_history = self.benchmarker.benchmark_history
        
        if not filtered_history:
            return {'error': 'No performance data available'}
        
        # Generate report by operation type
        operations = set(r.operation for r in filtered_history)
        report = {
            'report_generated_at': datetime.now().isoformat(),
            'total_operations': len(filtered_history),
            'model_filter': model_id,
            'operations_summary': {}
        }
        
        for operation in operations:
            op_results = [r for r in filtered_history if r.operation == operation]
            summary = self.benchmarker.get_benchmark_summary(operation)
            
            # Add operation-specific insights
            if operation == "store_model":
                summary['throughput_mb_per_second'] = []
                for result in op_results:
                    if result.file_size_bytes > 0 and result.duration_seconds > 0:
                        throughput = (result.file_size_bytes / 1024 / 1024) / result.duration_seconds
                        summary['throughput_mb_per_second'].append(throughput)
                
                if summary['throughput_mb_per_second']:
                    summary['mean_throughput_mb_per_second'] = np.mean(summary['throughput_mb_per_second'])
            
            report['operations_summary'][operation] = summary
        
        # Add overall insights
        all_durations = [r.duration_seconds for r in filtered_history if r.metadata.get('success', True)]
        if all_durations:
            report['overall_performance'] = {
                'average_operation_time': np.mean(all_durations),
                'fastest_operation': np.min(all_durations),
                'slowest_operation': np.max(all_durations),
                'total_time_spent': np.sum(all_durations)
            }
        
        return report
    
    def benchmark_model_comparison(
        self, 
        model_packages: List[ModelPackage],
        format_types: List[SerializationFormat] = None
    ) -> Dict[str, Any]:
        """
        Benchmark multiple models for comparison.
        
        Args:
            model_packages: List of model packages to benchmark
            format_types: Serialization formats to test (all if None)
            
        Returns:
            Comparison benchmark results
        """
        if format_types is None:
            format_types = list(SerializationFormat)
        
        comparison_results = {
            'benchmark_timestamp': datetime.now().isoformat(),
            'models_tested': len(model_packages),
            'formats_tested': [f.value for f in format_types],
            'results': []
        }
        
        for model_package in model_packages:
            model_results = {
                'model_id': model_package.model_id,
                'model_name': model_package.model_name,
                'task_type': model_package.task_type,
                'format_results': {}
            }
            
            for format_type in format_types:
                try:
                    # Benchmark storage
                    store_result, store_benchmark = self.store_model_with_benchmark(
                        model_package, format_type, f"Benchmark test - {format_type.value}"
                    )
                    
                    # Benchmark loading
                    load_result, load_benchmark = self.load_model_with_benchmark(
                        model_package.model_id, model_package.version
                    )
                    
                    model_results['format_results'][format_type.value] = {
                        'store_duration': store_benchmark.duration_seconds,
                        'store_memory': store_benchmark.memory_used_mb,
                        'store_size': store_benchmark.file_size_bytes,
                        'load_duration': load_benchmark.duration_seconds,
                        'load_memory': load_benchmark.memory_used_mb,
                        'total_duration': store_benchmark.duration_seconds + load_benchmark.duration_seconds
                    }
                    
                except Exception as e:
                    model_results['format_results'][format_type.value] = {
                        'error': str(e)
                    }
            
            comparison_results['results'].append(model_results)
        
        return comparison_results


# Convenience functions
def create_model_package(
    model_object: Any,
    model_id: str,
    model_name: str,
    task_type: str,
    feature_names: List[str],
    preprocessors: Optional[Dict[str, Any]] = None,
    training_config: Optional[Dict[str, Any]] = None,
    performance_metrics: Optional[Dict[str, Any]] = None,
    version: str = "1.0.0"
) -> ModelPackage:
    """
    Create a ModelPackage for storage.
    
    Args:
        model_object: Trained model object
        model_id: Unique model identifier
        model_name: Human-readable model name
        task_type: Task type (classification/regression)
        feature_names: List of feature names
        preprocessors: Dictionary of preprocessor objects
        training_config: Training configuration
        performance_metrics: Model performance metrics
        version: Model version
        
    Returns:
        ModelPackage object ready for storage
    """
    return ModelPackage(
        model_object=model_object,
        preprocessors=preprocessors or {},
        feature_metadata={'feature_names': feature_names},
        training_config=training_config or {},
        performance_metrics=performance_metrics or {},
        model_id=model_id,
        model_name=model_name,
        task_type=task_type,
        created_at=datetime.now().isoformat(),
        version=version
    ) 