"""Configuration management system for Photo Filter application."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import os

@dataclass
class ClusteringConfig:
    """Configuration for clustering algorithms."""
    time_threshold_hours: float = 6.0
    location_threshold_km: float = 1.0
    min_cluster_size: int = 1
    max_gap_hours: float = 2.0
    merge_threshold_hours: float = 3.0
    similarity_threshold: float = 0.7

@dataclass
class ProcessingConfig:
    """Configuration for media processing."""
    max_photos_per_event: int = 50
    batch_size: int = 16
    use_gpu: bool = True
    enable_vectorization: bool = True
    enable_geocoding: bool = True
    geocoding_cache_size: int = 10000

@dataclass
class PathConfig:
    """Configuration for file paths."""
    sample_photos_dir: str = "Sample_Photos"
    iphone_automatic_dir: str = "Sample_Photos/iPhone Automatic"
    pictures_dir: str = "Sample_Photos/Pictures"
    vector_db_dir: str = "vector_db"
    log_file: str = "logs/photo_filter.log"

@dataclass
class NamingConfig:
    """Configuration for event naming."""
    date_format: str = "%Y_%m_%d"
    include_location: bool = True
    include_duration_hints: bool = True
    max_name_length: int = 100
    use_llm_naming: bool = False
    llm_model: str = "gpt-3.5-turbo"

@dataclass
class FaceRecognitionConfig:
    """Configuration for face recognition features."""
    enable_face_detection: bool = False  # Disabled by default for privacy
    detection_model: str = "hog"  # "hog" (fast) or "cnn" (accurate)
    recognition_tolerance: float = 0.6  # Lower = stricter matching
    min_face_size: int = 50  # Minimum face size in pixels
    include_in_naming: bool = True  # Use people data in event names
    store_encodings: bool = True  # Cache face encodings for performance
    auto_cluster_faces: bool = True  # Automatically cluster unknown faces
    cluster_tolerance: float = 0.5  # Face clustering distance threshold

@dataclass
class PhotoFilterConfig:
    """Complete application configuration."""
    clustering: ClusteringConfig
    processing: ProcessingConfig
    paths: PathConfig
    naming: NamingConfig
    faces: FaceRecognitionConfig
    log_level: str = "INFO"
    version: str = "1.0.0"

class ConfigManager:
    """Manages application configuration with file persistence."""

    def __init__(self, config_file: Optional[Path] = None):
        """Initialize configuration manager.

        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file or Path("photo_filter_config.json")
        self.logger = logging.getLogger(__name__)
        self._config = None

    def load_config(self, create_default: bool = True) -> PhotoFilterConfig:
        """Load configuration from file or create default.

        Args:
            create_default: Create default config if file doesn't exist

        Returns:
            PhotoFilterConfig instance
        """
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)

                # Convert dict to config objects
                self._config = self._dict_to_config(config_data)
                self.logger.info(f"Configuration loaded from {self.config_file}")

            except Exception as e:
                self.logger.warning(f"Error loading config file: {e}")
                if create_default:
                    self._config = self._create_default_config()
                    self.logger.info("Using default configuration")
                else:
                    raise

        else:
            if create_default:
                self._config = self._create_default_config()
                self.save_config()  # Save default config
                self.logger.info(f"Created default configuration at {self.config_file}")
            else:
                raise FileNotFoundError(f"Configuration file not found: {self.config_file}")

        return self._config

    def save_config(self, config: Optional[PhotoFilterConfig] = None) -> None:
        """Save configuration to file.

        Args:
            config: Configuration to save (uses current if None)
        """
        if config is None:
            config = self._config

        if config is None:
            raise ValueError("No configuration to save")

        try:
            config_dict = self._config_to_dict(config)

            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)

            self.logger.info(f"Configuration saved to {self.config_file}")

        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise

    def get_config(self) -> PhotoFilterConfig:
        """Get current configuration.

        Returns:
            Current PhotoFilterConfig instance
        """
        if self._config is None:
            self._config = self.load_config()
        return self._config

    def update_clustering_config(self, **kwargs) -> None:
        """Update clustering configuration parameters.

        Args:
            **kwargs: Clustering parameters to update
        """
        config = self.get_config()

        for key, value in kwargs.items():
            if hasattr(config.clustering, key):
                setattr(config.clustering, key, value)
                self.logger.info(f"Updated clustering.{key} = {value}")
            else:
                self.logger.warning(f"Unknown clustering parameter: {key}")

        self.save_config(config)

    def update_processing_config(self, **kwargs) -> None:
        """Update processing configuration parameters.

        Args:
            **kwargs: Processing parameters to update
        """
        config = self.get_config()

        for key, value in kwargs.items():
            if hasattr(config.processing, key):
                setattr(config.processing, key, value)
                self.logger.info(f"Updated processing.{key} = {value}")
            else:
                self.logger.warning(f"Unknown processing parameter: {key}")

        self.save_config(config)

    def update_paths_config(self, **kwargs) -> None:
        """Update path configuration parameters.

        Args:
            **kwargs: Path parameters to update
        """
        config = self.get_config()

        for key, value in kwargs.items():
            if hasattr(config.paths, key):
                setattr(config.paths, key, value)
                self.logger.info(f"Updated paths.{key} = {value}")
            else:
                self.logger.warning(f"Unknown path parameter: {key}")

        self.save_config(config)

    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self._config = self._create_default_config()
        self.save_config()
        self.logger.info("Configuration reset to defaults")

    def validate_config(self, config: Optional[PhotoFilterConfig] = None) -> Dict[str, Any]:
        """Validate configuration parameters.

        Args:
            config: Configuration to validate (uses current if None)

        Returns:
            Dictionary with validation results
        """
        if config is None:
            config = self.get_config()

        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }

        # Validate clustering parameters
        clustering = config.clustering
        if clustering.time_threshold_hours <= 0:
            validation_results['errors'].append("time_threshold_hours must be positive")

        if clustering.location_threshold_km <= 0:
            validation_results['errors'].append("location_threshold_km must be positive")

        if clustering.min_cluster_size < 1:
            validation_results['errors'].append("min_cluster_size must be at least 1")

        if clustering.max_gap_hours > clustering.time_threshold_hours:
            validation_results['warnings'].append("max_gap_hours larger than time_threshold_hours")

        # Validate processing parameters
        processing = config.processing
        if processing.max_photos_per_event < 1:
            validation_results['errors'].append("max_photos_per_event must be at least 1")

        if processing.batch_size < 1:
            validation_results['errors'].append("batch_size must be at least 1")

        # Validate paths
        paths = config.paths
        base_dir = Path(paths.sample_photos_dir)
        if not base_dir.exists():
            validation_results['warnings'].append(f"Sample photos directory not found: {base_dir}")

        # Check if any errors found
        validation_results['valid'] = len(validation_results['errors']) == 0

        return validation_results

    def _create_default_config(self) -> PhotoFilterConfig:
        """Create default configuration."""
        return PhotoFilterConfig(
            clustering=ClusteringConfig(),
            processing=ProcessingConfig(),
            paths=PathConfig(),
            naming=NamingConfig(),
            faces=FaceRecognitionConfig()
        )

    def _config_to_dict(self, config: PhotoFilterConfig) -> Dict[str, Any]:
        """Convert config object to dictionary."""
        return {
            'clustering': asdict(config.clustering),
            'processing': asdict(config.processing),
            'paths': asdict(config.paths),
            'naming': asdict(config.naming),
            'faces': asdict(config.faces),
            'log_level': config.log_level,
            'version': config.version
        }

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> PhotoFilterConfig:
        """Convert dictionary to config object."""
        return PhotoFilterConfig(
            clustering=ClusteringConfig(**config_dict.get('clustering', {})),
            processing=ProcessingConfig(**config_dict.get('processing', {})),
            paths=PathConfig(**config_dict.get('paths', {})),
            naming=NamingConfig(**config_dict.get('naming', {})),
            faces=FaceRecognitionConfig(**config_dict.get('faces', {})),
            log_level=config_dict.get('log_level', 'INFO'),
            version=config_dict.get('version', '1.0.0')
        )

    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration.

        Returns:
            Dictionary with configuration summary
        """
        config = self.get_config()

        return {
            'config_file': str(self.config_file),
            'config_exists': self.config_file.exists(),
            'clustering': {
                'time_threshold_hours': config.clustering.time_threshold_hours,
                'location_threshold_km': config.clustering.location_threshold_km,
                'min_cluster_size': config.clustering.min_cluster_size
            },
            'processing': {
                'max_photos_per_event': config.processing.max_photos_per_event,
                'use_gpu': config.processing.use_gpu,
                'enable_vectorization': config.processing.enable_vectorization
            },
            'paths': {
                'sample_photos_dir': config.paths.sample_photos_dir,
                'vector_db_dir': config.paths.vector_db_dir
            },
            'validation': self.validate_config(config)
        }

# Global configuration manager instance
config_manager = None

def get_config_manager(config_file: Optional[Path] = None) -> ConfigManager:
    """Get global configuration manager instance.

    Args:
        config_file: Path to configuration file

    Returns:
        ConfigManager instance
    """
    global config_manager
    if config_manager is None:
        config_manager = ConfigManager(config_file)
    return config_manager

def get_config() -> PhotoFilterConfig:
    """Get current application configuration.

    Returns:
        PhotoFilterConfig instance
    """
    return get_config_manager().get_config()