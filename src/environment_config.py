"""
Environment-based configuration for Photo Filter AI.

This module provides configuration management for different environments:
- production: Real data, full functionality
- development: Local development with real data
- test: Isolated test environment with temporary data
"""

import os
from pathlib import Path
from typing import Optional
from enum import Enum

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    # python-dotenv not installed - environment variables must be set manually
    pass

class Environment(Enum):
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    TEST = "test"

class EnvironmentConfig:
    """Environment-based configuration manager."""

    def __init__(self):
        self._env = self._detect_environment()
        self._project_root = Path(__file__).parent.parent

    def _detect_environment(self) -> Environment:
        """Detect current environment from environment variables."""
        env_var = os.getenv('PHOTO_FILTER_ENV', '').lower()

        if env_var == 'test' or os.getenv('PHOTO_FILTER_TEST_MODE') == 'true':
            return Environment.TEST
        elif env_var == 'production':
            return Environment.PRODUCTION
        else:
            return Environment.DEVELOPMENT

    @property
    def environment(self) -> Environment:
        """Get current environment."""
        return self._env

    @property
    def is_test(self) -> bool:
        """Check if running in test environment."""
        return self._env == Environment.TEST

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self._env == Environment.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self._env == Environment.PRODUCTION

    @property
    def data_dir(self) -> Path:
        """Get data directory path for current environment."""
        if self.is_test:
            # Use temporary directory for tests
            test_dir = os.getenv('PHOTO_FILTER_DATA_DIR')
            if test_dir:
                return Path(test_dir)
            else:
                # Fallback to temp directory
                import tempfile
                return Path(tempfile.gettempdir()) / "photo_filter_test"
        elif self.is_production:
            # Production data directory
            return self._project_root / "data"
        else:
            # Development uses same as production but can be overridden
            dev_data_dir = os.getenv('PHOTO_FILTER_DEV_DATA_DIR')
            if dev_data_dir:
                return Path(dev_data_dir)
            return self._project_root / "data"

    @property
    def cache_dir(self) -> Path:
        """Get cache directory path for current environment."""
        if self.is_test:
            cache_dir = os.getenv('PHOTO_FILTER_CACHE_DIR')
            if cache_dir:
                return Path(cache_dir)
            return self.data_dir / "cache"
        else:
            return self.data_dir / "cache"

    @property
    def vector_db_dir(self) -> Path:
        """Get vector database directory path for current environment."""
        if self.is_test:
            vector_dir = os.getenv('PHOTO_FILTER_VECTOR_DB_DIR')
            if vector_dir:
                return Path(vector_dir)
            return self.data_dir / "vector_db"
        else:
            return self._project_root / "vector_db"

    @property
    def event_naming_cache_file(self) -> Path:
        """Get event naming cache file path for current environment."""
        # Check for explicit override first
        cache_file = os.getenv('PHOTO_FILTER_EVENT_CACHE_FILE')
        if cache_file:
            return Path(cache_file)

        return self.data_dir / "event_naming_cache.json"

    @property
    def log_level(self) -> str:
        """Get appropriate log level for environment."""
        if self.is_test:
            return os.getenv('PHOTO_FILTER_LOG_LEVEL', 'WARNING')
        elif self.is_development:
            return os.getenv('PHOTO_FILTER_LOG_LEVEL', 'DEBUG')
        else:
            return os.getenv('PHOTO_FILTER_LOG_LEVEL', 'INFO')

    def ensure_directories(self):
        """Ensure all required directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)

    def __str__(self):
        return f"EnvironmentConfig(env={self.environment.value}, data_dir={self.data_dir})"

# Global configuration instance
config = EnvironmentConfig()

# Convenience functions for common usage
def get_data_dir() -> Path:
    """Get data directory for current environment."""
    return config.data_dir

def get_cache_dir() -> Path:
    """Get cache directory for current environment."""
    return config.cache_dir

def get_event_naming_cache_file() -> Path:
    """Get event naming cache file for current environment."""
    return config.event_naming_cache_file

def is_test_environment() -> bool:
    """Check if running in test environment."""
    return config.is_test