"""
Common test fixtures for Photo Filter AI tests.

This module provides reusable fixtures for testing various components
of the photo organization system.
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


@pytest.fixture
def temp_test_dir():
    """Create a temporary directory for test isolation."""
    test_dir = Path(tempfile.mkdtemp(prefix="photo_filter_test_"))
    yield test_dir

    # Cleanup
    if test_dir.exists():
        shutil.rmtree(test_dir)


@pytest.fixture
def empty_cache_file(temp_test_dir):
    """Create an empty JSON cache file for testing."""
    cache_file = temp_test_dir / "test_cache.json"
    with open(cache_file, 'w') as f:
        json.dump({}, f)
    return cache_file


@pytest.fixture
def sample_cache_file(temp_test_dir):
    """Create a cache file with sample data for testing."""
    cache_file = temp_test_dir / "sample_cache.json"
    sample_data = {
        "morning|Short Event|Monday|weekday|Edmonton|general|unknown": "2024_01_15 - Morning Coffee",
        "afternoon|Medium Event|Saturday|weekend|Calgary|outdoor|unknown": "2024_01_20 - Afternoon Hike"
    }
    with open(cache_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    return cache_file


@pytest.fixture
def mock_cluster_data():
    """Create mock cluster data for event naming tests."""
    return {
        'files': [
            {'path': '/test/IMG_001.jpg', 'filename': 'IMG_001.jpg'},
            {'path': '/test/IMG_002.jpg', 'filename': 'IMG_002.jpg'}
        ],
        'start_time': datetime(2024, 1, 15, 9, 30),
        'end_time': datetime(2024, 1, 15, 10, 30),
        'location': {
            'city': 'Edmonton',
            'country': 'Canada',
            'coordinates': (53.5461, -113.4938)
        },
        'metadata': {
            'total_photos': 2,
            'duration_hours': 1.0,
            'main_location': 'Edmonton'
        }
    }


@pytest.fixture
def mock_media_file():
    """Create a mock MediaFile object for testing."""
    class MockMediaFile:
        def __init__(self, path: str, filename: str = None):
            self.path = Path(path)
            self.filename = filename or self.path.name
            self.time = datetime(2024, 1, 15, 9, 30)
            self.file_type = 'photo'
            self.event_folder = 'Test Event'

    return MockMediaFile


@pytest.fixture
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment with proper imports."""
    import sys
    from pathlib import Path

    # Add src to Python path
    project_root = Path(__file__).parent.parent.parent
    src_path = str(project_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    yield

    # Cleanup if needed
    pass


@pytest.fixture
def isolated_config(temp_test_dir):
    """Create isolated configuration for testing."""
    config_file = temp_test_dir / "test_config.json"
    test_config = {
        "processing": {
            "max_photos": 10,
            "time_threshold_hours": 2.0
        },
        "paths": {
            "cache_dir": str(temp_test_dir / "cache"),
            "vector_db_dir": str(temp_test_dir / "vector_db")
        }
    }

    with open(config_file, 'w') as f:
        json.dump(test_config, f, indent=2)

    return config_file