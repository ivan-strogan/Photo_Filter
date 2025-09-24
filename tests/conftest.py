"""
Pytest configuration and fixtures.

This file makes fixtures automatically available to all tests in the tests/ directory.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Import all common fixtures to make them available
from tests.fixtures.common_fixtures import *

# You can add additional test configuration here
pytest_plugins = []

def pytest_configure(config):
    """Global pytest configuration - runs once at test session start."""
    # Set environment to test mode
    os.environ['PHOTO_FILTER_ENV'] = 'test'
    os.environ['PHOTO_FILTER_TEST_MODE'] = 'true'

    # Create temporary directory for all test data
    global TEST_DATA_DIR
    TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="photo_filter_test_data_"))

    # Set test-specific environment variables
    os.environ['PHOTO_FILTER_DATA_DIR'] = str(TEST_DATA_DIR)
    os.environ['PHOTO_FILTER_CACHE_DIR'] = str(TEST_DATA_DIR / 'cache')
    os.environ['PHOTO_FILTER_VECTOR_DB_DIR'] = str(TEST_DATA_DIR / 'vector_db')

    # Create test directories
    (TEST_DATA_DIR / 'cache').mkdir(parents=True, exist_ok=True)
    (TEST_DATA_DIR / 'vector_db').mkdir(parents=True, exist_ok=True)

    # Create isolated test cache file
    test_cache_file = TEST_DATA_DIR / 'event_naming_cache.json'
    with open(test_cache_file, 'w') as f:
        import json
        json.dump({}, f)

    print(f"🧪 TEST SESSION: Environment = test")
    print(f"🧪 TEST SESSION: Using isolated test data directory: {TEST_DATA_DIR}")

def pytest_unconfigure(config):
    """Cleanup after test session."""
    global TEST_DATA_DIR
    if 'TEST_DATA_DIR' in globals() and TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR)
        print(f"🧹 TEST SESSION: Cleaned up test data directory: {TEST_DATA_DIR}")

@pytest.fixture(autouse=True)
def isolate_event_namer_cache(monkeypatch):
    """Automatically isolate EventNamer cache for all tests."""
    # This fixture runs for every test automatically
    test_cache_file = Path(tempfile.mkdtemp(prefix="test_event_cache_")) / "event_naming_cache.json"

    # Create empty cache file
    with open(test_cache_file, 'w') as f:
        import json
        json.dump({}, f)

    # Monkey patch the EventNamer to use test cache
    def mock_get_cache_file():
        return str(test_cache_file)

    # This will override any EventNamer cache file usage
    monkeypatch.setenv("PHOTO_FILTER_EVENT_CACHE_FILE", str(test_cache_file))

    yield test_cache_file

    # Cleanup
    if test_cache_file.parent.exists():
        shutil.rmtree(test_cache_file.parent)