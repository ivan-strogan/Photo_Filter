# Photo Filter AI App - Test Suite

This directory contains all test files for the Photo Filter AI application.

## Test Files

### Core Component Tests

- `test_basic.py` - Basic functionality tests
- `test_components.py` - Component integration tests
- `test_system.py` - End-to-end system tests

### Feature-Specific Tests

- `test_content_analyzer.py` - Computer vision content analysis tests
- `test_location_verification.py` - GPS metadata and geocoding tests
- `test_organized_scanner.py` - Existing photo library scanning tests
- `test_vectorization.py` - Photo vectorization and similarity tests

### Processing Tests

- `test_small_processing.py` - Small batch processing tests

## Running Tests

To run tests, activate the virtual environment and execute from the project root:

```bash
# Activate virtual environment
source venv_py311/bin/activate

# Run all tests (RSpec-style - recommended)
pytest tests/

# Run all tests with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_content_analyzer.py

# Run tests matching a pattern
pytest -k "event_naming"

# Run tests by category
pytest -m "unit"                # Run only unit tests
pytest -m "not slow"            # Skip slow tests
pytest -m "integration"         # Run only integration tests

# Run individual tests (legacy method - still works)
python tests/test_content_analyzer.py
python tests/test_location_verification.py
```

## Test Requirements

Some tests require additional dependencies:

- `torch` and `transformers` for advanced content analysis
- Internet connection for geocoding tests
- Sample photos in `Sample_Photos/iPhone Automatic/` directory

## Test Structure

### Directory Organization

- **Test files**: `test_*.py` files contain the actual test implementations
- **Fixtures**: `tests/fixtures/` contains reusable test fixtures and setup code
- **Artifacts**: `tests/artifacts/` contains test assets like sample images, mock data files, and other test resources
  - `tests/artifacts/photos/` - Sample photos for testing image processing functionality
- **conftest.py**: Makes fixtures automatically available to all tests

### Test Types (Markers)

- `@pytest.mark.unit` - Fast unit tests for individual components
- `@pytest.mark.integration` - Integration tests for component interaction
- `@pytest.mark.slow` - Tests that take longer to run (can be skipped with `-m "not slow"`)
- `@pytest.mark.system` - End-to-end system tests

### Available Fixtures

- `temp_test_dir` - Temporary directory for test isolation
- `empty_cache_file` - Empty JSON cache file
- `sample_cache_file` - Pre-populated cache with sample data
- `mock_cluster_data` - Mock photo cluster for event naming tests
- `mock_media_file` - Mock MediaFile object factory
- `isolated_config` - Isolated configuration for testing

Each test file includes:

- Setup and initialization (via fixtures)
- Component testing
- Error handling verification
- Automatic cleanup
- Detailed output for debugging
