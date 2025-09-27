# Test Architecture Documentation

## Overview

This project uses a modern two-tier test architecture designed to provide fast development feedback while ensuring comprehensive validation. The architecture separates **unit tests** (fast, mocked) from **integration tests** (slower, real components) to optimize the development workflow.

## Folder Structure

```
tests/
├── unit/                     # Fast unit tests (~10 seconds total)
│   ├── __init__.py
│   ├── test_clustering_face_unit.py                    # Face recognition unit tests
│   └── test_clustering_face_integration_fake_BACKUP.py # Legacy mocked tests (backup)
├── integration/              # Real component integration tests (~10 seconds total)
│   ├── __init__.py
│   └── test_clustering_face_integration.py             # Real photo face recognition tests
├── artifacts/               # Test data and fixtures
│   ├── photos/             # Real photos for integration testing
│   │   ├── Woman_Photo_1.jpeg    # Elena Rodriguez (known person)
│   │   ├── Woman_Photo_2.jpeg    # Elena Rodriguez (same person)
│   │   └── No_Faces_Photo.jpeg   # Photo with no detectable faces
│   └── temp_test_dbs/      # Temporary test databases (auto-created)
├── test_*.py               # Legacy test files (various components)
├── conftest.py            # Shared pytest fixtures
└── README.md             # This file
```

## Test Categories

### Unit Tests (`tests/unit/`)

**Purpose**: Fast tests that validate individual component behavior in isolation

**Characteristics**:
- **Speed**: 0.1-0.5 seconds per test, ~10 seconds total
- **Dependencies**: Mocked external dependencies (no real photos, no real face recognition)
- **Focus**: Component initialization, configuration, logic validation, edge cases
- **Markers**: `@pytest.mark.unit`

**When to use**:
- Testing component initialization and configuration
- Validating business logic with controlled inputs
- Testing error handling and edge cases
- Regression prevention for specific bugs (e.g., MediaCluster constructor fixes)

**Example**: Testing that `MediaClusteringEngine` properly stores face recognition components without actually performing face recognition.

### Integration Tests (`tests/integration/`)

**Purpose**: End-to-end validation with real components and data

**Characteristics**:
- **Speed**: 5-20 seconds per test, varies by complexity
- **Dependencies**: Real photos, real face recognition, actual file I/O
- **Focus**: End-to-end workflows, component interaction, regression prevention
- **Markers**: `@pytest.mark.integration`, `@pytest.mark.slow`

**When to use**:
- Testing complete workflows with real data
- Validating face recognition with actual photos
- Ensuring components work together correctly
- Regression prevention for complex issues (e.g., Issue #13)

**Example**: Testing that the clustering pipeline correctly identifies "Elena Rodriguez" in real photos from `tests/artifacts/photos/`.

## Development Workflow

### Fast Development Cycle (~10 seconds)

```bash
# Run only unit tests for quick feedback
pytest tests/unit/ -v

# Run specific unit test file
pytest tests/unit/test_clustering_face_unit.py -v

# Run unit tests by marker
pytest -m "unit" -v
```

### Pre-commit Validation (~20 seconds)

```bash
# Run both unit and integration tests
pytest tests/ -m "unit or integration" -v

# Run all tests with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### Component-specific Testing

```bash
# Face recognition tests only
pytest tests/ -k "face" -v

# Clustering tests only
pytest tests/ -k "clustering" -v

# Regression tests only
pytest -m "regression" -v
```

### Comprehensive Validation

```bash
# All tests including legacy files
pytest tests/ -v

# All tests excluding slow ones
pytest tests/ -m "not slow" -v
```

## Legacy Test Files

The project maintains backward compatibility with existing test files:

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

```bash
# Run individual legacy tests (still supported)
python tests/test_content_analyzer.py
python tests/test_location_verification.py
```

## Pytest Markers

The project uses pytest markers to categorize tests:

```python
@pytest.mark.unit         # Fast unit tests with mocked dependencies
@pytest.mark.integration  # Real component integration tests
@pytest.mark.slow         # Tests that take >5 seconds
@pytest.mark.regression   # Tests that prevent specific bug regressions
@pytest.mark.system       # System-level tests
```

### Marker Usage Examples

```bash
# Run fast tests only
pytest -m "unit"

# Run integration tests only
pytest -m "integration"

# Skip slow tests
pytest -m "not slow"

# Run regression prevention tests
pytest -m "regression"

# Combine markers
pytest -m "unit or (integration and not slow)"
```

## Face Recognition Test Coverage

### Unit Tests (`test_clustering_face_unit.py`)

**Tests 17 scenarios** including:
- Component initialization with/without face recognition
- FaceRecognizer parameter storage
- PeopleDatabase person management
- Clustering enhancement logic with mocked face detection
- MediaCluster constructor validation (validates bug fixes)
- Issue #13 regression prevention

### Integration Tests (`test_clustering_face_integration.py`)

**Tests 5 real-world scenarios** including:
- Real face recognition with actual photos
- Full clustering pipeline with real components
- End-to-end MediaClusteringEngine workflows
- Real people database creation and management
- Issue #13 regression prevention with real data

## Test Data Management

### Real Photos (`tests/artifacts/photos/`)

- **Woman_Photo_1.jpeg**: Contains Elena Rodriguez (used to train people database)
- **Woman_Photo_2.jpeg**: Contains Elena Rodriguez (used to test recognition)
- **No_Faces_Photo.jpeg**: Contains no detectable faces (negative test case)

### Temporary Test Databases

Integration tests create temporary people databases in `tests/artifacts/temp_test_dbs/` to avoid interfering with the main people database.

## Available Fixtures

- `temp_test_dir` - Temporary directory for test isolation
- `empty_cache_file` - Empty JSON cache file
- `sample_cache_file` - Pre-populated cache with sample data
- `mock_cluster_data` - Mock photo cluster for event naming tests
- `mock_media_file` - Mock MediaFile object factory
- `isolated_config` - Isolated configuration for testing

## Test Requirements

Some tests require additional dependencies:

- `torch` and `transformers` for advanced content analysis
- Internet connection for geocoding tests
- `dlib` and `face_recognition` for face recognition tests
- Sample photos in `Sample_Photos/iPhone Automatic/` directory

## Writing New Tests

### Unit Test Guidelines

```python
import pytest
from unittest.mock import Mock, MagicMock

@pytest.mark.unit
def test_component_with_mocked_dependencies():
    """Test component behavior with mocked dependencies."""
    # Create mocks
    mock_dependency = Mock(spec=SomeDependency)
    mock_dependency.some_method.return_value = "expected_result"

    # Test component
    component = SomeComponent(dependency=mock_dependency)
    result = component.do_something()

    # Verify behavior
    assert result == "expected_result"
    mock_dependency.some_method.assert_called_once()
```

### Integration Test Guidelines

```python
import pytest
from pathlib import Path

@pytest.mark.integration
@pytest.mark.slow
def test_real_workflow_with_actual_data():
    """Test complete workflow with real data."""
    # Use real components and data
    real_photos = list(Path("tests/artifacts/photos").glob("*.jpeg"))
    component = RealComponent()

    # Test end-to-end workflow
    result = component.process_photos(real_photos)

    # Verify real results
    assert len(result.detected_people) > 0
    assert "Elena Rodriguez" in result.detected_people
```

## Performance Expectations

### Current Performance (as of implementation)

- **Unit Tests**: 17 tests in ~10 seconds (0.6 seconds average)
- **Integration Tests**: 5 tests in ~10 seconds (2 seconds average)
- **Total Test Suite**: 22 tests in ~20 seconds
- **Legacy Tests**: Various (maintained for backward compatibility)

### Performance Targets

- **Unit tests**: Should remain under 15 seconds total
- **Integration tests**: Should remain under 30 seconds total
- **Individual test**: Unit tests <1 second, integration tests <10 seconds

## Best Practices

### Do's
- ✅ Use mocks in unit tests to isolate components
- ✅ Use real data in integration tests for end-to-end validation
- ✅ Add appropriate pytest markers to all tests
- ✅ Keep unit tests under 1 second each
- ✅ Use descriptive test names that explain what is being tested
- ✅ Create dedicated fixtures for common test scenarios

### Don'ts
- ❌ Don't use real photos in unit tests (use mocks)
- ❌ Don't use mocks in integration tests (use real components)
- ❌ Don't create tests that depend on external services
- ❌ Don't write tests that modify production databases
- ❌ Don't create overly complex test fixtures
- ❌ Don't ignore test performance (monitor test execution time)

This architecture ensures fast development feedback while maintaining comprehensive test coverage and preventing regressions.
