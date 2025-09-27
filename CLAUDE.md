# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Version Control Guidelines

**IMPORTANT**: This project is now under proper version control with GitHub. All changes must be committed and versioned properly:

- **Always commit meaningful changes** with descriptive commit messages
- **Follow conventional commit format**: `type: description` (e.g., `feat: add face recognition`, `fix: resolve event naming bug`, `docs: update README`)
- **Test before committing**: Run `pytest tests/` to ensure all tests pass
- **Environment files**: The `.env` file is ignored by git - only `.env.example` should be committed
- **No direct pushes**: All code changes should be properly staged, committed, and ready for review
- **Branch management**: Use feature branches for significant changes, merge to master when stable

## GitHub Workflow Process

**MANDATORY**: All code changes must follow this GitHub workflow to ensure proper tracking and audit trails:

### 1. Create GitHub Issue
```bash
# Create issue for any task, bug fix, or feature
gh issue create --title "type: brief description" --body "Detailed description with requirements and acceptance criteria"
```

### 2. Create Feature Branch
```bash
# Always include issue number in branch name
git checkout -b feature/ISSUE#-descriptive-name
# Example: git checkout -b feature/1-github-workflow-docs
```

### 3. Make Changes and Commit
```bash
# Follow conventional commit format
git add .
git commit -m "type: description - addresses issue #ISSUE#"
# Example: git commit -m "docs: add GitHub workflow process - addresses issue #1"
```

### 4. Create Pull Request
```bash
# Push branch and create PR with issue linking
git push -u origin feature/ISSUE#-descriptive-name
gh pr create --title "type: description" --body "## Summary
Brief description of changes

## Closes
Fixes #ISSUE#

## Test plan
- [ ] Tests pass
- [ ] Code follows project conventions"
```

### 5. Auto-close Issues
- Use "Fixes #ISSUE#", "Closes #ISSUE#", or "Resolves #ISSUE#" in PR body
- Issues automatically close when PR is merged
- Maintains complete audit trail from issue → branch → PR → merge

### Branch Naming Convention
- `feature/ISSUE#-short-description` - New features
- `fix/ISSUE#-short-description` - Bug fixes
- `docs/ISSUE#-short-description` - Documentation updates
- `refactor/ISSUE#-short-description` - Code refactoring

This workflow ensures every change is:
- ✅ Properly tracked with GitHub issues
- ✅ Linked to specific branches and PRs
- ✅ Documented with clear commit messages
- ✅ Automatically closed when merged
- ✅ Maintains complete project history

## Project Overview

Photo Filter AI is an intelligent photo organization system that automatically clusters and organizes photos/videos using temporal patterns, GPS location data, and computer vision analysis. It's designed to process iPhone photo exports and organize them into meaningful events.

## Development Environment

### Requirements
- **Python 3.11** (required for full ML functionality)
- For face recognition: `brew install cmake && brew install dlib`

### Virtual Environment Setup
```bash
# Create virtual environment with Python 3.11
python3.11 -m venv venv_py311

# Activate virtual environment (required for all development)
source venv_py311/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Core Commands

```bash
# Main CLI commands (run with virtual environment activated)
python main.py scan                    # Scan existing organized photos
python main.py process --dry-run       # Process new photos (preview mode)
python main.py pipeline               # Run complete processing pipeline
python main.py analyze-content --max-photos 10  # Analyze photo content
python main.py status                 # Show system status
python main.py config show            # Display current configuration
python main.py config update --time-threshold 8.0  # Update config parameters
```

### Testing

**Test Architecture**: The project uses a two-tier test architecture with separate unit and integration tests for optimal development workflow:

```
tests/
├── unit/                     # Fast unit tests (~15 seconds total)
│   ├── test_clustering_face_unit.py                    # Face recognition unit tests (17 tests)
│   ├── test_event_namer_unit.py                        # Event naming unit tests (6 tests)
│   └── test_clustering_face_integration_fake_BACKUP.py # Legacy mocked tests (backup)
├── integration/              # Slower integration tests (~15 seconds total)
│   ├── test_clustering_face_integration.py             # Real photo face recognition tests (5 tests)
│   └── test_event_naming_integration.py                # Event naming integration tests (5 tests)
├── artifacts/               # Test data and fixtures
│   └── photos/             # Real photos for integration testing
├── test_*.py               # Legacy test files (various components)
└── README.md              # Detailed testing documentation
```

#### Development Workflow Commands

```bash
# Development (fast feedback) - ~15 seconds
pytest tests/unit/ -v                    # Run only unit tests

# Pre-commit validation - ~30 seconds
pytest tests/ -m "unit or integration"   # Run both unit and integration tests

# Quick component testing
pytest tests/unit/test_clustering_face_unit.py -v        # Face recognition unit tests
pytest tests/integration/test_clustering_face_integration.py -v  # Face recognition integration tests

# Test by markers (recommended)
pytest -m "unit"                # Fast unit tests only
pytest -m "integration"         # Real component integration tests
pytest -m "regression"          # Regression prevention tests
pytest -m "not slow"            # Skip slow tests

# Pattern matching
pytest -k "face_recognition"    # Tests containing "face_recognition"
pytest -k "clustering"         # Tests containing "clustering"

# Coverage analysis
pytest tests/ --cov=src --cov-report=html

# All tests with verbose output
pytest tests/ -v
```

#### Legacy Test Files (Individual Components)

```bash
# Run individual component tests (legacy approach)
python tests/test_content_analyzer.py
python tests/test_location_verification.py
python tests/test_components.py

# System integration tests
python tests/system_integration_test.py
python tests/basic_system_test.py

# Demo scripts for specific features
python tests/intelligent_naming_demo.py
python test_sample_photos_demo.py
```

#### Face Recognition Testing

The face recognition system has comprehensive test coverage:

- **Unit Tests** (`tests/unit/test_clustering_face_unit.py`): Fast tests with mocked components, validates component initialization, configuration, and logic without external dependencies
- **Integration Tests** (`tests/integration/test_clustering_face_integration.py`): Real photo tests using actual face recognition with test photos from `tests/artifacts/photos/`
- **Regression Tests**: Prevents regression of Issue #13 (face recognition integration in clustering pipeline)

```bash
# Run face recognition tests specifically
pytest tests/ -k "face" -v                    # All face recognition tests
pytest tests/unit/test_clustering_face_unit.py -v        # Unit tests only
pytest tests/integration/test_clustering_face_integration.py -v  # Integration tests only
```

#### Event Naming Testing

The event naming system has comprehensive test coverage for LLM integration and location accuracy:

- **Unit Tests** (`tests/unit/test_event_namer_unit.py`): Fast tests with mocked LLM responses, validates prompt generation, location constraints, and anti-hallucination measures
- **Integration Tests** (`tests/integration/test_event_naming_integration.py`): End-to-end tests with real context processing and validation system verification
- **Regression Tests**: Prevents regression of Issue #14 (LLM location hallucination) and validates Issue #15 (validation system accuracy)

```bash
# Run event naming tests specifically
pytest tests/ -k "event_nam" -v               # All event naming tests
pytest tests/unit/test_event_namer_unit.py -v            # Unit tests only
pytest tests/integration/test_event_naming_integration.py -v  # Integration tests only
```

## Architecture

### Core Processing Pipeline
1. **PhotoOrganizerPipeline** (`src/photo_organizer_pipeline.py`) - Main orchestration engine for complete processing workflow
2. **MediaDetector** (`src/media_detector.py`) - Finds and parses iPhone media files (IMG_YYYYMMDD_HHMMSS format)
3. **TemporalClustering** (`src/temporal_clustering.py`) - Groups photos by time using multiple algorithms (`by_time`, `by_day`, `activity_periods`)
4. **ContentAnalyzer** (`src/content_analyzer.py`) - Computer vision analysis using CLIP/BLIP models
5. **MediaClustering** (`src/media_clustering.py`) - Comprehensive clustering engine combining temporal, location, and content signals
6. **OrganizedPhotosScanner** (`src/organized_photos_scanner.py`) - Scans existing organized photos to build vector database
7. **PhotoVectorizer** (`src/photo_vectorizer.py`) - Creates CLIP embeddings for photos and videos

### Key Modules
- **ConfigManager** (`src/config_manager.py`) - Centralized configuration management with JSON persistence
- **MetadataExtractor** (`src/metadata_extractor.py`) - EXIF/GPS metadata extraction for photos and videos
- **Geocoding** (`src/geocoding.py`) - Location services and reverse geocoding with caching
- **EventNamer** (`src/event_namer.py`) - Intelligent event naming with LLM integration (OpenAI/Ollama)
- **FolderOrganizer** (`src/folder_organizer.py`) - Creates organized folder structures
- **FileOrganizer** (`src/file_organizer.py`) - Media file moving/copying operations
- **VectorDatabase** (`src/vector_database.py`) - ChromaDB for CLIP embeddings and similarity matching
- **MediaValidator** (`src/media_validator.py`) - File validation and corruption detection
- **FaceRecognizer** (`src/face_recognizer.py`) - Face detection and person identification with caching
- **PeopleDatabase** (`src/people_database.py`) - Persistent storage for known people and face encodings
- **GPUUtils** (`src/gpu_utils.py`) - GPU acceleration management for CUDA/MPS

### Data Flow
```
Stage 0: VectorDB → OrganizedPhotosScanner → PhotoVectorizer → ChromaDB (learning from existing)
Stage 1: Scan → MediaDetector → MediaValidator → File selection
Stage 2: Clustering → TemporalClustering → LocationEnrichment → LocationRefinement → ConfidenceScoring
Stage 3: Naming → EventNamer (with LLM) → VectorSimilarity matching
Stage 4: Folders → FolderOrganizer → Conflict resolution
Stage 5: Organization → FileOrganizer → Media moving/copying
```

## Configuration

The system uses `photo_filter_config.json` for configuration. Key parameters:
- `clustering.time_threshold_hours`: Time proximity for clustering (default: 6.0)
- `clustering.location_threshold_km`: Geographic distance threshold (default: 1.0)
- `clustering.min_cluster_size`: Minimum files per cluster (default: 1)
- `processing.use_gpu`: Enable GPU acceleration for ML models
- `processing.enable_vectorization`: Enable CLIP embeddings for similarity
- `processing.max_photos_per_event`: Maximum photos to process per event (default: 50)
- `naming.use_llm_naming`: Enable LLM-powered event naming
- `naming.llm_model`: LLM model to use ("gpt-3.5-turbo" or local Ollama models)

## File Structure

```
Sample_Photos/
├── iPhone Automatic/     # Unorganized photos from iPhone export
└── Pictures/            # Organized photos in YYYY/YYYY_MM_DD - Event Name format
    └── YYYY/            # Year-based organization
        ├── Photos To Filter/  # Unfiltered photos scoped to this year
        ├── YYYY_MM_DD - Event Name/
        └── YYYY_MM_DD - Another Event/

Output format: "2024_10_24 - Mexico Vacation" or "2024_11_15 - Birthday Party - Quick Event"
```

## Development Notes

- **Requires Python 3.11** for full compatibility with all ML dependencies
- Always activate the virtual environment before running any commands: `source venv_py311/bin/activate`
- The project supports both CPU and GPU processing (CUDA/MPS)
- Extensive logging is available in `logs/photo_filter.log`
- The system maintains a location cache in `data/location_cache.json` for performance
- Vector embeddings are stored in `vector_db/` directory using ChromaDB
- Configuration is persisted in `photo_filter_config.json` with automatic validation
- Smart duplicate detection prevents re-processing already vectorized photos
- Face recognition system supports custom cache file paths for test isolation
- Most test files can be run directly as scripts for debugging specific components

## Known Issues & Bug Fixes Needed

See `todo.txt` for current bug fixes and improvements needed. Key priority items:
- Fix configuration inconsistency bug (MIN_CLUSTER_SIZE values)
- Replace debug print statements with proper logging
- Implement video GPS extraction
- Optimize GPS coordinate matching performance

## LLM Integration

The system supports multiple LLM backends for intelligent event naming:
- **OpenAI GPT models** (requires API key)
- **Local Ollama models** (llama3.1:8b, phi3:mini, etc.)
- **Template-based fallback** (rule-based naming when LLM unavailable)

Vector similarity matching helps learn from existing organized photos to improve naming consistency.