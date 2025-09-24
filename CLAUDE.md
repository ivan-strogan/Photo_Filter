# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

```bash
# Run all tests (RSpec-style - recommended)
pytest tests/

# Run all tests with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_content_analyzer.py

# Run tests matching a pattern
pytest -k "event_naming"

# Run tests by category/marker
pytest -m "unit"                # Run only unit tests
pytest -m "not slow"            # Skip slow tests
pytest -m "integration"         # Run only integration tests

# Show test coverage
pytest tests/ --cov=src

# Run individual test files (legacy method)
python tests/test_content_analyzer.py
python tests/test_location_verification.py
python tests/test_components.py

# Run comprehensive system tests
python tests/system_integration_test.py
python tests/basic_system_test.py

# Run demo scripts for testing specific features
python tests/intelligent_naming_demo.py
python test_sample_photos_demo.py
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