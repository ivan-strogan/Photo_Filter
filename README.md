# Photo Filter AI App

An intelligent photo organization system that automatically clusters and organizes photos and videos based on temporal patterns, GPS location data, and visual content analysis.

## Features

- 📸 **Smart Photo Detection**: Recognizes iPhone photo/video naming format (IMG_YYYYMMDD_HHMMSS.JPG/.MOV)
- 🕒 **Temporal Clustering**: Groups media by time proximity with intelligent algorithms
- 📍 **Location-Based Grouping**: Uses GPS metadata and reverse geocoding for location clustering
- 🤖 **Computer Vision Analysis**: Analyzes photo content for objects, scenes, and activities
- 🧠 **Vector Database**: Uses CLIP embeddings for visual similarity matching with intelligent caching
- 🎬 **Video Processing**: Intelligent frame extraction and vectorization for video files
- 🤖 **Smart Learning**: Learns from existing organized photos to improve future naming
- ⚡ **Duplicate Detection**: Skips already-processed photos for lightning-fast subsequent runs
- ⚙️ **Configurable Parameters**: Customizable clustering thresholds and processing settings
- 🚀 **GPU Acceleration**: Supports CUDA and Apple MPS for faster processing
- 📊 **Comprehensive Logging**: Detailed progress tracking and session reports

## Project Structure

```
Photo_Filter/
├── src/                     # Main source code
│   ├── config.py           # Configuration settings
│   ├── media_detector.py   # File detection and parsing
│   ├── metadata_extractor.py # EXIF and GPS metadata extraction
│   ├── content_analyzer.py # Computer vision content analysis
│   ├── temporal_clustering.py # Time-based clustering algorithms
│   ├── geocoding.py        # Location services and reverse geocoding
│   ├── media_clustering.py # Comprehensive clustering engine
│   ├── config_manager.py   # Configuration management system
│   ├── photo_vectorizer.py # CLIP-based image & video vectorization
│   ├── vector_database.py  # ChromaDB vector storage with smart caching
│   ├── organized_photos_scanner.py # Learns from existing organized photos
│   └── ...
├── tests/                  # Test suite
│   ├── test_content_analyzer.py
│   ├── test_location_verification.py
│   └── ...
├── data/                   # Data files (config, cache)
├── logs/                   # Log files
├── main.py                # CLI interface
├── requirements.txt       # Python dependencies
└── todo.txt              # Development roadmap
```

## Requirements

- **Python 3.11** (required for AI models)
- **CLIP and BLIP models** (required for content analysis)
- macOS, Linux, or Windows
- Optional: For face recognition features: CMake and dlib (see installation notes below)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository_url>
   cd Photo_Filter
   ```

2. **Create virtual environment with Python 3.11**
   ```bash
   python3.11 -m venv venv_py311
   source venv_py311/bin/activate  # On Windows: venv_py311\\Scripts\\activate
   ```

3. **Install core dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Optional: Install face recognition dependencies**
   ```bash
   # macOS with Homebrew
   brew install cmake
   brew install dlib

   # Then install face-recognition (already in requirements.txt)
   pip install face-recognition
   ```

## Usage

### Prerequisites - Start Ollama LLM Server

**IMPORTANT**: Before running the photo organization pipeline, you must start the Ollama LLM server for intelligent event naming.

```bash
# In a separate terminal window, start Ollama server
ollama serve
```

Keep this terminal running while you use the Photo Filter app. The event naming system requires an active LLM connection and will fail if Ollama is not running.

**Install Ollama** (if not already installed):
```bash
# macOS/Linux
curl https://ollama.ai/install.sh | sh

# Or download from https://ollama.ai
```

**Download the required model**:
```bash
ollama pull llama3.1:8b
```

### CLI Commands

```bash
# Activate virtual environment
source venv_py311/bin/activate

# Scan existing organized photos
python main.py scan

# Process new unorganized photos (preview mode - doesn't move files)
python main.py process --dry-run

# Process and actually organize photos (moves/copies files to folders)
python main.py process --no-dry-run

# Analyze photo content
python main.py analyze-content --max-photos 10

# Show system status
python main.py status

# Configuration management
python main.py config show
python main.py config update --time-threshold 8.0
python main.py config reset

# Face recognition management
python main.py faces scan              # Build face database from organized photos
python main.py faces add "John Doe" photo1.jpg photo2.jpg  # Add person
python main.py faces list              # List known people
python main.py faces remove "John Doe" # Remove person
python main.py faces status            # Show face recognition status
```

### Configuration

The system uses a JSON configuration file with customizable parameters:

```json
{
  "clustering": {
    "time_threshold_hours": 6.0,
    "location_threshold_km": 1.0,
    "min_cluster_size": 3
  },
  "processing": {
    "max_photos_per_event": 50,
    "use_gpu": true,
    "enable_vectorization": true
  }
}
```

## File Organization

### Input Structure
```
Sample_Photos/
├── iPhone Automatic/       # Unorganized photos from iPhone
│   ├── IMG_20241024_143000.JPG
│   ├── IMG_20241024_143015.MOV
│   └── ...
└── Pictures/              # Existing organized photos
    ├── 2024/
    │   ├── 2024_10_24 - Mexico Vacation/
    │   └── 2024_11_15 - Birthday Party/
    └── ...
```

### Output Format
The system suggests organized folder names like:
- `2024_10_24 - Mexico Vacation`
- `2024_11_15 - Birthday Party - Quick Event`
- `2024_12_25 - Edmonton - All Day`

## Clustering Algorithm

The system uses a multi-stage clustering approach:

1. **Temporal Clustering**: Groups photos by time proximity using three algorithms:
   - `by_time`: Continuous time-based grouping
   - `by_day`: Day-boundary aware clustering
   - `activity_periods`: Natural activity pattern detection

2. **Location Enhancement**: Refines clusters using GPS coordinates and reverse geocoding

3. **Content Analysis**: Analyzes visual content for objects, scenes, and activities

4. **Confidence Scoring**: Calculates cluster quality based on multiple signals

## Testing

Run the test suite:

```bash
# Run individual tests
python tests/test_content_analyzer.py
python tests/test_location_verification.py

# Run all tests (with pytest)
pytest tests/
```

## Development Status

**Completed Features (18/24):**
- ✅ Media detection and parsing
- ✅ Metadata extraction (photos & videos)
- ✅ Temporal clustering algorithms
- ✅ Location-based clustering
- ✅ Computer vision content analysis
- ✅ Vector database integration
- ✅ Configuration management
- ✅ CLI interface
- ✅ Face detection and recognition

**In Progress:**
- ⚠️ LLM integration for intelligent event naming

**Remaining Features:**
- 🔄 Video content analysis
- 🔄 Automated folder creation
- 🔄 Media moving/copying system

## Dependencies

### Core Dependencies
- `click` - CLI interface
- `Pillow` - Image processing
- `exifread` - EXIF metadata extraction
- `geopy` - Geocoding services
- `numpy` - Numerical operations
- `python-dateutil` - Date parsing

### Required ML Dependencies
- `torch` - PyTorch for neural networks (REQUIRED)
- `transformers` - Hugging Face models for CLIP/BLIP (REQUIRED)
- `sentence-transformers` - Text embeddings
- `chromadb` - Vector database

Note: The application requires local AI models for content analysis. Install all dependencies from requirements.txt.

## Configuration

Key settings can be adjusted via the configuration system:

- **Time Threshold**: How close in time photos should be to cluster together
- **Location Threshold**: Geographic distance for location-based clustering
- **Cluster Size**: Minimum number of photos to form an event
- **GPU Usage**: Enable/disable GPU acceleration
- **Processing Limits**: Maximum photos per event for efficiency

## License

[Add your license here]

## Development Workflow

**MANDATORY**: All contributions must follow this GitHub workflow for proper tracking:

### 1. Create Issue First
```bash
gh issue create --title "type: description" --body "Requirements and acceptance criteria"
```

### 2. Create Feature Branch
```bash
git checkout -b feature/ISSUE#-description
# Example: git checkout -b feature/1-github-workflow-docs
```

### 3. Make Changes & Commit
```bash
git add .
git commit -m "type: description - addresses issue #ISSUE#"
```

### 4. Create Pull Request
```bash
git push -u origin feature/ISSUE#-description
gh pr create --title "type: description" --body "Fixes #ISSUE#"
```

Issues automatically close when PRs are merged, maintaining complete audit trails.

## Contributing

See [CLAUDE.md](./CLAUDE.md) for detailed development guidelines including:
- Architecture overview and data flow
- Testing procedures with pytest
- Configuration management
- Core commands and virtual environment setup# Photo_Filter
