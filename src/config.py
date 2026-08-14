"""Configuration settings for the Photo Filter application."""

import os
from pathlib import Path

try:
    from .environment_config import get_iphone_automatic_dir
except ImportError:
    from environment_config import get_iphone_automatic_dir

# Base paths
BASE_DIR = Path(__file__).parent.parent
SAMPLE_PHOTOS_DIR = BASE_DIR / "Sample_Photos"
# Overridable via PHOTO_FILTER_IPHONE_AUTOMATIC_DIR in .env - see
# environment_config.py. Defaults to the real Sample_Photos folder.
IPHONE_AUTOMATIC_DIR = get_iphone_automatic_dir()
PICTURES_DIR = SAMPLE_PHOTOS_DIR / "Pictures"

# Data directories
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Vector database settings
VECTOR_DB_DIR = BASE_DIR / "vector_db"
COLLECTION_NAME = "photo_embeddings"

# Supported file formats
PHOTO_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
VIDEO_EXTENSIONS = {'.mov', '.mp4', '.avi'}
SUPPORTED_EXTENSIONS = PHOTO_EXTENSIONS | VIDEO_EXTENSIONS

# File naming pattern - more flexible to handle existing files
FILENAME_PATTERN = r"IMG_(\d{8})_(\d{6})(?:_\d+)?\.(JPG|MOV|PNG|jpg|mov|png)$"

# Clustering parameters
TIME_THRESHOLD_HOURS = 6  # Group photos within 6 hours
LOCATION_THRESHOLD_KM = 1.0  # Group photos within 1km
MIN_CLUSTER_SIZE = 1  # Minimum photos for an event (allow single-photo events)

# GPU settings
USE_GPU = True  # Enable GPU if available
DEVICE = "cuda" if USE_GPU else "cpu"

# API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")