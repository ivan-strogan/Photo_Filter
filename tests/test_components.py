#!/usr/bin/env python3
"""Test script for individual components."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.media_detector import MediaDetector
from src.metadata_extractor import MetadataExtractor
from src.vector_database import VectorDatabase
from src.gpu_utils import gpu_manager

def test_media_detection():
    """Test media file detection."""
    print("=== Testing Media Detection ===")

    detector = MediaDetector()

    # Test filename parsing
    test_files = [
        "IMG_20221014_061257.JPG",
        "IMG_20141025_163037.MOV",
        "IMG_20160312_142530.PNG",
        "invalid_filename.jpg"
    ]

    for filename in test_files:
        result = detector.parse_filename(filename)
        print(f"{filename}: {result}")

    # Test directory scanning
    print("\nScanning iPhone Automatic folder...")
    iphone_files = detector.scan_iphone_automatic()
    print(f"Found {len(iphone_files)} files")

    if iphone_files:
        print(f"First file: {iphone_files[0].filename} ({iphone_files[0].file_type})")
        stats = detector.get_media_stats(iphone_files)
        print(f"Stats: {stats}")

def test_metadata_extraction():
    """Test metadata extraction from photos."""
    print("\n=== Testing Metadata Extraction ===")

    detector = MediaDetector()
    extractor = MetadataExtractor()

    # Get first photo from iPhone Automatic
    iphone_files = detector.scan_iphone_automatic()
    photo_files = [f for f in iphone_files if f.file_type == 'photo']

    if photo_files:
        first_photo = photo_files[0]
        print(f"Extracting metadata from: {first_photo.filename}")

        metadata = extractor.extract_photo_metadata(first_photo)

        print("Extracted metadata:")
        for key, value in metadata.items():
            if value is not None:
                print(f"  {key}: {value}")

def test_gpu_detection():
    """Test GPU detection and configuration."""
    print("\n=== Testing GPU Detection ===")

    gpu_manager.print_gpu_summary()

    print(f"Device: {gpu_manager.get_device()}")
    print(f"GPU Available: {gpu_manager.is_gpu_available()}")
    print(f"Optimal batch size: {gpu_manager.get_optimal_batch_size()}")

    if gpu_manager.is_gpu_available():
        memory_info = gpu_manager.get_memory_usage()
        print(f"Memory info: {memory_info}")

def test_vector_database():
    """Test vector database operations."""
    print("\n=== Testing Vector Database ===")

    # Initialize database
    db = VectorDatabase()

    # Get stats
    stats = db.get_database_stats()
    print(f"Database stats: {stats}")

    # Test adding a dummy embedding
    import numpy as np
    dummy_embedding = np.random.rand(512)
    dummy_metadata = {
        'filename': 'test_photo.jpg',
        'creation_time': '2024-01-01',
        'camera_make': 'Test Camera'
    }

    success = db.add_photo_embedding(
        photo_id="test_001",
        embedding=dummy_embedding,
        metadata=dummy_metadata
    )

    print(f"Added test embedding: {success}")

    # Test search
    similar = db.search_similar_photos(dummy_embedding, n_results=5)
    print(f"Found {len(similar)} similar photos")

if __name__ == "__main__":
    test_media_detection()
    test_metadata_extraction()
    test_gpu_detection()
    test_vector_database()

    print("\n=== All Tests Completed ===")