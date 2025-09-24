#!/usr/bin/env python3
"""Basic test script without ML dependencies."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def test_media_detection():
    """Test media file detection."""
    print("=== Testing Media Detection ===")

    from src.media_detector import MediaDetector

    detector = MediaDetector()

    # Test filename parsing
    test_files = [
        "IMG_20221014_061257.JPG",
        "IMG_20141025_163037.MOV",
        "IMG_20160312_142530.PNG",
        "invalid_filename.jpg"
    ]

    print("Testing filename parsing:")
    for filename in test_files:
        result = detector.parse_filename(filename)
        if result:
            dt, ext = result
            print(f"  ✅ {filename}: {dt.strftime('%Y-%m-%d %H:%M:%S')} (.{ext})")
        else:
            print(f"  ❌ {filename}: Failed to parse")

    # Test directory scanning
    print("\nScanning iPhone Automatic folder...")
    iphone_files = detector.scan_iphone_automatic()
    print(f"Found {len(iphone_files)} files")

    if iphone_files:
        print(f"First few files:")
        for i, f in enumerate(iphone_files[:5]):
            print(f"  {f.filename} ({f.file_type}) - {f.time}")

        stats = detector.get_media_stats(iphone_files)
        print(f"\nStats: {stats}")

    print("\nScanning Pictures library...")
    pictures_files = detector.scan_pictures_library()
    print(f"Found {len(pictures_files)} organized files")

    if pictures_files:
        # Group by event folder
        events = {}
        for photo in pictures_files:
            event = getattr(photo, 'event_folder', 'Unknown')
            if event not in events:
                events[event] = []
            events[event].append(photo)

        print(f"Found {len(events)} event folders:")
        for event, photos in events.items():
            print(f"  {event}: {len(photos)} files")

def test_metadata_extraction():
    """Test metadata extraction from photos."""
    print("\n=== Testing Metadata Extraction ===")

    from src.media_detector import MediaDetector
    from src.metadata_extractor import MetadataExtractor

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
    else:
        print("No photos found for metadata extraction")

def test_basic_gpu():
    """Test basic GPU detection without PyTorch."""
    print("\n=== Testing Basic System Info ===")

    import platform
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")

    # Check if PyTorch would be available
    try:
        import torch
        print(f"PyTorch available: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if hasattr(torch.backends, 'mps'):
            print(f"MPS available: {torch.backends.mps.is_available()}")
    except ImportError:
        print("PyTorch not installed - GPU features will be CPU-only")

if __name__ == "__main__":
    test_media_detection()
    test_metadata_extraction()
    test_basic_gpu()

    print("\n=== Basic Tests Completed ===")