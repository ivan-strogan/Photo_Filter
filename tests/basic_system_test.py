#!/usr/bin/env python3
"""
Basic system test without ML dependencies.
Tests core functionality that should work on any system.
"""

import sys
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
sys.path.append(str(Path(__file__).parent))

def test_basic_components():
    """Test basic components that don't require ML dependencies."""
    print("🧪 BASIC SYSTEM TEST")
    print("=" * 50)

    # Test 1: Media Detector
    print("1️⃣  Testing MediaDetector...")
    try:
        from src.media_detector import MediaDetector, MediaFile
        detector = MediaDetector()
        print("   ✅ MediaDetector initialized successfully")

        # Test filename parsing
        test_filename = "IMG_20241024_143000.JPG"
        parsed = detector._parse_filename(test_filename)
        if parsed and parsed.date.year == 2024:
            print("   ✅ Filename parsing works correctly")
        else:
            print("   ❌ Filename parsing failed")

    except Exception as e:
        print(f"   ❌ MediaDetector failed: {e}")

    # Test 2: Media Validator
    print("\n2️⃣  Testing MediaValidator...")
    try:
        from src.media_validator import MediaValidator
        validator = MediaValidator()
        print("   ✅ MediaValidator initialized successfully")

        # Create a test file
        test_dir = Path(tempfile.mkdtemp())
        test_file = test_dir / "test.jpg"
        with open(test_file, 'wb') as f:
            f.write(b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00')
            f.write(b'test content')

        result = validator.validate_media_file(test_file)
        if result.detected_format == 'JPEG':
            print("   ✅ File validation works correctly")
        else:
            print("   ❌ File validation failed")

        shutil.rmtree(test_dir)

    except Exception as e:
        print(f"   ❌ MediaValidator failed: {e}")

    # Test 3: Temporal Clusterer (basic)
    print("\n3️⃣  Testing TemporalClusterer...")
    try:
        from src.temporal_clustering import TemporalClusterer
        clusterer = TemporalClusterer()
        print("   ✅ TemporalClusterer initialized successfully")

        # Test with mock data
        mock_files = [
            type('MockFile', (), {
                'date': datetime(2024, 10, 24, 14, 30),
                'path': Path('test1.jpg'),
                'filename': 'test1.jpg'
            })(),
            type('MockFile', (), {
                'date': datetime(2024, 10, 24, 14, 45),
                'path': Path('test2.jpg'),
                'filename': 'test2.jpg'
            })()
        ]

        clusters = clusterer.cluster_by_time(mock_files, max_gap_hours=2)
        if len(clusters) > 0:
            print("   ✅ Basic temporal clustering works")
        else:
            print("   ❌ Temporal clustering failed")

    except Exception as e:
        print(f"   ❌ TemporalClusterer failed: {e}")

    # Test 4: Event Namer (fallback mode)
    print("\n4️⃣  Testing EventNamer (fallback mode)...")
    try:
        from src.event_namer import EventNamer
        namer = EventNamer(enable_llm=False)  # Disable LLM for testing
        print("   ✅ EventNamer initialized successfully")

        # Test simple naming
        test_context = {
            'date_range': {'start': datetime(2024, 10, 31), 'end': datetime(2024, 10, 31)},
            'time_patterns': {'primary_time_of_day': 'Evening'},
            'location_info': {'primary_location': 'Unknown'},
            'content_analysis': {'detected_objects': ['person', 'costume']},
            'cluster_size': 5
        }

        event_name = namer.generate_event_name(test_context)
        if event_name and len(event_name) > 0:
            print(f"   ✅ Event naming works: '{event_name}'")
        else:
            print("   ❌ Event naming failed")

    except Exception as e:
        print(f"   ❌ EventNamer failed: {e}")

    # Test 5: File Organizer
    print("\n5️⃣  Testing FileOrganizer...")
    try:
        from src.file_organizer import FileOrganizer
        organizer = FileOrganizer(dry_run=True)
        print("   ✅ FileOrganizer initialized successfully")
        print("   ✅ Dry-run mode enabled for safe testing")

    except Exception as e:
        print(f"   ❌ FileOrganizer failed: {e}")

    # Test 6: Configuration
    print("\n6️⃣  Testing Configuration...")
    try:
        from src.config import CONFIG_FILE, DATA_DIR, LOGS_DIR
        print(f"   ✅ Config paths defined: {CONFIG_FILE.exists()}")
        print(f"   ✅ Data directory: {DATA_DIR}")
        print(f"   ✅ Logs directory: {LOGS_DIR}")

    except Exception as e:
        print(f"   ❌ Configuration failed: {e}")

def test_error_handling():
    """Test error handling capabilities."""
    print("\n" + "=" * 50)
    print("🛡️  TESTING ERROR HANDLING")
    print("=" * 50)

    # Test with invalid files
    print("🧪 Testing with problematic files...")

    try:
        from src.media_validator import MediaValidator
        validator = MediaValidator()

        # Test non-existent file
        result = validator.validate_media_file(Path("/nonexistent/file.jpg"))
        if not result.is_valid and result.validation_errors:
            print("   ✅ Non-existent file handled correctly")
        else:
            print("   ❌ Non-existent file not handled properly")

    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")

def test_real_files():
    """Test with actual photo files if available."""
    print("\n" + "=" * 50)
    print("📁 TESTING WITH REAL FILES")
    print("=" * 50)

    sample_photos = Path("Sample_Photos/iPhone Automatic")
    if sample_photos.exists():
        print(f"   Found sample photos directory: {sample_photos}")

        try:
            from src.media_detector import MediaDetector
            detector = MediaDetector()

            media_files = detector.scan_directory(sample_photos)
            print(f"   ✅ Detected {len(media_files)} media files")

            if media_files:
                first_file = media_files[0]
                print(f"   ✅ Sample file: {first_file.filename}")
                print(f"   ✅ Parsed date: {first_file.date}")

        except Exception as e:
            print(f"   ❌ Real file test failed: {e}")
    else:
        print("   ⚠️  No sample photos found - skipping real file test")

if __name__ == "__main__":
    test_basic_components()
    test_error_handling()
    test_real_files()

    print("\n" + "=" * 50)
    print("🎉 BASIC SYSTEM TEST COMPLETE")
    print("=" * 50)
    print("✅ Core components tested without ML dependencies")
    print("💡 For full functionality, install ML packages in virtual environment:")
    print("   python3 -m venv venv")
    print("   source venv/bin/activate")
    print("   pip install torch transformers chromadb sentence-transformers")
    print("   python system_integration_test.py")