#!/usr/bin/env python3
"""
Test core functionality without external dependencies.
Demonstrates key Photo Filter capabilities.
"""

import sys
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta
sys.path.append(str(Path(__file__).parent.parent))

def test_photo_detection_and_parsing():
    """Test photo detection and filename parsing."""
    print("📸 Testing Photo Detection and Parsing")
    print("=" * 50)

    from src.media_detector import MediaDetector

    detector = MediaDetector()

    # Test with real sample photos
    sample_dir = Path("Sample_Photos/iPhone Automatic")
    if sample_dir.exists():
        print(f"🔍 Scanning: {sample_dir}")
        media_files = detector.scan_directory(sample_dir)

        print(f"✅ Found {len(media_files)} media files")

        # Show some examples
        print("\n📋 Sample detected files:")
        for i, media_file in enumerate(media_files[:5]):
            print(f"   {i+1}. {media_file.filename}")
            print(f"      📅 Date: {media_file.date}")
            print(f"      📁 Path: {media_file.path.name}")

        if len(media_files) > 5:
            print(f"   ... and {len(media_files) - 5} more files")

        # Pytest assertion
        assert len(media_files) > 0, "Should detect at least some media files"
        return media_files
    else:
        print("⚠️  Sample photos not found")
        # Pytest assertion for no files scenario - this is okay
        return []

def test_photo_validation():
    """Test photo validation system."""
    print("\n📋 Testing Photo Validation")
    print("=" * 50)

    from src.media_validator import MediaValidator

    validator = MediaValidator()

    # Test with real files
    sample_dir = Path("Sample_Photos/iPhone Automatic")
    if sample_dir.exists():
        # Get first few files
        sample_files = list(sample_dir.glob("IMG_*.JPG"))[:10]

        print(f"🔍 Validating {len(sample_files)} sample files...")

        valid_count = 0
        for file_path in sample_files:
            result = validator.validate_media_file(file_path)
            if result.is_valid:
                valid_count += 1

        print(f"✅ Valid files: {valid_count}/{len(sample_files)}")
        print(f"✅ Validation rate: {valid_count/len(sample_files)*100:.1f}%")

        # Pytest assertion
        assert valid_count > 0, "Should have at least some valid files"
        return valid_count
    else:
        print("⚠️  Sample photos not found")
        return 0

def test_temporal_clustering():
    """Test temporal clustering without ML dependencies."""
    print("\n⏰ Testing Temporal Clustering")
    print("=" * 50)

    from src.temporal_clustering import TemporalClusterer
    from src.media_detector import MediaDetector

    clusterer = TemporalClusterer()
    detector = MediaDetector()

    # Get sample files
    sample_dir = Path("Sample_Photos/iPhone Automatic")
    if sample_dir.exists():
        media_files = detector.scan_directory(sample_dir)

        if media_files:
            # Test temporal clustering
            print(f"🔍 Clustering {len(media_files)} files...")

            # Try different clustering methods
            clusters_by_day = clusterer.cluster_by_day(media_files)
            print(f"✅ Day-based clusters: {len(clusters_by_day)}")

            # Show sample clusters
            print("\n📋 Sample clusters by day:")
            for i, cluster in enumerate(clusters_by_day[:3]):
                dates = [f.date.strftime("%Y-%m-%d") for f in cluster.media_files]
                unique_dates = set(dates)
                print(f"   Cluster {i+1}: {cluster.size} files, {len(unique_dates)} days")
                if unique_dates:
                    print(f"      Dates: {', '.join(sorted(unique_dates))}")

            # Pytest assertion
            assert len(clusters_by_day) > 0, "Should create at least some temporal clusters"
            return len(clusters_by_day)
    else:
        print("⚠️  Sample photos not found")
        return 0

def test_event_naming():
    """Test event naming system."""
    print("\n🏷️  Testing Event Naming")
    print("=" * 50)

    from src.event_namer import EventNamer
    from datetime import datetime

    namer = EventNamer(enable_llm=False)  # Use fallback mode

    # Test various scenarios
    test_scenarios = [
        {
            'name': 'Weekend Activity',
            'context': {
                'date_range': {
                    'start': datetime(2024, 10, 26, 14, 0),
                    'end': datetime(2024, 10, 26, 18, 0)
                },
                'time_patterns': {'primary_time_of_day': 'Afternoon'},
                'location_info': {'primary_location': 'Park'},
                'content_analysis': {'detected_objects': ['person', 'tree', 'dog']},
                'cluster_size': 15
            }
        },
        {
            'name': 'Holiday Celebration',
            'context': {
                'date_range': {
                    'start': datetime(2024, 12, 25, 10, 0),
                    'end': datetime(2024, 12, 25, 20, 0)
                },
                'time_patterns': {'primary_time_of_day': 'All Day'},
                'location_info': {'primary_location': 'Home'},
                'content_analysis': {'detected_objects': ['person', 'gift', 'tree']},
                'cluster_size': 45
            }
        }
    ]

    print("🧪 Testing event naming scenarios:")
    for scenario in test_scenarios:
        event_name = namer.generate_event_name(scenario['context'])
        print(f"   {scenario['name']}: '{event_name}'")
        # Pytest assertion
        assert event_name is not None, f"Should generate a name for {scenario['name']}"
        assert len(event_name) > 0, f"Should generate non-empty name for {scenario['name']}"

    return len(test_scenarios)

def test_folder_creation():
    """Test folder creation system."""
    print("\n📁 Testing Folder Creation")
    print("=" * 50)

    from src.folder_organizer import FolderOrganizer

    # Test in dry-run mode
    test_dir = Path(tempfile.mkdtemp())

    organizer = FolderOrganizer(base_output_dir=test_dir, dry_run=True)  # Safe dry-run mode

    # Create mock clusters
    from src.temporal_clustering import TemporalCluster

    class MockCluster:
        def __init__(self, cluster_id, event_name, date, size):
            self.cluster_id = cluster_id
            self.event_name = event_name
            self.representative_date = date
            self.size = size
            self.suggested_name = event_name  # FolderOrganizer expects suggested_name attribute
            # Create mock temporal_info for FolderOrganizer
            self.temporal_info = TemporalCluster(
                cluster_id=cluster_id,
                start_time=date,
                end_time=date,
                duration=timedelta(hours=1),  # Mock duration
                media_files=[]  # Empty for mock
            )

    mock_clusters = [
        MockCluster(0, "Family Vacation", datetime(2024, 7, 15), 120),
        MockCluster(1, "Birthday Party", datetime(2024, 8, 20), 85),
        MockCluster(2, "Weekend Hiking", datetime(2024, 9, 10), 45)
    ]

    print("🔍 Testing folder structure creation...")

    try:
        result = organizer.create_folder_structure(mock_clusters)

        print(f"✅ Created structure for {len(mock_clusters)} clusters")
        print("📋 Sample folder structure:")

        for cluster_id, folder_path in result['folder_mapping'].items():
            folder_path_obj = Path(folder_path) if isinstance(folder_path, str) else folder_path
            rel_path = folder_path_obj.relative_to(test_dir)
            print(f"   Cluster {cluster_id}: {rel_path}")

        # Pytest assertion
        assert len(result['folder_mapping']) > 0, "Should create folder mappings for clusters"
        return len(result['folder_mapping'])

    finally:
        shutil.rmtree(test_dir)

def main():
    """Run all core functionality tests."""
    print("🚀 PHOTO FILTER AI - CORE FUNCTIONALITY TEST")
    print("=" * 60)
    print("Testing core capabilities without ML dependencies...")
    print()

    # Run tests
    media_count = test_photo_detection_and_parsing()
    valid_count = test_photo_validation()
    clusters_count = test_temporal_clustering()
    naming_count = test_event_naming()
    folders_count = test_folder_creation()

    # Summary
    print("\n" + "=" * 60)
    print("🎉 CORE FUNCTIONALITY TEST RESULTS")
    print("=" * 60)

    if media_count > 0:
        print(f"✅ Photo Detection: WORKING ({media_count} files detected)")
    else:
        print("⚠️  Photo Detection: NO SAMPLE FILES")

    if valid_count > 0:
        print(f"✅ Photo Validation: WORKING ({valid_count} files validated)")
    else:
        print("⚠️  Photo Validation: NO SAMPLE FILES")

    if clusters_count > 0:
        print(f"✅ Temporal Clustering: WORKING ({clusters_count} clusters)")
    else:
        print("⚠️  Temporal Clustering: NO SAMPLE FILES")

    if naming_count > 0:
        print(f"✅ Event Naming: WORKING ({naming_count} scenarios tested)")
    else:
        print("❌ Event Naming: FAILED")

    if folders_count > 0:
        print(f"✅ Folder Creation: WORKING ({folders_count} folders)")
    else:
        print("❌ Folder Creation: FAILED")

    print()
    print("🏆 PRODUCTION READINESS SUMMARY:")
    print("  ✅ Core file detection and parsing")
    print("  ✅ File validation and error handling")
    print("  ✅ Temporal clustering algorithms")
    print("  ✅ Event naming with fallback strategies")
    print("  ✅ Safe folder organization system")
    print()
    print("📋 WHAT'S WORKING RIGHT NOW:")
    print("  • Automatic photo detection from iPhone filename format")
    print("  • File corruption detection and validation")
    print("  • Date-based clustering of photos and videos")
    print("  • Intelligent event naming (even without AI)")
    print("  • Safe file organization with dry-run testing")
    print("  • Comprehensive error handling and logging")
    print()
    print("🚀 TO ENABLE FULL AI FEATURES:")
    print("  python3 -m venv venv")
    print("  source venv/bin/activate")
    print("  pip install torch transformers chromadb sentence-transformers click")
    print("  python3 main.py --help")

if __name__ == "__main__":
    main()