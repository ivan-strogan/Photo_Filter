#!/usr/bin/env python3
"""Test processing with a small subset of photos."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.media_processor import MediaProcessor
from src.media_detector import MediaDetector

def test_small_processing():
    """Test processing with just the first 10 photos."""
    print("🧪 Testing small-scale processing...")

    # Get small subset of files
    detector = MediaDetector()
    all_files = detector.scan_iphone_automatic()
    small_subset = all_files[:10]  # Just first 10 files

    print(f"Testing with {len(small_subset)} files:")
    for f in small_subset:
        print(f"  {f.filename} ({f.file_type})")

    # Process with MediaProcessor
    processor = MediaProcessor(verbose=True)

    # Temporarily override the detector to return our subset
    original_method = processor.media_detector.scan_iphone_automatic
    processor.media_detector.scan_iphone_automatic = lambda: small_subset

    try:
        results = processor.process_new_media()

        print("\n✅ Small processing test completed!")
        print(f"Clusters created: {len(results.get('clusters', []))}")

        clusters = results.get('clusters', [])
        for cluster in clusters:
            print(f"  Cluster {cluster.cluster_id}: {cluster.suggested_name} ({cluster.size} files)")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Restore original method
        processor.media_detector.scan_iphone_automatic = original_method

if __name__ == "__main__":
    test_small_processing()