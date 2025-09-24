#!/usr/bin/env python3
"""
Test to catch the bug where event names are not being cached.

This test specifically reproduces the issue where the pipeline processes photos
but no event names are generated or cached, resulting in 0% naming success rate.
"""

import sys
import os
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime

# Add src to path and change to project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
os.chdir(project_root)

# Import modules
try:
    from src.photo_organizer_pipeline import PhotoOrganizerPipeline
    from src.config_manager import get_config
    from src.event_namer import EventNamer
except ImportError:
    # Fallback for when running as script
    import photo_organizer_pipeline
    import config_manager
    import event_namer
    PhotoOrganizerPipeline = photo_organizer_pipeline.PhotoOrganizerPipeline
    get_config = config_manager.get_config
    EventNamer = event_namer.EventNamer

class TestEventNamingBug:
    def __init__(self):
        """Initialize test with isolated environment."""
        # Create temporary directories for test isolation
        self.test_dir = Path(tempfile.mkdtemp(prefix="event_naming_test_"))
        self.test_cache_file = self.test_dir / "test_event_naming_cache.json"

        print(f"🧪 TEST: Using test directory: {self.test_dir}")

        # Ensure test cache starts empty
        with open(self.test_cache_file, 'w') as f:
            json.dump({}, f)

    def cleanup(self):
        """Clean up test files."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            print(f"🧹 TEST: Cleaned up test directory")

    def test_event_namer_directly(self):
        """Test EventNamer class directly to see if it works in isolation."""
        print(f"\n🔬 TEST: Testing EventNamer directly...")

        # Create EventNamer with test cache
        namer = event_namer.EventNamer(cache_file=self.test_cache_file, enable_llm=False)

        # Create mock cluster data
        cluster_data = {
            'files': [{'path': 'test1.jpg'}, {'path': 'test2.jpg'}],
            'start_time': datetime(2024, 10, 25, 14, 30),
            'end_time': datetime(2024, 10, 25, 16, 30),
            'location_info': {
                'city': 'Edmonton',
                'country': 'Canada',
                'coordinates': (53.5461, -113.4938)
            },
            'content_analysis': {
                'activities': ['party', 'celebration'],
                'objects': ['cake', 'people'],
                'scenes': ['indoor']
            },
            'confidence_score': 0.8
        }

        print(f"🔬 TEST: Calling generate_event_name with mock data...")
        event_name = namer.generate_event_name(cluster_data)
        print(f"🔬 TEST: Generated event name: {event_name}")

        # Check if name was cached
        with open(self.test_cache_file, 'r') as f:
            cache_data = json.load(f)

        print(f"🔬 TEST: Cache after direct test: {len(cache_data)} entries")
        if cache_data:
            print(f"🔬 TEST: Sample cache entry: {list(cache_data.items())[0]}")

        return len(cache_data) > 0, event_name

    def test_pipeline_naming(self):
        """Test event naming through the full pipeline."""
        print(f"\n🏭 TEST: Testing event naming through pipeline...")

        # Find some actual photos to test with
        sample_photos_dir = Path("Sample_Photos/iPhone Automatic")
        if not sample_photos_dir.exists():
            print(f"❌ TEST: Sample photos directory not found: {sample_photos_dir}")
            return False, "No sample photos found"

        # Get a few photo files
        photo_files = list(sample_photos_dir.glob("*.jpg"))[:3]
        if len(photo_files) < 2:
            print(f"❌ TEST: Not enough photos found in {sample_photos_dir}")
            return False, "Insufficient photos"

        print(f"🏭 TEST: Found {len(photo_files)} photos to test with")

        # Create config with test cache file
        config = config_manager.ConfigManager()
        config.naming.use_llm_naming = False  # Disable LLM for reliable testing

        # Initialize pipeline with modified cache location
        pipeline = photo_organizer_pipeline.PhotoOrganizerPipeline(config=config)

        # Temporarily replace the event namer's cache file
        original_cache_file = pipeline.clustering_engine.event_namer.cache_file
        pipeline.clustering_engine.event_namer.cache_file = self.test_cache_file
        pipeline.clustering_engine.event_namer.naming_cache = {}  # Clear in-memory cache

        try:
            # Process the photos
            print(f"🏭 TEST: Processing {len(photo_files)} photos through pipeline...")

            # Convert photo paths to the format expected by pipeline
            photo_paths = [str(p) for p in photo_files]

            # Run the pipeline in dry-run mode
            result = pipeline.run_pipeline(
                input_directory=str(sample_photos_dir.parent),
                max_photos=len(photo_files),
                dry_run=True
            )

            print(f"🏭 TEST: Pipeline completed with result: {result}")

            # Check if names were cached
            with open(self.test_cache_file, 'r') as f:
                cache_data = json.load(f)

            print(f"🏭 TEST: Cache after pipeline: {len(cache_data)} entries")

            # Check naming statistics
            naming_stats = result.get('naming_results', {})
            success_rate = naming_stats.get('success_rate', 0)
            print(f"🏭 TEST: Naming success rate: {success_rate}%")

            return len(cache_data) > 0, f"Success rate: {success_rate}%, Cache entries: {len(cache_data)}"

        finally:
            # Restore original cache file
            pipeline.clustering_engine.event_namer.cache_file = original_cache_file

    def test_debug_pipeline_stages(self):
        """Test each stage of the pipeline to find where naming fails."""
        print(f"\n🐛 TEST: Debugging pipeline stages...")

        # Find some photos
        sample_photos_dir = Path("Sample_Photos/iPhone Automatic")
        if not sample_photos_dir.exists():
            return False, "No sample photos"

        photo_files = list(sample_photos_dir.glob("*.jpg"))[:2]
        if len(photo_files) < 2:
            return False, "Insufficient photos"

        config = config_manager.ConfigManager()
        config.naming.use_llm_naming = False

        pipeline = photo_organizer_pipeline.PhotoOrganizerPipeline(config=config)

        # Test Stage 1: File scanning
        print(f"🐛 TEST: Stage 1 - Scanning files...")
        scan_result = pipeline._stage_scan_and_validate(str(sample_photos_dir.parent), max_photos=2)
        print(f"🐛 TEST: Scan result: {len(scan_result.get('files', []))} files found")

        if not scan_result.get('files'):
            return False, "No files found in scan stage"

        # Test Stage 2: Clustering
        print(f"🐛 TEST: Stage 2 - Clustering...")
        cluster_result = pipeline._stage_cluster_media(scan_result['files'])
        clusters = cluster_result.get('clusters', [])
        print(f"🐛 TEST: Clustering result: {len(clusters)} clusters created")

        if not clusters:
            return False, "No clusters created"

        # Test Stage 3: Event naming
        print(f"🐛 TEST: Stage 3 - Event naming...")
        naming_result = pipeline._stage_generate_event_names(clusters)
        named_clusters = naming_result.get('clusters', [])
        print(f"🐛 TEST: Naming result: {len(named_clusters)} named clusters")

        # Check if any clusters have names
        clusters_with_names = [c for c in named_clusters if hasattr(c, 'suggested_name') and c.suggested_name]
        print(f"🐛 TEST: Clusters with suggested names: {len(clusters_with_names)}")

        if clusters_with_names:
            print(f"🐛 TEST: Sample suggested name: {clusters_with_names[0].suggested_name}")

        # Check cache
        with open(self.test_cache_file, 'r') as f:
            cache_data = json.load(f)

        print(f"🐛 TEST: Final cache entries: {len(cache_data)}")

        return len(cache_data) > 0, f"Named clusters: {len(clusters_with_names)}, Cache: {len(cache_data)}"

def main():
    """Run the event naming bug test."""
    print("🚨 EVENT NAMING BUG TEST")
    print("=" * 50)

    test = TestEventNamingBug()

    try:
        # Test 1: Direct EventNamer test
        direct_success, direct_result = test.test_event_namer_directly()
        print(f"\n✅ Direct EventNamer test: {'PASS' if direct_success else 'FAIL'} - {direct_result}")

        # Test 2: Pipeline naming test
        pipeline_success, pipeline_result = test.test_pipeline_naming()
        print(f"✅ Pipeline naming test: {'PASS' if pipeline_success else 'FAIL'} - {pipeline_result}")

        # Test 3: Debug pipeline stages
        debug_success, debug_result = test.test_debug_pipeline_stages()
        print(f"✅ Debug pipeline test: {'PASS' if debug_success else 'FAIL'} - {debug_result}")

        # Overall result
        print(f"\n🎯 OVERALL TEST RESULT:")
        if direct_success and pipeline_success:
            print(f"✅ ALL TESTS PASSED - Event naming is working correctly")
        elif direct_success and not pipeline_success:
            print(f"❌ BUG DETECTED - EventNamer works directly but fails in pipeline")
        elif not direct_success:
            print(f"❌ CRITICAL BUG - EventNamer fails even in isolation")
        else:
            print(f"⚠️ PARTIAL FAILURE - Mixed results")

    except Exception as e:
        print(f"💥 TEST EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

    finally:
        test.cleanup()

if __name__ == "__main__":
    main()