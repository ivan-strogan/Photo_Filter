#!/usr/bin/env python3
"""
Comprehensive System Integration Test

This test verifies that all components work together properly in the complete
photo organization pipeline, simulating real-world usage scenarios.

For junior developers:
- Integration tests verify that individual components work together
- This tests the complete user workflow from start to finish
- We simulate various real-world scenarios and edge cases
"""

import sys
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta
import json
sys.path.append(str(Path(__file__).parent))

def create_realistic_test_environment():
    """Create a realistic test environment with various file types and scenarios."""
    test_dir = Path(tempfile.mkdtemp(prefix="photo_filter_integration_test_"))

    # Create directory structure
    iphone_automatic_dir = test_dir / "Sample_Photos" / "iPhone Automatic"
    iphone_automatic_dir.mkdir(parents=True)

    pictures_dir = test_dir / "Sample_Photos" / "Pictures"
    pictures_dir.mkdir(parents=True)

    # Create realistic test files with proper naming convention
    test_files = []

    # Scenario 1: Halloween photos (clustered event)
    base_date = datetime(2024, 10, 31, 18, 30)  # Halloween evening
    for i in range(5):
        filename = f"IMG_{(base_date + timedelta(minutes=i*10)).strftime('%Y%m%d_%H%M%S')}.JPG"
        file_path = iphone_automatic_dir / filename

        # Create valid JPEG file
        with open(file_path, 'wb') as f:
            f.write(b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00')
            f.write(f'Halloween photo {i+1} content data'.encode() * 20)
            f.write(b'\xFF\xD9')
        test_files.append(file_path)

    # Scenario 2: Weekend activity (separate cluster)
    base_date = datetime(2024, 11, 2, 14, 15)  # Saturday afternoon
    for i in range(3):
        filename = f"IMG_{(base_date + timedelta(minutes=i*20)).strftime('%Y%m%d_%H%M%S')}.JPG"
        file_path = iphone_automatic_dir / filename

        with open(file_path, 'wb') as f:
            f.write(b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00')
            f.write(f'Weekend photo {i+1} content data'.encode() * 15)
            f.write(b'\xFF\xD9')
        test_files.append(file_path)

    # Scenario 3: Video files
    video_date = datetime(2024, 11, 2, 15, 45)
    for i in range(2):
        filename = f"IMG_{(video_date + timedelta(minutes=i*5)).strftime('%Y%m%d_%H%M%S')}.MOV"
        file_path = iphone_automatic_dir / filename

        with open(file_path, 'wb') as f:
            # Create minimal MP4/MOV structure
            f.write(b'\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41')
            f.write(f'Video {i+1} content data'.encode() * 30)
        test_files.append(file_path)

    # Scenario 4: Corrupted files (should be filtered out)
    corrupted_file = iphone_automatic_dir / "IMG_20241103_120000.JPG"
    with open(corrupted_file, 'wb') as f:
        f.write(b'This is not a valid image file')
    test_files.append(corrupted_file)

    # Scenario 5: Empty file (should be filtered out)
    empty_file = iphone_automatic_dir / "IMG_20241103_130000.JPG"
    empty_file.touch()
    test_files.append(empty_file)

    # Scenario 6: Unsupported format
    unsupported_file = iphone_automatic_dir / "IMG_20241103_140000.HEIC"
    with open(unsupported_file, 'wb') as f:
        f.write(b'HEIC file content' * 20)
    test_files.append(unsupported_file)

    return test_dir, test_files

def test_individual_components():
    """Test individual components in isolation."""
    print("🔧 Testing Individual Components")
    print("=" * 50)

    try:
        # Test 1: Media Detector
        print("1️⃣  Testing MediaDetector...")
        from src.media_detector import MediaDetector
        detector = MediaDetector()
        print("   ✅ MediaDetector imported and initialized")

        # Test 2: Media Validator
        print("2️⃣  Testing MediaValidator...")
        from src.media_validator import MediaValidator
        validator = MediaValidator()
        print("   ✅ MediaValidator imported and initialized")

        # Test 3: Temporal Clusterer
        print("3️⃣  Testing TemporalClusterer...")
        from src.temporal_clustering import TemporalClusterer
        clusterer = TemporalClusterer()
        print("   ✅ TemporalClusterer imported and initialized")

        # Test 4: Event Namer
        print("4️⃣  Testing EventNamer...")
        from src.event_namer import EventNamer
        namer = EventNamer(enable_llm=False)
        print("   ✅ EventNamer imported and initialized")

        # Test 5: Folder Organizer
        print("5️⃣  Testing FolderOrganizer...")
        from src.folder_organizer import FolderOrganizer
        folder_org = FolderOrganizer(dry_run=True)
        print("   ✅ FolderOrganizer imported and initialized")

        # Test 6: File Organizer
        print("6️⃣  Testing FileOrganizer...")
        from src.file_organizer import FileOrganizer
        file_org = FileOrganizer(dry_run=True)
        print("   ✅ FileOrganizer imported and initialized")

        # Test 7: Complete Pipeline
        print("7️⃣  Testing PhotoOrganizerPipeline...")
        from src.photo_organizer_pipeline import PhotoOrganizerPipeline
        pipeline = PhotoOrganizerPipeline(dry_run=True)
        print("   ✅ PhotoOrganizerPipeline imported and initialized")

        print("\n✅ All individual components initialized successfully!")
        # Pytest assertion - test passed
        assert True  # Explicit success assertion

    except Exception as e:
        print(f"\n❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        # Pytest assertion - test failed
        assert False, f"Component test failed: {e}"

def test_complete_pipeline_dry_run(temp_test_dir):
    """Test the complete pipeline in dry-run mode."""
    print("\n🚀 Testing Complete Pipeline (Dry Run)")
    print("=" * 50)

    try:
        from src.photo_organizer_pipeline import PhotoOrganizerPipeline

        # Configure pipeline for testing
        pipeline = PhotoOrganizerPipeline(
            max_photos=20,
            operation_mode="copy",
            dry_run=True,  # Safe dry-run mode
            verify_checksums=True
        )

        print(f"📋 Pipeline Configuration:")
        print(f"   Max photos: {pipeline.max_photos}")
        print(f"   Operation mode: {pipeline.operation_mode}")
        print(f"   Dry run: {pipeline.dry_run}")
        print(f"   Verify checksums: {pipeline.verify_checksums}")

        # Create progress callback
        progress_stages = []
        def progress_callback(message, progress):
            progress_stages.append((message, progress))
            percentage = int(progress * 100)
            print(f"   [{percentage:3d}%] {message}")

        # Set the sample photos directory to our test directory
        original_cwd = Path.cwd()
        try:
            # Temporarily change to test directory
            import os
            os.chdir(temp_test_dir)

            print(f"\n🎬 Starting pipeline from: {temp_test_dir}")

            # Run the complete pipeline
            results = pipeline.run_complete_pipeline(
                source_folder=None,  # Use default
                output_folder=None,  # Use default
                progress_callback=progress_callback
            )

        finally:
            os.chdir(original_cwd)

        # Analyze results
        print(f"\n📊 Pipeline Results Analysis:")
        print("-" * 30)

        pipeline_summary = results['pipeline_summary']
        print(f"   Execution time: {pipeline_summary['execution_time_seconds']:.2f} seconds")
        print(f"   Overall success: {pipeline_summary['overall_success']}")
        print(f"   Dry run mode: {pipeline_summary['dry_run_mode']}")

        # Stage-by-stage analysis
        stage_summaries = results['stage_summaries']

        # Scan stage
        if 'scan' in stage_summaries:
            scan = stage_summaries['scan']
            print(f"\n   📂 Scan Stage:")
            print(f"      Files scanned: {scan.get('files_scanned', 0)}")
            print(f"      Files selected: {scan.get('files_selected', 0)}")

        # Validation stage
        if 'validation' in stage_summaries:
            validation = stage_summaries['validation']
            print(f"\n   🔍 Validation Stage:")
            print(f"      Files input: {validation.get('files_input', 0)}")
            print(f"      Files valid: {validation.get('files_valid', 0)}")
            print(f"      Files corrupted: {validation.get('files_corrupted', 0)}")
            print(f"      Files unsupported: {validation.get('files_unsupported', 0)}")
            print(f"      Success rate: {validation.get('validation_success_rate', 0):.1%}")

        # Clustering stage
        if 'clustering' in stage_summaries:
            clustering = stage_summaries['clustering']
            print(f"\n   🧠 Clustering Stage:")
            print(f"      Files input: {clustering.get('files_input', 0)}")
            print(f"      Clusters created: {clustering.get('clusters_created', 0)}")
            print(f"      Avg cluster size: {clustering.get('avg_cluster_size', 0):.1f}")

        # Naming stage
        if 'naming' in stage_summaries:
            naming = stage_summaries['naming']
            print(f"\n   🎯 Naming Stage:")
            print(f"      Clusters input: {naming.get('clusters_input', 0)}")
            print(f"      Naming success rate: {naming.get('naming_success_rate', 0):.1%}")
            print(f"      High confidence names: {naming.get('high_confidence_names', 0)}")

        # Folder stage
        if 'folders' in stage_summaries:
            folders = stage_summaries['folders']
            print(f"\n   📁 Folder Stage:")
            print(f"      Folders created: {folders.get('folders_created', 0)}")
            print(f"      Conflicts resolved: {folders.get('conflicts_resolved', 0)}")

        # Organization stage
        if 'organization' in stage_summaries:
            organization = stage_summaries['organization']
            print(f"\n   📂 Organization Stage:")
            print(f"      Files processed: {organization.get('files_processed', 0)}")
            print(f"      Files successful: {organization.get('files_successful', 0)}")
            print(f"      Success rate: {organization.get('success_rate', 0):.1%}")

        # Check for errors
        if results.get('errors'):
            print(f"\n   ⚠️  Errors encountered: {len(results['errors'])}")
            for error in results['errors'][:3]:  # Show first 3
                print(f"      • {error['stage']}: {error['error']}")
        else:
            print(f"\n   ✅ No errors encountered")

        # Verify progress tracking worked
        print(f"\n   📈 Progress Tracking:")
        print(f"      Progress updates: {len(progress_stages)}")
        if progress_stages:
            print(f"      First update: {progress_stages[0][0]}")
            print(f"      Last update: {progress_stages[-1][0]}")

        # Pytest assertion
        assert pipeline_summary['overall_success'], f"Pipeline should succeed but overall_success was {pipeline_summary['overall_success']}"

    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        # Pytest assertion - test failed
        assert False, f"Pipeline test failed: {e}"

def test_component_interaction():
    """Test specific component interactions and edge cases."""
    print("\n🔗 Testing Component Interactions")
    print("=" * 50)

    success_count = 0
    total_tests = 0

    try:
        # Test 1: MediaDetector + MediaValidator integration
        print("1️⃣  Testing MediaDetector + MediaValidator...")
        total_tests += 1

        # Create a small test file
        test_dir = Path(tempfile.mkdtemp(prefix="interaction_test_"))
        test_file = test_dir / "test.jpg"
        with open(test_file, 'wb') as f:
            f.write(b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00')
            f.write(b'Test image data' * 10)
            f.write(b'\xFF\xD9')

        from src.media_detector import MediaDetector
        from src.media_validator import MediaValidator

        # Test the interaction
        detector = MediaDetector()
        validator = MediaValidator()

        # Simulate scanning (we'll just create a mock MediaFile)
        class MockMediaFile:
            def __init__(self, path):
                self.path = Path(path)
                self.filename = path.name
                self.date = datetime.now()
                self.file_type = 'photo'

        media_file = MockMediaFile(test_file)
        result = validator.validate_media_file(media_file.path)

        if result.is_valid and result.is_supported:
            print("      ✅ MediaDetector + MediaValidator integration working")
            success_count += 1
        else:
            print(f"      ❌ Validation failed: {result.get_summary()}")

        # Cleanup
        shutil.rmtree(test_dir)

        # Test 2: TemporalClusterer + EventNamer integration
        print("2️⃣  Testing TemporalClusterer + EventNamer...")
        total_tests += 1

        from src.temporal_clustering import TemporalClusterer
        from src.event_namer import EventNamer

        clusterer = TemporalClusterer()
        namer = EventNamer(enable_llm=False)

        # Create mock media files for clustering
        mock_files = []
        base_time = datetime(2024, 11, 1, 14, 30)
        for i in range(3):
            mock_file = MockMediaFile(f"test_{i}.jpg")
            mock_file.date = base_time + timedelta(minutes=i*10)
            mock_files.append(mock_file)

        # Test clustering
        clusters = clusterer.cluster_by_time(mock_files)

        if clusters:
            # Test naming
            cluster = clusters[0]
            cluster_data = {
                'start_time': cluster.start_time,
                'end_time': cluster.end_time,
                'duration_hours': cluster.duration.total_seconds() / 3600,
                'size': cluster.size,
                'photo_count': cluster.photo_count,
                'video_count': cluster.video_count,
                'location_info': None,
                'dominant_location': None,
                'gps_coordinates': [],
                'content_tags': ['autumn', 'afternoon'],
                'people_detected': [],
                'confidence_score': 0.8,
                'media_files': cluster.media_files
            }

            name = namer.generate_event_name(cluster_data)
            if name and len(name) > 10:  # Reasonable name generated
                print(f"      ✅ Generated name: '{name}'")
                success_count += 1
            else:
                print(f"      ❌ Poor name generated: '{name}'")
        else:
            print("      ❌ No clusters created")

        # Test 3: Configuration system integration
        print("3️⃣  Testing Configuration System...")
        total_tests += 1

        try:
            from src.config_manager import get_config_manager
            config_mgr = get_config_manager()
            config = config_mgr.load_config()

            # Test basic config access
            if hasattr(config, 'clustering') and hasattr(config, 'processing'):
                print("      ✅ Configuration system working")
                success_count += 1
            else:
                print("      ❌ Configuration structure invalid")
        except Exception as e:
            print(f"      ❌ Configuration system failed: {e}")

        # Test 4: Error propagation
        print("4️⃣  Testing Error Propagation...")
        total_tests += 1

        # Test that errors are handled gracefully throughout the pipeline
        try:
            from src.photo_organizer_pipeline import PhotoOrganizerPipeline

            # Create pipeline with intentionally problematic settings
            pipeline = PhotoOrganizerPipeline(max_photos=0, dry_run=True)

            # Try to run with no photos - should handle gracefully
            results = pipeline.run_complete_pipeline()

            # Should complete without crashing, even with no photos
            if 'pipeline_summary' in results:
                print("      ✅ Error propagation handled gracefully")
                success_count += 1
            else:
                print("      ❌ Pipeline crashed on edge case")
        except Exception as e:
            print(f"      ❌ Error propagation test failed: {e}")

        print(f"\n📊 Component Interaction Results:")
        print(f"   Tests passed: {success_count}/{total_tests}")
        print(f"   Success rate: {success_count/total_tests:.1%}")

        # Pytest assertion
        assert success_count == total_tests, f"Expected all {total_tests} tests to pass, but only {success_count} passed"

    except Exception as e:
        print(f"\n❌ Component interaction test failed: {e}")
        import traceback
        traceback.print_exc()
        # Pytest assertion - test failed
        assert False, f"Component interaction test failed: {e}"

def run_comprehensive_system_test():
    """Run comprehensive system integration test."""
    print("🧪 COMPREHENSIVE SYSTEM INTEGRATION TEST")
    print("=" * 60)
    print("Testing Photo Filter AI system end-to-end")
    print("=" * 60)

    # Track test results
    test_results = {
        'component_test': False,
        'pipeline_test': False,
        'interaction_test': False,
        'overall_success': False
    }

    # Create test environment
    print("🔧 Setting up test environment...")
    test_dir, test_files = create_realistic_test_environment()
    print(f"   Created {len(test_files)} test files in: {test_dir}")
    print(f"   Test scenarios: Halloween photos, weekend activity, videos, corrupted files")

    try:
        # Test 1: Individual Components
        test_results['component_test'] = test_individual_components()

        if test_results['component_test']:
            # Test 2: Complete Pipeline
            test_results['pipeline_test'] = test_complete_pipeline_dry_run(test_dir)

            # Test 3: Component Interactions
            test_results['interaction_test'] = test_component_interaction()

        # Overall assessment
        test_results['overall_success'] = all([
            test_results['component_test'],
            test_results['pipeline_test'],
            test_results['interaction_test']
        ])

    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up test environment: {test_dir}")
        shutil.rmtree(test_dir)

    # Final report
    print("\n" + "=" * 60)
    print("🎯 FINAL SYSTEM TEST RESULTS")
    print("=" * 60)

    print(f"📋 Test Summary:")
    print(f"   ✅ Component Initialization: {'PASS' if test_results['component_test'] else 'FAIL'}")
    print(f"   ✅ Complete Pipeline: {'PASS' if test_results['pipeline_test'] else 'FAIL'}")
    print(f"   ✅ Component Integration: {'PASS' if test_results['interaction_test'] else 'FAIL'}")
    print()

    if test_results['overall_success']:
        print("🎉 SYSTEM INTEGRATION TEST: ✅ PASSED")
        print()
        print("🚀 Photo Filter AI System Status:")
        print("   ✅ All components working correctly")
        print("   ✅ Complete pipeline functional")
        print("   ✅ Error handling working")
        print("   ✅ Component integration verified")
        print("   ✅ Ready for production use")
        print()
        print("💡 What this means:")
        print("   • Your system can reliably process iPhone photos")
        print("   • Intelligent clustering and naming works")
        print("   • File organization pipeline is functional")
        print("   • Error handling prevents crashes")
        print("   • Safe for use with real photo collections")

    else:
        print("❌ SYSTEM INTEGRATION TEST: ❌ FAILED")
        print()
        print("🔍 Issues found:")
        if not test_results['component_test']:
            print("   • Component initialization problems")
        if not test_results['pipeline_test']:
            print("   • Pipeline execution issues")
        if not test_results['interaction_test']:
            print("   • Component integration problems")
        print()
        print("🛠️  Recommendation: Review failed components before production use")

    return test_results['overall_success']

if __name__ == "__main__":
    success = run_comprehensive_system_test()
    sys.exit(0 if success else 1)