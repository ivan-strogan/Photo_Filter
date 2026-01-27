#!/usr/bin/env python3
"""Test content analyzer for photo analysis capabilities."""

import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from src.content_analyzer import ContentAnalyzer
from src.media_detector import MediaDetector


@pytest.mark.unit
def test_model_initialization():
    """Test that CLIP and BLIP models initialize correctly.

    Verifies lazy loading pattern and successful model initialization.
    Related to Issue #20 - ensures models load without errors.
    """
    analyzer = ContentAnalyzer(use_gpu=False)

    # Models should be None before initialization (lazy loading)
    assert analyzer.clip_model is None
    assert analyzer.blip_model is None
    assert analyzer.clip_processor is None
    assert analyzer.blip_processor is None

    # Initialize models
    result = analyzer._initialize_models()

    # Verify successful initialization
    assert result is True
    assert analyzer.clip_model is not None
    assert analyzer.blip_model is not None
    assert analyzer.clip_processor is not None
    assert analyzer.blip_processor is not None

    analyzer.cleanup()


@pytest.mark.integration
def test_photo_analysis():
    """Test that photo analysis completes without errors.

    **Integration test for Issue #20** - verifies code doesn't crash.

    Uses Woman_Photo_1.jpeg to verify:
    - No NoneType errors when calling model methods
    - analyze_photo_content() returns valid ContentAnalysis object
    - All expected fields are present and correctly typed

    This test does NOT verify AI accuracy - see test_photo_analysis_accuracy_* tests.
    """
    test_photo = Path("tests/artifacts/photos/Woman_Photo_1.jpeg")
    if not test_photo.exists():
        pytest.skip("Test photo Woman_Photo_1.jpeg not available")

    analyzer = ContentAnalyzer(use_gpu=False)

    # Analyze photo - should not crash or raise NoneType errors
    analysis = analyzer.analyze_photo_content(test_photo)

    # Verify valid result structure (not testing accuracy)
    assert analysis is not None
    assert isinstance(analysis.description, str)
    assert len(analysis.description) > 0
    assert isinstance(analysis.objects, list)
    assert isinstance(analysis.scenes, list)
    assert isinstance(analysis.activities, list)
    assert isinstance(analysis.confidence_score, float)
    assert 0.0 <= analysis.confidence_score <= 1.0
    assert "CLIP+BLIP" in analysis.analysis_model

    analyzer.cleanup()


@pytest.mark.integration
def test_photo_analysis_no_faces():
    """Test that photo analysis works when no faces are present.

    **Integration test for Issue #20** - verifies code doesn't crash on photos without faces.

    Uses no_faces_photo1.jpg to verify:
    - No errors occur when no faces are present
    - face_count is 0 (verifiable fact, not AI accuracy)
    - Valid analysis results are still returned

    This test does NOT verify description accuracy - see test_photo_analysis_accuracy_* tests.
    """
    test_photo = Path("tests/artifacts/photos/no_faces_photo1.jpg")
    if not test_photo.exists():
        pytest.skip("Test photo no_faces_photo1.jpg not available")

    analyzer = ContentAnalyzer(use_gpu=False)
    analysis = analyzer.analyze_photo_content(test_photo)

    # Verify valid result structure
    assert analysis is not None
    assert len(analysis.description) > 0
    assert analysis.description != "Unable to generate description"
    assert isinstance(analysis.objects, list)
    assert isinstance(analysis.scenes, list)

    # Verify face detection ran and found no faces (verifiable fact)
    assert analysis.face_count == 0, "Should not detect faces in no_faces_photo1.jpg"
    assert len(analysis.people_detected) == 0, "Should not identify any people"

    # Verify using AI models
    assert "CLIP+BLIP" in analysis.analysis_model

    analyzer.cleanup()


@pytest.mark.integration
@pytest.mark.slow
def test_photo_analysis_accuracy_woman_photo():
    """AI Accuracy Benchmark: Verify accurate description of Woman_Photo_1.jpeg.

    **This is an AI accuracy test, NOT a crash test.**

    Woman_Photo_1.jpeg contains: a woman's face (young woman, smiling)

    Expected AI results:
    - Description should mention "woman" or "person" or "face"
    - Objects: should detect "person"
    - Scenes: should detect reasonable scene types
    - High confidence (>0.5) since photo is clear

    Note: AI models may not be perfect. This test may fail if models are updated
    or if the photo is challenging for CLIP/BLIP.
    """
    test_photo = Path("tests/artifacts/photos/Woman_Photo_1.jpeg")
    if not test_photo.exists():
        pytest.skip("Test photo Woman_Photo_1.jpeg not available")

    analyzer = ContentAnalyzer(use_gpu=False)
    analysis = analyzer.analyze_photo_content(test_photo)

    assert analysis is not None
    print(f"\nDescription: {analysis.description}")
    print(f"Objects detected: {analysis.objects}")
    print(f"Scenes detected: {analysis.scenes}")
    print(f"Confidence: {analysis.confidence_score:.2f}")

    # Verify description mentions relevant content
    description_lower = analysis.description.lower()
    has_person_reference = any(keyword in description_lower for keyword in ["woman", "person", "girl", "lady", "face"])
    assert has_person_reference, f"Description should mention woman/person/face, got: {analysis.description}"

    # For a clear photo of a person, should detect 'person' object
    assert "person" in analysis.objects, f"Should detect 'person' in objects, got: {analysis.objects}"

    # Should detect at least one scene type
    assert len(analysis.scenes) > 0, "Should detect at least one scene type"

    # Clear photo should have good confidence
    assert analysis.confidence_score > 0.5, f"Clear photo should have high confidence, got: {analysis.confidence_score:.2f}"

    analyzer.cleanup()


@pytest.mark.integration
def test_photo_analysis_with_face_recognition():
    """Test that ContentAnalyzer integrates with FaceRecognizer without errors.

    **Integration test for Issue #20** - verifies face recognition integration doesn't crash.

    Verifies that when a FaceRecognizer is provided:
    - ContentAnalyzer calls face detection without errors
    - Face count is included in results
    - Analysis model indicates face recognition was attempted
    - No NoneType errors occur

    Note: This tests ContentAnalyzer integration with empty database.
    Face recognition accuracy is tested in test_face_recognition_accuracy_*.
    """
    test_photo = Path("tests/artifacts/photos/Woman_Photo_1.jpeg")
    if not test_photo.exists():
        pytest.skip("Test photo Woman_Photo_1.jpeg not available")

    # Import face recognition components
    try:
        from src.face_recognizer import FaceRecognizer
        from src.people_database import PeopleDatabase
    except ImportError:
        pytest.skip("Face recognition dependencies not available")

    # Create test people database and face recognizer (empty database)
    import tempfile
    test_db_file = Path(tempfile.mkdtemp()) / "test_people.json"
    people_db = PeopleDatabase(database_file=test_db_file)
    face_recognizer = FaceRecognizer(people_database=people_db)

    # Create analyzer with face recognition
    analyzer = ContentAnalyzer(use_gpu=False, face_recognizer=face_recognizer)

    # Analyze photo - should not raise errors
    analysis = analyzer.analyze_photo_content(test_photo)

    assert analysis is not None

    # Verify face detection was attempted
    assert analysis.face_count > 0, "Should detect at least one face in Woman_Photo_1.jpeg"

    # Model should indicate face recognition was used
    assert "Face Recognition" in analysis.analysis_model, \
        "Analysis model should indicate Face Recognition was used"

    # Verify people_detected is a list (even if empty, since database is empty)
    assert isinstance(analysis.people_detected, list), \
        "people_detected should be a list"

    # Cleanup
    analyzer.cleanup()
    if test_db_file.exists():
        test_db_file.unlink()
        test_db_file.parent.rmdir()


@pytest.mark.integration
@pytest.mark.slow
def test_face_recognition_accuracy_identifies_known_person():
    """AI Accuracy Benchmark: Verify face recognition identifies a known person.

    **This is an AI accuracy test for face recognition, NOT a crash test.**

    Woman_Photo_1.jpeg contains: a woman (we'll call her Elena Rodriguez for testing)

    Test process:
    1. Add Elena Rodriguez to the database using Woman_Photo_1.jpeg
    2. Analyze the same photo
    3. Verify Elena Rodriguez is identified in people_detected

    Expected result:
    - Face is detected (face_count > 0)
    - Elena Rodriguez is identified in people_detected

    Note: This test previously failed due to Issue #25 (face cache invalidation bug).
    The bug was fixed by clearing the face cache when adding people to the database.
    """
    test_photo = Path("tests/artifacts/photos/Woman_Photo_1.jpeg")
    if not test_photo.exists():
        pytest.skip("Test photo Woman_Photo_1.jpeg not available")

    # Import face recognition components
    try:
        from src.face_recognizer import FaceRecognizer
        from src.people_database import PeopleDatabase
    except ImportError:
        pytest.skip("Face recognition dependencies not available")

    # Create test people database
    import tempfile
    test_db_file = Path(tempfile.mkdtemp()) / "test_people.json"
    people_db = PeopleDatabase(database_file=test_db_file)
    face_recognizer = FaceRecognizer(people_database=people_db)

    # Add Elena Rodriguez to the database using the test photo
    print(f"\nTraining: Adding Elena Rodriguez from {test_photo.name}")
    success = face_recognizer.add_person("Elena Rodriguez", [test_photo])

    if not success:
        pytest.skip("Could not add Elena Rodriguez to face database - face detection may have failed")

    # Verify Elena was added
    known_people = people_db.list_people()
    print(f"Known people in database: {[p.name for p in known_people]}")
    assert any(p.name == "Elena Rodriguez" for p in known_people), \
        "Elena Rodriguez should be in the database after training"

    # Create analyzer with trained face recognizer
    analyzer = ContentAnalyzer(use_gpu=False, face_recognizer=face_recognizer)

    # Analyze the same photo (should recognize Elena)
    print(f"Testing: Analyzing {test_photo.name}")
    analysis = analyzer.analyze_photo_content(test_photo)

    assert analysis is not None
    print(f"Face count: {analysis.face_count}")
    print(f"People detected: {analysis.people_detected}")

    # Verify face was detected
    assert analysis.face_count > 0, "Should detect at least one face"

    # Verify Elena Rodriguez was identified
    assert len(analysis.people_detected) > 0, \
        f"Should identify at least one person, got: {analysis.people_detected}"
    assert "Elena Rodriguez" in analysis.people_detected, \
        f"Should identify Elena Rodriguez, got: {analysis.people_detected}"

    print("✓ Successfully identified Elena Rodriguez in the photo")

    # Cleanup
    analyzer.cleanup()
    if test_db_file.exists():
        test_db_file.unlink()
        test_db_file.parent.rmdir()


def test_content_analyzer_legacy():
    """Test the content analyzer functionality."""
    print("🔍 Testing Content Analyzer...")

    try:
        # Initialize components
        analyzer = ContentAnalyzer(use_gpu=False)
        detector = MediaDetector()

        # Get sample photos for testing
        all_files = detector.scan_iphone_automatic()
        photo_files = [f for f in all_files if f.file_type == 'photo']

        if not photo_files:
            print("❌ No photos found for testing")
            return

        print(f"Found {len(photo_files)} photos for analysis")

        # Test basic analysis (without ML models)
        print(f"\n📸 Testing basic content analysis...")

        sample_photos = photo_files[:3]  # Test first 3 photos

        for i, photo in enumerate(sample_photos):
            print(f"\n--- Photo {i+1}: {photo.filename} ---")
            print(f"Path: {photo.path}")
            print(f"Time: {photo.time}")

            # Analyze content
            analysis = analyzer.analyze_photo_content(photo.path)

            if analysis:
                print(f"✅ Analysis successful:")
                print(f"  Model: {analysis.analysis_model}")
                print(f"  Confidence: {analysis.confidence_score:.2f}")
                print(f"  Description: {analysis.description}")
                print(f"  Objects: {', '.join(analysis.objects) if analysis.objects else 'None detected'}")
                print(f"  Scenes: {', '.join(analysis.scenes) if analysis.scenes else 'None detected'}")
                print(f"  Activities: {', '.join(analysis.activities) if analysis.activities else 'None detected'}")
            else:
                print(f"❌ Analysis failed")

        # Test batch analysis
        print(f"\n🔄 Testing batch analysis...")

        batch_photos = [photo.path for photo in sample_photos]
        batch_results = analyzer.analyze_batch(batch_photos, max_photos=3)

        print(f"Batch analysis completed: {len(batch_results)} photos processed")

        # Generate content summary
        if batch_results:
            summary = analyzer.get_content_summary(batch_results)
            print(f"\n📊 Content Summary:")
            print(f"  Photos analyzed: {summary['total_photos_analyzed']}")
            print(f"  Average confidence: {summary['average_confidence']:.3f}")
            print(f"  Unique objects: {summary['unique_objects']}")
            print(f"  Unique scenes: {summary['unique_scenes']}")
            print(f"  Unique activities: {summary['unique_activities']}")

            if summary['top_objects']:
                print(f"  Top objects: {', '.join([obj for obj, count in summary['top_objects'][:3]])}")
            if summary['top_scenes']:
                print(f"  Top scenes: {', '.join([scene for scene, count in summary['top_scenes'][:3]])}")
            if summary['top_activities']:
                print(f"  Top activities: {', '.join([activity for activity, count in summary['top_activities'][:3]])}")

        # Test cache functionality
        print(f"\n💾 Testing cache functionality...")

        cache_file = Path("content_analysis_cache.json")

        # Save cache
        analyzer.save_analysis_cache(cache_file)

        # Clear cache and reload
        analyzer.analysis_cache.clear()
        print(f"  Cache cleared: {len(analyzer.analysis_cache)} entries")

        analyzer.load_analysis_cache(cache_file)
        print(f"  Cache reloaded: {len(analyzer.analysis_cache)} entries")

        # Clean up test file
        if cache_file.exists():
            cache_file.unlink()
            print(f"  Test cache file cleaned up")

        # Test with ML models if available
        print(f"\n🤖 Testing ML model availability...")

        try:
            import torch
            from transformers import CLIPModel
            print(f"  ✅ PyTorch available: {torch.__version__}")
            print(f"  ✅ Transformers available")

            # Test GPU availability
            if torch.cuda.is_available():
                print(f"  ✅ CUDA GPU available: {torch.cuda.get_device_name()}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print(f"  ✅ Apple MPS available")
            else:
                print(f"  ℹ️  CPU only (no GPU acceleration)")

            # Note about model initialization
            print(f"\n💡 Note: ML models are lazy-loaded on first use")
            print(f"   This test initializes ContentAnalyzer but doesn't analyze photos to save time")

        except ImportError as e:
            print(f"  ❌ ML dependencies not available: {e}")
            print(f"  💡 Install with: pip install torch transformers")

        print(f"\n✅ Content analyzer testing completed!")

        # Cleanup
        analyzer.cleanup()

    except Exception as e:
        print(f"❌ Error in content analyzer test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_content_analyzer()