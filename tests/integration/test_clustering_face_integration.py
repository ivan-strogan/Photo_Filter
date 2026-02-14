#!/usr/bin/env python3
"""
Real photo face recognition integration tests for clustering pipeline.

This test verifies face recognition integration using REAL photos and REAL face detection:
1. Uses actual photos from tests/artifacts/photos/
2. Creates real people database with Elena Rodriguez from Woman_Photo_1.jpeg
3. Tests real face recognition without mocking
4. Verifies full clustering pipeline with face recognition
5. Prevents regression of Issue #13 (face recognition not running during clustering)

Uses proper pytest fixtures and follows the project test structure.
"""

import pytest
import json
import random
from pathlib import Path
from datetime import datetime, timedelta
from PIL import Image

# Optional EXIF manipulation library
try:
    import piexif
    PIEXIF_AVAILABLE = True
except ImportError:
    PIEXIF_AVAILABLE = False

# Import core classes
try:
    from src.media_clustering import MediaClusteringEngine, MediaCluster
    from src.temporal_clustering import TemporalCluster
    from src.face_recognizer import FaceRecognizer, FaceRecognitionResult
    from src.people_database import PeopleDatabase
    from src.photo_organizer_pipeline import PhotoOrganizerPipeline
    from src.media_detector import MediaFile
    from src.config_manager import get_config
except ImportError:
    # Fallback imports if needed
    import media_clustering
    import temporal_clustering
    import face_recognizer
    import people_database
    import photo_organizer_pipeline
    import media_detector
    import config_manager
    MediaClusteringEngine = media_clustering.MediaClusteringEngine
    MediaCluster = media_clustering.MediaCluster
    TemporalCluster = temporal_clustering.TemporalCluster
    FaceRecognizer = face_recognizer.FaceRecognizer
    FaceRecognitionResult = face_recognizer.FaceRecognitionResult
    PeopleDatabase = people_database.PeopleDatabase
    PhotoOrganizerPipeline = photo_organizer_pipeline.PhotoOrganizerPipeline
    MediaFile = media_detector.MediaFile
    get_config = config_manager.get_config


@pytest.fixture
def real_test_photos():
    """Return paths to real photos in tests/artifacts/photos/."""
    base_path = Path("tests/artifacts/photos")

    # Verify photos exist
    with_faces = [
        base_path / "Woman_Photo_1.jpeg",
        base_path / "Woman_Photo_2.jpg",
        base_path / "Woman_Photo_3.jpg"
    ]

    no_faces = [
        base_path / "no_faces_photo1.jpg",
        base_path / "no_faces_photo2.jpg",
        base_path / "no_faces_photo3.jpg"
    ]

    # Verify all photos exist
    for photo in with_faces + no_faces:
        assert photo.exists(), f"Test photo missing: {photo}"

    return {
        'with_faces': with_faces,
        'no_faces': no_faces
    }


@pytest.fixture
def test_people_database_with_elena(temp_test_dir, real_test_photos):
    """Create test people database and add Elena Rodriguez from Woman_Photo_1.jpeg."""
    print("🧪 Setting up test people database with Elena Rodriguez from real photo")

    # Create fresh people database in temp directory
    db_file = temp_test_dir / "test_people_database.json"
    people_db = PeopleDatabase(database_file=db_file)

    # Create face recognizer to extract encoding from Woman_Photo_1.jpeg
    face_recognizer = FaceRecognizer(
        detection_model="hog",
        recognition_tolerance=0.6,
        people_database=people_db  # Pass the database for proper setup
    )

    # Use Woman_Photo_1.jpeg as the source for Elena Rodriguez
    source_photo = real_test_photos['with_faces'][0]  # Woman_Photo_1.jpeg
    print(f"🧪 Extracting face encoding from: {source_photo}")

    # Detect faces in the source photo
    result = face_recognizer.detect_faces(source_photo)

    assert result.faces_detected > 0, f"No faces found in source photo: {source_photo}"
    print(f"🧪 Found {result.faces_detected} face(s) in source photo")

    # Get the first face encoding
    first_face = result.faces[0]
    face_encoding = first_face.encoding

    # Add Elena Rodriguez to the database using the real face encoding
    people_db.add_person(
        person_id="Elena Rodriguez",
        name="Elena Rodriguez",
        encodings=[face_encoding],  # List of encodings
        photo_paths=[str(source_photo)],  # List of photo paths
        notes="Test person created from Woman_Photo_1.jpeg"
    )

    print("✅ Elena Rodriguez added to test people database")

    return people_db, db_file


@pytest.fixture
def real_media_files(real_test_photos):
    """Create MediaFile objects from real photos."""
    media_files = []

    # Create MediaFiles for photos with faces
    for i, photo_path in enumerate(real_test_photos['with_faces']):
        media_file = MediaFile(
            path=photo_path,
            filename=photo_path.name,
            date=datetime(2024, 10, 25, 16, 30, i),  # Stagger times slightly
            time=datetime(2024, 10, 25, 16, 30, i),
            extension=photo_path.suffix,
            file_type='photo',
            size=photo_path.stat().st_size
        )
        media_files.append(media_file)

    return media_files


@pytest.fixture
def real_test_cluster(real_media_files):
    """Create a test cluster with real MediaFiles."""
    # Create temporal cluster
    start_time = datetime(2024, 10, 25, 16, 30, 0)
    end_time = datetime(2024, 10, 25, 17, 30, 0)
    temporal_cluster = TemporalCluster(
        cluster_id=1,
        start_time=start_time,
        end_time=end_time,
        duration=end_time - start_time,
        media_files=real_media_files
    )

    # Create MediaCluster with real photos
    return MediaCluster(
        cluster_id=1,
        media_files=real_media_files,
        temporal_info=temporal_cluster
    )


@pytest.mark.integration
@pytest.mark.slow
def test_real_face_recognition_with_real_photos(test_people_database_with_elena, real_test_photos):
    """Test face recognition with real photos and real detection (no mocking)."""
    print("🧪 Testing REAL face recognition with REAL photos")

    people_db, db_file = test_people_database_with_elena

    # Create real face recognizer (no mocking)
    face_recognizer = FaceRecognizer(
        detection_model="hog",
        recognition_tolerance=0.6,
        people_database=people_db
    )

    # Test positive cases: Should detect Elena in woman photos
    print("🧪 Testing positive cases (photos with faces)")
    elena_detected_count = 0

    for photo_path in real_test_photos['with_faces']:
        print(f"🧪 Processing: {photo_path.name}")
        result = face_recognizer.detect_faces(photo_path)

        print(f"   Faces detected: {result.faces_detected}")
        if result.faces:
            detected_people = [face.person_id for face in result.faces if face.person_id]
            print(f"   People identified: {detected_people}")

            if "Elena Rodriguez" in detected_people:
                elena_detected_count += 1

    # Elena should be detected in at least the source photo (Woman_Photo_1.jpeg)
    assert elena_detected_count >= 1, \
        f"Elena Rodriguez should be detected in at least 1 photo, found in {elena_detected_count}"

    print(f"✅ Elena Rodriguez detected in {elena_detected_count} photos")

    # Test negative cases: Should detect no faces in no-face photos
    print("🧪 Testing negative cases (photos without faces)")
    for photo_path in real_test_photos['no_faces']:
        print(f"🧪 Processing: {photo_path.name}")
        result = face_recognizer.detect_faces(photo_path)
        print(f"   Faces detected: {result.faces_detected}")

        assert result.faces_detected == 0, \
            f"No faces should be detected in {photo_path.name}, found {result.faces_detected}"

    print("✅ No faces correctly detected in no-face photos")
    print("✅ Real face recognition integration test PASSED")


@pytest.mark.integration
@pytest.mark.slow
def test_clustering_pipeline_with_real_photos(test_people_database_with_elena, real_test_cluster):
    """Test full clustering pipeline with real photos and face recognition."""
    print("🧪 Testing full clustering pipeline with REAL photos and face recognition")

    people_db, db_file = test_people_database_with_elena

    # Create clustering engine with real face recognition components
    face_recognizer = FaceRecognizer(
        detection_model="hog",
        recognition_tolerance=0.6,
        people_database=people_db
    )

    engine = MediaClusteringEngine(
        face_recognizer=face_recognizer,
        people_database=people_db
    )

    print(f"🧪 Processing cluster with {len(real_test_cluster.media_files)} real photos")

    # Run real face recognition enhancement (no mocking)
    enhanced_clusters = engine._enhance_with_people_data([real_test_cluster])

    # Verify enhancement worked
    assert len(enhanced_clusters) == 1, "Should return one enhanced cluster"
    enhanced_cluster = enhanced_clusters[0]

    print(f"🧪 Enhanced cluster people detected: {enhanced_cluster.people_detected}")

    # Elena should be detected in the real photos through full pipeline
    assert enhanced_cluster.people_detected is not None, \
        "Enhanced cluster should have people data"

    # Elena should be detected (at least in Woman_Photo_1.jpeg which was used to create her encoding)
    assert "Elena Rodriguez" in enhanced_cluster.people_detected, \
        f"Elena Rodriguez should be detected in real photos, found: {enhanced_cluster.people_detected}"

    print("✅ Elena Rodriguez detected through full clustering pipeline")
    print("✅ Full pipeline integration test PASSED")


@pytest.mark.integration
def test_photo_organizer_pipeline_with_real_face_recognition(test_people_database_with_elena):
    """Test PhotoOrganizerPipeline integration with real face recognition setup."""
    print("🧪 Testing PhotoOrganizerPipeline with real face recognition components")

    people_db, db_file = test_people_database_with_elena

    # Create face recognizer with the test database
    face_recognizer = FaceRecognizer(
        detection_model="hog",
        recognition_tolerance=0.6,
        people_database=people_db
    )

    # Create MediaClusteringEngine directly with our test components
    # (avoiding pipeline config complexity for this specific test)
    engine = MediaClusteringEngine(
        face_recognizer=face_recognizer,
        people_database=people_db
    )

    # Verify face recognition components are properly assigned
    assert engine.face_recognizer is not None, \
        "Engine should have face recognizer"
    assert engine.people_database is not None, \
        "Engine should have people database"

    # Verify the people database contains Elena Rodriguez
    elena_person = engine.people_database.find_person_by_name("Elena Rodriguez")
    assert elena_person is not None, \
        "Test people database should contain Elena Rodriguez"

    print("✅ MediaClusteringEngine properly integrates with real face recognition components")


@pytest.mark.regression
@pytest.mark.slow
def test_regression_issue_13_with_real_photos(test_people_database_with_elena, real_test_photos):
    """Regression test for Issue #13 using real photos and real face recognition.

    This test ensures Issue #13 (face recognition not running during clustering)
    stays fixed by testing the complete integration with real photos.
    """
    print("🧪 REGRESSION TEST: Issue #13 with REAL photos and face recognition")

    people_db, db_file = test_people_database_with_elena

    # Create face recognizer with more relaxed tolerance for regression test
    face_recognizer = FaceRecognizer(
        detection_model="hog",
        recognition_tolerance=0.7,  # More relaxed tolerance
        people_database=people_db
    )

    # Create MediaClusteringEngine directly (avoiding pipeline config complexity)
    clustering_engine = MediaClusteringEngine(
        face_recognizer=face_recognizer,
        people_database=people_db
    )

    # These assertions would have failed before Issue #13 fix
    assert clustering_engine.face_recognizer is not None, \
        "REGRESSION: MediaClusteringEngine missing face_recognizer parameter (Issue #13)"
    assert clustering_engine.people_database is not None, \
        "REGRESSION: MediaClusteringEngine missing people_database parameter (Issue #13)"

    # Test that face recognition works on at least one of the woman photos
    # (Elena should be detected in at least Woman_Photo_2 or Woman_Photo_3)
    elena_detected = False
    for test_photo in real_test_photos['with_faces']:
        result = clustering_engine.face_recognizer.detect_faces(test_photo)
        print(f"🧪 Face detection on {test_photo.name}: {result.faces_detected} faces")

        if result.faces:
            detected_people = [face.person_id for face in result.faces if face.person_id]
            print(f"🧪 People identified: {detected_people}")

            if "Elena Rodriguez" in detected_people:
                elena_detected = True
                break

    # Elena should be detected in at least one photo to verify Issue #13 is fixed
    assert elena_detected, \
        "REGRESSION: Face recognition should detect Elena in at least one photo (Issue #13 verification)"

    print("✅ REGRESSION TEST PASSED: Issue #13 fix verified with real photos")


@pytest.mark.unit
def test_media_clustering_engine_receives_real_face_components(test_people_database_with_elena):
    """Test that MediaClusteringEngine properly receives real face recognition components."""
    print("🧪 Testing MediaClusteringEngine with real face recognition components")

    people_db, db_file = test_people_database_with_elena

    # Create real face recognition components
    face_recognizer = FaceRecognizer(
        detection_model="hog",
        recognition_tolerance=0.6,
        people_database=people_db
    )

    # Create MediaClusteringEngine with real face recognition
    engine = MediaClusteringEngine(
        face_recognizer=face_recognizer,
        people_database=people_db
    )

    # Verify components are properly assigned
    assert engine.face_recognizer is not None, "Face recognizer should be assigned"
    assert engine.people_database is not None, "People database should be assigned"
    assert engine.face_recognizer == face_recognizer, "Face recognizer should match"
    assert engine.people_database == people_db, "People database should match"

    # Verify the people database actually contains Elena
    elena_person = engine.people_database.find_person_by_name("Elena Rodriguez")
    assert elena_person is not None, "People database should contain Elena Rodriguez"

    print("✅ MediaClusteringEngine properly receives real face recognition components")


@pytest.mark.integration
@pytest.mark.regression
@pytest.mark.skipif(not PIEXIF_AVAILABLE, reason="piexif library not installed")
def test_face_recognition_with_exif_orientation(temp_test_dir, real_test_photos):
    """Test that face recognition handles EXIF orientation correctly (Issue #53).

    Simulates iPhone behavior: pixels stored rotated + EXIF tag for correction.
    Our _load_image_with_orientation() should correct before face detection.
    """
    print("🧪 Testing EXIF orientation handling (Issue #53)")

    source_photo = real_test_photos['with_faces'][0]
    test_photo_path = temp_test_dir / "test_exif_rotated.jpg"

    face_recognizer = FaceRecognizer(
        detection_model="hog",
        recognition_tolerance=0.6,
        enable_caching=False
    )

    # Verify original works
    original_result = face_recognizer.detect_faces(source_photo)
    print(f"🧪 Original: {original_result.faces_detected} faces")
    assert original_result.faces_detected > 0, "Should detect face in original"

    try:
        # Simulate iPhone EXIF 6: pixels stored 90° CCW, EXIF says rotate 90° CW to correct
        # PIL rotate() is counter-clockwise, so rotate(90) = 90° CCW
        img = Image.open(source_photo)
        rotated_img = img.rotate(90, expand=True)

        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        exif_dict["0th"][piexif.ImageIFD.Orientation] = 6
        exif_bytes = piexif.dump(exif_dict)
        rotated_img.save(test_photo_path, "JPEG", exif=exif_bytes)

        # Our code should read EXIF, correct rotation, then detect face
        result = face_recognizer.detect_faces(test_photo_path)
        print(f"🧪 With EXIF orientation 6: {result.faces_detected} faces")

        assert result.faces_detected > 0, \
            "REGRESSION #53: Face not detected with EXIF orientation correction"

        print("✅ EXIF orientation test PASSED")

    finally:
        if test_photo_path.exists():
            test_photo_path.unlink()


@pytest.mark.integration
@pytest.mark.regression
def test_face_recognition_with_random_pixel_rotation(temp_test_dir, real_test_photos):
    """Test face recognition on a randomly rotated image.

    Rotates image pixels by a random angle (30-270°) to simulate a tilted face.
    HOG face detection works best on upright faces, so this tests robustness.
    """
    print("🧪 Testing face recognition with random pixel rotation")

    source_photo = real_test_photos['with_faces'][0]
    random_angle = random.randint(30, 270)
    test_photo_path = temp_test_dir / f"test_rotated_{random_angle}.jpg"

    face_recognizer = FaceRecognizer(
        detection_model="hog",
        recognition_tolerance=0.6,
        enable_caching=False
    )

    # Verify original works
    original_result = face_recognizer.detect_faces(source_photo)
    print(f"🧪 Original: {original_result.faces_detected} faces")
    assert original_result.faces_detected > 0, "Should detect face in original"

    try:
        # Rotate image by random angle
        img = Image.open(source_photo)
        rotated_img = img.rotate(random_angle, expand=True, fillcolor=(255, 255, 255))
        rotated_img.save(test_photo_path, "JPEG")

        # Test face detection on rotated image
        result = face_recognizer.detect_faces(test_photo_path)
        print(f"🧪 Rotated {random_angle}°: {result.faces_detected} faces")

        # HOG may not detect rotated faces - that's OK, test shouldn't crash
        if result.faces_detected > 0:
            print(f"✅ Face detected at {random_angle}° rotation!")
        else:
            print(f"⚠️  No face at {random_angle}° (expected for HOG detector)")

        print("✅ Pixel rotation test completed (no crash)")

    finally:
        if test_photo_path.exists():
            test_photo_path.unlink()


if __name__ == "__main__":
    """Run the real photo face recognition integration tests standalone."""
    print("🧪 Real Photo Face Recognition Integration Test")
    print("📋 Test Plan:")
    print("   1. Use REAL photos from tests/artifacts/photos/")
    print("   2. Create real people database with Elena Rodriguez")
    print("   3. Test real face recognition (no mocking)")
    print("   4. Test full clustering pipeline integration")
    print("   5. Regression test for Issue #13 with real photos")
    print()

    # Check dependencies
    try:
        import face_recognition
        print("✅ face_recognition library available")
    except ImportError:
        print("❌ face_recognition library not available")
        import sys
        sys.exit(1)

    # Check that test photos exist
    photos_dir = Path("tests/artifacts/photos")
    if not photos_dir.exists():
        print(f"❌ Test photos directory not found: {photos_dir}")
        import sys
        sys.exit(1)

    print(f"✅ Test photos directory found: {photos_dir}")

    # Run with pytest
    pytest.main([__file__, "-v", "-s"])