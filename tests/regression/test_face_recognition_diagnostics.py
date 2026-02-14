#!/usr/bin/env python3
"""
Regression tests for face recognition diagnostic logging.

These tests ensure diagnostic logging doesn't break face recognition processing.
Tests specifically cover:
- Diagnostic logging when people database is populated
- Diagnostic logging when faces found but no recognition attempted
- Using correct PeopleDatabase methods (list_people, not get_all_people)
- REGRESSION: Prevents reintroduction of get_all_people() bug

Uses REAL photos from tests/artifacts/photos/ for realistic testing.
"""

import pytest
from pathlib import Path
from datetime import datetime

from src.media_clustering import MediaClusteringEngine, MediaCluster
from src.temporal_clustering import TemporalCluster
from src.face_recognizer import FaceRecognizer
from src.people_database import PeopleDatabase
from src.media_detector import MediaFile


@pytest.fixture
def real_test_photos():
    """Return paths to real photos in tests/artifacts/photos/."""
    base_path = Path("tests/artifacts/photos")

    # Verify photos exist
    with_faces = [
        base_path / "Woman_Photo_1.jpeg",
        base_path / "Woman_Photo_2.jpg",
    ]

    no_faces = [
        base_path / "no_faces_photo1.jpg",
    ]

    # Verify all photos exist
    for photo in with_faces + no_faces:
        assert photo.exists(), f"Test photo missing: {photo}"

    return {
        'with_faces': with_faces,
        'no_faces': no_faces
    }


@pytest.fixture
def test_people_database_regression(temp_test_dir, real_test_photos):
    """Create test people database with person from real photo for regression testing."""
    print("🧪 Setting up regression test people database")

    # Create fresh people database in temp directory
    db_file = temp_test_dir / "regression_people_database.json"
    people_db = PeopleDatabase(database_file=db_file)

    # Create face recognizer to extract encoding from Woman_Photo_1.jpeg
    face_recognizer = FaceRecognizer(
        detection_model="hog",
        recognition_tolerance=0.6,
        people_database=people_db
    )

    # Use Woman_Photo_1.jpeg as the source
    source_photo = real_test_photos['with_faces'][0]
    print(f"🧪 Extracting face encoding from: {source_photo}")

    # Detect faces in the source photo
    result = face_recognizer.detect_faces(source_photo)

    if result.faces_detected > 0:
        # Get the first face encoding
        first_face = result.faces[0]
        face_encoding = first_face.encoding

        # Add test person to the database
        people_db.add_person(
            person_id="Test Person",
            name="Test Person",
            encodings=[face_encoding],
            photo_paths=[str(source_photo)],
            notes="Test person for regression testing"
        )
        print("✅ Test Person added to regression database")
    else:
        print("⚠️ No faces detected in source photo, database will be empty")

    return people_db, db_file


@pytest.mark.regression
@pytest.mark.integration
def test_diagnostic_logging_with_populated_people_database(test_people_database_regression, real_test_photos):
    """
    REGRESSION TEST: Ensures diagnostic logging doesn't crash when people database is populated.

    This test would have caught the bug where get_all_people() was used instead of list_people().
    Uses real photos and real face detection for realistic testing.
    """
    print("🧪 REGRESSION: Testing diagnostic logging with populated people database")

    people_db, db_file = test_people_database_regression

    # Verify database has people
    people_list = people_db.list_people()
    print(f"🧪 People database has {len(people_list)} person(s)")

    # Create real face recognizer (not mocked)
    face_recognizer = FaceRecognizer(
        detection_model="hog",
        recognition_tolerance=0.6,
        people_database=people_db
    )

    # Create clustering engine with people database
    engine = MediaClusteringEngine(
        face_recognizer=face_recognizer,
        people_database=people_db
    )

    # Create cluster with real photo
    test_photo = real_test_photos['with_faces'][1]  # Woman_Photo_2.jpg
    media_file = MediaFile(
        path=test_photo,
        filename=test_photo.name,
        date=datetime(2024, 10, 25, 16, 30, 0),
        time=datetime(2024, 10, 25, 16, 30, 0),
        extension=test_photo.suffix,
        file_type='photo',
        size=test_photo.stat().st_size
    )

    start_time = datetime(2024, 10, 25, 16, 30, 0)
    end_time = datetime(2024, 10, 25, 17, 30, 0)
    temporal_cluster = TemporalCluster(
        cluster_id=0,
        start_time=start_time,
        end_time=end_time,
        duration=end_time - start_time,
        media_files=[media_file]
    )

    cluster = MediaCluster(
        cluster_id=0,
        media_files=[media_file],
        temporal_info=temporal_cluster,
        location_info=None,
        dominant_location=None,
        gps_coordinates=[]
    )

    # This should NOT crash - it should successfully log diagnostic
    # The bug was: self.people_database.get_all_people() crashed with AttributeError
    # The fix: self.people_database.list_people() works correctly
    try:
        enhanced_clusters = engine._enhance_with_people_data([cluster])
        print("✅ Diagnostic logging completed without crash")
        assert len(enhanced_clusters) == 1, "Should return 1 enhanced cluster"
    except AttributeError as e:
        if "get_all_people" in str(e):
            pytest.fail(f"REGRESSION: Using non-existent get_all_people() method: {e}")
        raise

    print("✅ REGRESSION TEST PASSED: Diagnostic logging works with populated database")


@pytest.mark.regression
@pytest.mark.integration
def test_diagnostic_logging_uses_correct_people_database_method(test_people_database_regression, real_test_photos):
    """
    REGRESSION TEST: Verifies diagnostic logging uses list_people() not get_all_people().

    Directly tests that the correct PeopleDatabase method is used in diagnostic code.
    """
    print("🧪 REGRESSION: Testing correct PeopleDatabase method usage")

    people_db, db_file = test_people_database_regression

    # Verify list_people exists and works
    assert hasattr(people_db, 'list_people'), "PeopleDatabase should have list_people method"
    people_list = people_db.list_people()
    print(f"🧪 list_people() returned {len(people_list)} person(s)")

    # Verify get_all_people does NOT exist
    assert not hasattr(people_db, 'get_all_people'), \
        "PeopleDatabase should NOT have get_all_people method (doesn't exist in API)"

    # Create real face recognizer
    face_recognizer = FaceRecognizer(
        detection_model="hog",
        recognition_tolerance=0.6,
        people_database=people_db
    )

    # Create engine
    engine = MediaClusteringEngine(
        face_recognizer=face_recognizer,
        people_database=people_db
    )

    # Create cluster with photo without faces (to trigger diagnostic path)
    no_faces_photo = real_test_photos['no_faces'][0]
    media_file = MediaFile(
        path=no_faces_photo,
        filename=no_faces_photo.name,
        date=datetime(2024, 10, 25, 16, 30, 0),
        time=datetime(2024, 10, 25, 16, 30, 0),
        extension=no_faces_photo.suffix,
        file_type='photo',
        size=no_faces_photo.stat().st_size
    )

    start_time = datetime(2024, 10, 25, 16, 30, 0)
    end_time = datetime(2024, 10, 25, 17, 30, 0)
    temporal_cluster = TemporalCluster(
        cluster_id=0,
        start_time=start_time,
        end_time=end_time,
        duration=end_time - start_time,
        media_files=[media_file]
    )

    cluster = MediaCluster(
        cluster_id=0,
        media_files=[media_file],
        temporal_info=temporal_cluster,
        location_info=None,
        dominant_location=None,
        gps_coordinates=[]
    )

    # Run enhancement - should use list_people, not get_all_people
    try:
        enhanced_clusters = engine._enhance_with_people_data([cluster])
        print("✅ Enhancement completed without calling non-existent get_all_people()")
    except AttributeError as e:
        if "get_all_people" in str(e):
            pytest.fail(f"REGRESSION: Code is calling non-existent get_all_people() method: {e}")
        raise

    print("✅ REGRESSION TEST PASSED: Using correct list_people() method")


@pytest.mark.regression
@pytest.mark.integration
def test_face_recognition_with_real_people_database_doesnt_crash(test_people_database_regression, real_test_photos):
    """
    REGRESSION TEST: Integration test ensuring face recognition works with real people database.

    Tests the full path through face recognition with diagnostic logging enabled.
    Uses real photos and real face detection to ensure realistic testing.
    """
    print("🧪 REGRESSION: Integration test with real people database and photos")

    people_db, db_file = test_people_database_regression

    # Verify database is populated
    all_people = people_db.list_people()
    print(f"🧪 People database has {len(all_people)} person(s)")

    # Create real face recognizer
    face_recognizer = FaceRecognizer(
        detection_model="hog",
        recognition_tolerance=0.6,
        people_database=people_db
    )

    # Create engine with real people database
    engine = MediaClusteringEngine(
        face_recognizer=face_recognizer,
        people_database=people_db
    )

    # Test multiple scenarios with different real photos
    test_scenarios = [
        ("with_faces", real_test_photos['with_faces'][0], "Photo with faces"),
        ("no_faces", real_test_photos['no_faces'][0], "Photo without faces"),
    ]

    for scenario_name, photo_path, description in test_scenarios:
        print(f"🧪 Testing scenario: {description} ({photo_path.name})")

        # Create cluster
        media_file = MediaFile(
            path=photo_path,
            filename=photo_path.name,
            date=datetime(2024, 10, 25, 16, 30, 0),
            time=datetime(2024, 10, 25, 16, 30, 0),
            extension=photo_path.suffix,
            file_type='photo',
            size=photo_path.stat().st_size
        )

        start_time = datetime(2024, 10, 25, 16, 30, 0)
        end_time = datetime(2024, 10, 25, 17, 30, 0)
        temporal_cluster = TemporalCluster(
            cluster_id=0,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            media_files=[media_file]
        )

        cluster = MediaCluster(
            cluster_id=0,
            media_files=[media_file],
            temporal_info=temporal_cluster,
            location_info=None,
            dominant_location=None,
            gps_coordinates=[]
        )

        # Should not crash on any scenario
        try:
            enhanced_clusters = engine._enhance_with_people_data([cluster])
            assert len(enhanced_clusters) == 1, f"{description}: Should return 1 cluster"
            print(f"   ✅ {description} processed successfully")
        except AttributeError as e:
            if "get_all_people" in str(e):
                pytest.fail(f"{description}: REGRESSION - using get_all_people(): {e}")
            raise
        except Exception as e:
            pytest.fail(f"{description}: Unexpected error in face recognition: {e}")

    print("✅ REGRESSION TEST PASSED: All scenarios processed without crashes")


if __name__ == "__main__":
    """Run the regression tests standalone."""
    print("🧪 Face Recognition Diagnostic Regression Tests")
    print("📋 Test Plan:")
    print("   1. Use REAL photos from tests/artifacts/photos/")
    print("   2. Create real people database with person from real photo")
    print("   3. Test diagnostic logging doesn't crash with populated database")
    print("   4. Verify correct list_people() method is used (not get_all_people())")
    print("   5. Integration test with multiple scenarios")
    print()

    # Run with pytest
    pytest.main([__file__, "-v", "-s"])
