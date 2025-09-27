#!/usr/bin/env python3
"""
Unit tests for face recognition clustering components.

These are FAST unit tests that test individual components in isolation using
mocked dependencies. No real photos, no real face recognition, no slow operations.

Focus:
- Component initialization and configuration
- Logic validation with controlled inputs
- Error handling and edge cases
- The specific bugs we fixed (MediaCluster constructor, etc.)

Run with: pytest tests/unit/test_clustering_face_unit.py -v
Expected time: <3 seconds total
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
from datetime import datetime, timedelta

# Import core classes
try:
    from src.media_clustering import MediaClusteringEngine, MediaCluster
    from src.temporal_clustering import TemporalCluster
    from src.face_recognizer import FaceRecognizer, FaceRecognitionResult, Face
    from src.people_database import PeopleDatabase
    from src.media_detector import MediaFile
except ImportError:
    # Fallback imports if needed
    import media_clustering
    import temporal_clustering
    import face_recognizer
    import people_database
    import media_detector
    MediaClusteringEngine = media_clustering.MediaClusteringEngine
    MediaCluster = media_clustering.MediaCluster
    TemporalCluster = temporal_clustering.TemporalCluster
    FaceRecognizer = face_recognizer.FaceRecognizer
    FaceRecognitionResult = face_recognizer.FaceRecognitionResult
    Face = face_recognizer.Face
    PeopleDatabase = people_database.PeopleDatabase
    MediaFile = media_detector.MediaFile


# ===== MOCK FIXTURES =====

@pytest.fixture
def mock_people_database():
    """Mock PeopleDatabase for unit tests."""
    mock_db = Mock(spec=PeopleDatabase)
    mock_db.find_person_by_name.return_value = None  # Default: no person found
    mock_db.list_people.return_value = []  # Default: empty database
    return mock_db


@pytest.fixture
def mock_face_recognizer():
    """Mock FaceRecognizer for unit tests."""
    mock_recognizer = Mock(spec=FaceRecognizer)
    # Default: no faces detected
    mock_result = Mock(spec=FaceRecognitionResult)
    mock_result.faces_detected = 0
    mock_result.faces = []
    mock_recognizer.detect_faces.return_value = mock_result
    return mock_recognizer


@pytest.fixture
def mock_face_with_elena():
    """Mock Face object representing Elena Rodriguez."""
    mock_face = Mock(spec=Face)
    mock_face.person_id = "Elena Rodriguez"
    mock_face.confidence = 0.8
    mock_face.encoding = [0.1] * 128  # Mock 128-dimensional encoding
    return mock_face


@pytest.fixture
def mock_face_recognition_result_with_elena(mock_face_with_elena):
    """Mock FaceRecognitionResult with Elena detected."""
    mock_result = Mock(spec=FaceRecognitionResult)
    mock_result.image_path = "mock_photo.jpg"
    mock_result.faces_detected = 1
    mock_result.faces = [mock_face_with_elena]
    mock_result.processing_time = 0.1
    mock_result.error = None
    # Mock the get_people_detected method to return Elena
    mock_result.get_people_detected.return_value = ["Elena Rodriguez"]
    return mock_result


@pytest.fixture
def sample_media_files():
    """Create sample MediaFile objects for testing."""
    media_files = []
    for i in range(3):
        media_file = MediaFile(
            path=Path(f"mock_photo_{i}.jpg"),
            filename=f"mock_photo_{i}.jpg",
            date=datetime(2024, 10, 25, 16, 30, i),
            time=datetime(2024, 10, 25, 16, 30, i),
            extension='.jpg',
            file_type='photo',
            size=1024
        )
        media_files.append(media_file)
    return media_files


@pytest.fixture
def sample_temporal_cluster(sample_media_files):
    """Create sample TemporalCluster for testing."""
    return TemporalCluster(
        cluster_id=1,
        start_time=datetime(2024, 10, 25, 16, 30, 0),
        end_time=datetime(2024, 10, 25, 17, 30, 0),
        duration=timedelta(hours=1),
        media_files=sample_media_files
    )


@pytest.fixture
def sample_media_cluster(sample_media_files, sample_temporal_cluster):
    """Create sample MediaCluster for testing."""
    return MediaCluster(
        cluster_id=1,
        media_files=sample_media_files,
        temporal_info=sample_temporal_cluster
    )


# ===== COMPONENT INITIALIZATION TESTS =====

@pytest.mark.unit
def test_media_clustering_engine_with_face_components(mock_face_recognizer, mock_people_database):
    """Test MediaClusteringEngine properly stores face recognition components."""
    print("🧪 Testing MediaClusteringEngine component initialization")

    # Create engine with face recognition components
    engine = MediaClusteringEngine(
        face_recognizer=mock_face_recognizer,
        people_database=mock_people_database
    )

    # Verify components are properly assigned
    assert engine.face_recognizer is not None, "Face recognizer should be assigned"
    assert engine.people_database is not None, "People database should be assigned"
    assert engine.face_recognizer == mock_face_recognizer, "Face recognizer should match"
    assert engine.people_database == mock_people_database, "People database should match"

    print("✅ MediaClusteringEngine properly stores face recognition components")


@pytest.mark.unit
def test_media_clustering_engine_without_face_components():
    """Test MediaClusteringEngine works without face recognition components."""
    print("🧪 Testing MediaClusteringEngine without face recognition")

    # Create engine without face recognition
    engine = MediaClusteringEngine()

    # Verify components are None
    assert engine.face_recognizer is None, "Face recognizer should be None"
    assert engine.people_database is None, "People database should be None"

    print("✅ MediaClusteringEngine works without face recognition components")


@pytest.mark.unit
def test_face_recognizer_initialization_parameters():
    """Test FaceRecognizer stores initialization parameters correctly."""
    print("🧪 Testing FaceRecognizer parameter storage")

    # Create mock people database
    mock_people_db = Mock(spec=PeopleDatabase)

    # Create FaceRecognizer with specific parameters
    face_recognizer = FaceRecognizer(
        detection_model="hog",
        recognition_tolerance=0.6,
        people_database=mock_people_db
    )

    # Verify parameters are stored
    assert face_recognizer.detection_model == "hog", "Detection model should be stored"
    assert face_recognizer.recognition_tolerance == 0.6, "Recognition tolerance should be stored"
    assert face_recognizer.people_database == mock_people_db, "People database should be stored"

    print("✅ FaceRecognizer properly stores initialization parameters")


# ===== PEOPLE DATABASE UNIT TESTS =====

@pytest.mark.unit
def test_people_database_add_person_success(temp_test_dir):
    """Test successfully adding a person to people database."""
    print("🧪 Testing PeopleDatabase add_person success case")

    # Create temporary database file
    db_file = temp_test_dir / "test_people_db.json"
    people_db = PeopleDatabase(database_file=db_file)

    # Mock face encoding
    mock_encoding = [0.1] * 128  # 128-dimensional encoding

    # Add person to database
    success = people_db.add_person(
        person_id="Test Person",
        name="Test Person",
        encodings=[mock_encoding],
        photo_paths=["mock_photo.jpg"],
        notes="Unit test person"
    )

    # Verify person was added successfully
    assert success, "Adding person should succeed"

    # Verify person can be found
    person = people_db.find_person_by_name("Test Person")
    assert person is not None, "Added person should be findable"
    assert person.name == "Test Person", "Person name should match"

    print("✅ PeopleDatabase successfully adds person")


@pytest.mark.unit
def test_people_database_find_nonexistent_person(temp_test_dir):
    """Test finding non-existent person returns None."""
    print("🧪 Testing PeopleDatabase find_person with non-existent person")

    # Create empty database
    db_file = temp_test_dir / "empty_db.json"
    people_db = PeopleDatabase(database_file=db_file)

    # Try to find non-existent person
    person = people_db.find_person_by_name("Non-existent Person")
    assert person is None, "Non-existent person should return None"

    print("✅ PeopleDatabase correctly returns None for non-existent person")


# ===== CLUSTERING LOGIC UNIT TESTS =====

@pytest.mark.unit
def test_enhance_with_people_data_with_detected_faces(
    mock_face_recognizer, mock_people_database, sample_media_cluster, mock_face_recognition_result_with_elena
):
    """Test people enhancement logic when faces are detected."""
    print("🧪 Testing clustering enhancement with detected faces")

    # Configure mock face recognizer to return Elena Rodriguez
    mock_face_recognizer.detect_faces.return_value = mock_face_recognition_result_with_elena

    # Create engine with mocked components
    engine = MediaClusteringEngine(
        face_recognizer=mock_face_recognizer,
        people_database=mock_people_database
    )

    # Run enhancement
    enhanced_clusters = engine._enhance_with_people_data([sample_media_cluster])

    # Verify enhancement worked
    assert len(enhanced_clusters) == 1, "Should return one enhanced cluster"
    enhanced_cluster = enhanced_clusters[0]

    # Verify face recognition was called for each photo
    assert mock_face_recognizer.detect_faces.call_count == len(sample_media_cluster.media_files), \
        "Face detection should be called for each photo"

    # Verify people were detected
    assert enhanced_cluster.people_detected is not None, "Enhanced cluster should have people data"
    assert "Elena Rodriguez" in enhanced_cluster.people_detected, "Elena should be detected"

    print("✅ Clustering enhancement works when faces are detected")


@pytest.mark.unit
def test_enhance_with_people_data_no_faces_detected(
    mock_face_recognizer, mock_people_database, sample_media_cluster
):
    """Test people enhancement when no faces are detected."""
    print("🧪 Testing clustering enhancement with no faces detected")

    # Configure mock to return no faces
    mock_result = Mock(spec=FaceRecognitionResult)
    mock_result.faces_detected = 0
    mock_result.faces = []
    mock_face_recognizer.detect_faces.return_value = mock_result

    # Create engine with mocked components
    engine = MediaClusteringEngine(
        face_recognizer=mock_face_recognizer,
        people_database=mock_people_database
    )

    # Run enhancement
    enhanced_clusters = engine._enhance_with_people_data([sample_media_cluster])

    # Verify enhancement handled no faces gracefully
    assert len(enhanced_clusters) == 1, "Should return one enhanced cluster"
    enhanced_cluster = enhanced_clusters[0]

    # Verify no people were detected
    assert enhanced_cluster.people_detected == [], "No people should be detected when no faces found"

    print("✅ Clustering enhancement handles no faces detected correctly")


# ===== MEDIA CLUSTER CONSTRUCTOR TESTS (Bug Fix Validation) =====

@pytest.mark.unit
def test_media_cluster_constructor_correct_attributes(sample_media_files, sample_temporal_cluster):
    """Test MediaCluster constructor uses correct attributes (validates our bug fix)."""
    print("🧪 Testing MediaCluster constructor with correct attributes")

    # This test validates the bug fix we made:
    # - Changed primary_location -> dominant_location
    # - Ensured temporal_info is required
    # - Fixed attribute naming issues

    # Create MediaCluster with correct attributes
    cluster = MediaCluster(
        cluster_id=1,
        media_files=sample_media_files,
        temporal_info=sample_temporal_cluster,  # Required
        location_info=None,
        dominant_location="Test Location",  # Not primary_location
        gps_coordinates=[(37.7749, -122.4194)],
        content_tags=["test", "photo"],
        people_detected=["Elena Rodriguez"],
        confidence_score=0.8,
        suggested_name="Test Event",
        metadata={"test": "data"}
    )

    # Verify all attributes are accessible (this would fail before our fix)
    assert cluster.cluster_id == 1
    assert cluster.media_files == sample_media_files
    assert cluster.temporal_info == sample_temporal_cluster
    assert cluster.dominant_location == "Test Location"  # Not primary_location
    assert cluster.people_detected == ["Elena Rodriguez"]
    assert cluster.confidence_score == 0.8

    print("✅ MediaCluster constructor works with correct attributes")


@pytest.mark.unit
def test_media_cluster_enhancement_creation_process(sample_media_files, sample_temporal_cluster):
    """Test the specific MediaCluster enhancement process that was failing."""
    print("🧪 Testing MediaCluster enhancement creation process")

    # Create original cluster
    original_cluster = MediaCluster(
        cluster_id=1,
        media_files=sample_media_files,
        temporal_info=sample_temporal_cluster
    )

    # Create enhanced cluster (this exact pattern was failing before our fix)
    enhanced_cluster = MediaCluster(
        cluster_id=original_cluster.cluster_id,
        media_files=original_cluster.media_files,
        temporal_info=original_cluster.temporal_info,  # Required (was missing)
        location_info=original_cluster.location_info,
        dominant_location=original_cluster.dominant_location,  # Was primary_location (wrong)
        gps_coordinates=original_cluster.gps_coordinates,
        content_tags=original_cluster.content_tags,
        people_detected=["Elena Rodriguez"],  # Enhancement: add people
        confidence_score=original_cluster.confidence_score,
        suggested_name=original_cluster.suggested_name,
        metadata=original_cluster.metadata.copy() if original_cluster.metadata else {}
    )

    # Verify enhancement creation succeeded
    assert enhanced_cluster.cluster_id == original_cluster.cluster_id
    assert enhanced_cluster.people_detected == ["Elena Rodriguez"]
    assert enhanced_cluster.temporal_info == original_cluster.temporal_info

    print("✅ MediaCluster enhancement creation process works (bug fix validated)")


# ===== ISSUE #13 REGRESSION PREVENTION =====

@pytest.mark.unit
@pytest.mark.regression
def test_issue_13_core_fix_validation():
    """Test the core Issue #13 fix: MediaClusteringEngine receives face recognition parameters."""
    print("🧪 REGRESSION TEST: Issue #13 core fix validation")

    # This test validates the core Issue #13 fix:
    # MediaClusteringEngine should receive face_recognizer and people_database parameters
    # Before the fix, these parameters were not passed to the constructor

    # Mock components
    mock_face_recognizer = Mock(spec=FaceRecognizer)
    mock_people_database = Mock(spec=PeopleDatabase)

    # The bug was that MediaClusteringEngine was initialized without these parameters
    # This should work after our fix
    engine = MediaClusteringEngine(
        face_recognizer=mock_face_recognizer,  # This parameter was missing before fix
        people_database=mock_people_database    # This parameter was missing before fix
    )

    # Verify the Issue #13 fix
    assert engine.face_recognizer is not None, \
        "REGRESSION: MediaClusteringEngine should receive face_recognizer (Issue #13)"
    assert engine.people_database is not None, \
        "REGRESSION: MediaClusteringEngine should receive people_database (Issue #13)"
    assert engine.face_recognizer == mock_face_recognizer, \
        "REGRESSION: Face recognizer should be properly assigned (Issue #13)"
    assert engine.people_database == mock_people_database, \
        "REGRESSION: People database should be properly assigned (Issue #13)"

    print("✅ REGRESSION TEST PASSED: Issue #13 core fix validated")


if __name__ == "__main__":
    """Run the face recognition unit tests standalone."""
    print("🧪 Face Recognition Clustering Unit Tests")
    print("📋 Test Categories:")
    print("   1. Component initialization tests (mocked dependencies)")
    print("   2. People database unit tests (temp files)")
    print("   3. Clustering logic tests (mocked face detection)")
    print("   4. MediaCluster constructor tests (bug fix validation)")
    print("   5. Issue #13 regression prevention (core logic)")
    print()
    print("⚡ Expected time: <3 seconds total")
    print("🔧 Testing approach: Fast unit tests with mocked components")
    print()

    # Run with pytest
    pytest.main([__file__, "-v"])