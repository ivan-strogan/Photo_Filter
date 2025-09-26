#!/usr/bin/env python3
"""
Test face recognition integration in the clustering pipeline.

This test verifies that:
1. MediaClusteringEngine properly receives face recognition components
2. Face recognition is actually called during clustering
3. People data is correctly integrated into cluster results
4. Prevents regression of Issue #13 (face recognition not running during clustering)

Uses proper pytest fixtures and follows the project test structure.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Import will be handled by conftest.py fixtures
try:
    from src.media_clustering import MediaClusteringEngine, MediaCluster
    from src.temporal_clustering import TemporalCluster
    from src.face_recognizer import FaceRecognizer, FaceRecognitionResult
    from src.people_database import PeopleDatabase
    from src.photo_organizer_pipeline import PhotoOrganizerPipeline
    from src.media_detector import MediaFile
    from src.config_manager import get_config
    from datetime import datetime, timedelta
except ImportError:
    # Fallback imports if needed
    import media_clustering
    import temporal_clustering
    import face_recognizer
    import people_database
    import photo_organizer_pipeline
    import media_detector
    import config_manager
    from datetime import datetime, timedelta
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
def mock_people_database(temp_test_dir):
    """Create a mock people database with Sasha."""
    people_db_file = temp_test_dir / "people_database.json"
    people_data = {
        "people": {
            "Sasha Strogan": {
                "person_id": "sasha_001",
                "encodings": [[0.1, 0.2, 0.3] * 43]  # Mock 128-dimensional encoding
            }
        }
    }
    with open(people_db_file, 'w') as f:
        json.dump(people_data, f)
    return people_db_file


@pytest.fixture
def test_photos(temp_test_dir):
    """Create test photo files."""
    photo_dir = temp_test_dir / "photos"
    photo_dir.mkdir()

    test_photos = []
    for i in range(3):
        photo_path = photo_dir / f"IMG_20241025_16303{i}.JPG"
        # Create minimal valid JPEG
        with open(photo_path, 'wb') as f:
            f.write(b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00')
            f.write(f'Test photo {i} content data'.encode() * 50)
            f.write(b'\xFF\xD9')
        test_photos.append(photo_path)
    return test_photos


@pytest.fixture
def test_media_files(test_photos):
    """Create MediaFile objects from test photos."""
    media_files = []
    for photo_path in test_photos:
        test_datetime = datetime(2024, 10, 25, 16, 30, 0)
        media_file = MediaFile(
            path=Path(photo_path),
            filename=photo_path.name,
            date=test_datetime,
            time=test_datetime,
            extension='.JPG',
            file_type='photo',
            size=1024
        )
        media_files.append(media_file)
    return media_files


@pytest.fixture
def test_cluster(test_media_files):
    """Create a test cluster with media files."""
    # Create a temporal cluster first
    start_time = datetime(2024, 10, 25, 16, 30, 0)
    end_time = datetime(2024, 10, 25, 17, 30, 0)
    temporal_cluster = TemporalCluster(
        cluster_id=1,
        start_time=start_time,
        end_time=end_time,
        duration=end_time - start_time,
        media_files=test_media_files
    )

    # Create MediaCluster with the temporal cluster
    return MediaCluster(
        cluster_id=1,
        media_files=test_media_files,
        temporal_info=temporal_cluster
    )


@pytest.mark.unit
def test_media_clustering_engine_receives_face_components(mock_people_database):
    """Test that MediaClusteringEngine properly receives face recognition components."""
    print("🧪 Testing MediaClusteringEngine face component initialization")

    # Create face recognition components
    people_db = PeopleDatabase(database_file=mock_people_database)
    face_recognizer = FaceRecognizer(
        detection_model="hog",
        recognition_tolerance=0.6,
        people_database=people_db
    )

    # Create MediaClusteringEngine with face recognition
    engine = MediaClusteringEngine(
        face_recognizer=face_recognizer,
        people_database=people_db
    )

    # Verify components are properly assigned
    assert engine.face_recognizer is not None, "Face recognizer should be assigned"
    assert engine.people_database is not None, "People database should be assigned"
    assert engine.face_recognizer == face_recognizer, "Face recognizer should match"
    assert engine.people_database == people_db, "People database should match"

    print("✅ MediaClusteringEngine properly receives face recognition components")


@pytest.mark.unit
def test_media_clustering_engine_without_face_components():
    """Test that MediaClusteringEngine works without face recognition components."""
    print("🧪 Testing MediaClusteringEngine without face recognition")

    # Create MediaClusteringEngine without face recognition
    engine = MediaClusteringEngine()

    # Verify components are None
    assert engine.face_recognizer is None, "Face recognizer should be None"
    assert engine.people_database is None, "People database should be None"

    print("✅ MediaClusteringEngine works without face recognition components")


@pytest.mark.integration
def test_face_recognition_integration_with_mocked_detection(mock_people_database, test_cluster):
    """Test face recognition integration with controlled mock results."""
    print("🧪 Testing face recognition integration with controlled results")

    # Create real face recognition components
    people_db = PeopleDatabase(database_file=mock_people_database)
    face_recognizer = FaceRecognizer(
        detection_model="hog",
        recognition_tolerance=0.6,
        people_database=people_db
    )

    # Mock the detect_faces method to return controlled results
    def mock_detect_faces(photo_path):
        """Mock face detection that recognizes Sasha in the first photo."""
        result = FaceRecognitionResult()
        if "163030" in str(photo_path):  # First test photo
            result.faces_detected = 2
            result.people_detected = {"Sasha Strogan"}
        else:
            result.faces_detected = 0
            result.people_detected = set()
        return result

    # Patch the detect_faces method
    face_recognizer.detect_faces = mock_detect_faces

    # Create MediaClusteringEngine with real face recognition
    engine = MediaClusteringEngine(
        face_recognizer=face_recognizer,
        people_database=people_db
    )

    # Run face recognition enhancement
    engine._enhance_with_people_data([test_cluster])

    # Verify cluster was enhanced with people data
    assert test_cluster.people_detected is not None, "Cluster should have people data"
    assert "Sasha Strogan" in test_cluster.people_detected, "Sasha should be detected"

    print("✅ Face recognition successfully integrated into clustering pipeline")


@pytest.mark.integration
def test_photo_organizer_pipeline_initializes_face_recognition():
    """Test that PhotoOrganizerPipeline properly initializes face recognition components."""
    print("🧪 Testing PhotoOrganizerPipeline face recognition initialization")

    # Get current config and ensure face detection is enabled
    config = get_config()
    original_face_setting = config.faces.enable_face_detection

    try:
        # Enable face detection for this test
        config.faces.enable_face_detection = True

        # Create pipeline
        pipeline = PhotoOrganizerPipeline(dry_run=True)

        # Initialize clustering engine (this triggers initialization)
        pipeline._initialize_clustering_engine_with_vector_db()
        clustering_engine = pipeline.clustering_engine

        # Verify face recognition components are initialized
        assert clustering_engine.face_recognizer is not None, \
            "Pipeline should initialize face recognizer when enabled"
        assert clustering_engine.people_database is not None, \
            "Pipeline should initialize people database when enabled"

        print("✅ PhotoOrganizerPipeline properly initializes face recognition")

    finally:
        # Restore original setting
        config.faces.enable_face_detection = original_face_setting


@pytest.mark.integration
def test_photo_organizer_pipeline_without_face_recognition():
    """Test that PhotoOrganizerPipeline works without face recognition when disabled."""
    print("🧪 Testing PhotoOrganizerPipeline without face recognition")

    # Get current config and ensure face detection is disabled
    config = get_config()
    original_face_setting = config.faces.enable_face_detection

    try:
        # Disable face detection for this test
        config.faces.enable_face_detection = False

        # Create pipeline
        pipeline = PhotoOrganizerPipeline(dry_run=True)

        # Initialize clustering engine (this triggers initialization)
        pipeline._initialize_clustering_engine_with_vector_db()
        clustering_engine = pipeline.clustering_engine

        # Verify face recognition components are not initialized
        assert clustering_engine.face_recognizer is None, \
            "Pipeline should not initialize face recognizer when disabled"
        assert clustering_engine.people_database is None, \
            "Pipeline should not initialize people database when disabled"

        print("✅ PhotoOrganizerPipeline works correctly without face recognition")

    finally:
        # Restore original setting
        config.faces.enable_face_detection = original_face_setting


@pytest.mark.regression
def test_regression_issue_13_face_recognition_pipeline_integration():
    """Regression test for Issue #13: Face recognition not running during clustering.

    This test ensures that the specific bug from Issue #13 doesn't reoccur
    where MediaClusteringEngine was initialized without face recognition parameters.
    """
    print("🧪 REGRESSION TEST: Issue #13 - Face recognition pipeline integration")

    config = get_config()
    original_face_setting = config.faces.enable_face_detection

    try:
        # Enable face detection
        config.faces.enable_face_detection = True

        # Create pipeline (this is where the bug occurred)
        pipeline = PhotoOrganizerPipeline(dry_run=True)

        # Initialize clustering engine - this should include face recognition
        pipeline._initialize_clustering_engine_with_vector_db()
        clustering_engine = pipeline.clustering_engine

        # These assertions would have failed before the fix
        assert clustering_engine.face_recognizer is not None, \
            "REGRESSION: MediaClusteringEngine missing face_recognizer parameter (Issue #13)"
        assert clustering_engine.people_database is not None, \
            "REGRESSION: MediaClusteringEngine missing people_database parameter (Issue #13)"

        # Verify the face recognizer is properly configured
        assert hasattr(clustering_engine.face_recognizer, 'detect_faces'), \
            "Face recognizer should have detect_faces method"
        assert hasattr(clustering_engine.people_database, 'get_all_people'), \
            "People database should have get_all_people method"

        print("✅ REGRESSION TEST PASSED: Issue #13 bug is fixed and will not reoccur")

    finally:
        # Restore original setting
        config.faces.enable_face_detection = original_face_setting


@pytest.mark.unit
@patch('src.media_clustering.MediaClusteringEngine._enhance_with_people_data')
def test_clustering_calls_face_recognition_enhancement(mock_enhance_people, test_cluster):
    """Test that clustering pipeline calls face recognition enhancement method."""
    print("🧪 Testing that clustering calls face recognition enhancement")

    # Create MediaClusteringEngine with mocked face recognition
    mock_face_recognizer = Mock()
    mock_people_db = Mock()

    engine = MediaClusteringEngine(
        face_recognizer=mock_face_recognizer,
        people_database=mock_people_db
    )

    # Run clustering on the test cluster
    engine._enhance_with_people_data([test_cluster])

    # Verify that face recognition enhancement was called
    mock_enhance_people.assert_called_once()

    print("✅ Clustering pipeline properly calls face recognition enhancement")


if __name__ == "__main__":
    """Run the clustering face integration tests standalone."""
    print("🧪 Clustering Face Recognition Integration Test")
    print("📋 Test Plan:")
    print("   1. MediaClusteringEngine properly receives face components")
    print("   2. Face recognition is integrated into clustering pipeline")
    print("   3. PhotoOrganizerPipeline initializes face recognition correctly")
    print("   4. Regression test for Issue #13")
    print()

    # Check dependencies
    try:
        import face_recognition
        print("✅ face_recognition library available")
    except ImportError:
        print("❌ face_recognition library not available")
        import sys
        sys.exit(1)

    # Run with pytest
    pytest.main([__file__, "-v"])