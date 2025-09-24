#!/usr/bin/env python3
"""End-to-end system test."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.media_detector import MediaDetector
from src.metadata_extractor import MetadataExtractor
from src.photo_vectorizer import PhotoVectorizer
from src.vector_database import VectorDatabase

def test_organized_photos_workflow():
    """Test workflow for scanning organized photos (Pictures folder)."""
    print("=== Testing Organized Photos Workflow ===")

    detector = MediaDetector()
    extractor = MetadataExtractor()
    vectorizer = PhotoVectorizer()
    db = VectorDatabase()

    # Scan Pictures library
    print("Scanning Pictures library...")
    organized_files = detector.scan_pictures_library()
    photo_files = [f for f in organized_files if f.file_type == 'photo']

    print(f"Found {len(photo_files)} organized photos")

    # Group by event folder
    events = {}
    for photo in photo_files:
        event = getattr(photo, 'event_folder', 'Unknown')
        if event not in events:
            events[event] = []
        events[event].append(photo)

    print(f"Found {len(events)} event folders:")
    for event, photos in events.items():
        print(f"  {event}: {len(photos)} photos")

    # Test with one event (limit for testing)
    if events:
        test_event = list(events.keys())[0]
        test_photos = events[test_event][:3]  # First 3 photos

        print(f"\nTesting with event: {test_event}")

        # Extract metadata and vectorize
        print("Processing photos...")
        for photo in test_photos:
            metadata = extractor.extract_photo_metadata(photo)
            print(f"  {photo.filename}: extracted metadata")

        # Vectorize
        vectorization_results = vectorizer.vectorize_media_files(test_photos)

        # Add to database with event folder info
        for i, (photo_id, embedding) in enumerate(vectorization_results):
            if embedding is not None:
                photo = test_photos[i]
                metadata = extractor.extract_photo_metadata(photo)
                success = db.add_photo_embedding(
                    photo_id=photo_id,
                    embedding=embedding,
                    metadata=metadata,
                    event_folder=test_event
                )
                print(f"  Added to DB: {photo_id} -> {test_event}")

    vectorizer.cleanup()

def test_unorganized_photos_workflow():
    """Test workflow for processing unorganized photos (iPhone Automatic)."""
    print("\n=== Testing Unorganized Photos Workflow ===")

    detector = MediaDetector()
    extractor = MetadataExtractor()
    vectorizer = PhotoVectorizer()
    db = VectorDatabase()

    # Scan iPhone Automatic folder
    print("Scanning iPhone Automatic folder...")
    unorganized_files = detector.scan_iphone_automatic()
    photo_files = [f for f in unorganized_files if f.file_type == 'photo']

    print(f"Found {len(photo_files)} unorganized photos")

    if photo_files:
        # Test with first few photos
        test_photos = photo_files[:3]

        print(f"Testing with {len(test_photos)} photos")

        # Process each photo
        for photo in test_photos:
            print(f"\nProcessing: {photo.filename}")

            # Extract metadata
            metadata = extractor.extract_photo_metadata(photo)
            gps = metadata.get('gps_coordinates')
            camera = metadata.get('camera_make')

            print(f"  GPS: {gps}")
            print(f"  Camera: {camera}")
            print(f"  Date: {photo.time}")

            # Vectorize
            vectorization_results = vectorizer.vectorize_media_files([photo])

            if vectorization_results and vectorization_results[0][1] is not None:
                photo_id, embedding = vectorization_results[0]

                # Search for similar organized photos
                similar_photos = db.search_similar_photos(
                    query_embedding=embedding,
                    n_results=5,
                    filter_organized=True
                )

                print(f"  Found {len(similar_photos)} similar organized photos:")
                for sim_photo in similar_photos:
                    event = sim_photo['metadata'].get('event_folder', 'Unknown')
                    distance = sim_photo['distance']
                    print(f"    {event}: similarity={1-distance:.3f}")

    vectorizer.cleanup()

def test_full_system():
    """Test the complete system workflow."""
    print("=== Full System Test ===")

    # Test database stats
    db = VectorDatabase()
    stats = db.get_database_stats()
    print(f"Database stats: {stats}")

    # Test organized photos workflow
    test_organized_photos_workflow()

    # Test unorganized photos workflow
    test_unorganized_photos_workflow()

    print("\n=== Full System Test Completed ===")

if __name__ == "__main__":
    test_full_system()