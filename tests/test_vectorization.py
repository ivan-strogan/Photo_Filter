#!/usr/bin/env python3
"""Test photo vectorization pipeline."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.media_detector import MediaDetector
from src.photo_vectorizer import PhotoVectorizer
from src.vector_database import VectorDatabase
from src.metadata_extractor import MetadataExtractor

def test_photo_vectorization():
    """Test complete photo vectorization workflow."""
    print("=== Testing Photo Vectorization ===")

    # Initialize components
    detector = MediaDetector()
    extractor = MetadataExtractor()
    vectorizer = PhotoVectorizer()
    db = VectorDatabase()

    print(f"Vectorizer info: {vectorizer.get_model_info()}")

    # Get sample photos from iPhone Automatic
    print("\nScanning for photos...")
    iphone_files = detector.scan_iphone_automatic()
    photo_files = [f for f in iphone_files if f.file_type == 'photo']

    if not photo_files:
        print("No photos found for testing!")
        return

    # Test with first 5 photos for speed
    test_photos = photo_files[:5]
    print(f"Testing with {len(test_photos)} photos")

    # Extract metadata for each photo
    print("\nExtracting metadata...")
    for i, photo in enumerate(test_photos):
        metadata = extractor.extract_photo_metadata(photo)
        test_photos[i].metadata = metadata
        print(f"  {photo.filename}: {metadata.get('width', 'N/A')}x{metadata.get('height', 'N/A')}")

    # Vectorize photos
    print("\nVectorizing photos...")
    vectorization_results = vectorizer.vectorize_media_files(test_photos)

    success_count = sum(1 for _, emb in vectorization_results if emb is not None)
    print(f"Successfully vectorized {success_count}/{len(test_photos)} photos")

    # Add to database
    print("\nAdding to vector database...")
    for i, (photo_id, embedding) in enumerate(vectorization_results):
        if embedding is not None:
            photo = test_photos[i]
            success = db.add_photo_embedding(
                photo_id=photo_id,
                embedding=embedding,
                metadata=photo.metadata
            )
            print(f"  Added {photo_id}: {success}")

    # Test similarity search
    if vectorization_results and vectorization_results[0][1] is not None:
        print("\nTesting similarity search...")
        query_embedding = vectorization_results[0][1]
        similar_photos = db.search_similar_photos(
            query_embedding=query_embedding,
            n_results=3,
            filter_organized=False
        )

        print(f"Found {len(similar_photos)} similar photos:")
        for photo in similar_photos:
            print(f"  {photo['photo_id']}: distance={photo['distance']:.3f}")

    # Cleanup
    vectorizer.cleanup()
    print("\nVectorization test completed!")

if __name__ == "__main__":
    test_photo_vectorization()