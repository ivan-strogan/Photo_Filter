"""Scanner for building vector database from existing organized photos."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .media_detector import MediaDetector
from .metadata_extractor import MetadataExtractor
from .logging_utils import ProgressTracker

# Import ML components only when needed
PhotoVectorizer = None
VectorDatabase = None

class OrganizedPhotosScanner:
    """Scans existing organized photos to build initial vector database."""

    def __init__(self, use_gpu: bool = True):
        """Initialize the scanner.

        Args:
            use_gpu: Whether to use GPU acceleration for vectorization
        """
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.media_detector = MediaDetector()
        self.metadata_extractor = MetadataExtractor()
        self.vector_db = None  # Initialize when needed

        # Initialize vectorizer only when needed (GPU dependencies)
        self.vectorizer = None
        self.use_gpu = use_gpu

    def _initialize_vectorizer(self):
        """Lazy initialization of vectorizer (requires ML dependencies)."""
        if self.vectorizer is None:
            try:
                global PhotoVectorizer
                if PhotoVectorizer is None:
                    from .photo_vectorizer import PhotoVectorizer

                self.vectorizer = PhotoVectorizer()
                self.logger.info("Photo vectorizer initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize vectorizer: {e}")
                self.logger.warning("Vector similarity features will not be available")
                return False
        return True

    def _initialize_vector_db(self):
        """Lazy initialization of vector database."""
        if self.vector_db is None:
            try:
                global VectorDatabase
                if VectorDatabase is None:
                    from .vector_database import VectorDatabase

                self.vector_db = VectorDatabase()
                self.logger.info("Vector database initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize vector database: {e}")
                return False
        return True

    def scan_and_build_database(self,
                               max_photos_per_event: int = 50,
                               skip_large_events: bool = True,
                               max_events: int = 5,
                               quick_scan: bool = True) -> Dict[str, Any]:
        """Scan organized photos and build vector database.

        Args:
            max_photos_per_event: Maximum photos to process per event folder
            skip_large_events: Skip events with too many photos (processing efficiency)
            max_events: Maximum number of event folders to process (for quick initial scan)
            quick_scan: If True, only process a subset of events for faster initialization

        Returns:
            Dictionary with scan results
        """
        self.logger.info("Starting organized photos scan to build vector database")

        # Get all organized photos
        organized_files = self.media_detector.scan_pictures_library()
        photo_files = [f for f in organized_files if f.file_type == 'photo']

        self.logger.info(f"Found {len(photo_files)} organized photos")

        # Group by event folder
        events = self._group_by_event_folder(photo_files)

        self.logger.info(f"Found {len(events)} event folders")

        # Filter events for processing
        filtered_events = self._filter_events_for_processing(
            events, max_photos_per_event, skip_large_events
        )

        # Apply max_events limit if specified
        if max_events is not None and len(filtered_events) > max_events:
            # Take the first max_events (could also sort by size, date, etc.)
            filtered_events = dict(list(filtered_events.items())[:max_events])
            self.logger.info(f"Limited to {max_events} events for quick scan")

        self.logger.info(f"Processing {len(filtered_events)} event folders")

        # Process each event
        results = {
            'total_events_found': len(events),
            'events_processed': len(filtered_events),
            'photos_processed': 0,
            'photos_vectorized': 0,
            'photos_added_to_db': 0,
            'events_details': [],
            'skipped_events': []
        }

        for event_name, event_photos in filtered_events.items():
            self.logger.info(f"📁 Processing event: {event_name} ({len(event_photos)} photos)")

            event_result = self._process_event(event_name, event_photos)
            results['events_details'].append(event_result)

            results['photos_processed'] += event_result['photos_processed']
            results['photos_vectorized'] += event_result['photos_vectorized']
            results['photos_added_to_db'] += event_result['photos_added_to_db']

            # Log completion status
            success_rate = event_result['photos_vectorized'] / max(1, event_result['photos_processed'])
            self.logger.info(f"✅ Completed event: {event_name} - {event_result['photos_vectorized']}/{event_result['photos_processed']} photos vectorized ({success_rate:.1%} success)")

        # Add skipped events info
        for event_name, event_photos in events.items():
            if event_name not in filtered_events:
                results['skipped_events'].append({
                    'event_name': event_name,
                    'photo_count': len(event_photos),
                    'reason': 'too_large' if len(event_photos) > max_photos_per_event else 'filtered'
                })

        self.logger.info(f"Scan completed: {results['photos_added_to_db']} photos added to vector database")
        return results

    def _group_by_event_folder(self, photo_files: List) -> Dict[str, List]:
        """Group photos by their event folder."""
        events = {}

        for photo in photo_files:
            event_folder = getattr(photo, 'event_folder', 'Unknown')

            # Skip generic folders that don't represent events
            if self._should_skip_folder(event_folder):
                continue

            if event_folder not in events:
                events[event_folder] = []
            events[event_folder].append(photo)

        return events

    def _should_skip_folder(self, folder_name: str) -> bool:
        """Check if folder should be skipped during processing."""
        skip_keywords = [
            'filter', 'sort', 'organize', 'temp', 'temporary',
            'misc', 'random', 'unsorted', 'unknown', 'need to'
        ]

        folder_lower = folder_name.lower()
        return any(keyword in folder_lower for keyword in skip_keywords)

    def _filter_events_for_processing(self,
                                    events: Dict[str, List],
                                    max_photos_per_event: int,
                                    skip_large_events: bool) -> Dict[str, List]:
        """Filter events for efficient processing."""
        filtered = {}

        for event_name, photos in events.items():
            if skip_large_events and len(photos) > max_photos_per_event:
                # Take a sample from large events
                sample_size = min(max_photos_per_event, len(photos))
                # Take photos from beginning, middle, and end for diversity
                step = len(photos) // sample_size
                sampled_photos = photos[::max(1, step)][:sample_size]
                filtered[event_name] = sampled_photos

                self.logger.info(f"Sampling {len(sampled_photos)} photos from {event_name} (total: {len(photos)})")
            else:
                filtered[event_name] = photos

        return filtered

    def _process_event(self, event_name: str, event_photos: List) -> Dict[str, Any]:
        """Process a single event folder."""
        result = {
            'event_name': event_name,
            'photos_processed': len(event_photos),
            'photos_vectorized': 0,
            'photos_added_to_db': 0,
            'errors': []
        }

        # Initialize vectorizer and database if needed
        if not self._initialize_vectorizer():
            result['errors'].append("Vectorizer not available")
            return result

        if not self._initialize_vector_db():
            result['errors'].append("Vector database not available")
            return result

        try:
            # Create progress tracker
            progress = ProgressTracker(len(event_photos), f"Processing {event_name}")

            # Process photos in batches
            batch_size = 10  # Small batches for memory efficiency

            for i in range(0, len(event_photos), batch_size):
                batch = event_photos[i:i + batch_size]

                # Filter out photos that already exist in database
                photos_to_vectorize = []
                skipped_count = 0

                print(f"🔍 DEBUG: Processing batch of {len(batch)} photos")
                for photo in batch:
                    # Use same ID format as vectorizer (avoid redundant timestamps)
                    photo_id = self._create_file_id(photo)
                    print(f"🔍 DEBUG: Checking photo_id='{photo_id}' filename='{photo.filename}' (was using path: {str(photo.path)})")

                    photo_exists = self.vector_db.photo_exists(photo_id)
                    print(f"🔍 DEBUG: vector_db.photo_exists('{photo_id}') returned: {photo_exists}")

                    if photo_exists:
                        print(f"⏭️  CACHE HIT: Skipping already vectorized: {photo.filename}")
                        self.logger.info(f"⏭️  Skipping already vectorized: {photo.filename}")
                        result['photos_vectorized'] += 1  # Count as processed
                        skipped_count += 1
                    else:
                        print(f"🆕 DEBUG: Photo NOT in database, will vectorize: {photo.filename}")
                        photos_to_vectorize.append(photo)

                # Only vectorize photos that don't exist in database
                print(f"🔍 DEBUG: After filtering - {len(photos_to_vectorize)} photos need vectorization, {skipped_count} skipped")
                if photos_to_vectorize:
                    print(f"🚀 DEBUG: Starting vectorization of {len(photos_to_vectorize)} photos...")
                    vectorization_results = self.vectorizer.vectorize_media_files(photos_to_vectorize)
                    print(f"✅ DEBUG: Vectorization completed, got {len(vectorization_results)} results")
                else:
                    print(f"⏭️  DEBUG: All photos were cached, no vectorization needed")

                # Process each new photo
                if photos_to_vectorize:
                    print(f"💾 DEBUG: Processing {len(vectorization_results)} vectorization results...")
                    for j, (photo_id, embedding) in enumerate(vectorization_results):
                        if embedding is not None:
                            photo = photos_to_vectorize[j]
                            print(f"💾 DEBUG: Adding photo {j+1}/{len(vectorization_results)}: {photo.filename} (id: {photo_id[:50]}...)")

                            # Extract metadata
                            metadata = self.metadata_extractor.extract_photo_metadata(photo)

                            # Add to vector database
                            print(f"💾 DEBUG: Calling vector_db.add_photo_embedding for {photo.filename}")
                            success = self.vector_db.add_photo_embedding(
                                photo_id=photo_id,
                                embedding=embedding,
                                metadata=metadata,
                                event_folder=event_name
                            )
                            print(f"💾 DEBUG: add_photo_embedding returned: {success} for {photo.filename}")

                            # Verify it was actually added
                            exists_after_add = self.vector_db.photo_exists(photo_id)
                            print(f"💾 DEBUG: After adding, photo_exists check: {exists_after_add} for {photo.filename}")

                            if success:
                                result['photos_vectorized'] += 1
                                result['photos_added_to_db'] += 1
                                self.logger.info(f"✅ Added new photo to database: {photo.filename}")
                            else:
                                result['errors'].append(f"Failed to add {photo_id} to database")

                if skipped_count > 0:
                    self.logger.info(f"📊 Batch summary: {skipped_count} skipped, {len(photos_to_vectorize)} new photos processed")

                # Update progress
                progress.update(len(batch))

            progress.close()

        except Exception as e:
            error_msg = f"Error processing event {event_name}: {e}"
            self.logger.error(error_msg)
            result['errors'].append(error_msg)

        return result

    def get_database_summary(self) -> Dict[str, Any]:
        """Get summary of current vector database state."""
        try:
            if not self._initialize_vector_db():
                return {'error': 'Vector database not available'}

            stats = self.vector_db.get_database_stats()
            event_folders = self.vector_db.get_all_event_folders()

            # Get photos per event
            event_counts = {}
            for event in event_folders:
                photos = self.vector_db.get_organized_photos_by_event(event)
                event_counts[event] = len(photos)

            return {
                'database_stats': stats,
                'event_folders': event_folders,
                'event_photo_counts': event_counts,
                'top_events': sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            }

        except Exception as e:
            self.logger.error(f"Error getting database summary: {e}")
            return {'error': str(e)}

    def search_similar_to_organized_photo(self,
                                        event_name: str,
                                        n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for photos similar to those in an organized event.

        Args:
            event_name: Name of the event to use as reference
            n_results: Number of similar photos to return

        Returns:
            List of similar photos with metadata
        """
        if not self._initialize_vectorizer():
            return []

        try:
            # Get photos from the event
            event_photos = self.vector_db.get_organized_photos_by_event(event_name)

            if not event_photos:
                self.logger.warning(f"No photos found for event: {event_name}")
                return []

            # Use the first photo as reference
            reference_photo = event_photos[0]
            reference_embedding = reference_photo.get('embedding')

            if reference_embedding is None:
                self.logger.warning(f"No embedding found for reference photo in {event_name}")
                return []

            # Search for similar photos
            similar_photos = self.vector_db.search_similar_photos(
                query_embedding=reference_embedding,
                n_results=n_results,
                filter_organized=True
            )

            return similar_photos

        except Exception as e:
            self.logger.error(f"Error searching similar photos: {e}")
            return []

    def _create_file_id(self, media_file) -> str:
        """Create a unique ID for a media file (matching photo_vectorizer logic).

        Args:
            media_file: MediaFile object

        Returns:
            Unique file ID string
        """
        filename = media_file.filename

        # Check if filename already contains timestamp (iPhone format: IMG_YYYYMMDD_HHMMSS.JPG)
        import re
        iphone_pattern = r'^(IMG|MOV)_\d{8}_\d{6}\.(JPG|MOV|jpg|mov)$'

        if re.match(iphone_pattern, filename):
            # Filename already has timestamp, use it directly
            print(f"📱 SCANNER: Using iPhone filename as photo_id: {filename}")
            return filename
        else:
            # Non-iPhone filename, add timestamp prefix
            timestamp = media_file.time.strftime("%Y%m%d_%H%M%S")
            photo_id = f"{timestamp}_{filename}"
            print(f"📷 SCANNER: Created photo_id with timestamp: {photo_id}")
            return photo_id

    def cleanup(self):
        """Clean up resources."""
        if self.vectorizer:
            self.vectorizer.cleanup()