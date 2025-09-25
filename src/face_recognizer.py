"""
Face detection and recognition system for Photo Filter AI.

This module provides face detection, encoding, and recognition capabilities
to identify people in photos and enhance clustering and event naming.

For junior developers:
- Uses the face_recognition library for detection and encoding
- Implements caching to avoid re-processing faces
- Provides both single photo and batch processing capabilities
- Includes error handling for corrupted or problematic images
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import numpy as np
from PIL import Image
import hashlib
import warnings

# Suppress deprecation warning from face_recognition_models package
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

# Optional face recognition dependencies
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

from .config import DATA_DIR

@dataclass
class Face:
    """Represents a detected face with location and encoding."""
    top: int                    # Top pixel coordinate
    right: int                  # Right pixel coordinate
    bottom: int                 # Bottom pixel coordinate
    left: int                   # Left pixel coordinate
    encoding: Optional[np.ndarray] = None  # 128-dimensional face encoding
    confidence: float = 0.0     # Detection confidence score
    person_id: Optional[str] = None        # Identified person (if known)

    def to_dict(self) -> Dict[str, Any]:
        """Convert face to dictionary for JSON serialization."""
        return {
            'top': self.top,
            'right': self.right,
            'bottom': self.bottom,
            'left': self.left,
            'encoding': self.encoding.tolist() if self.encoding is not None else None,
            'confidence': self.confidence,
            'person_id': self.person_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Face':
        """Create Face from dictionary."""
        encoding = np.array(data['encoding']) if data.get('encoding') else None
        return cls(
            top=data['top'],
            right=data['right'],
            bottom=data['bottom'],
            left=data['left'],
            encoding=encoding,
            confidence=data.get('confidence', 0.0),
            person_id=data.get('person_id')
        )

@dataclass
class FaceRecognitionResult:
    """Result of face recognition analysis."""
    image_path: str
    faces_detected: int
    faces: List[Face]
    processing_time: float
    error: Optional[str] = None

    def get_people_detected(self) -> List[str]:
        """Get list of identified people in the photo."""
        people = []
        for face in self.faces:
            if face.person_id and face.person_id not in people:
                people.append(face.person_id)
        return people

class FaceRecognizer:
    """Face detection and recognition system."""

    def __init__(self,
                 detection_model: str = "hog",
                 recognition_tolerance: float = 0.6,
                 min_face_size: int = 50,
                 enable_caching: bool = True,
                 cache_file: Optional[Path] = None,
                 people_database=None):
        """Initialize face recognizer.

        Args:
            detection_model: "hog" (fast) or "cnn" (accurate, requires GPU)
            recognition_tolerance: Lower values = stricter matching (0.0-1.0)
            min_face_size: Minimum face size in pixels to detect
            enable_caching: Whether to cache face encodings
            cache_file: Path to cache file (defaults to DATA_DIR/face_cache.json)
            people_database: PeopleDatabase instance to use for persistence
        """
        self.logger = logging.getLogger(__name__)

        if not FACE_RECOGNITION_AVAILABLE:
            self.logger.warning("face_recognition library not available. Face recognition disabled.")
            self.enabled = False
            return

        self.enabled = True
        self.detection_model = detection_model
        self.recognition_tolerance = recognition_tolerance
        self.min_face_size = min_face_size
        self.enable_caching = enable_caching

        # Cache for face encodings
        self.face_cache = {}
        self.cache_file = cache_file or (DATA_DIR / "face_cache.json")

        # Use PeopleDatabase for persistence
        if people_database is None:
            from .people_database import PeopleDatabase
            self.people_database = PeopleDatabase()
        else:
            self.people_database = people_database

        # Load cache if it exists
        if enable_caching:
            self._load_cache()

        self.logger.info(f"Face recognizer initialized (model: {detection_model}, tolerance: {recognition_tolerance})")

    def _load_cache(self) -> None:
        """Load face encoding cache from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)

                # Convert back to Face objects
                for image_path, face_list in cache_data.items():
                    self.face_cache[image_path] = [Face.from_dict(face_data) for face_data in face_list]

                self.logger.info(f"Loaded face cache with {len(self.face_cache)} images")
        except Exception as e:
            self.logger.warning(f"Could not load face cache: {e}")

    def _save_cache(self) -> None:
        """Save face encoding cache to disk."""
        if not self.enable_caching:
            return

        try:
            # Ensure data directory exists
            DATA_DIR.mkdir(parents=True, exist_ok=True)

            # Convert Face objects to dictionaries
            cache_data = {}
            for image_path, face_list in self.face_cache.items():
                cache_data[image_path] = [face.to_dict() for face in face_list]

            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)

            self.logger.debug(f"Saved face cache with {len(self.face_cache)} images")
        except Exception as e:
            self.logger.error(f"Could not save face cache: {e}")

    def _get_image_hash(self, image_path: Path) -> str:
        """Generate hash for image file for cache key."""
        try:
            stat = image_path.stat()
            # Use file size and modification time for hash
            content = f"{image_path}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(content.encode()).hexdigest()
        except Exception:
            return str(image_path)

    def detect_faces(self, image_path: Path) -> FaceRecognitionResult:
        """Detect faces in a single image.

        Args:
            image_path: Path to image file

        Returns:
            FaceRecognitionResult with detected faces
        """
        import time
        start_time = time.time()

        if not self.enabled:
            return FaceRecognitionResult(
                image_path=str(image_path),
                faces_detected=0,
                faces=[],
                processing_time=0.0,
                error="Face recognition not available"
            )

        # Check cache first
        cache_key = self._get_image_hash(image_path)
        if self.enable_caching and cache_key in self.face_cache:
            cached_faces = self.face_cache[cache_key]
            processing_time = time.time() - start_time
            self.logger.debug(f"Using cached faces for {image_path.name}: {len(cached_faces)} faces")

            return FaceRecognitionResult(
                image_path=str(image_path),
                faces_detected=len(cached_faces),
                faces=cached_faces,
                processing_time=processing_time
            )

        try:
            self.logger.info(f"🔍 Detecting faces in: {image_path.name}")

            # Load image
            image = face_recognition.load_image_file(str(image_path))

            # Detect face locations
            face_locations = face_recognition.face_locations(
                image,
                model=self.detection_model
            )

            faces = []
            if face_locations:
                # Extract face encodings
                face_encodings = face_recognition.face_encodings(image, face_locations)

                for i, (top, right, bottom, left) in enumerate(face_locations):
                    # Check if face is large enough
                    face_width = right - left
                    face_height = bottom - top

                    if face_width >= self.min_face_size and face_height >= self.min_face_size:
                        encoding = face_encodings[i] if i < len(face_encodings) else None

                        face = Face(
                            top=top,
                            right=right,
                            bottom=bottom,
                            left=left,
                            encoding=encoding,
                            confidence=1.0  # face_recognition doesn't provide confidence scores
                        )

                        # Try to identify person if we have known people
                        if encoding is not None:
                            identification_result = self._identify_person(encoding)
                            if identification_result:
                                # Extract just the person_id from the tuple (person_id, confidence)
                                face.person_id = identification_result[0] if isinstance(identification_result, tuple) else identification_result

                        faces.append(face)

            # Cache the results
            if self.enable_caching:
                self.face_cache[cache_key] = faces
                self._save_cache()

            processing_time = time.time() - start_time

            if faces:
                self.logger.info(f"✅ Detected {len(faces)} faces in {image_path.name} ({processing_time:.2f}s)")
            else:
                self.logger.debug(f"No faces detected in {image_path.name}")

            return FaceRecognitionResult(
                image_path=str(image_path),
                faces_detected=len(faces),
                faces=faces,
                processing_time=processing_time
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error detecting faces in {image_path.name}: {e}"
            self.logger.error(error_msg)

            return FaceRecognitionResult(
                image_path=str(image_path),
                faces_detected=0,
                faces=[],
                processing_time=processing_time,
                error=error_msg
            )

    def detect_faces_batch(self, image_paths: List[Path]) -> List[FaceRecognitionResult]:
        """Detect faces in multiple images.

        Args:
            image_paths: List of image file paths

        Returns:
            List of FaceRecognitionResult objects
        """
        results = []

        self.logger.info(f"Processing {len(image_paths)} images for face detection...")

        for i, image_path in enumerate(image_paths):
            self.logger.debug(f"Processing image {i+1}/{len(image_paths)}: {image_path.name}")
            result = self.detect_faces(image_path)
            results.append(result)

        # Summary statistics
        total_faces = sum(r.faces_detected for r in results)
        successful = sum(1 for r in results if r.error is None)

        self.logger.info(f"Batch processing complete: {total_faces} faces detected in {successful}/{len(image_paths)} images")

        return results

    def _identify_person(self, unknown_encoding: np.ndarray) -> Optional[str]:
        """Identify a person from face encoding.

        Args:
            unknown_encoding: Face encoding to identify

        Returns:
            Person ID if match found, None otherwise
        """
        return self.people_database.identify_person(unknown_encoding, self.recognition_tolerance)

    def add_person(self, person_id: str, image_paths: List[Path]) -> bool:
        """Add a person to the known people database.

        Args:
            person_id: Unique identifier for the person
            image_paths: List of image paths containing this person

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            self.logger.error("Face recognition not available")
            return False

        try:
            encodings = []

            for image_path in image_paths:
                result = self.detect_faces(image_path)

                if result.error:
                    self.logger.warning(f"Could not process {image_path}: {result.error}")
                    continue

                # Use all face encodings from this image
                for face in result.faces:
                    if face.encoding is not None:
                        encodings.append(face.encoding)

            if not encodings:
                self.logger.error(f"No face encodings found for person {person_id}")
                return False

            # Use PeopleDatabase for persistence
            success = self.people_database.add_person(
                person_id=person_id,
                name=person_id,  # Use person_id as name for now
                encodings=encodings,
                photo_paths=[str(p) for p in image_paths]
            )

            if success:
                self.logger.info(f"Added person '{person_id}' with {len(encodings)} face encodings")
                return True
            else:
                return False

        except Exception as e:
            self.logger.error(f"Error adding person {person_id}: {e}")
            return False

    def remove_person(self, person_id: str) -> bool:
        """Remove a person from the known people database.

        Args:
            person_id: Person to remove

        Returns:
            True if removed, False if not found
        """
        return self.people_database.remove_person(person_id)

    def list_known_people(self) -> List[str]:
        """Get list of known people IDs."""
        people = self.people_database.list_people()
        return [person.person_id for person in people]

    def get_statistics(self) -> Dict[str, Any]:
        """Get face recognition statistics."""
        cache_size = len(self.face_cache) if self.enable_caching else 0
        db_stats = self.people_database.get_statistics()
        known_people_count = db_stats['total_people']
        total_known_encodings = db_stats['total_encodings']

        return {
            'enabled': self.enabled,
            'detection_model': self.detection_model,
            'recognition_tolerance': self.recognition_tolerance,
            'min_face_size': self.min_face_size,
            'cache_enabled': self.enable_caching,
            'cached_images': cache_size,
            'known_people': known_people_count,
            'total_known_encodings': total_known_encodings
        }

    def clear_cache(self) -> None:
        """Clear the face detection cache."""
        self.face_cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
        self.logger.info("Face detection cache cleared")

    def cleanup(self) -> None:
        """Cleanup resources and save cache."""
        if self.enable_caching and self.face_cache:
            self._save_cache()
        self.logger.info("Face recognizer cleanup completed")