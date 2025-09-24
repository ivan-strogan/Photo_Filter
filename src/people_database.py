"""
People database for storing and managing known face encodings.

This module provides persistent storage for known people and their face encodings,
enabling person identification across photos and events.

For junior developers:
- Stores face encodings as JSON with base64 encoding for numpy arrays
- Provides clustering to group similar faces automatically
- Implements confidence scoring for person identification
- Supports manual labeling and learning from organized photos
"""

import logging
import json
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime

# Optional dependencies
try:
    from sklearn.cluster import DBSCAN
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False

from .config import DATA_DIR

@dataclass
class PersonRecord:
    """Record for a known person in the database."""
    person_id: str
    name: str
    encodings: List[np.ndarray]
    photo_paths: List[str]
    confidence_scores: List[float]
    created_date: datetime
    last_seen_date: datetime
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'person_id': self.person_id,
            'name': self.name,
            'encodings': [self._encode_array(enc) for enc in self.encodings],
            'photo_paths': self.photo_paths,
            'confidence_scores': self.confidence_scores,
            'created_date': self.created_date.isoformat(),
            'last_seen_date': self.last_seen_date.isoformat(),
            'notes': self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonRecord':
        """Create PersonRecord from dictionary."""
        return cls(
            person_id=data['person_id'],
            name=data['name'],
            encodings=[cls._decode_array(enc) for enc in data['encodings']],
            photo_paths=data['photo_paths'],
            confidence_scores=data.get('confidence_scores', []),
            created_date=datetime.fromisoformat(data['created_date']),
            last_seen_date=datetime.fromisoformat(data['last_seen_date']),
            notes=data.get('notes', '')
        )

    @staticmethod
    def _encode_array(arr: np.ndarray) -> str:
        """Encode numpy array to base64 string."""
        return base64.b64encode(arr.tobytes()).decode('utf-8')

    @staticmethod
    def _decode_array(encoded: str) -> np.ndarray:
        """Decode base64 string to numpy array."""
        bytes_data = base64.b64decode(encoded.encode('utf-8'))
        return np.frombuffer(bytes_data, dtype=np.float64)

class PeopleDatabase:
    """Database for managing known people and their face encodings."""

    def __init__(self, database_file: Optional[Path] = None):
        """Initialize people database.

        Args:
            database_file: Path to database JSON file
        """
        self.logger = logging.getLogger(__name__)
        self.database_file = database_file or (DATA_DIR / "people_database.json")
        self.people: Dict[str, PersonRecord] = {}

        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Load existing database
        self._load_database()

        self.logger.info(f"People database initialized with {len(self.people)} known people")

    def _load_database(self) -> None:
        """Load people database from JSON file."""
        try:
            if self.database_file.exists():
                with open(self.database_file, 'r') as f:
                    data = json.load(f)

                self.people = {
                    person_id: PersonRecord.from_dict(person_data)
                    for person_id, person_data in data.items()
                }

                self.logger.info(f"Loaded {len(self.people)} people from database")
        except Exception as e:
            self.logger.warning(f"Could not load people database: {e}")
            self.people = {}

    def _save_database(self) -> None:
        """Save people database to JSON file."""
        try:
            data = {
                person_id: person.to_dict()
                for person_id, person in self.people.items()
            }

            with open(self.database_file, 'w') as f:
                json.dump(data, f, indent=2)

            self.logger.debug(f"Saved people database with {len(self.people)} people")
        except Exception as e:
            self.logger.error(f"Could not save people database: {e}")

    def add_person(self,
                   person_id: str,
                   name: str,
                   encodings: List[np.ndarray],
                   photo_paths: List[str],
                   notes: str = "") -> bool:
        """Add a new person to the database.

        Args:
            person_id: Unique identifier for the person
            name: Human-readable name
            encodings: List of face encodings for this person
            photo_paths: Paths to photos used for training
            notes: Optional notes about the person

        Returns:
            True if added successfully, False if person already exists
        """
        if person_id in self.people:
            self.logger.warning(f"Person {person_id} already exists in database")
            return False

        now = datetime.now()
        person_record = PersonRecord(
            person_id=person_id,
            name=name,
            encodings=encodings,
            photo_paths=photo_paths,
            confidence_scores=[1.0] * len(encodings),  # Default high confidence
            created_date=now,
            last_seen_date=now,
            notes=notes
        )

        self.people[person_id] = person_record
        self._save_database()

        self.logger.info(f"Added person '{name}' ({person_id}) with {len(encodings)} encodings")
        return True

    def update_person(self,
                     person_id: str,
                     new_encodings: Optional[List[np.ndarray]] = None,
                     new_photo_paths: Optional[List[str]] = None,
                     new_name: Optional[str] = None,
                     new_notes: Optional[str] = None) -> bool:
        """Update an existing person's information.

        Args:
            person_id: Person to update
            new_encodings: Additional face encodings
            new_photo_paths: Additional photo paths
            new_name: New name for the person
            new_notes: New notes

        Returns:
            True if updated successfully, False if person not found
        """
        if person_id not in self.people:
            self.logger.warning(f"Person {person_id} not found in database")
            return False

        person = self.people[person_id]

        if new_encodings:
            person.encodings.extend(new_encodings)
            person.confidence_scores.extend([1.0] * len(new_encodings))

        if new_photo_paths:
            person.photo_paths.extend(new_photo_paths)

        if new_name:
            person.name = new_name

        if new_notes:
            person.notes = new_notes

        person.last_seen_date = datetime.now()
        self._save_database()

        self.logger.info(f"Updated person '{person.name}' ({person_id})")
        return True

    def remove_person(self, person_id: str) -> bool:
        """Remove a person from the database.

        Args:
            person_id: Person to remove

        Returns:
            True if removed, False if not found
        """
        if person_id in self.people:
            name = self.people[person_id].name
            del self.people[person_id]
            self._save_database()
            self.logger.info(f"Removed person '{name}' ({person_id})")
            return True
        return False

    def get_person(self, person_id: str) -> Optional[PersonRecord]:
        """Get person record by ID."""
        return self.people.get(person_id)

    def find_person_by_name(self, name: str) -> Optional[PersonRecord]:
        """Find person by name (case-insensitive)."""
        for person in self.people.values():
            if person.name.lower() == name.lower():
                return person
        return None

    def list_people(self) -> List[PersonRecord]:
        """Get list of all people in the database."""
        return list(self.people.values())

    def identify_person(self,
                       unknown_encoding: np.ndarray,
                       tolerance: float = 0.6) -> Optional[Tuple[str, float]]:
        """Identify a person from their face encoding.

        Args:
            unknown_encoding: Face encoding to identify
            tolerance: Recognition tolerance (lower = stricter)

        Returns:
            Tuple of (person_id, confidence) if match found, None otherwise
        """
        try:
            import face_recognition
        except ImportError:
            self.logger.error("face_recognition library not available")
            return None

        best_match = None
        best_distance = float('inf')

        for person_id, person in self.people.items():
            # Calculate distances to all encodings for this person
            distances = face_recognition.face_distance(person.encodings, unknown_encoding)

            # Use the best (lowest) distance
            min_distance = min(distances) if distances.size > 0 else float('inf')

            if min_distance < tolerance and min_distance < best_distance:
                best_distance = min_distance
                # Convert distance to confidence (0-1, higher = more confident)
                confidence = 1.0 - min_distance
                best_match = (person_id, confidence)

        if best_match:
            person_id, confidence = best_match
            person = self.people[person_id]
            self.logger.debug(f"Identified {person.name} with confidence {confidence:.3f}")

        return best_match

    def cluster_unknown_faces(self,
                            encodings: List[np.ndarray],
                            min_cluster_size: int = 2,
                            eps: float = 0.5) -> List[List[int]]:
        """Cluster unknown face encodings to group similar faces.

        Args:
            encodings: List of face encodings
            min_cluster_size: Minimum faces per cluster
            eps: Maximum distance between faces in same cluster

        Returns:
            List of clusters, each containing indices of similar faces
        """
        if not CLUSTERING_AVAILABLE:
            self.logger.warning("Clustering not available (scikit-learn not installed)")
            return []

        if len(encodings) < min_cluster_size:
            return []

        try:
            # Convert to numpy array
            encoding_matrix = np.array(encodings)

            # Use DBSCAN clustering
            clustering = DBSCAN(eps=eps, min_samples=min_cluster_size, metric='euclidean')
            cluster_labels = clustering.fit_predict(encoding_matrix)

            # Group indices by cluster
            clusters = {}
            for idx, label in enumerate(cluster_labels):
                if label >= 0:  # -1 means noise/outlier
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(idx)

            cluster_list = list(clusters.values())
            self.logger.info(f"Found {len(cluster_list)} face clusters from {len(encodings)} encodings")

            return cluster_list

        except Exception as e:
            self.logger.error(f"Error clustering faces: {e}")
            return []

    def suggest_person_from_folder_name(self, folder_name: str) -> List[str]:
        """Extract potential person names from folder names.

        Args:
            folder_name: Event folder name (e.g., "2024_10_15 - Sarah Wedding")

        Returns:
            List of potential person names found
        """
        import re

        # Common patterns for extracting names from folder names
        patterns = [
            r"(\w+)\s+(?:Wedding|Birthday|Party|Graduation)",  # "Sarah Wedding"
            r"(\w+)\s*&\s*(\w+)",  # "Sarah & Mike"
            r"(\w+)\s+and\s+(\w+)",  # "Sarah and Mike"
            r"(?:with|and)\s+(\w+)",  # "Party with Sarah"
        ]

        potential_names = []
        folder_lower = folder_name.lower()

        for pattern in patterns:
            matches = re.findall(pattern, folder_lower, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    potential_names.extend(match)
                else:
                    potential_names.append(match)

        # Filter out common words that aren't names
        common_words = {
            'party', 'wedding', 'birthday', 'vacation', 'trip', 'dinner',
            'lunch', 'event', 'celebration', 'photos', 'pics', 'day',
            'night', 'weekend', 'holiday', 'christmas', 'new', 'year'
        }

        names = []
        for name in potential_names:
            name = name.strip().title()
            if len(name) >= 2 and name.lower() not in common_words:
                names.append(name)

        return list(set(names))  # Remove duplicates

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        total_encodings = sum(len(person.encodings) for person in self.people.values())
        avg_encodings = total_encodings / len(self.people) if self.people else 0

        # Recent activity (last 30 days)
        recent_threshold = datetime.now().timestamp() - (30 * 24 * 3600)
        recent_people = sum(
            1 for person in self.people.values()
            if person.last_seen_date.timestamp() > recent_threshold
        )

        return {
            'total_people': len(self.people),
            'total_encodings': total_encodings,
            'average_encodings_per_person': round(avg_encodings, 1),
            'recent_activity_count': recent_people,
            'database_file_size': self.database_file.stat().st_size if self.database_file.exists() else 0
        }

    def export_person_list(self) -> List[Dict[str, Any]]:
        """Export list of people for external use."""
        return [
            {
                'person_id': person.person_id,
                'name': person.name,
                'encoding_count': len(person.encodings),
                'photo_count': len(person.photo_paths),
                'created_date': person.created_date.isoformat(),
                'last_seen_date': person.last_seen_date.isoformat(),
                'notes': person.notes
            }
            for person in self.people.values()
        ]

    def backup_database(self, backup_path: Path) -> bool:
        """Create backup of the database.

        Args:
            backup_path: Path for backup file

        Returns:
            True if backup created successfully
        """
        try:
            import shutil
            shutil.copy2(self.database_file, backup_path)
            self.logger.info(f"Database backed up to {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Could not create backup: {e}")
            return False