"""Vector database for storing and searching photo embeddings."""

import chromadb
from chromadb.config import Settings
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import logging

from .config import VECTOR_DB_DIR, COLLECTION_NAME

class VectorDatabase:
    """Manages ChromaDB vector database for photo embeddings and metadata."""

    def __init__(self, persist_directory: Path = None):
        """Initialize the vector database.

        Args:
            persist_directory: Directory to persist the database
        """
        self.persist_directory = persist_directory or VECTOR_DB_DIR
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                allow_reset=True,
                anonymized_telemetry=False
            )
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(COLLECTION_NAME)
            self.logger.info(f"Loaded existing collection '{COLLECTION_NAME}'")
        except:
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Photo embeddings and metadata for intelligent clustering"}
            )
            self.logger.info(f"Created new collection '{COLLECTION_NAME}'")

    def add_photo_embedding(self,
                          photo_id: str,
                          embedding: np.ndarray,
                          metadata: Dict[str, Any],
                          event_folder: Optional[str] = None) -> bool:
        """Add a photo embedding to the database.

        Args:
            photo_id: Unique identifier for the photo
            embedding: Photo embedding vector
            metadata: Photo metadata dictionary
            event_folder: Event folder name if photo is already organized

        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"💾 VECTOR_DB DEBUG: add_photo_embedding called for photo_id: {photo_id[:50]}...")

            # Prepare metadata for ChromaDB (only string, int, float, bool values)
            chroma_metadata = self._prepare_metadata_for_chroma(metadata)

            # Add event folder if provided
            if event_folder:
                chroma_metadata['event_folder'] = event_folder
                chroma_metadata['is_organized'] = True
            else:
                chroma_metadata['is_organized'] = False

            print(f"💾 VECTOR_DB DEBUG: Prepared metadata for {photo_id[:50]}..., event_folder={event_folder}")

            # Convert embedding to list for ChromaDB
            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            print(f"💾 VECTOR_DB DEBUG: Converted embedding to list, size: {len(embedding_list)}")

            # Add to collection
            print(f"💾 VECTOR_DB DEBUG: About to add to ChromaDB collection...")
            self.collection.add(
                embeddings=[embedding_list],
                metadatas=[chroma_metadata],
                ids=[photo_id]
            )
            print(f"💾 VECTOR_DB DEBUG: Successfully added {photo_id[:50]}... to ChromaDB collection")

            return True

        except Exception as e:
            self.logger.error(f"Error adding photo embedding {photo_id}: {e}")
            return False

    def photo_exists(self, photo_id: str) -> bool:
        """Check if a photo already exists in the database.

        Args:
            photo_id: Unique identifier for the photo

        Returns:
            True if photo exists, False otherwise
        """
        try:
            print(f"🔍 VECTOR_DB DEBUG: Checking if photo_id '{photo_id}' exists in collection")
            result = self.collection.get(ids=[photo_id])
            exists = len(result['ids']) > 0
            print(f"🔍 VECTOR_DB DEBUG: Query result for '{photo_id}': {len(result['ids'])} entries found -> exists={exists}")
            return exists
        except Exception as e:
            print(f"💥 VECTOR_DB DEBUG: Exception checking photo_id '{photo_id}': {e}")
            self.logger.error(f"Error checking if photo {photo_id} exists: {e}")
            return False

    def add_batch_embeddings(self,
                           photo_ids: List[str],
                           embeddings: List[np.ndarray],
                           metadatas: List[Dict[str, Any]],
                           event_folders: Optional[List[str]] = None) -> int:
        """Add multiple photo embeddings in batch.

        Args:
            photo_ids: List of photo IDs
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            event_folders: Optional list of event folder names

        Returns:
            Number of successfully added embeddings
        """
        if not photo_ids or len(photo_ids) != len(embeddings) or len(photo_ids) != len(metadatas):
            self.logger.error("Mismatched lengths in batch embeddings")
            return 0

        try:
            chroma_metadatas = []
            chroma_embeddings = []

            for i, (photo_id, embedding, metadata) in enumerate(zip(photo_ids, embeddings, metadatas)):
                # Prepare metadata
                chroma_metadata = self._prepare_metadata_for_chroma(metadata)

                # Add event folder if provided
                if event_folders and i < len(event_folders) and event_folders[i]:
                    chroma_metadata['event_folder'] = event_folders[i]
                    chroma_metadata['is_organized'] = True
                else:
                    chroma_metadata['is_organized'] = False

                chroma_metadatas.append(chroma_metadata)

                # Convert embedding
                embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                chroma_embeddings.append(embedding_list)

            # Add batch to collection
            self.collection.add(
                embeddings=chroma_embeddings,
                metadatas=chroma_metadatas,
                ids=photo_ids
            )

            self.logger.info(f"Successfully added {len(photo_ids)} embeddings to database")
            return len(photo_ids)

        except Exception as e:
            self.logger.error(f"Error adding batch embeddings: {e}")
            return 0

    def search_similar_photos(self,
                            query_embedding: np.ndarray,
                            n_results: int = 10,
                            filter_organized: bool = True) -> List[Dict[str, Any]]:
        """Search for similar photos based on embedding.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filter_organized: If True, only search organized photos

        Returns:
            List of similar photos with metadata and distances
        """
        try:
            # Prepare query embedding
            query_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding

            # Prepare filter
            where_filter = None
            if filter_organized:
                where_filter = {"is_organized": {"$eq": True}}

            # Search
            results = self.collection.query(
                query_embeddings=[query_list],
                n_results=n_results,
                where=where_filter
            )

            # Format results
            similar_photos = []
            if results['ids'] and results['ids'][0]:
                for i, photo_id in enumerate(results['ids'][0]):
                    photo_data = {
                        'photo_id': photo_id,
                        'distance': results['distances'][0][i],
                        'metadata': results['metadatas'][0][i]
                    }
                    similar_photos.append(photo_data)

            return similar_photos

        except Exception as e:
            self.logger.error(f"Error searching similar photos: {e}")
            return []

    def get_organized_photos_by_event(self, event_folder: str) -> List[Dict[str, Any]]:
        """Get all photos from a specific event folder.

        Args:
            event_folder: Event folder name

        Returns:
            List of photos in the event
        """
        try:
            results = self.collection.get(
                where={"$and": [
                    {"event_folder": {"$eq": event_folder}},
                    {"is_organized": {"$eq": True}}
                ]}
            )

            photos = []
            if results['ids']:
                for i, photo_id in enumerate(results['ids']):
                    photo_data = {
                        'photo_id': photo_id,
                        'metadata': results['metadatas'][i],
                        'embedding': results['embeddings'][i] if results.get('embeddings') else None
                    }
                    photos.append(photo_data)

            return photos

        except Exception as e:
            self.logger.error(f"Error getting photos for event {event_folder}: {e}")
            return []

    def get_all_event_folders(self) -> List[str]:
        """Get list of all event folders in the database.

        Returns:
            List of unique event folder names
        """
        try:
            results = self.collection.get(
                where={"is_organized": {"$eq": True}}
            )

            event_folders = set()
            if results['metadatas']:
                for metadata in results['metadatas']:
                    if 'event_folder' in metadata:
                        event_folders.add(metadata['event_folder'])

            return sorted(list(event_folders))

        except Exception as e:
            self.logger.error(f"Error getting event folders: {e}")
            return []

    def delete_photo(self, photo_id: str) -> bool:
        """Delete a photo from the database.

        Args:
            photo_id: Photo ID to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[photo_id])
            return True
        except Exception as e:
            self.logger.error(f"Error deleting photo {photo_id}: {e}")
            return False

    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database.

        Returns:
            Dictionary with database statistics
        """
        try:
            count = self.collection.count()

            # Get organized vs unorganized counts
            organized_results = self.collection.get(where={"is_organized": {"$eq": True}})
            organized_count = len(organized_results['ids']) if organized_results['ids'] else 0

            unorganized_count = count - organized_count

            # Get event folders
            event_folders = self.get_all_event_folders()

            return {
                'total_photos': count,
                'organized_photos': organized_count,
                'unorganized_photos': unorganized_count,
                'event_folders_count': len(event_folders),
                'event_folders': event_folders
            }

        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {
                'total_photos': 0,
                'organized_photos': 0,
                'unorganized_photos': 0,
                'event_folders_count': 0,
                'event_folders': []
            }

    def _prepare_metadata_for_chroma(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metadata for ChromaDB by converting to supported types.

        Args:
            metadata: Original metadata dictionary

        Returns:
            ChromaDB-compatible metadata dictionary
        """
        chroma_metadata = {}

        for key, value in metadata.items():
            # Skip None values
            if value is None:
                continue

            # Convert to supported types (string, int, float, bool)
            if isinstance(value, (str, int, float, bool)):
                chroma_metadata[key] = value
            elif isinstance(value, (list, tuple)):
                # Convert lists/tuples to JSON strings
                if len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
                    # GPS coordinates
                    chroma_metadata[f"{key}_lat"] = float(value[0])
                    chroma_metadata[f"{key}_lon"] = float(value[1])
                else:
                    chroma_metadata[key] = json.dumps(value)
            else:
                # Convert other types to string
                chroma_metadata[key] = str(value)

        return chroma_metadata

    def reset_database(self) -> bool:
        """Reset the entire database (delete all data).

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_collection(COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Photo embeddings and metadata for intelligent clustering"}
            )
            self.logger.info("Database reset successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error resetting database: {e}")
            return False