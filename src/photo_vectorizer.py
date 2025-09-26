"""Photo vectorization pipeline using computer vision models."""

import os
# Set environment variables for fast processing before importing transformers
os.environ['TRANSFORMERS_USE_FAST'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Suppress tokenizer warnings

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from tqdm import tqdm
import cv2

from .gpu_utils import gpu_manager
from .media_detector import MediaFile

class PhotoVectorizer:
    """Creates vector embeddings for photos using pre-trained vision models."""

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        """Initialize the photo vectorizer.

        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.logger = logging.getLogger(__name__)
        self.device = gpu_manager.get_device()
        self.model_name = model_name
        self.model = None
        self.batch_size = gpu_manager.get_optimal_batch_size(base_batch_size=16)

        self._load_model()

    def _load_model(self) -> None:
        """Load the vision model."""
        try:
            self.logger.info(f"Loading model {self.model_name} on {self.device}")

            # Load CLIP model for image embeddings with fast processing enabled
            try:
                self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
                self.logger.info("CLIP model loaded with fast processing enabled")
            except Exception as e:
                self.logger.warning(f"Could not load with fast processing, falling back: {e}")
                self.model = SentenceTransformer(self.model_name)

            # Configure for GPU if available
            if gpu_manager.is_gpu_available():
                gpu_manager.configure_model_for_device(self.model)
                gpu_manager.optimize_for_inference()

            self.logger.info(f"Model loaded successfully with batch size {self.batch_size}")

        except Exception as e:
            self.logger.error(f"Error loading model {self.model_name}: {e}")
            raise

    def vectorize_single_photo(self, image_path: Path) -> Optional[np.ndarray]:
        """Create vector embedding for a single photo.

        Args:
            image_path: Path to the image file

        Returns:
            Vector embedding as numpy array, or None if failed
        """
        photo_name = image_path.name
        try:
            # Check if file format is supported before processing
            if not self._is_supported_image_format(image_path):
                self.logger.warning(f"⚠️  Skipping unsupported format: {photo_name}")
                return None

            self.logger.info(f"🔄 Vectorizing: {photo_name}")

            # Load and preprocess image
            image = Image.open(image_path)

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize if too large (for memory efficiency)
            max_size = 1024
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # Create embedding
            with torch.no_grad():
                embedding = self.model.encode(
                    image,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )

            self.logger.info(f"✅ Successfully vectorized: {photo_name}")
            return embedding

        except Exception as e:
            self.logger.error(f"❌ Failed to vectorize {photo_name}: {e}")
            return None

    def vectorize_batch_photos(self, image_paths: List[Path]) -> List[Optional[np.ndarray]]:
        """Create vector embeddings for a batch of photos.

        Args:
            image_paths: List of image file paths

        Returns:
            List of vector embeddings (None for failed images)
        """
        embeddings = []

        try:
            # Process in batches for memory efficiency
            for i in range(0, len(image_paths), self.batch_size):
                batch_paths = image_paths[i:i + self.batch_size]
                batch_images = []
                valid_indices = []

                # Load batch images
                for idx, image_path in enumerate(batch_paths):
                    photo_name = image_path.name
                    try:
                        # Check if file format is supported before processing
                        if not self._is_supported_image_format(image_path):
                            self.logger.warning(f"⚠️  Skipping unsupported format: {photo_name}")
                            embeddings.append(None)
                            continue

                        self.logger.info(f"🔄 Loading for batch: {photo_name}")
                        image = Image.open(image_path)

                        # Convert to RGB if necessary
                        if image.mode != 'RGB':
                            image = image.convert('RGB')

                        # Resize if too large
                        max_size = 1024
                        if max(image.size) > max_size:
                            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                        batch_images.append(image)
                        valid_indices.append(i + idx)

                    except Exception as e:
                        self.logger.error(f"❌ Failed to load {photo_name}: {e}")
                        embeddings.append(None)

                # Process valid images in batch
                if batch_images:
                    try:
                        with torch.no_grad():
                            batch_embeddings = self.model.encode(
                                batch_images,
                                convert_to_numpy=True,
                                normalize_embeddings=True,
                                batch_size=len(batch_images)
                            )

                        # Add embeddings in correct order and log success
                        embedding_idx = 0
                        for original_idx in range(i, i + len(batch_paths)):
                            if original_idx in valid_indices:
                                photo_name = image_paths[original_idx].name
                                embeddings.append(batch_embeddings[embedding_idx])
                                self.logger.info(f"✅ Successfully vectorized: {photo_name}")
                                embedding_idx += 1
                            # Note: None was already added for invalid images

                    except Exception as e:
                        self.logger.error(f"Error processing batch: {e}")
                        # Add None for all images in this batch and log individual failures
                        for idx in range(len(batch_images)):
                            photo_name = image_paths[i + idx].name
                            self.logger.error(f"❌ Failed to vectorize (batch error): {photo_name}")
                            embeddings.append(None)

                # Clear GPU cache periodically
                if gpu_manager.is_gpu_available() and i % (self.batch_size * 4) == 0:
                    gpu_manager.clear_gpu_cache()

        except Exception as e:
            self.logger.error(f"Error in batch vectorization: {e}")

        return embeddings

    def _is_supported_image_format(self, image_path: Path) -> bool:
        """Check if the image format is supported by PIL/CLIP.

        Args:
            image_path: Path to the image file

        Returns:
            True if format is supported, False otherwise
        """
        # Supported image extensions for CLIP/PIL
        supported_extensions = {
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
            '.webp', '.gif', '.ppm', '.pgm', '.pbm'
        }

        file_extension = image_path.suffix.lower()

        # Check extension
        if file_extension not in supported_extensions:
            return False

        # Additional check: try to verify it's actually an image
        try:
            with Image.open(image_path) as img:
                img.verify()  # Verify it's a valid image
            return True
        except Exception:
            return False

    def _is_supported_video_format(self, video_path: Path) -> bool:
        """Check if the video format is supported.

        Args:
            video_path: Path to the video file

        Returns:
            True if format is supported, False otherwise
        """
        # Supported video extensions
        supported_extensions = {
            '.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv',
            '.webm', '.m4v', '.3gp', '.3g2'
        }

        file_extension = video_path.suffix.lower()
        return file_extension in supported_extensions

    def _vectorize_videos(self, video_paths: List[Path]) -> List[Optional[np.ndarray]]:
        """Vectorize video files by extracting key frames and using CLIP.

        Args:
            video_paths: List of video file paths

        Returns:
            List of embeddings (or None for failed videos)
        """
        import cv2

        embeddings = []

        for video_path in tqdm(video_paths, desc="Vectorizing videos"):
            try:
                embedding = self._extract_video_embedding(video_path)
                embeddings.append(embedding)
            except Exception as e:
                self.logger.error(f"❌ Failed to vectorize video {video_path.name}: {e}")
                embeddings.append(None)

        return embeddings

    def _extract_video_embedding(self, video_path: Path) -> Optional[np.ndarray]:
        """Extract embedding from video by analyzing key frames.

        Args:
            video_path: Path to video file

        Returns:
            Average embedding from key frames, or None if failed
        """
        import cv2

        try:
            self.logger.info(f"🎬 Processing video: {video_path.name}")

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.logger.error(f"Could not open video: {video_path.name}")
                return None

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Extract 3-5 frames evenly distributed throughout the video
            num_frames_to_extract = min(5, max(3, total_frames // 30))  # At most 5 frames
            frame_indices = [int(i * total_frames / num_frames_to_extract)
                           for i in range(num_frames_to_extract)]

            frame_embeddings = []

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    # Convert OpenCV frame (BGR) to PIL Image (RGB)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)

                    # Resize if needed
                    if max(pil_image.size) > 1024:
                        pil_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

                    # Get embedding for this frame
                    with torch.no_grad():
                        frame_embedding = self.model.encode(
                            pil_image,
                            convert_to_numpy=True,
                            normalize_embeddings=True
                        )
                        frame_embeddings.append(frame_embedding)

            cap.release()

            if frame_embeddings:
                # Average the frame embeddings to get video embedding
                video_embedding = np.mean(frame_embeddings, axis=0)
                self.logger.info(f"✅ Successfully processed video: {video_path.name} ({len(frame_embeddings)} frames)")
                return video_embedding
            else:
                self.logger.error(f"No valid frames extracted from: {video_path.name}")
                return None

        except Exception as e:
            self.logger.error(f"Error processing video {video_path.name}: {e}")
            return None

    def vectorize_media_files(self, media_files: List[MediaFile]) -> List[Tuple[str, Optional[np.ndarray]]]:
        """Vectorize a list of media files (photos only).

        Args:
            media_files: List of MediaFile objects

        Returns:
            List of tuples (file_id, embedding)
        """
        # Separate photos and videos for different processing
        photo_files = []
        video_files = []
        skipped_count = 0

        for mf in media_files:
            if mf.file_type == 'photo':
                if self._is_supported_image_format(mf.path):
                    photo_files.append(mf)
                else:
                    self.logger.warning(f"⚠️  Skipping unsupported image format: {mf.filename}")
                    skipped_count += 1
            elif mf.file_type == 'video':
                if self._is_supported_video_format(mf.path):
                    video_files.append(mf)
                else:
                    self.logger.warning(f"⚠️  Skipping unsupported video format: {mf.filename}")
                    skipped_count += 1

        total_files = len(photo_files) + len(video_files)
        if total_files == 0:
            self.logger.info("No supported media files to vectorize")
            return []

        if skipped_count > 0:
            self.logger.info(f"📊 Processing {len(photo_files)} photos, {len(video_files)} videos, skipped {skipped_count} unsupported formats")
        else:
            self.logger.info(f"Processing {len(photo_files)} photos and {len(video_files)} videos...")

        # Process photos and videos separately, then combine results
        results = []

        # Process photos with CLIP
        if photo_files:
            image_paths = [mf.path for mf in photo_files]
            photo_ids = [self._create_file_id(mf) for mf in photo_files]
            photo_embeddings = self._vectorize_images(image_paths)
            results.extend(zip(photo_ids, photo_embeddings))

        # Process videos by extracting frames and using CLIP
        if video_files:
            video_paths = [mf.path for mf in video_files]
            video_ids = [self._create_file_id(mf) for mf in video_files]
            video_embeddings = self._vectorize_videos(video_paths)
            results.extend(zip(video_ids, video_embeddings))

        return results

    def _vectorize_images(self, image_paths: List[Path]) -> List[Optional[np.ndarray]]:
        """Vectorize image files using CLIP."""
        embeddings = []

        for i in tqdm(range(0, len(image_paths), self.batch_size),
                     desc="Vectorizing photos",
                     unit="batch"):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_embeddings = self.vectorize_batch_photos(batch_paths)
            embeddings.extend(batch_embeddings)

        return embeddings

    def _create_file_id(self, media_file: MediaFile) -> str:
        """Create a unique ID for a media file.

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
            self.logger.debug(f"📱 VECTORIZER: Using iPhone filename as photo_id: {filename}")
            return filename
        else:
            # Non-iPhone filename, add timestamp prefix
            timestamp = media_file.time.strftime("%Y%m%d_%H%M%S")
            photo_id = f"{timestamp}_{filename}"
            self.logger.debug(f"📷 VECTORIZER: Created photo_id with timestamp: {photo_id}")
            return photo_id

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Normalize embeddings
            emb1_norm = embedding1 / np.linalg.norm(embedding1)
            emb2_norm = embedding2 / np.linalg.norm(embedding2)

            # Compute cosine similarity
            similarity = np.dot(emb1_norm, emb2_norm)

            # Ensure result is in [0, 1] range
            return max(0.0, min(1.0, float(similarity)))

        except Exception as e:
            self.logger.warning(f"Error computing similarity: {e}")
            return 0.0

    def find_similar_photos(self,
                          query_embedding: np.ndarray,
                          candidate_embeddings: List[Tuple[str, np.ndarray]],
                          threshold: float = 0.8,
                          max_results: int = 10) -> List[Tuple[str, float]]:
        """Find similar photos based on embedding similarity.

        Args:
            query_embedding: Query photo embedding
            candidate_embeddings: List of (photo_id, embedding) tuples
            threshold: Similarity threshold (0-1)
            max_results: Maximum number of results to return

        Returns:
            List of (photo_id, similarity_score) tuples, sorted by similarity
        """
        similarities = []

        for photo_id, embedding in candidate_embeddings:
            if embedding is not None:
                similarity = self.compute_similarity(query_embedding, embedding)
                if similarity >= threshold:
                    similarities.append((photo_id, similarity))

        # Sort by similarity (descending) and limit results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'batch_size': self.batch_size,
            'gpu_available': gpu_manager.is_gpu_available(),
            'gpu_info': gpu_manager.get_gpu_info()
        }

    def cleanup(self) -> None:
        """Cleanup resources and clear GPU cache."""
        if gpu_manager.is_gpu_available():
            gpu_manager.clear_gpu_cache()

        if hasattr(self.model, 'cpu'):
            self.model.cpu()

        self.logger.info("Vectorizer cleanup completed")