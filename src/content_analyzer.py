"""
Content analysis for photos using computer vision models.

This module analyzes photos to generate natural-language descriptions, using a
vision LLM served by Ollama (model selected via PHOTO_FILTER_CAPTION_MODEL, see
environment_config.py), plus face recognition for identifying people. CLIP is
used elsewhere in the project (PhotoVectorizer) for embeddings/similarity, but
not here for classification - see temp/naming_diagnostic_plan.md.

For junior developers:
- Implements caching to avoid re-analyzing the same photos
- Captioning is a live HTTP call to Ollama - no local model to load
- Face recognition (when enabled) runs locally via the injected FaceRecognizer
"""

import base64
import io
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

# Always import PIL for basic image handling (this is required)
from PIL import Image, ImageOps

from .environment_config import get_caption_model, get_ollama_url

# Ollama HTTP calls - optional dependency, same pattern as event_namer.py
try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

@dataclass
class ContentAnalysis:
    """
    Results from photo content analysis.

    This is a data container holding everything we learned about a photo.

    For junior developers:
    - description: Natural language description of the photo
    - confidence_score: How confident we are (0.0 to 1.0)
    - analysis_model: Which model was used (<vision model>, <vision model>+Face Recognition)
    - people_detected: List of identified people in the photo
    - face_count: Number of faces detected
    """
    description: str           # Natural language description
    confidence_score: float    # Confidence 0.0-1.0
    analysis_model: str        # Which model was used
    people_detected: List[str] # Identified people in the photo
    face_count: int            # Number of faces detected

class ContentAnalyzer:
    """Analyzes photo content using computer vision models."""

    def __init__(self, use_gpu: bool = True, face_recognizer=None,
                 vision_model: Optional[str] = None, ollama_url: Optional[str] = None):
        """Initialize content analyzer.

        Args:
            use_gpu: Unused here (no local model runs in this class); accepted
                for constructor compatibility with the shared processing.use_gpu config
            face_recognizer: Optional FaceRecognizer instance for people detection
            vision_model: Ollama model used for captioning (default: PHOTO_FILTER_CAPTION_MODEL)
            ollama_url: Ollama server URL (default: PHOTO_FILTER_OLLAMA_URL)
        """
        self.logger = logging.getLogger(__name__)
        self.use_gpu = use_gpu
        self.face_recognizer = face_recognizer
        self.vision_model = vision_model or get_caption_model()
        self.ollama_url = ollama_url or get_ollama_url()

        # Cache for analysis results
        self.analysis_cache = {}

    def analyze_photo_content(self, photo_path: Path) -> Optional[ContentAnalysis]:
        """Analyze photo content to generate a description and identify people.

        Args:
            photo_path: Path to photo file

        Returns:
            ContentAnalysis object or None if analysis fails
        """
        try:
            # Check cache first
            cache_key = str(photo_path)
            if cache_key in self.analysis_cache:
                return self.analysis_cache[cache_key]

            # Load and preprocess image. EXIF-transpose before anything else -
            # sideways/upside-down photos (common with phone orientation
            # metadata) caption and classify badly otherwise.
            image = ImageOps.exif_transpose(Image.open(photo_path))
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Perform comprehensive analysis
            analysis = self._comprehensive_analysis(image, photo_path)

            # Cache result
            self.analysis_cache[cache_key] = analysis

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing photo content {photo_path}: {e}")
            return None

    def _comprehensive_analysis(self, image: Image.Image, photo_path: Path) -> ContentAnalysis:
        """Perform comprehensive content analysis using the Ollama vision model."""
        try:
            # Generate image description using the Ollama vision model
            description = self._generate_description(image)

            # Perform face recognition if available
            people_detected, face_count = self._analyze_faces(photo_path)

            # Calculate overall confidence
            confidence = self._calculate_confidence(description)

            return ContentAnalysis(
                description=description,
                confidence_score=confidence,
                analysis_model=f"{self.vision_model}+Face Recognition" if self.face_recognizer else self.vision_model,
                people_detected=people_detected,
                face_count=face_count
            )

        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {e}")
            raise RuntimeError(f"Content analysis failed for {photo_path.name}: {e}") from e

    _CAPTION_PROMPT = (
        "Describe this photo in 2-3 sentences, covering: the setting, any visible "
        "occasion clues (decorations, cake, rings, attire), how many people are "
        "visible and what they are doing. Also note any details that would indicate "
        "whether the setting is a private home or a commercial/retail business "
        "(e.g. price tags, sales racks, store signage, checkout counters vs. "
        "personal belongings, home decor, family photos on the wall). "
        "Describe only what is visible; do not guess names or relationships.")

    def _generate_description(self, image: Image.Image) -> str:
        """Generate natural language description via the Ollama vision model."""
        try:
            buf = io.BytesIO()
            image.convert("RGB").save(buf, format="JPEG", quality=90)
            image_b64 = base64.b64encode(buf.getvalue()).decode()

            payload = {
                "model": self.vision_model,
                "prompt": self._CAPTION_PROMPT,
                "images": [image_b64],
                "stream": False,
                "think": False,
                "options": {"temperature": 0.0, "num_predict": 350}
            }
            # Generous timeout - the local Ollama server is often shared with
            # other concurrent work, which can make an otherwise-quick
            # captioning call take much longer than usual to get scheduled.
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=1800)
            response.raise_for_status()
            return response.json().get("response", "").strip()

        except Exception as e:
            self.logger.error(f"Error generating description: {e}")
            return "Unable to generate description"

    def _calculate_confidence(self, description: str) -> float:
        """Calculate confidence score for the analysis based on captioning success."""
        if description and description != "Unable to generate description":
            return 0.8
        return 0.0

    def analyze_batch(self, photo_paths: List[Path],
                     max_photos: Optional[int] = None) -> Dict[str, ContentAnalysis]:
        """Analyze multiple photos in batch.

        Args:
            photo_paths: List of photo file paths
            max_photos: Maximum number of photos to analyze

        Returns:
            Dictionary mapping photo paths to analysis results
        """
        results = {}

        # Limit batch size if specified
        if max_photos:
            photo_paths = photo_paths[:max_photos]

        self.logger.info(f"Starting batch content analysis of {len(photo_paths)} photos")

        for i, photo_path in enumerate(photo_paths):
            self.logger.info(f"Analyzing photo {i+1}/{len(photo_paths)}: {photo_path.name}")

            analysis = self.analyze_photo_content(photo_path)
            if analysis:
                results[str(photo_path)] = analysis

        self.logger.info(f"Completed batch analysis: {len(results)} photos analyzed")
        return results

    def get_content_summary(self, analyses: Dict[str, ContentAnalysis]) -> Dict[str, Any]:
        """Generate summary statistics from multiple content analyses.

        Args:
            analyses: Dictionary of photo analyses

        Returns:
            Summary statistics
        """
        if not analyses:
            return {"error": "No analyses provided"}

        total_confidence = sum(analysis.confidence_score for analysis in analyses.values())

        # Representative vision-model captions - unique, non-empty, in analysis order.
        # These carry scene understanding a tag vocabulary can't express.
        sample_captions = []
        for analysis in analyses.values():
            description = (analysis.description or "").strip()
            if description and description not in sample_captions:
                sample_captions.append(description)

        return {
            "total_photos_analyzed": len(analyses),
            "average_confidence": total_confidence / len(analyses),
            "sample_captions": sample_captions[:3]
        }

    def save_analysis_cache(self, cache_file: Path):
        """Save analysis cache to file."""
        try:
            # Convert ContentAnalysis objects to dicts for JSON serialization
            cache_data = {}
            for key, analysis in self.analysis_cache.items():
                cache_data[key] = {
                    "description": analysis.description,
                    "confidence_score": analysis.confidence_score,
                    "analysis_model": analysis.analysis_model,
                    "people_detected": analysis.people_detected,
                    "face_count": analysis.face_count
                }

            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)

            self.logger.info(f"Analysis cache saved to {cache_file}")

        except Exception as e:
            self.logger.error(f"Error saving analysis cache: {e}")

    def load_analysis_cache(self, cache_file: Path):
        """Load analysis cache from file."""
        try:
            if not cache_file.exists():
                return

            with open(cache_file, 'r') as f:
                cache_data = json.load(f)

            # Convert dicts back to ContentAnalysis objects
            for key, data in cache_data.items():
                self.analysis_cache[key] = ContentAnalysis(**data)

            self.logger.info(f"Analysis cache loaded from {cache_file} ({len(cache_data)} entries)")

        except Exception as e:
            self.logger.error(f"Error loading analysis cache: {e}")

    def _analyze_faces(self, photo_path: Path) -> tuple[List[str], int]:
        """Analyze faces in the photo using face recognition.

        Args:
            photo_path: Path to the photo file

        Returns:
            Tuple of (people_detected, face_count)
        """
        if not self.face_recognizer or not self.face_recognizer.enabled:
            return [], 0

        try:
            # Use face recognizer to detect and identify faces
            result = self.face_recognizer.detect_faces(photo_path)

            if result.error:
                self.logger.warning(f"Face recognition failed for {photo_path.name}: {result.error}")
                return [], 0

            # Extract people names and face count
            people_detected = result.get_people_detected()
            face_count = result.faces_detected

            if face_count > 0:
                self.logger.debug(f"Found {face_count} faces in {photo_path.name}, identified: {people_detected}")

            return people_detected, face_count

        except Exception as e:
            self.logger.error(f"Error in face analysis for {photo_path.name}: {e}")
            return [], 0

    def cleanup(self):
        """Clean up resources."""
        self.analysis_cache.clear()