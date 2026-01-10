"""
Content analysis for photos using computer vision models.

This module analyzes photos to understand what's in them - objects, scenes,
activities, and provides natural language descriptions. It can work with
or without advanced ML models.

For junior developers:
- This demonstrates "graceful degradation" - works even when ML deps aren't available
- Uses lazy loading pattern - models loaded only when needed
- Implements caching to avoid re-analyzing the same photos
- Shows how to handle optional dependencies with try/except imports
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json

# Always import PIL for basic image handling (this is required)
from PIL import Image

# Import ML components only when needed - this is "optional dependency" pattern
# The app works without these, but has more features when they're available
try:
    import torch                                          # PyTorch for deep learning
    from transformers import BlipProcessor, BlipForConditionalGeneration  # Image captioning
    from transformers import CLIPProcessor, CLIPModel     # Image classification
    import requests                                       # For potential API calls
    TRANSFORMERS_AVAILABLE = True                         # Flag to track if ML is available
except ImportError:
    # ML libraries not installed - we'll use basic analysis instead
    TRANSFORMERS_AVAILABLE = False

@dataclass
class ContentAnalysis:
    """
    Results from photo content analysis.

    This is a data container holding everything we learned about a photo.

    For junior developers:
    - objects: Things we detected (person, car, dog, food, etc.)
    - scenes: Where the photo was taken (indoor, outdoor, restaurant, etc.)
    - activities: What's happening (eating, vacation, celebration, etc.)
    - description: Natural language description of the photo
    - confidence_score: How confident we are (0.0 to 1.0)
    - analysis_model: Which model was used (Basic, CLIP+BLIP, etc.)
    - people_detected: List of identified people in the photo
    - face_count: Number of faces detected
    """
    objects: List[str]         # Things detected in the photo
    scenes: List[str]          # Scene types (indoor/outdoor, etc.)
    activities: List[str]      # Activities happening
    description: str           # Natural language description
    confidence_score: float    # Confidence 0.0-1.0
    analysis_model: str        # Which model was used
    people_detected: List[str] # Identified people in the photo
    face_count: int            # Number of faces detected

class ContentAnalyzer:
    """Analyzes photo content using computer vision models."""

    def __init__(self, use_gpu: bool = True, enable_local_models: bool = True, face_recognizer=None):
        """Initialize content analyzer.

        Args:
            use_gpu: Whether to use GPU acceleration
            enable_local_models: Whether to use local CLIP/BLIP models
            face_recognizer: Optional FaceRecognizer instance for people detection
        """
        self.logger = logging.getLogger(__name__)
        self.use_gpu = use_gpu
        self.enable_local_models = enable_local_models
        self.face_recognizer = face_recognizer

        # Model components (lazy loaded)
        self.clip_model = None
        self.clip_processor = None
        self.blip_model = None
        self.blip_processor = None

        # Cache for analysis results
        self.analysis_cache = {}

        # Predefined categories for classification
        self.scene_categories = [
            "outdoor", "indoor", "nature", "urban", "beach", "mountain",
            "forest", "restaurant", "home", "office", "park", "street",
            "garden", "lake", "river", "building", "church", "museum"
        ]

        self.activity_categories = [
            "eating", "drinking", "walking", "running", "sitting", "standing",
            "playing", "working", "reading", "cooking", "driving", "swimming",
            "shopping", "meeting", "celebration", "vacation", "sports", "exercise"
        ]

        self.object_categories = [
            "person", "car", "bicycle", "motorcycle", "bus", "truck", "boat",
            "airplane", "dog", "cat", "bird", "horse", "cow", "elephant",
            "food", "cake", "pizza", "wine", "beer", "coffee", "book",
            "phone", "laptop", "tv", "clock", "flower", "tree", "building"
        ]

    def _initialize_models(self) -> bool:
        """Initialize computer vision models.

        This application requires local AI models. If models fail to initialize,
        an error will be raised.
        """
        if not TRANSFORMERS_AVAILABLE:
            error_msg = ("CRITICAL: Transformers library not available. "
                        "Install required dependencies: pip install transformers torch")
            self.logger.error(error_msg)
            raise ImportError(error_msg)

        if not self.enable_local_models:
            error_msg = ("CRITICAL: Local models are disabled but required for this application. "
                        "ContentAnalyzer must be initialized with enable_local_models=True")
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # Initialize CLIP for object/scene classification
            if self.clip_model is None:
                self.logger.info("Loading CLIP model for content classification...")
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

                if self.use_gpu and torch.cuda.is_available():
                    self.clip_model = self.clip_model.cuda()
                    self.logger.info("CLIP model loaded on GPU")
                else:
                    self.logger.info("CLIP model loaded on CPU")

            # Initialize BLIP for image captioning
            if self.blip_model is None:
                self.logger.info("Loading BLIP model for image captioning...")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

                if self.use_gpu and torch.cuda.is_available():
                    self.blip_model = self.blip_model.cuda()
                    self.logger.info("BLIP model loaded on GPU")
                else:
                    self.logger.info("BLIP model loaded on CPU")

            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize vision models: {e}")
            return False

    def analyze_photo_content(self, photo_path: Path) -> Optional[ContentAnalysis]:
        """Analyze photo content to extract objects, scenes, and activities.

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

            # Load and preprocess image
            image = Image.open(photo_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Initialize models if needed (raises error if unavailable)
            self._initialize_models()

            # Perform comprehensive analysis
            analysis = self._comprehensive_analysis(image, photo_path)

            # Cache result
            self.analysis_cache[cache_key] = analysis

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing photo content {photo_path}: {e}")
            return None

    def _comprehensive_analysis(self, image: Image.Image, photo_path: Path) -> ContentAnalysis:
        """Perform comprehensive content analysis using CLIP and BLIP."""
        try:
            # Generate image description using BLIP
            description = self._generate_description(image)

            # Classify scenes using CLIP
            scenes = self._classify_scenes(image)

            # Classify objects using CLIP
            objects = self._classify_objects(image)

            # Infer activities from description and classifications
            activities = self._infer_activities(description, objects, scenes)

            # Perform face recognition if available
            people_detected, face_count = self._analyze_faces(photo_path)

            # Calculate overall confidence
            confidence = self._calculate_confidence(description, scenes, objects, activities)

            return ContentAnalysis(
                objects=objects,
                scenes=scenes,
                activities=activities,
                description=description,
                confidence_score=confidence,
                analysis_model="CLIP+BLIP+Face Recognition" if self.face_recognizer else "CLIP+BLIP",
                people_detected=people_detected,
                face_count=face_count
            )

        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {e}")
            raise RuntimeError(f"Content analysis failed for {photo_path.name}: {e}") from e

    def _generate_description(self, image: Image.Image) -> str:
        """Generate natural language description using BLIP."""
        try:
            inputs = self.blip_processor(image, return_tensors="pt")

            if self.use_gpu and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50)

            description = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return description

        except Exception as e:
            self.logger.error(f"Error generating description: {e}")
            return "Unable to generate description"

    def _classify_scenes(self, image: Image.Image) -> List[str]:
        """Classify scene types using CLIP."""
        try:
            # Prepare scene prompts
            scene_prompts = [f"a photo of {scene}" for scene in self.scene_categories]

            # Process image and text
            inputs = self.clip_processor(
                text=scene_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            )

            if self.use_gpu and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

            # Get top scenes (threshold > 0.1)
            scene_scores = probs[0].cpu().numpy()
            top_scenes = []

            for i, score in enumerate(scene_scores):
                if score > 0.1:
                    top_scenes.append((self.scene_categories[i], float(score)))

            # Sort by confidence and return top 3
            top_scenes.sort(key=lambda x: x[1], reverse=True)
            return [scene for scene, _ in top_scenes[:3]]

        except Exception as e:
            self.logger.error(f"Error classifying scenes: {e}")
            return []

    def _classify_objects(self, image: Image.Image) -> List[str]:
        """Classify objects using CLIP."""
        try:
            # Prepare object prompts
            object_prompts = [f"a photo containing {obj}" for obj in self.object_categories]

            # Process image and text
            inputs = self.clip_processor(
                text=object_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            )

            if self.use_gpu and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

            # Get top objects (threshold > 0.15)
            object_scores = probs[0].cpu().numpy()
            top_objects = []

            for i, score in enumerate(object_scores):
                if score > 0.15:
                    top_objects.append((self.object_categories[i], float(score)))

            # Sort by confidence and return top 5
            top_objects.sort(key=lambda x: x[1], reverse=True)
            return [obj for obj, _ in top_objects[:5]]

        except Exception as e:
            self.logger.error(f"Error classifying objects: {e}")
            return []

    def _infer_activities(self, description: str, objects: List[str], scenes: List[str]) -> List[str]:
        """Infer activities from description and detected elements."""
        activities = []
        description_lower = description.lower()

        # Activity inference rules
        activity_rules = {
            "eating": ["eating", "food", "restaurant", "kitchen", "dining"],
            "drinking": ["drinking", "coffee", "beer", "wine", "cafe"],
            "vacation": ["beach", "mountain", "vacation", "travel", "tourist"],
            "celebration": ["cake", "party", "celebration", "birthday", "wedding"],
            "shopping": ["shopping", "store", "mall", "market"],
            "sports": ["playing", "ball", "field", "court", "sports"],
            "cooking": ["cooking", "kitchen", "chef", "preparing"],
            "working": ["office", "computer", "laptop", "meeting", "work"],
            "walking": ["walking", "street", "park", "path"],
            "driving": ["car", "driving", "road", "traffic"]
        }

        # Check description and context
        for activity, keywords in activity_rules.items():
            if any(keyword in description_lower for keyword in keywords):
                activities.append(activity)
            elif any(keyword in objects for keyword in keywords):
                activities.append(activity)
            elif any(keyword in scenes for keyword in keywords):
                activities.append(activity)

        return list(set(activities))  # Remove duplicates

    def _calculate_confidence(self, description: str, scenes: List[str],
                            objects: List[str], activities: List[str]) -> float:
        """Calculate overall confidence score for analysis."""
        confidence = 0.0

        # Base confidence from successful analysis
        if description and description != "Unable to generate description":
            confidence += 0.3

        # Confidence from detected elements
        if scenes:
            confidence += 0.25
        if objects:
            confidence += 0.25
        if activities:
            confidence += 0.2

        # Quality bonus for rich analysis
        total_elements = len(scenes) + len(objects) + len(activities)
        if total_elements >= 5:
            confidence += 0.1
        elif total_elements >= 3:
            confidence += 0.05

        return min(confidence, 1.0)

    def _basic_content_analysis(self, image: Image.Image, photo_path: Path) -> ContentAnalysis:
        """Fallback basic analysis when ML models unavailable."""
        # Extract basic info from filename and path
        filename = photo_path.name.lower()
        parent_folder = photo_path.parent.name.lower()

        # Basic scene inference from path
        scenes = []
        if any(keyword in parent_folder for keyword in ["outdoor", "nature", "park"]):
            scenes.append("outdoor")
        elif any(keyword in parent_folder for keyword in ["indoor", "home", "house"]):
            scenes.append("indoor")

        # Basic object inference
        objects = []
        if "birthday" in parent_folder or "party" in parent_folder:
            objects.extend(["person", "cake"])

        # Basic activity inference
        activities = []
        if "vacation" in parent_folder or "trip" in parent_folder:
            activities.append("vacation")
        elif "birthday" in parent_folder:
            activities.append("celebration")

        # Perform face recognition if available (even in basic mode)
        people_detected, face_count = self._analyze_faces(photo_path)

        return ContentAnalysis(
            objects=objects,
            scenes=scenes,
            activities=activities,
            description="Basic analysis from filename and folder",
            confidence_score=0.3,
            analysis_model="Basic+Face Recognition" if self.face_recognizer else "Basic",
            people_detected=people_detected,
            face_count=face_count
        )

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

        # Aggregate data
        all_objects = []
        all_scenes = []
        all_activities = []
        total_confidence = 0.0

        for analysis in analyses.values():
            all_objects.extend(analysis.objects)
            all_scenes.extend(analysis.scenes)
            all_activities.extend(analysis.activities)
            total_confidence += analysis.confidence_score

        # Count frequencies
        from collections import Counter
        object_counts = Counter(all_objects)
        scene_counts = Counter(all_scenes)
        activity_counts = Counter(all_activities)

        return {
            "total_photos_analyzed": len(analyses),
            "average_confidence": total_confidence / len(analyses),
            "top_objects": object_counts.most_common(10),
            "top_scenes": scene_counts.most_common(5),
            "top_activities": activity_counts.most_common(5),
            "unique_objects": len(object_counts),
            "unique_scenes": len(scene_counts),
            "unique_activities": len(activity_counts)
        }

    def save_analysis_cache(self, cache_file: Path):
        """Save analysis cache to file."""
        try:
            # Convert ContentAnalysis objects to dicts for JSON serialization
            cache_data = {}
            for key, analysis in self.analysis_cache.items():
                cache_data[key] = {
                    "objects": analysis.objects,
                    "scenes": analysis.scenes,
                    "activities": analysis.activities,
                    "description": analysis.description,
                    "confidence_score": analysis.confidence_score,
                    "analysis_model": analysis.analysis_model
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

        # Clear GPU memory if used
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()