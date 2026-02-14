"""
Intelligent event naming using LLM and multi-signal analysis.

This module generates human-readable event names by analyzing photos, metadata,
locations, timing patterns, and content to create meaningful folder names.

For junior developers:
- Shows how to combine multiple AI signals for decision making
- Demonstrates prompt engineering for specific tasks
- Uses fallback strategies when APIs aren't available
- Implements caching for performance and cost optimization
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import re

# Optional LLM dependencies
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Local LLM support via Ollama
try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from .config import DATA_DIR
except ImportError:
    from config import DATA_DIR

# Vector database support
try:
    from .vector_database import VectorDatabase
    from .photo_vectorizer import PhotoVectorizer
    VECTOR_DB_AVAILABLE = True
except ImportError:
    try:
        from vector_database import VectorDatabase
        from photo_vectorizer import PhotoVectorizer
        VECTOR_DB_AVAILABLE = True
    except ImportError:
        VECTOR_DB_AVAILABLE = False

class EventNamer:
    """
    Generates intelligent event names using LLM and multi-signal analysis.

    This class takes a cluster of photos and generates a human-readable
    event name by analyzing multiple signals:
    - Temporal patterns (time of day, duration, frequency)
    - Location data (GPS coordinates, reverse geocoding)
    - Content analysis (objects, scenes, activities detected)
    - Calendar context (holidays, weekends, seasons)
    - Similar events in existing organization

    For junior developers:
    - This demonstrates "ensemble methods" - combining multiple AI systems
    - Uses prompt engineering to get consistent, useful results from LLM
    - Implements graceful degradation when LLM isn't available
    - Shows how to structure complex decision-making logic
    """

    def __init__(self, api_key: Optional[str] = None, enable_llm: bool = True,
                 ollama_model: str = "llama3.1:8b", ollama_url: str = "http://localhost:11434",
                 vector_db: Optional[Any] = None, photo_vectorizer: Optional[Any] = None):
        """
        Initialize the event namer.

        Args:
            api_key: OpenAI API key (optional, can use environment variable)
            enable_llm: Whether to use LLM for naming (fallback to rule-based)
            ollama_model: Local Ollama model to use (e.g., "llama3.1:8b", "phi3:mini")
            ollama_url: Ollama server URL (default: localhost)
            vector_db: Vector database instance for finding similar organized photos
            photo_vectorizer: Photo vectorizer for creating embeddings

        For junior developers:
        - API keys should never be hardcoded - use environment variables
        - Always provide fallback options when external services might fail
        - Initialize expensive resources (like API clients) lazily
        """
        self.logger = logging.getLogger(__name__)
        self.enable_llm = enable_llm

        # Setup dedicated LLM interaction logger
        self.llm_logger = self._setup_llm_logger()

        # LLM preferences: OpenAI -> Ollama -> Templates
        # Only use OpenAI if we have an API key
        self.use_openai = enable_llm and OPENAI_AVAILABLE and api_key is not None
        self.use_ollama = enable_llm and OLLAMA_AVAILABLE
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url

        # LLM clients (initialized lazily)
        self.openai_client = None
        self.api_key = api_key

        # Caching for performance and cost optimization
        self.naming_cache = {}

        # Use environment-aware cache file path
        try:
            from .environment_config import get_event_naming_cache_file
        except ImportError:
            from environment_config import get_event_naming_cache_file

        self.cache_file = get_event_naming_cache_file()
        self._load_cache()

        # Vector database for finding similar organized photos
        self.vector_db = vector_db
        self.photo_vectorizer = photo_vectorizer
        self.enable_vector_similarity = VECTOR_DB_AVAILABLE and vector_db is not None and photo_vectorizer is not None

        # Knowledge bases for intelligent naming
        self.holiday_patterns = self._load_holiday_patterns()
        self.activity_templates = self._load_activity_templates()
        self.location_nicknames = self._load_location_nicknames()

    def _setup_llm_logger(self) -> logging.Logger:
        """
        Setup dedicated logger for LLM prompt/response interactions.

        Returns:
            Configured logger for LLM interactions
        """
        llm_logger = logging.getLogger('llm_interactions')
        llm_logger.setLevel(logging.INFO)

        # Create logs directory if it doesn't exist
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)

        # Create file handler for LLM interactions
        log_file = log_dir / 'llm_prompts_responses.log'
        handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        handler.setLevel(logging.INFO)

        # Create detailed formatter
        formatter = logging.Formatter(
            '\n{"timestamp": "%(asctime)s", "level": "%(levelname)s"}\n%(message)s\n' + '='*80,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)

        # Remove existing handlers to avoid duplicates
        llm_logger.handlers = []
        llm_logger.addHandler(handler)

        # Don't propagate to root logger
        llm_logger.propagate = False

        return llm_logger

    def _initialize_llm_client(self) -> Tuple[bool, str]:
        """
        Lazy initialization of LLM client (OpenAI or Ollama).

        Returns:
            Tuple of (success, provider) where provider is "openai", "ollama", or "none"

        For junior developers:
        - Lazy loading means we only create expensive resources when needed
        - This pattern saves memory and startup time
        - Always handle potential failures in resource initialization
        - Try multiple providers in order of preference
        """
        if not self.enable_llm:
            return False, "none"

        # Try OpenAI first
        if self.use_openai and self.openai_client is None:
            try:
                if self.api_key:
                    self.openai_client = OpenAI(api_key=self.api_key)
                else:
                    # Try to use environment variable
                    self.openai_client = OpenAI()  # Uses OPENAI_API_KEY env var

                # Test the connection with a simple call
                self.openai_client.models.list()
                self.logger.info("OpenAI client initialized successfully")
                return True, "openai"

            except Exception as e:
                self.logger.warning(f"Could not initialize OpenAI client: {e}")
                self.use_openai = False

        # Try Ollama if OpenAI failed
        if self.use_ollama:
            print(f"DEBUG: Trying Ollama at {self.ollama_url} with model {self.ollama_model}")
            try:
                # Test Ollama connection
                response = requests.get(f"{self.ollama_url}/api/version", timeout=5)
                print(f"DEBUG: Ollama version check status: {response.status_code}")
                if response.status_code == 200:
                    # Test model availability
                    test_data = {
                        "model": self.ollama_model,
                        "prompt": "test",
                        "stream": False
                    }
                    print(f"DEBUG: Testing model with data: {test_data}")
                    test_response = requests.post(
                        f"{self.ollama_url}/api/generate",
                        json=test_data,
                        timeout=60
                    )
                    print(f"DEBUG: Model test response status: {test_response.status_code}")
                    if test_response.status_code == 200:
                        self.logger.info(f"Ollama client initialized successfully with model: {self.ollama_model}")
                        print(f"DEBUG: Ollama initialization SUCCESS!")
                        return True, "ollama"
                    else:
                        self.logger.warning(f"Ollama model {self.ollama_model} not available")
                        print(f"DEBUG: Model test failed with status {test_response.status_code}")
                else:
                    self.logger.warning("Ollama server not responding")
                    print(f"DEBUG: Ollama server not responding, status: {response.status_code}")
            except Exception as e:
                self.logger.warning(f"Could not connect to Ollama: {e}")
                print(f"DEBUG: Ollama connection exception: {e}")
                self.use_ollama = False

        # No LLM available
        self.logger.info("No LLM providers available, using template-based naming")
        return False, "none"

    def generate_event_name(self, cluster_data: Dict[str, Any]) -> str:
        """
        Generate an intelligent event name for a photo cluster.

        Args:
            cluster_data: Dictionary containing cluster information:
                - files: List of MediaFile objects
                - start_time: datetime when cluster starts
                - end_time: datetime when cluster ends
                - location_info: GPS and geocoding data
                - content_analysis: Objects, scenes, activities detected
                - confidence_score: How confident we are in the clustering

        Returns:
            Human-readable event name (e.g., "2024_10_25 - Halloween Party - Edmonton")

        For junior developers:
        - This is the main "orchestrator" method that coordinates everything
        - Notice how we try multiple approaches in order of sophistication
        - Each approach has fallbacks for when data isn't available
        """
        print(f"🎯 EVENT NAMING: Starting event name generation for cluster with {len(cluster_data.get('files', []))} files")

        # Create detailed diagnostic log
        self._log_diagnostics("=== EVENT NAMING DIAGNOSTICS START ===")

        # Log file information clearly
        files = cluster_data.get('files', []) or cluster_data.get('media_files', [])
        self._log_diagnostics(f"Cluster size: {len(files)} files")

        if files:
            # Log first few filenames for identification
            sample_size = min(5, len(files))
            self._log_diagnostics(f"Files in cluster (showing {sample_size}/{len(files)}):")
            for i, media_file in enumerate(files[:sample_size]):
                if hasattr(media_file, 'filename'):
                    filename = media_file.filename
                elif hasattr(media_file, 'path'):
                    filename = media_file.path.name if hasattr(media_file.path, 'name') else str(media_file.path)
                else:
                    filename = str(media_file)
                self._log_diagnostics(f"  [{i+1}] {filename}")
            if len(files) > sample_size:
                self._log_diagnostics(f"  ... and {len(files) - sample_size} more files")

        self._log_diagnostics(f"Cluster data keys: {list(cluster_data.keys())}")
        self._log_diagnostics(f"LLM enabled: {self.enable_llm}")
        self._log_diagnostics(f"Vector similarity enabled: {self.enable_vector_similarity}")

        try:
            # Extract key information from cluster
            print(f"🔍 EVENT NAMING: Building context from cluster data...")
            context = self._build_event_context(cluster_data)
            print(f"🔍 EVENT NAMING: Context built - location: {context.get('location', {}).get('city', 'Unknown')}")

            # Debug content analysis
            content = context.get('content', {})
            print(f"🔍 CONTENT DEBUG: Activities: {content.get('activities', [])[:3]}")
            print(f"🔍 CONTENT DEBUG: Scenes: {content.get('scenes', [])[:3]}")
            print(f"🔍 CONTENT DEBUG: Objects: {content.get('objects', [])[:3]}")
            print(f"🔍 CONTENT DEBUG: Primary activity: {content.get('primary_activity', 'unknown')}")

            # Log detailed context information
            self._log_diagnostics("--- EXTRACTED CONTEXT ---")
            for section_key, section_data in context.items():
                self._log_diagnostics(f"{section_key}: {section_data}")

            # Check cache first (save API costs and time)
            cache_key = self._generate_cache_key(context)
            print(f"💾 EVENT NAMING: Cache key: {cache_key}")
            self._log_diagnostics(f"Cache key: {cache_key}")
            if cache_key in self.naming_cache:
                cached_name = self.naming_cache[cache_key]
                print(f"💾 EVENT NAMING: Found cached name: {cached_name}")
                self._log_diagnostics(f"CACHE HIT: {cached_name}")
                self.logger.debug(f"Using cached name for similar event")
                return cached_name

            print(f"💾 EVENT NAMING: No cached name found, generating new one...")
            self._log_diagnostics("CACHE MISS - generating new name")

            # Try different naming approaches in order of sophistication
            event_name = None

            # Approach 1: LLM-based intelligent naming (most sophisticated)
            print(f"🤖 EVENT NAMING: LLM enabled: {self.enable_llm}")
            if self.enable_llm:
                print(f"🤖 EVENT NAMING: Attempting LLM-based naming...")
                self._log_diagnostics("--- ATTEMPTING LLM NAMING ---")
                event_name = self._generate_llm_name(context)
                print(f"🤖 EVENT NAMING: LLM result: {event_name}")
                self._log_diagnostics(f"LLM result: {event_name}")
            else:
                self._log_diagnostics("LLM naming DISABLED")

            # Approach 2: Template-based naming (DISABLED - generated poor generic names)
            # if not event_name:
            #     print(f"📋 EVENT NAMING: Attempting template-based naming...")
            #     event_name = self._generate_template_name(context)
            #     print(f"📋 EVENT NAMING: Template result: {event_name}")

            # Approach 3: Simple rule-based naming (DISABLED - generated poor generic names)
            # if not event_name:
            #     print(f"⚙️ EVENT NAMING: Attempting simple rule-based naming...")
            #     event_name = self._generate_simple_name(context)
            #     print(f"⚙️ EVENT NAMING: Simple result: {event_name}")

            print(f"✅ EVENT NAMING: Generated name before validation: {event_name}")

            # Check if we have a name to validate
            if not event_name:
                print(f"❌ EVENT NAMING: No name generated (LLM timeout/failure), skipping validation")
                self._log_diagnostics("NO EVENT NAME GENERATED - LLM timeout or failure")
                self._log_diagnostics("=== EVENT NAMING DIAGNOSTICS END ===")
                return None

            # Validate the name before caching
            is_valid = self._validate_event_name(event_name, context)
            print(f"🔍 EVENT NAMING: Validation result: {is_valid}")

            if is_valid:
                # Cache the result for similar future events
                print(f"💾 EVENT NAMING: Caching validated name: {event_name}")
                self.naming_cache[cache_key] = event_name
                self._save_cache()
                print(f"💾 EVENT NAMING: Cache saved successfully")
            else:
                # Name was rejected by validation - return None to indicate no good name found
                print(f"❌ EVENT NAMING: Name rejected by validation, no fallback used")
                event_name = None
                self.logger.info(f"Event name rejected by validation, no fallback applied")

            # Add final diagnostics
            if event_name:
                self._log_diagnostics(f"SUCCESS: Final event name: {event_name}")
            else:
                self._log_diagnostics("NO EVENT NAME GENERATED - returning None to skip event")

            self._log_diagnostics("=== EVENT NAMING DIAGNOSTICS END ===")
            print(f"🎉 EVENT NAMING: Final event name: {event_name}")
            self.logger.info(f"Generated event name: {event_name}")
            return event_name

        except Exception as e:
            print(f"💥 EVENT NAMING: Exception occurred: {e}")
            self.logger.error(f"Error generating event name: {e}")
            # Ultimate fallback - always return something reasonable
            fallback_name = self._generate_fallback_name(cluster_data)
            print(f"🆘 EVENT NAMING: Using ultimate fallback: {fallback_name}")
            return fallback_name


    def _validate_event_name(self, event_name: str, context: Dict[str, Any]) -> bool:
        """
        Validate event name for obvious issues before caching.

        Args:
            event_name: Generated event name
            context: Event context used for generation

        Returns:
            True if name is acceptable, False if it should be rejected
        """
        location = context['location']
        temporal = context['temporal']

        # Extract the descriptive part (after date) and location suffix
        parts = event_name.split(' - ')
        description = parts[1] if len(parts) > 1 else ''
        name_location = parts[2] if len(parts) > 2 else ''

        actual_location = location.get('city', '')

        print(f"🔍 VALIDATION DEBUG: Description: '{description}'")
        print(f"🔍 VALIDATION DEBUG: Location in name: '{name_location}'")
        print(f"🔍 VALIDATION DEBUG: Actual location: '{actual_location}'")

        # Validate location consistency: if we provided a city, LLM should use it
        # If actual_location exists, the name should contain it (not "Unknown" or different city)
        if actual_location:
            # Check if the actual location appears in the name
            if name_location and actual_location.lower() in name_location.lower():
                print(f"✅ VALIDATION DEBUG: Location matches - '{actual_location}' found in '{name_location}'")
            elif name_location.lower() == 'unknown':
                print(f"❌ VALIDATION DEBUG: LLM returned 'Unknown' when we provided '{actual_location}'")
                self.logger.warning(f"Rejecting name with 'Unknown' when location was provided: {event_name}")
                return False
            elif name_location and actual_location.lower() not in name_location.lower():
                print(f"❌ VALIDATION DEBUG: Location mismatch - expected '{actual_location}', got '{name_location}'")
                self.logger.warning(f"Rejecting name with wrong location: {event_name}")
                return False

        # Reject seasonal mismatches (only for obvious cases)
        if 'Beach' in description and temporal['season'] == 'winter':
            self.logger.warning(f"Rejecting seasonally inappropriate name: {event_name}")
            return False

        # Removed the generic terms validation as it was too restrictive
        print(f"✅ VALIDATION DEBUG: Name passed validation: {event_name}")
        return True

    def _contains_meta_text(self, event_name: str) -> bool:
        """
        Check if event name contains meta-text instead of actual event description.

        Args:
            event_name: Generated event name to check

        Returns:
            True if meta-text detected, False if clean
        """
        # Extract description part (after date)
        parts = event_name.split(' - ', 1)
        description = parts[1].lower() if len(parts) > 1 else event_name.lower()

        # Meta-text phrases that indicate LLM is explaining instead of naming
        meta_phrases = [
            "here are",
            "here is",
            "options for",
            "option for",
            "folder name",
            "short name",
            "few options",
            "could be",
            "suggestions",
            "suggest",
            "create",
            "name for",
            "photos from"
        ]

        return any(phrase in description for phrase in meta_phrases)

    def _build_event_context(self, cluster_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build comprehensive context for event naming.

        This method extracts and structures all available information about
        the photo cluster to help make intelligent naming decisions.

        Args:
            cluster_data: Raw cluster information

        Returns:
            Structured context dictionary

        For junior developers:
        - This is a "data preparation" step - crucial for good AI results
        - We normalize and structure messy real-world data
        - Missing data is handled gracefully with defaults
        - The better the context, the better the naming decisions
        """
        files = cluster_data.get('files', []) or cluster_data.get('media_files', [])
        start_time = cluster_data.get('start_time')
        end_time = cluster_data.get('end_time')
        location_info = cluster_data.get('location_info') or {}
        content_analysis = cluster_data.get('content_analysis') or {}

        # Calculate event characteristics
        duration = end_time - start_time if start_time and end_time else timedelta(0)

        # Use provided counts if available, otherwise calculate from files
        photo_count = cluster_data.get('photo_count', len([f for f in files if getattr(f, 'file_type', None) == 'photo']))
        video_count = cluster_data.get('video_count', len([f for f in files if getattr(f, 'file_type', None) == 'video']))

        # Temporal context
        temporal_context = {
            'date': start_time.strftime('%Y_%m_%d') if start_time else 'unknown_date',
            'time_of_day': self._classify_time_of_day(start_time) if start_time else 'unknown',
            'day_of_week': start_time.strftime('%A') if start_time else 'unknown',
            'duration_hours': duration.total_seconds() / 3600,
            'duration_category': self._classify_duration(duration),
            'season': self._get_season(start_time) if start_time else 'unknown',
            'is_weekend': start_time.weekday() >= 5 if start_time else False,
            'is_holiday': self._check_holiday(start_time) if start_time else False
        }

        # Location context - handle both location_info object and dominant_location string
        dominant_location = cluster_data.get('dominant_location', '')

        # Handle both location_info object and dictionary formats
        def safe_get_location_attr(obj, attr, default=''):
            if hasattr(obj, attr):
                return getattr(obj, attr, default)
            elif isinstance(obj, dict):
                return obj.get(attr, default)
            return default

        location_context = {
            'has_gps': bool(safe_get_location_attr(location_info, 'latitude')) or bool(cluster_data.get('gps_coordinates')),
            'city': safe_get_location_attr(location_info, 'city') or self._extract_city_from_location_string(dominant_location),
            'state': safe_get_location_attr(location_info, 'state'),
            'country': safe_get_location_attr(location_info, 'country'),
            'venue_type': self._classify_venue_type(location_info),
            'location_nickname': self._get_location_nickname(location_info) or self._extract_city_from_location_string(dominant_location),
            'full_location': dominant_location
        }

        # Content context - handle both content_analysis and direct content_tags
        content_tags = cluster_data.get('content_tags', [])

        content_context = {
            'objects': content_analysis.get('top_objects', []) or content_tags,
            'scenes': content_analysis.get('top_scenes', []) or content_tags,
            'activities': content_analysis.get('top_activities', []) or content_tags,
            'confidence': content_analysis.get('average_confidence', 0.0),
            'primary_activity': self._identify_primary_activity(content_analysis) or self._identify_activity_from_tags(content_tags),
            'event_type': self._classify_event_type(content_analysis, temporal_context) or self._classify_event_from_tags(content_tags),
            'content_tags': content_tags
        }

        # Media context
        media_context = {
            'total_files': len(files),
            'photo_count': photo_count,
            'video_count': video_count,
            'media_ratio': photo_count / max(1, photo_count + video_count),
            'capture_pattern': self._analyze_capture_pattern(files)
        }

        # People context - extract face recognition and people information
        people_detected = cluster_data.get('people_detected', [])
        face_count = cluster_data.get('metadata', {}).get('total_faces_detected', 0)
        people_consistency = cluster_data.get('metadata', {}).get('people_consistency_score', 0.0)

        people_context = {
            'people_detected': people_detected,
            'people_count': len(people_detected),
            'face_count': face_count,
            'people_consistency': people_consistency,
            'has_people': len(people_detected) > 0,
            'main_people': self._format_people_names(people_detected),
            'people_category': self._classify_people_category(len(people_detected), people_consistency)
        }

        # Vector similarity context - find similar organized photos
        similarity_context = self._analyze_similar_organized_photos(files, temporal_context, location_context)

        return {
            'temporal': temporal_context,
            'location': location_context,
            'content': content_context,
            'media': media_context,
            'people': people_context,
            'similarity': similarity_context,
            'raw_data': cluster_data  # Keep original data for reference
        }

    def _generate_llm_name(self, context: Dict[str, Any]) -> Optional[str]:
        """
        Generate event name using LLM (OpenAI GPT or Ollama).

        Args:
            context: Structured event context

        Returns:
            LLM-generated event name or None if failed

        For junior developers:
        - This demonstrates "prompt engineering" - how to ask AI for specific results
        - The prompt includes examples, constraints, and clear instructions
        - We parse and validate the AI response
        - Always have error handling for API calls
        - Shows how to support multiple LLM providers with the same interface
        """
        try:
            prompt = self._build_naming_prompt(context)
            raw_name = None

            print(f"🤖 LLM DEBUG: use_ollama={self.use_ollama}, use_openai={self.use_openai}")

            # Try Ollama directly (skip initialization test to avoid timeout issues)
            provider = "none"
            if self.use_ollama:
                print(f"🤖 LLM DEBUG: Attempting Ollama query with full detailed prompt...")
                raw_name = self._query_ollama(prompt)
                print(f"🤖 LLM DEBUG: Ollama result: {raw_name}")
                provider = "ollama"
            # TODO: Remove OpenAI support - no longer needed, Ollama provides local LLM
            elif self.use_openai:
                print(f"🤖 LLM DEBUG: Attempting OpenAI query...")
                success, init_provider = self._initialize_llm_client()
                print(f"🤖 LLM DEBUG: OpenAI init: success={success}, provider={init_provider}")
                if success and init_provider == "openai":
                    raw_name = self._query_openai(prompt)
                    print(f"🤖 LLM DEBUG: OpenAI result: {raw_name}")
                    provider = "openai"
            else:
                print(f"🤖 LLM DEBUG: No LLM provider configured")
                return None

            if not raw_name:
                print(f"🤖 LLM DEBUG: No result from {provider} provider")
                return None

            # For simple Ollama responses, skip validation since they're already formatted
            if provider == "ollama":
                self.logger.info(f"LLM (ollama) generated name: {raw_name}")
                return raw_name

            # Extract and clean the response for other providers
            clean_name = self._clean_event_name(raw_name)

            # Validate the name meets our requirements
            if self._validate_event_name(clean_name, context):
                self.logger.info(f"LLM ({provider}) generated name: {clean_name}")
                return clean_name
            else:
                self.logger.warning(f"LLM name failed validation: {raw_name}")
                return None

        except Exception as e:
            self.logger.warning(f"LLM naming failed: {e}")
            return None

    # TODO: Remove this method - OpenAI support no longer needed
    def _query_openai(self, prompt: str) -> Optional[str]:
        """Query OpenAI for event name generation."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant that creates descriptive, concise folder names for photo events. Follow the requested format exactly."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=100,
                temperature=0.3,  # Lower temperature for more consistent results
                timeout=10  # Don't wait too long
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.warning(f"OpenAI query failed: {e}")
            return None

    def _query_ollama(self, prompt: str) -> Optional[str]:
        """Query Ollama with the full detailed prompt."""
        try:
            # Log the prompt being sent
            self.llm_logger.info(f"PROMPT TO OLLAMA (model: {self.ollama_model}):\n{prompt}")

            data = {
                "model": self.ollama_model,
                "prompt": prompt,  # Use the provided prompt from _build_naming_prompt()
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 30  # Allow room for descriptive event names
                }
            }
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=data,
                timeout=300  # 5 minutes - quality over speed, give LLM time to process detailed prompt
            )
            if response.status_code == 200:
                result = response.json()
                ollama_response = result.get("response", "").strip()

                # Log the raw response from Ollama
                self.llm_logger.info(f"RESPONSE FROM OLLAMA:\n{ollama_response}")

                # Format the response as a proper folder name
                if ollama_response:
                    # Clean and format the response
                    clean_name = ollama_response.replace('"', '').replace("'", "").strip()

                    # Take only first line if multi-line
                    clean_name = clean_name.split('\n')[0].strip()

                    # Validate quality - reject if contains meta-text
                    if self._contains_meta_text(clean_name):
                        self.logger.warning(f"❌ LLM output contains meta-text, rejecting: {clean_name}")
                        self.llm_logger.warning(f"REJECTED (meta-text detected): {clean_name}")
                        return None

                    self.llm_logger.info(f"ACCEPTED (after cleaning): {clean_name}")
                    return clean_name

                self.llm_logger.warning("RESPONSE WAS EMPTY")
                return None
            else:
                self.logger.warning(f"Ollama returned status {response.status_code}")
                self.llm_logger.error(f"OLLAMA ERROR: Status {response.status_code}")
                return None

        except Exception as e:
            self.logger.warning(f"Ollama query failed: {e}")
            self.llm_logger.error(f"OLLAMA EXCEPTION: {str(e)}")
            return None

    def _build_naming_prompt(self, context: Dict[str, Any]) -> str:
        """
        Build a detailed prompt for the LLM to generate event names.

        This method creates a comprehensive prompt that gives the LLM
        all the context it needs to make intelligent naming decisions.

        Args:
            context: Structured event context

        Returns:
            Formatted prompt string

        For junior developers:
        - Good prompts are crucial for getting good AI results
        - Include examples of desired output format
        - Provide clear constraints and rules
        - Give context about the use case
        """
        temporal = context['temporal']
        location = context['location']
        content = context['content']
        media = context['media']

        prompt = f"""Create a descriptive folder name for a photo event with this information:

**Event Details:**
- Date: {temporal['date']}
- Time of day: {temporal['time_of_day']}
- Duration: {temporal['duration_category']} ({temporal['duration_hours']:.1f} hours)
- Day: {temporal['day_of_week']} ({'weekend' if temporal['is_weekend'] else 'weekday'})
- Season: {temporal['season']}
- Holiday: {'Yes' if temporal['is_holiday'] else 'No'}

**Location:**
- City: {location['city'] or 'Unknown'}
- Venue type: {location['venue_type']}
- GPS available: {'Yes' if location['has_gps'] else 'No'}

**Content Analysis:**
- Activities: {', '.join([item[0] if isinstance(item, tuple) else str(item) for item in content['activities'][:3]]) if content['activities'] else 'None detected'}
- Scenes: {', '.join([item[0] if isinstance(item, tuple) else str(item) for item in content['scenes'][:3]]) if content['scenes'] else 'None detected'}
- Objects: {', '.join([item[0] if isinstance(item, tuple) else str(item) for item in content['objects'][:3]]) if content['objects'] else 'None detected'}
- Event type: {content['event_type']}

**People Detected:**
- People: {context['people']['main_people'] if context['people']['has_people'] else 'None identified'}
- People count: {context['people']['people_count']}
- Face count: {context['people']['face_count']}
- Category: {context['people']['people_category']}

**Media:**
- Total files: {media['total_files']}
- Photos: {media['photo_count']}, Videos: {media['video_count']}

**Format Requirements:**
- Start with date: YYYY_MM_DD
- Add descriptive event name
- Add location if known
- Keep under 60 characters total
- Use title case
- No special characters except hyphens and underscores

**IMPORTANT CONSTRAINTS:**
- ONLY use the provided location: {location['city'] or 'Unknown'}
- DO NOT invent or change the location - use EXACTLY what is provided
- Be specific and descriptive, avoid generic terms like "Photoshoot", "Event Name", "Outing"
- Consider the season and weather for the location
- If no specific activity detected, use time/duration/setting context

**Examples for Edmonton (winter city):**
- 2024_01_15 - Indoor Family Gathering - Edmonton
- 2024_07_20 - Summer Festival - Edmonton
- 2024_11_10 - Autumn Photography Session - Edmonton
- 2024_12_25 - Christmas Morning - Home
- 2024_06_15 - Outdoor Concert - Edmonton
- 2024_03_20 - Spring Garden Visit - Edmonton

**Examples for other locations:**
- 2024_08_10 - Beach Day - Vancouver
- 2024_09_05 - Mountain Hiking - Calgary

**CRITICAL OUTPUT INSTRUCTION:**
Generate ONLY the folder name using the EXACT location provided above.
Do NOT output:
- Explanations or commentary
- Multiple options or lines
- Meta-text like "Here are some options..." or "I suggest..."
- Just the single folder name, nothing else

Output only the folder name now:"""

        return prompt

    def _generate_template_name(self, context: Dict[str, Any]) -> str:
        """
        Generate event name using template-based approach.

        This is a sophisticated fallback that uses predefined templates
        based on detected patterns, activities, and context.

        Args:
            context: Structured event context

        Returns:
            Template-based event name

        For junior developers:
        - This shows how to build rule-based AI as a fallback
        - Templates provide consistency and good results
        - Multiple templates can be combined for different scenarios
        """
        temporal = context['temporal']
        location = context['location']
        content = context['content']
        people = context['people']

        # Start with date
        base_name = temporal['date']

        # Determine event type and get appropriate template
        event_type = content['event_type']
        primary_activity = content['primary_activity']

        # Template selection logic
        if temporal['is_holiday']:
            template = self._get_holiday_template(temporal, content)
        elif event_type in self.activity_templates:
            template = self.activity_templates[event_type]
        elif primary_activity in self.activity_templates:
            template = self.activity_templates[primary_activity]
        elif temporal['is_weekend'] and temporal['duration_hours'] > 4:
            template = "Weekend Event"
        elif temporal['time_of_day'] == 'morning' and 'outdoor' in content['scenes']:
            template = "Morning Activity"
        elif temporal['time_of_day'] == 'evening' and 'indoor' in content['scenes']:
            template = "Evening Gathering"
        else:
            template = temporal['duration_category']

        # Add people if available and appropriate
        people_part = ""
        if people['has_people'] and people['people_count'] <= 4:
            # Only add people names for small groups
            people_part = f" - {people['main_people']}"

        # Add location if available
        location_part = ""
        if location['city']:
            if location['location_nickname']:
                location_part = f" - {location['location_nickname']}"
            else:
                location_part = f" - {location['city']}"

        # Assemble final name with priority: date - people - template - location
        if people_part:
            return f"{base_name}{people_part} {template}{location_part}"
        else:
            return f"{base_name} - {template}{location_part}"

    def _generate_simple_name(self, context: Dict[str, Any]) -> str:
        """
        Generate simple rule-based event name.

        This is the most basic naming approach that always works,
        providing reasonable names based on simple rules.

        Args:
            context: Structured event context

        Returns:
            Simple rule-based event name
        """
        temporal = context['temporal']
        location = context['location']

        base_name = temporal['date']

        # Simple rules for event description
        if temporal['duration_hours'] < 1:
            event_desc = "Quick Photos"
        elif temporal['duration_hours'] > 8:
            event_desc = "All Day Event"
        elif temporal['is_weekend']:
            event_desc = "Weekend Activity"
        else:
            event_desc = "Event"

        # Add location if available
        if location['city']:
            return f"{base_name} - {event_desc} - {location['city']}"
        else:
            return f"{base_name} - {event_desc}"

    def _generate_fallback_name(self, cluster_data: Dict[str, Any]) -> str:
        """
        Ultimate fallback name generation.

        This method always returns a valid name, even if all other
        approaches fail. It's the "safety net" of the naming system.

        Args:
            cluster_data: Raw cluster data

        Returns:
            Basic fallback name
        """
        start_time = cluster_data.get('start_time')
        files = cluster_data.get('files', [])

        if start_time:
            date_str = start_time.strftime('%Y_%m_%d')
        else:
            date_str = "unknown_date"

        return f"{date_str} - Photos ({len(files)} files)"

    def _analyze_similar_organized_photos(self, files: List[Any], temporal_context: Dict[str, Any], location_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze similar organized photos using vector database to inform naming.

        Args:
            files: List of MediaFile objects in current cluster
            temporal_context: Temporal information about the cluster
            location_context: Location information about the cluster

        Returns:
            Dictionary with similar photo analysis and naming suggestions
        """
        if not self.enable_vector_similarity:
            return {
                'enabled': False,
                'similar_photos': [],
                'naming_patterns': [],
                'confidence': 0.0
            }

        try:
            # Get photo files only
            photo_files = [f for f in files if getattr(f, 'file_type', None) == 'photo']
            if not photo_files:
                return {
                    'enabled': True,
                    'similar_photos': [],
                    'naming_patterns': [],
                    'confidence': 0.0,
                    'message': 'No photo files to analyze'
                }

            # Vectorize a representative sample of photos from the cluster
            sample_photos = photo_files[:3]  # Use first 3 photos as representatives
            vectorization_results = self.photo_vectorizer.vectorize_media_files(sample_photos)

            # Find similar organized photos for each sample
            similar_events = []
            all_event_folders = set()

            for photo_id, embedding in vectorization_results:
                if embedding is not None:
                    # Search for similar photos in organized collection
                    similar_photos = self.vector_db.search_similar_photos(
                        embedding,
                        n_results=10,
                        filter_organized=True
                    )

                    for similar_photo in similar_photos:
                        metadata = similar_photo['metadata']
                        if 'event_folder' in metadata:
                            event_folder = metadata['event_folder']
                            all_event_folders.add(event_folder)
                            similar_events.append({
                                'event_folder': event_folder,
                                'distance': similar_photo['distance'],
                                'similarity': 1.0 - similar_photo['distance'],  # Convert distance to similarity
                                'metadata': metadata
                            })

            # Extract naming patterns from similar event folders
            naming_patterns = self._extract_naming_patterns(list(all_event_folders))

            # Calculate confidence based on similarity scores
            if similar_events:
                avg_similarity = sum(event['similarity'] for event in similar_events) / len(similar_events)
                confidence = min(avg_similarity, 1.0)
            else:
                confidence = 0.0

            # Sort similar events by similarity
            similar_events.sort(key=lambda x: x['similarity'], reverse=True)

            self.logger.info(f"Found {len(similar_events)} similar photos from {len(all_event_folders)} event folders with avg confidence {confidence:.3f}")

            return {
                'enabled': True,
                'similar_photos': similar_events[:5],  # Top 5 most similar
                'naming_patterns': naming_patterns,
                'confidence': confidence,
                'total_similar_events': len(all_event_folders),
                'total_similar_photos': len(similar_events)
            }

        except Exception as e:
            self.logger.warning(f"Error analyzing similar organized photos: {e}")
            return {
                'enabled': True,
                'similar_photos': [],
                'naming_patterns': [],
                'confidence': 0.0,
                'error': str(e)
            }

    def _extract_naming_patterns(self, event_folders: List[str]) -> List[Dict[str, Any]]:
        """
        Extract naming patterns from similar event folder names.

        Args:
            event_folders: List of event folder names from similar photos

        Returns:
            List of naming patterns with frequency and examples
        """
        if not event_folders:
            return []

        patterns = {}

        for folder_name in event_folders:
            # Extract components from folder names
            # Expected format: YYYY_MM_DD - Event Name - Location
            parts = folder_name.split(' - ')

            if len(parts) >= 2:
                # Extract event type/activity from the middle part
                event_name = parts[1].strip() if len(parts) > 1 else ''
                location_name = parts[2].strip() if len(parts) > 2 else ''

                # Look for common patterns in event names
                if event_name:
                    # Normalize event name to find patterns
                    normalized_event = event_name.lower()

                    # Group similar event types
                    pattern_key = self._normalize_event_pattern(normalized_event)

                    if pattern_key not in patterns:
                        patterns[pattern_key] = {
                            'pattern': pattern_key,
                            'examples': [],
                            'count': 0,
                            'locations': set()
                        }

                    patterns[pattern_key]['examples'].append(folder_name)
                    patterns[pattern_key]['count'] += 1
                    if location_name:
                        patterns[pattern_key]['locations'].add(location_name)

        # Convert to list and sort by frequency
        pattern_list = []
        for pattern_data in patterns.values():
            pattern_data['locations'] = list(pattern_data['locations'])
            pattern_list.append(pattern_data)

        pattern_list.sort(key=lambda x: x['count'], reverse=True)
        return pattern_list[:5]  # Return top 5 patterns

    def _normalize_event_pattern(self, event_name: str) -> str:
        """
        Normalize event names to identify common patterns.

        Args:
            event_name: Raw event name from folder

        Returns:
            Normalized pattern string
        """
        # Common event pattern keywords
        if any(word in event_name for word in ['party', 'celebration', 'birthday']):
            return 'party'
        elif any(word in event_name for word in ['trip', 'vacation', 'travel', 'visit']):
            return 'trip'
        elif any(word in event_name for word in ['dinner', 'lunch', 'breakfast', 'meal']):
            return 'meal'
        elif any(word in event_name for word in ['walk', 'hike', 'outdoor', 'park']):
            return 'outdoor'
        elif any(word in event_name for word in ['work', 'meeting', 'office']):
            return 'work'
        elif any(word in event_name for word in ['family', 'gathering', 'reunion']):
            return 'family'
        elif any(word in event_name for word in ['shopping', 'store', 'mall']):
            return 'shopping'
        elif any(word in event_name for word in ['sports', 'game', 'match']):
            return 'sports'
        else:
            # Return first significant word
            words = event_name.split()
            for word in words:
                if len(word) > 3 and word not in ['the', 'and', 'with', 'from']:
                    return word
            return 'event'

    # Helper methods for context analysis
    def _classify_time_of_day(self, dt: datetime) -> str:
        """Classify time of day into categories."""
        hour = dt.hour
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'

    def _classify_duration(self, duration: timedelta) -> str:
        """Classify event duration into categories."""
        hours = duration.total_seconds() / 3600
        if hours < 0.5:
            return 'Quick Event'
        elif hours < 2:
            return 'Short Event'
        elif hours < 6:
            return 'Medium Event'
        elif hours < 12:
            return 'Long Event'
        else:
            return 'Extended Event'

    def _get_season(self, dt: datetime) -> str:
        """Determine season from date."""
        month = dt.month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'

    def _check_holiday(self, dt: datetime) -> bool:
        """Check if date is a major holiday."""
        # Simple holiday detection - can be expanded
        month, day = dt.month, dt.day
        holidays = [
            (1, 1),   # New Year's Day
            (2, 14),  # Valentine's Day
            (7, 1),   # Canada Day
            (7, 4),   # Independence Day
            (10, 31), # Halloween
            (12, 25), # Christmas
            (12, 31), # New Year's Eve
        ]
        return (month, day) in holidays

    def _classify_venue_type(self, location_info: Any) -> str:
        """Classify venue type from location information."""
        # This would be more sophisticated with full geocoding data
        return getattr(location_info, 'venue_type', 'unknown')

    def _get_location_nickname(self, location_info: Any) -> str:
        """Get friendly nickname for location."""
        city = getattr(location_info, 'city', '')
        if city in self.location_nicknames:
            return self.location_nicknames[city]
        return city

    def _extract_city_from_location_string(self, location_string: str) -> str:
        """Extract city name from a location string like 'Edmonton, Alberta, Canada'."""
        if not location_string:
            return ''

        # Split by comma and take the first part (usually the city)
        parts = [part.strip() for part in location_string.split(',')]
        if parts:
            return parts[0]
        return ''

    def _identify_primary_activity(self, content_analysis: Dict[str, Any]) -> str:
        """Identify the primary activity from content analysis."""
        activities = content_analysis.get('top_activities', [])
        if activities:
            # top_activities is a list of tuples: [('activity_name', count), ...]
            # Extract just the activity name from the first tuple
            first_activity = activities[0]
            if isinstance(first_activity, tuple):
                return first_activity[0]  # Get activity name from tuple
            return str(first_activity)  # Fallback to string conversion
        return 'unknown'

    def _identify_activity_from_tags(self, content_tags: List[str]) -> str:
        """Identify primary activity from content tags."""
        if not content_tags:
            return 'general'

        # Map common tags to activities
        tag_to_activity = {
            'outdoor': 'outdoor_activity',
            'nature': 'outdoor_activity',
            'indoor': 'indoor_activity',
            'celebration': 'party',
            'costume': 'party',
            'family': 'family_gathering',
            'travel': 'travel',
            'urban': 'city_exploration',
            'food': 'dining'
        }

        for tag in content_tags:
            if tag in tag_to_activity:
                return tag_to_activity[tag]

        return content_tags[0] if content_tags else 'general'

    def _classify_event_from_tags(self, content_tags: List[str]) -> str:
        """Classify event type from content tags."""
        if not content_tags:
            return 'general'

        # Map tags to event types
        for tag in content_tags:
            if tag in ['celebration', 'costume', 'party']:
                return 'celebration'
            elif tag in ['outdoor', 'nature']:
                return 'outdoor_event'
            elif tag in ['indoor', 'family']:
                return 'indoor_event'
            elif tag in ['travel', 'urban']:
                return 'travel_event'

        return 'general'

    def _classify_event_type(self, content_analysis: Dict[str, Any],
                           temporal_context: Dict[str, Any]) -> str:
        """Classify overall event type."""
        activities = content_analysis.get('top_activities', [])
        objects = content_analysis.get('top_objects', [])

        # Pattern matching for event types
        if 'celebration' in activities or 'cake' in objects:
            return 'celebration'
        elif 'vacation' in activities:
            return 'vacation'
        elif 'eating' in activities or 'food' in objects:
            return 'dining'
        elif temporal_context['is_weekend'] and temporal_context['duration_hours'] > 4:
            return 'weekend_activity'
        else:
            return 'general'

    def _analyze_capture_pattern(self, files: List) -> str:
        """Analyze how photos were captured (burst, spread out, etc.)."""
        if len(files) < 2:
            return 'single'

        # Calculate time gaps between photos
        print(f"🐛 DEBUG: Calculating time gaps for {len(files)} files")
        if files:
            file_type = files[0].file_type if hasattr(files[0], 'file_type') else 'unknown'
            print(f"🐛 DEBUG: First file type: {file_type}")

        # Handle both MediaFile objects and dictionary formats
        times = []
        for f in files:
            if hasattr(f, 'date'):
                # MediaFile object
                times.append(f.date)
            elif isinstance(f, dict):
                # Dictionary format - try different timestamp keys
                timestamp = f.get('timestamp') or f.get('date') or f.get('time')
                if timestamp:
                    times.append(timestamp)
                else:
                    print(f"🐛 DEBUG: Dictionary file has no timestamp: {f}")
            else:
                print(f"🐛 DEBUG: Unknown file type: {type(f)} - {f}")

        if not times:
            print(f"🐛 DEBUG: No valid timestamps found, returning 'unknown'")
            return 'unknown'

        times = sorted(times)
        gaps = [(times[i+1] - times[i]).total_seconds() for i in range(len(times)-1)]
        avg_gap = sum(gaps) / len(gaps)

        if avg_gap < 60:  # Less than 1 minute average
            return 'burst'
        elif avg_gap < 600:  # Less than 10 minutes
            return 'continuous'
        else:
            return 'sporadic'

    # Utility methods
    def _generate_cache_key(self, context: Dict[str, Any]) -> str:
        """Generate cache key for similar events."""
        # Create a key based on major characteristics
        temporal = context['temporal']
        location = context['location']
        content = context['content']

        key_parts = [
            temporal['time_of_day'],
            temporal['duration_category'],
            temporal['day_of_week'],
            'weekend' if temporal['is_weekend'] else 'weekday',
            location['city'],
            content['event_type'],
            content['primary_activity']
        ]

        return "|".join(str(part) for part in key_parts)

    def _clean_event_name(self, name: str) -> str:
        """Clean and validate event name."""
        # Remove extra whitespace and standardize format
        name = re.sub(r'\s+', ' ', name.strip())

        # Ensure proper format
        if not name.startswith('20'):  # Doesn't start with year
            # Try to extract date if present
            date_match = re.search(r'20\d{2}_\d{2}_\d{2}', name)
            if date_match:
                date_part = date_match.group()
                rest = name.replace(date_part, '').strip(' -')
                name = f"{date_part} - {rest}" if rest else date_part

        return name


    # Configuration loading methods
    def _load_holiday_patterns(self) -> Dict[str, str]:
        """Load holiday naming patterns."""
        return {
            'christmas': 'Christmas Celebration',
            'halloween': 'Halloween Party',
            'thanksgiving': 'Thanksgiving Dinner',
            'birthday': 'Birthday Party',
            'wedding': 'Wedding Celebration',
            'graduation': 'Graduation Ceremony'
        }

    def _load_activity_templates(self) -> Dict[str, str]:
        """Load activity-based naming templates."""
        return {
            'celebration': 'Celebration',
            'vacation': 'Vacation Day',
            'dining': 'Dinner Event',
            'outdoor': 'Outdoor Activity',
            'shopping': 'Shopping Trip',
            'sports': 'Sports Event',
            'work': 'Work Event',
            'family': 'Family Gathering'
        }

    def _load_location_nicknames(self) -> Dict[str, str]:
        """Load location nicknames."""
        return {
            'Edmonton': 'Edmonton',
            'Calgary': 'Calgary',
            'Vancouver': 'Vancouver',
            'Toronto': 'Toronto'
        }

    def _get_holiday_template(self, temporal: Dict[str, Any],
                            content: Dict[str, Any]) -> str:
        """Get holiday-specific template."""
        date = temporal['date']
        month_day = date.split('_')[1:3]  # Get MM_DD

        # Simple holiday mapping
        if month_day == ['12', '25']:
            return 'Christmas Morning'
        elif month_day == ['10', '31']:
            return 'Halloween Party'
        elif month_day == ['01', '01']:
            return 'New Year Celebration'
        else:
            return 'Holiday Celebration'

    # Cache management
    def _load_cache(self):
        """Load naming cache from file."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    self.naming_cache = json.load(f)
                self.logger.debug(f"Loaded {len(self.naming_cache)} cached names")
        except Exception as e:
            self.logger.warning(f"Could not load naming cache: {e}")
            self.naming_cache = {}

    def _save_cache(self):
        """Save naming cache to file."""
        try:
            print(f"💾 CACHE DEBUG: _save_cache() called")
            print(f"💾 CACHE DEBUG: Cache file path: {self.cache_file}")
            print(f"💾 CACHE DEBUG: Cache contents: {len(self.naming_cache)} entries")
            if self.naming_cache:
                print(f"💾 CACHE DEBUG: Sample cache entry: {list(self.naming_cache.items())[0]}")

            # Ensure data directory exists
            DATA_DIR.mkdir(exist_ok=True)
            print(f"💾 CACHE DEBUG: Data directory created/verified: {DATA_DIR}")

            with open(self.cache_file, 'w') as f:
                json.dump(self.naming_cache, f, indent=2)
            print(f"💾 CACHE DEBUG: Successfully wrote cache to file")

            # Verify the file was actually written
            with open(self.cache_file, 'r') as f:
                saved_data = json.load(f)
            print(f"💾 CACHE DEBUG: Verification - file contains {len(saved_data)} entries")

        except Exception as e:
            print(f"💥 CACHE DEBUG: Exception in _save_cache(): {e}")
            self.logger.warning(f"Could not save naming cache: {e}")
            import traceback
            traceback.print_exc()

    def _log_diagnostics(self, message: str):
        """Log detailed diagnostics to a separate diagnostics file."""
        try:
            diagnostics_file = DATA_DIR / "event_naming_diagnostics.log"

            # Ensure data directory exists
            DATA_DIR.mkdir(parents=True, exist_ok=True)

            with open(diagnostics_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().isoformat()
                f.write(f"[{timestamp}] {message}\n")
        except Exception as e:
            # Don't let diagnostic logging break the main functionality
            self.logger.warning(f"Could not write diagnostics: {e}")

    def _format_people_names(self, people_detected: List[str]) -> str:
        """Format people names for event naming.

        Args:
            people_detected: List of detected people names

        Returns:
            Formatted string of people names
        """
        if not people_detected:
            return ""

        if len(people_detected) == 1:
            return people_detected[0]
        elif len(people_detected) == 2:
            return f"{people_detected[0]} & {people_detected[1]}"
        elif len(people_detected) <= 4:
            return ", ".join(people_detected[:-1]) + f" & {people_detected[-1]}"
        else:
            return f"{people_detected[0]} & {len(people_detected)-1} others"

    def _classify_people_category(self, people_count: int, consistency: float) -> str:
        """Classify the people category for event naming.

        Args:
            people_count: Number of unique people detected
            consistency: People consistency score (0.0-1.0)

        Returns:
            People category string
        """
        if people_count == 0:
            return "no_people"
        elif people_count == 1 and consistency > 0.7:
            return "solo"
        elif people_count == 2 and consistency > 0.6:
            return "couple"
        elif people_count <= 4 and consistency > 0.5:
            return "small_group"
        elif people_count <= 8:
            return "group"
        else:
            return "large_group"

    def cleanup(self):
        """Clean up resources."""
        self._save_cache()