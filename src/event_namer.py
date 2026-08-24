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

# Venue resolution (fills in venue names reverse geocoding didn't surface)
try:
    from .venue_resolver import VenueResolver, VenueSearchError
    VENUE_RESOLVER_AVAILABLE = True
except ImportError:
    try:
        from venue_resolver import VenueResolver, VenueSearchError
        VENUE_RESOLVER_AVAILABLE = True
    except ImportError:
        VENUE_RESOLVER_AVAILABLE = False

# Naming prompt building blocks shared across the three mode-specific
# prompts (home/away/unknown) - see EventNamer._build_naming_prompt. Split
# into three prompts instead of one with home-vs-away conditional logic,
# because the conditional kept bleeding across modes: fixing an away-city
# bug broke home-city clusters, which then got rejected by validation and
# produced no name at all. Resolving mode in Python and handing each prompt
# only the ONE instruction that actually applies removes the gate entirely.
_NAMING_PROMPT_INTRO = """You are helping organize a personal photo library. Photos have already
been grouped into this cluster based on when and where they were taken - all
of them belong in one folder together. Below is everything known about the
cluster: timing, location, a description of each photo, and any people
identified. Use this information to generate a good, human-friendly folder
name."""

_NAMING_VENUE_CONSTRAINTS = """- The raw geo data above may include a more specific venue name than the City field (e.g. a mall, restaurant, or business) - if it names a real, specific place, use that actual name rather than inventing a generic category description from the photos; only invent a generic setting description when no real venue name is available
- If the venue is a shopping center/mall, prefer the mall's own name over the name or category of an individual store inside it (e.g. use "Capilano Mall", not "Electronics Store" or a specific store's brand name) - the mall name is more useful and recognizable for organizing photos than a specific store visit. Still include the general activity from the photos alongside the mall name rather than dropping it (e.g. "Furniture Shopping at Capilano Mall" - not just "Capilano Mall" alone) - but keep the activity general and safely supported by what's shown (e.g. "Furniture Shopping", not "Interior Design Consultation" - browsing furniture displays does not confirm a formal consultation took place)"""

_NAMING_GENERAL_CONSTRAINTS = """- Be specific and descriptive, avoid generic terms like "Photoshoot", "Event Name", "Outing"
- Prefer the photo descriptions and people above over generic season/time labels
- If the Date/Holiday above indicates a specific occasion, prefer that over decor visible in the photos when they conflict (e.g. a Christmas tree still up in photos taken on New Year's Day - trust the date, not the decor)
- Consider the season and weather for the location
- If no specific activity detected, use time/duration/setting context"""

_NAMING_OUTPUT_INSTRUCTIONS = """**CRITICAL OUTPUT INSTRUCTION:**
Generate ONLY the folder name using the EXACT location provided above.
Do NOT output:
- Explanations or commentary
- Multiple options or lines
- Meta-text like "Here are some options..." or "I suggest..."
- Just the single folder name, nothing else

Output only the folder name now:"""

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
                 ollama_model: Optional[str] = None, ollama_url: str = "http://localhost:11434",
                 vector_db: Optional[Any] = None, photo_vectorizer: Optional[Any] = None,
                 home_city: Optional[str] = None, home_state: Optional[str] = None,
                 enable_venue_search: bool = True):
        """
        Initialize the event namer.

        Args:
            api_key: OpenAI API key (optional, can use environment variable)
            enable_llm: Whether to use LLM for naming (fallback to rule-based)
            ollama_model: Local Ollama model to use (default: PHOTO_FILTER_NAMING_MODEL
                env var, e.g. "llama3.1:8b", "gemma3:12b")
            ollama_url: Ollama server URL (default: localhost)
            vector_db: Vector database instance for finding similar organized photos
            photo_vectorizer: Photo vectorizer for creating embeddings
            home_city: User's home city (default: PHOTO_FILTER_HOME_CITY env var).
                Events there don't need the city stated in the name; events
                elsewhere do.
            home_state: User's home state/province (default: PHOTO_FILTER_HOME_STATE
                env var). Disambiguates home_city from same-named cities elsewhere.
            enable_venue_search: Whether to look up a specific venue name (via
                src/venue_resolver.py) when reverse geocoding didn't surface
                one - e.g. a ski hill or restaurant with no business name in
                the geocoded address. Requires network access (Ollama +
                OpenStreetMap Overpass API); fails closed (no venue name
                added) on any error.

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
        self.ollama_url = ollama_url

        # LLM clients (initialized lazily)
        self.openai_client = None
        self.api_key = api_key

        # Caching for performance and cost optimization
        self.naming_cache = {}

        # Use environment-aware cache file path
        try:
            from .environment_config import (get_event_naming_cache_file, get_home_city,
                                              get_home_state, get_naming_model)
        except ImportError:
            from environment_config import (get_event_naming_cache_file, get_home_city,
                                             get_home_state, get_naming_model)

        self.cache_file = get_event_naming_cache_file()
        self._load_cache()
        self.home_city = home_city or get_home_city()
        self.home_state = home_state or get_home_state()
        self.ollama_model = ollama_model or get_naming_model()

        # Venue resolution: fills in a venue name (e.g. a ski hill or
        # restaurant) when reverse geocoding didn't surface one
        self.venue_resolver = (
            VenueResolver(ollama_url=self.ollama_url)
            if (self.use_ollama and enable_venue_search and VENUE_RESOLVER_AVAILABLE)
            else None
        )

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
        # Get files from either key (cluster_data uses 'media_files')
        files = cluster_data.get('files', []) or cluster_data.get('media_files', [])
        print(f"🎯 EVENT NAMING: Starting event name generation for cluster with {len(files)} files")

        # Create detailed diagnostic log
        self._log_diagnostics("=== EVENT NAMING DIAGNOSTICS START ===")
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

            # Check cache first (save API costs and time). A None key means
            # there's no real content signal to key on (issue #76) - always
            # skip the cache in that case rather than risk a collision with
            # an unrelated cluster.
            cache_key = self._generate_cache_key(context)
            print(f"💾 EVENT NAMING: Cache key: {cache_key}")
            self._log_diagnostics(f"Cache key: {cache_key}")
            if cache_key is not None and cache_key in self.naming_cache:
                cached_name = self.naming_cache[cache_key]
                print(f"💾 EVENT NAMING: Found cached name: {cached_name}")
                self._log_diagnostics(f"CACHE HIT: {cached_name}")
                self.logger.debug(f"Using cached name for similar event")
                return cached_name

            print(f"💾 EVENT NAMING: No cached name found, generating new one...")
            self._log_diagnostics("CACHE MISS - generating new name")

            # Resolve a specific venue name (e.g. a ski hill or restaurant)
            # if reverse geocoding didn't surface one - only runs on a cache
            # miss, since it's the expensive path (network calls)
            if self.venue_resolver and not self._resolve_venue(context):
                print(f"❌ EVENT NAMING: Venue search failed, skipping this cycle")
                self._log_diagnostics("=== EVENT NAMING DIAGNOSTICS END ===")
                return None

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
                # Only cache results with sufficient content confidence
                # This prevents low-quality results from polluting the cache.
                # content_confidence is caption-generation confidence, not
                # scene/object detection confidence, so it does NOT catch the
                # empty-scenes/objects case - that's what the None cache_key
                # guard below is for (issue #76).
                content_confidence = context['content'].get('confidence', 0.0)
                min_cache_confidence = 0.5

                if cache_key is not None and content_confidence >= min_cache_confidence:
                    # Cache the result for similar future events
                    print(f"💾 EVENT NAMING: Caching validated name: {event_name} (confidence: {content_confidence:.2f})")
                    self.naming_cache[cache_key] = event_name
                    self._save_cache()
                    print(f"💾 EVENT NAMING: Cache saved successfully")
                else:
                    print(f"💾 EVENT NAMING: Skipping cache (confidence {content_confidence:.2f} < {min_cache_confidence})")
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

        # Extract the descriptive part (after date) for the seasonal check
        # below. Location is no longer expected in a fixed third segment -
        # home events correctly omit it, away events fold it into the
        # description (issue #72) - so the hallucination checks below scan
        # the whole name instead of a specific segment (issue #74).
        parts = event_name.split(' - ')
        description = parts[1] if len(parts) > 1 else ''

        actual_location = (location.get('city') or '').strip()
        name_lower = event_name.lower()

        print(f"VALIDATION DEBUG: Description: '{description}'")
        print(f"VALIDATION DEBUG: Actual location: '{actual_location}'")

        if actual_location:
            actual_lower = actual_location.lower()
            home_lower = self.home_city.lower()
            is_home_event = actual_lower == home_lower

            # A venue can legitimately have the home city baked into its own
            # name (e.g. "Edmonton EXPO Centre") - only check the venue's own
            # name field, not the full geocoded address, since almost every
            # address's display_name ends with "..., Edmonton, ..." regardless
            # of what's actually at that location (issue #72/#74)
            raw_geo = location.get('raw_geo') or {}
            venue_own_name = (raw_geo.get('name') or '').lower()
            home_city_in_venue_name = home_lower in venue_own_name

            # LLM used a placeholder instead of the location we gave it
            if re.search(r'\bunknown\b', name_lower) and actual_lower not in name_lower:
                print(f"VALIDATION DEBUG: Rejecting - 'Unknown' used when '{actual_location}' was provided")
                self.logger.warning(f"Rejecting name with 'Unknown' when location was provided: {event_name}")
                return False

            # Home events were told not to state the city - if it shows up
            # anyway, the LLM ignored the naming guidance
            if is_home_event and not home_city_in_venue_name and re.search(rf'\b{re.escape(home_lower)}\b', name_lower):
                print(f"VALIDATION DEBUG: Rejecting - home city '{self.home_city}' stated when guidance said not to")
                self.logger.warning(f"Rejecting home-city name that states the city anyway: {event_name}")
                return False

            # Away events can't also be home - stating the home city while
            # actually elsewhere is a direct contradiction (the original
            # issue #14 failure mode: real GPS location replaced with a
            # different one the LLM defaulted to)
            if not is_home_event and not home_city_in_venue_name and re.search(rf'\b{re.escape(home_lower)}\b', name_lower):
                print(f"VALIDATION DEBUG: Rejecting - states home city '{self.home_city}' while actually in '{actual_location}'")
                self.logger.warning(f"Rejecting name stating home city '{self.home_city}' for an away event in '{actual_location}': {event_name}")
                return False

        # Reject seasonal mismatches (only for obvious cases)
        if 'Beach' in description and temporal['season'] == 'winter':
            self.logger.warning(f"Rejecting seasonally inappropriate name: {event_name}")
            return False

        print(f"VALIDATION DEBUG: Name passed validation: {event_name}")
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

    def _resolve_venue(self, context: Dict[str, Any]) -> bool:
        """Fill in a specific venue name (e.g. a ski hill or restaurant)
        when reverse geocoding didn't surface one, using self.venue_resolver.
        Mutates context['location']['raw_geo']['name'] in place if a venue
        is found.

        Returns:
            True if naming should proceed (a venue was resolved, or none
            was needed). False if the venue search itself failed (not just
            "found nothing") and naming should be skipped this cycle rather
            than proceed with an unenriched raw_geo that could surface a
            misleading field (e.g. a bare street address) as if it were a
            confirmed venue - see VenueSearchError in venue_resolver.py.
        """
        location = context['location']
        raw_geo = location.get('raw_geo')
        captions = context['content'].get('sample_captions') or []
        if not raw_geo or not captions:
            return True

        try:
            confirmed_venue = self.venue_resolver.resolve(raw_geo, captions)
        except VenueSearchError as e:
            self.logger.warning(f"Venue search failed, skipping naming this cycle: {e}")
            self._log_diagnostics(f"VENUE SEARCH FAILED - skipping naming: {e}")
            return False

        if confirmed_venue:
            raw_geo['name'] = confirmed_venue
            location['raw_geo'] = raw_geo
        return True

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
        holiday_name = self._check_holiday(start_time) if start_time else ''
        temporal_context = {
            'date': start_time.strftime('%Y_%m_%d') if start_time else 'unknown_date',
            'time_of_day': self._classify_time_of_day(start_time) if start_time else 'unknown',
            'day_of_week': start_time.strftime('%A') if start_time else 'unknown',
            'duration_hours': duration.total_seconds() / 3600,
            'duration_category': self._classify_duration(duration),
            'season': self._get_season(start_time) if start_time else 'unknown',
            'is_weekend': start_time.weekday() >= 5 if start_time else False,
            'is_holiday': bool(holiday_name),
            'holiday_name': holiday_name
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

        # GPS spread across the cluster - distinguishes a single-venue event
        # from a multi-location day or trip (issue #72)
        gps_coords = cluster_data.get('gps_coordinates') or []
        gps_spread_km = None
        if len(gps_coords) >= 2:
            lat_km = (max(c[0] for c in gps_coords) - min(c[0] for c in gps_coords)) * 111.0
            lon_km = (max(c[1] for c in gps_coords) - min(c[1] for c in gps_coords)) * 68.0
            gps_spread_km = max(lat_km, lon_km)

        # Neighbourhood/suburb from the reverse-geocode raw data when present
        raw_geo = safe_get_location_attr(location_info, 'raw_data', {}) or {}
        geo_address = raw_geo.get('address', {}) if isinstance(raw_geo, dict) else {}
        area = geo_address.get('neighbourhood') or geo_address.get('quarter') or geo_address.get('suburb') or ''

        location_context = {
            'has_gps': bool(safe_get_location_attr(location_info, 'latitude')) or bool(cluster_data.get('gps_coordinates')),
            'city': safe_get_location_attr(location_info, 'city') or self._extract_city_from_location_string(dominant_location),
            'state': safe_get_location_attr(location_info, 'state'),
            'country': safe_get_location_attr(location_info, 'country'),
            'raw_geo': raw_geo if isinstance(raw_geo, dict) and raw_geo else None,
            'location_nickname': self._get_location_nickname(location_info) or self._extract_city_from_location_string(dominant_location),
            'full_location': dominant_location,
            'area': area,
            'gps_spread_km': gps_spread_km
        }

        # Content context - handle both content_analysis and direct content_tags.
        # content_tags are OBJECT tags, so they may only stand in for objects -
        # falling back to them for scenes/activities put objects in the
        # activities field ("Activities: person, phone", issue #72).
        content_tags = cluster_data.get('content_tags', [])

        content_context = {
            'objects': content_analysis.get('top_objects', []) or content_tags,
            'scenes': content_analysis.get('top_scenes', []),
            'activities': content_analysis.get('top_activities', []),
            'confidence': content_analysis.get('average_confidence', 0.0),
            'primary_activity': self._identify_primary_activity(content_analysis) or self._identify_activity_from_tags(content_tags),
            'event_type': self._classify_event_type(content_analysis, temporal_context) or self._classify_event_from_tags(content_tags),
            'content_tags': content_tags,
            'sample_captions': content_analysis.get('sample_captions', [])
        }

        # Media context
        media_context = {
            'total_files': len(files),
            'photo_count': photo_count,
            'video_count': video_count,
            'media_ratio': photo_count / max(1, photo_count + video_count),
            'capture_pattern': self._analyze_capture_pattern(files)
        }

        # People context - extract face recognition and people information.
        # total_faces_detected is often unpopulated; never report fewer faces
        # than identified people (the "Face count: 0" contradiction, issue #72)
        people_detected = cluster_data.get('people_detected', [])
        face_count = max(cluster_data.get('metadata', {}).get('total_faces_detected', 0),
                         len(people_detected))
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
                "think": False,  # thinking models otherwise burn num_predict on hidden
                                  # reasoning tokens and return an empty response
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

    def _determine_location_mode(self, location: Dict[str, Any]) -> str:
        """Resolve home/away/unknown from data already available - GPS
        presence and city vs home_city - so the naming prompt never has to
        ask the LLM to correctly gate a home-vs-away conditional itself."""
        if not location.get('has_gps'):
            return 'unknown'
        city = (location.get('city') or '').strip().lower()
        if city == (self.home_city or '').strip().lower():
            return 'home'
        return 'away'

    def _build_naming_event_details_block(self, temporal: Dict[str, Any]) -> str:
        return f"""**Event Details:**
- Date: {temporal['date']}
- Time of day: {temporal['time_of_day']}
- Duration: {temporal['duration_category']} ({temporal['duration_hours']:.1f} hours)
- Day: {temporal['day_of_week']} ({'weekend' if temporal['is_weekend'] else 'weekday'})
- Season: {temporal['season']}
- Holiday: {temporal.get('holiday_name') or ('Yes' if temporal['is_holiday'] else 'No')}"""

    def _build_naming_photos_block(self, content: Dict[str, Any]) -> str:
        # Numbered so the LLM knows exactly how many photos are in the
        # cluster and that each caption maps to one of them (issue #72)
        captions = content.get('sample_captions') or []
        if not captions:
            return ""
        total = len(captions)
        lines = "\n".join(f'- Photo {i+1} of {total}: "{c}"' for i, c in enumerate(captions))
        return f"\n\n**Photos in this cluster ({total} total):**\n{lines}"

    def _build_naming_people_block(self, people: Dict[str, Any]) -> str:
        # Naming guidance only for solo/couple (<=2) - past that, "use their
        # name" breaks down grammatically for a possessive folder name
        # ("Jane, John & Amy's Birthday" reads like only Amy's) and a name
        # list doesn't scale as a title anyway (issue #80 follow-up)
        guidance = ""
        if people['has_people'] and people['people_count'] <= 2:
            guidance = (f"\n- {people['main_people']} appears in this event; if the event "
                        f"centers on them, use their name (e.g. \"{people['main_people']}'s Birthday\")")
        return f"""**People Detected:**
- People: {people['main_people'] if people['has_people'] else 'None identified'}
- People count: {people['people_count']}
- Face count: {people['face_count']}
- Category: {people['people_category']}{guidance}"""

    def _build_naming_media_block(self, media: Dict[str, Any], similarity: Optional[Dict[str, Any]]) -> str:
        # Similar past events from the organized library - the pattern-
        # matching signal that #67 found was computed but never sent
        similarity_block = ""
        similarity = similarity or {}
        if similarity.get('enabled') and similarity.get('similar_photos'):
            best_per_folder: Dict[str, float] = {}
            for match in similarity['similar_photos']:
                folder = match.get('event_folder')
                score = match.get('similarity', 0.0)
                if folder and score > best_per_folder.get(folder, -1.0):
                    best_per_folder[folder] = score
            top_matches = sorted(best_per_folder.items(), key=lambda kv: kv[1], reverse=True)[:5]
            if top_matches:
                lines = "\n".join(f"- {folder} (similarity: {score:.2f})" for folder, score in top_matches)
                similarity_block = (
                    "\n\n**Similar Past Events (minor signal - formatting/style reference only):**\n"
                    f"{lines}\n"
                    "- These are visually similar past photos, not necessarily the same occasion - "
                    "only borrow their naming STYLE (e.g. date format, how specific the wording is), "
                    "never their subject/occasion unless the photo descriptions above actually support it"
                )
        return f"""**Media:**
- Total files: {media['total_files']}
- Photos: {media['photo_count']}, Videos: {media['video_count']}{similarity_block}"""

    def _build_naming_prompt_home(self, context: Dict[str, Any]) -> str:
        temporal, location, content = context['temporal'], context['location'], context['content']
        people, media = context['people'], context['media']

        gps_spread_km = location.get('gps_spread_km')
        location_spread = (f"Multi-location ({gps_spread_km:.1f} km spread across the cluster)"
                            if gps_spread_km is not None and gps_spread_km >= 2.0 else "Single venue")
        area = location.get('area') or ''
        area_line = f"\n- Area: {area}" if area else ""
        # Raw reverse-geocode data, unfiltered - lets the LLM find a specific
        # venue name itself (e.g. a business name buried in an unexpected
        # field like address.road, the way "Ed's Bowling" was found) rather
        # than us pre-guessing which field is "the" venue name (issue #72)
        raw_geo = location.get('raw_geo')
        raw_geo_block = (f"\n\nHere is the raw geo data for the cluster - look for a specific "
                          f"venue/business name anywhere in it, not just the top-level \"name\" "
                          f"field, since it may be buried in the address details:\n"
                          f"{json.dumps(raw_geo, indent=2)}") if raw_geo else ""

        location_section = f"""**General Location Information:**
- This event took place at {self.home_city}, {self.home_state} which is the person's home city - never state the city or state in the folder name
- Location spread: {location_spread}{area_line}{raw_geo_block}"""

        format_requirements = f"""**Format Requirements:**
- Start with date: YYYY_MM_DD
- Add descriptive event name
- If the raw geo data above names a real, specific venue/business anywhere in it, use that actual name instead of inventing a generic description or category - this is the most important location signal, and takes priority over everything else below
- Never state a city or state in the folder name - this is the person's home city
- If a venue name is used, do NOT also state the Area/neighbourhood - a venue name and an Area are mutually exclusive, never combine them (e.g. "Furniture Shopping at Capilano Mall", not "Furniture Shopping at Capilano Mall in Bonnie Doon")
- Only when there's no venue name at all, this is a private residence: fold the Area/neighbourhood into the description if one is available (e.g. "Movie Night with Friends in Bonnie Doon"); if no Area is available either, just use the descriptive name alone with no location mentioned
- Keep under 60 characters total
- Use title case
- No special characters except hyphens and underscores"""

        constraints = f"""**IMPORTANT CONSTRAINTS:**
- DO NOT invent or state a city or state anywhere in the name
{_NAMING_VENUE_CONSTRAINTS}
{_NAMING_GENERAL_CONSTRAINTS}"""

        examples = f"""**Examples (home location is {self.home_city}, {self.home_state} - never stated):**
- 2024_01_15 - Sarah's Birthday Dinner in Strathcona
- 2024_07_20 - Canada Day Festival
- 2024_12_25 - Christmas Morning
- 2024_03_08 - Game Night with Friends in Riverbend
- 2024_02_10 - Skating at Hawrelak Park"""

        return "\n\n".join([
            _NAMING_PROMPT_INTRO, self._build_naming_event_details_block(temporal),
            location_section + self._build_naming_photos_block(content),
            self._build_naming_people_block(people),
            self._build_naming_media_block(media, context.get('similarity')),
            format_requirements, constraints, examples, _NAMING_OUTPUT_INSTRUCTIONS,
        ])

    def _build_naming_prompt_away(self, context: Dict[str, Any]) -> str:
        temporal, location, content = context['temporal'], context['location'], context['content']
        people, media = context['people'], context['media']

        gps_spread_km = location.get('gps_spread_km')
        location_spread = (f"Multi-location ({gps_spread_km:.1f} km spread across the cluster)"
                            if gps_spread_km is not None and gps_spread_km >= 2.0 else "Single venue")
        raw_geo = location.get('raw_geo')
        raw_geo_block = f"\n\nHere is the raw geo data for the cluster:\n{json.dumps(raw_geo, indent=2)}" if raw_geo else ""
        city = location['city'] or 'Unknown'

        location_section = f"""**Location:**
- City: {city}
- State: {location['state'] or 'Unknown'}
- Country: {location['country'] or 'Unknown'}
- Location spread: {location_spread}{raw_geo_block}"""

        format_requirements = f"""**Format Requirements:**
- Start with date: YYYY_MM_DD
- Add descriptive event name
- Always fold the city ("{city}") into the description naturally (e.g. "Sarah's Birthday Dinner in Toronto", "Ramen Night at Sakura Sushi in Winnipeg") - do this whether or not a venue name is used, don't just append "- City"
- If a specific venue name is available (from the raw geo data above), use it alongside the city instead of inventing a generic description
- Keep under 60 characters total
- Use title case
- No special characters except hyphens and underscores"""

        constraints = f"""**IMPORTANT CONSTRAINTS:**
- ONLY use the provided city: {city}
- DO NOT invent or change the location - use EXACTLY what is provided
{_NAMING_VENUE_CONSTRAINTS}
{_NAMING_GENERAL_CONSTRAINTS}"""

        examples = """**Examples (always state the city):**
- 2024_08_10 - Vancouver Beach Day
- 2018_02_09 - Mexico Trip
- 2023_01_15 - Elena's Birthday in Toronto
- 2024_11_02 - Ramen Night at Sakura Sushi in Winnipeg"""

        return "\n\n".join([
            _NAMING_PROMPT_INTRO, self._build_naming_event_details_block(temporal),
            location_section + self._build_naming_photos_block(content),
            self._build_naming_people_block(people),
            self._build_naming_media_block(media, context.get('similarity')),
            format_requirements, constraints, examples, _NAMING_OUTPUT_INSTRUCTIONS,
        ])

    def _build_naming_prompt_unknown(self, context: Dict[str, Any]) -> str:
        temporal, content = context['temporal'], context['content']
        people, media = context['people'], context['media']

        location_section = "**Location:**\n- Unknown - no GPS data is available for this event"

        format_requirements = """**Format Requirements:**
- Start with date: YYYY_MM_DD
- Add descriptive event name
- Do not mention any location - no city, no area, nothing - we have no location data for this event
- Keep under 60 characters total
- Use title case
- No special characters except hyphens and underscores"""

        constraints = f"""**IMPORTANT CONSTRAINTS:**
- DO NOT invent a location - none is available
{_NAMING_GENERAL_CONSTRAINTS}"""

        examples = """**Examples (no location data - never mention a location):**
- 2024_01_15 - Sarah's Birthday Dinner
- 2024_07_20 - Canada Day Festival
- 2024_03_08 - Game Night with Friends"""

        return "\n\n".join([
            _NAMING_PROMPT_INTRO, self._build_naming_event_details_block(temporal),
            location_section + self._build_naming_photos_block(content),
            self._build_naming_people_block(people),
            self._build_naming_media_block(media, context.get('similarity')),
            format_requirements, constraints, examples, _NAMING_OUTPUT_INSTRUCTIONS,
        ])

    def _build_naming_prompt(self, context: Dict[str, Any]) -> str:
        """
        Build a detailed prompt for the LLM to generate event names.

        Dispatches to one of three mode-specific prompt builders (home/away/
        unknown) based on GPS + city vs home_city, resolved in
        _determine_location_mode. See the module-level comment above
        _NAMING_PROMPT_INTRO for why this is split into three prompts
        instead of one with home-vs-away conditional logic.

        Args:
            context: Structured event context

        Returns:
            Formatted prompt string
        """
        mode = self._determine_location_mode(context['location'])
        builder = {
            'home': self._build_naming_prompt_home,
            'away': self._build_naming_prompt_away,
            'unknown': self._build_naming_prompt_unknown,
        }[mode]
        return builder(context)

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

    def _check_holiday(self, dt: datetime) -> str:
        """Return the holiday name if this date is a major holiday, else ''.

        Fixed-date holidays only (Canada + Eastern Europe, matching this
        library's actual events) - variable-date holidays (Easter, Family
        Day, Thanksgiving, Victoria Day, Labour Day) would need real date
        calculation and aren't covered yet.
        """
        month, day = dt.month, dt.day
        holidays = {
            (1, 1): "New Year's Day",
            (1, 7): "Orthodox Christmas",
            (1, 14): "Old New Year",
            (2, 14): "Valentine's Day",
            (3, 8): "International Women's Day",
            (7, 1): "Canada Day",
            (10, 31): "Halloween",
            (11, 11): "Remembrance Day",
            (12, 25): "Christmas",
            (12, 26): "Boxing Day",
            (12, 31): "New Year's Eve",
        }
        return holidays.get((month, day), '')

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
    def _generate_cache_key(self, context: Dict[str, Any]) -> Optional[str]:
        """Generate cache key for similar events.

        The cache key must be specific enough to avoid false cache hits where
        different content gets the same cached name. Includes:
        - Temporal: time_of_day, duration, day, weekend/weekday
        - Location: city
        - Content: event_type, primary_activity, top scenes, top objects
        - People: people_category (solo/couple/group/no_people)

        Returns None when scenes and objects are both empty (issue #76):
        with no real content signal, the key collapses to fixed placeholder
        strings, so two genuinely different small events on the same city/
        weekday/time-of-day (e.g. two different stops on the same trip) can
        collide on an identical key. None means "don't trust this key" -
        callers must skip the cache entirely (no read, no write) rather than
        treat it as a normal cache key.
        """
        temporal = context['temporal']
        location = context['location']
        content = context['content']
        people = context.get('people', {})

        # Get top scenes and objects for more specific cache key
        # Note: scenes/objects may be tuples like ('home', 6) or strings
        raw_scenes = content.get('scenes', [])
        raw_objects = content.get('objects', [])

        if not raw_scenes and not raw_objects:
            return None

        # Extract scene names (handle both tuple and string formats)
        scenes = [s[0] if isinstance(s, tuple) else s for s in raw_scenes]
        objects = [o[0] if isinstance(o, tuple) else o for o in raw_objects]

        # Sort and join top 2 scenes for consistency
        scene_key = '_'.join(sorted(scenes[:2])) if scenes else 'unknown_scene'
        # Sort and join top 3 objects for consistency
        object_key = '_'.join(sorted(objects[:3])) if objects else 'unknown_objects'
        # People category for social context
        people_category = people.get('people_category', 'unknown')

        key_parts = [
            temporal['time_of_day'],
            temporal['duration_category'],
            temporal['day_of_week'],
            'weekend' if temporal['is_weekend'] else 'weekday',
            location['city'],
            content['event_type'],
            content['primary_activity'],
            scene_key,
            object_key,
            people_category
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

        Uses first names only (e.g. "Jane" not "Jane Smith") - full names
        read as overly formal for a personal photo folder name.

        Only names solo/couple (1-2 people, e.g. "Jane" or "Jane & John") -
        past 2, a name list doesn't scale as a folder title ("Jane, John,
        Amy & Sam's Birthday" reads like only Sam's) and there's no single
        name a group event would center on, so it falls back to a plain
        headcount instead of naming anyone (issue #80 follow-up).

        Args:
            people_detected: List of detected people names

        Returns:
            Formatted string of people first names, or a headcount for 3+
        """
        if not people_detected:
            return ""

        first_names = [name.split()[0] if name.split() else name for name in people_detected]

        if len(first_names) == 1:
            return first_names[0]
        elif len(first_names) == 2:
            return f"{first_names[0]} & {first_names[1]}"
        else:
            return f"a group of {len(first_names)}"

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