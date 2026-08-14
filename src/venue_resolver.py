"""
Venue resolution for event naming.

Reverse geocoding (Nominatim) frequently returns a street address with no
business/venue name at all - a ski hill, a restaurant, a shop. This module
fills that gap with a three-step pipeline, validated against 16 real photo
clusters (see temp/diagnostics/2026-07-12_naming_prompt_competition/):

1. judge() - LLM call classifying the cluster as one of:
   - "venue_known": the raw geo data already names a real, specific place
   - "business_unknown": no venue name, but the photos suggest a business/
     public place - worth searching for
   - "residence": no venue name, and the photos suggest a private home -
     not worth searching, since it's not a business
2. search_nearby_named_features() - deterministic OpenStreetMap Overpass API
   search with an adaptive radius (shrinks in dense urban areas, expands in
   rural ones), no LLM involved
3. verify() - LLM call reasoning over the search candidates against the
   photo captions to decide if any of them plausibly matches; explicitly
   allowed to say "no match" rather than force a guess

resolve() orchestrates all three and returns a confirmed venue name, or None.

For junior developers:
- The judge/search/verify split exists because reliable LLM tool-calling
  (letting the model decide when/how to search) turned out to be fragile on
  local models - see reference notes on schema anchoring bias and the
  documented gemma3/gemma4 Apple Silicon empty-response bug. Splitting the
  work into a deterministic search step plus two narrowly-scoped LLM calls
  sidesteps that fragility entirely.
- Every network call (Overpass, Ollama) fails closed: on any error, resolve()
  returns None rather than raising, so a flaky API never breaks event naming.
"""

import json
import logging
import math
import time
from typing import Any, Dict, List, Optional

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from .environment_config import get_venue_judge_model, get_ollama_url
except ImportError:
    from environment_config import get_venue_judge_model, get_ollama_url

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_USER_AGENT = "photo_filter_app (personal photo organizer, https://github.com)"

# Adaptive-radius search bounds
MIN_RADIUS_M = 100
MAX_RADIUS_M = 8000
TARGET_MIN_RESULTS = 1
TARGET_MAX_RESULTS = 15
MAX_SEARCH_ITERATIONS = 5
OVERPASS_RETRIES = 3

# Tags that identify a named place worth surfacing as a venue candidate
POI_TAG_REGEX = r'^(shop|amenity|leisure|tourism|piste:type)$'
LANDUSE_REGEX = r'^(winter_sports|retail|commercial)$'
RELEVANT_TAG_KEYS = ('shop', 'amenity', 'leisure', 'tourism', 'piste:type', 'sport', 'landuse')


class VenueSearchError(RuntimeError):
    """Raised when the Overpass search itself fails (network error, rate
    limit, timeout) after exhausting retries - distinct from a search that
    ran successfully and genuinely found zero candidates. Callers should
    treat this as "couldn't determine, try again later", not as
    confirmation that no venue exists - silently falling back to naming
    with incomplete data produces misleading folder names."""


class VenueResolver:
    """Judge/search/verify pipeline for filling in venue names that reverse
    geocoding didn't surface."""

    def __init__(self, ollama_model: Optional[str] = None, ollama_url: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.ollama_model = ollama_model or get_venue_judge_model()
        self.ollama_url = ollama_url or get_ollama_url()

    def resolve(self, raw_geo: Dict[str, Any], captions: List[str]) -> Optional[str]:
        """Run the full judge -> search -> verify pipeline for one cluster.

        Args:
            raw_geo: The cluster's raw reverse-geocode dict (Nominatim format).
            captions: Photo captions for the cluster.

        Returns:
            A confirmed venue name, or None if no real venue could be
            identified (search ran and found nothing plausible, or a search
            wasn't warranted). Deliberately does not guess.

        Raises:
            VenueSearchError: if the Overpass search itself failed (network
                error, rate limit) rather than genuinely finding no results.
                Callers should treat this as "couldn't determine, try again
                later" - NOT catch-and-proceed with an unenriched raw_geo,
                since that can surface a misleading non-venue field (e.g. a
                business-district name) as if it were a confirmed venue.
        """
        if not REQUESTS_AVAILABLE or not raw_geo or not captions:
            return None

        category = self.judge(raw_geo, captions)
        if category != 'business_unknown':
            return None

        lat, lon = raw_geo.get('lat'), raw_geo.get('lon')
        if lat is None or lon is None:
            return None

        try:
            candidates = self.search_nearby_named_features(float(lat), float(lon))
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Venue search failed: {e}")
            return None

        if not candidates:
            return None

        result = self.verify(candidates, captions)
        return result.get('match') if result else None

    # ---------- step 1: judge ----------

    def judge(self, raw_geo: Dict[str, Any], captions: List[str]) -> Optional[str]:
        """Classify a cluster as venue_known / business_unknown / residence."""
        prompt = self._build_judge_prompt(raw_geo, captions)
        response = self._query_ollama_json(prompt, num_predict=100)
        return response.get('category') if response else None

    @staticmethod
    def _build_judge_prompt(raw_geo: Dict[str, Any], captions: List[str]) -> str:
        photos = [{"photo": f"{i+1} of {len(captions)}", "description": c}
                  for i, c in enumerate(captions)]
        return f"""This is one step in a photo-organization pipeline. Photos have already been
grouped into clusters by time and GPS location. For each cluster, we reverse-geocode
the GPS coordinates to get raw location data, and we generate a caption for each photo.
Before we hand this off to a final step that names the folder, we need to check
whether the location data actually identifies a specific place, and if not, whether
it's worth searching for one.

Raw reverse-geocode data for this cluster's GPS point:
{json.dumps(raw_geo, indent=2)}

Photo descriptions for this cluster:
{json.dumps(photos, indent=2)}

Task: classify this cluster into exactly one of three categories:

- "venue_known": the raw geo data above already specifies a real, specific venue/place
  name (a business, mall, landmark, restaurant, etc.) - not just a generic street
  address or an OSM class/type code with no name.
- "business_unknown": the raw geo data does NOT specify a venue name, AND the photo
  descriptions suggest a business, public place, or outdoor activity (not someone's
  home) - e.g. a store, restaurant, ski hill, park, event venue. Worth searching for.
- "residence": the raw geo data does NOT specify a venue name, AND the photo
  descriptions suggest a private home (living room, someone's kitchen, a couch, etc.)
  - not worth searching for a business, since it's not one.

Respond with a JSON object only, no other text, in this exact format:
{{"category": "venue_known" or "business_unknown" or "residence", "reason": "one short sentence"}}"""

    # ---------- step 2a: search (deterministic, no LLM) ----------

    def search_nearby_named_features(self, lat: float, lon: float,
                                      start_radius_m: int = 1000) -> List[Dict[str, Any]]:
        """Adaptive-radius Overpass search: shrinks the radius when too many
        results come back (dense urban area), expands when too few (rural
        area), and returns results sorted by actual distance."""
        radius = start_radius_m
        results: List[Dict[str, Any]] = []
        for _ in range(MAX_SEARCH_ITERATIONS):
            elements = self._query_overpass(lat, lon, radius)
            results = self._dedupe_and_sort(elements, lat, lon)
            count = len(results)

            if count > TARGET_MAX_RESULTS and radius > MIN_RADIUS_M:
                radius = max(MIN_RADIUS_M, radius // 2)
                time.sleep(2)
                continue
            if count < TARGET_MIN_RESULTS and radius < MAX_RADIUS_M:
                radius = min(MAX_RADIUS_M, radius * 3)
                time.sleep(2)
                continue
            return results[:TARGET_MAX_RESULTS]

        return results[:TARGET_MAX_RESULTS]

    def _query_overpass(self, lat: float, lon: float, radius_m: int) -> List[Dict[str, Any]]:
        query = f"""
[out:json][timeout:30];
(
  node(around:{radius_m},{lat},{lon})[~"{POI_TAG_REGEX}"~"."]["name"];
  way(around:{radius_m},{lat},{lon})[~"{POI_TAG_REGEX}"~"."]["name"];
  relation(around:{radius_m},{lat},{lon})[~"{POI_TAG_REGEX}"~"."]["name"];
  node(around:{radius_m},{lat},{lon})["sport"]["name"];
  way(around:{radius_m},{lat},{lon})["sport"]["name"];
  relation(around:{radius_m},{lat},{lon})["sport"]["name"];
  way(around:{radius_m},{lat},{lon})["landuse"~"{LANDUSE_REGEX}"]["name"];
  relation(around:{radius_m},{lat},{lon})["landuse"~"{LANDUSE_REGEX}"]["name"];
);
out tags center;
"""
        headers = {"User-Agent": OVERPASS_USER_AGENT}
        last_err = None
        for attempt in range(OVERPASS_RETRIES):
            try:
                resp = requests.post(OVERPASS_URL, data={"data": query}, headers=headers, timeout=45)
                if resp.status_code in (429, 504):
                    last_err = f"HTTP {resp.status_code}"
                    time.sleep(5 * (attempt + 1))
                    continue
                resp.raise_for_status()
                return resp.json().get("elements", [])
            except requests.exceptions.RequestException as e:
                last_err = e
                time.sleep(3)
        raise VenueSearchError(f"Overpass query failed after {OVERPASS_RETRIES} attempts: {last_err}")

    @staticmethod
    def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371000
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * R * math.asin(math.sqrt(a))

    @classmethod
    def _dedupe_and_sort(cls, elements: List[Dict[str, Any]], lat: float, lon: float) -> List[Dict[str, Any]]:
        results = []
        seen = set()
        for el in elements:
            tags = el.get("tags", {})
            name = tags.get("name")
            if not name or name in seen:
                continue
            seen.add(name)
            if "center" in el:
                elat, elon = el["center"]["lat"], el["center"]["lon"]
            else:
                elat, elon = el.get("lat"), el.get("lon")
            dist = cls._haversine_m(lat, lon, elat, elon) if elat is not None else None
            relevant = {k: v for k, v in tags.items() if k in RELEVANT_TAG_KEYS}
            results.append({"name": name, "tags": relevant,
                             "distance_m": round(dist) if dist is not None else None})
        results.sort(key=lambda r: (r["distance_m"] is None, r["distance_m"]))
        return results

    # ---------- step 2b: verify ----------

    def verify(self, candidates: List[Dict[str, Any]], captions: List[str]) -> Optional[Dict[str, Any]]:
        """Reason over the search candidates against the photo captions to
        decide whether any plausibly matches. Returns {"match": name or
        None, "reason": ...}, or None if the Ollama call itself failed."""
        prompt = self._build_verify_prompt(candidates, captions)
        return self._query_ollama_json(prompt, num_predict=150)

    @staticmethod
    def _build_verify_prompt(candidates: List[Dict[str, Any]], captions: List[str]) -> str:
        photos = [{"photo": f"{i+1} of {len(captions)}", "description": c}
                  for i, c in enumerate(captions)]
        return f"""This is one step in a photo-organization pipeline. We searched for named
places near the GPS location where these photos were taken, since the original
reverse-geocoded location data didn't identify a specific venue. Below are the
candidate places found nearby (closest first), and the photo descriptions for
this cluster.

Candidate places found nearby, closest first:
{json.dumps(candidates, indent=2)}

Photo descriptions for this cluster:
{json.dumps(photos, indent=2)}

Task: decide whether any candidate above plausibly matches what's shown in the
photos. A candidate counts as a match if its type (from its tags) is
consistent with the activity/setting described in the photos - it does not
need to be explicitly named in the photo descriptions, just plausible. If
multiple candidates are plausible, prefer the overall venue/business/resort
name over a specific sub-feature within it (e.g. prefer a ski resort's own
name over one of its individual trail/run names; prefer a mall's own name
over one of its parking lots or transit stops) - do not just pick whichever
is closest if a broader, more recognizable candidate is also plausible. If
none of the candidates are a plausible fit for what's shown in the photos, do
not force a guess - say there is no match.

Respond with a JSON object only, no other text, in this exact format:
{{"match": "<candidate name>" or null, "reason": "one short sentence"}}"""

    # ---------- shared Ollama helper ----------

    def _query_ollama_json(self, prompt: str, num_predict: int) -> Optional[Dict[str, Any]]:
        try:
            payload = {
                "model": self.ollama_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,  # thinking models otherwise burn num_predict on hidden
                                  # reasoning tokens and return an empty response
                "format": "json",
                "options": {"temperature": 0.1, "num_predict": num_predict},
            }
            response = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=60)
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
            return json.loads(content)
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Venue resolver Ollama call failed: {e}")
            return None
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Venue resolver got unparseable Ollama response: {e}")
            return None
