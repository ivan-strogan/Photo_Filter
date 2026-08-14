"""Reverse geocoding utilities for location-based clustering."""

from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
from pathlib import Path

try:
    from .environment_config import get_location_cache_file
except ImportError:
    from environment_config import get_location_cache_file

# Proximity threshold (km) used by find_representative_location to group
# GPS points when picking a cluster's representative location. Deliberately
# much tighter than cluster_locations_by_proximity's default 1.0km
# cross-venue threshold - this one targets GPS-lag-level noise (e.g. a
# phone's first-shot-after-opening-camera position fix), not real distance
# between different places.
REPRESENTATIVE_LOCATION_THRESHOLD_KM = 0.15

@dataclass
class LocationInfo:
    """Location information from reverse geocoding."""
    latitude: float
    longitude: float
    address: str
    city: str
    state: str
    country: str
    raw_data: Dict[str, Any]

class LocationGeocoder:
    """Handles reverse geocoding and location clustering."""

    def __init__(self, user_agent: str = "photo_filter_app"):
        """Initialize the geocoder.

        Args:
            user_agent: User agent string for the geocoding service
        """
        self.logger = logging.getLogger(__name__)
        self.geocoder = Nominatim(user_agent=user_agent)
        self.cache_file = get_location_cache_file()
        self.location_cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict]:
        """Load location cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading location cache: {e}")
        return {}

    def _save_cache(self) -> None:
        """Save location cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.location_cache, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Error saving location cache: {e}")

    def _create_cache_key(self, lat: float, lon: float, precision: int = 3) -> str:
        """Create cache key for coordinates.

        Args:
            lat: Latitude
            lon: Longitude
            precision: Decimal places for caching (reduces API calls)

        Returns:
            Cache key string
        """
        return f"{lat:.{precision}f},{lon:.{precision}f}"

    def reverse_geocode(self, latitude: float, longitude: float) -> Optional[LocationInfo]:
        """Perform reverse geocoding on coordinates.

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate

        Returns:
            LocationInfo object or None if geocoding fails
        """
        # Check cache first
        cache_key = self._create_cache_key(latitude, longitude)
        if cache_key in self.location_cache:
            cached_data = self.location_cache[cache_key]
            return LocationInfo(
                latitude=latitude,
                longitude=longitude,
                address=cached_data.get('address', ''),
                city=cached_data.get('city', ''),
                state=cached_data.get('state', ''),
                country=cached_data.get('country', ''),
                raw_data=cached_data.get('raw_data', {})
            )

        try:
            # Add delay to respect rate limits
            time.sleep(1)

            location = self.geocoder.reverse(f"{latitude}, {longitude}")

            if location and location.raw:
                raw_data = location.raw
                address_components = raw_data.get('address', {})

                # Extract location components
                city = self._extract_city(address_components)
                state = self._extract_state(address_components)
                country = address_components.get('country', '')
                address = location.address

                location_info = LocationInfo(
                    latitude=latitude,
                    longitude=longitude,
                    address=address,
                    city=city,
                    state=state,
                    country=country,
                    raw_data=raw_data
                )

                # Cache the result
                self.location_cache[cache_key] = {
                    'address': address,
                    'city': city,
                    'state': state,
                    'country': country,
                    'raw_data': raw_data
                }
                self._save_cache()

                return location_info

        except (GeocoderTimedOut, GeocoderServiceError) as e:
            self.logger.warning(f"Geocoding error for ({latitude}, {longitude}): {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error in geocoding: {e}")

        return None

    def _extract_city(self, address_components: Dict[str, str]) -> str:
        """Extract city name from address components."""
        # Try different possible city field names
        city_fields = ['city', 'town', 'village', 'municipality', 'suburb', 'neighbourhood']
        for field in city_fields:
            if field in address_components:
                return address_components[field]
        return ''

    def _extract_state(self, address_components: Dict[str, str]) -> str:
        """Extract state/province name from address components."""
        # Try different possible state field names
        state_fields = ['state', 'province', 'region', 'county']
        for field in state_fields:
            if field in address_components:
                return address_components[field]
        return ''

    def calculate_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate distance between two coordinates in kilometers.

        Args:
            coord1: First coordinate (lat, lon)
            coord2: Second coordinate (lat, lon)

        Returns:
            Distance in kilometers
        """
        try:
            return geodesic(coord1, coord2).kilometers
        except Exception as e:
            self.logger.warning(f"Error calculating distance: {e}")
            return float('inf')

    def cluster_locations_by_proximity(self,
                                     locations: List[Tuple[float, float]],
                                     threshold_km: float = 1.0) -> List[List[int]]:
        """Cluster locations based on proximity.

        Args:
            locations: List of (lat, lon) tuples
            threshold_km: Distance threshold in kilometers

        Returns:
            List of clusters, where each cluster is a list of location indices
        """
        if not locations:
            return []

        clusters = []
        visited = [False] * len(locations)

        for i, location in enumerate(locations):
            if visited[i]:
                continue

            # Start new cluster
            cluster = [i]
            visited[i] = True

            # Find all locations within threshold
            for j in range(i + 1, len(locations)):
                if visited[j]:
                    continue

                distance = self.calculate_distance(location, locations[j])
                if distance <= threshold_km:
                    cluster.append(j)
                    visited[j] = True

            clusters.append(cluster)

        return clusters

    def find_representative_location(self, gps_coordinates: List[Tuple[float, float]],
                                      locations: List[LocationInfo],
                                      threshold_km: float = REPRESENTATIVE_LOCATION_THRESHOLD_KM
                                      ) -> Optional[LocationInfo]:
        """Pick a representative LocationInfo for a cluster using density
        (majority-count) grouping, not distance-averaging.

        A naive "use whichever photo is first chronologically" (the
        previous behavior) lets a single outlier GPS reading - e.g. a
        phone's first shot after opening the camera, which often uses a
        stale cell/wifi position fix before the true GPS lock completes -
        determine the location for an entire cluster of otherwise-
        consistent photos. A geometric centroid/average was considered and
        rejected: the average of two genuinely different real places (e.g.
        a multi-stop trip) is often a point in neither of them.

        Instead, this groups the cluster's GPS points by tight proximity
        (threshold_km, deliberately much tighter than
        cluster_locations_by_proximity's default 1.0km cross-venue
        threshold - this one is sized to catch GPS-lag-level noise, not
        real distance between different places) and returns the first
        chronological photo's location within the LARGEST group. See
        temp/diagnostics/2026-08-11_location_representative_point_bug/NOTES.md
        for the full investigation, including why this fixed a real case
        (29 photos at the true location + 1 outlier ~330m away) without
        changing any of the other 9 real clusters that had no outlier.

        Args:
            gps_coordinates: (lat, lon) tuples in chronological order,
                index-aligned with `locations`.
            locations: LocationInfo objects, index-aligned with
                `gps_coordinates`.
            threshold_km: Proximity threshold for grouping (default 150m).

        Returns:
            The representative LocationInfo, or None if `locations` is empty.
        """
        if not locations:
            return None
        if len(locations) == 1:
            return locations[0]

        groups = self.cluster_locations_by_proximity(gps_coordinates, threshold_km=threshold_km)
        largest_group = max(groups, key=len)
        return locations[largest_group[0]]

    def get_location_summary(self, locations: List[LocationInfo]) -> Dict[str, Any]:
        """Get summary of location distribution.

        Args:
            locations: List of LocationInfo objects

        Returns:
            Dictionary with location statistics
        """
        if not locations:
            return {
                'total_locations': 0,
                'countries': {},
                'states': {},
                'cities': {}
            }

        countries = {}
        states = {}
        cities = {}

        for location in locations:
            # Count countries
            if location.country:
                countries[location.country] = countries.get(location.country, 0) + 1

            # Count states
            if location.state:
                state_key = f"{location.state}, {location.country}"
                states[state_key] = states.get(state_key, 0) + 1

            # Count cities
            if location.city:
                city_key = f"{location.city}, {location.state}, {location.country}"
                cities[city_key] = cities.get(city_key, 0) + 1

        return {
            'total_locations': len(locations),
            'unique_countries': len(countries),
            'unique_states': len(states),
            'unique_cities': len(cities),
            'countries': dict(sorted(countries.items(), key=lambda x: x[1], reverse=True)),
            'states': dict(sorted(states.items(), key=lambda x: x[1], reverse=True)),
            'cities': dict(sorted(cities.items(), key=lambda x: x[1], reverse=True))
        }

    def find_most_common_location(self, locations: List[LocationInfo]) -> Optional[str]:
        """Find the most common location from a list.

        Args:
            locations: List of LocationInfo objects

        Returns:
            Most common location string or None
        """
        if not locations:
            return None

        location_counts = {}
        for location in locations:
            # Create location string (city, state format)
            if location.city and location.state:
                location_str = f"{location.city}, {location.state}"
            elif location.city:
                location_str = location.city
            elif location.state:
                location_str = location.state
            else:
                location_str = location.country

            location_counts[location_str] = location_counts.get(location_str, 0) + 1

        if location_counts:
            return max(location_counts, key=location_counts.get)
        return None

    def cleanup_cache(self, max_entries: int = 10000) -> None:
        """Clean up location cache if it gets too large.

        Args:
            max_entries: Maximum number of cache entries to keep
        """
        if len(self.location_cache) > max_entries:
            self.logger.info(f"Cleaning up location cache (current size: {len(self.location_cache)})")
            # Keep only the most recent entries (simplified cleanup)
            cache_items = list(self.location_cache.items())
            self.location_cache = dict(cache_items[-max_entries:])
            self._save_cache()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the geocoding cache.

        Returns:
            Dictionary with cache statistics
        """
        return {
            'cache_size': len(self.location_cache),
            'cache_file_exists': self.cache_file.exists(),
            'cache_file_size_bytes': self.cache_file.stat().st_size if self.cache_file.exists() else 0
        }