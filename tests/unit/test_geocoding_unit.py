#!/usr/bin/env python3
"""
Unit tests for src/geocoding.py's LocationGeocoder.

These are FAST unit tests - no network calls (find_representative_location
and its dependencies, cluster_locations_by_proximity/calculate_distance,
are pure math over already-known coordinates - no reverse geocoding
involved).

Focus:
- find_representative_location(): the fix for a real bug where a cluster's
  location was picked from whichever photo happened to be first
  chronologically, with no outlier rejection - see
  temp/diagnostics/2026-08-11_location_representative_point_bug/NOTES.md
  for the full investigation (a 30-photo cluster's location came from one
  outlier photo ~330m from where the other 29 were actually taken, due to
  a common iPhone first-shot GPS lag quirk).

Run with: pytest tests/unit/test_geocoding_unit.py -v
Expected time: <1 second total
"""

import pytest

from src.geocoding import LocationGeocoder, LocationInfo


@pytest.fixture
def geocoder():
    return LocationGeocoder()


def make_location(lat, lon, city="Edmonton"):
    return LocationInfo(latitude=lat, longitude=lon, address="", city=city,
                         state="Alberta", country="Canada", raw_data={})


# ===== find_representative_location() =====

@pytest.mark.unit
def test_find_representative_location_empty_returns_none(geocoder):
    assert geocoder.find_representative_location([], []) is None


@pytest.mark.unit
def test_find_representative_location_single_photo_returns_it(geocoder):
    loc = make_location(53.5, -113.5)
    result = geocoder.find_representative_location([(53.5, -113.5)], [loc])
    assert result is loc


@pytest.mark.unit
def test_find_representative_location_all_tight_returns_first_chronological(geocoder):
    """When every photo agrees (single group), the first chronological
    photo is still a reasonable representative - same as the old behavior
    for the common case with no outlier."""
    coords = [(53.5000, -113.5000), (53.5001, -113.5001), (53.5002, -113.5002)]
    locs = [make_location(*c) for c in coords]
    result = geocoder.find_representative_location(coords, locs)
    assert result is locs[0]


@pytest.mark.unit
def test_find_representative_location_ignores_single_outlier():
    """Regression test for the real bug: a lone outlier (e.g. GPS lag on
    the first shot after opening the camera) must NOT determine the
    cluster's location when the vast majority of photos agree on
    somewhere else - even though old behavior (locations[0]) would have
    picked exactly this outlier, since it's first."""
    geocoder = LocationGeocoder()
    # Outlier ~330m away (matches the real cluster 0 case), listed FIRST
    # chronologically - exactly reproducing what broke before this fix
    outlier_coord = (53.5384, -113.6166)
    majority_coords = [(53.5392, -113.6118)] * 5  # 5 photos, same tight spot
    # perturb slightly so they're distinct points but still well within
    # the 150m grouping threshold of each other
    majority_coords = [(53.5392 + i * 0.00001, -113.6118 + i * 0.00001) for i in range(5)]

    coords = [outlier_coord] + majority_coords
    locs = [make_location(*outlier_coord, city="OutlierCity")] + \
           [make_location(*c, city="MajorityCity") for c in majority_coords]

    result = geocoder.find_representative_location(coords, locs)

    assert result.city == "MajorityCity", (
        "the majority group (5 photos) must win over the single outlier, "
        "even though the outlier is first chronologically"
    )


@pytest.mark.unit
def test_find_representative_location_matches_real_cluster_0_case():
    """Exact real-world numbers from the investigation: 1 outlier photo +
    6 tightly-clustered photos (the other real photo from that cluster was
    excluded here to keep the test self-contained - see NOTES.md for the
    full 7-photo dataset)."""
    geocoder = LocationGeocoder()
    outlier = (53.538372, -113.616562)  # IMG_20160101_001421.JPG - the outlier
    majority = [
        (53.539219, -113.611769),  # IMG_20160101_001923.JPG
        (53.539333, -113.611700),
        (53.539147, -113.611850),
        (53.539250, -113.611820),
        (53.539200, -113.611780),
        (53.539280, -113.611790),
    ]
    coords = [outlier] + majority
    locs = [make_location(*outlier, city="Place LaRue")] + \
           [make_location(*c, city="Glenwood") for c in majority]

    result = geocoder.find_representative_location(coords, locs)

    assert result.city == "Glenwood"


@pytest.mark.unit
def test_find_representative_location_two_distinct_places_picks_larger_group():
    """Multi-location cluster (e.g. a trip with two real stops) - the
    larger group wins, not a meaningless geometric average of both. This
    does NOT pretend the cluster is single-venue; existing gps_spread_km
    "Multi-location" detection elsewhere already flags that separately."""
    geocoder = LocationGeocoder()
    place_a = [(51.0, -114.0), (51.0001, -114.0001), (51.0002, -114.0002)]  # 3 photos
    place_b = [(51.5, -114.5), (51.5001, -114.5001)]  # 2 photos, far from place_a

    coords = place_a + place_b
    locs = [make_location(*c, city="PlaceA") for c in place_a] + \
           [make_location(*c, city="PlaceB") for c in place_b]

    result = geocoder.find_representative_location(coords, locs)

    assert result.city == "PlaceA", "the larger group (3 photos) should win over the smaller one (2 photos)"


@pytest.mark.unit
def test_find_representative_location_uses_tighter_threshold_than_venue_splitting(geocoder):
    """The grouping threshold must be smaller than
    cluster_locations_by_proximity's default 1.0km cross-venue threshold -
    otherwise this would just merge genuinely different nearby venues
    together instead of only catching GPS-lag-level noise."""
    from src.geocoding import REPRESENTATIVE_LOCATION_THRESHOLD_KM
    assert REPRESENTATIVE_LOCATION_THRESHOLD_KM < 1.0


# ===== cluster_locations_by_proximity() - existing method, sanity coverage =====

@pytest.mark.unit
def test_cluster_locations_by_proximity_groups_nearby_points(geocoder):
    close_points = [(53.5, -113.5), (53.5001, -113.5001)]
    far_point = [(51.0, -114.0)]
    clusters = geocoder.cluster_locations_by_proximity(close_points + far_point, threshold_km=1.0)
    # the two close points should be in the same cluster, the far one separate
    cluster_sizes = sorted(len(c) for c in clusters)
    assert cluster_sizes == [1, 2]


@pytest.mark.unit
def test_calculate_distance_zero_for_identical_points(geocoder):
    assert geocoder.calculate_distance((53.5, -113.5), (53.5, -113.5)) == 0
