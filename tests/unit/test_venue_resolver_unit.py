#!/usr/bin/env python3
"""
Unit tests for src/venue_resolver.py (VenueResolver).

These are FAST unit tests using mocked HTTP calls - no real network, no
real LLM/Overpass calls.

Focus:
- judge() / verify() JSON response handling (valid, unparseable, request failure)
- Overpass search: adaptive radius behavior, dedup/sort, distance calculation
- VenueSearchError: raised on a genuine search failure (network error,
  persistent rate limit), distinct from a search that ran cleanly and found
  zero results - this distinction is what fixes the "cluster 9 street
  address" bug (see temp/diagnostics/2026-08-04_prompt_separation_competition/)
- resolve() orchestration: category gating, error propagation, happy path

Run with: pytest tests/unit/test_venue_resolver_unit.py -v
Expected time: <3 seconds total
"""

import json
import pytest
from unittest.mock import Mock, patch
import requests

from src.venue_resolver import VenueResolver, VenueSearchError


# ===== FIXTURES =====

@pytest.fixture
def resolver():
    return VenueResolver(ollama_model="qwen3:14b", ollama_url="http://localhost:11434")


def _ollama_response(content_obj):
    """Build a mock requests.Response for a successful Ollama /api/chat call."""
    resp = Mock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"message": {"content": json.dumps(content_obj)}}
    return resp


def _overpass_response(status_code=200, elements=None):
    resp = Mock()
    resp.status_code = status_code
    if status_code == 200:
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"elements": elements or []}
    else:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(str(status_code))
    return resp


def _overpass_element(name, lat, lon, tags_extra=None):
    return {
        "tags": {"name": name, "amenity": "restaurant", **(tags_extra or {})},
        "lat": lat,
        "lon": lon,
    }


# ===== judge() =====

@pytest.mark.unit
def test_judge_returns_category_from_valid_json_response(resolver):
    with patch("src.venue_resolver.requests.post", return_value=_ollama_response(
            {"category": "business_unknown", "reason": "no venue name, looks like a restaurant"})):
        result = resolver.judge({"lat": "53.0", "lon": "-113.0"}, ["A photo of a restaurant."])
    assert result == "business_unknown"


@pytest.mark.unit
def test_judge_returns_none_on_unparseable_response(resolver):
    bad_resp = Mock()
    bad_resp.status_code = 200
    bad_resp.raise_for_status.return_value = None
    bad_resp.json.return_value = {"message": {"content": "not valid json"}}
    with patch("src.venue_resolver.requests.post", return_value=bad_resp):
        result = resolver.judge({"lat": "53.0", "lon": "-113.0"}, ["caption"])
    assert result is None


@pytest.mark.unit
def test_judge_returns_none_on_request_exception(resolver):
    with patch("src.venue_resolver.requests.post", side_effect=requests.exceptions.ConnectionError("down")):
        result = resolver.judge({"lat": "53.0", "lon": "-113.0"}, ["caption"])
    assert result is None


# ===== verify() =====

@pytest.mark.unit
def test_verify_returns_match_dict_from_valid_json_response(resolver):
    candidates = [{"name": "Rabbit Hill Snow Resort", "tags": {"sport": "skiing"}, "distance_m": 132}]
    with patch("src.venue_resolver.requests.post", return_value=_ollama_response(
            {"match": "Rabbit Hill Snow Resort", "reason": "matches ski setting"})):
        result = resolver.verify(candidates, ["A snowy hill photo."])
    assert result == {"match": "Rabbit Hill Snow Resort", "reason": "matches ski setting"}


@pytest.mark.unit
def test_verify_returns_none_match_when_llm_says_no_match(resolver):
    candidates = [{"name": "7-Eleven", "tags": {"shop": "convenience"}, "distance_m": 50}]
    with patch("src.venue_resolver.requests.post", return_value=_ollama_response(
            {"match": None, "reason": "none of the candidates fit"})):
        result = resolver.verify(candidates, ["A close-up of a sticker."])
    assert result["match"] is None


@pytest.mark.unit
def test_verify_returns_none_on_unparseable_response(resolver):
    bad_resp = Mock()
    bad_resp.status_code = 200
    bad_resp.raise_for_status.return_value = None
    bad_resp.json.return_value = {"message": {"content": "{broken"}}
    with patch("src.venue_resolver.requests.post", return_value=bad_resp):
        result = resolver.verify([{"name": "X", "tags": {}, "distance_m": 1}], ["caption"])
    assert result is None


# ===== _query_overpass / VenueSearchError =====
# Regression coverage for the "cluster 9" bug: a transient Overpass failure
# was silently returning [] (treated as "genuinely no venue"), letting
# naming fall through to an unenriched raw_geo that surfaced a misleading
# non-venue field ("Fourth Street BRZ") as if it were a confirmed venue.

@pytest.mark.unit
def test_persistent_504_raises_venue_search_error(resolver):
    with patch("src.venue_resolver.requests.post", return_value=_overpass_response(status_code=504)), \
         patch("src.venue_resolver.time.sleep"):
        with pytest.raises(VenueSearchError):
            resolver.search_nearby_named_features(53.0, -113.0)


@pytest.mark.unit
def test_connection_error_raises_venue_search_error(resolver):
    with patch("src.venue_resolver.requests.post", side_effect=requests.exceptions.ConnectionError("boom")), \
         patch("src.venue_resolver.time.sleep"):
        with pytest.raises(VenueSearchError):
            resolver.search_nearby_named_features(53.0, -113.0)


@pytest.mark.unit
def test_genuine_zero_results_does_not_raise(resolver):
    with patch("src.venue_resolver.requests.post", return_value=_overpass_response(elements=[])), \
         patch("src.venue_resolver.time.sleep"):
        results = resolver.search_nearby_named_features(53.0, -113.0)
    assert results == []


@pytest.mark.unit
def test_resolve_propagates_venue_search_error_instead_of_swallowing_it(resolver):
    """resolve() must NOT catch VenueSearchError into a plain None - that
    would make "couldn't determine" indistinguishable from "confirmed no
    venue", which is exactly the bug this exception type exists to fix."""
    raw_geo = {"lat": "53.0", "lon": "-113.0", "address": {}}
    with patch.object(resolver, "judge", return_value="business_unknown"), \
         patch.object(resolver, "search_nearby_named_features", side_effect=VenueSearchError("simulated")):
        with pytest.raises(VenueSearchError):
            resolver.resolve(raw_geo, ["a restaurant photo"])


@pytest.mark.unit
def test_resolve_returns_none_when_search_finds_nothing(resolver):
    """The legitimate negative case: search ran cleanly and found zero
    candidates - no exception, just None."""
    raw_geo = {"lat": "53.0", "lon": "-113.0", "address": {}}
    with patch.object(resolver, "judge", return_value="business_unknown"), \
         patch.object(resolver, "search_nearby_named_features", return_value=[]):
        result = resolver.resolve(raw_geo, ["a restaurant photo"])
    assert result is None


# ===== search_nearby_named_features: adaptive radius =====

@pytest.mark.unit
def test_search_shrinks_radius_when_too_many_results(resolver):
    """Dense urban area: first call returns >15 results, so the radius
    should shrink and a second, smaller-radius query should run."""
    many_elements = [_overpass_element(f"Place {i}", 53.0 + i * 0.0001, -113.0) for i in range(20)]
    few_elements = [_overpass_element("Place 1", 53.0001, -113.0)]

    call_radii = []
    def fake_query_overpass(lat, lon, radius_m):
        call_radii.append(radius_m)
        return many_elements if len(call_radii) == 1 else few_elements

    with patch.object(resolver, "_query_overpass", side_effect=fake_query_overpass), \
         patch("src.venue_resolver.time.sleep"):
        results = resolver.search_nearby_named_features(53.0, -113.0, start_radius_m=1000)

    assert len(call_radii) >= 2, "should have retried with a different radius"
    assert call_radii[1] < call_radii[0], "radius should shrink after too many results"
    assert len(results) <= 15


@pytest.mark.unit
def test_search_expands_radius_when_too_few_results(resolver):
    """Rural area: first call returns zero results, so the radius should
    expand and a second, larger-radius query should run."""
    call_radii = []
    def fake_query_overpass(lat, lon, radius_m):
        call_radii.append(radius_m)
        return [] if len(call_radii) == 1 else [_overpass_element("Remote Lodge", 53.0, -113.0)]

    with patch.object(resolver, "_query_overpass", side_effect=fake_query_overpass), \
         patch("src.venue_resolver.time.sleep"):
        results = resolver.search_nearby_named_features(53.0, -113.0, start_radius_m=1000)

    assert len(call_radii) >= 2, "should have retried with a different radius"
    assert call_radii[1] > call_radii[0], "radius should expand after too few results"
    assert len(results) == 1


# ===== _dedupe_and_sort / _haversine_m =====

@pytest.mark.unit
def test_dedupe_and_sort_removes_duplicate_names(resolver):
    elements = [
        _overpass_element("Same Place", 53.001, -113.0),
        _overpass_element("Same Place", 53.002, -113.0),  # duplicate name, should be dropped
        _overpass_element("Other Place", 53.003, -113.0),
    ]
    results = resolver._dedupe_and_sort(elements, 53.0, -113.0)
    names = [r["name"] for r in results]
    assert names.count("Same Place") == 1
    assert "Other Place" in names


@pytest.mark.unit
def test_dedupe_and_sort_orders_by_distance(resolver):
    elements = [
        _overpass_element("Far Place", 53.01, -113.0),
        _overpass_element("Near Place", 53.0001, -113.0),
    ]
    results = resolver._dedupe_and_sort(elements, 53.0, -113.0)
    assert results[0]["name"] == "Near Place"
    assert results[0]["distance_m"] < results[1]["distance_m"]


@pytest.mark.unit
def test_haversine_distance_zero_for_identical_points(resolver):
    assert resolver._haversine_m(53.0, -113.0, 53.0, -113.0) == 0


@pytest.mark.unit
def test_haversine_distance_positive_for_different_points(resolver):
    dist = resolver._haversine_m(53.0, -113.0, 53.1, -113.1)
    assert dist > 0


# ===== resolve() orchestration =====

@pytest.mark.unit
def test_resolve_skips_search_when_venue_already_known(resolver):
    """category=venue_known means raw_geo already names a real place -
    resolve() should not search at all."""
    raw_geo = {"lat": "53.0", "lon": "-113.0", "name": "Original Joe's"}
    with patch.object(resolver, "judge", return_value="venue_known") as mock_judge, \
         patch.object(resolver, "search_nearby_named_features") as mock_search:
        result = resolver.resolve(raw_geo, ["dinner photo"])
    mock_judge.assert_called_once()
    mock_search.assert_not_called()
    assert result is None


@pytest.mark.unit
def test_resolve_skips_search_when_residence(resolver):
    """category=residence means it's not worth searching for a business."""
    raw_geo = {"lat": "53.0", "lon": "-113.0"}
    with patch.object(resolver, "judge", return_value="residence"), \
         patch.object(resolver, "search_nearby_named_features") as mock_search:
        result = resolver.resolve(raw_geo, ["living room photo"])
    mock_search.assert_not_called()
    assert result is None


@pytest.mark.unit
def test_resolve_returns_none_when_raw_geo_missing(resolver):
    assert resolver.resolve(None, ["a caption"]) is None
    assert resolver.resolve({}, ["a caption"]) is None


@pytest.mark.unit
def test_resolve_returns_none_when_captions_missing(resolver):
    assert resolver.resolve({"lat": "53.0", "lon": "-113.0"}, []) is None
    assert resolver.resolve({"lat": "53.0", "lon": "-113.0"}, None) is None


@pytest.mark.unit
def test_resolve_full_happy_path_returns_confirmed_venue(resolver):
    """judge -> business_unknown, search -> candidates, verify -> a match:
    the full pipeline should return the confirmed venue name."""
    raw_geo = {"lat": "53.3807457", "lon": "-113.6704144"}
    captions = ["Two people smiling outdoors at night in a snowy environment."]
    candidates = [{"name": "Rabbit Hill Snow Resort", "tags": {"sport": "skiing"}, "distance_m": 132}]

    with patch.object(resolver, "judge", return_value="business_unknown"), \
         patch.object(resolver, "search_nearby_named_features", return_value=candidates), \
         patch.object(resolver, "verify", return_value={"match": "Rabbit Hill Snow Resort", "reason": "ski setting"}):
        result = resolver.resolve(raw_geo, captions)

    assert result == "Rabbit Hill Snow Resort"
