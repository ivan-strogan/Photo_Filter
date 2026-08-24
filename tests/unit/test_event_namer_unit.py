#!/usr/bin/env python3
"""
Unit tests for event naming components.

These are FAST unit tests that test individual methods in isolation using
mocked dependencies. No real LLM calls, no real file operations.

Focus:
- EventNamer method behavior with controlled inputs
- Prompt generation logic validation
- Location constraint enforcement
- Error handling and edge cases

Run with: pytest tests/unit/test_event_namer_unit.py -v
Expected time: <3 seconds total
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import json

# Import core classes
try:
    from src.event_namer import EventNamer
except ImportError:
    import event_namer
    EventNamer = event_namer.EventNamer


# ===== MOCK FIXTURES =====

@pytest.fixture
def mock_context_edmonton():
    """Mock context data for Edmonton location (Issue #14 reproduction)."""
    return {
        'temporal': {
            'date': '2014_10_25',
            'time_of_day': 'afternoon',
            'day_of_week': 'Saturday',
            'duration_hours': 0.87,
            'duration_category': 'Short Event',
            'season': 'fall',
            'is_weekend': True,
            'is_holiday': False
        },
        'location': {
            'has_gps': True,
            'city': 'Edmonton',
            'state': 'Alberta',
            'country': 'Canada',
            'venue_type': 'unknown',
            'location_nickname': 'Edmonton',
            'full_location': 'Edmonton, Alberta'
        },
        'content': {
            'objects': [],
            'scenes': [],
            'activities': [],
            'confidence': 0.0,
            'primary_activity': 'unknown',
            'event_type': 'general',
            'content_tags': []
        },
        'media': {
            'total_files': 4,
            'photo_count': 4,
            'video_count': 0,
            'media_ratio': 1.0,
            'capture_pattern': 'burst'
        },
        'people': {
            'people_detected': [],
            'people_count': 0,
            'face_count': 0,
            'people_consistency': 0.0,
            'has_people': False,
            'main_people': '',
            'people_category': 'no_people'
        }
    }


@pytest.fixture
def mock_event_namer():
    """Mock EventNamer for unit tests."""
    # Create EventNamer with mocked dependencies
    with patch('requests.post') as mock_post:
        event_namer = EventNamer(
            enable_llm=True,
            ollama_model="llama3.1:8b",
            ollama_url="http://localhost:11434"
        )
        event_namer.use_ollama = True
        event_namer.use_openai = False
        return event_namer


# ===== UNIT TESTS FOR ISSUE #14 FIX =====

@pytest.mark.unit
def test_query_ollama_simple_includes_location_constraint(mock_event_namer, mock_context_edmonton):
    """Test that _query_ollama_simple includes location constraints (Issue #14 fix)."""
    print("🧪 Testing Ollama simple query includes location constraints")

    # Mock the requests.post call to capture the prompt
    with patch('requests.post') as mock_post:
        # Mock successful Ollama response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': '2014_10_25 - Afternoon Shopping - Edmonton'
        }
        mock_post.return_value = mock_response

        # Build prompt from context and call Ollama (new method names)
        prompt = mock_event_namer._build_naming_prompt(mock_context_edmonton)
        result = mock_event_namer._query_ollama(prompt)

        # Verify the method was called
        assert mock_post.called, "requests.post should be called"

        # Verify location constraints are in the built prompt (home mode - Edmonton
        # is the configured home city, so the prompt withholds rather than states it)
        assert 'Edmonton' in prompt, "Prompt should include actual location (Edmonton)"
        assert 'DO NOT invent or state a city or state anywhere in the name' in prompt, \
            "Prompt should have location constraint"
        assert 'never state the city or state in the folder name' in prompt, \
            "Prompt should warn against hallucination"

        # Verify result includes correct location
        assert result is not None, "Should return a result"
        assert 'Edmonton' in result, "Result should contain correct location"

    print("✅ Ollama simple query includes proper location constraints")


@pytest.mark.unit
def test_query_ollama_prevents_location_hallucination(mock_event_namer, mock_context_edmonton):
    """Test that the prompt explicitly prevents location hallucination."""
    print("🧪 Testing Ollama prompt prevents location hallucination")

    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'response': 'Test Event - Edmonton'}
        mock_post.return_value = mock_response

        # Build prompt and call Ollama (new method names)
        prompt = mock_event_namer._build_naming_prompt(mock_context_edmonton)
        mock_event_namer._query_ollama(prompt)

        # Verify specific anti-hallucination constraints (home mode)
        assert 'DO NOT invent or state a city or state anywhere in the name' in prompt, \
            "Prompt should explicitly warn against location hallucination"
        assert 'never state the city or state in the folder name' in prompt, \
            "Prompt should enforce withholding the home city"
        assert prompt.count('Edmonton') >= 2, \
            "Edmonton should be mentioned multiple times for emphasis"

    print("✅ Ollama prompt includes strong anti-hallucination constraints")


@pytest.mark.unit
def test_query_ollama_with_unknown_location(mock_event_namer, mock_context_edmonton):
    """Test behavior when location is unknown."""
    print("🧪 Testing Ollama query with unknown location")

    # Create a copy of context with no location
    import copy
    context_no_location = copy.deepcopy(mock_context_edmonton)
    context_no_location['location']['city'] = None  # No city information
    context_no_location['location']['has_gps'] = False

    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'response': 'Test Event - Unknown'}
        mock_post.return_value = mock_response

        # Build prompt and call Ollama (new method names)
        prompt = mock_event_namer._build_naming_prompt(context_no_location)
        result = mock_event_namer._query_ollama(prompt)

        # Verify unknown location handling - no GPS routes to the dedicated
        # "unknown" mode prompt, which omits location entirely rather than
        # stating "Unknown" as if it were a real city value
        assert 'Unknown' in prompt, "Should use 'Unknown' when no city provided"
        assert 'DO NOT invent a location - none is available' in prompt, \
            "Should still enforce a location constraint even with no GPS"

    print("✅ Ollama handles unknown location correctly")


@pytest.mark.unit
def test_query_ollama_format_requirements(mock_event_namer, mock_context_edmonton):
    """Test that the prompt includes proper format requirements."""
    print("🧪 Testing Ollama prompt includes format requirements")

    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'response': '2014_10_25 - Test - Edmonton'}
        mock_post.return_value = mock_response

        # Build prompt and call Ollama (new method names)
        prompt = mock_event_namer._build_naming_prompt(mock_context_edmonton)
        mock_event_namer._query_ollama(prompt)

        # Verify format requirements (updated for new prompt format)
        assert 'Format Requirements' in prompt, "Prompt should include format instructions section"
        assert 'YYYY_MM_DD' in prompt, "Prompt should specify date format"
        assert 'Examples (home location is' in prompt, \
            "Prompt should include example outputs"

    print("✅ Ollama prompt includes proper format requirements")


@pytest.mark.unit
def test_query_ollama_error_handling(mock_event_namer, mock_context_edmonton):
    """Test error handling in Ollama query."""
    print("🧪 Testing Ollama query error handling")

    # Build a test prompt
    prompt = mock_event_namer._build_naming_prompt(mock_context_edmonton)

    # Test HTTP error
    with patch('requests.post') as mock_post:
        mock_post.side_effect = Exception("Connection error")

        result = mock_event_namer._query_ollama(prompt)
        assert result is None, "Should return None on connection error"

    # Test empty response
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'response': ''}
        mock_post.return_value = mock_response

        result = mock_event_namer._query_ollama(prompt)
        assert result is None, "Should return None on empty response"

    print("✅ Ollama query handles errors correctly")


# ===== REGRESSION TESTS FOR ISSUE #14 =====

@pytest.mark.unit
@pytest.mark.regression
def test_issue_14_regression_prevention(mock_event_namer, mock_context_edmonton):
    """Regression test to prevent Issue #14 from reoccurring."""
    print("🧪 REGRESSION TEST: Issue #14 - LLM location hallucination prevention")

    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'response': '2014_10_25 - Shopping Trip - Edmonton'}
        mock_post.return_value = mock_response

        # This exact scenario caused Edmonton -> Paris hallucination before fix
        # Build prompt and call Ollama (new method names)
        prompt = mock_event_namer._build_naming_prompt(mock_context_edmonton)
        result = mock_event_namer._query_ollama(prompt)

        # Core regression prevention checks (home mode - Edmonton is home city)
        assert 'Edmonton' in prompt, \
            "REGRESSION: Prompt must include actual location from GPS data"
        assert 'DO NOT invent or state a city or state anywhere in the name' in prompt, \
            "REGRESSION: Prompt must enforce location constraint"
        assert 'never state the city or state in the folder name' in prompt, \
            "REGRESSION: Prompt must warn against stating an invented/wrong city"

        # Verify result quality
        assert result is not None, "REGRESSION: Should generate a result"
        if result:
            assert 'Edmonton' in result, "REGRESSION: Result should contain correct location"

    print("✅ REGRESSION TEST PASSED: Issue #14 prevention mechanisms in place")


# ===== REGRESSION TESTS FOR ISSUE #74 (VALIDATOR BROKEN BY HOME/AWAY FORMAT) =====

@pytest.mark.unit
@pytest.mark.regression
def test_validate_event_name_home_event_without_city_passes(mock_event_namer, mock_context_edmonton):
    """Home-city event names correctly omit the city (Issue #74)."""
    assert mock_event_namer._validate_event_name(
        '2024_01_15 - Christmas Morning', mock_context_edmonton
    ) is True


@pytest.mark.unit
@pytest.mark.regression
def test_validate_event_name_away_event_with_correct_city_passes(mock_event_namer, mock_context_edmonton):
    """Away-event names that fold the correct city into the description pass (Issue #74)."""
    import copy
    context = copy.deepcopy(mock_context_edmonton)
    context['location']['city'] = 'Calgary'

    assert mock_event_namer._validate_event_name(
        '2024_01_15 - Calgary Trip', context
    ) is True


@pytest.mark.unit
@pytest.mark.regression
def test_validate_event_name_rejects_home_city_stated_at_home(mock_event_namer, mock_context_edmonton):
    """Home-city event names that state the city anyway are rejected (Issue #74)."""
    assert mock_event_namer._validate_event_name(
        '2024_01_15 - Christmas Morning - Edmonton', mock_context_edmonton
    ) is False


@pytest.mark.unit
@pytest.mark.regression
def test_validate_event_name_rejects_home_city_stated_while_away(mock_event_namer, mock_context_edmonton):
    """Rejects a name stating the home city while the event is actually elsewhere -
    the original Issue #14 failure mode (real GPS location replaced with a
    different one), now detected without relying on a fixed name segment."""
    import copy
    context = copy.deepcopy(mock_context_edmonton)
    context['location']['city'] = 'Calgary'

    assert mock_event_namer._validate_event_name(
        '2024_01_15 - Weekend in Edmonton', context
    ) is False


@pytest.mark.unit
@pytest.mark.regression
def test_validate_event_name_rejects_unknown_when_city_known(mock_event_namer, mock_context_edmonton):
    """Rejects the 'Unknown' placeholder when a real city was provided (Issue #14)."""
    assert mock_event_namer._validate_event_name(
        '2024_01_15 - Test Event - Unknown', mock_context_edmonton
    ) is False


# ===== UNIT TESTS FOR ISSUE #41 FIX (META-TEXT DETECTION) =====

@pytest.mark.unit
def test_contains_meta_text_detects_bad_outputs(mock_event_namer):
    """Test that _contains_meta_text correctly identifies meta-text phrases."""
    print("🧪 Testing meta-text detection for bad outputs")

    # Test cases from actual bad outputs (Issue #41)
    bad_outputs = [
        "2014_10_25 - Here is a short folder name for photos from",
        "2014_10_27 - Here are a few options:\\n\\n1. 201",
        "2014_10_30 - Here are a few options for a short folder name",
        "2016_02_15 - Here are a few options for a short folder name",
        "2014_11_08 - Here is a short folder name for photos from",
        "2016_01_01 - Create a folder name like this",
        "2016_03_19 - Suggestions for your photos",
        "2015_12_31 - Could be named something like",
    ]

    for bad_output in bad_outputs:
        result = mock_event_namer._contains_meta_text(bad_output)
        assert result == True, f"Should detect meta-text in: {bad_output}"

    print(f"✅ Detected meta-text in all {len(bad_outputs)} bad outputs")


@pytest.mark.unit
def test_contains_meta_text_allows_good_outputs(mock_event_namer):
    """Test that _contains_meta_text allows clean event names."""
    print("🧪 Testing meta-text detection allows good outputs")

    # Good event names that should NOT trigger meta-text detection
    good_outputs = [
        "2014_10_25 - Weekend Shopping - Edmonton",
        "2016_01_01 - New Year's Party - Edmonton",
        "2014_11_08 - Morning Coffee - Calgary",
        "2016_03_19 - Family Dinner - Edmonton",
        "2015_12_31 - New Year's Eve Celebration - Edmonton",
        "2016_02_15 - Valentine's Day - Calgary",
        "2014_10_30 - Halloween Party - Edmonton",
    ]

    for good_output in good_outputs:
        result = mock_event_namer._contains_meta_text(good_output)
        assert result == False, f"Should NOT detect meta-text in: {good_output}"

    print(f"✅ Accepted all {len(good_outputs)} good outputs")


@pytest.mark.unit
def test_query_ollama_rejects_meta_text_responses(mock_event_namer, mock_context_edmonton):
    """Test that _query_ollama rejects responses containing meta-text."""
    print("🧪 Testing Ollama query rejects meta-text responses")

    with patch('requests.post') as mock_post:
        # Simulate LLM returning meta-text (should be rejected)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'Here are a few options for a short folder name for your photos'
        }
        mock_post.return_value = mock_response

        # Build prompt and call Ollama (new method names)
        prompt = mock_event_namer._build_naming_prompt(mock_context_edmonton)
        result = mock_event_namer._query_ollama(prompt)

        # Should return None because meta-text was detected
        assert result is None, "Should reject response containing meta-text"

    print("✅ Ollama query correctly rejects meta-text responses")


@pytest.mark.unit
def test_query_ollama_accepts_clean_responses(mock_event_namer, mock_context_edmonton):
    """Test that _query_ollama accepts clean event names."""
    print("🧪 Testing Ollama query accepts clean responses")

    with patch('requests.post') as mock_post:
        # Simulate LLM returning clean event name
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': '2014_10_25 - Weekend Shopping - Edmonton'
        }
        mock_post.return_value = mock_response

        # Build prompt and call Ollama (new method names)
        prompt = mock_event_namer._build_naming_prompt(mock_context_edmonton)
        result = mock_event_namer._query_ollama(prompt)

        # Should return the clean event name
        assert result is not None, "Should accept clean response"
        assert 'Weekend Shopping' in result, "Should return the event name"
        assert 'Edmonton' in result, "Should include location"

    print("✅ Ollama query correctly accepts clean responses")


@pytest.mark.unit
def test_query_ollama_simple_directive_prompt_structure(mock_event_namer, mock_context_edmonton):
    """Test that Ollama prompt uses directive structure to prevent meta-text (Issue #41 fix)."""
    print("🧪 Testing Ollama prompt uses directive structure")

    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'response': 'Test Event - Edmonton'}
        mock_post.return_value = mock_response

        # Build prompt and call Ollama
        prompt = mock_event_namer._build_naming_prompt(mock_context_edmonton)
        result = mock_event_namer._query_ollama(prompt)

        # Verify directive structure (not explanatory)
        assert 'Generate ONLY the folder name' in prompt, \
            "Prompt should use directive 'Generate ONLY' instruction"
        assert 'Do NOT output:' in prompt, \
            "Prompt should include negative directive"
        assert 'Here are some options' in prompt, \
            "Prompt should show examples of WRONG meta-text output"
        assert 'Examples (home location is' in prompt, \
            "Prompt should show positive examples"

        # Verify it does NOT use explanatory phrasing
        assert 'Create a short folder name' not in prompt, \
            "Prompt should NOT use explanatory 'Create a...' phrasing"

    print("✅ Ollama prompt uses directive structure to prevent meta-text")


# ===== REGRESSION TESTS FOR ISSUE #41 =====

@pytest.mark.unit
@pytest.mark.regression
def test_issue_41_regression_prevention(mock_event_namer, mock_context_edmonton):
    """Regression test to prevent Issue #41 (LLM meta-text) from reoccurring."""
    print("🧪 REGRESSION TEST: Issue #41 - LLM meta-text prevention")

    with patch('requests.post') as mock_post:
        # Build prompt once
        prompt = mock_event_namer._build_naming_prompt(mock_context_edmonton)

        # Simulate what happened before fix: LLM returns meta-text
        mock_response_bad = Mock()
        mock_response_bad.status_code = 200
        mock_response_bad.json.return_value = {
            'response': '2014_10_25 - Here are a few options for a short folder name'
        }
        mock_post.return_value = mock_response_bad

        # This exact scenario caused meta-text output before fix
        result_bad = mock_event_namer._query_ollama(prompt)

        # Core regression prevention check: meta-text should be REJECTED
        assert result_bad is None, \
            "REGRESSION: Meta-text responses must be rejected by validation"

        # Now test that clean output is accepted
        mock_response_good = Mock()
        mock_response_good.status_code = 200
        mock_response_good.json.return_value = {
            'response': '2014_10_25 - Weekend Shopping - Edmonton'
        }
        mock_post.return_value = mock_response_good

        result_good = mock_event_namer._query_ollama(prompt)

        # Clean output should be accepted
        assert result_good is not None, \
            "REGRESSION: Clean event names must be accepted"
        assert 'Weekend Shopping' in result_good, \
            "REGRESSION: Should return actual event name"

        # Verify the prompt includes directive structure
        assert 'Generate ONLY the folder name' in prompt, \
            "REGRESSION: Prompt must use directive structure"
        assert 'Do NOT output:' in prompt, \
            "REGRESSION: Prompt must include negative directive"

    print("✅ REGRESSION TEST PASSED: Issue #41 prevention mechanisms in place")


# ===== UNIT TESTS FOR ISSUE #54 FIX (CACHE KEY GRANULARITY) =====

@pytest.mark.unit
def test_cache_key_includes_scene_and_object_info(mock_event_namer, mock_context_edmonton):
    """Test that cache key includes scene and object information (Issue #54 fix)."""
    print("🧪 Testing cache key includes scene and object info")

    import copy

    # Context with specific scenes and objects
    context_fitting_room = copy.deepcopy(mock_context_edmonton)
    context_fitting_room['content']['scenes'] = ['indoor', 'store']
    context_fitting_room['content']['objects'] = ['person', 'mirror', 'clothing']

    # Different context with different scenes and objects
    context_office = copy.deepcopy(mock_context_edmonton)
    context_office['content']['scenes'] = ['indoor', 'office']
    context_office['content']['objects'] = ['person', 'laptop', 'desk']

    # Generate cache keys
    key_fitting_room = mock_event_namer._generate_cache_key(context_fitting_room)
    key_office = mock_event_namer._generate_cache_key(context_office)

    # Keys should be different due to different content
    assert key_fitting_room != key_office, \
        "Cache keys should differ for different scenes/objects"

    # Keys should contain scene and object info
    assert 'indoor' in key_fitting_room, "Cache key should contain scene info"
    assert 'store' in key_fitting_room, "Cache key should contain scene info"
    assert 'mirror' in key_fitting_room or 'clothing' in key_fitting_room, \
        "Cache key should contain object info"

    print("✅ Cache key correctly includes scene and object info")


@pytest.mark.unit
def test_cache_key_includes_people_category(mock_event_namer, mock_context_edmonton):
    """Test that cache key includes people category (Issue #54 fix)."""
    print("🧪 Testing cache key includes people category")

    import copy

    # Context with solo person - non-empty scenes/objects, otherwise the
    # cache key is None regardless of people category (issue #76)
    context_solo = copy.deepcopy(mock_context_edmonton)
    context_solo['content']['scenes'] = ['kitchen']
    context_solo['content']['objects'] = ['table']
    context_solo['people']['people_category'] = 'solo'

    # Context with group
    context_group = copy.deepcopy(mock_context_edmonton)
    context_group['content']['scenes'] = ['kitchen']
    context_group['content']['objects'] = ['table']
    context_group['people']['people_category'] = 'group'

    # Generate cache keys
    key_solo = mock_event_namer._generate_cache_key(context_solo)
    key_group = mock_event_namer._generate_cache_key(context_group)

    # Keys should be different due to different people category
    assert key_solo != key_group, \
        "Cache keys should differ for different people categories"

    # Keys should contain people category
    assert 'solo' in key_solo, "Solo cache key should contain 'solo'"
    assert 'group' in key_group, "Group cache key should contain 'group'"

    print("✅ Cache key correctly includes people category")


@pytest.mark.unit
@pytest.mark.regression
def test_cache_key_none_when_content_empty(mock_event_namer, mock_context_edmonton):
    """Issue #76 regression: with no scenes/objects at all, the cache key
    used to collapse to fixed 'unknown_scene'/'unknown_objects' placeholders,
    so two genuinely different small events (same city/weekday/time-of-day)
    could collide on an identical key. _generate_cache_key must return None
    in this case so callers skip the cache entirely instead of risking a
    collision."""
    print("🧪 Testing cache key is None when content is empty")

    import copy

    # Context with empty content
    context_empty = copy.deepcopy(mock_context_edmonton)
    context_empty['content']['scenes'] = []
    context_empty['content']['objects'] = []

    key = mock_event_namer._generate_cache_key(context_empty)

    assert key is None, (
        "Cache key must be None when scenes and objects are both empty - "
        f"got {key!r}, which risks colliding with an unrelated cluster")

    print("✅ Cache key is None for empty content, as expected")


@pytest.mark.unit
@pytest.mark.regression
def test_issue_54_regression_different_content_different_keys(mock_event_namer, mock_context_edmonton):
    """Regression test: different photo content must produce different cache keys (Issue #54)."""
    print("🧪 REGRESSION TEST: Issue #54 - Cache key granularity")

    import copy

    # Simulate the original bug scenario:
    # Two photos with same temporal/location but different content

    # Photo 1: Fitting room (the misclassified photo)
    context_fitting = copy.deepcopy(mock_context_edmonton)
    context_fitting['content']['scenes'] = ['indoor', 'store']
    context_fitting['content']['objects'] = ['person', 'mirror', 'clothing']
    context_fitting['content']['activities'] = ['shopping']
    context_fitting['content']['primary_activity'] = 'shopping'
    context_fitting['people']['people_category'] = 'solo'

    # Photo 2: Office (what it was incorrectly cached as)
    context_office = copy.deepcopy(mock_context_edmonton)
    context_office['content']['scenes'] = ['indoor', 'office']
    context_office['content']['objects'] = ['person', 'laptop', 'desk']
    context_office['content']['activities'] = ['working']
    context_office['content']['primary_activity'] = 'working'
    context_office['people']['people_category'] = 'no_people'

    # Generate cache keys
    key_fitting = mock_event_namer._generate_cache_key(context_fitting)
    key_office = mock_event_namer._generate_cache_key(context_office)

    # CRITICAL: Keys MUST be different to prevent the Issue #54 bug
    assert key_fitting != key_office, \
        "REGRESSION #54: Different content MUST produce different cache keys"

    print("✅ REGRESSION TEST PASSED: Issue #54 cache key granularity fix verified")


@pytest.mark.unit
@pytest.mark.regression
def test_issue_62_cache_key_handles_tuple_format(mock_event_namer, mock_context_edmonton):
    """Regression test: cache key must handle tuple format from content analyzer (Issue #62).

    The ContentAnalyzer returns scenes/objects as tuples like ('home', 6) where
    the second element is a count. The cache key generation must extract just
    the names and not crash when encountering this format.
    """
    print("🧪 REGRESSION TEST: Issue #62 - Tuple format handling in cache key")

    import copy

    # Use TUPLE FORMAT as returned by ContentAnalyzer in production
    context_tuple = copy.deepcopy(mock_context_edmonton)
    context_tuple['content']['scenes'] = [('home', 6), ('indoor', 5), ('urban', 2)]
    context_tuple['content']['objects'] = [('tree', 3), ('wine', 3), ('person', 1)]

    # Should NOT crash - this was the Issue #62 bug
    key = mock_event_namer._generate_cache_key(context_tuple)

    # Key should contain the scene/object names (not tuples)
    assert 'home' in key or 'indoor' in key, \
        "Cache key should contain scene name from tuple"
    assert '(' not in key, \
        "REGRESSION #62: Cache key should NOT contain tuple parentheses"
    assert '6' not in key, \
        "REGRESSION #62: Cache key should NOT contain tuple counts"

    print("✅ REGRESSION TEST PASSED: Issue #62 tuple format handling verified")


@pytest.mark.unit
@pytest.mark.regression
def test_issue_62_event_naming_flow_completes(mock_event_namer, mock_context_edmonton):
    """Regression test: event naming flow must not crash before reaching LLM (Issue #62).

    This test ensures the entire event naming pipeline works with production-like
    data formats. If any step crashes (like cache key generation), this test fails.
    """
    print("🧪 REGRESSION TEST: Issue #62 - Event naming flow completion")

    import copy
    from unittest.mock import patch

    # Create context with TUPLE FORMAT (production format)
    context = copy.deepcopy(mock_context_edmonton)
    context['content']['scenes'] = [('home', 6), ('indoor', 5)]
    context['content']['objects'] = [('tree', 3), ('wine', 3)]
    context['content']['activities'] = [('drinking', 3)]

    # Create minimal cluster_data for generate_event_name
    # Note: start_time and end_time must be datetime objects, not strings
    cluster_data = {
        'start_time': datetime(2024, 1, 1, 12, 0, 0),
        'end_time': datetime(2024, 1, 1, 14, 0, 0),
        'duration_hours': 2.0,
        'size': 5,
        'photo_count': 5,
        'video_count': 0,
        'location_info': None,
        'dominant_location': 'Edmonton, Alberta',
        'gps_coordinates': [(53.5, -113.5)],
        'content_tags': ['tree', 'wine'],
        'content_analysis': {
            'top_scenes': [('home', 6), ('indoor', 5)],
            'top_objects': [('tree', 3), ('wine', 3)],
            'top_activities': [('drinking', 3)],
            'average_confidence': 0.9
        },
        'people_detected': ['Test Person'],
        'confidence_score': 0.8,
        'media_files': []
    }

    # Mock the LLM call to return a valid name (we just want to verify flow completes).
    # This is a home-city event (Edmonton), so the name correctly omits the city.
    with patch.object(mock_event_namer, '_generate_llm_name', return_value='2024_01_01 - Test Event'):
        # This should NOT crash - the flow must complete to the LLM call
        result = mock_event_namer.generate_event_name(cluster_data)

    # Verify we got a result (not None, which would indicate early crash)
    assert result is not None, \
        "REGRESSION #62: Event naming must not crash before returning a result"

    print("✅ REGRESSION TEST PASSED: Issue #62 event naming flow completes")


# ===== UNIT TESTS FOR ISSUE #67 (SIMILAR PAST EVENTS) =====

@pytest.mark.unit
def test_similarity_section_included_when_populated(mock_event_namer, mock_context_edmonton):
    """Prompt includes past event names when similarity context has matches (Issue #67)."""
    import copy
    context = copy.deepcopy(mock_context_edmonton)
    context['similarity'] = {
        'enabled': True,
        'similar_photos': [
            {'event_folder': "2023_01_15 - Elena's Birthday", 'similarity': 0.91},
            {'event_folder': "2022_01_14 - Elena's Birthday", 'similarity': 0.87},
        ]
    }

    prompt = mock_event_namer._build_naming_prompt(context)

    assert 'Similar Past Events' in prompt
    assert "2023_01_15 - Elena's Birthday" in prompt
    assert '0.91' in prompt


@pytest.mark.unit
def test_similarity_section_omitted_when_empty(mock_event_namer, mock_context_edmonton):
    """Prompt omits the similarity section cleanly when no context is provided (Issue #67)."""
    # mock_context_edmonton has no 'similarity' key at all
    prompt = mock_event_namer._build_naming_prompt(mock_context_edmonton)
    assert 'Similar Past Events' not in prompt


@pytest.mark.unit
def test_similarity_section_omitted_when_disabled_or_no_matches(mock_event_namer, mock_context_edmonton):
    """Prompt omits the similarity section when enabled but no similar photos were found (Issue #67)."""
    import copy
    context = copy.deepcopy(mock_context_edmonton)
    context['similarity'] = {'enabled': True, 'similar_photos': []}

    prompt = mock_event_namer._build_naming_prompt(context)
    assert 'Similar Past Events' not in prompt


@pytest.mark.unit
def test_similarity_section_dedupes_and_caps_at_five(mock_event_namer, mock_context_edmonton):
    """Repeated folders collapse to their best score, and the list is capped at 5 (Issue #67)."""
    import copy
    context = copy.deepcopy(mock_context_edmonton)
    context['similarity'] = {
        'enabled': True,
        'similar_photos': [
            {'event_folder': 'Folder A', 'similarity': 0.5},
            {'event_folder': 'Folder A', 'similarity': 0.8},  # same folder, higher score should win
            {'event_folder': 'Folder B', 'similarity': 0.7},
            {'event_folder': 'Folder C', 'similarity': 0.6},
            {'event_folder': 'Folder D', 'similarity': 0.4},
            {'event_folder': 'Folder E', 'similarity': 0.3},
            {'event_folder': 'Folder F', 'similarity': 0.2},
        ]
    }

    prompt = mock_event_namer._build_naming_prompt(context)

    assert 'Folder A (similarity: 0.80)' in prompt
    assert 'Folder A (similarity: 0.50)' not in prompt
    assert 'Folder F' not in prompt, "Should cap at the top 5 matches"


# ===== UNIT TESTS FOR ISSUE #72 (RICHER PROMPT CONTEXT) =====

@pytest.mark.unit
def test_location_spread_multi_location(mock_event_namer, mock_context_edmonton):
    """A wide GPS spread is labeled multi-location instead of just 'GPS available: Yes' (Issue #72)."""
    import copy
    context = copy.deepcopy(mock_context_edmonton)
    context['location']['gps_spread_km'] = 45.2

    prompt = mock_event_namer._build_naming_prompt(context)
    assert 'Multi-location' in prompt
    assert '45.2 km' in prompt


@pytest.mark.unit
def test_location_spread_single_venue(mock_event_namer, mock_context_edmonton):
    """A cluster with GPS but no meaningful spread is labeled a single venue (Issue #72)."""
    prompt = mock_event_namer._build_naming_prompt(mock_context_edmonton)
    assert 'Single venue' in prompt


@pytest.mark.unit
def test_sample_captions_included_in_prompt(mock_event_namer, mock_context_edmonton):
    """Representative vision-model captions are surfaced in the prompt (Issue #72)."""
    import copy
    context = copy.deepcopy(mock_context_edmonton)
    context['content']['sample_captions'] = [
        'A man kneels by a decorated Christmas tree holding a wrapped gift.'
    ]

    prompt = mock_event_namer._build_naming_prompt(context)
    assert 'Photos in this cluster' in prompt
    assert 'Photo 1 of 1' in prompt
    assert 'A man kneels by a decorated Christmas tree holding a wrapped gift.' in prompt


@pytest.mark.unit
def test_sample_captions_omitted_when_empty(mock_event_namer, mock_context_edmonton):
    """No photos section is added when there are no captions (Issue #72)."""
    prompt = mock_event_namer._build_naming_prompt(mock_context_edmonton)
    assert 'Photos in this cluster' not in prompt


@pytest.mark.unit
def test_people_guidance_included_for_small_group(mock_event_namer, mock_context_edmonton):
    """The prompt explicitly permits using a person's name for small, identified groups (Issue #72)."""
    import copy
    context = copy.deepcopy(mock_context_edmonton)
    context['people']['has_people'] = True
    context['people']['people_count'] = 1
    context['people']['main_people'] = 'Test Person'

    prompt = mock_event_namer._build_naming_prompt(context)
    assert 'Test Person' in prompt
    assert 'use their name' in prompt


@pytest.mark.unit
def test_people_guidance_omitted_when_no_people(mock_event_namer, mock_context_edmonton):
    """No people-naming guidance is added when nobody was identified (Issue #72)."""
    prompt = mock_event_namer._build_naming_prompt(mock_context_edmonton)
    assert 'use their name' not in prompt


@pytest.mark.unit
@pytest.mark.regression
def test_people_guidance_omitted_for_groups_larger_than_two(mock_event_namer, mock_context_edmonton):
    """Issue #80 follow-up: the "use their name" guidance must not appear
    for 3+ people - there's no single name a group event centers on, and
    the possessive example ("Jane, John & Amy's Birthday") reads wrong for
    a group anyway."""
    import copy
    context = copy.deepcopy(mock_context_edmonton)
    context['people']['has_people'] = True
    context['people']['people_count'] = 3
    context['people']['main_people'] = 'a group of 3'

    prompt = mock_event_namer._build_naming_prompt(context)
    assert 'use their name' not in prompt


@pytest.mark.unit
@pytest.mark.regression
def test_format_people_names_uses_first_name_only(mock_event_namer):
    """Issue #80: people names in the folder name should be first-name-only
    (e.g. "Jane" not "Jane Smith") - a full surname reads too formal for a
    personal photo folder."""
    assert mock_event_namer._format_people_names(['Jane Smith']) == 'Jane'
    assert mock_event_namer._format_people_names(['Jane Smith', 'John Doe']) == 'Jane & John'


@pytest.mark.unit
@pytest.mark.regression
def test_format_people_names_no_names_past_two_people(mock_event_namer):
    """Issue #80 follow-up: past 2 people, don't list any names - a name
    list doesn't scale as a folder title ("Jane, John, Amy & Sam's Birthday"
    reads like only Sam's) and there's no single name a group event centers
    on. Falls back to a plain headcount instead."""
    assert mock_event_namer._format_people_names(
        ['Jane Smith', 'John Doe', 'Amy Lee']) == 'a group of 3'
    assert mock_event_namer._format_people_names(
        ['Jane Smith', 'John Doe', 'Amy Lee', 'Sam Fox', 'Kim Park']) == 'a group of 5'


@pytest.mark.unit
def test_format_people_names_single_token_name_unchanged(mock_event_namer):
    """A name with no surname on file (just one token) is used as-is."""
    assert mock_event_namer._format_people_names(['Jane']) == 'Jane'


@pytest.mark.unit
def test_format_people_names_empty_list(mock_event_namer):
    assert mock_event_namer._format_people_names([]) == ''


@pytest.mark.unit
def test_examples_are_not_uniformly_seasonal_generic(mock_event_namer, mock_context_edmonton):
    """The example list mixes in people/content-driven names, not just <Season> <Generic> (Issue #72)."""
    prompt = mock_event_namer._build_naming_prompt(mock_context_edmonton)
    assert "Sarah's Birthday Dinner" in prompt or "Elena's Birthday" in prompt


# ===== UNIT TESTS FOR THE THREE-PROMPT NAMING ARCHITECTURE =====
#
# The single-prompt design asked the model to correctly GATE a conditional
# ("state the city only if away, never if home") and it kept bleeding
# across that gate - fixing an away-city bug broke home-city clusters. The
# fix resolves mode (home/away/unknown) in Python from data already
# available (GPS + city vs home_city), then builds a mode-specific prompt
# with only the ONE instruction that actually applies - no gate to get
# wrong. Validated against 48 real-cluster test runs (0 bugs) before being
# ported into production - see
# temp/diagnostics/2026-08-04_prompt_separation_competition/STATUS.md

@pytest.mark.unit
def test_determine_location_mode_unknown_when_no_gps(mock_event_namer, mock_context_edmonton):
    import copy
    context = copy.deepcopy(mock_context_edmonton)
    context['location']['has_gps'] = False
    context['location']['city'] = None

    assert mock_event_namer._determine_location_mode(context['location']) == 'unknown'


@pytest.mark.unit
def test_determine_location_mode_home_when_city_matches_home_city(mock_event_namer, mock_context_edmonton):
    # mock_context_edmonton's city ('Edmonton') matches the default home_city
    assert mock_event_namer._determine_location_mode(mock_context_edmonton['location']) == 'home'


@pytest.mark.unit
def test_determine_location_mode_home_is_case_insensitive(mock_event_namer, mock_context_edmonton):
    import copy
    context = copy.deepcopy(mock_context_edmonton)
    context['location']['city'] = 'EDMONTON'

    assert mock_event_namer._determine_location_mode(context['location']) == 'home'


@pytest.mark.unit
def test_determine_location_mode_away_when_city_differs(mock_event_namer, mock_context_edmonton):
    import copy
    context = copy.deepcopy(mock_context_edmonton)
    context['location']['city'] = 'Calgary'

    assert mock_event_namer._determine_location_mode(context['location']) == 'away'


@pytest.mark.unit
def test_build_naming_prompt_dispatches_to_home_builder(mock_event_namer, mock_context_edmonton):
    prompt = mock_event_namer._build_naming_prompt(mock_context_edmonton)
    assert "which is the person's home city" in prompt
    assert 'never state the city or state in the folder name' in prompt


@pytest.mark.unit
def test_build_naming_prompt_dispatches_to_away_builder(mock_event_namer, mock_context_edmonton):
    import copy
    context = copy.deepcopy(mock_context_edmonton)
    context['location']['city'] = 'Calgary'

    prompt = mock_event_namer._build_naming_prompt(context)
    assert 'Always fold the city ("Calgary")' in prompt
    assert '- City: Calgary' in prompt


@pytest.mark.unit
def test_build_naming_prompt_dispatches_to_unknown_builder(mock_event_namer, mock_context_edmonton):
    import copy
    context = copy.deepcopy(mock_context_edmonton)
    context['location']['has_gps'] = False
    context['location']['city'] = None

    prompt = mock_event_namer._build_naming_prompt(context)
    assert 'no GPS data is available' in prompt
    assert 'Edmonton' not in prompt, "unknown mode should never mention the home city"


@pytest.mark.unit
def test_home_prompt_never_states_city_even_with_area(mock_event_namer, mock_context_edmonton):
    import copy
    context = copy.deepcopy(mock_context_edmonton)
    context['location']['area'] = 'Aldergrove'

    prompt = mock_event_namer._build_naming_prompt_home(context)
    assert '- Area: Aldergrove' in prompt
    assert '- City:' not in prompt, "home prompt should never have a City: field to state"


@pytest.mark.unit
def test_home_prompt_omits_area_line_when_not_available(mock_event_namer, mock_context_edmonton):
    prompt = mock_event_namer._build_naming_prompt_home(mock_context_edmonton)
    assert '- Area:' not in prompt


@pytest.mark.unit
def test_home_prompt_includes_venue_area_mutual_exclusivity_rule(mock_event_namer, mock_context_edmonton):
    """Regression coverage: a venue name and an Area were sometimes combined
    (e.g. "Dinner at Original Joes in Westmount") until this rule was made
    explicit and unconditional - see STATUS.md 'Home prompt - fixes made'."""
    prompt = mock_event_namer._build_naming_prompt_home(mock_context_edmonton)
    assert 'do NOT also state the Area/neighbourhood' in prompt
    assert 'never combine them' in prompt


@pytest.mark.unit
def test_home_prompt_directs_search_for_venue_name_in_raw_geo(mock_event_namer, mock_context_edmonton):
    """Regression coverage: 'Ed's Bowling' was only findable because the
    prompt directs the model to look beyond the top-level 'name' field."""
    import copy
    context = copy.deepcopy(mock_context_edmonton)
    context['location']['raw_geo'] = {'name': '123 Some Street', 'address': {'road': "Ed's Bowling"}}

    prompt = mock_event_namer._build_naming_prompt_home(context)
    assert 'not just the top-level "name" field' in prompt
    assert "Ed's Bowling" in prompt  # raw geo JSON dump should be present verbatim


@pytest.mark.unit
def test_away_prompt_always_states_city_regardless_of_venue(mock_event_namer, mock_context_edmonton):
    """Regression coverage: a venue name used to make the model skip the
    city-folding rule entirely (cluster 9 lost 'in Calgary') until this was
    made unconditional - see STATUS.md."""
    import copy
    context = copy.deepcopy(mock_context_edmonton)
    context['location']['city'] = 'Calgary'
    context['location']['raw_geo'] = {'name': 'Elephant & Castle'}

    prompt = mock_event_namer._build_naming_prompt_away(context)
    assert 'do this whether or not a venue name is used' in prompt
    assert '- City: Calgary' in prompt


@pytest.mark.unit
def test_away_prompt_does_not_reference_home_city(mock_event_namer, mock_context_edmonton):
    """The away prompt never needs to reference 'home' as a concept - the
    home/away decision is already resolved in Python before this prompt is
    built, so referencing it here just reintroduces unnecessary
    comparison-reasoning (removed per user direction)."""
    import copy
    context = copy.deepcopy(mock_context_edmonton)
    context['location']['city'] = 'Calgary'

    prompt = mock_event_namer._build_naming_prompt_away(context)
    assert 'home' not in prompt.lower()


@pytest.mark.unit
def test_unknown_prompt_never_mentions_city_or_area(mock_event_namer, mock_context_edmonton):
    import copy
    context = copy.deepcopy(mock_context_edmonton)
    context['location']['has_gps'] = False
    context['location']['city'] = None
    context['location']['area'] = 'Some Neighbourhood'  # should still never appear

    prompt = mock_event_namer._build_naming_prompt_unknown(context)
    assert '- City:' not in prompt
    assert '- Area:' not in prompt
    assert 'Some Neighbourhood' not in prompt


# ===== UNIT TESTS FOR VENUE RESOLVER WIRING (_resolve_venue) =====

@pytest.mark.unit
def test_resolve_venue_injects_confirmed_venue_into_raw_geo(mock_event_namer, mock_context_edmonton):
    import copy
    context = copy.deepcopy(mock_context_edmonton)
    context['location']['raw_geo'] = {'lat': '53.0', 'lon': '-113.0'}
    context['content']['sample_captions'] = ['A photo of a ski hill.']

    mock_event_namer.venue_resolver = Mock()
    mock_event_namer.venue_resolver.resolve.return_value = 'Rabbit Hill Snow Resort'

    proceed = mock_event_namer._resolve_venue(context)

    assert proceed is True
    assert context['location']['raw_geo']['name'] == 'Rabbit Hill Snow Resort'


@pytest.mark.unit
def test_resolve_venue_returns_false_on_venue_search_error(mock_event_namer, mock_context_edmonton):
    from src.venue_resolver import VenueSearchError
    import copy
    context = copy.deepcopy(mock_context_edmonton)
    context['location']['raw_geo'] = {'lat': '53.0', 'lon': '-113.0'}
    context['content']['sample_captions'] = ['A photo of a restaurant.']

    mock_event_namer.venue_resolver = Mock()
    mock_event_namer.venue_resolver.resolve.side_effect = VenueSearchError('simulated failure')

    proceed = mock_event_namer._resolve_venue(context)

    assert proceed is False, "a genuine search failure should signal 'skip naming', not proceed with stale data"


@pytest.mark.unit
def test_resolve_venue_skips_when_no_raw_geo(mock_event_namer, mock_context_edmonton):
    context = mock_context_edmonton  # no raw_geo key at all
    mock_event_namer.venue_resolver = Mock()

    proceed = mock_event_namer._resolve_venue(context)

    assert proceed is True
    mock_event_namer.venue_resolver.resolve.assert_not_called()


@pytest.mark.unit
def test_resolve_venue_skips_when_no_captions(mock_event_namer, mock_context_edmonton):
    import copy
    context = copy.deepcopy(mock_context_edmonton)
    context['location']['raw_geo'] = {'lat': '53.0', 'lon': '-113.0'}
    # content.sample_captions left unset (empty)

    mock_event_namer.venue_resolver = Mock()
    proceed = mock_event_namer._resolve_venue(context)

    assert proceed is True
    mock_event_namer.venue_resolver.resolve.assert_not_called()


@pytest.mark.unit
def test_venue_resolver_disabled_by_default_flag(mock_context_edmonton):
    """enable_venue_search=False should mean no VenueResolver is constructed
    at all, regardless of whether the module is importable."""
    with patch('requests.post'):
        namer = EventNamer(enable_llm=True, ollama_model="llama3.1:8b", enable_venue_search=False)
    assert namer.venue_resolver is None


@pytest.mark.unit
def test_generate_event_name_returns_none_when_venue_search_fails(mock_event_namer, tmp_path):
    """End-to-end wiring check: a VenueSearchError during naming should
    make generate_event_name return None (skip this cycle) rather than
    proceed with an unenriched, potentially-misleading raw_geo."""
    from src.venue_resolver import VenueSearchError
    from datetime import datetime

    mock_event_namer.cache_file = tmp_path / "isolated_cache.json"
    mock_event_namer.naming_cache = {}
    mock_event_namer.venue_resolver = Mock()
    mock_event_namer.venue_resolver.resolve.side_effect = VenueSearchError('simulated failure')
    mock_event_namer._query_ollama = Mock(return_value="Should not be reached")

    cluster_data = {
        'files': [],
        'start_time': datetime(2024, 1, 15, 14, 0),
        'end_time': datetime(2024, 1, 15, 15, 0),
        'location_info': {
            'city': 'Calgary', 'state': 'Alberta', 'country': 'Canada',
            'latitude': 51.0, 'raw_data': {'lat': '51.0', 'lon': '-114.0'},
        },
        'content_analysis': {
            'average_confidence': 0.8,
            'sample_captions': ['A photo of a restaurant.'],
        },
    }

    result = mock_event_namer.generate_event_name(cluster_data)

    assert result is None
    mock_event_namer.venue_resolver.resolve.assert_called_once()
    mock_event_namer._query_ollama.assert_not_called()


def _make_people_cluster_data(people_detected):
    """cluster_data for the end-to-end people-naming tests below - real
    scenes/objects so the naming cache key isn't None (issue #76) and
    doesn't interfere with these tests."""
    from datetime import datetime
    return {
        'files': [],
        'start_time': datetime(2024, 1, 15, 14, 0),
        'end_time': datetime(2024, 1, 15, 15, 0),
        'location_info': {
            'city': 'Edmonton', 'state': 'Alberta', 'country': 'Canada',
            'latitude': 53.5, 'raw_data': {'lat': '53.5', 'lon': '-113.5'},
        },
        'content_analysis': {
            'average_confidence': 0.8,
            'top_scenes': ['backyard'],
            'top_objects': ['patio'],
            'sample_captions': ['A group of people on a patio.'],
        },
        'people_detected': people_detected,
    }


@pytest.mark.unit
@pytest.mark.regression
def test_end_to_end_two_people_from_face_db_appear_in_prompt(mock_event_namer, tmp_path):
    """Issue #80 end-to-end: with 2 people_detected (as real face
    recognition against the people database would produce), the actual
    prompt sent to the LLM must show their first names and the
    "use their name" guidance - not just the isolated _format_people_names
    helper in a vacuum."""
    mock_event_namer.cache_file = tmp_path / "isolated_cache.json"
    mock_event_namer.naming_cache = {}
    mock_event_namer.venue_resolver = None
    mock_event_namer._query_ollama = Mock(return_value="2024_01_15 - Jane & John's Patio Afternoon")

    cluster_data = _make_people_cluster_data(['Jane Smith', 'John Doe'])
    mock_event_namer.generate_event_name(cluster_data)

    prompt = mock_event_namer._query_ollama.call_args[0][0]
    assert 'Jane & John' in prompt
    assert 'Jane Smith' not in prompt and 'John Doe' not in prompt, (
        "Full surnames must not reach the prompt, only first names")
    assert 'use their name' in prompt


@pytest.mark.unit
@pytest.mark.regression
def test_end_to_end_three_people_from_face_db_no_names_in_prompt(mock_event_namer, tmp_path):
    """Issue #80 end-to-end: with 3 people_detected, the actual prompt sent
    to the LLM must not name anyone (no first names, no "use their name"
    guidance) - falls back to a plain headcount instead."""
    mock_event_namer.cache_file = tmp_path / "isolated_cache.json"
    mock_event_namer.naming_cache = {}
    mock_event_namer.venue_resolver = None
    mock_event_namer._query_ollama = Mock(return_value="2024_01_15 - Patio Afternoon with Friends")

    cluster_data = _make_people_cluster_data(['Jane Smith', 'John Doe', 'Amy Lee'])
    mock_event_namer.generate_event_name(cluster_data)

    prompt = mock_event_namer._query_ollama.call_args[0][0]
    for name in ('Jane', 'John', 'Amy'):
        assert name not in prompt, f"'{name}' must not appear in the prompt for a 3-person group"
    assert 'use their name' not in prompt
    assert 'a group of 3' in prompt


@pytest.mark.unit
@pytest.mark.regression
def test_issue_76_regression_empty_content_clusters_dont_share_cached_name(mock_event_namer, tmp_path):
    """Issue #76 regression: two different real clusters on the same day/
    city/weekday with no scene/object content analysis (a real, common case
    for small clusters) used to collide on the same cache key and silently
    share a cached name - e.g. a Calgary sushi dinner cluster and a
    separate same-day foosball cluster both ended up named after the sushi
    dinner. Each must get its own freshly-generated name instead."""
    from datetime import datetime

    mock_event_namer.cache_file = tmp_path / "isolated_cache.json"
    mock_event_namer.naming_cache = {}
    mock_event_namer.venue_resolver = None
    mock_event_namer._query_ollama = Mock(side_effect=[
        "2016_01_30 - Sushi Dinner at Pocket Holic in Calgary",
        "2016_01_30 - Foosball with Friends in Calgary",
    ])

    def make_cluster(start_hour, start_minute):
        return {
            'files': [],
            'start_time': datetime(2016, 1, 30, start_hour, start_minute),
            'end_time': datetime(2016, 1, 30, start_hour, start_minute + 20),
            'location_info': {
                'city': 'Calgary', 'state': 'Alberta', 'country': 'Canada',
                'latitude': 51.0, 'raw_data': {'lat': '51.0', 'lon': '-114.0'},
            },
            'content_analysis': {'average_confidence': 0.8},
        }

    name_1 = mock_event_namer.generate_event_name(make_cluster(18, 39))
    name_2 = mock_event_namer.generate_event_name(make_cluster(20, 9))

    assert name_1 != name_2, (
        f"Two different clusters got the same name via a cache collision: {name_1!r}")
    assert mock_event_namer._query_ollama.call_count == 2, (
        "Second cluster must generate its own name, not reuse a cached one")


if __name__ == "__main__":
    """Run the event namer unit tests standalone."""
    print("🧪 Event Namer Unit Tests")
    print("📋 Test Categories:")
    print("   1. Ollama simple query location constraint tests (Issue #14)")
    print("   2. Meta-text detection and validation tests (Issue #41)")
    print("   3. Prompt generation validation tests")
    print("   4. Error handling tests")
    print("   5. Regression prevention tests (Issues #14, #41)")
    print("   6. Cache key granularity tests (Issue #54)")
    print()
    print("⚡ Expected time: <5 seconds total")
    print("🔧 Testing approach: Fast unit tests with mocked dependencies")
    print()

    # Run with pytest
    pytest.main([__file__, "-v"])