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

        # Call the method
        result = mock_event_namer._query_ollama_simple(mock_context_edmonton)

        # Verify the method was called with location constraints
        assert mock_post.called, "requests.post should be called"
        call_args = mock_post.call_args
        posted_data = call_args[1]['json']  # Second argument, 'json' key
        prompt = posted_data['prompt']

        # Verify location constraints are in the prompt
        assert 'Edmonton' in prompt, "Prompt should include actual location (Edmonton)"
        assert 'ONLY use Edmonton' in prompt, "Prompt should have location constraint"
        assert 'NOT Paris, Vegas' in prompt or 'NOT Paris' in prompt, "Prompt should warn against hallucination"

        # Verify result includes correct location
        assert result is not None, "Should return a result"
        assert 'Edmonton' in result, "Result should contain correct location"

    print("✅ Ollama simple query includes proper location constraints")


@pytest.mark.unit
def test_query_ollama_simple_prevents_location_hallucination(mock_event_namer, mock_context_edmonton):
    """Test that the prompt explicitly prevents location hallucination."""
    print("🧪 Testing Ollama prompt prevents location hallucination")

    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'response': 'Test Event - Edmonton'}
        mock_post.return_value = mock_response

        # Call the method
        mock_event_namer._query_ollama_simple(mock_context_edmonton)

        # Extract the prompt
        call_args = mock_post.call_args
        posted_data = call_args[1]['json']
        prompt = posted_data['prompt']

        # Verify specific anti-hallucination constraints
        assert 'NOT Paris, Vegas' in prompt or 'NOT Paris' in prompt, \
            "Prompt should explicitly warn against common hallucinated cities"
        assert 'ONLY use Edmonton' in prompt, \
            "Prompt should enforce using only the provided location"
        assert prompt.count('Edmonton') >= 2, \
            "Edmonton should be mentioned multiple times for emphasis"

    print("✅ Ollama prompt includes strong anti-hallucination constraints")


@pytest.mark.unit
def test_query_ollama_simple_with_unknown_location(mock_event_namer):
    """Test behavior when location is unknown."""
    print("🧪 Testing Ollama simple query with unknown location")

    context_no_location = {
        'temporal': {
            'date': '2014_10_25',
            'time_of_day': 'afternoon',
            'is_holiday': False
        },
        'location': {
            'city': None,  # No city information
            'has_gps': False
        }
    }

    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'response': 'Test Event - Unknown'}
        mock_post.return_value = mock_response

        # Call the method
        result = mock_event_namer._query_ollama_simple(context_no_location)

        # Extract the prompt
        call_args = mock_post.call_args
        posted_data = call_args[1]['json']
        prompt = posted_data['prompt']

        # Verify unknown location handling
        assert 'Unknown' in prompt, "Should use 'Unknown' when no city provided"
        assert 'ONLY use Unknown' in prompt, \
            "Should still enforce location constraint even with Unknown"

    print("✅ Ollama handles unknown location correctly")


@pytest.mark.unit
def test_query_ollama_simple_format_requirements(mock_event_namer, mock_context_edmonton):
    """Test that the prompt includes proper format requirements."""
    print("🧪 Testing Ollama prompt includes format requirements")

    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'response': '2014_10_25 - Test - Edmonton'}
        mock_post.return_value = mock_response

        # Call the method
        mock_event_namer._query_ollama_simple(mock_context_edmonton)

        # Extract the prompt
        call_args = mock_post.call_args
        posted_data = call_args[1]['json']
        prompt = posted_data['prompt']

        # Verify format requirements
        assert 'exact format' in prompt or 'Format:' in prompt, "Prompt should include format instructions"
        assert '2014_10_25 - Event Name - Edmonton' in prompt or '2014_10_25 - Family BBQ - Edmonton' in prompt, \
            "Prompt should show correct format example"
        assert 'Example' in prompt or 'CORRECT output' in prompt, "Prompt should include examples"

    print("✅ Ollama prompt includes proper format requirements")


@pytest.mark.unit
def test_query_ollama_simple_error_handling(mock_event_namer, mock_context_edmonton):
    """Test error handling in Ollama simple query."""
    print("🧪 Testing Ollama simple query error handling")

    # Test HTTP error
    with patch('requests.post') as mock_post:
        mock_post.side_effect = Exception("Connection error")

        result = mock_event_namer._query_ollama_simple(mock_context_edmonton)
        assert result is None, "Should return None on connection error"

    # Test empty response
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'response': ''}
        mock_post.return_value = mock_response

        result = mock_event_namer._query_ollama_simple(mock_context_edmonton)
        assert result is None, "Should return None on empty response"

    print("✅ Ollama simple query handles errors correctly")


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
        result = mock_event_namer._query_ollama_simple(mock_context_edmonton)

        # Extract and verify the prompt has location constraints
        call_args = mock_post.call_args
        posted_data = call_args[1]['json']
        prompt = posted_data['prompt']

        # Core regression prevention checks
        assert 'Edmonton' in prompt, \
            "REGRESSION: Prompt must include actual location from GPS data"
        assert 'ONLY use Edmonton' in prompt, \
            "REGRESSION: Prompt must enforce location constraint"
        assert 'NOT Paris' in prompt or 'NOT Paris, Vegas' in prompt, \
            "REGRESSION: Prompt must warn against hallucinated locations"

        # Verify result quality
        assert result is not None, "REGRESSION: Should generate a result"
        if result:
            assert 'Edmonton' in result, "REGRESSION: Result should contain correct location"

    print("✅ REGRESSION TEST PASSED: Issue #14 prevention mechanisms in place")


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
def test_query_ollama_simple_rejects_meta_text_responses(mock_event_namer, mock_context_edmonton):
    """Test that _query_ollama_simple rejects responses containing meta-text."""
    print("🧪 Testing Ollama query rejects meta-text responses")

    with patch('requests.post') as mock_post:
        # Simulate LLM returning meta-text (should be rejected)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'Here are a few options for a short folder name for your photos'
        }
        mock_post.return_value = mock_response

        # Call the method
        result = mock_event_namer._query_ollama_simple(mock_context_edmonton)

        # Should return None because meta-text was detected
        assert result is None, "Should reject response containing meta-text"

    print("✅ Ollama query correctly rejects meta-text responses")


@pytest.mark.unit
def test_query_ollama_simple_accepts_clean_responses(mock_event_namer, mock_context_edmonton):
    """Test that _query_ollama_simple accepts clean event names."""
    print("🧪 Testing Ollama query accepts clean responses")

    with patch('requests.post') as mock_post:
        # Simulate LLM returning clean event name
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': '2014_10_25 - Weekend Shopping - Edmonton'
        }
        mock_post.return_value = mock_response

        # Call the method
        result = mock_event_namer._query_ollama_simple(mock_context_edmonton)

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

        # Call the method
        mock_event_namer._query_ollama_simple(mock_context_edmonton)

        # Extract the prompt
        call_args = mock_post.call_args
        posted_data = call_args[1]['json']
        prompt = posted_data['prompt']

        # Verify directive structure (not explanatory)
        assert 'Output ONLY a folder name' in prompt, \
            "Prompt should use directive 'Output ONLY' instruction"
        assert 'WRONG - Do NOT output:' in prompt, \
            "Prompt should include negative examples"
        assert 'Here are options' in prompt, \
            "Prompt should show examples of WRONG meta-text output"
        assert 'Examples of CORRECT output:' in prompt, \
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
        # Simulate what happened before fix: LLM returns meta-text
        mock_response_bad = Mock()
        mock_response_bad.status_code = 200
        mock_response_bad.json.return_value = {
            'response': '2014_10_25 - Here are a few options for a short folder name'
        }
        mock_post.return_value = mock_response_bad

        # This exact scenario caused meta-text output before fix
        result_bad = mock_event_namer._query_ollama_simple(mock_context_edmonton)

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

        result_good = mock_event_namer._query_ollama_simple(mock_context_edmonton)

        # Clean output should be accepted
        assert result_good is not None, \
            "REGRESSION: Clean event names must be accepted"
        assert 'Weekend Shopping' in result_good, \
            "REGRESSION: Should return actual event name"

        # Verify the prompt includes directive structure
        call_args = mock_post.call_args
        posted_data = call_args[1]['json']
        prompt = posted_data['prompt']

        assert 'Output ONLY a folder name' in prompt, \
            "REGRESSION: Prompt must use directive structure"
        assert 'WRONG - Do NOT output:' in prompt, \
            "REGRESSION: Prompt must include negative examples"

    print("✅ REGRESSION TEST PASSED: Issue #41 prevention mechanisms in place")


if __name__ == "__main__":
    """Run the event namer unit tests standalone."""
    print("🧪 Event Namer Unit Tests")
    print("📋 Test Categories:")
    print("   1. Ollama simple query location constraint tests (Issue #14)")
    print("   2. Meta-text detection and validation tests (Issue #41)")
    print("   3. Prompt generation validation tests")
    print("   4. Error handling tests")
    print("   5. Regression prevention tests (Issues #14, #41)")
    print()
    print("⚡ Expected time: <5 seconds total")
    print("🔧 Testing approach: Fast unit tests with mocked dependencies")
    print()

    # Run with pytest
    pytest.main([__file__, "-v"])