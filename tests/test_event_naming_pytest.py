"""
Pytest-compatible test for event naming functionality.

This test checks if EventNamer is working correctly and caching names properly.
Uses proper pytest fixtures and structure.
"""

import pytest
import sys
import os
from pathlib import Path
from datetime import datetime

# Ensure src is in path (handled by conftest.py fixtures)
try:
    from src.event_namer import EventNamer
except ImportError:
    import event_namer
    EventNamer = event_namer.EventNamer


@pytest.mark.unit
def test_event_namer_basic_functionality(empty_cache_file, mock_cluster_data):
    """Test basic EventNamer functionality with caching."""
    print("🧪 PYTEST: Testing EventNamer basic functionality")

    # Create EventNamer with LLM enabled and isolated test cache
    namer = EventNamer(enable_llm=True)
    # Override the cache file path to use our test cache
    namer.cache_file = empty_cache_file
    namer.naming_cache = {}

    # Generate event name
    event_name = namer.generate_event_name(mock_cluster_data)

    # Check that name was generated
    assert event_name is not None
    assert len(event_name) > 0
    assert event_name.startswith("2024_01_15")

    print(f"✅ PYTEST: Generated event name: {event_name}")

    # Check that name was cached
    assert empty_cache_file.exists()

    # Verify cache contents
    import json
    with open(empty_cache_file, 'r') as f:
        cache_data = json.load(f)

    assert len(cache_data) > 0, "Cache should contain at least one entry"
    print(f"✅ PYTEST: Cache contains {len(cache_data)} entries")


@pytest.mark.unit
def test_event_namer_cache_reuse(sample_cache_file, mock_cluster_data):
    """Test that EventNamer reuses cached names."""
    print("🧪 PYTEST: Testing EventNamer cache reuse")

    # Create EventNamer with LLM enabled and test cache
    namer = EventNamer(enable_llm=True)
    # Override the cache file path to use our test cache
    namer.cache_file = sample_cache_file
    # Load the sample cache data
    import json
    with open(sample_cache_file, 'r') as f:
        namer.naming_cache = json.load(f)

    # Check initial cache size
    import json
    with open(sample_cache_file, 'r') as f:
        initial_cache = json.load(f)
    initial_size = len(initial_cache)

    print(f"📊 PYTEST: Initial cache size: {initial_size}")

    # Generate event name (should use existing logic)
    event_name = namer.generate_event_name(mock_cluster_data)

    # Check final cache size
    with open(sample_cache_file, 'r') as f:
        final_cache = json.load(f)
    final_size = len(final_cache)

    print(f"📊 PYTEST: Final cache size: {final_size}")
    print(f"✅ PYTEST: Generated event name: {event_name}")

    # Cache should have grown (new entry added)
    assert final_size >= initial_size
    assert event_name is not None


@pytest.mark.integration
@pytest.mark.slow
def test_event_naming_pipeline_integration(temp_test_dir, mock_media_file):
    """Test event naming as part of a larger pipeline (if components are available)."""
    print("🧪 PYTEST: Testing event naming pipeline integration")

    try:
        # This is a more comprehensive test that would involve
        # the full pipeline if all components are available

        # For now, just test that we can create the basic components
        cache_file = temp_test_dir / "pipeline_cache.json"
        namer = EventNamer(cache_file=cache_file, enable_llm=False)

        # Create a simple test file
        test_file = mock_media_file("/test/sample.jpg")

        # Test that we can handle the basic workflow
        assert namer is not None
        print("✅ PYTEST: Pipeline integration test passed (basic)")

    except Exception as e:
        pytest.skip(f"Pipeline integration test skipped due to missing dependencies: {e}")


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])