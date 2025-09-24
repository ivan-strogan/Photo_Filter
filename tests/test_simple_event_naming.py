#!/usr/bin/env python3
"""
Simple test to check if EventNamer is working and caching names.
"""

import sys
import os
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime

# Add src to path and set environment
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
os.chdir(Path(__file__).parent.parent)

# Import just the EventNamer
import event_namer

def test_event_namer_caching():
    """Test if EventNamer generates and caches names correctly."""
    print("🧪 SIMPLE EVENT NAMING TEST")
    print("=" * 40)

    # Create temporary cache file
    test_dir = Path(tempfile.mkdtemp(prefix="simple_naming_test_"))
    test_cache_file = test_dir / "test_cache.json"

    try:
        # Initialize with empty cache
        with open(test_cache_file, 'w') as f:
            json.dump({}, f)

        print(f"📁 Using test cache: {test_cache_file}")

        # Create EventNamer (disable LLM for consistency)
        namer = event_namer.EventNamer(enable_llm=False)
        # Set the cache file path manually
        namer.cache_file = test_cache_file
        namer.naming_cache = {}  # Clear the in-memory cache
        print(f"🎯 EventNamer initialized with LLM enabled: {namer.enable_llm}")
        print(f"🎯 Cache file set to: {namer.cache_file}")

        # Create test cluster data
        cluster_data = {
            'files': [
                {'path': 'photo1.jpg', 'timestamp': datetime(2024, 10, 25, 14, 30)},
                {'path': 'photo2.jpg', 'timestamp': datetime(2024, 10, 25, 14, 45)}
            ],
            'start_time': datetime(2024, 10, 25, 14, 30),
            'end_time': datetime(2024, 10, 25, 16, 30),
            'location_info': {
                'city': 'Edmonton',
                'country': 'Canada',
                'coordinates': (53.5461, -113.4938)
            },
            'content_analysis': {
                'activities': ['celebration', 'party'],
                'objects': ['cake', 'people', 'decorations'],
                'scenes': ['indoor', 'gathering']
            },
            'confidence_score': 0.8
        }

        print(f"📊 Test cluster: {len(cluster_data['files'])} files, location: {cluster_data['location_info']['city']}")

        # Test 1: Generate first name
        print(f"\n🔥 TEST 1: Generate first event name...")
        event_name_1 = namer.generate_event_name(cluster_data)
        print(f"✅ Generated name: {event_name_1}")

        # Check cache after first generation
        with open(test_cache_file, 'r') as f:
            cache_data_1 = json.load(f)
        print(f"💾 Cache entries after first generation: {len(cache_data_1)}")

        # Test 2: Generate with same data (should use cache)
        print(f"\n🔥 TEST 2: Generate same event name again (should use cache)...")
        event_name_2 = namer.generate_event_name(cluster_data)
        print(f"✅ Generated name: {event_name_2}")

        # Check cache after second generation
        with open(test_cache_file, 'r') as f:
            cache_data_2 = json.load(f)
        print(f"💾 Cache entries after second generation: {len(cache_data_2)}")

        # Test 3: Generate with different data
        cluster_data_different = cluster_data.copy()
        cluster_data_different['location_info'] = {
            'city': 'Calgary',
            'country': 'Canada',
            'coordinates': (51.0447, -114.0719)
        }

        print(f"\n🔥 TEST 3: Generate with different location (Calgary)...")
        event_name_3 = namer.generate_event_name(cluster_data_different)
        print(f"✅ Generated name: {event_name_3}")

        # Check final cache
        with open(test_cache_file, 'r') as f:
            cache_data_3 = json.load(f)
        print(f"💾 Final cache entries: {len(cache_data_3)}")

        # Display cache contents
        if cache_data_3:
            print(f"\n📋 Cache contents:")
            for i, (key, value) in enumerate(cache_data_3.items(), 1):
                print(f"   {i}. {key[:50]}... → {value}")

        # Results
        print(f"\n📊 TEST RESULTS:")
        print(f"   Event name 1: {event_name_1}")
        print(f"   Event name 2: {event_name_2}")
        print(f"   Event name 3: {event_name_3}")
        print(f"   Names 1&2 same (cache hit): {event_name_1 == event_name_2}")
        print(f"   Names 1&3 different (different data): {event_name_1 != event_name_3}")
        print(f"   Total cache entries: {len(cache_data_3)}")

        # Success criteria
        success = (
            event_name_1 and  # Generated a name
            event_name_1 == event_name_2 and  # Cache was used
            event_name_1 != event_name_3 and  # Different data gives different name
            len(cache_data_3) > 0  # Names were actually cached
        )

        if success:
            print(f"\n✅ ALL TESTS PASSED - EventNamer is working correctly!")
            # Pytest assertion - test passed
            assert True  # Explicit success assertion
        else:
            print(f"\n❌ TESTS FAILED - EventNamer has issues:")
            failure_reasons = []
            if not event_name_1:
                failure_reasons.append("Failed to generate event name")
                print(f"   - Failed to generate event name")
            if event_name_1 != event_name_2:
                failure_reasons.append("Cache not working (names should be same)")
                print(f"   - Cache not working (names should be same)")
            if event_name_1 == event_name_3:
                failure_reasons.append("Not generating different names for different data")
                print(f"   - Not generating different names for different data")
            if len(cache_data_3) == 0:
                failure_reasons.append("Names not being cached")
                print(f"   - Names not being cached")
            # Pytest assertion - test failed
            assert False, f"Event naming test failed: {'; '.join(failure_reasons)}"

    except Exception as e:
        print(f"💥 TEST EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        # Pytest assertion - test failed
        assert False, f"Event naming test failed with exception: {e}"

    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"🧹 Cleaned up test directory")

if __name__ == "__main__":
    success = test_event_namer_caching()
    exit(0 if success else 1)