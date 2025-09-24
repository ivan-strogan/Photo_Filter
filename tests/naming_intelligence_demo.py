#!/usr/bin/env python3
"""
Naming Intelligence Demo: Shows the improved event naming system.

For junior developers:
This demo shows how the EventNamer class generates intelligent folder names
by analyzing temporal patterns, location data, and event context.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
sys.path.append(str(Path(__file__).parent))

# Import only what we need for the naming demo
from src.event_namer import EventNamer

def create_sample_cluster_data():
    """Create sample cluster data to demonstrate naming intelligence."""

    # Sample cluster 1: Weekend photography session
    cluster1 = {
        'start_time': datetime(2024, 10, 26, 14, 30),  # Saturday afternoon
        'end_time': datetime(2024, 10, 26, 17, 45),    # Saturday evening
        'duration_hours': 3.25,
        'size': 15,
        'photo_count': 14,
        'video_count': 1,
        'location_info': None,  # Will be enhanced when geocoding works
        'dominant_location': 'Edmonton, Alberta, Canada',
        'gps_coordinates': [(53.5461, -113.4938), (53.5445, -113.4937)],  # Edmonton coordinates
        'content_tags': ['outdoor', 'nature', 'autumn'],
        'people_detected': [],
        'confidence_score': 0.85,
        'media_files': []  # Empty for demo
    }

    # Sample cluster 2: Quick indoor photos
    cluster2 = {
        'start_time': datetime(2024, 10, 31, 19, 15),  # Halloween evening
        'end_time': datetime(2024, 10, 31, 19, 45),    # 30 minutes
        'duration_hours': 0.5,
        'size': 8,
        'photo_count': 7,
        'video_count': 1,
        'location_info': None,
        'dominant_location': 'Edmonton, Alberta, Canada',
        'gps_coordinates': [(53.5461, -113.4938)],
        'content_tags': ['indoor', 'costume', 'celebration'],
        'people_detected': ['person1', 'person2'],
        'confidence_score': 0.92,
        'media_files': []
    }

    # Sample cluster 3: All-day event
    cluster3 = {
        'start_time': datetime(2024, 11, 15, 9, 00),   # Friday morning
        'end_time': datetime(2024, 11, 15, 22, 30),    # Friday night
        'duration_hours': 13.5,
        'size': 45,
        'photo_count': 38,
        'video_count': 7,
        'location_info': None,
        'dominant_location': 'Calgary, Alberta, Canada',
        'gps_coordinates': [(51.0447, -114.0719), (51.0456, -114.0712)],  # Calgary coordinates
        'content_tags': ['outdoor', 'urban', 'travel'],
        'people_detected': ['person1', 'person3'],
        'confidence_score': 0.78,
        'media_files': []
    }

    # Sample cluster 4: No location data
    cluster4 = {
        'start_time': datetime(2024, 12, 25, 16, 30),  # Christmas Day
        'end_time': datetime(2024, 12, 25, 18, 15),
        'duration_hours': 1.75,
        'size': 12,
        'photo_count': 10,
        'video_count': 2,
        'location_info': None,
        'dominant_location': None,
        'gps_coordinates': [],
        'content_tags': ['indoor', 'family', 'celebration'],
        'people_detected': ['person1', 'person2', 'person3'],
        'confidence_score': 0.88,
        'media_files': []
    }

    return [cluster1, cluster2, cluster3, cluster4]

def naming_intelligence_demo():
    print("🧠 Event Naming Intelligence Demo")
    print("=" * 50)
    print()

    print("This demo shows how our EventNamer generates intelligent folder names")
    print("by analyzing multiple signals from photo clusters.")
    print()

    # Initialize EventNamer
    event_namer = EventNamer(enable_llm=False)  # Start with template-based naming

    print("🎯 Naming Strategy:")
    print(f"   LLM Enabled: {event_namer.enable_llm}")
    print("   Using: Template-based intelligent naming")
    print()

    # Get sample cluster data
    sample_clusters = create_sample_cluster_data()

    print("📁 Generated Event Names:")
    print()

    for i, cluster_data in enumerate(sample_clusters, 1):
        try:
            # Generate intelligent name
            suggested_name = event_namer.generate_event_name(cluster_data)

            print(f"  {i}. 📂 {suggested_name}")
            print(f"     └── {cluster_data['size']} files ({cluster_data['photo_count']} photos, {cluster_data['video_count']} videos)")
            print(f"     └── Duration: {cluster_data['duration_hours']:.1f} hours")
            print(f"     └── Time: {cluster_data['start_time'].strftime('%Y-%m-%d %H:%M')} to {cluster_data['end_time'].strftime('%H:%M')}")

            # Show location if available
            if cluster_data['dominant_location']:
                clean_location = cluster_data['dominant_location'].replace(', Alberta, Canada', '').replace(', Canada', '')
                print(f"     └── Location: {clean_location}")

            # Show content tags if available
            if cluster_data['content_tags']:
                print(f"     └── Content: {', '.join(cluster_data['content_tags'])}")

            # Show people if detected
            if cluster_data['people_detected']:
                print(f"     └── People: {len(cluster_data['people_detected'])} detected")

            print(f"     └── Confidence: {cluster_data['confidence_score']:.2f}")
            print()

        except Exception as e:
            print(f"  {i}. ❌ Error generating name: {e}")
            print()

    print("🎉 Naming Intelligence Features:")
    print("   ✅ Date-based naming (YYYY_MM_DD format)")
    print("   ✅ Time-of-day analysis (morning, afternoon, evening)")
    print("   ✅ Duration-based hints (Quick, All Day, etc.)")
    print("   ✅ Location integration (city names)")
    print("   ✅ Holiday detection (Christmas, Halloween, etc.)")
    print("   ✅ Weekend vs weekday recognition")
    print("   ✅ Season detection (Spring, Summer, Fall, Winter)")
    print("   ✅ Activity pattern analysis")
    print()

    print("🚀 Future Enhancements:")
    print("   🔮 LLM-based naming for complex events")
    print("   🔮 Content analysis integration")
    print("   🔮 Face recognition for people-based events")
    print("   🔮 Activity detection from photo content")
    print("   🔮 Smart event type classification")

if __name__ == "__main__":
    naming_intelligence_demo()