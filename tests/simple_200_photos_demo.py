#!/usr/bin/env python3
"""
Simple 200 Photos Demo: Shows intelligent folder names without ML dependencies.

For junior developers:
This demo shows what folder names would look like for the first 200 photos
using our intelligent naming system, with simulated metadata and content analysis.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.media_detector import MediaDetector
from src.temporal_clustering import TemporalClusterer
from src.event_namer import EventNamer

def simple_200_photos_demo():
    print("📸 First 200 Photos - Intelligent Folder Preview")
    print("=" * 60)

    # Get first 200 photos
    detector = MediaDetector()
    all_files = detector.scan_iphone_automatic()
    photo_files = [f for f in all_files if f.file_type == 'photo']

    # Take first 200 photos
    first_200 = sorted(photo_files, key=lambda x: x.date)[:200]
    print(f"📊 Analyzing first 200 photos from {len(photo_files)} total photos")
    print(f"📅 Date range: {first_200[0].date.strftime('%Y-%m-%d')} to {first_200[-1].date.strftime('%Y-%m-%d')}")
    print()

    # Perform temporal clustering
    clusterer = TemporalClusterer(time_threshold_hours=6.0, min_cluster_size=2)

    print("🔄 Analyzing temporal patterns...")
    clusters = clusterer.cluster_by_time(first_200)

    # Filter and merge clusters
    merged_clusters = clusterer.merge_nearby_clusters(clusters, merge_threshold_hours=6.0)
    final_clusters = clusterer.filter_small_clusters(merged_clusters)

    print(f"📁 Generated {len(final_clusters)} clusters from first 200 photos")
    print()

    # Initialize intelligent naming
    event_namer = EventNamer(enable_llm=False)

    print("🧠 Intelligent Folder Names:")
    print()

    for i, cluster in enumerate(final_clusters, 1):
        # Analyze temporal context for intelligent naming
        start_time = cluster.start_time
        duration_hours = cluster.duration.total_seconds() / 3600

        # Generate context-aware content tags
        content_tags = []

        # Season detection
        month = start_time.month
        if month in [12, 1, 2]:
            content_tags.append('winter')
            if month == 12 and start_time.day >= 20:
                content_tags.append('holiday_season')
        elif month in [3, 4, 5]:
            content_tags.append('spring')
        elif month in [6, 7, 8]:
            content_tags.append('summer')
        else:
            content_tags.append('autumn')
            if month == 10 and start_time.day >= 25:
                content_tags.append('halloween')

        # Time of day context
        hour = start_time.hour
        if 6 <= hour < 12:
            content_tags.append('morning')
        elif 12 <= hour < 17:
            content_tags.append('afternoon')
        elif 17 <= hour < 21:
            content_tags.append('evening')

        # Weekend context
        if start_time.weekday() >= 5:
            content_tags.append('weekend')

        # Duration-based activity hints
        if duration_hours < 1:
            content_tags.append('quick')
        elif duration_hours > 8:
            content_tags.append('all_day')
        elif 2 <= duration_hours <= 4:
            content_tags.append('activity')

        # Special date detection
        if month == 12 and start_time.day == 25:
            content_tags.extend(['christmas', 'celebration'])
        elif month == 10 and start_time.day == 31:
            content_tags.extend(['halloween', 'celebration'])
        elif month == 1 and start_time.day == 1:
            content_tags.extend(['new_year', 'celebration'])

        # Mock GPS data for some photos (simulate realistic GPS availability)
        has_gps = (i % 3 == 0)  # About 1/3 of clusters have GPS
        gps_coordinates = [(53.5461, -113.4938)] if has_gps else []
        dominant_location = "Edmonton, Alberta, Canada" if has_gps else None

        # Prepare cluster data for intelligent naming
        cluster_data = {
            'start_time': cluster.start_time,
            'end_time': cluster.end_time,
            'duration_hours': duration_hours,
            'size': cluster.size,
            'photo_count': cluster.photo_count,
            'video_count': cluster.video_count,
            'location_info': None,
            'dominant_location': dominant_location,
            'gps_coordinates': gps_coordinates,
            'content_tags': content_tags,
            'people_detected': [],
            'confidence_score': 0.75 + (i % 3) * 0.1,  # Vary confidence 0.75-0.95
            'media_files': cluster.media_files
        }

        # Generate intelligent folder name
        try:
            folder_name = event_namer.generate_event_name(cluster_data)
        except Exception as e:
            # Simple fallback
            date_str = cluster.start_time.strftime("%Y_%m_%d")
            folder_name = f"{date_str} - Photos"

        # Display cluster details
        confidence_emoji = "🎯" if cluster_data['confidence_score'] > 0.8 else "📊"
        location_emoji = "📍" if has_gps else "📅"

        print(f"  {i:2d}. {confidence_emoji} 📂 {folder_name}")
        print(f"      └── {cluster.size} files ({cluster.photo_count} photos, {cluster.video_count} videos)")
        print(f"      └── {location_emoji} {cluster.start_time.strftime('%Y-%m-%d %H:%M')} → {cluster.end_time.strftime('%H:%M')} ({duration_hours:.1f}h)")

        if has_gps:
            print(f"      └── 📍 Location: Edmonton")
        if content_tags:
            relevant_tags = [tag for tag in content_tags if tag not in ['morning', 'afternoon', 'evening']][:3]
            if relevant_tags:
                print(f"      └── 🏷️  Context: {', '.join(relevant_tags)}")

        print()

        # Show first 20 clusters for detailed view
        if i >= 20:
            remaining_clusters = len(final_clusters) - 20
            if remaining_clusters > 0:
                print(f"... and {remaining_clusters} more clusters with intelligent names")
                print()
                # Show a few more examples without details
                for j in range(21, min(len(final_clusters) + 1, 26)):
                    cluster = final_clusters[j-1]
                    date_str = cluster.start_time.strftime("%Y_%m_%d")
                    duration_h = cluster.duration.total_seconds() / 3600

                    # Quick name generation
                    if cluster.start_time.month == 12 and cluster.start_time.day == 25:
                        name = f"{date_str} - Christmas Photos"
                    elif cluster.start_time.weekday() >= 5 and duration_h > 3:
                        name = f"{date_str} - Weekend Activity"
                    elif duration_h < 1:
                        name = f"{date_str} - Quick Photos"
                    else:
                        name = f"{date_str} - Event"

                    print(f"  {j:2d}. 📂 {name} ({cluster.size} files)")
            break

    # Summary
    total_files_in_clusters = sum(c.size for c in final_clusters)
    print(f"\n📊 Summary for First 200 Photos:")
    print(f"   📸 Photos analyzed: 200")
    print(f"   📁 Intelligent folders created: {len(final_clusters)}")
    print(f"   📋 Files organized: {total_files_in_clusters}")
    print(f"   📈 Organization rate: {total_files_in_clusters/200:.1%}")
    print(f"   📏 Average folder size: {total_files_in_clusters/len(final_clusters):.1f} files")

    # Date span analysis
    date_span = (final_clusters[-1].end_time - final_clusters[0].start_time).days
    print(f"   📅 Time span: {date_span} days")
    print()

    print("✨ Intelligent Naming Features Demonstrated:")
    print("   🎄 Holiday detection (Christmas, Halloween, New Year)")
    print("   🌸 Seasonal awareness (Winter, Spring, Summer, Autumn)")
    print("   ⏰ Time context (Morning, Afternoon, Evening)")
    print("   📅 Weekend vs weekday recognition")
    print("   ⏱️  Duration classification (Quick, Activity, All Day)")
    print("   📍 Location integration (when GPS available)")
    print("   🎯 Confidence-based quality scoring")
    print()

    print("🚀 This is just template-based naming!")
    print("   With LLM enabled: Even smarter names like 'Family Birthday Party - Edmonton'")
    print("   With content analysis: Activity-specific names like 'Mountain Hiking Trip'")
    print("   With face detection: People-based events like 'Sarah & Mike Wedding'")

if __name__ == "__main__":
    simple_200_photos_demo()