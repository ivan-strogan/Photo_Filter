#!/usr/bin/env python3
"""
First 200 Photos Demo: Shows intelligent folder names for first 200 photos.

For junior developers:
This demo shows how the complete clustering and naming system works together
to generate meaningful folder names from real photo data.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.media_detector import MediaDetector
from src.temporal_clustering import TemporalClusterer
from src.metadata_extractor import MetadataExtractor
from src.event_namer import EventNamer

def analyze_first_200_photos():
    print("📸 First 200 Photos - Intelligent Folder Names")
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

    # Try different clustering approaches to see which works best
    print("🔄 Analyzing temporal patterns...")
    suggested_method = clusterer.suggest_best_clustering_method(first_200)
    print(f"📋 Recommended clustering method: {suggested_method}")

    if suggested_method == "by_day":
        clusters = clusterer.cluster_by_day(first_200)
    elif suggested_method == "activity_periods":
        clusters = clusterer.cluster_by_activity_periods(first_200)
    else:
        clusters = clusterer.cluster_by_time(first_200)

    # Filter and merge clusters
    merged_clusters = clusterer.merge_nearby_clusters(clusters, merge_threshold_hours=6.0)
    final_clusters = clusterer.filter_small_clusters(merged_clusters)

    print(f"📁 Generated {len(final_clusters)} clusters from first 200 photos")
    print()

    # Initialize intelligent naming
    event_namer = EventNamer(enable_llm=False)  # Use template-based naming
    metadata_extractor = MetadataExtractor()

    print("🧠 Intelligent Folder Names:")
    print()

    for i, cluster in enumerate(final_clusters, 1):
        # Extract metadata for location analysis
        gps_coordinates = []
        for media_file in cluster.media_files[:5]:  # Sample first 5 files for performance
            try:
                metadata = metadata_extractor.extract_photo_metadata(media_file)
                gps_coords = metadata.get('gps_coordinates')
                if gps_coords and len(gps_coords) == 2:
                    gps_coordinates.append(gps_coords)
            except Exception:
                pass  # Skip files with metadata issues

        # Simulate content analysis based on season/time
        content_tags = []
        start_time = cluster.start_time

        # Season detection
        month = start_time.month
        if month in [12, 1, 2]:
            content_tags.append('winter')
        elif month in [3, 4, 5]:
            content_tags.append('spring')
        elif month in [6, 7, 8]:
            content_tags.append('summer')
        else:
            content_tags.append('autumn')

        # Time of day inference
        hour = start_time.hour
        if 6 <= hour < 12:
            content_tags.append('morning')
        elif 12 <= hour < 17:
            content_tags.append('afternoon')
        elif 17 <= hour < 21:
            content_tags.append('evening')
        else:
            content_tags.append('night')

        # Weekend vs weekday
        if start_time.weekday() >= 5:  # Saturday or Sunday
            content_tags.append('weekend')

        # Duration-based activity inference
        duration_hours = cluster.duration.total_seconds() / 3600
        if duration_hours < 1:
            content_tags.append('quick')
        elif duration_hours > 8:
            content_tags.append('all_day')

        # Mock location (simplified - would use reverse geocoding in real system)
        dominant_location = None
        if gps_coordinates:
            # For demo, assume Edmonton area if GPS available
            dominant_location = "Edmonton, Alberta, Canada"

        # Prepare cluster data for naming
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
            'people_detected': [],  # Would be populated by face detection
            'confidence_score': 0.8,  # Mock confidence
            'media_files': cluster.media_files
        }

        # Generate intelligent name
        try:
            folder_name = event_namer.generate_event_name(cluster_data)
        except Exception as e:
            # Fallback naming
            date_str = cluster.start_time.strftime("%Y_%m_%d")
            if duration_hours < 1:
                folder_name = f"{date_str} - Quick Photos"
            elif duration_hours > 8:
                folder_name = f"{date_str} - All Day Event"
            else:
                folder_name = f"{date_str} - Event"

        # Display cluster info
        print(f"  {i:2d}. 📂 {folder_name}")
        print(f"      └── {cluster.size} files ({cluster.photo_count} photos, {cluster.video_count} videos)")
        print(f"      └── {cluster.start_time.strftime('%Y-%m-%d %H:%M')} to {cluster.end_time.strftime('%H:%M')}")
        print(f"      └── Duration: {duration_hours:.1f} hours")

        if gps_coordinates:
            print(f"      └── Location: {len(gps_coordinates)} GPS points")
        if content_tags:
            print(f"      └── Context: {', '.join(content_tags[:3])}")  # Show first 3 tags

        print()

        # Stop at 15 clusters for readability
        if i >= 15:
            remaining_clusters = len(final_clusters) - 15
            if remaining_clusters > 0:
                print(f"... and {remaining_clusters} more clusters")
            break

    # Summary statistics
    total_files_in_clusters = sum(c.size for c in final_clusters)
    print("\n📊 Clustering Summary:")
    print(f"   Photos processed: 200")
    print(f"   Files in clusters: {total_files_in_clusters}")
    print(f"   Unclustered files: {200 - total_files_in_clusters}")
    print(f"   Total clusters: {len(final_clusters)}")
    print(f"   Average cluster size: {total_files_in_clusters / len(final_clusters):.1f} files")

    # Show date coverage
    date_range_days = (final_clusters[-1].end_time - final_clusters[0].start_time).days
    print(f"   Date span: {date_range_days} days")
    print()

    print("💡 Naming Intelligence Features in Action:")
    print("   ✅ Holiday detection (Christmas, Halloween, etc.)")
    print("   ✅ Season awareness (Winter, Spring, Summer, Autumn)")
    print("   ✅ Time of day context (Morning, Afternoon, Evening)")
    print("   ✅ Weekend vs weekday recognition")
    print("   ✅ Duration-based classification (Quick, All Day)")
    print("   ✅ GPS location integration (when available)")
    print("   ✅ Activity pattern detection")

if __name__ == "__main__":
    analyze_first_200_photos()