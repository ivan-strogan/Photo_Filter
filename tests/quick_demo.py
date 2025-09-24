#!/usr/bin/env python3
"""Quick demo to show folder naming for first batch of photos."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.media_detector import MediaDetector
from src.temporal_clustering import TemporalClusterer

def quick_demo():
    print("🔍 Quick Demo: Folder Names for First 100 Photos")
    print("=" * 60)

    # Get first 100 photos
    detector = MediaDetector()
    all_files = detector.scan_iphone_automatic()
    photo_files = [f for f in all_files if f.file_type == 'photo']

    # Take first 100 photos
    first_100 = sorted(photo_files, key=lambda x: x.date)[:100]
    print(f"Analyzing first 100 photos from {len(photo_files)} total photos")
    print(f"Date range: {first_100[0].date} to {first_100[-1].date}")
    print()

    # Quick temporal clustering
    clusterer = TemporalClusterer(time_threshold_hours=6.0, min_cluster_size=2)
    clusters = clusterer.cluster_by_time(first_100)

    print(f"📅 Generated {len(clusters)} clusters from first 100 photos:")
    print()

    for i, cluster in enumerate(clusters[:10]):  # Show first 10 clusters
        start_date = cluster.start_time.strftime("%Y_%m_%d")

        # Determine event type based on duration and size
        if cluster.duration.total_seconds() < 3600:  # Less than 1 hour
            event_type = "Quick Event"
        elif cluster.duration.total_seconds() > 86400:  # More than 1 day
            event_type = "Multi-Day Event"
        elif cluster.duration.total_seconds() > 8 * 3600:  # More than 8 hours
            event_type = "All Day"
        else:
            event_type = "Event"

        # Create folder name
        if event_type == "Quick Event" and cluster.size <= 3:
            folder_name = f"{start_date} - Quick Photos"
        elif event_type == "All Day":
            folder_name = f"{start_date} - All Day Event"
        else:
            folder_name = f"{start_date} - {event_type}"

        print(f"  📁 {folder_name}")
        print(f"     └── {cluster.size} files, {cluster.duration}")
        print(f"     └── Photos: {cluster.photo_count}, Videos: {cluster.video_count}")
        print()

    if len(clusters) > 10:
        print(f"... and {len(clusters) - 10} more clusters")

    print("\n💡 These are basic folder names. With full ML analysis, they would be more descriptive!")
    print("    Example with content analysis: '2014_10_25 - Birthday Party - Edmonton'")

if __name__ == "__main__":
    quick_demo()