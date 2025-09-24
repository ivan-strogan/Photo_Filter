#!/usr/bin/env python3
"""
Test the Folder Organizer system.

For junior developers:
This script tests the folder creation system by creating sample clusters
and demonstrating how the folder organizer would create intelligent folders.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.media_detector import MediaDetector
from src.temporal_clustering import TemporalClusterer
from src.folder_organizer import FolderOrganizer
from src.event_namer import EventNamer

def test_folder_organizer():
    print("📁 Testing Folder Organizer System")
    print("=" * 50)

    # Get sample photos
    detector = MediaDetector()
    all_files = detector.scan_iphone_automatic()
    photo_files = [f for f in all_files if f.file_type == 'photo']

    # Use first 30 photos for testing
    test_photos = sorted(photo_files, key=lambda x: x.date)[:30]
    print(f"📸 Testing with {len(test_photos)} photos")

    # Create temporal clusters
    clusterer = TemporalClusterer(time_threshold_hours=6.0, min_cluster_size=2)
    clusters = clusterer.cluster_by_time(test_photos)

    print(f"🎯 Created {len(clusters)} temporal clusters")

    # Add intelligent names to clusters
    event_namer = EventNamer(enable_llm=False)

    enhanced_clusters = []
    for i, cluster in enumerate(clusters):
        # Create MediaCluster-like object for testing
        class TestCluster:
            def __init__(self, temporal_cluster, cluster_id):
                self.cluster_id = cluster_id
                self.temporal_info = temporal_cluster
                self.media_files = temporal_cluster.media_files
                self.size = temporal_cluster.size
                self.photo_count = temporal_cluster.photo_count
                self.video_count = temporal_cluster.video_count
                self.duration_hours = temporal_cluster.duration.total_seconds() / 3600
                self.has_location = False
                self.gps_coordinates = []
                self.dominant_location = None
                self.location_info = None
                self.content_tags = []
                self.people_detected = []
                self.confidence_score = 0.8

                # Generate intelligent name
                cluster_data = {
                    'start_time': temporal_cluster.start_time,
                    'end_time': temporal_cluster.end_time,
                    'duration_hours': self.duration_hours,
                    'size': self.size,
                    'photo_count': self.photo_count,
                    'video_count': self.video_count,
                    'location_info': None,
                    'dominant_location': None,
                    'gps_coordinates': [],
                    'content_tags': self._generate_content_tags(),
                    'people_detected': [],
                    'confidence_score': self.confidence_score,
                    'media_files': temporal_cluster.media_files
                }

                try:
                    self.suggested_name = event_namer.generate_event_name(cluster_data)
                except Exception:
                    # Fallback naming
                    date_str = temporal_cluster.start_time.strftime("%Y_%m_%d")
                    if self.duration_hours < 1:
                        self.suggested_name = f"{date_str} - Quick Photos"
                    else:
                        self.suggested_name = f"{date_str} - Event"

            def _generate_content_tags(self):
                """Generate context-aware content tags for testing."""
                tags = []
                start_time = self.temporal_info.start_time

                # Season
                month = start_time.month
                if month in [12, 1, 2]:
                    tags.append('winter')
                elif month in [3, 4, 5]:
                    tags.append('spring')
                elif month in [6, 7, 8]:
                    tags.append('summer')
                else:
                    tags.append('autumn')

                # Weekend
                if start_time.weekday() >= 5:
                    tags.append('weekend')

                # Duration
                if self.duration_hours < 1:
                    tags.append('quick')
                elif self.duration_hours > 6:
                    tags.append('all_day')

                return tags

        enhanced_cluster = TestCluster(cluster, i)
        enhanced_clusters.append(enhanced_cluster)

    print("🧠 Generated intelligent names for clusters")

    # Test folder organizer in dry-run mode
    print("\n📁 Testing Folder Organizer (Dry Run Mode)")
    folder_organizer = FolderOrganizer(dry_run=True)

    # Show preview
    preview = folder_organizer.preview_folder_structure(enhanced_clusters)
    print("\n" + preview)

    # Test the actual folder creation logic
    print("\n🔧 Testing Folder Creation Logic")
    result = folder_organizer.create_folder_structure(enhanced_clusters)

    print(f"\n📊 Folder Creation Results:")
    print(f"   Folders that would be created: {result['operation_summary']['folders_created']}")
    print(f"   Conflicts resolved: {result['operation_summary']['conflicts_resolved']}")
    print(f"   Errors: {result['operation_summary']['errors']}")
    print(f"   Total clusters processed: {result['operation_summary']['total_clusters_processed']}")

    # Show year organization
    if result['year_organization']:
        print(f"\n📅 Year Organization:")
        for year, folders in result['year_organization'].items():
            print(f"   {year}: {len(folders)} folders")

    # Show folder mapping sample
    if result['folder_mapping']:
        print(f"\n📂 Sample Folder Mappings:")
        for cluster_id, folder_path in list(result['folder_mapping'].items())[:5]:
            print(f"   Cluster {cluster_id} → {folder_path}")

    print(f"\n✅ Folder Organizer test completed successfully!")
    print(f"💡 This was a dry run - no actual folders were created.")
    print(f"📁 Folders would be created in: {result['organized_directory']}")

if __name__ == "__main__":
    test_folder_organizer()