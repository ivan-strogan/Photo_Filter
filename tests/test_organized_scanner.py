#!/usr/bin/env python3
"""Test organized photos scanner without ML dependencies."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.organized_photos_scanner import OrganizedPhotosScanner

def test_organized_scanner():
    """Test the organized photos scanner."""
    print("📚 Testing Organized Photos Scanner...")

    try:
        # Initialize scanner
        scanner = OrganizedPhotosScanner(use_gpu=False)

        # Test grouping functionality first (no ML required)
        print("\n🔍 Analyzing existing organization...")

        organized_files = scanner.media_detector.scan_pictures_library()
        photo_files = [f for f in organized_files if f.file_type == 'photo']

        print(f"Found {len(photo_files)} organized photos")

        # Group by event
        events = scanner._group_by_event_folder(photo_files)

        print(f"\n📁 Event Folders Found:")
        for event_name, photos in events.items():
            print(f"  {event_name}: {len(photos)} photos")

        # Test filtering
        filtered_events = scanner._filter_events_for_processing(
            events, max_photos_per_event=20, skip_large_events=True
        )

        print(f"\n🔧 After Filtering (max 20 photos per event):")
        for event_name, photos in filtered_events.items():
            original_count = len(events[event_name])
            filtered_count = len(photos)
            if filtered_count != original_count:
                print(f"  {event_name}: {filtered_count}/{original_count} photos (sampled)")
            else:
                print(f"  {event_name}: {filtered_count} photos")

        # Test skip logic
        test_folders = [
            "2022_10_14 - Mexico Trip",
            "Need To Filter",
            "Photos To Filter",
            "2022_10_24 - Daniel's First Birthday"
        ]

        print(f"\n🚫 Folder Skip Logic Test:")
        for folder in test_folders:
            should_skip = scanner._should_skip_folder(folder)
            status = "SKIP" if should_skip else "PROCESS"
            print(f"  {folder}: {status}")

        # Get database summary (without vectorization)
        print(f"\n📊 Current Database State:")
        try:
            summary = scanner.get_database_summary()
            if 'error' in summary:
                print(f"  Database not yet initialized: {summary['error']}")
            else:
                stats = summary['database_stats']
                print(f"  Total photos in DB: {stats['total_photos']}")
                print(f"  Organized photos: {stats['organized_photos']}")
                print(f"  Event folders: {stats['event_folders_count']}")
        except Exception as e:
            print(f"  Database check failed: {e}")

        print(f"\n✅ Organized scanner analysis completed successfully!")

        # Provide summary
        total_events = len(events)
        processable_events = len(filtered_events)
        total_photos = sum(len(photos) for photos in filtered_events.values())

        print(f"\n📋 Summary:")
        print(f"  Total events found: {total_events}")
        print(f"  Events ready for processing: {processable_events}")
        print(f"  Photos ready for vectorization: {total_photos}")

        # Cleanup
        scanner.cleanup()

    except Exception as e:
        print(f"❌ Error in organized scanner test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_organized_scanner()