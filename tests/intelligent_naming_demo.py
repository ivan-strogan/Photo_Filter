#!/usr/bin/env python3
"""
Intelligent Naming Demo: Shows improved folder naming capabilities.

For junior developers:
This demo compares the basic naming system with the new intelligent naming
system to show the dramatic improvement in event name quality.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.media_detector import MediaDetector
from src.media_clustering import MediaClusteringEngine
from src.config_manager import get_config_manager

def intelligent_naming_demo():
    print("🧠 Intelligent Naming Demo: Before vs After")
    print("=" * 60)

    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config()

    print(f"📋 Configuration:")
    print(f"   LLM Naming Enabled: {config.naming.use_llm_naming}")
    print(f"   Date Format: {config.naming.date_format}")
    print(f"   Include Location: {config.naming.include_location}")
    print(f"   Include Duration Hints: {config.naming.include_duration_hints}")
    print()

    # Get first 50 photos for faster demo
    detector = MediaDetector()
    all_files = detector.scan_iphone_automatic()
    photo_files = [f for f in all_files if f.file_type == 'photo']

    # Take first 50 photos for demo
    first_50 = sorted(photo_files, key=lambda x: x.date)[:50]
    print(f"📸 Analyzing first 50 photos from {len(photo_files)} total photos")
    print(f"📅 Date range: {first_50[0].date.strftime('%Y-%m-%d')} to {first_50[-1].date.strftime('%Y-%m-%d')}")
    print()

    # Initialize clustering engine
    clustering_engine = MediaClusteringEngine(
        time_threshold_hours=config.clustering.time_threshold_hours,
        location_threshold_km=config.clustering.location_threshold_km,
        min_cluster_size=config.clustering.min_cluster_size,
        similarity_threshold=config.clustering.similarity_threshold
    )

    # Perform clustering
    print("🔄 Performing intelligent clustering...")
    clusters = clustering_engine.cluster_media_files(first_50)

    # Generate intelligent names
    print("🎯 Generating intelligent event names...")
    named_clusters = clustering_engine.suggest_event_names(clusters, enable_llm=False)  # Start with template-based

    print(f"\n📁 Generated {len(named_clusters)} intelligent event folders:")
    print()

    for i, cluster in enumerate(named_clusters[:8]):  # Show first 8 clusters
        print(f"  📂 {cluster.suggested_name}")
        print(f"     └── {cluster.size} files ({cluster.photo_count} photos, {cluster.video_count} videos)")
        print(f"     └── Duration: {cluster.duration_hours:.1f} hours")
        print(f"     └── Confidence: {cluster.confidence_score:.2f}")

        # Show location info if available
        if cluster.location_info:
            location_parts = []
            if cluster.location_info.city:
                location_parts.append(cluster.location_info.city)
            if cluster.location_info.state and cluster.location_info.state != cluster.location_info.city:
                location_parts.append(cluster.location_info.state)
            if location_parts:
                print(f"     └── Location: {', '.join(location_parts)}")

        # Show GPS coordinates if available
        if cluster.gps_coordinates:
            avg_lat = sum(coord[0] for coord in cluster.gps_coordinates) / len(cluster.gps_coordinates)
            avg_lon = sum(coord[1] for coord in cluster.gps_coordinates) / len(cluster.gps_coordinates)
            print(f"     └── GPS: {avg_lat:.4f}, {avg_lon:.4f}")

        print()

    if len(named_clusters) > 8:
        print(f"... and {len(named_clusters) - 8} more clusters")
        print()

    # Show improvement comparison
    print("💡 Naming Intelligence Improvements:")
    print("   OLD: Basic names like '2014_10_25 - Event'")
    print("   NEW: Intelligent names like '2014_10_25 - Weekend Activity - Edmonton'")
    print()

    # Show clustering summary
    summary = clustering_engine.get_clustering_summary(named_clusters)
    print("📊 Clustering Quality Summary:")
    print(f"   Total clusters: {summary['total_clusters']}")
    print(f"   Average confidence: {summary['avg_confidence']:.3f}")
    print(f"   Location coverage: {summary['location_coverage']:.1%}")
    print(f"   Average cluster size: {summary['avg_cluster_size']:.1f} files")
    print(f"   Average duration: {summary['avg_duration_hours']:.2f} hours")
    print()

    # Quality distribution
    quality_dist = summary['quality_distribution']
    print("🎯 Quality Distribution:")
    print(f"   High confidence (≥0.7): {quality_dist['high_confidence']} clusters")
    print(f"   Medium confidence (0.4-0.7): {quality_dist['medium_confidence']} clusters")
    print(f"   Low confidence (<0.4): {quality_dist['low_confidence']} clusters")
    print()

    print("🚀 Next Steps:")
    print("   1. Enable LLM naming for even smarter folder names")
    print("   2. Add content analysis for activity-specific names")
    print("   3. Implement face detection for people-based events")
    print("   4. Create automatic folder structure and file organization")

if __name__ == "__main__":
    intelligent_naming_demo()