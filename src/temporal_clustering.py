"""
Temporal clustering algorithm for grouping photos and videos by time proximity.

This module implements the core logic for grouping photos taken around the same time
into "events". The idea is that photos taken within a few hours probably belong to
the same event (like a birthday party, vacation day, etc.).

For junior developers:
- Uses different algorithms: by_time, by_day, activity_periods
- Demonstrates algorithm selection based on data characteristics
- Shows how to work with datetime objects and timedeltas
- Implements the "cluster merging" pattern for refining results
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

from .media_detector import MediaFile

@dataclass
class TemporalCluster:
    """
    Represents a cluster of media files grouped by time.

    A temporal cluster is a group of photos/videos that were taken close together
    in time, suggesting they're from the same event or activity.

    For junior developers:
    - cluster_id: Unique identifier for this cluster
    - start_time/end_time: Time range covered by this cluster
    - duration: How long the event lasted (end_time - start_time)
    - media_files: All the photos/videos in this cluster
    - Properties provide convenient access to computed values
    """
    cluster_id: int                 # Unique ID for this cluster
    start_time: datetime           # When the first photo was taken
    end_time: datetime             # When the last photo was taken
    duration: timedelta            # How long the event lasted
    media_files: List[MediaFile]   # All photos/videos in this cluster

    @property
    def size(self) -> int:
        """
        Number of media files in the cluster.

        Using @property makes this look like an attribute but computed dynamically.
        This is useful because the size might change if we add/remove files.
        """
        return len(self.media_files)

    @property
    def photo_count(self) -> int:
        """
        Number of photos in the cluster.

        For junior developers:
        - sum(1 for f in ... if condition) is a Python idiom for counting
        - It's equivalent to: count = 0; for f in media_files: if f.file_type == 'photo': count += 1
        """
        return sum(1 for f in self.media_files if f.file_type == 'photo')

    @property
    def video_count(self) -> int:
        """Number of videos in the cluster."""
        return sum(1 for f in self.media_files if f.file_type == 'video')

class TemporalClusterer:
    """Groups media files into temporal clusters based on time proximity."""

    def __init__(self,
                 time_threshold_hours: float = 6.0,
                 min_cluster_size: int = 1,
                 max_gap_hours: float = 2.0):
        """Initialize the temporal clusterer.

        Args:
            time_threshold_hours: Maximum time gap to consider files in same event
            min_cluster_size: Minimum number of files to form a cluster
            max_gap_hours: Maximum gap between consecutive files in same cluster
        """
        self.time_threshold_hours = time_threshold_hours
        self.min_cluster_size = min_cluster_size
        self.max_gap_hours = max_gap_hours
        self.logger = logging.getLogger(__name__)

    def cluster_by_time(self, media_files: List[MediaFile]) -> List[TemporalCluster]:
        """Cluster media files by temporal proximity.

        Args:
            media_files: List of MediaFile objects

        Returns:
            List of TemporalCluster objects
        """
        if not media_files:
            return []

        # Sort files by timestamp
        sorted_files = sorted(media_files, key=lambda x: x.time)

        clusters = []
        current_cluster_files = [sorted_files[0]]
        cluster_id = 0

        for i in range(1, len(sorted_files)):
            current_file = sorted_files[i]
            previous_file = sorted_files[i - 1]

            # Calculate time gap
            time_gap = current_file.time - previous_file.time
            gap_hours = time_gap.total_seconds() / 3600

            # Check if file belongs to current cluster
            if gap_hours <= self.max_gap_hours:
                current_cluster_files.append(current_file)
            else:
                # Finalize current cluster if it meets minimum size
                if len(current_cluster_files) >= self.min_cluster_size:
                    cluster = self._create_cluster(cluster_id, current_cluster_files)
                    clusters.append(cluster)
                    cluster_id += 1

                # Start new cluster
                current_cluster_files = [current_file]

        # Handle the last cluster
        if len(current_cluster_files) >= self.min_cluster_size:
            cluster = self._create_cluster(cluster_id, current_cluster_files)
            clusters.append(cluster)

        self.logger.info(f"Created {len(clusters)} temporal clusters from {len(media_files)} files")
        return clusters

    def cluster_by_day(self, media_files: List[MediaFile]) -> List[TemporalCluster]:
        """Cluster media files by day.

        Args:
            media_files: List of MediaFile objects

        Returns:
            List of TemporalCluster objects (one per day)
        """
        if not media_files:
            return []

        # Group files by date
        date_groups = {}
        for media_file in media_files:
            date_key = media_file.date
            if date_key not in date_groups:
                date_groups[date_key] = []
            date_groups[date_key].append(media_file)

        clusters = []
        cluster_id = 0

        for date, files in sorted(date_groups.items()):
            if len(files) >= self.min_cluster_size:
                cluster = self._create_cluster(cluster_id, files)
                clusters.append(cluster)
                cluster_id += 1

        self.logger.info(f"Created {len(clusters)} daily clusters from {len(media_files)} files")
        return clusters

    def cluster_by_activity_periods(self, media_files: List[MediaFile]) -> List[TemporalCluster]:
        """Cluster media files by detecting activity periods.

        This method detects periods of high photo/video activity and groups them.

        Args:
            media_files: List of MediaFile objects

        Returns:
            List of TemporalCluster objects
        """
        if not media_files:
            return []

        # Sort files by timestamp
        sorted_files = sorted(media_files, key=lambda x: x.time)

        # Detect activity periods using sliding window approach
        clusters = []
        cluster_id = 0
        i = 0

        while i < len(sorted_files):
            # Start potential cluster
            cluster_start = i
            cluster_files = [sorted_files[i]]

            # Look ahead for activity period
            j = i + 1
            while j < len(sorted_files):
                time_gap = sorted_files[j].time - sorted_files[j-1].time
                gap_hours = time_gap.total_seconds() / 3600

                # If gap is small, add to cluster
                if gap_hours <= self.max_gap_hours:
                    cluster_files.append(sorted_files[j])
                    j += 1
                # If gap is moderate but we have a good cluster, try to extend
                elif gap_hours <= self.time_threshold_hours and len(cluster_files) >= self.min_cluster_size:
                    # Look ahead to see if there's more activity
                    if j + 1 < len(sorted_files):
                        next_gap = sorted_files[j+1].time - sorted_files[j].time
                        next_gap_hours = next_gap.total_seconds() / 3600
                        if next_gap_hours <= self.max_gap_hours:
                            cluster_files.append(sorted_files[j])
                            j += 1
                            continue
                    break
                else:
                    break

            # Create cluster if it meets criteria
            if len(cluster_files) >= self.min_cluster_size:
                cluster = self._create_cluster(cluster_id, cluster_files)
                clusters.append(cluster)
                cluster_id += 1
                i = j
            else:
                i += 1

        self.logger.info(f"Created {len(clusters)} activity-based clusters from {len(media_files)} files")
        return clusters

    def _create_cluster(self, cluster_id: int, files: List[MediaFile]) -> TemporalCluster:
        """Create a TemporalCluster from a list of files.

        Args:
            cluster_id: Unique identifier for the cluster
            files: List of MediaFile objects

        Returns:
            TemporalCluster object
        """
        sorted_files = sorted(files, key=lambda x: x.time)
        start_time = sorted_files[0].time
        end_time = sorted_files[-1].time
        duration = end_time - start_time

        return TemporalCluster(
            cluster_id=cluster_id,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            media_files=sorted_files
        )

    def merge_nearby_clusters(self,
                             clusters: List[TemporalCluster],
                             merge_threshold_hours: float = 3.0) -> List[TemporalCluster]:
        """Merge clusters that are close in time.

        Args:
            clusters: List of TemporalCluster objects
            merge_threshold_hours: Maximum time gap to merge clusters

        Returns:
            List of merged TemporalCluster objects
        """
        if len(clusters) <= 1:
            return clusters

        # Sort clusters by start time
        sorted_clusters = sorted(clusters, key=lambda x: x.start_time)
        merged_clusters = []
        current_cluster = sorted_clusters[0]

        for i in range(1, len(sorted_clusters)):
            next_cluster = sorted_clusters[i]

            # Calculate gap between clusters
            time_gap = next_cluster.start_time - current_cluster.end_time
            gap_hours = time_gap.total_seconds() / 3600

            if gap_hours <= merge_threshold_hours:
                # Merge clusters
                merged_files = current_cluster.media_files + next_cluster.media_files
                current_cluster = self._create_cluster(current_cluster.cluster_id, merged_files)
            else:
                # Add current cluster to results and start new one
                merged_clusters.append(current_cluster)
                current_cluster = next_cluster

        # Add the last cluster
        merged_clusters.append(current_cluster)

        self.logger.info(f"Merged {len(clusters)} clusters into {len(merged_clusters)} clusters")
        return merged_clusters

    def filter_small_clusters(self, clusters: List[TemporalCluster]) -> List[TemporalCluster]:
        """Remove clusters that are too small.

        Args:
            clusters: List of TemporalCluster objects

        Returns:
            List of filtered TemporalCluster objects
        """
        filtered = [c for c in clusters if c.size >= self.min_cluster_size]

        if len(filtered) != len(clusters):
            self.logger.info(f"Filtered out {len(clusters) - len(filtered)} small clusters")

        return filtered

    def get_clustering_stats(self, clusters: List[TemporalCluster]) -> Dict[str, Any]:
        """Get statistics about temporal clustering results.

        Args:
            clusters: List of TemporalCluster objects

        Returns:
            Dictionary with clustering statistics
        """
        if not clusters:
            return {
                'total_clusters': 0,
                'total_files': 0,
                'avg_cluster_size': 0,
                'avg_duration_hours': 0,
                'largest_cluster_size': 0,
                'longest_duration_hours': 0
            }

        total_files = sum(c.size for c in clusters)
        avg_cluster_size = total_files / len(clusters)

        durations_hours = [c.duration.total_seconds() / 3600 for c in clusters]
        avg_duration = sum(durations_hours) / len(durations_hours)

        largest_cluster = max(clusters, key=lambda x: x.size)
        longest_cluster = max(clusters, key=lambda x: x.duration)

        return {
            'total_clusters': len(clusters),
            'total_files': total_files,
            'avg_cluster_size': round(avg_cluster_size, 1),
            'avg_duration_hours': round(avg_duration, 2),
            'largest_cluster_size': largest_cluster.size,
            'longest_duration_hours': round(longest_cluster.duration.total_seconds() / 3600, 2),
            'cluster_sizes': [c.size for c in clusters],
            'duration_hours': durations_hours
        }

    def suggest_best_clustering_method(self, media_files: List[MediaFile]) -> str:
        """Suggest the best clustering method based on file distribution.

        Args:
            media_files: List of MediaFile objects

        Returns:
            Suggested clustering method name
        """
        if not media_files:
            return "none"

        # Analyze time distribution
        sorted_files = sorted(media_files, key=lambda x: x.time)

        if len(sorted_files) < self.min_cluster_size:
            return "none"

        # Calculate statistics
        total_span = sorted_files[-1].time - sorted_files[0].time
        span_days = total_span.days

        # Calculate average gap between consecutive photos
        gaps = []
        for i in range(1, len(sorted_files)):
            gap = sorted_files[i].time - sorted_files[i-1].time
            gaps.append(gap.total_seconds() / 3600)  # Convert to hours

        if gaps:
            avg_gap_hours = sum(gaps) / len(gaps)
            median_gap_hours = sorted(gaps)[len(gaps) // 2]

            # Decision logic
            if span_days <= 1:
                return "activity_periods"  # Single day, look for activity bursts
            elif avg_gap_hours > 24:
                return "by_day"  # Large gaps, cluster by day
            elif median_gap_hours <= self.max_gap_hours:
                return "activity_periods"  # Frequent photos, detect activity periods
            else:
                return "by_time"  # Default temporal clustering

        return "by_time"