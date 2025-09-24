"""Comprehensive media clustering combining location, time, content, and similarity data."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import numpy as np

from .media_detector import MediaFile
from .temporal_clustering import TemporalClusterer, TemporalCluster
from .geocoding import LocationGeocoder, LocationInfo
from .metadata_extractor import MetadataExtractor
from .event_namer import EventNamer

@dataclass
class MediaCluster:
    """Comprehensive media cluster with multiple signals."""
    cluster_id: int
    media_files: List[MediaFile]
    temporal_info: TemporalCluster
    location_info: Optional[LocationInfo] = None
    dominant_location: Optional[str] = None
    gps_coordinates: List[Tuple[float, float]] = field(default_factory=list)
    content_tags: List[str] = field(default_factory=list)
    people_detected: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    suggested_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """Number of media files in cluster."""
        return len(self.media_files)

    @property
    def photo_count(self) -> int:
        """Number of photos in cluster."""
        return sum(1 for f in self.media_files if f.file_type == 'photo')

    @property
    def video_count(self) -> int:
        """Number of videos in cluster."""
        return sum(1 for f in self.media_files if f.file_type == 'video')

    @property
    def duration_hours(self) -> float:
        """Duration of cluster in hours."""
        return self.temporal_info.duration.total_seconds() / 3600

    @property
    def has_location(self) -> bool:
        """Whether cluster has location data."""
        return bool(self.gps_coordinates)

class MediaClusteringEngine:
    """Advanced clustering engine combining multiple signals."""

    def __init__(self,
                 time_threshold_hours: float = 6.0,
                 location_threshold_km: float = 1.0,
                 min_cluster_size: int = 1,
                 similarity_threshold: float = 0.7,
                 vector_db: Optional[Any] = None,
                 photo_vectorizer: Optional[Any] = None,
                 face_recognizer: Optional[Any] = None,
                 people_database: Optional[Any] = None):
        """Initialize the clustering engine.

        Args:
            time_threshold_hours: Time threshold for temporal clustering
            location_threshold_km: Distance threshold for location clustering
            min_cluster_size: Minimum files for a valid cluster
            similarity_threshold: Threshold for content similarity
            vector_db: Vector database instance for finding similar organized photos
            photo_vectorizer: Photo vectorizer for creating embeddings
            face_recognizer: Face recognizer instance for people detection
            people_database: People database for person identification
        """
        self.time_threshold_hours = time_threshold_hours
        self.location_threshold_km = location_threshold_km
        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold

        # Face recognition components
        self.face_recognizer = face_recognizer
        self.people_database = people_database

        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.temporal_clusterer = TemporalClusterer(
            time_threshold_hours=time_threshold_hours,
            min_cluster_size=min_cluster_size
        )
        self.geocoder = LocationGeocoder()
        self.metadata_extractor = MetadataExtractor()
        # Initialize EventNamer with configuration and vector components
        try:
            from .config_manager import get_config
            config = get_config()
            self.event_namer = EventNamer(
                enable_llm=config.naming.use_llm_naming,
                vector_db=vector_db,
                photo_vectorizer=photo_vectorizer
            )
        except Exception:
            # Fallback if config system not available
            self.event_namer = EventNamer(
                enable_llm=False,
                vector_db=vector_db,
                photo_vectorizer=photo_vectorizer
            )

    def cluster_media_files(self, media_files: List[MediaFile]) -> List[MediaCluster]:
        """Perform comprehensive clustering of media files.

        Args:
            media_files: List of MediaFile objects

        Returns:
            List of MediaCluster objects
        """
        if not media_files:
            return []

        self.logger.info(f"Starting comprehensive clustering of {len(media_files)} files")

        # Step 1: Temporal clustering (primary signal)
        temporal_clusters = self._perform_temporal_clustering(media_files)

        # Step 2: Enhance with location data
        enhanced_clusters = self._enhance_with_location_data(temporal_clusters)

        # Step 3: Refine with location-based splitting
        location_refined_clusters = self._refine_with_location_clustering(enhanced_clusters)

        # Step 4: Enhance with people-based clustering
        people_enhanced_clusters = self._enhance_with_people_data(location_refined_clusters)

        # Step 5: Calculate confidence scores
        final_clusters = self._calculate_confidence_scores(people_enhanced_clusters)

        self.logger.info(f"Created {len(final_clusters)} comprehensive clusters")
        return final_clusters

    def _perform_temporal_clustering(self, media_files: List[MediaFile]) -> List[TemporalCluster]:
        """Perform temporal clustering as the primary signal."""
        self.logger.info("Performing temporal clustering...")

        # Suggest best clustering method based on data distribution
        suggested_method = self.temporal_clusterer.suggest_best_clustering_method(media_files)
        self.logger.info(f"Using temporal clustering method: {suggested_method}")

        # Apply the suggested clustering method
        if suggested_method == "by_day":
            clusters = self.temporal_clusterer.cluster_by_day(media_files)
        elif suggested_method == "activity_periods":
            clusters = self.temporal_clusterer.cluster_by_activity_periods(media_files)
        else:  # "by_time" or fallback
            clusters = self.temporal_clusterer.cluster_by_time(media_files)

        # Merge nearby clusters if beneficial
        merged_clusters = self.temporal_clusterer.merge_nearby_clusters(
            clusters, merge_threshold_hours=self.time_threshold_hours
        )

        # Filter out small clusters
        print(f"DEBUG TEMPORAL: Before filter - {len(merged_clusters)} clusters, {sum(c.size for c in merged_clusters)} total files")
        filtered_clusters = self.temporal_clusterer.filter_small_clusters(merged_clusters)
        print(f"DEBUG TEMPORAL: After filter - {len(filtered_clusters)} clusters, {sum(c.size for c in filtered_clusters)} total files")

        if len(merged_clusters) != len(filtered_clusters):
            removed_clusters = len(merged_clusters) - len(filtered_clusters)
            print(f"DEBUG TEMPORAL: Removed {removed_clusters} clusters")

        return filtered_clusters

    def _enhance_with_location_data(self, temporal_clusters: List[TemporalCluster]) -> List[MediaCluster]:
        """Enhance temporal clusters with location information."""
        self.logger.info("Enhancing clusters with location data...")

        print(f"DEBUG LOCATION: Input temporal clusters: {len(temporal_clusters)}, files: {sum(c.size for c in temporal_clusters)}")
        media_clusters = []

        for i, temporal_cluster in enumerate(temporal_clusters):
            # Extract GPS coordinates from files
            gps_coordinates = []
            locations = []

            for media_file in temporal_cluster.media_files:
                metadata = self.metadata_extractor.extract_photo_metadata(media_file)
                gps_coords = metadata.get('gps_coordinates')

                if gps_coords and len(gps_coords) == 2:
                    gps_coordinates.append(gps_coords)

                    # Perform reverse geocoding (with caching)
                    location_info = self.geocoder.reverse_geocode(gps_coords[0], gps_coords[1])
                    if location_info:
                        locations.append(location_info)

            # Determine dominant location
            dominant_location = None
            representative_location = None

            if locations:
                dominant_location = self.geocoder.find_most_common_location(locations)
                representative_location = locations[0]  # Use first location as representative

            # Create MediaCluster
            media_cluster = MediaCluster(
                cluster_id=i,
                media_files=temporal_cluster.media_files,
                temporal_info=temporal_cluster,
                location_info=representative_location,
                dominant_location=dominant_location,
                gps_coordinates=gps_coordinates
            )

            media_clusters.append(media_cluster)

        print(f"DEBUG LOCATION: Output media clusters: {len(media_clusters)}, files: {sum(len(c.media_files) for c in media_clusters)}")
        return media_clusters

    def _refine_with_location_clustering(self, clusters: List[MediaCluster]) -> List[MediaCluster]:
        """Refine clusters by splitting those with diverse locations."""
        self.logger.info("Refining clusters with location-based splitting...")

        print(f"DEBUG REFINE: Input clusters: {len(clusters)}, files: {sum(len(c.media_files) for c in clusters)}")
        refined_clusters = []
        cluster_id_counter = 0

        for cluster in clusters:
            if not cluster.gps_coordinates or len(cluster.gps_coordinates) < 2:
                # No location data or single location - keep as is
                cluster.cluster_id = cluster_id_counter
                refined_clusters.append(cluster)
                cluster_id_counter += 1
                continue

            # Check if locations are diverse enough to split
            location_clusters = self.geocoder.cluster_locations_by_proximity(
                cluster.gps_coordinates,
                threshold_km=self.location_threshold_km
            )

            if len(location_clusters) <= 1:
                # All locations are close - keep as single cluster
                cluster.cluster_id = cluster_id_counter
                refined_clusters.append(cluster)
                cluster_id_counter += 1
            else:
                # Split into multiple location-based clusters
                self.logger.info(f"Splitting cluster into {len(location_clusters)} location-based clusters")

                # First, collect all files that have GPS coordinates
                files_with_gps = []
                files_without_gps = []

                for media_file in cluster.media_files:
                    metadata = self.metadata_extractor.extract_photo_metadata(media_file)
                    file_gps = metadata.get('gps_coordinates')
                    if file_gps and len(file_gps) == 2:
                        files_with_gps.append(media_file)
                    else:
                        files_without_gps.append(media_file)

                # Create location-based clusters for files with GPS
                for loc_cluster_indices in location_clusters:
                    # Create new cluster for this location group
                    location_files = []
                    location_coords = []

                    for idx in loc_cluster_indices:
                        # Find corresponding media file
                        gps_coord = cluster.gps_coordinates[idx]

                        # Find media files with this GPS coordinate
                        for media_file in files_with_gps:
                            metadata = self.metadata_extractor.extract_photo_metadata(media_file)
                            file_gps = metadata.get('gps_coordinates')

                            if (file_gps and len(file_gps) == 2 and
                                abs(file_gps[0] - gps_coord[0]) < 0.001 and
                                abs(file_gps[1] - gps_coord[1]) < 0.001):
                                location_files.append(media_file)
                                location_coords.append(gps_coord)
                                break

                    # Always create cluster regardless of size (min_cluster_size = 1)
                    if location_files:
                        # Create temporal cluster for these files
                        sub_temporal_cluster = self.temporal_clusterer._create_cluster(
                            cluster_id_counter, location_files
                        )

                        # Determine location for this sub-cluster
                        if location_coords:
                            representative_gps = location_coords[0]
                            location_info = self.geocoder.reverse_geocode(
                                representative_gps[0], representative_gps[1]
                            )
                        else:
                            location_info = None

                        new_cluster = MediaCluster(
                            cluster_id=cluster_id_counter,
                            media_files=location_files,
                            temporal_info=sub_temporal_cluster,
                            location_info=location_info,
                            gps_coordinates=location_coords
                        )

                        refined_clusters.append(new_cluster)
                        cluster_id_counter += 1

                # Create a separate cluster for files without GPS coordinates
                if files_without_gps:
                    # Create temporal cluster for files without GPS
                    sub_temporal_cluster = self.temporal_clusterer._create_cluster(
                        cluster_id_counter, files_without_gps
                    )

                    no_gps_cluster = MediaCluster(
                        cluster_id=cluster_id_counter,
                        media_files=files_without_gps,
                        temporal_info=sub_temporal_cluster,
                        location_info=None,
                        gps_coordinates=[]
                    )

                    refined_clusters.append(no_gps_cluster)
                    cluster_id_counter += 1

        print(f"DEBUG REFINE: Output clusters: {len(refined_clusters)}, files: {sum(len(c.media_files) for c in refined_clusters)}")
        return refined_clusters

    def _enhance_with_people_data(self, clusters: List[MediaCluster]) -> List[MediaCluster]:
        """Enhance clusters with people detection and grouping.

        Args:
            clusters: List of media clusters to enhance

        Returns:
            Enhanced clusters with people information
        """
        if not self.face_recognizer or not self.face_recognizer.enabled:
            self.logger.debug("Face recognition not available, skipping people enhancement")
            return clusters

        self.logger.info("Enhancing clusters with people data...")
        print(f"DEBUG PEOPLE: Input clusters: {len(clusters)}, files: {sum(len(c.media_files) for c in clusters)}")

        enhanced_clusters = []

        for cluster in clusters:
            try:
                # Analyze faces in all photos in this cluster
                cluster_people = set()
                total_faces = 0
                people_consistency_score = 0.0

                # Process each photo for face detection
                photos_with_people = []
                for media_file in cluster.media_files:
                    if media_file.file_type == 'photo':
                        try:
                            result = self.face_recognizer.detect_faces(media_file.path)
                            if not result.error:
                                people_detected = result.get_people_detected()
                                face_count = result.faces_detected

                                if people_detected:
                                    cluster_people.update(people_detected)
                                    photos_with_people.append((media_file, people_detected))

                                total_faces += face_count

                        except Exception as e:
                            self.logger.warning(f"Face detection failed for {media_file.filename}: {e}")

                # Calculate people consistency (how many photos have the same people)
                if photos_with_people and len(cluster_people) > 0:
                    # Count how often each person appears
                    person_counts = {}
                    for _, people in photos_with_people:
                        for person in people:
                            person_counts[person] = person_counts.get(person, 0) + 1

                    # Calculate consistency as percentage of photos containing main people
                    main_people = [person for person, count in person_counts.items()
                                 if count >= len(photos_with_people) * 0.3]  # Person in at least 30% of photos

                    if main_people:
                        people_consistency_score = len([p for _, people in photos_with_people
                                                      if any(person in people for person in main_people)]) / len(photos_with_people)

                # Update cluster with people information
                enhanced_cluster = MediaCluster(
                    cluster_id=cluster.cluster_id,
                    media_files=cluster.media_files,
                    primary_location=cluster.primary_location,
                    date_range=cluster.date_range,
                    suggested_name=cluster.suggested_name,
                    confidence_score=cluster.confidence_score,
                    people_detected=list(cluster_people),
                    metadata=cluster.metadata.copy() if cluster.metadata else {}
                )

                # Add people-related metadata
                enhanced_cluster.metadata.update({
                    'total_faces_detected': total_faces,
                    'unique_people_count': len(cluster_people),
                    'people_consistency_score': people_consistency_score,
                    'photos_with_people': len(photos_with_people),
                    'main_people': person_counts if photos_with_people else {}
                })

                enhanced_clusters.append(enhanced_cluster)

                if cluster_people:
                    self.logger.debug(f"Cluster {cluster.cluster_id}: Found {len(cluster_people)} people, "
                                    f"{total_faces} faces, consistency: {people_consistency_score:.2f}")

            except Exception as e:
                self.logger.error(f"Error enhancing cluster {cluster.cluster_id} with people data: {e}")
                enhanced_clusters.append(cluster)  # Keep original cluster on error

        # Optionally split or merge clusters based on people consistency
        final_clusters = self._refine_clusters_by_people(enhanced_clusters)

        print(f"DEBUG PEOPLE: Output clusters: {len(final_clusters)}, files: {sum(len(c.media_files) for c in final_clusters)}")
        return final_clusters

    def _refine_clusters_by_people(self, clusters: List[MediaCluster]) -> List[MediaCluster]:
        """Refine clusters by splitting/merging based on people consistency.

        Args:
            clusters: Clusters with people data

        Returns:
            Refined clusters based on people patterns
        """
        # For now, just return clusters as-is
        # Future enhancement: split clusters with low people consistency
        # or merge nearby clusters with same people
        return clusters

    def _calculate_confidence_scores(self, clusters: List[MediaCluster]) -> List[MediaCluster]:
        """Calculate confidence scores for clusters based on multiple factors."""
        self.logger.info("Calculating cluster confidence scores...")

        print(f"DEBUG CONFIDENCE: Input clusters: {len(clusters)}, files: {sum(len(c.media_files) for c in clusters)}")
        for cluster in clusters:
            score = 0.0
            factors = 0

            # Time factor (duration and density)
            if cluster.duration_hours > 0:
                # Prefer events that span reasonable time (not too short, not too long)
                if 0.5 <= cluster.duration_hours <= 12:
                    score += 0.3
                elif cluster.duration_hours <= 24:
                    score += 0.2
                else:
                    score += 0.1
                factors += 1

            # Size factor
            if cluster.size >= self.min_cluster_size:
                size_score = min(0.3, cluster.size / 20)  # Cap at 20 files for full score
                score += size_score
                factors += 1

            # Location factor
            if cluster.has_location:
                score += 0.2
                factors += 1

                # Bonus for location consistency
                if len(set(cluster.gps_coordinates)) <= 3:  # Most photos from same location
                    score += 0.1

            # Media type diversity factor
            if cluster.photo_count > 0 and cluster.video_count > 0:
                score += 0.1  # Bonus for mixed media

            # People consistency factor
            people_consistency = cluster.metadata.get('people_consistency_score', 0.0) if cluster.metadata else 0.0
            if people_consistency > 0:
                score += people_consistency * 0.2  # People consistency can add up to 0.2
                factors += 1

                # Bonus for identified people (vs just faces)
                unique_people = cluster.metadata.get('unique_people_count', 0) if cluster.metadata else 0
                if unique_people > 0 and len(cluster.people_detected) > 0:
                    score += 0.1  # Bonus for having identified people

            # Normalize score
            if factors > 0:
                cluster.confidence_score = score / factors
            else:
                cluster.confidence_score = 0.0

        print(f"DEBUG CONFIDENCE: Output clusters: {len(clusters)}, files: {sum(len(c.media_files) for c in clusters)}")
        return clusters

    def get_clustering_summary(self, clusters: List[MediaCluster]) -> Dict[str, Any]:
        """Get comprehensive summary of clustering results."""
        if not clusters:
            return {
                'total_clusters': 0,
                'total_files': 0,
                'avg_confidence': 0.0,
                'clusters_with_location': 0,
                'avg_cluster_size': 0.0,
                'avg_duration_hours': 0.0
            }

        total_files = sum(c.size for c in clusters)
        clusters_with_location = sum(1 for c in clusters if c.has_location)
        avg_confidence = sum(c.confidence_score for c in clusters) / len(clusters)
        avg_size = total_files / len(clusters)
        avg_duration = sum(c.duration_hours for c in clusters) / len(clusters)

        # Quality distribution
        high_confidence = sum(1 for c in clusters if c.confidence_score >= 0.7)
        medium_confidence = sum(1 for c in clusters if 0.4 <= c.confidence_score < 0.7)
        low_confidence = sum(1 for c in clusters if c.confidence_score < 0.4)

        return {
            'total_clusters': len(clusters),
            'total_files': total_files,
            'avg_confidence': round(avg_confidence, 3),
            'clusters_with_location': clusters_with_location,
            'location_coverage': round(clusters_with_location / len(clusters), 3),
            'avg_cluster_size': round(avg_size, 1),
            'avg_duration_hours': round(avg_duration, 2),
            'quality_distribution': {
                'high_confidence': high_confidence,
                'medium_confidence': medium_confidence,
                'low_confidence': low_confidence
            },
            'size_distribution': {
                'small_clusters': sum(1 for c in clusters if c.size < 5),
                'medium_clusters': sum(1 for c in clusters if 5 <= c.size < 15),
                'large_clusters': sum(1 for c in clusters if c.size >= 15)
            }
        }

    def suggest_event_names(self, clusters: List[MediaCluster], enable_llm: bool = False) -> List[MediaCluster]:
        """Generate intelligent event names using the EventNamer system.

        For junior developers:
        This method uses our sophisticated EventNamer to generate meaningful
        folder names like "2024_10_25 - Halloween Party - Edmonton" instead
        of generic names like "2024_10_25 - Event".

        Args:
            clusters: List of media clusters to name
            enable_llm: Whether to use LLM for advanced naming (requires API key)
        """
        print(f"🎯 CLUSTERING: suggest_event_names called with {len(clusters)} clusters, enable_llm={enable_llm}")
        self.logger.info("Generating intelligent event names for clusters...")

        # Update EventNamer LLM setting if provided
        if hasattr(self, 'event_namer'):
            self.event_namer.enable_llm = enable_llm

        for cluster in clusters:
            try:
                # Prepare cluster data for the EventNamer
                cluster_data = {
                    'start_time': cluster.temporal_info.start_time,
                    'end_time': cluster.temporal_info.end_time,
                    'duration_hours': cluster.duration_hours,
                    'size': cluster.size,
                    'photo_count': cluster.photo_count,
                    'video_count': cluster.video_count,
                    'location_info': cluster.location_info,
                    'dominant_location': cluster.dominant_location,
                    'gps_coordinates': cluster.gps_coordinates,
                    'content_tags': cluster.content_tags,
                    'people_detected': cluster.people_detected,
                    'confidence_score': cluster.confidence_score,
                    'media_files': cluster.media_files
                }

                # Generate intelligent event name
                print(f"📝 CLUSTERING: About to call EventNamer for cluster {cluster.cluster_id} with {len(cluster.media_files)} files")
                suggested_name = self.event_namer.generate_event_name(cluster_data)
                cluster.suggested_name = suggested_name
                print(f"📝 CLUSTERING: EventNamer returned: {suggested_name}")

                self.logger.debug(f"Generated name for cluster {cluster.cluster_id}: '{suggested_name}'")

            except Exception as e:
                self.logger.warning(f"Error generating name for cluster {cluster.cluster_id}: {e}")
                # Fallback to simple naming
                cluster.suggested_name = self._generate_fallback_name(cluster)

        return clusters

    def _generate_fallback_name(self, cluster: MediaCluster) -> str:
        """Generate a simple fallback name if intelligent naming fails.

        For junior developers:
        This is our "safety net" - if the smart naming system fails for any
        reason, we still generate a basic but usable folder name.
        """
        date_str = cluster.temporal_info.start_time.strftime("%Y_%m_%d")

        # Add simple duration hint
        if cluster.duration_hours < 2:
            return f"{date_str} - Quick Event"
        elif cluster.duration_hours > 8:
            return f"{date_str} - All Day Event"
        else:
            return f"{date_str} - Event"