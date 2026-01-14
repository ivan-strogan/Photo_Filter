"""Main media processor that orchestrates the photo organization pipeline."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from .media_detector import MediaDetector
from .metadata_extractor import MetadataExtractor
from .media_clustering import MediaClusteringEngine, MediaCluster
from .content_analyzer import ContentAnalyzer
from .logging_utils import setup_logging, get_logger, ProgressTracker
from .config import *

class MediaProcessor:
    """Main orchestrator for the photo filtering and organization pipeline."""

    def __init__(self, verbose: bool = False):
        """Initialize the media processor.

        Args:
            verbose: Enable verbose logging
        """
        # Set up logging
        log_level = "DEBUG" if verbose else "INFO"
        self.logger_manager = setup_logging(log_level)
        self.logger = logging.getLogger("MediaProcessor")

        # Initialize components
        self.media_detector = MediaDetector()
        self.metadata_extractor = MetadataExtractor()
        self.content_analyzer = ContentAnalyzer(use_gpu=USE_GPU)
        self.clustering_engine = MediaClusteringEngine(
            time_threshold_hours=TIME_THRESHOLD_HOURS,
            location_threshold_km=LOCATION_THRESHOLD_KM,
            min_cluster_size=MIN_CLUSTER_SIZE
        )

        self.logger.info("MediaProcessor initialized successfully")

    def scan_existing_library(self) -> Dict[str, Any]:
        """Scan the existing Pictures library to understand current organization.

        Returns:
            Dictionary with scan results
        """
        self.logger_manager.log_operation_start("Scan Existing Library", {
            "directory": str(PICTURES_DIR)
        })

        try:
            # Scan Pictures library
            progress = self.logger_manager.create_progress_tracker(1, "Scanning Pictures library")
            organized_files = self.media_detector.scan_pictures_library()
            progress.update(1)
            scan_summary = progress.close()

            # Get statistics
            stats = self.media_detector.get_media_stats(organized_files)
            self.logger_manager.log_file_processing_stats(stats)

            # Analyze organization patterns
            organization_analysis = self._analyze_existing_organization(organized_files)

            results = {
                'scan_summary': scan_summary,
                'file_stats': stats,
                'organization_analysis': organization_analysis,
                'organized_files': organized_files
            }

            self.logger_manager.log_operation_end("Scan Existing Library", {
                "total_files": len(organized_files),
                "event_folders": len(organization_analysis.get('event_folders', {}))
            })

            return results

        except Exception as e:
            self.logger_manager.log_error_with_context(e, {
                "operation": "scan_existing_library",
                "directory": str(PICTURES_DIR)
            })
            raise

    def process_new_media(self) -> Dict[str, Any]:
        """Process new media from iPhone Automatic folder.

        Returns:
            Dictionary with processing results
        """
        self.logger_manager.log_operation_start("Process New Media", {
            "directory": str(IPHONE_AUTOMATIC_DIR)
        })

        try:
            # Step 1: Scan unorganized files
            progress = self.logger_manager.create_progress_tracker(1, "Scanning unorganized files")
            unorganized_files = self.media_detector.scan_iphone_automatic()
            progress.update(1)
            progress.close()

            if not unorganized_files:
                self.logger.info("No unorganized files found")
                return {'clusters': [], 'summary': {}}

            # Get file statistics
            file_stats = self.media_detector.get_media_stats(unorganized_files)
            self.logger_manager.log_file_processing_stats(file_stats)

            # Step 2: Perform comprehensive clustering
            self.logger.info(f"Starting clustering of {len(unorganized_files)} files")
            clusters = self.clustering_engine.cluster_media_files(unorganized_files)

            # Step 3: Analyze content for photo clusters
            self.logger.info("Analyzing cluster content...")
            clusters = self._analyze_cluster_content(clusters)

            # Step 4: Suggest event names
            clusters = self.clustering_engine.suggest_event_names(clusters)

            # Step 4: Get clustering summary
            clustering_summary = self.clustering_engine.get_clustering_summary(clusters)
            self.logger_manager.log_clustering_results(clustering_summary)

            results = {
                'file_stats': file_stats,
                'clusters': clusters,
                'clustering_summary': clustering_summary,
                'unorganized_files': unorganized_files
            }

            self.logger_manager.log_operation_end("Process New Media", {
                "total_files": len(unorganized_files),
                "clusters_created": len(clusters),
                "avg_confidence": clustering_summary.get('avg_confidence', 0)
            })

            return results

        except Exception as e:
            self.logger_manager.log_error_with_context(e, {
                "operation": "process_new_media",
                "directory": str(IPHONE_AUTOMATIC_DIR)
            })
            raise

    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete photo organization pipeline.

        Returns:
            Dictionary with complete pipeline results
        """
        self.logger_manager.log_operation_start("Complete Pipeline")

        try:
            pipeline_start = datetime.now()

            # Step 1: Scan existing library
            self.logger.info("Phase 1: Scanning existing library...")
            existing_library_results = self.scan_existing_library()

            # Step 2: Process new media
            self.logger.info("Phase 2: Processing new media...")
            new_media_results = self.process_new_media()

            # Step 3: Generate recommendations
            self.logger.info("Phase 3: Generating recommendations...")
            recommendations = self._generate_organization_recommendations(
                existing_library_results,
                new_media_results
            )

            pipeline_end = datetime.now()
            pipeline_duration = pipeline_end - pipeline_start

            # Compile complete results
            complete_results = {
                'pipeline_info': {
                    'start_time': pipeline_start.isoformat(),
                    'end_time': pipeline_end.isoformat(),
                    'duration_seconds': pipeline_duration.total_seconds()
                },
                'existing_library': existing_library_results,
                'new_media': new_media_results,
                'recommendations': recommendations
            }

            # Save session report
            self.logger_manager.save_session_report(complete_results)

            self.logger_manager.log_operation_end("Complete Pipeline", {
                "duration_minutes": round(pipeline_duration.total_seconds() / 60, 2),
                "new_clusters": len(new_media_results.get('clusters', [])),
                "recommendations": len(recommendations.get('suggested_actions', []))
            })

            return complete_results

        except Exception as e:
            self.logger_manager.log_error_with_context(e, {
                "operation": "run_complete_pipeline"
            })
            raise

    def _analyze_existing_organization(self, organized_files: List) -> Dict[str, Any]:
        """Analyze the existing organization patterns.

        Args:
            organized_files: List of organized media files

        Returns:
            Dictionary with organization analysis
        """
        event_folders = {}
        yearly_distribution = {}

        for media_file in organized_files:
            # Get event folder
            event_folder = getattr(media_file, 'event_folder', 'Unknown')
            if event_folder not in event_folders:
                event_folders[event_folder] = {
                    'count': 0,
                    'photos': 0,
                    'videos': 0,
                    'date_range': [None, None]
                }

            event_info = event_folders[event_folder]
            event_info['count'] += 1

            if media_file.file_type == 'photo':
                event_info['photos'] += 1
            elif media_file.file_type == 'video':
                event_info['videos'] += 1

            # Update date range
            file_date = media_file.date
            if event_info['date_range'][0] is None or file_date < event_info['date_range'][0]:
                event_info['date_range'][0] = file_date
            if event_info['date_range'][1] is None or file_date > event_info['date_range'][1]:
                event_info['date_range'][1] = file_date

            # Yearly distribution
            year = file_date.year
            yearly_distribution[year] = yearly_distribution.get(year, 0) + 1

        return {
            'event_folders': event_folders,
            'yearly_distribution': yearly_distribution,
            'total_event_folders': len(event_folders),
            'naming_patterns': self._analyze_naming_patterns(event_folders)
        }

    def _analyze_naming_patterns(self, event_folders: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze naming patterns in existing event folders.

        Args:
            event_folders: Dictionary of event folder information

        Returns:
            Dictionary with naming pattern analysis
        """
        patterns = {
            'date_prefixed': 0,
            'location_mentioned': 0,
            'descriptive_names': 0,
            'to_filter_folders': 0
        }

        for folder_name in event_folders.keys():
            folder_lower = folder_name.lower()

            # Check for date prefix (YYYY_MM_DD pattern)
            if any(char.isdigit() for char in folder_name[:10]):
                patterns['date_prefixed'] += 1

            # Check for location mentions (rough heuristic)
            location_keywords = ['mexico', 'trip', 'vacation', 'birthday', 'wedding', 'calgary', 'edmonton']
            if any(keyword in folder_lower for keyword in location_keywords):
                patterns['location_mentioned'] += 1

            # Check for descriptive names
            if len(folder_name.split()) > 2:
                patterns['descriptive_names'] += 1

            # Check for "to filter" folders
            if 'filter' in folder_lower or 'sort' in folder_lower:
                patterns['to_filter_folders'] += 1

        return patterns

    def _analyze_cluster_content(self, clusters: List[MediaCluster]) -> List[MediaCluster]:
        """Analyze photo content for each cluster to enhance event identification.

        Args:
            clusters: List of media clusters

        Returns:
            Enhanced clusters with content analysis
        """
        try:
            for cluster in clusters:
                # Only analyze photos (skip videos for now)
                photo_files = [f for f in cluster.media_files if f.file_type == 'photo']

                if not photo_files:
                    continue

                self.logger.debug(f"Analyzing content for cluster with {len(photo_files)} photos")

                # Analyze content for a sample of photos (max 5 to avoid excessive processing)
                sample_size = min(5, len(photo_files))
                sample_photos = photo_files[:sample_size]

                # Convert to Path objects for analysis
                photo_paths = [f.path for f in sample_photos]

                # Perform batch content analysis
                content_results = self.content_analyzer.analyze_batch(photo_paths, max_photos=sample_size)

                if content_results:
                    # Get content summary
                    content_summary = self.content_analyzer.get_content_summary(content_results)

                    # Add content information to cluster
                    cluster.content_analysis = {
                        'analyzed_photos': len(content_results),
                        'average_confidence': content_summary.get('average_confidence', 0.0),
                        'top_objects': [obj for obj, count in content_summary.get('top_objects', [])[:3]],
                        'top_scenes': [scene for scene, count in content_summary.get('top_scenes', [])[:2]],
                        'top_activities': [activity for activity, count in content_summary.get('top_activities', [])[:2]]
                    }

                    self.logger.debug(f"Content analysis complete: {cluster.content_analysis}")
                else:
                    self.logger.debug("No content analysis results available")

        except Exception as e:
            self.logger.warning(f"Error analyzing cluster content: {e}")
            # Continue processing even if content analysis fails

        return clusters

    def _generate_organization_recommendations(self,
                                            existing_library: Dict[str, Any],
                                            new_media: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations for organizing new media.

        Args:
            existing_library: Results from existing library scan
            new_media: Results from new media processing

        Returns:
            Dictionary with organization recommendations
        """
        recommendations = {
            'suggested_actions': [],
            'folder_name_suggestions': [],
            'quality_assessment': {},
            'organization_strategy': 'temporal_location'
        }

        clusters = new_media.get('clusters', [])
        if not clusters:
            return recommendations

        # Generate folder name suggestions
        for i, cluster in enumerate(clusters):
            if cluster.confidence_score >= 0.5:  # Only suggest high-quality clusters
                action = {
                    'cluster_id': cluster.cluster_id,
                    'suggested_folder_name': cluster.suggested_name,
                    'file_count': cluster.size,
                    'confidence': cluster.confidence_score,
                    'action_type': 'create_folder',
                    'priority': 'high' if cluster.confidence_score >= 0.7 else 'medium'
                }
                recommendations['suggested_actions'].append(action)

        # Quality assessment
        clustering_summary = new_media.get('clustering_summary', {})
        quality_dist = clustering_summary.get('quality_distribution', {})

        recommendations['quality_assessment'] = {
            'total_clusters': len(clusters),
            'high_quality_clusters': quality_dist.get('high_confidence', 0),
            'medium_quality_clusters': quality_dist.get('medium_confidence', 0),
            'low_quality_clusters': quality_dist.get('low_confidence', 0),
            'average_confidence': clustering_summary.get('avg_confidence', 0),
            'location_coverage': clustering_summary.get('location_coverage', 0)
        }

        # Strategy recommendation based on existing patterns
        organization_analysis = existing_library.get('organization_analysis', {})
        naming_patterns = organization_analysis.get('naming_patterns', {})

        if naming_patterns.get('date_prefixed', 0) > naming_patterns.get('descriptive_names', 0):
            recommendations['organization_strategy'] = 'date_focused'
        elif naming_patterns.get('location_mentioned', 0) > len(clusters) // 2:
            recommendations['organization_strategy'] = 'location_focused'
        else:
            recommendations['organization_strategy'] = 'temporal_location'

        return recommendations

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and configuration.

        Returns:
            Dictionary with system status
        """
        return {
            'directories': {
                'base_dir': str(BASE_DIR),
                'sample_photos_dir': str(SAMPLE_PHOTOS_DIR),
                'iphone_automatic_dir': str(IPHONE_AUTOMATIC_DIR),
                'pictures_dir': str(PICTURES_DIR),
                'vector_db_dir': str(VECTOR_DB_DIR)
            },
            'configuration': {
                'time_threshold_hours': TIME_THRESHOLD_HOURS,
                'location_threshold_km': LOCATION_THRESHOLD_KM,
                'min_cluster_size': MIN_CLUSTER_SIZE,
                'supported_extensions': list(SUPPORTED_EXTENSIONS)
            },
            'logging': self.logger_manager.get_log_summary() if self.logger_manager else {},
            'components_initialized': {
                'media_detector': self.media_detector is not None,
                'metadata_extractor': self.metadata_extractor is not None,
                'clustering_engine': self.clustering_engine is not None
            }
        }