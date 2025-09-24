"""
Complete Photo Organization Pipeline.

This module combines all components to provide a complete end-to-end photo
organization system: clustering, naming, folder creation, and file organization.

For junior developers:
- Shows how to orchestrate multiple complex systems
- Demonstrates proper error handling across system boundaries
- Implements comprehensive progress tracking and reporting
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import json

from .media_detector import MediaDetector
from .media_clustering import MediaClusteringEngine
from .folder_organizer import FolderOrganizer
from .file_organizer import FileOrganizer
from .media_validator import MediaValidator
from .organized_photos_scanner import OrganizedPhotosScanner
from .config_manager import get_config

class PhotoOrganizerPipeline:
    """
    Complete photo organization pipeline that orchestrates all components.

    This class provides a single interface to run the complete photo organization
    workflow from scanning unorganized photos to creating organized folder structures
    with intelligently moved files.

    For junior developers:
    - This is a "facade" pattern - single interface to complex subsystems
    - Pipeline pattern - data flows through multiple processing stages
    - Observer pattern - progress callbacks allow UI updates
    """

    def __init__(self,
                 max_photos: int = 100,
                 operation_mode: str = "copy",
                 dry_run: bool = True,
                 verify_checksums: bool = True):
        """
        Initialize the photo organization pipeline.

        Args:
            max_photos: Maximum number of photos to process
            operation_mode: "copy" (safer) or "move" (more efficient)
            dry_run: If True, only simulate operations
            verify_checksums: If True, verify file integrity
        """
        self.logger = logging.getLogger(__name__)
        self.max_photos = max_photos
        self.operation_mode = operation_mode
        self.dry_run = dry_run
        self.verify_checksums = verify_checksums

        # Initialize basic components (clustering engine will be initialized after vector DB)
        self.media_detector = MediaDetector()
        self.media_validator = MediaValidator(enable_deep_validation=True)
        self.folder_organizer = FolderOrganizer(dry_run=dry_run)
        self.file_organizer = FileOrganizer(
            operation_mode=operation_mode,
            dry_run=dry_run,
            verify_checksums=verify_checksums
        )
        self.organized_photos_scanner = OrganizedPhotosScanner()

        # Clustering engine will be initialized after vector database setup
        self.clustering_engine = None

        # Pipeline state
        self.pipeline_results = {
            'stage_results': {},
            'overall_stats': {},
            'errors': [],
            'warnings': []
        }

        self.logger.info(f"PhotoOrganizerPipeline initialized")
        self.logger.info(f"Max photos: {max_photos}")
        self.logger.info(f"Operation mode: {operation_mode}")
        self.logger.info(f"Dry run: {dry_run}")

    def run_complete_pipeline(self,
                            source_folder: Optional[Path] = None,
                            output_folder: Optional[Path] = None,
                            progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run the complete photo organization pipeline.

        Args:
            source_folder: Source folder to scan (default: iPhone Automatic)
            output_folder: Output folder for organized photos (default: from config)
            progress_callback: Optional callback for progress updates

        Returns:
            Comprehensive results dictionary

        For junior developers:
        - This is the main orchestrator method
        - Each stage builds on the previous stage's results
        - Comprehensive error handling ensures partial failures are handled gracefully
        """
        self.logger.info("Starting complete photo organization pipeline")
        pipeline_start_time = datetime.now()

        try:
            # Stage 0: Initialize vector database with existing organized photos
            self.logger.info("Stage 0: Building vector database from existing organized photos...")
            if progress_callback:
                progress_callback("Analyzing existing organized photos...", 0.05)

            vector_results = self._stage_initialize_vector_database()
            self.pipeline_results['stage_results']['vector_db'] = vector_results

            # Stage 1: Scan and detect media files
            self.logger.info("Stage 1: Scanning media files...")
            if progress_callback:
                progress_callback("Scanning media files...", 0.08)

            scan_results = self._stage_scan_media_files(source_folder)
            self.pipeline_results['stage_results']['scan'] = scan_results

            if not scan_results['media_files']:
                self.logger.warning("No media files found to process")
                return self._generate_pipeline_report(pipeline_start_time)

            # Stage 2: Validate media files
            self.logger.info("Stage 2: Validating media files...")
            if progress_callback:
                progress_callback("Validating file integrity and formats...", 0.2)

            validation_results = self._stage_validate_media_files(scan_results['media_files'])
            self.pipeline_results['stage_results']['validation'] = validation_results

            if not validation_results['valid_media_files']:
                self.logger.warning("No valid media files found after validation")
                return self._generate_pipeline_report(pipeline_start_time)

            # Stage 3: Perform intelligent clustering
            self.logger.info("Stage 3: Performing intelligent clustering...")
            if progress_callback:
                progress_callback("Clustering photos by time and location...", 0.35)

            clustering_results = self._stage_cluster_media_files(validation_results['valid_media_files'])
            self.pipeline_results['stage_results']['clustering'] = clustering_results

            print(f"🔍 PIPELINE DEBUG: Clustering stage returned {len(clustering_results.get('clusters', []))} clusters")
            print(f"🔍 PIPELINE DEBUG: Clustering results keys: {clustering_results.keys()}")

            if not clustering_results['clusters']:
                print(f"❌ PIPELINE DEBUG: No clusters found, skipping naming stage")
                self.logger.warning("No clusters created from media files")
                return self._generate_pipeline_report(pipeline_start_time)

            # Stage 4: Generate intelligent event names
            self.logger.info("Stage 4: Generating intelligent event names...")
            if progress_callback:
                progress_callback("Generating intelligent folder names...", 0.5)

            print(f"🔍 PIPELINE DEBUG: About to call naming stage with {len(clustering_results.get('clusters', []))} clusters from clustering_results")
            naming_results = self._stage_generate_event_names(clustering_results['clusters'])
            self.pipeline_results['stage_results']['naming'] = naming_results

            # Stage 5: Create folder structure
            self.logger.info("Stage 5: Creating folder structure...")
            if progress_callback:
                progress_callback("Creating organized folder structure...", 0.7)

            folder_results = self._stage_create_folder_structure(
                naming_results['named_clusters'],
                output_folder
            )
            self.pipeline_results['stage_results']['folders'] = folder_results

            # Stage 6: Organize files (move/copy to folders)
            self.logger.info("Stage 6: Organizing files...")
            if progress_callback:
                progress_callback("Moving/copying files to organized folders...", 0.9)

            organization_results = self._stage_organize_files(
                naming_results['named_clusters'],
                folder_results['folder_mapping'],
                progress_callback
            )
            self.pipeline_results['stage_results']['organization'] = organization_results

            # Final stage: Generate comprehensive report
            if progress_callback:
                progress_callback("Generating final report...", 1.0)

            final_report = self._generate_pipeline_report(pipeline_start_time)

            self.logger.info("Photo organization pipeline completed successfully")
            return final_report

        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {e}")
            self.pipeline_results['errors'].append({
                'stage': 'pipeline',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return self._generate_pipeline_report(pipeline_start_time)

    def _stage_scan_media_files(self, source_folder: Optional[Path]) -> Dict[str, Any]:
        """Stage 1: Scan and detect media files."""
        try:
            # Always scan iPhone Automatic folder (hardcoded)
            all_files = self.media_detector.scan_iphone_automatic()

            # Filter to photos and videos, limit by max_photos
            photo_files = [f for f in all_files if f.file_type == 'photo']
            video_files = [f for f in all_files if f.file_type == 'video']

            # Take first N photos as requested
            selected_photos = sorted(photo_files, key=lambda x: x.date)[:self.max_photos]

            # Also include videos from the same time period if any
            if selected_photos:
                earliest_date = selected_photos[0].date
                latest_date = selected_photos[-1].date
                selected_videos = [
                    v for v in video_files
                    if earliest_date <= v.date <= latest_date
                ]
            else:
                selected_videos = []

            media_files = selected_photos + selected_videos

            return {
                'media_files': media_files,
                'total_scanned': len(all_files),
                'photos_found': len(photo_files),
                'videos_found': len(video_files),
                'photos_selected': len(selected_photos),
                'videos_selected': len(selected_videos),
                'date_range': (
                    media_files[0].date.isoformat() if media_files else None,
                    media_files[-1].date.isoformat() if media_files else None
                )
            }

        except Exception as e:
            self.logger.error(f"Media scanning failed: {e}")
            self.pipeline_results['errors'].append({
                'stage': 'scan',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return {'media_files': [], 'error': str(e)}

    def _stage_validate_media_files(self, media_files: List[Any]) -> Dict[str, Any]:
        """Stage 2: Validate media files for corruption and format support."""
        try:
            # Create progress callback for validation
            def validation_progress(progress, current, total):
                # This is an internal progress, don't expose to main callback
                pass

            # Get file paths from media files
            file_paths = [media_file.path for media_file in media_files]

            # Perform batch validation
            validation_results = self.media_validator.validate_batch(
                file_paths, validation_progress
            )

            # Filter valid files
            valid_files = self.media_validator.filter_valid_files(validation_results)
            corrupted_files = self.media_validator.filter_corrupted_files(validation_results)

            # Create valid MediaFile objects
            valid_media_files = [
                mf for mf in media_files
                if mf.path in valid_files
            ]

            # Get validation summary
            validation_summary = self.media_validator.get_validation_summary()

            # Log validation results
            self.logger.info(f"Validation completed: {len(valid_media_files)}/{len(media_files)} files valid")

            if corrupted_files:
                self.logger.warning(f"Found {len(corrupted_files)} corrupted files")
                for corrupted_file in corrupted_files[:5]:  # Log first 5
                    self.logger.warning(f"Corrupted: {corrupted_file.name}")

            # Log unsupported files
            unsupported_count = validation_summary['unsupported_files']
            if unsupported_count > 0:
                self.logger.warning(f"Found {unsupported_count} unsupported files")

            return {
                'valid_media_files': valid_media_files,
                'validation_results': validation_results,
                'validation_summary': validation_summary,
                'files_input': len(media_files),
                'files_valid': len(valid_media_files),
                'files_corrupted': len(corrupted_files),
                'files_unsupported': unsupported_count,
                'validation_success_rate': validation_summary['success_rate']
            }

        except Exception as e:
            self.logger.error(f"Media validation failed: {e}")
            self.pipeline_results['errors'].append({
                'stage': 'validation',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            # Return original files if validation fails
            return {
                'valid_media_files': media_files,
                'error': str(e),
                'files_input': len(media_files),
                'files_valid': len(media_files),
                'validation_fallback': True
            }

    def _stage_cluster_media_files(self, media_files: List[Any]) -> Dict[str, Any]:
        """Stage 2: Perform intelligent clustering."""
        try:
            print(f"DEBUG CLUSTERING: Input files: {len(media_files)}")

            # Check for files with None timestamps
            files_with_time = [f for f in media_files if f.time is not None]
            files_without_time = [f for f in media_files if f.time is None]
            if files_without_time:
                print(f"DEBUG CLUSTERING: Files without timestamps: {len(files_without_time)}")
                for f in files_without_time[:3]:
                    print(f"  - {f.filename}")

            clusters = self.clustering_engine.cluster_media_files(media_files)
            clustering_summary = self.clustering_engine.get_clustering_summary(clusters)

            files_in_clusters = sum(c.size for c in clusters)
            print(f"DEBUG CLUSTERING: Files in clusters: {files_in_clusters}")
            print(f"DEBUG CLUSTERING: Missing from clustering: {len(media_files) - files_in_clusters}")

            if len(media_files) != files_in_clusters:
                # Find which files didn't make it
                clustered_filenames = set()
                for cluster in clusters:
                    for f in cluster.media_files:
                        clustered_filenames.add(f.filename)

                missing_files = [f.filename for f in media_files if f.filename not in clustered_filenames]
                print(f"DEBUG CLUSTERING: Missing files: {missing_files[:5]}")

            return {
                'clusters': clusters,
                'clustering_summary': clustering_summary,
                'clusters_created': len(clusters),
                'files_clustered': files_in_clusters
            }

        except Exception as e:
            print(f"💥 CLUSTERING STAGE ERROR: {e}")
            import traceback
            print(f"💥 CLUSTERING STAGE TRACEBACK:")
            traceback.print_exc()
            self.logger.error(f"Clustering failed: {e}")
            self.pipeline_results['errors'].append({
                'stage': 'clustering',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return {'clusters': [], 'error': str(e)}

    def _stage_generate_event_names(self, clusters: List[Any]) -> Dict[str, Any]:
        """Stage 3: Generate intelligent event names."""
        try:
            # Use LLM if enabled in config
            enable_llm = self.config.naming.use_llm_naming if hasattr(self, 'config') else True
            print(f"🚀 PIPELINE: About to call suggest_event_names with {len(clusters)} clusters, enable_llm={enable_llm}")
            named_clusters = self.clustering_engine.suggest_event_names(clusters, enable_llm=enable_llm)
            print(f"🚀 PIPELINE: suggest_event_names completed, returned {len(named_clusters)} clusters")

            # Calculate naming quality statistics
            high_confidence_names = sum(1 for c in named_clusters if c.confidence_score >= 0.7)
            named_clusters_count = sum(1 for c in named_clusters if c.suggested_name)

            return {
                'named_clusters': named_clusters,
                'total_clusters': len(named_clusters),
                'clusters_with_names': named_clusters_count,
                'high_confidence_names': high_confidence_names,
                'naming_success_rate': named_clusters_count / max(1, len(named_clusters))
            }

        except Exception as e:
            self.logger.error(f"Event naming failed: {e}")
            self.pipeline_results['errors'].append({
                'stage': 'naming',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return {'named_clusters': clusters, 'error': str(e)}

    def _stage_create_folder_structure(self,
                                     named_clusters: List[Any],
                                     output_folder: Optional[Path]) -> Dict[str, Any]:
        """Stage 4: Create folder structure."""
        try:
            if output_folder:
                self.folder_organizer = FolderOrganizer(
                    base_output_dir=output_folder,
                    dry_run=self.dry_run
                )

            folder_results = self.folder_organizer.create_folder_structure(named_clusters)
            return folder_results

        except Exception as e:
            self.logger.error(f"Folder creation failed: {e}")
            self.pipeline_results['errors'].append({
                'stage': 'folders',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return {'folder_mapping': {}, 'error': str(e)}

    def _stage_organize_files(self,
                            named_clusters: List[Any],
                            folder_mapping: Dict[int, Path],
                            progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Stage 5: Organize files into folders."""
        try:
            # Debug: Count files going into organization
            total_files_to_organize = sum(len(cluster.media_files) for cluster in named_clusters)
            print(f"DEBUG ORGANIZATION: Input clusters: {len(named_clusters)}")
            print(f"DEBUG ORGANIZATION: Total files to organize: {total_files_to_organize}")

            # Create progress wrapper for file organization
            def file_progress_wrapper(progress, current, total):
                if progress_callback:
                    # Map file progress to overall pipeline progress (90-99%)
                    overall_progress = 0.9 + (progress * 0.09)
                    progress_callback(f"Organizing files ({current}/{total})", overall_progress)

            organization_results = self.file_organizer.organize_files(
                named_clusters,
                folder_mapping,
                file_progress_wrapper
            )

            return organization_results

        except Exception as e:
            self.logger.error(f"File organization failed: {e}")
            self.pipeline_results['errors'].append({
                'stage': 'organization',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return {'operation_summary': {'total_files_processed': 0}, 'error': str(e)}

    def _stage_initialize_vector_database(self) -> Dict[str, Any]:
        """Stage 0: Initialize vector database with existing organized photos."""
        try:
            self.logger.info("Initializing vector database with existing organized photos...")

            # Check configuration to see if vectorization is enabled
            config = get_config()
            if not config.processing.enable_vectorization:
                self.logger.info("Vectorization disabled in configuration, skipping vector database initialization")
                # Initialize clustering engine without vector database components
                self._initialize_clustering_engine_with_vector_db(vector_db=None, photo_vectorizer=None)
                return {
                    'vectorization_enabled': False,
                    'photos_scanned': 0,
                    'photos_vectorized': 0,
                    'photos_added_to_db': 0,
                    'event_folders_processed': 0
                }

            # Check if vector database already exists and skip if it has enough data
            db_summary = self.organized_photos_scanner.get_database_summary()
            if db_summary and db_summary.get('total_photos', 0) > 10:
                self.logger.info(f"Vector database already exists with {db_summary['total_photos']} photos, skipping rebuild")
                scan_results = {
                    'total_photos_found': db_summary.get('total_photos', 0),
                    'photos_vectorized': 0,
                    'photos_added_to_db': 0,
                    'event_folders_processed': db_summary.get('event_folders_count', 0),
                    'execution_time_seconds': 0.0,
                    'skipped_existing': True
                }
            else:
                # Scan existing organized photos and build vector database (full scan for first time)
                self.logger.info("Building vector database for first time - this may take several hours...")
                scan_results = self.organized_photos_scanner.scan_and_build_database(
                    max_photos_per_event=50,  # Full scan
                    skip_large_events=False,  # Process all events
                    max_events=None,  # No limit on events
                    quick_scan=False  # Full comprehensive scan
                )

            # Get database statistics
            db_summary = self.organized_photos_scanner.get_database_summary()

            self.logger.info(f"Vector database initialized with {scan_results.get('photos_added_to_db', 0)} photos from {scan_results.get('event_folders_processed', 0)} event folders")

            # Initialize clustering engine with vector database components
            self._initialize_clustering_engine_with_vector_db()

            return {
                'vectorization_enabled': True,
                'photos_scanned': scan_results.get('total_photos_found', 0),
                'photos_vectorized': scan_results.get('photos_vectorized', 0),
                'photos_added_to_db': scan_results.get('photos_added_to_db', 0),
                'event_folders_processed': scan_results.get('event_folders_processed', 0),
                'database_stats': db_summary,
                'execution_time_seconds': scan_results.get('execution_time_seconds', 0)
            }

        except Exception as e:
            self.logger.warning(f"Vector database initialization failed: {e}")
            # Initialize clustering engine without vector database components as fallback
            self._initialize_clustering_engine_with_vector_db(vector_db=None, photo_vectorizer=None)
            self.pipeline_results['warnings'].append({
                'stage': 'vector_db',
                'warning': f"Could not initialize vector database: {str(e)}",
                'timestamp': datetime.now().isoformat()
            })
            return {
                'vectorization_enabled': False,
                'photos_scanned': 0,
                'photos_vectorized': 0,
                'photos_added_to_db': 0,
                'event_folders_processed': 0,
                'error': str(e)
            }

    def _initialize_clustering_engine_with_vector_db(self, vector_db=None, photo_vectorizer=None):
        """Initialize clustering engine with vector database components."""
        try:
            # Get vector database and photo vectorizer from scanner if not provided
            if vector_db is None and photo_vectorizer is None:
                if hasattr(self.organized_photos_scanner, 'vector_db') and self.organized_photos_scanner.vector_db:
                    vector_db = self.organized_photos_scanner.vector_db
                if hasattr(self.organized_photos_scanner, 'vectorizer') and self.organized_photos_scanner.vectorizer:
                    photo_vectorizer = self.organized_photos_scanner.vectorizer

            # Initialize clustering engine with vector components
            self.clustering_engine = MediaClusteringEngine(
                vector_db=vector_db,
                photo_vectorizer=photo_vectorizer
            )

            if vector_db and photo_vectorizer:
                self.logger.info("Clustering engine initialized with vector database support")
            else:
                self.logger.info("Clustering engine initialized without vector database support")

        except Exception as e:
            self.logger.warning(f"Error initializing clustering engine with vector DB: {e}")
            # Fallback to basic clustering engine
            self.clustering_engine = MediaClusteringEngine()
            self.logger.info("Clustering engine initialized with basic configuration")

    def _generate_pipeline_report(self, pipeline_start_time: datetime) -> Dict[str, Any]:
        """Generate comprehensive pipeline report."""
        pipeline_end_time = datetime.now()
        pipeline_duration = (pipeline_end_time - pipeline_start_time).total_seconds()

        # Extract key statistics from each stage
        vector_results = self.pipeline_results['stage_results'].get('vector_db', {})
        scan_results = self.pipeline_results['stage_results'].get('scan', {})
        validation_results = self.pipeline_results['stage_results'].get('validation', {})
        clustering_results = self.pipeline_results['stage_results'].get('clustering', {})
        naming_results = self.pipeline_results['stage_results'].get('naming', {})
        folder_results = self.pipeline_results['stage_results'].get('folders', {})
        organization_results = self.pipeline_results['stage_results'].get('organization', {})

        # Calculate overall success metrics
        files_scanned = scan_results.get('total_scanned', 0)
        files_selected = len(scan_results.get('media_files', []))
        files_validated = validation_results.get('files_valid', 0)
        files_corrupted = validation_results.get('files_corrupted', 0)
        clusters_created = clustering_results.get('clusters_created', 0)
        folders_created = folder_results.get('operation_summary', {}).get('folders_created', 0)
        files_organized = organization_results.get('operation_summary', {}).get('successful_operations', 0)

        return {
            'pipeline_summary': {
                'execution_time_seconds': pipeline_duration,
                'dry_run_mode': self.dry_run,
                'operation_mode': self.operation_mode,
                'max_photos_limit': self.max_photos,
                'overall_success': len(self.pipeline_results['errors']) == 0
            },
            'stage_summaries': {
                'vector_db': {
                    'vectorization_enabled': vector_results.get('vectorization_enabled', False),
                    'photos_scanned': vector_results.get('photos_scanned', 0),
                    'photos_vectorized': vector_results.get('photos_vectorized', 0),
                    'photos_added_to_db': vector_results.get('photos_added_to_db', 0),
                    'event_folders_processed': vector_results.get('event_folders_processed', 0),
                    'execution_time_seconds': vector_results.get('execution_time_seconds', 0)
                },
                'scan': {
                    'files_scanned': files_scanned,
                    'files_selected': files_selected,
                    'selection_rate': files_selected / max(1, files_scanned)
                },
                'validation': {
                    'files_input': files_selected,
                    'files_valid': files_validated,
                    'files_corrupted': files_corrupted,
                    'files_unsupported': validation_results.get('files_unsupported', 0),
                    'validation_success_rate': validation_results.get('validation_success_rate', 0)
                },
                'clustering': {
                    'files_input': files_validated,
                    'clusters_created': clusters_created,
                    'avg_cluster_size': files_validated / max(1, clusters_created)
                },
                'naming': {
                    'clusters_input': clusters_created,
                    'naming_success_rate': naming_results.get('naming_success_rate', 0),
                    'high_confidence_names': naming_results.get('high_confidence_names', 0)
                },
                'folders': {
                    'folders_created': folders_created,
                    'conflicts_resolved': folder_results.get('operation_summary', {}).get('conflicts_resolved', 0)
                },
                'organization': {
                    'files_processed': organization_results.get('operation_summary', {}).get('total_files_processed', 0),
                    'files_successful': files_organized,
                    'success_rate': organization_results.get('operation_summary', {}).get('success_rate', 0)
                }
            },
            'detailed_results': self.pipeline_results['stage_results'],
            'errors': self.pipeline_results['errors'],
            'warnings': self.pipeline_results['warnings'],
            'output_directory': folder_results.get('organized_directory'),
            'timestamp': datetime.now().isoformat()
        }

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'configuration': {
                'max_photos': self.max_photos,
                'operation_mode': self.operation_mode,
                'dry_run': self.dry_run,
                'verify_checksums': self.verify_checksums
            },
            'stage_results_available': list(self.pipeline_results['stage_results'].keys()),
            'errors_count': len(self.pipeline_results['errors']),
            'warnings_count': len(self.pipeline_results['warnings'])
        }

    def save_pipeline_report(self, report: Dict[str, Any], output_file: Optional[Path] = None) -> Path:
        """Save pipeline report to file."""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(f"photo_organization_report_{timestamp}.json")

        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            self.logger.info(f"Pipeline report saved to: {output_file}")
            return output_file

        except Exception as e:
            self.logger.error(f"Failed to save pipeline report: {e}")
            raise