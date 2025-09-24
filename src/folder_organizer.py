"""
Automated folder creation and organization system.

This module creates intelligent folder structures based on event clustering
and handles naming conflicts, permissions, and organizational logic.

For junior developers:
- Shows how to safely create file system structures
- Demonstrates conflict resolution strategies
- Implements proper error handling for file operations
- Uses defensive programming patterns for file system safety
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import shutil
import json

# Use typing for MediaCluster to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .media_clustering import MediaCluster
from .config_manager import get_config

class FolderOrganizer:
    """
    Creates and manages intelligent folder structures for photo organization.

    This class takes clustered photos with intelligent names and creates
    the actual folder structure, handling conflicts and ensuring safe operations.

    For junior developers:
    - Always validate paths before file operations
    - Use atomic operations where possible
    - Create comprehensive logs for troubleshooting
    - Never delete original files without explicit user permission
    """

    def __init__(self, base_output_dir: Optional[Path] = None, dry_run: bool = True):
        """
        Initialize the folder organizer.

        Args:
            base_output_dir: Base directory for organized photos (default from config)
            dry_run: If True, only simulate operations without creating folders

        For junior developers:
        - dry_run mode is essential for testing - always implement it
        - Default parameters make the class easy to use
        - Logging helps track what the system is doing
        """
        self.logger = logging.getLogger(__name__)
        self.dry_run = dry_run

        # Get configuration
        try:
            config = get_config()
            self.base_dir = base_output_dir or Path(config.paths.sample_photos_dir)
        except Exception:
            self.base_dir = base_output_dir or Path("Sample_Photos")

        # Create organized photos directory
        self.organized_dir = self.base_dir / "Organized_Photos"
        self.conflict_log = []
        self.created_folders = []
        self.operation_summary = {
            'folders_created': 0,
            'conflicts_resolved': 0,
            'errors': 0,
            'total_clusters_processed': 0
        }

        self.logger.info(f"FolderOrganizer initialized")
        self.logger.info(f"Base directory: {self.base_dir}")
        self.logger.info(f"Organized directory: {self.organized_dir}")
        self.logger.info(f"Dry run mode: {self.dry_run}")

    def create_folder_structure(self, clusters: List[Any]) -> Dict[str, Any]:
        """
        Create the complete folder structure for organized photo clusters.

        Args:
            clusters: List of media clusters with suggested names

        Returns:
            Dictionary with operation results and folder mapping

        For junior developers:
        - This is the main orchestrator method
        - Always return detailed results for debugging
        - Handle errors gracefully without stopping the entire process
        """
        self.logger.info(f"Creating folder structure for {len(clusters)} clusters")
        self.operation_summary['total_clusters_processed'] = len(clusters)

        if not self.dry_run:
            self._ensure_base_directories_exist()

        folder_mapping = {}  # Maps cluster_id to final folder path
        year_organization = {}  # Groups folders by year for better organization

        for cluster in clusters:
            try:
                folder_path, created = self._create_cluster_folder(cluster)

                if folder_path:
                    folder_mapping[cluster.cluster_id] = folder_path

                    # Track year organization
                    year = cluster.temporal_info.start_time.year
                    if year not in year_organization:
                        year_organization[year] = []
                    year_organization[year].append({
                        'cluster_id': cluster.cluster_id,
                        'folder_name': folder_path.name,
                        'file_count': cluster.size,
                        'suggested_name': cluster.suggested_name
                    })

                    if created:
                        self.operation_summary['folders_created'] += 1

            except Exception as e:
                self.logger.error(f"Error creating folder for cluster {cluster.cluster_id}: {e}")
                self.operation_summary['errors'] += 1

        # Create year-based organization summary
        self._create_year_organization_summary(year_organization)

        # Generate operation report
        report = self._generate_operation_report(folder_mapping, year_organization)

        self.logger.info(f"Folder creation completed: {self.operation_summary['folders_created']} folders created")
        return report

    def _ensure_base_directories_exist(self) -> None:
        """Ensure base directory structure exists."""
        try:
            self.organized_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Ensured directory exists: {self.organized_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create base directory {self.organized_dir}: {e}")
            raise

    def _create_cluster_folder(self, cluster: Any) -> Tuple[Optional[Path], bool]:
        """
        Create folder for a single cluster, handling conflicts.

        Args:
            cluster: Media cluster to create folder for

        Returns:
            Tuple of (folder_path, was_created)

        For junior developers:
        - This method handles the complex logic of naming conflicts
        - Notice how we try multiple strategies before giving up
        - Always log what decisions were made for troubleshooting
        """
        if not cluster.suggested_name:
            self.logger.warning(f"Cluster {cluster.cluster_id} has no suggested name, skipping")
            return None, False

        # Get year for organization
        year = cluster.temporal_info.start_time.year
        year_dir = self.organized_dir / str(year)

        # Clean the suggested name for use as folder name
        clean_name = self._clean_folder_name(cluster.suggested_name)
        base_folder_path = year_dir / clean_name

        # Handle naming conflicts
        final_folder_path, conflict_resolved = self._resolve_naming_conflict(
            base_folder_path, cluster
        )

        if conflict_resolved:
            self.operation_summary['conflicts_resolved'] += 1

        # Create the folder
        created = False
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would create folder: {final_folder_path}")
            created = True
        else:
            try:
                final_folder_path.parent.mkdir(parents=True, exist_ok=True)
                final_folder_path.mkdir(exist_ok=True)
                created = True
                self.created_folders.append(final_folder_path)
                self.logger.info(f"Created folder: {final_folder_path}")

                # Create a metadata file for the folder
                self._create_folder_metadata(final_folder_path, cluster)

            except Exception as e:
                self.logger.error(f"Failed to create folder {final_folder_path}: {e}")
                return None, False

        return final_folder_path, created

    def _clean_folder_name(self, name: str) -> str:
        """
        Clean folder name to be filesystem-safe.

        For junior developers:
        - File systems have different rules for valid characters
        - Always sanitize user-generated content before using as filenames
        - Keep names readable while ensuring compatibility
        """
        # Remove or replace problematic characters
        forbidden_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        clean_name = name

        for char in forbidden_chars:
            clean_name = clean_name.replace(char, '_')

        # Remove multiple spaces and leading/trailing spaces
        clean_name = ' '.join(clean_name.split())

        # Limit length to reasonable size
        if len(clean_name) > 100:
            clean_name = clean_name[:97] + "..."

        # Ensure it's not empty
        if not clean_name.strip():
            clean_name = "Unnamed_Event"

        return clean_name

    def _resolve_naming_conflict(self, folder_path: Path, cluster: Any) -> Tuple[Path, bool]:
        """
        Resolve naming conflicts when folder already exists.

        Args:
            folder_path: Proposed folder path
            cluster: Media cluster data

        Returns:
            Tuple of (final_path, conflict_was_resolved)

        For junior developers:
        - Naming conflicts are common - always have a strategy
        - Multiple resolution approaches increase success rate
        - Document your conflict resolution strategy
        """
        if not folder_path.exists() and self.dry_run:
            return folder_path, False

        if not folder_path.exists():
            return folder_path, False

        # Folder exists - resolve conflict
        self.logger.info(f"Naming conflict detected for: {folder_path}")
        conflict_info = {
            'original_path': str(folder_path),
            'cluster_id': cluster.cluster_id,
            'timestamp': datetime.now().isoformat()
        }

        # Strategy 1: Add time suffix for same-day events
        base_name = folder_path.name
        time_suffix = cluster.temporal_info.start_time.strftime("_%H%M")

        new_path = folder_path.parent / f"{base_name}{time_suffix}"
        if not new_path.exists():
            conflict_info['resolution'] = 'time_suffix'
            conflict_info['final_path'] = str(new_path)
            self.conflict_log.append(conflict_info)
            return new_path, True

        # Strategy 2: Add incremental number
        for i in range(2, 100):  # Try up to 99 variations
            new_path = folder_path.parent / f"{base_name}_{i:02d}"
            if not new_path.exists():
                conflict_info['resolution'] = f'incremental_{i}'
                conflict_info['final_path'] = str(new_path)
                self.conflict_log.append(conflict_info)
                return new_path, True

        # Strategy 3: Use cluster ID as last resort
        new_path = folder_path.parent / f"{base_name}_cluster_{cluster.cluster_id}"
        conflict_info['resolution'] = 'cluster_id_fallback'
        conflict_info['final_path'] = str(new_path)
        self.conflict_log.append(conflict_info)

        return new_path, True

    def _create_folder_metadata(self, folder_path: Path, cluster: Any) -> None:
        """
        Create metadata file for the folder with cluster information.

        For junior developers:
        - Metadata files help with future processing and debugging
        - JSON is a good format for structured data
        - Hidden files (starting with .) keep metadata separate from user content
        """
        metadata = {
            'cluster_id': cluster.cluster_id,
            'suggested_name': cluster.suggested_name,
            'created_date': datetime.now().isoformat(),
            'temporal_info': {
                'start_time': cluster.temporal_info.start_time.isoformat(),
                'end_time': cluster.temporal_info.end_time.isoformat(),
                'duration_hours': cluster.duration_hours
            },
            'file_counts': {
                'total_files': cluster.size,
                'photos': cluster.photo_count,
                'videos': cluster.video_count
            },
            'location_info': {
                'has_gps': cluster.has_location,
                'dominant_location': cluster.dominant_location,
                'gps_count': len(cluster.gps_coordinates)
            },
            'confidence_score': cluster.confidence_score,
            'content_tags': cluster.content_tags,
            'people_detected': len(cluster.people_detected)
        }

        metadata_file = folder_path / ".cluster_metadata.json"

        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.debug(f"Created metadata file: {metadata_file}")
        except Exception as e:
            self.logger.warning(f"Failed to create metadata file {metadata_file}: {e}")

    def _create_year_organization_summary(self, year_organization: Dict[int, List[Dict]]) -> None:
        """Create summary files for year-based organization."""
        for year, folders in year_organization.items():
            year_dir = self.organized_dir / str(year)
            summary_file = year_dir / ".year_summary.json"

            summary = {
                'year': year,
                'total_folders': len(folders),
                'total_files': sum(f['file_count'] for f in folders),
                'folders': folders,
                'created_date': datetime.now().isoformat()
            }

            if not self.dry_run:
                try:
                    year_dir.mkdir(parents=True, exist_ok=True)
                    with open(summary_file, 'w') as f:
                        json.dump(summary, f, indent=2)
                    self.logger.debug(f"Created year summary: {summary_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to create year summary {summary_file}: {e}")

    def _generate_operation_report(self, folder_mapping: Dict[int, Path],
                                 year_organization: Dict[int, List[Dict]]) -> Dict[str, Any]:
        """Generate comprehensive operation report."""
        return {
            'operation_summary': self.operation_summary.copy(),
            'folder_mapping': {k: str(v) for k, v in folder_mapping.items()},
            'year_organization': year_organization,
            'conflict_log': self.conflict_log.copy(),
            'created_folders': [str(f) for f in self.created_folders],
            'base_directory': str(self.base_dir),
            'organized_directory': str(self.organized_dir),
            'dry_run': self.dry_run,
            'timestamp': datetime.now().isoformat()
        }

    def preview_folder_structure(self, clusters: List[Any]) -> str:
        """
        Generate a text preview of the folder structure that would be created.

        For junior developers:
        - Preview functions are invaluable for user confidence
        - Text output is easy to read and debug
        - Shows the user exactly what will happen before doing it
        """
        preview_lines = []
        preview_lines.append("📁 Folder Structure Preview")
        preview_lines.append("=" * 50)

        # Group by year
        year_groups = {}
        for cluster in clusters:
            year = cluster.temporal_info.start_time.year
            if year not in year_groups:
                year_groups[year] = []
            year_groups[year].append(cluster)

        total_folders = 0
        total_files = 0

        for year in sorted(year_groups.keys()):
            clusters_in_year = year_groups[year]
            year_files = sum(c.size for c in clusters_in_year)

            preview_lines.append(f"\n📅 {year} ({len(clusters_in_year)} folders, {year_files} files)")
            preview_lines.append("   │")

            for i, cluster in enumerate(clusters_in_year):
                is_last = (i == len(clusters_in_year) - 1)
                connector = "   └──" if is_last else "   ├──"

                clean_name = self._clean_folder_name(cluster.suggested_name or "Unnamed")
                duration = cluster.duration_hours
                confidence = cluster.confidence_score

                preview_lines.append(f"{connector} 📂 {clean_name}")
                preview_lines.append(f"   {'    ' if is_last else '   │   '}└── {cluster.size} files, {duration:.1f}h, confidence: {confidence:.2f}")

                total_folders += 1
                total_files += cluster.size

        preview_lines.append(f"\n📊 Summary:")
        preview_lines.append(f"   Total folders: {total_folders}")
        preview_lines.append(f"   Total files: {total_files}")
        preview_lines.append(f"   Output directory: {self.organized_dir}")

        return "\n".join(preview_lines)

    def get_operation_summary(self) -> Dict[str, Any]:
        """Get current operation summary."""
        return {
            'summary': self.operation_summary.copy(),
            'conflicts': len(self.conflict_log),
            'created_folders_count': len(self.created_folders),
            'dry_run_mode': self.dry_run
        }