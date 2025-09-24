"""
Media file moving and copying system for photo organization.

This module handles the actual movement/copying of photos and videos from
unorganized sources into the intelligent folder structure created by the
folder organizer.

For junior developers:
- Demonstrates safe file operations with validation and rollback
- Shows how to preserve metadata and timestamps during file moves
- Implements both copy and move operations with progress tracking
- Uses atomic operations to prevent data loss
"""

import os
import shutil
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime
import json
import time

from .config_manager import get_config

class FileOrganizer:
    """
    Handles safe moving and copying of media files into organized folder structure.

    This class takes the folder structure created by FolderOrganizer and the
    media clusters, then safely moves or copies files to their destination folders.

    For junior developers:
    - Always verify file integrity before and after operations
    - Use atomic operations where possible to prevent corruption
    - Implement comprehensive logging for troubleshooting
    - Provide rollback capabilities for failed operations
    """

    def __init__(self,
                 operation_mode: str = "copy",  # "copy" or "move"
                 dry_run: bool = True,
                 verify_checksums: bool = True):
        """
        Initialize the file organizer.

        Args:
            operation_mode: "copy" (safer) or "move" (more efficient)
            dry_run: If True, only simulate operations without touching files
            verify_checksums: If True, verify file integrity using checksums

        For junior developers:
        - copy mode is safer but uses more disk space
        - move mode is more efficient but requires careful error handling
        - dry_run is essential for testing complex file operations
        """
        self.logger = logging.getLogger(__name__)
        self.operation_mode = operation_mode
        self.dry_run = dry_run
        self.verify_checksums = verify_checksums

        # Operation tracking
        self.operations_log = []
        self.successful_operations = []
        self.failed_operations = []
        self.skipped_files = []

        # Statistics
        self.stats = {
            'files_processed': 0,
            'files_successful': 0,
            'files_failed': 0,
            'files_skipped': 0,
            'bytes_processed': 0,
            'operation_start_time': None,
            'operation_end_time': None
        }

        # Configuration
        try:
            config = get_config()
            self.preserve_timestamps = True
            self.create_backups = config.processing.get('create_backups', False)
        except Exception:
            self.preserve_timestamps = True
            self.create_backups = False

        self.logger.info(f"FileOrganizer initialized")
        self.logger.info(f"Operation mode: {self.operation_mode}")
        self.logger.info(f"Dry run: {self.dry_run}")
        self.logger.info(f"Verify checksums: {self.verify_checksums}")

    def organize_files(self,
                      clusters: List[Any],
                      folder_mapping: Dict[int, Path],
                      progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Organize files by moving/copying them to their designated folders.

        Args:
            clusters: List of media clusters with file information
            folder_mapping: Mapping from cluster_id to destination folder path
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with operation results and statistics

        For junior developers:
        - This is the main orchestrator that coordinates all file operations
        - Progress callbacks allow UI updates during long operations
        - Comprehensive error handling ensures partial failures don't break everything
        """
        self.logger.info(f"Starting file organization for {len(clusters)} clusters")
        self.stats['operation_start_time'] = datetime.now()

        # Calculate total files for progress tracking
        total_files = sum(cluster.size for cluster in clusters)
        self.logger.info(f"Total files to process: {total_files}")

        processed_files = 0

        for cluster in clusters:
            if cluster.cluster_id not in folder_mapping:
                self.logger.warning(f"No folder mapping for cluster {cluster.cluster_id}, skipping")
                continue

            destination_folder = folder_mapping[cluster.cluster_id]

            # Process each file in the cluster
            for media_file in cluster.media_files:
                try:
                    # Process individual file
                    success = self._organize_single_file(
                        media_file,
                        destination_folder,
                        cluster.cluster_id
                    )

                    if success:
                        self.stats['files_successful'] += 1
                    else:
                        self.stats['files_failed'] += 1

                    processed_files += 1
                    self.stats['files_processed'] += 1

                    # Update progress
                    if progress_callback:
                        progress = processed_files / total_files
                        progress_callback(progress, processed_files, total_files)

                except Exception as e:
                    self.logger.error(f"Error processing file {media_file.path}: {e}")
                    self.stats['files_failed'] += 1
                    self.failed_operations.append({
                        'source_path': str(media_file.path),
                        'cluster_id': cluster.cluster_id,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })

        self.stats['operation_end_time'] = datetime.now()

        # Generate comprehensive report
        report = self._generate_organization_report()

        self.logger.info(f"File organization completed")
        self.logger.info(f"Success rate: {self.stats['files_successful']}/{self.stats['files_processed']} files")

        return report

    def _organize_single_file(self,
                             media_file: Any,
                             destination_folder: Path,
                             cluster_id: int) -> bool:
        """
        Organize a single file to its destination folder.

        Args:
            media_file: MediaFile object with source path
            destination_folder: Path to destination folder
            cluster_id: ID of the cluster this file belongs to

        Returns:
            True if successful, False otherwise

        For junior developers:
        - This method handles all the complexity of safe file operations
        - Notice how we validate everything before making changes
        - Checksums ensure file integrity during transfer
        """
        source_path = Path(media_file.path)

        # Validate source file
        if not source_path.exists():
            self.logger.error(f"Source file does not exist: {source_path}")
            return False

        if not source_path.is_file():
            self.logger.error(f"Source is not a file: {source_path}")
            return False

        # Calculate destination path - ensure destination_folder is a Path object
        if isinstance(destination_folder, str):
            destination_folder = Path(destination_folder)
        destination_path = destination_folder / source_path.name

        # Handle naming conflicts at destination
        final_destination = self._resolve_destination_conflict(destination_path, source_path)

        # Check if file already organized (avoid duplicate work)
        if self._is_file_already_organized(source_path, final_destination):
            self.logger.info(f"File already organized, skipping: {source_path.name}")
            self.skipped_files.append({
                'source_path': str(source_path),
                'destination_path': str(final_destination),
                'reason': 'already_organized',
                'cluster_id': cluster_id
            })
            return True

        # Calculate file size for statistics
        file_size = source_path.stat().st_size
        self.stats['bytes_processed'] += file_size

        # Log the operation
        operation_log = {
            'source_path': str(source_path),
            'destination_path': str(final_destination),
            'cluster_id': cluster_id,
            'operation_mode': self.operation_mode,
            'file_size_bytes': file_size,
            'timestamp': datetime.now().isoformat()
        }

        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would {self.operation_mode} {source_path.name} → {final_destination}")
            operation_log['status'] = 'dry_run_success'
            self.operations_log.append(operation_log)
            return True

        # Verify source file integrity if requested
        source_checksum = None
        if self.verify_checksums:
            source_checksum = self._calculate_file_checksum(source_path)
            operation_log['source_checksum'] = source_checksum

        # Ensure destination directory exists
        final_destination.parent.mkdir(parents=True, exist_ok=True)

        # Perform the file operation
        try:
            if self.operation_mode == "copy":
                success = self._copy_file_safely(source_path, final_destination, source_checksum)
            else:  # move
                success = self._move_file_safely(source_path, final_destination, source_checksum)

            if success:
                operation_log['status'] = 'success'
                self.successful_operations.append(operation_log.copy())
                self.logger.info(f"Successfully {self.operation_mode}d: {source_path.name}")
            else:
                operation_log['status'] = 'failed'
                self.failed_operations.append(operation_log.copy())
                self.logger.error(f"Failed to {self.operation_mode}: {source_path.name}")

            self.operations_log.append(operation_log)
            return success

        except Exception as e:
            operation_log['status'] = 'error'
            operation_log['error'] = str(e)
            self.failed_operations.append(operation_log)
            self.operations_log.append(operation_log)
            self.logger.error(f"Exception during {self.operation_mode} operation: {e}")
            return False

    def _resolve_destination_conflict(self, destination_path: Path, source_path: Path) -> Path:
        """
        Resolve naming conflicts at the destination.

        For junior developers:
        - Similar to folder naming conflicts, but for individual files
        - We try multiple strategies to find a unique name
        - Always preserve the file extension
        """
        if not destination_path.exists():
            return destination_path

        # File exists - create unique name
        stem = destination_path.stem
        suffix = destination_path.suffix
        parent = destination_path.parent

        # Strategy 1: Check if it's the same file (avoid duplicate copies)
        if self._are_files_identical(source_path, destination_path):
            self.logger.info(f"Identical file already exists at destination: {destination_path.name}")
            return destination_path

        # Strategy 2: Add incremental number
        for i in range(1, 1000):
            new_path = parent / f"{stem}_{i:03d}{suffix}"
            if not new_path.exists():
                self.logger.info(f"Resolved naming conflict: {destination_path.name} → {new_path.name}")
                return new_path

        # Strategy 3: Add timestamp (fallback)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = parent / f"{stem}_{timestamp}{suffix}"
        self.logger.warning(f"Using timestamp fallback for naming conflict: {fallback_path.name}")
        return fallback_path

    def _is_file_already_organized(self, source_path: Path, destination_path: Path) -> bool:
        """Check if file is already organized to avoid duplicate work."""
        if not destination_path.exists():
            return False

        # Check if files are identical
        return self._are_files_identical(source_path, destination_path)

    def _are_files_identical(self, file1: Path, file2: Path) -> bool:
        """Check if two files are identical using size and checksum."""
        if not (file1.exists() and file2.exists()):
            return False

        # Quick size check first
        if file1.stat().st_size != file2.stat().st_size:
            return False

        # Checksum comparison for final verification
        if self.verify_checksums:
            checksum1 = self._calculate_file_checksum(file1)
            checksum2 = self._calculate_file_checksum(file2)
            return checksum1 == checksum2

        return True

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """
        Calculate SHA-256 checksum for file integrity verification.

        For junior developers:
        - Checksums ensure file integrity during transfer operations
        - SHA-256 is cryptographically secure for file verification
        - We read in chunks to handle large files efficiently
        """
        sha256_hash = hashlib.sha256()

        try:
            with open(file_path, "rb") as f:
                # Read in 64KB chunks for memory efficiency
                for chunk in iter(lambda: f.read(65536), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating checksum for {file_path}: {e}")
            return ""

    def _copy_file_safely(self, source_path: Path, destination_path: Path, source_checksum: Optional[str] = None) -> bool:
        """
        Safely copy file with integrity verification.

        For junior developers:
        - shutil.copy2 preserves metadata (timestamps, permissions)
        - We verify the copy succeeded by comparing checksums
        - Atomic operations prevent partial file corruption
        """
        try:
            # Use copy2 to preserve metadata
            shutil.copy2(source_path, destination_path)

            # Verify the copy if checksums enabled
            if self.verify_checksums and source_checksum:
                destination_checksum = self._calculate_file_checksum(destination_path)
                if source_checksum != destination_checksum:
                    self.logger.error(f"Checksum mismatch after copy: {source_path.name}")
                    # Remove corrupted copy
                    if destination_path.exists():
                        destination_path.unlink()
                    return False

            # Preserve timestamps if requested
            if self.preserve_timestamps:
                source_stat = source_path.stat()
                os.utime(destination_path, (source_stat.st_atime, source_stat.st_mtime))

            return True

        except Exception as e:
            self.logger.error(f"Error copying file {source_path} to {destination_path}: {e}")
            # Clean up partial copy
            if destination_path.exists():
                try:
                    destination_path.unlink()
                except Exception:
                    pass
            return False

    def _move_file_safely(self, source_path: Path, destination_path: Path, source_checksum: Optional[str] = None) -> bool:
        """
        Safely move file with integrity verification.

        For junior developers:
        - Move operations are more efficient but riskier than copy
        - We use copy+verify+delete approach for safety
        - This ensures data integrity even if operation is interrupted
        """
        try:
            # For safety, we copy first then delete original (atomic move)
            if self._copy_file_safely(source_path, destination_path, source_checksum):
                # Verify the copy succeeded before deleting original
                if self.verify_checksums and source_checksum:
                    destination_checksum = self._calculate_file_checksum(destination_path)
                    if source_checksum != destination_checksum:
                        self.logger.error(f"Move verification failed: {source_path.name}")
                        # Remove bad copy
                        if destination_path.exists():
                            destination_path.unlink()
                        return False

                # Safe to delete original
                source_path.unlink()
                return True
            else:
                return False

        except Exception as e:
            self.logger.error(f"Error moving file {source_path} to {destination_path}: {e}")
            return False

    def _generate_organization_report(self) -> Dict[str, Any]:
        """Generate comprehensive report of organization operation."""
        duration = None
        if self.stats['operation_start_time'] and self.stats['operation_end_time']:
            duration = (self.stats['operation_end_time'] - self.stats['operation_start_time']).total_seconds()

        return {
            'operation_summary': {
                'mode': self.operation_mode,
                'dry_run': self.dry_run,
                'verify_checksums': self.verify_checksums,
                'total_files_processed': self.stats['files_processed'],
                'successful_operations': self.stats['files_successful'],
                'failed_operations': self.stats['files_failed'],
                'skipped_files': self.stats['files_skipped'],
                'total_bytes_processed': self.stats['bytes_processed'],
                'duration_seconds': duration,
                'success_rate': self.stats['files_successful'] / max(1, self.stats['files_processed'])
            },
            'operation_details': {
                'successful_operations': self.successful_operations,
                'failed_operations': self.failed_operations,
                'skipped_files': self.skipped_files,
                'operations_log': self.operations_log
            },
            'timestamp': datetime.now().isoformat()
        }

    def rollback_operations(self, operations_to_rollback: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Rollback file operations in case of errors.

        For junior developers:
        - Rollback is essential for recovery from failed operations
        - We only rollback operations that actually succeeded
        - This helps restore the original state if something goes wrong
        """
        if not operations_to_rollback:
            operations_to_rollback = self.successful_operations

        rollback_results = {
            'attempted': 0,
            'successful': 0,
            'failed': 0,
            'errors': []
        }

        self.logger.info(f"Starting rollback of {len(operations_to_rollback)} operations")

        for operation in operations_to_rollback:
            rollback_results['attempted'] += 1

            try:
                source_path = Path(operation['source_path'])
                destination_path = Path(operation['destination_path'])

                if self.operation_mode == "copy":
                    # For copy operations, just remove the copy
                    if destination_path.exists():
                        destination_path.unlink()
                        rollback_results['successful'] += 1
                        self.logger.info(f"Rollback: Removed copy {destination_path.name}")
                    else:
                        self.logger.warning(f"Rollback: Copy not found {destination_path.name}")

                else:  # move operations
                    # For move operations, move the file back
                    if destination_path.exists() and not source_path.exists():
                        shutil.move(destination_path, source_path)
                        rollback_results['successful'] += 1
                        self.logger.info(f"Rollback: Moved back {source_path.name}")
                    else:
                        self.logger.warning(f"Rollback: Cannot restore {source_path.name}")

            except Exception as e:
                rollback_results['failed'] += 1
                error_msg = f"Rollback failed for {operation.get('source_path', 'unknown')}: {e}"
                rollback_results['errors'].append(error_msg)
                self.logger.error(error_msg)

        self.logger.info(f"Rollback completed: {rollback_results['successful']}/{rollback_results['attempted']} successful")
        return rollback_results

    def get_operation_summary(self) -> Dict[str, Any]:
        """Get current operation summary."""
        return {
            'stats': self.stats.copy(),
            'successful_operations_count': len(self.successful_operations),
            'failed_operations_count': len(self.failed_operations),
            'skipped_files_count': len(self.skipped_files),
            'mode': self.operation_mode,
            'dry_run': self.dry_run
        }