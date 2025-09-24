"""Logging and progress tracking utilities."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from tqdm import tqdm
import json

from .config import LOGS_DIR

class ProgressTracker:
    """Tracks progress for long-running operations."""

    def __init__(self, total: int, description: str = "Processing"):
        """Initialize progress tracker.

        Args:
            total: Total number of items to process
            description: Description of the operation
        """
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = datetime.now()
        self.progress_bar = tqdm(
            total=total,
            desc=description,
            unit="items",
            ncols=100
        )

    def update(self, increment: int = 1, description: Optional[str] = None) -> None:
        """Update progress.

        Args:
            increment: Number of items processed
            description: Optional description update
        """
        self.current += increment
        self.progress_bar.update(increment)

        if description:
            self.progress_bar.set_description(description)

    def set_description(self, description: str) -> None:
        """Update the progress description."""
        self.progress_bar.set_description(description)

    def close(self) -> Dict[str, Any]:
        """Close progress tracker and return summary.

        Returns:
            Dictionary with operation summary
        """
        self.progress_bar.close()
        end_time = datetime.now()
        duration = end_time - self.start_time

        return {
            'description': self.description,
            'total_items': self.total,
            'processed_items': self.current,
            'completion_rate': self.current / self.total if self.total > 0 else 0,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'items_per_second': self.current / duration.total_seconds() if duration.total_seconds() > 0 else 0
        }

class PhotoFilterLogger:
    """Centralized logging system for Photo Filter application."""

    def __init__(self, log_level: str = "INFO", log_file: Optional[Path] = None):
        """Initialize the logger.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_file: Optional file path for log output
        """
        self.log_level = getattr(logging, log_level.upper())
        # Use logs directory from config, ensure it exists
        LOGS_DIR.mkdir(exist_ok=True)
        self.log_file = log_file or (LOGS_DIR / "photo_filter.log")
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._setup_logging()

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        # Create custom formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)

        # Clear existing handlers
        root_logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(self.log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Log session start
        self.log_session_start()

    def log_session_start(self) -> None:
        """Log the start of a new session."""
        logger = logging.getLogger("PhotoFilter")
        logger.info(f"{'='*60}")
        logger.info(f"Starting Photo Filter session: {self.session_id}")
        logger.info(f"Log level: {logging.getLevelName(self.log_level)}")
        logger.info(f"Log file: {self.log_file}")
        logger.info(f"{'='*60}")

    def log_operation_start(self, operation: str, details: Dict[str, Any] = None) -> None:
        """Log the start of a major operation.

        Args:
            operation: Name of the operation
            details: Optional details about the operation
        """
        logger = logging.getLogger("PhotoFilter")
        logger.info(f"Starting operation: {operation}")

        if details:
            for key, value in details.items():
                logger.info(f"  {key}: {value}")

    def log_operation_end(self, operation: str, results: Dict[str, Any] = None) -> None:
        """Log the end of a major operation.

        Args:
            operation: Name of the operation
            results: Optional results from the operation
        """
        logger = logging.getLogger("PhotoFilter")
        logger.info(f"Completed operation: {operation}")

        if results:
            for key, value in results.items():
                logger.info(f"  {key}: {value}")

    def log_error_with_context(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """Log an error with additional context.

        Args:
            error: Exception that occurred
            context: Additional context information
        """
        logger = logging.getLogger("PhotoFilter")
        logger.error(f"Error occurred: {type(error).__name__}: {error}")

        if context:
            logger.error("Error context:")
            for key, value in context.items():
                logger.error(f"  {key}: {value}")

    def create_progress_tracker(self, total: int, description: str) -> ProgressTracker:
        """Create a progress tracker for an operation.

        Args:
            total: Total number of items to process
            description: Description of the operation

        Returns:
            ProgressTracker instance
        """
        logger = logging.getLogger("PhotoFilter")
        logger.info(f"Starting progress tracking: {description} ({total} items)")

        return ProgressTracker(total, description)

    def log_clustering_results(self, results: Dict[str, Any]) -> None:
        """Log clustering results in a structured format.

        Args:
            results: Clustering results dictionary
        """
        logger = logging.getLogger("PhotoFilter.Clustering")

        logger.info("Clustering Results Summary:")
        logger.info(f"  Total clusters: {results.get('total_clusters', 0)}")
        logger.info(f"  Total files: {results.get('total_files', 0)}")
        logger.info(f"  Average confidence: {results.get('avg_confidence', 0):.3f}")
        logger.info(f"  Location coverage: {results.get('location_coverage', 0):.1%}")
        logger.info(f"  Average cluster size: {results.get('avg_cluster_size', 0):.1f}")
        logger.info(f"  Average duration: {results.get('avg_duration_hours', 0):.2f} hours")

        # Quality distribution
        quality_dist = results.get('quality_distribution', {})
        if quality_dist:
            logger.info("Quality distribution:")
            logger.info(f"  High confidence: {quality_dist.get('high_confidence', 0)}")
            logger.info(f"  Medium confidence: {quality_dist.get('medium_confidence', 0)}")
            logger.info(f"  Low confidence: {quality_dist.get('low_confidence', 0)}")

    def log_file_processing_stats(self, stats: Dict[str, Any]) -> None:
        """Log file processing statistics.

        Args:
            stats: File processing statistics
        """
        logger = logging.getLogger("PhotoFilter.FileProcessing")

        logger.info("File Processing Statistics:")
        logger.info(f"  Total files scanned: {stats.get('total_files', 0)}")
        logger.info(f"  Photos: {stats.get('photos', 0)}")
        logger.info(f"  Videos: {stats.get('videos', 0)}")
        logger.info(f"  Total size: {stats.get('total_size_mb', 0):.1f} MB")

        date_range = stats.get('date_range', (None, None))
        if date_range[0] and date_range[1]:
            logger.info(f"  Date range: {date_range[0]} to {date_range[1]}")

    def save_session_report(self, report_data: Dict[str, Any]) -> Path:
        """Save a detailed session report to file.

        Args:
            report_data: Session report data

        Returns:
            Path to the saved report file
        """
        report_file = Path(f"photo_filter_report_{self.session_id}.json")

        report_data['session_info'] = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'log_file': str(self.log_file),
            'log_level': logging.getLevelName(self.log_level)
        }

        try:
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)

            logger = logging.getLogger("PhotoFilter")
            logger.info(f"Session report saved: {report_file}")

        except Exception as e:
            logger = logging.getLogger("PhotoFilter")
            logger.error(f"Failed to save session report: {e}")

        return report_file

    def get_log_summary(self) -> Dict[str, Any]:
        """Get a summary of the current logging session.

        Returns:
            Dictionary with logging session summary
        """
        log_size = 0
        if self.log_file.exists():
            log_size = self.log_file.stat().st_size

        return {
            'session_id': self.session_id,
            'log_level': logging.getLevelName(self.log_level),
            'log_file': str(self.log_file),
            'log_file_size_bytes': log_size,
            'start_time': self.session_id  # Session ID contains start time
        }

# Global logger instance
photo_filter_logger = None

def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> PhotoFilterLogger:
    """Set up global logging for the application.

    Args:
        log_level: Logging level
        log_file: Optional log file path

    Returns:
        PhotoFilterLogger instance
    """
    global photo_filter_logger
    photo_filter_logger = PhotoFilterLogger(log_level, log_file)
    return photo_filter_logger

def get_logger() -> Optional[PhotoFilterLogger]:
    """Get the global logger instance.

    Returns:
        PhotoFilterLogger instance or None if not set up
    """
    return photo_filter_logger