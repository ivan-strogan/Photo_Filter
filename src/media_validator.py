"""
Media file validation and corruption detection system.

This module provides comprehensive validation for photo and video files,
detecting corruption, unsupported formats, and other file integrity issues.

For junior developers:
- Shows how to validate media files without loading entire file into memory
- Demonstrates defensive programming for handling real-world file issues
- Implements comprehensive file format validation
- Uses magic numbers for reliable file type detection
"""

import os
import logging
import mimetypes
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import struct
import hashlib

# Optional PIL dependency for advanced image validation
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class MediaValidationResult:
    """
    Result of media file validation.

    For junior developers:
    - Data classes are perfect for structured validation results
    - Boolean flags make it easy to check validation status
    - Error messages provide actionable feedback for debugging
    """

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.is_valid = True
        self.is_supported = True
        self.file_type = None  # 'photo', 'video', 'unknown'
        self.detected_format = None  # 'JPEG', 'PNG', 'MP4', etc.
        self.file_size_bytes = 0
        self.corruption_detected = False
        self.validation_errors = []
        self.validation_warnings = []
        self.validation_details = {}

    def add_error(self, error_message: str, error_type: str = "general"):
        """Add validation error."""
        self.validation_errors.append({
            'message': error_message,
            'type': error_type,
            'timestamp': datetime.now().isoformat()
        })
        self.is_valid = False

    def add_warning(self, warning_message: str, warning_type: str = "general"):
        """Add validation warning."""
        self.validation_warnings.append({
            'message': warning_message,
            'type': warning_type,
            'timestamp': datetime.now().isoformat()
        })

    def get_summary(self) -> str:
        """Get human-readable validation summary."""
        if self.is_valid and self.is_supported:
            return f"✅ Valid {self.detected_format} {self.file_type}"
        elif not self.is_supported:
            return f"⚠️  Unsupported {self.detected_format or 'format'}"
        elif self.corruption_detected:
            return f"❌ Corrupted {self.detected_format or 'file'}"
        else:
            return f"❌ Invalid file ({len(self.validation_errors)} errors)"

class MediaValidator:
    """
    Comprehensive media file validator for photos and videos.

    This class validates media files for corruption, format support,
    and other integrity issues before processing.

    For junior developers:
    - Magic number validation is more reliable than file extensions
    - Memory-efficient validation doesn't load entire files
    - Comprehensive logging helps debug file issues
    """

    # Magic number signatures for file format detection
    MAGIC_NUMBERS = {
        # Image formats
        b'\xFF\xD8\xFF': 'JPEG',
        b'\x89PNG\r\n\x1a\n': 'PNG',
        b'GIF87a': 'GIF87a',
        b'GIF89a': 'GIF89a',
        b'RIFF': 'WEBP',  # Need to check for WEBP after RIFF
        b'BM': 'BMP',
        b'II*\x00': 'TIFF_LE',  # Little endian TIFF
        b'MM\x00*': 'TIFF_BE',  # Big endian TIFF

        # Video formats
        b'\x00\x00\x00\x14ftyp': 'MP4',
        b'\x00\x00\x00\x18ftyp': 'MP4',
        b'\x00\x00\x00\x1cftyp': 'MP4',
        b'\x00\x00\x00 ftyp': 'MP4',
        b'RIFF': 'AVI',  # Need additional check
        b'\x1aE\xdf\xa3': 'WEBM',
        b'fLaC': 'FLAC',
        b'OggS': 'OGG',
    }

    SUPPORTED_IMAGE_FORMATS = {
        'JPEG', 'PNG', 'GIF87a', 'GIF89a', 'BMP', 'TIFF_LE', 'TIFF_BE'
    }

    SUPPORTED_VIDEO_FORMATS = {
        'MP4', 'AVI', 'WEBM', 'MOV'
    }

    def __init__(self, enable_deep_validation: bool = True):
        """
        Initialize media validator.

        Args:
            enable_deep_validation: Whether to perform deep file structure validation
        """
        self.logger = logging.getLogger(__name__)
        self.enable_deep_validation = enable_deep_validation

        # Validation statistics
        self.validation_stats = {
            'files_validated': 0,
            'valid_files': 0,
            'corrupted_files': 0,
            'unsupported_files': 0,
            'validation_errors': 0
        }

        self.logger.info(f"MediaValidator initialized (deep validation: {enable_deep_validation})")

    def validate_media_file(self, file_path: Union[str, Path]) -> MediaValidationResult:
        """
        Perform comprehensive validation of a media file.

        Args:
            file_path: Path to the media file to validate

        Returns:
            MediaValidationResult with validation details

        For junior developers:
        - This is the main entry point for file validation
        - Each validation step builds on the previous ones
        - Early returns prevent unnecessary processing of invalid files
        """
        file_path = Path(file_path)
        result = MediaValidationResult(file_path)

        self.validation_stats['files_validated'] += 1

        try:
            # Step 1: Basic file existence and access validation
            if not self._validate_file_existence(file_path, result):
                return result

            # Step 2: File size validation
            if not self._validate_file_size(file_path, result):
                return result

            # Step 3: File format detection and validation
            if not self._validate_file_format(file_path, result):
                return result

            # Step 4: Deep file structure validation (optional)
            if self.enable_deep_validation:
                self._validate_file_structure(file_path, result)

            # Step 5: Security validation (basic checks)
            self._validate_file_security(file_path, result)

            # Update statistics
            if result.is_valid:
                if result.is_supported:
                    self.validation_stats['valid_files'] += 1
                else:
                    self.validation_stats['unsupported_files'] += 1
            else:
                if result.corruption_detected:
                    self.validation_stats['corrupted_files'] += 1
                else:
                    self.validation_stats['validation_errors'] += 1

            self.logger.debug(f"Validation completed for {file_path.name}: {result.get_summary()}")
            return result

        except Exception as e:
            result.add_error(f"Validation exception: {e}", "exception")
            self.logger.error(f"Exception validating {file_path}: {e}")
            return result

    def _validate_file_existence(self, file_path: Path, result: MediaValidationResult) -> bool:
        """Validate basic file existence and accessibility."""
        if not file_path.exists():
            result.add_error("File does not exist", "file_access")
            return False

        if not file_path.is_file():
            result.add_error("Path is not a regular file", "file_access")
            return False

        try:
            # Test read access
            with open(file_path, 'rb') as f:
                f.read(1)
        except PermissionError:
            result.add_error("Permission denied reading file", "file_access")
            return False
        except Exception as e:
            result.add_error(f"Cannot read file: {e}", "file_access")
            return False

        return True

    def _validate_file_size(self, file_path: Path, result: MediaValidationResult) -> bool:
        """Validate file size constraints."""
        try:
            file_size = file_path.stat().st_size
            result.file_size_bytes = file_size
            result.validation_details['file_size_bytes'] = file_size

            # Check for empty files
            if file_size == 0:
                result.add_error("File is empty (0 bytes)", "file_size")
                return False

            # Check for suspiciously small files
            if file_size < 100:  # Less than 100 bytes is suspicious for media
                result.add_warning("File is very small for media content", "file_size")

            # Check for extremely large files (>1GB)
            if file_size > 1024 * 1024 * 1024:
                result.add_warning("File is very large (>1GB)", "file_size")

            return True

        except Exception as e:
            result.add_error(f"Cannot determine file size: {e}", "file_size")
            return False

    def _validate_file_format(self, file_path: Path, result: MediaValidationResult) -> bool:
        """Validate file format using magic numbers and extension."""
        try:
            # Read first 32 bytes for magic number detection
            with open(file_path, 'rb') as f:
                header = f.read(32)

            if len(header) < 8:
                result.add_error("File too small to determine format", "format")
                return False

            # Detect format using magic numbers
            detected_format = self._detect_format_from_magic(header)

            if not detected_format:
                # Fallback to extension-based detection
                detected_format = self._detect_format_from_extension(file_path)
                if detected_format:
                    result.add_warning("Format detected from extension only (no magic number match)", "format")

            if detected_format:
                result.detected_format = detected_format
                result.validation_details['detected_format'] = detected_format

                # Determine file type (photo/video)
                if detected_format in self.SUPPORTED_IMAGE_FORMATS:
                    result.file_type = 'photo'
                elif detected_format in self.SUPPORTED_VIDEO_FORMATS:
                    result.file_type = 'video'
                else:
                    result.file_type = 'unknown'
                    result.is_supported = False
                    result.add_warning(f"Unsupported format: {detected_format}", "format")

                # Check format support
                if detected_format not in (self.SUPPORTED_IMAGE_FORMATS | self.SUPPORTED_VIDEO_FORMATS):
                    result.is_supported = False
                    result.add_error(f"Unsupported media format: {detected_format}", "format")

            else:
                result.add_error("Unknown or corrupted file format", "format")
                result.file_type = 'unknown'
                result.is_supported = False
                return False

            return True

        except Exception as e:
            result.add_error(f"Format validation failed: {e}", "format")
            return False

    def _detect_format_from_magic(self, header: bytes) -> Optional[str]:
        """Detect file format from magic number in header."""
        for magic_bytes, format_name in self.MAGIC_NUMBERS.items():
            if header.startswith(magic_bytes):
                # Special handling for RIFF containers
                if magic_bytes == b'RIFF' and len(header) >= 12:
                    # Check RIFF subtype
                    riff_type = header[8:12]
                    if riff_type == b'WEBP':
                        return 'WEBP'
                    elif riff_type == b'AVI ':
                        return 'AVI'
                    else:
                        return 'RIFF_UNKNOWN'
                return format_name

        return None

    def _detect_format_from_extension(self, file_path: Path) -> Optional[str]:
        """Fallback format detection from file extension."""
        extension = file_path.suffix.lower()

        extension_map = {
            '.jpg': 'JPEG', '.jpeg': 'JPEG',
            '.png': 'PNG',
            '.gif': 'GIF89a',
            '.bmp': 'BMP',
            '.tiff': 'TIFF_LE', '.tif': 'TIFF_LE',
            '.mp4': 'MP4',
            '.avi': 'AVI',
            '.mov': 'MOV',
            '.webm': 'WEBM'
        }

        return extension_map.get(extension)

    def _validate_file_structure(self, file_path: Path, result: MediaValidationResult) -> None:
        """Perform deep validation of file structure."""
        if result.file_type == 'photo' and PIL_AVAILABLE:
            self._validate_image_structure(file_path, result)
        elif result.file_type == 'video':
            self._validate_video_structure(file_path, result)

    def _validate_image_structure(self, file_path: Path, result: MediaValidationResult) -> None:
        """Validate image file structure using PIL."""
        try:
            with Image.open(file_path) as img:
                # Basic PIL validation
                img.verify()  # Verify the image is not corrupted

                # Re-open for additional checks (verify() closes the image)
                with Image.open(file_path) as img2:
                    width, height = img2.size
                    result.validation_details['image_width'] = width
                    result.validation_details['image_height'] = height
                    result.validation_details['image_mode'] = img2.mode

                    # Check for reasonable dimensions
                    if width < 10 or height < 10:
                        result.add_warning("Image dimensions are very small", "image_structure")

                    if width > 50000 or height > 50000:
                        result.add_warning("Image dimensions are very large", "image_structure")

        except Exception as e:
            result.add_error(f"Image structure validation failed: {e}", "image_structure")
            result.corruption_detected = True

    def _validate_video_structure(self, file_path: Path, result: MediaValidationResult) -> None:
        """Basic video file structure validation."""
        try:
            # Basic MP4/MOV validation by checking atom structure
            if result.detected_format in ['MP4', 'MOV']:
                self._validate_mp4_structure(file_path, result)

        except Exception as e:
            result.add_error(f"Video structure validation failed: {e}", "video_structure")
            result.corruption_detected = True

    def _validate_mp4_structure(self, file_path: Path, result: MediaValidationResult) -> None:
        """Validate MP4/MOV file atom structure."""
        try:
            with open(file_path, 'rb') as f:
                # Check for basic MP4 atoms
                f.seek(0)
                header = f.read(32)

                if b'ftyp' not in header:
                    result.add_error("MP4 file missing required ftyp atom", "video_structure")
                    result.corruption_detected = True
                    return

                # Basic atom parsing
                f.seek(0)
                atoms_found = []
                bytes_read = 0
                max_read = min(file_path.stat().st_size, 1024 * 1024)  # Read max 1MB for validation

                while bytes_read < max_read:
                    try:
                        atom_size_data = f.read(4)
                        if len(atom_size_data) < 4:
                            break

                        atom_size = struct.unpack('>I', atom_size_data)[0]
                        atom_type = f.read(4)

                        if len(atom_type) < 4:
                            break

                        atoms_found.append(atom_type.decode('ascii', errors='ignore'))

                        if atom_size == 0:  # Atom extends to end of file
                            break
                        elif atom_size < 8:  # Invalid atom size
                            result.add_warning("Invalid MP4 atom size encountered", "video_structure")
                            break

                        # Skip to next atom
                        f.seek(f.tell() + atom_size - 8)
                        bytes_read += atom_size

                    except (struct.error, UnicodeDecodeError, OSError):
                        break

                result.validation_details['mp4_atoms_found'] = atoms_found

                # Check for essential atoms
                if 'ftyp' not in atoms_found:
                    result.add_error("MP4 missing ftyp atom", "video_structure")
                    result.corruption_detected = True

        except Exception as e:
            result.add_warning(f"MP4 structure check incomplete: {e}", "video_structure")

    def _validate_file_security(self, file_path: Path, result: MediaValidationResult) -> None:
        """Basic security validation."""
        # Check for suspicious file names
        filename = file_path.name.lower()

        suspicious_patterns = ['.exe', '.scr', '.bat', '.cmd', '.vbs', '.js']
        if any(pattern in filename for pattern in suspicious_patterns):
            result.add_error("Suspicious executable content in filename", "security")
            return

        # Check file size vs format consistency
        if result.file_type == 'photo' and result.file_size_bytes > 100 * 1024 * 1024:  # >100MB
            result.add_warning("Photo file unusually large for format", "security")

    def validate_batch(self, file_paths: List[Path],
                      progress_callback: Optional[callable] = None) -> Dict[str, MediaValidationResult]:
        """
        Validate multiple media files efficiently.

        Args:
            file_paths: List of file paths to validate
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping file paths to validation results
        """
        results = {}
        total_files = len(file_paths)

        self.logger.info(f"Starting batch validation of {total_files} files")

        for i, file_path in enumerate(file_paths):
            try:
                result = self.validate_media_file(file_path)
                results[str(file_path)] = result

                # Progress callback
                if progress_callback:
                    progress = (i + 1) / total_files
                    progress_callback(progress, i + 1, total_files)

            except Exception as e:
                self.logger.error(f"Batch validation error for {file_path}: {e}")
                # Create error result
                error_result = MediaValidationResult(Path(file_path))
                error_result.add_error(f"Batch validation exception: {e}", "batch_error")
                results[str(file_path)] = error_result

        self.logger.info(f"Batch validation completed: {len(results)} results")
        return results

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation statistics summary."""
        total_files = self.validation_stats['files_validated']

        return {
            'total_files_validated': total_files,
            'valid_files': self.validation_stats['valid_files'],
            'corrupted_files': self.validation_stats['corrupted_files'],
            'unsupported_files': self.validation_stats['unsupported_files'],
            'validation_errors': self.validation_stats['validation_errors'],
            'success_rate': self.validation_stats['valid_files'] / max(1, total_files),
            'corruption_rate': self.validation_stats['corrupted_files'] / max(1, total_files),
            'support_rate': (self.validation_stats['valid_files'] + self.validation_stats['corrupted_files']) / max(1, total_files)
        }

    def filter_valid_files(self, validation_results: Dict[str, MediaValidationResult]) -> List[Path]:
        """Filter validation results to return only valid, supported files."""
        valid_files = []

        for file_path, result in validation_results.items():
            if result.is_valid and result.is_supported:
                valid_files.append(Path(file_path))

        return valid_files

    def filter_corrupted_files(self, validation_results: Dict[str, MediaValidationResult]) -> List[Path]:
        """Filter validation results to return only corrupted files."""
        corrupted_files = []

        for file_path, result in validation_results.items():
            if result.corruption_detected:
                corrupted_files.append(Path(file_path))

        return corrupted_files