"""
Media file detection and parsing utilities.

This module handles the detection and parsing of iPhone media files that follow
the standard naming convention: IMG_YYYYMMDD_HHMMSS.ext

For junior developers:
- Uses regex (regular expressions) to parse filenames
- Path class is the modern way to handle file paths in Python
- dataclass decorator automatically creates __init__, __repr__, etc.
- Type hints (List, Optional, Tuple) help catch bugs and improve readability
"""

# Standard library imports
import re                    # Regular expressions for pattern matching
import os                    # Operating system interface
from pathlib import Path     # Modern path handling
from datetime import datetime  # Date and time operations
from typing import List, Optional, Tuple  # Type hints for better code
from dataclasses import dataclass  # Automatic class generation

# Import our configuration settings
from .config import FILENAME_PATTERN, SUPPORTED_EXTENSIONS, IPHONE_AUTOMATIC_DIR, PICTURES_DIR

@dataclass
class MediaFile:
    """
    Represents a media file with parsed metadata.

    This is a data container that holds all the information we extract
    from a media file. Using @dataclass automatically creates __init__,
    __repr__, and comparison methods for us.

    For junior developers:
    - path: Full file path (e.g., /Users/name/Photos/IMG_20241024_143000.JPG)
    - filename: Just the filename (e.g., IMG_20241024_143000.JPG)
    - date/time: When the photo was taken (parsed from filename)
    - extension: File extension (.jpg, .mov, etc.)
    - file_type: 'photo' or 'video' (determined from extension)
    - size: File size in bytes
    """
    path: Path          # Full path to the file
    filename: str       # Just the filename part
    date: datetime      # When photo was taken
    time: datetime      # Same as date (kept for backward compatibility)
    extension: str      # File extension (.jpg, .mov, etc.)
    file_type: str      # 'photo' or 'video'
    size: int           # File size in bytes

class MediaDetector:
    """
    Detects and parses media files in the specified directories.

    This class is responsible for finding iPhone photos and videos that follow
    the standard naming pattern and extracting useful information from them.

    For junior developers:
    - This uses the "composition" design pattern - it has methods that work together
    - Regular expressions (regex) are used to parse filenames
    - We return Optional types when operations might fail
    - The class stores a compiled regex for efficiency
    """

    def __init__(self):
        """
        Initialize the MediaDetector.

        We compile the regex pattern once during initialization for performance.
        Compiling regex patterns is expensive, so we do it once and reuse it.
        """
        # Compile the regex pattern from our config
        # re.IGNORECASE makes it work with both .jpg and .JPG
        self.filename_regex = re.compile(FILENAME_PATTERN, re.IGNORECASE)

    def parse_filename(self, filename: str) -> Optional[Tuple[datetime, str]]:
        """
        Parse filename to extract date/time and extension.

        iPhone photos follow this pattern: IMG_20241024_143000.JPG
        - IMG_ is the prefix
        - 20241024 is the date (YYYYMMDD)
        - 143000 is the time (HHMMSS)
        - .JPG is the extension

        Args:
            filename: Filename in format IMG_YYYYMMDD_HHMMSS.ext

        Returns:
            Tuple of (datetime, extension) or None if parsing fails

        For junior developers:
        - Optional[Tuple[...]] means this might return None
        - We use regex groups to extract parts of the filename
        - String slicing [4:6] gets characters 4 and 5
        - ValueError happens if date/time is invalid (like month 99)
        """
        # Try to match our regex pattern against the filename
        match = self.filename_regex.match(filename)
        if not match:
            # Filename doesn't match expected pattern
            return None

        # Extract the captured groups from regex
        # Groups are: (date_string, time_string, extension)
        date_str, time_str, ext = match.groups()

        try:
            # Parse date components from YYYYMMDD format
            year = int(date_str[:4])    # First 4 characters: YYYY
            month = int(date_str[4:6])  # Next 2 characters: MM
            day = int(date_str[6:8])    # Last 2 characters: DD

            # Parse time components from HHMMSS format
            hour = int(time_str[:2])    # First 2 characters: HH
            minute = int(time_str[2:4]) # Next 2 characters: MM
            second = int(time_str[4:6]) # Last 2 characters: SS

            # Create datetime object
            dt = datetime(year, month, day, hour, minute, second)
            return dt, ext.lower()  # Return tuple with lowercase extension
        except ValueError:
            # Invalid date/time values (like month 99, hour 25, etc.)
            return None

    def get_file_type(self, extension: str) -> str:
        """Determine if file is photo or video based on extension."""
        ext_lower = extension.lower()
        # Add dot if not present
        if not ext_lower.startswith('.'):
            ext_lower = '.' + ext_lower

        if ext_lower in {'.jpg', '.jpeg', '.png'}:
            return 'photo'
        elif ext_lower in {'.mov', '.mp4', '.avi'}:
            return 'video'
        else:
            return 'unknown'

    def scan_directory(self, directory: Path) -> List[MediaFile]:
        """Scan directory for media files matching the expected pattern.

        Args:
            directory: Directory path to scan

        Returns:
            List of MediaFile objects
        """
        media_files = []

        if not directory.exists():
            return media_files

        for file_path in directory.iterdir():
            if not file_path.is_file():
                continue

            filename = file_path.name
            parsed = self.parse_filename(filename)

            if parsed is None:
                continue

            dt, extension = parsed
            file_type = self.get_file_type(extension)

            if file_type == 'unknown':
                continue

            try:
                file_size = file_path.stat().st_size
            except OSError:
                continue

            media_file = MediaFile(
                path=file_path,
                filename=filename,
                date=dt.date(),
                time=dt,
                extension=extension,
                file_type=file_type,
                size=file_size
            )

            media_files.append(media_file)

        return sorted(media_files, key=lambda x: x.time)

    def scan_iphone_automatic(self) -> List[MediaFile]:
        """Scan iPhone Automatic folder for unprocessed media files."""
        return self.scan_directory(IPHONE_AUTOMATIC_DIR)

    def scan_pictures_library(self) -> List[MediaFile]:
        """Recursively scan Pictures library for existing organized media files."""
        media_files = []

        if not PICTURES_DIR.exists():
            return media_files

        # Recursively scan all subdirectories
        for root, dirs, files in os.walk(PICTURES_DIR):
            root_path = Path(root)

            for filename in files:
                file_path = root_path / filename
                parsed = self.parse_filename(filename)

                if parsed is None:
                    continue

                dt, extension = parsed
                file_type = self.get_file_type(extension)

                if file_type == 'unknown':
                    continue

                try:
                    file_size = file_path.stat().st_size
                except OSError:
                    continue

                # Extract event folder info from path
                relative_path = file_path.relative_to(PICTURES_DIR)
                event_folder = None
                if len(relative_path.parts) >= 2:
                    event_folder = relative_path.parts[1]  # Year/Event folder

                media_file = MediaFile(
                    path=file_path,
                    filename=filename,
                    date=dt.date(),
                    time=dt,
                    extension=extension,
                    file_type=file_type,
                    size=file_size
                )

                # Add event folder info as attribute
                media_file.event_folder = event_folder
                media_files.append(media_file)

        return sorted(media_files, key=lambda x: x.time)

    def get_media_stats(self, media_files: List[MediaFile]) -> dict:
        """Get statistics about media files."""
        total_files = len(media_files)
        photos = sum(1 for f in media_files if f.file_type == 'photo')
        videos = sum(1 for f in media_files if f.file_type == 'video')
        total_size = sum(f.size for f in media_files)

        if total_files > 0:
            date_range = (
                min(f.date for f in media_files),
                max(f.date for f in media_files)
            )
        else:
            date_range = (None, None)

        return {
            'total_files': total_files,
            'photos': photos,
            'videos': videos,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'date_range': date_range
        }