"""Media metadata extraction utilities."""

import exifread
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import cv2
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
import logging

from .media_detector import MediaFile

class MetadataExtractor:
    """Extracts metadata from photos and videos."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_photo_metadata(self, media_file: MediaFile) -> Dict[str, Any]:
        """Extract comprehensive metadata from photo file.

        Args:
            media_file: MediaFile object

        Returns:
            Dictionary containing extracted metadata
        """
        metadata = {
            'filename': media_file.filename,
            'file_path': str(media_file.path),
            'file_size': media_file.size,
            'creation_time': media_file.time,
            'gps_coordinates': None,
            'camera_make': None,
            'camera_model': None,
            'lens_model': None,
            'focal_length': None,
            'aperture': None,
            'iso': None,
            'shutter_speed': None,
            'flash': None,
            'orientation': None,
            'width': None,
            'height': None,
            'color_space': None
        }

        try:
            # Method 1: Use PIL for basic EXIF data
            self._extract_pil_metadata(media_file.path, metadata)

            # Method 2: Use exifread for more detailed GPS data
            self._extract_exifread_metadata(media_file.path, metadata)

        except Exception as e:
            self.logger.warning(f"Error extracting metadata from {media_file.filename}: {e}")

        return metadata

    def _extract_pil_metadata(self, file_path: Path, metadata: Dict[str, Any]) -> None:
        """Extract metadata using PIL."""
        try:
            with Image.open(file_path) as img:
                metadata['width'] = img.width
                metadata['height'] = img.height
                metadata['color_space'] = img.mode

                # Get EXIF data
                exif_data = img._getexif()
                if exif_data is not None:
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)

                        if tag == 'Make':
                            metadata['camera_make'] = str(value).strip()
                        elif tag == 'Model':
                            metadata['camera_model'] = str(value).strip()
                        elif tag == 'LensModel':
                            metadata['lens_model'] = str(value).strip()
                        elif tag == 'FocalLength':
                            if isinstance(value, tuple) and len(value) == 2:
                                metadata['focal_length'] = round(value[0] / value[1], 1)
                        elif tag == 'FNumber':
                            if isinstance(value, tuple) and len(value) == 2:
                                metadata['aperture'] = round(value[0] / value[1], 1)
                        elif tag == 'ISOSpeedRatings':
                            metadata['iso'] = value
                        elif tag == 'ExposureTime':
                            if isinstance(value, tuple) and len(value) == 2:
                                metadata['shutter_speed'] = f"1/{int(value[1]/value[0])}"
                        elif tag == 'Flash':
                            metadata['flash'] = 'On' if value & 1 else 'Off'
                        elif tag == 'Orientation':
                            metadata['orientation'] = value
                        elif tag == 'GPSInfo':
                            gps_coords = self._extract_gps_from_pil(value)
                            if gps_coords:
                                metadata['gps_coordinates'] = gps_coords

        except Exception as e:
            self.logger.debug(f"PIL metadata extraction failed for {file_path}: {e}")

    def _extract_exifread_metadata(self, file_path: Path, metadata: Dict[str, Any]) -> None:
        """Extract metadata using exifread for more detailed GPS info."""
        try:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)

                # Extract GPS coordinates with higher precision
                if metadata.get('gps_coordinates') is None:
                    gps_coords = self._extract_gps_from_exifread(tags)
                    if gps_coords:
                        metadata['gps_coordinates'] = gps_coords

                # Fill in any missing camera data
                if not metadata.get('camera_make') and 'Image Make' in tags:
                    metadata['camera_make'] = str(tags['Image Make']).strip()

                if not metadata.get('camera_model') and 'Image Model' in tags:
                    metadata['camera_model'] = str(tags['Image Model']).strip()

        except Exception as e:
            self.logger.debug(f"Exifread metadata extraction failed for {file_path}: {e}")

    def _extract_gps_from_pil(self, gps_info: Dict) -> Optional[Tuple[float, float]]:
        """Extract GPS coordinates from PIL GPS info."""
        try:
            if not gps_info:
                return None

            gps_data = {}
            for key, value in gps_info.items():
                tag = GPSTAGS.get(key, key)
                gps_data[tag] = value

            if 'GPSLatitude' in gps_data and 'GPSLongitude' in gps_data:
                lat = self._convert_gps_coordinate(
                    gps_data['GPSLatitude'],
                    gps_data.get('GPSLatitudeRef', 'N')
                )
                lon = self._convert_gps_coordinate(
                    gps_data['GPSLongitude'],
                    gps_data.get('GPSLongitudeRef', 'E')
                )

                if lat is not None and lon is not None:
                    return (lat, lon)

        except Exception as e:
            self.logger.debug(f"GPS extraction from PIL failed: {e}")

        return None

    def _extract_gps_from_exifread(self, tags: Dict) -> Optional[Tuple[float, float]]:
        """Extract GPS coordinates from exifread tags."""
        try:
            lat_tag = tags.get('GPS GPSLatitude')
            lat_ref_tag = tags.get('GPS GPSLatitudeRef')
            lon_tag = tags.get('GPS GPSLongitude')
            lon_ref_tag = tags.get('GPS GPSLongitudeRef')

            if lat_tag and lon_tag:
                lat = self._convert_exifread_gps(lat_tag, lat_ref_tag)
                lon = self._convert_exifread_gps(lon_tag, lon_ref_tag)

                if lat is not None and lon is not None:
                    return (lat, lon)

        except Exception as e:
            self.logger.debug(f"GPS extraction from exifread failed: {e}")

        return None

    def _convert_gps_coordinate(self, coord, ref) -> Optional[float]:
        """Convert GPS coordinate from degrees/minutes/seconds to decimal."""
        try:
            if isinstance(coord, (list, tuple)) and len(coord) >= 3:
                degrees = float(coord[0])
                minutes = float(coord[1])
                seconds = float(coord[2])

                decimal = degrees + (minutes / 60) + (seconds / 3600)

                if ref in ('S', 'W'):
                    decimal = -decimal

                return decimal
        except (ValueError, TypeError, IndexError):
            pass

        return None

    def _convert_exifread_gps(self, coord_tag, ref_tag) -> Optional[float]:
        """Convert exifread GPS coordinate to decimal."""
        try:
            coords = [float(x.num) / float(x.den) for x in coord_tag.values]
            if len(coords) >= 3:
                decimal = coords[0] + (coords[1] / 60) + (coords[2] / 3600)

                if ref_tag and str(ref_tag) in ('S', 'W'):
                    decimal = -decimal

                return decimal
        except (ValueError, TypeError, AttributeError):
            pass

        return None

    def extract_video_metadata(self, media_file: MediaFile) -> Dict[str, Any]:
        """Extract metadata from video file.

        Args:
            media_file: MediaFile object

        Returns:
            Dictionary containing extracted metadata
        """
        metadata = {
            'filename': media_file.filename,
            'file_path': str(media_file.path),
            'file_size': media_file.size,
            'creation_time': media_file.time,
            'duration': None,
            'fps': None,
            'width': None,
            'height': None,
            'codec': None,
            'bitrate': None,
            'gps_coordinates': None  # Videos can also have GPS data
        }

        try:
            # Use OpenCV to extract video metadata
            cap = cv2.VideoCapture(str(media_file.path))

            if cap.isOpened():
                # Basic video properties
                metadata['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                metadata['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                metadata['fps'] = cap.get(cv2.CAP_PROP_FPS)

                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if frame_count > 0 and metadata['fps'] > 0:
                    metadata['duration'] = frame_count / metadata['fps']

                # Try to get codec info
                fourcc = cap.get(cv2.CAP_PROP_FOURCC)
                if fourcc:
                    codec_bytes = int(fourcc).to_bytes(4, byteorder='little')
                    try:
                        metadata['codec'] = codec_bytes.decode('ascii').strip()
                    except UnicodeDecodeError:
                        metadata['codec'] = 'Unknown'

                cap.release()

            # TODO: Extract GPS from video metadata (requires different approach)
            # Some video formats store GPS data in metadata that requires
            # specialized libraries like ffprobe

        except Exception as e:
            self.logger.warning(f"Error extracting video metadata from {media_file.filename}: {e}")

        return metadata