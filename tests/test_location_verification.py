#!/usr/bin/env python3
"""Test location metadata interpretation for Edmonton verification."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.media_detector import MediaDetector
from src.metadata_extractor import MetadataExtractor
from src.geocoding import LocationGeocoder

def test_location_verification():
    """Test location metadata interpretation with Edmonton photos."""
    print("🌍 Testing GPS metadata interpretation...")

    # Initialize components
    detector = MediaDetector()
    extractor = MetadataExtractor()
    geocoder = LocationGeocoder()

    # Get sample of photos
    all_files = detector.scan_iphone_automatic()
    photo_files = [f for f in all_files if f.file_type == 'photo']

    # Test first 10 photos with GPS data
    photos_with_gps = []

    print(f"Checking GPS data in {len(photo_files)} photos...")

    for i, photo in enumerate(photo_files[:20]):  # Check first 20
        metadata = extractor.extract_photo_metadata(photo)
        gps_coords = metadata.get('gps_coordinates')

        if gps_coords and len(gps_coords) == 2:
            lat, lon = gps_coords
            photos_with_gps.append({
                'filename': photo.filename,
                'coordinates': (lat, lon),
                'metadata': metadata
            })

            print(f"\n📸 {photo.filename}")
            print(f"  GPS: {lat:.6f}, {lon:.6f}")
            print(f"  Camera: {metadata.get('camera_make', 'Unknown')} {metadata.get('camera_model', '')}")
            print(f"  Date: {photo.time}")

            # Perform reverse geocoding
            print("  🔍 Looking up location...")
            location_info = geocoder.reverse_geocode(lat, lon)

            if location_info:
                print(f"  📍 Location: {location_info.address}")
                print(f"  🏙️  City: {location_info.city}")
                print(f"  🏛️  State: {location_info.state}")
                print(f"  🇨🇦 Country: {location_info.country}")
            else:
                print("  ❌ Location lookup failed")

            if len(photos_with_gps) >= 5:  # Limit to first 5 with GPS
                break

    print(f"\n📊 Summary:")
    print(f"  Total photos checked: {min(20, len(photo_files))}")
    print(f"  Photos with GPS data: {len(photos_with_gps)}")

    if photos_with_gps:
        # Analyze GPS coordinate patterns
        latitudes = [p['coordinates'][0] for p in photos_with_gps]
        longitudes = [p['coordinates'][1] for p in photos_with_gps]

        avg_lat = sum(latitudes) / len(latitudes)
        avg_lon = sum(longitudes) / len(longitudes)

        print(f"\n🎯 GPS Analysis:")
        print(f"  Average coordinates: {avg_lat:.6f}, {avg_lon:.6f}")
        print(f"  Latitude range: {min(latitudes):.6f} to {max(latitudes):.6f}")
        print(f"  Longitude range: {min(longitudes):.6f} to {max(longitudes):.6f}")

        # Edmonton reference coordinates (approximately)
        edmonton_lat, edmonton_lon = 53.5461, -113.4938

        print(f"\n🏙️  Edmonton Reference: {edmonton_lat:.4f}, {edmonton_lon:.4f}")

        # Check if coordinates are in Edmonton area
        lat_diff = abs(avg_lat - edmonton_lat)
        lon_diff = abs(avg_lon - edmonton_lon)

        print(f"  Difference from Edmonton center:")
        print(f"    Latitude: {lat_diff:.4f}° ({lat_diff * 111:.1f} km)")  # 1° ≈ 111 km
        print(f"    Longitude: {lon_diff:.4f}° ({lon_diff * 111 * 0.6:.1f} km)")  # Adjust for latitude

        if lat_diff < 0.5 and lon_diff < 0.5:  # Within ~50km
            print("  ✅ Coordinates appear to be in Edmonton area!")
        else:
            print("  ⚠️  Coordinates may not be in Edmonton area")

    else:
        print("  ❌ No photos with GPS data found in sample")

if __name__ == "__main__":
    test_location_verification()