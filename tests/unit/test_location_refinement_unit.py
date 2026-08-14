#!/usr/bin/env python3
"""
Unit tests for location-based cluster refinement.

These are FAST unit tests with mocked geocoding/metadata - no network calls,
no real photos. They protect against Issue #70: burst photos with identical
GPS coordinates being collapsed into N copies of one file during location
refinement, silently dropping the sibling files from organization.

Run with: pytest tests/unit/test_location_refinement_unit.py -v
Expected time: <3 seconds total
"""

import pytest
from unittest.mock import Mock
from pathlib import Path
from datetime import datetime

try:
    from src.media_clustering import MediaClusteringEngine, MediaCluster
    from src.media_detector import MediaFile
    from src.geocoding import LocationGeocoder, LocationInfo
except ImportError:
    import media_clustering
    import media_detector
    import geocoding
    MediaClusteringEngine = media_clustering.MediaClusteringEngine
    MediaCluster = media_clustering.MediaCluster
    MediaFile = media_detector.MediaFile
    LocationGeocoder = geocoding.LocationGeocoder
    LocationInfo = geocoding.LocationInfo


# Two locations far enough apart to force a cluster split (>1 km)
COORD_HOME = (53.5379, -113.6974)   # Edmonton - west side
COORD_DOWNTOWN = (53.5461, -113.4938)  # Edmonton - downtown


def make_media_file(name: str, when: datetime) -> MediaFile:
    """Create a mock MediaFile like the ones in test_clustering_face_unit.py."""
    return MediaFile(
        path=Path(f"mock_photos/{name}"),
        filename=name,
        date=when,
        time=when,
        extension='.jpg',
        file_type='photo',
        size=1024
    )


def make_engine_with_mocks(gps_by_filename):
    """Create a MediaClusteringEngine with mocked geocoder and metadata extractor.

    Args:
        gps_by_filename: dict mapping filename to (lat, lon) or None
    """
    engine = MediaClusteringEngine()

    def fake_metadata(media_file):
        return {'gps_coordinates': gps_by_filename.get(media_file.filename)}

    engine.metadata_extractor = Mock()
    engine.metadata_extractor.extract_photo_metadata.side_effect = fake_metadata

    engine.geocoder = Mock()
    engine.geocoder.reverse_geocode.return_value = None
    return engine


def build_cluster(media_files, gps_by_filename):
    """Build a MediaCluster whose gps_coordinates align with media_files order,
    mirroring how _enrich_with_location_data constructs clusters."""
    gps_coordinates = [gps_by_filename[f.filename] for f in media_files
                       if gps_by_filename.get(f.filename) is not None]
    return MediaCluster(
        cluster_id=0,
        media_files=media_files,
        temporal_info=None,
        gps_coordinates=gps_coordinates
    )


@pytest.mark.unit
@pytest.mark.regression
def test_burst_photos_stay_distinct_after_location_split():
    """Issue #70 regression: burst photos share identical GPS - refinement must
    keep them as distinct files, not N copies of the first one."""
    when = datetime(2016, 4, 29, 14, 7, 1)  # burst: identical second-resolution time
    burst = [make_media_file(f"IMG_20160429_140701_{i}.JPG", when) for i in range(6)]
    downtown = [make_media_file(f"IMG_20160429_1801{i:02d}.JPG",
                                datetime(2016, 4, 29, 18, 1, i)) for i in range(2)]
    files = burst + downtown

    gps = {f.filename: COORD_HOME for f in burst}
    gps.update({f.filename: COORD_DOWNTOWN for f in downtown})

    engine = make_engine_with_mocks(gps)
    # Force a split: indices 0-5 (burst at home) and 6-7 (downtown)
    engine.geocoder.cluster_locations_by_proximity.return_value = [
        [0, 1, 2, 3, 4, 5], [6, 7]
    ]

    cluster = build_cluster(files, gps)
    refined = engine._refine_with_location_clustering([cluster])

    # Every input file appears exactly once across all refined clusters
    all_paths = [str(f.path) for c in refined for f in c.media_files]
    assert len(all_paths) == len(files), (
        f"Expected {len(files)} files after refinement, got {len(all_paths)}")
    assert len(set(all_paths)) == len(files), (
        f"Duplicate files in refined clusters: {sorted(all_paths)}")

    # The split produced the expected group sizes
    sizes = sorted(len(c.media_files) for c in refined)
    assert sizes == [2, 6], f"Expected clusters of size 2 and 6, got {sizes}"


@pytest.mark.unit
@pytest.mark.regression
def test_no_gps_files_kept_in_separate_cluster_after_split():
    """Files without GPS in a split cluster must survive into their own cluster."""
    when = datetime(2016, 4, 29, 14, 7, 1)
    with_gps = [make_media_file(f"IMG_A_{i}.JPG", when) for i in range(2)]
    with_gps += [make_media_file(f"IMG_B_{i}.JPG", when) for i in range(2)]
    no_gps = [make_media_file("IMG_NOGPS.JPG", when)]
    files = with_gps + no_gps

    gps = {"IMG_A_0.JPG": COORD_HOME, "IMG_A_1.JPG": COORD_HOME,
           "IMG_B_0.JPG": COORD_DOWNTOWN, "IMG_B_1.JPG": COORD_DOWNTOWN,
           "IMG_NOGPS.JPG": None}

    engine = make_engine_with_mocks(gps)
    engine.geocoder.cluster_locations_by_proximity.return_value = [[0, 1], [2, 3]]

    cluster = build_cluster(files, gps)
    refined = engine._refine_with_location_clustering([cluster])

    all_paths = [str(f.path) for c in refined for f in c.media_files]
    assert len(all_paths) == 5
    assert len(set(all_paths)) == 5
    # Three clusters: two location groups plus one no-GPS cluster
    assert len(refined) == 3
    no_gps_clusters = [c for c in refined if not c.gps_coordinates]
    assert len(no_gps_clusters) == 1
    assert no_gps_clusters[0].media_files[0].filename == "IMG_NOGPS.JPG"


@pytest.mark.unit
@pytest.mark.regression
def test_misaligned_gps_falls_back_without_duplicating_or_losing_files():
    """If gps_coordinates and files_with_gps counts diverge, the fallback
    coordinate matching must still never duplicate a file, and unmatched
    files must be preserved rather than dropped."""
    when = datetime(2016, 12, 3, 17, 11, 25)
    files = [make_media_file(f"IMG_20161203_17112{i}.JPG", when) for i in range(4)]
    gps = {f.filename: COORD_HOME for f in files}

    engine = make_engine_with_mocks(gps)
    engine.geocoder.cluster_locations_by_proximity.return_value = [[0, 1], [2]]

    cluster = build_cluster(files, gps)
    # Simulate the misalignment: 3 coordinates for 4 GPS-bearing files
    cluster.gps_coordinates = [COORD_HOME, COORD_HOME, COORD_DOWNTOWN]

    refined = engine._refine_with_location_clustering([cluster])

    all_paths = [str(f.path) for c in refined for f in c.media_files]
    assert len(set(all_paths)) == len(all_paths), (
        f"Duplicate files in fallback path: {sorted(all_paths)}")
    assert len(set(all_paths)) == 4, (
        f"Files lost in fallback path: expected 4, got {len(set(all_paths))}")


@pytest.mark.unit
@pytest.mark.regression
def test_split_group_uses_majority_location_not_first_coordinate():
    """Regression: _refine_with_location_clustering used to set a split
    sub-cluster's location from location_coords[0] - the chronologically
    first photo - instead of the majority-group representative point. A
    30-photo New Year's cluster's first photo was a ~330m GPS-lag outlier
    (common iPhone first-shot-after-opening-camera quirk); after
    find_representative_location was wired into _enhance_with_location_data
    (see test_geocoding_unit.py / NOTES.md in
    temp/diagnostics/2026-08-11_location_representative_point_bug/), this
    second call site - reached only when a cluster later gets split by
    _refine_with_location_clustering - still used location_coords[0] and
    silently reintroduced the same wrong location. Uses a real
    LocationGeocoder (reverse_geocode stubbed, no network) so the real
    find_representative_location/cluster_locations_by_proximity math runs."""
    outlier_file = make_media_file("IMG_20160101_001421.JPG",
                                    datetime(2016, 1, 1, 0, 14, 21))
    majority_files = [make_media_file(f"IMG_20160101_00192{i}.JPG",
                                       datetime(2016, 1, 1, 0, 19, 20 + i))
                       for i in range(5)]
    # A second, distant group - forces cluster_locations_by_proximity to
    # return >1 group, which is what actually triggers the split branch
    # (and the buggy location_coords[0] line within it) in production.
    afternoon_file = make_media_file("IMG_20160101_163218.JPG",
                                      datetime(2016, 1, 1, 16, 32, 18))
    files = [outlier_file] + majority_files + [afternoon_file]

    outlier_coord = (53.538372, -113.616562)  # the real GPS-lag outlier
    majority_base = (53.539219, -113.611769)  # ~330m away - the real majority spot
    gps = {outlier_file.filename: outlier_coord}
    gps.update({f.filename: (majority_base[0] + i * 0.00001, majority_base[1] + i * 0.00001)
                for i, f in enumerate(majority_files)})
    gps[afternoon_file.filename] = COORD_DOWNTOWN

    def make_location(lat, lon, neighbourhood):
        return LocationInfo(latitude=lat, longitude=lon, address="", city="Edmonton",
                             state="Alberta", country="Canada",
                             raw_data={"address": {"neighbourhood": neighbourhood}})

    def fake_reverse_geocode(lat, lon):
        if abs(lat - outlier_coord[0]) < 0.0001 and abs(lon - outlier_coord[1]) < 0.0001:
            return make_location(lat, lon, "Place LaRue")
        return make_location(lat, lon, "Glenwood")

    engine = make_engine_with_mocks(gps)
    engine.geocoder = LocationGeocoder()
    engine.geocoder.reverse_geocode = fake_reverse_geocode
    # Real (unmocked) distance clustering: the 6 midnight-area points (which
    # internally span the ~330m GPS-lag outlier) fall within the outer 1km
    # split threshold as one group vs. the distant afternoon photo - this is
    # what actually triggers the split branch that had the bug. The same
    # real method, at a tighter threshold, is what find_representative_
    # location uses inside the fix - must not be mocked, or it can't tell
    # the outlier from the majority.

    cluster = build_cluster(files, gps)
    refined = engine._refine_with_location_clustering([cluster])

    assert len(refined) == 2
    midnight_cluster = next(c for c in refined
                             if outlier_file.filename in [f.filename for f in c.media_files])
    neighbourhood = midnight_cluster.location_info.raw_data["address"]["neighbourhood"]
    assert neighbourhood == "Glenwood", (
        f"Expected majority location 'Glenwood', got '{neighbourhood}' - "
        "location_coords[0] (the outlier) was used instead of "
        "find_representative_location")


@pytest.mark.unit
def test_single_location_cluster_unchanged():
    """Clusters whose locations are all within threshold must pass through intact."""
    when = datetime(2019, 1, 1, 0, 6, 5)
    files = [make_media_file(f"IMG_20190101_00060{i}.JPG", when) for i in range(3)]
    gps = {f.filename: COORD_HOME for f in files}

    engine = make_engine_with_mocks(gps)
    engine.geocoder.cluster_locations_by_proximity.return_value = [[0, 1, 2]]

    cluster = build_cluster(files, gps)
    refined = engine._refine_with_location_clustering([cluster])

    assert len(refined) == 1
    assert [f.filename for f in refined[0].media_files] == [f.filename for f in files]
