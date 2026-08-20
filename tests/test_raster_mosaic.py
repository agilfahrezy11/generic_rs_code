import sys
from pathlib import Path

from rasterio.crs import CRS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "raster_mosaicking"))

from raster_mosaic import choose_target_crs_for_extent


def test_single_utm_zone_returns_utm_crs():
    crs = choose_target_crs_for_extent((100.0, -1.0, 101.0, 1.0))
    assert crs == CRS.from_epsg(32647)


def test_multi_utm_extent_falls_back_to_web_mercator():
    crs = choose_target_crs_for_extent((90.0, -15.0, 150.0, 20.0))
    assert crs == CRS.from_epsg(3857)
