"""
mosaic_commodities.py

End-to-end mosaicking of FAO commodity presence/absence (0/1) binary
rasters across SEA + Bhutan + PNG.

WHAT IT DOES
------------
1. For each commodity/year folder, discovers every .tif tile inside it
   (country-level tiles, plus Indonesia's multi-island / auto-sharded
   exports such as cocoa_binary_IDN_ISLD_2020-0000000000-0000000000).
2. Merges all tiles with rasterio.merge using the binary-safe ``max`` rule,
    so a presence (1) always wins over absence (0) in overlaps.
3. Streams the merge to disk in bounded windows, avoiding one giant
    in-memory array for large regional mosaics.
4. Optionally clips the final mosaic to the dissolved AOI boundary from the
    master shapefile, to trim pixels outside the AOI.

OUTPUT
------
{OUTPUT_DIR}/{commodity}_combine_{year}_SEA.tif
e.g. rubber_combine_2020_SEA.tif, palm_combine_2025_SEA.tif, ...

Countries with an all-zero raster for a given commodity are handled
automatically: they just contribute 0s to the mosaic like any other tile,
no special casing needed.
"""

import re
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from pathlib import Path
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds, reproject, calculate_default_transform, Resampling
from rasterio.windows import Window, from_bounds, transform as window_transform
from rasterio.transform import from_origin
from rasterio.mask import mask as rio_mask
from rasterio.features import geometry_mask

# --------------------------------------------------------------------------
# CONFIG - edit paths here
# --------------------------------------------------------------------------
folders_2020 = [
    #Path(r"C:\FAO_commodities_presence_absence\raw_ee\cocoa_2020"),
    #Path(r"C:\FAO_commodities_presence_absence\raw_ee\coffee_2020"),
    #Path(r"C:\FAO_commodities_presence_absence\raw_ee\palm_2020"),
    #Path(r"C:\FAO_commodities_presence_absence\raw_ee\rubber_2020"),
    #Path(r"C:\FAO_commodities_presence_absence\raw_ee\maize_2020"),
    #Path(r"C:\FAO_commodities_presence_absence\raw_ee\timber_2020"),
    Path(r"C:\FAO_commodities_presence_absence\intermediate_process\rice_2020"),
    
]
folders_2025 = [
    #Path(r"C:\FAO_commodities_presence_absence\raw_ee\cocoa_2025"),
    #Path(r"C:\FAO_commodities_presence_absence\raw_ee\coffee_2025"),
    #Path(r"C:\FAO_commodities_presence_absence\raw_ee\palm_2025"),
    #Path(r"C:\FAO_commodities_presence_absence\raw_ee\rubber_2025"),
    #Path(r"C:\FAO_commodities_presence_absence\raw_ee\maize_2025"),
    Path(r"C:\FAO_commodities_presence_absence\intermediate_process\rice_2023"),
]

master_shapefile_path = Path(
    r"C:\Users\AFahrezi\Documents\GitHub\generic_rs_code\raster_mosaicking\AOI\FAO_RCP51_Country.shp"
)

OUTPUT_DIR = Path(r"C:\FAO_commodities_presence_absence\mosaicked")
TARGET_CRS = "EPSG:4326"
FILL_VALUE = 0          # absence / background value
DTYPE = "uint8"
CLIP_TO_AOI = False      # set True to clip final mosaic to dissolved AOI boundary

# Pin the output resolution explicitly instead of letting GDAL auto-estimate it.
# calculate_default_transform's automatic resolution guess is NOT guaranteed to
# match your real source pixel size and can silently collapse the mosaic to a
# much coarser grid (the "giant flat blocks" symptom). 100 m at the equator is
# ~0.0008983 deg; this scales it slightly for typical SEA latitudes. Adjust if
# your data's true native resolution differs.
TARGET_RESOLUTION_METERS = 100
TARGET_RESOLUTION_DEG = TARGET_RESOLUTION_METERS / 111_320.0  # ~0.000898

FOLDER_NAME_RE = re.compile(r"^(?P<commodity>[a-zA-Z]+)_(?P<year>\d{4})$")

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def parse_folder_name(folder: Path):
    """cocoa_2020 -> ('cocoa', '2020')"""
    m = FOLDER_NAME_RE.match(folder.name)
    if not m:
        raise ValueError(f"Could not parse commodity/year from folder name: {folder.name}")
    return m.group("commodity"), m.group("year")


def list_tif_files(folder: Path):
    files = sorted(list(folder.glob("*.tif")) + list(folder.glob("*.tiff")))
    if not files:
        print(f"  [WARN] No raster files found in {folder}")
    return files


def compute_dst_grid(files, target_crs=TARGET_CRS):
    """
    Union bounds across all source tiles (expressed in target_crs), combined
    with the native source resolution when all sources already use target_crs
    (falls back to TARGET_RESOLUTION_DEG otherwise). Returns
    (transform, width, height).

    Bounds are snapped to the reference raster's pixel grid to avoid a
    fractional-pixel shift that would make rasterio resample otherwise
    identical source tiles during the merge.
    """
    minx = miny = np.inf
    maxx = maxy = -np.inf
    reference_res = None
    reference_origin = None
    native_grid = True

    for f in files:
        with rasterio.open(f) as src:
            if src.crs is None:
                raise ValueError(f"Raster has no CRS: {f}")

            if src.crs == CRS.from_string(target_crs):
                if reference_res is None:
                    reference_res = src.res
                    reference_origin = (src.transform.c, src.transform.f)
                elif not np.allclose(src.res, reference_res, rtol=0, atol=1e-12):
                    native_grid = False
            else:
                native_grid = False

            bounds_t = transform_bounds(src.crs, target_crs, *src.bounds, densify_pts=21)
            minx = min(minx, bounds_t[0])
            miny = min(miny, bounds_t[1])
            maxx = max(maxx, bounds_t[2])
            maxy = max(maxy, bounds_t[3])

            print(f"    [DIAG] {f.name}: crs={src.crs}, res={src.res}, size={src.width}x{src.height}")

    if native_grid and reference_res is not None and reference_origin is not None:
        xres, yres = reference_res
        origin_x, origin_y = reference_origin
        minx = origin_x + np.floor((minx - origin_x) / xres) * xres
        maxx = origin_x + np.ceil((maxx - origin_x) / xres) * xres
        miny = origin_y - np.ceil((origin_y - miny) / yres) * yres
        maxy = origin_y - np.floor((origin_y - maxy) / yres) * yres
        print(f"    [DIAG] Preserving native grid: res={xres}, {yres}")
    else:
        xres = yres = TARGET_RESOLUTION_DEG
        print(f"    [DIAG] Using fallback target grid: res={xres}, {yres}")

    width = int(np.ceil((maxx - minx) / xres))
    height = int(np.ceil((maxy - miny) / yres))
    dst_transform = from_origin(minx, maxy, xres, yres)
    return dst_transform, width, height


def extract_iso3(filename: str, commodity: str):
    """
    cocoa_binary_KHM_2020.tif -> 'KHM'
    cocoa_binary_IDN_ISLD_2020-0000000000-0000000000.tif -> 'IDN'
    (island/shard suffixes are ignored -- they all belong to the same country
    polygon for clipping purposes)
    """
    m = re.search(rf"{re.escape(commodity)}_binary_([A-Z]{{3}})", filename)
    return m.group(1) if m else None


def build_country_geometries(shapefile_path: Path, target_crs=TARGET_CRS):
    """
    Loads the master AOI shapefile once and returns a dict of
    {ISO_A3: [shapely geometries]} in target_crs, for per-tile clipping.
    Countries with multiple polygon parts (e.g. archipelagos) keep all parts.
    """
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs is None or gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(target_crs)

    geoms_by_iso3 = {}
    for iso3, group in gdf.groupby("ISO_A3"):
        geoms_by_iso3[iso3] = list(group.geometry)
    return geoms_by_iso3


def mosaic_folder(folder: Path, output_dir: Path):
    commodity, year = parse_folder_name(folder)
    print(f"\n=== {commodity} {year} ===")

    files = list_tif_files(folder)
    if not files:
        return

    print("  Preparing merge...")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{commodity}_combine_{year}_final.tif"

    print(f"  Reprojecting sources to {TARGET_CRS} as needed...")
    source_files = [rasterio.open(path) for path in files]
    src_files = []
    try:
        for path, src in zip(files, source_files):
            if src.crs is None:
                raise ValueError(f"Raster has no CRS and cannot be standardized: {path}")

            vrt = WarpedVRT(
                src,
                crs=TARGET_CRS,
                resampling=Resampling.nearest,
                src_nodata=src.nodata if src.nodata is not None else FILL_VALUE,
                nodata=FILL_VALUE,
            )
            src_files.append(vrt)

            if src.crs != CRS.from_string(TARGET_CRS):
                print(f"    standardized {path.name}: {src.crs} -> {TARGET_CRS}")

        output_profile = src_files[0].profile.copy()
        output_profile.update(
            driver="GTiff",
            nodata=FILL_VALUE,
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER",
        )

        print(f"  Merging {len(files)} rasters with max overlap rule...")
        # This is the notebook's merge logic, written directly to disk so
        # memory use is bounded instead of allocating the full mosaic.
        merge(
            src_files,
            method="max",
            nodata=FILL_VALUE,
            mem_limit=128,
            dst_path=out_path,
            dst_kwds=output_profile,
        )
    finally:
        for src in src_files:
            src.close()
        for src in source_files:
            src.close()

    print(f"  Wrote {out_path}")

    if CLIP_TO_AOI:
        clip_to_aoi(out_path)


def clip_to_aoi(raster_path: Path):
    """Optional: trim mosaic to the dissolved AOI boundary from the master shapefile."""
    sea_gdf = gpd.read_file(master_shapefile_path)
    if sea_gdf.crs is None or sea_gdf.crs.to_string() != TARGET_CRS:
        sea_gdf = sea_gdf.to_crs(TARGET_CRS)
    dissolved = [sea_gdf.union_all()] if hasattr(sea_gdf, "union_all") else [sea_gdf.unary_union]

    with rasterio.open(raster_path) as src:
        out_image, out_transform = rio_mask(src, dissolved, crop=True, nodata=FILL_VALUE)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        })

    with rasterio.open(raster_path, "w", **out_meta) as dst:
        dst.write(out_image)
    print(f"  Clipped to AOI: {raster_path}")


def main():
    for folder in folders_2020 + folders_2025:
        mosaic_folder(folder, OUTPUT_DIR)


if __name__ == "__main__":
    main()