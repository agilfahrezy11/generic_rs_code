"""
mosaic_commodities.py

End-to-end mosaicking of FAO commodity presence/absence (0/1) binary
rasters across SEA + Bhutan + PNG.

WHAT IT DOES
------------
1. For each commodity/year folder, discovers every .tif tile inside it
   (country-level tiles, plus Indonesia's multi-island / auto-sharded
   exports such as cocoa_binary_IDN_ISLD_2020-0000000000-0000000000).
2. Reprojects any tile that isn't EPSG:4326 on the fly (nearest-neighbour,
   since the data is categorical 0/1 -> no interpolation allowed).
3. Writes every tile into one shared output canvas, one small window at a
   time (not one giant in-memory array). This is the fix for the OOM you
   hit mosaicking the 10 m rice rasters over the full SEA extent: memory
   use here is bounded by the size of a single input tile, never by the
   size of the full SEA mosaic.
4. Where tiles overlap (e.g. shard seams), combines with np.maximum so a
   "presence" (1) from any tile always wins over "absence" (0) — safe for
   binary data and avoids seam artifacts overwriting good pixels.
5. Optionally clips the final mosaic to the dissolved AOI boundary from
   the master shapefile, to trim any warp halo beyond real land area.

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
from rasterio.crs import CRS
from pathlib import Path
from rasterio.warp import transform_bounds, reproject, calculate_default_transform, Resampling
from rasterio.windows import Window, from_bounds, transform as window_transform
from rasterio.transform import from_origin
from rasterio.mask import mask as rio_mask

# --------------------------------------------------------------------------
# CONFIG - edit paths here
# --------------------------------------------------------------------------
folders_2020 = [
    Path(r"C:\FAO_commodities_presence_absence\raw_ee\cocoa_2020"),
    Path(r"C:\FAO_commodities_presence_absence\raw_ee\coffee_2020"),
    Path(r"C:\FAO_commodities_presence_absence\raw_ee\palm_2020"),
    Path(r"C:\FAO_commodities_presence_absence\raw_ee\rubber_2020"),
    Path(r"C:\FAO_commodities_presence_absence\raw_ee\maize_2020"),
    Path(r"C:\FAO_commodities_presence_absence\raw_ee\timber_2020"),
]
folders_2025 = [
    Path(r"C:\FAO_commodities_presence_absence\raw_ee\cocoa_2025"),
    Path(r"C:\FAO_commodities_presence_absence\raw_ee\coffee_2025"),
    Path(r"C:\FAO_commodities_presence_absence\raw_ee\palm_2025"),
    Path(r"C:\FAO_commodities_presence_absence\raw_ee\rubber_2025"),
    Path(r"C:\FAO_commodities_presence_absence\raw_ee\maize_2025"),
    Path(r"C:\FAO_commodities_presence_absence\raw_ee\timber_2025"),
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
    with the native source resolution when all sources already use target_crs.
    Returns
    (transform, width, height).

    The bounds are snapped to the reference raster's pixel grid. This avoids
    a fractional-pixel shift that would make rasterio resample otherwise
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

            # diagnostic only -- confirms what the source file's native
            # resolution actually is, in its own CRS units
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


def mosaic_folder(folder: Path, output_dir: Path):
    commodity, year = parse_folder_name(folder)
    print(f"\n=== {commodity} {year} ===")

    files = list_tif_files(folder)
    if not files:
        return

    print("  Computing output grid...")
    dst_transform, dst_width, dst_height = compute_dst_grid(files)
    print(f"  Grid: {dst_width} x {dst_height} px, res={dst_transform.a:.8f} deg")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{commodity}_combine_{year}_SEA.tif"

    profile = {
        "driver": "GTiff",
        "dtype": DTYPE,
        "count": 1,
        "height": dst_height,
        "width": dst_width,
        "crs": TARGET_CRS,
        "transform": dst_transform,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": 0,
    }

    with rasterio.open(out_path, "w+", **profile) as dst:
        for f in files:
            with rasterio.open(f) as src:
                src_bounds_t = transform_bounds(src.crs, TARGET_CRS, *src.bounds, densify_pts=21)

                win = from_bounds(*src_bounds_t, transform=dst_transform)
                win = win.round_offsets(op="floor").round_lengths(op="ceil")
                win = win.intersection(Window(0, 0, dst_width, dst_height)) # type: ignore

                if win.width <= 0 or win.height <= 0:
                    print(f"    [SKIP] {f.name} falls outside output grid")
                    continue

                win_transform = window_transform(win, dst_transform)
                w_h, w_w = int(round(win.height)), int(round(win.width))

                local = np.full((w_h, w_w), FILL_VALUE, dtype=DTYPE)

                reproject(
                    source=rasterio.band(src, 1),
                    destination=local,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=win_transform,
                    dst_crs=TARGET_CRS,
                    dst_nodata=FILL_VALUE,
                    resampling=Resampling.nearest,
                )

                existing = dst.read(1, window=win)
                merged = np.maximum(existing, local).astype(DTYPE)
                dst.write(merged, 1, window=win)

                print(f"    merged {f.name}")

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