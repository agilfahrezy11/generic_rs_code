#!/usr/bin/env python3
"""
mosaic_rasters.py

Mosaic split raster tiles (e.g. Earth Engine exports that got auto-sharded)
back into single rasters, grouped by region.

Uses rasterio.merge, which reads/writes in windows internally rather than
naively stacking full arrays -> memory-safe for large GEE tile sets, and
gives pixel-accurate control over overlap resolution (method) and
resampling.

Usage
-----
    # Auto-group files by a regex captured from the filename (default:
    # everything before the first tile index, e.g. "cocoa_prob_idn-0000000000-0000000000.tif"
    # groups by "cocoa_prob_idn")
    python mosaic_rasters.py -i ./exports -o ./mosaics

    # Explicit pattern, single region
    python mosaic_rasters.py -i ./exports/kambera_*.tif -o ./mosaics/kambera.tif

    # Choose overlap method / resampling / nodata / compression
    python mosaic_rasters.py -i ./exports -o ./mosaics \
        --method max --resampling bilinear --nodata 0 --compress zstd
"""

import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# GEE sharded exports typically look like: <name>-<row>-<col>.tif or <name>0000000000-0000000000.tif
TILE_SUFFIX_RE = re.compile(r"[-_]?\d{4,}[-_]\d{4,}$")


def group_key(path: Path) -> str:
    """Derive a region/group name by stripping trailing tile-index suffixes."""
    stem = path.stem
    key = TILE_SUFFIX_RE.sub("", stem)
    return key or stem


def choose_target_crs_for_extent(bounds: tuple[float, float, float, float]) -> CRS:
    """Choose a projected CRS that is appropriate for the raster extent.

    A single UTM CRS is valid only when the study area lies in one UTM zone.
    Multi-zone regions such as Southeast Asia should use a continuous global
    projected CRS instead of pretending the full extent is one UTM zone.
    """
    minx, miny, maxx, maxy = bounds
    if not (minx <= maxx and miny <= maxy):
        raise ValueError(f"Invalid bounds: {bounds}")

    min_zone = int((minx + 180.0) // 6.0) + 1
    max_zone = int((maxx + 180.0) // 6.0) + 1

    if miny >= -80 and maxy <= 84 and min_zone == max_zone:
        zone = min_zone
        epsg = 32600 + zone if maxy >= 0 else 32700 + zone
        return CRS.from_epsg(epsg)

    return CRS.from_epsg(3857)


def reproject_raster_to_crs(raster_path: Path, output_path: Path, dst_crs: CRS,
                           target_resolution: float = 100.0,
                           nodata: float | None = 0, compress: str = "lzw") -> Path:
    """Reproject a raster to a projected CRS and resample to a target ground resolution."""
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")

        if src.crs == dst_crs:
            # Keep the original raster if the CRS already matches, but still allow
            # output to be written with a tailored resolution.
            # This is mainly here to support explicit projected datasets.
            raise ValueError(f"Raster already in target CRS: {dst_crs}")

        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=(target_resolution, target_resolution),
        )

        kwargs = src.meta.copy()
        kwargs.update({
            "driver": "GTiff",
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "nodata": nodata,
            "compress": compress,
        })

        array = src.read()
        destination = np.empty((src.count, height, width), dtype=array.dtype)
        reproject(
            source=array,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            src_nodata=nodata,
            dst_nodata=nodata,
            resampling=Resampling.average,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **kwargs) as dst:
            dst.write(destination)

    return output_path


def discover_groups(input_dir: Path, pattern: str) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(input_dir.glob(pattern)):
        if f.is_file():
            groups[group_key(f)].append(f)
    return groups


def mosaic_group(files: list[Path], out_path: Path, method: str,
                  resampling: Resampling, nodata: float | None,
                  compress: str, dtype: str | None) -> None:
    if len(files) == 1:
        log.info("Single file for %s, copying as-is -> %s", out_path.name, out_path)
        srcs = [rasterio.open(files[0])]
    else:
        srcs = [rasterio.open(f) for f in files]

    log.info("Mosaicking %d tile(s) -> %s", len(srcs), out_path)

    mosaic_arr, out_transform = merge(
        srcs,
        method=method,
        resampling=resampling,
        nodata=nodata,
    )

    profile = srcs[0].profile.copy()
    profile.update(
        driver="GTiff",
        height=mosaic_arr.shape[1],
        width=mosaic_arr.shape[2],
        transform=out_transform,
        count=mosaic_arr.shape[0],
        dtype=dtype or mosaic_arr.dtype,
        compress=compress,
        predictor=2 if compress in ("deflate", "zstd", "lzw") else 1,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER",
    )
    if nodata is not None:
        profile["nodata"] = nodata

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic_arr.astype(profile["dtype"]))
        if len(files) > 1:
            dst.build_overviews([2, 4, 8, 16], Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")

    for s in srcs:
        s.close()


def mosaic_folder(folder: Path, output_dir: Path, method: str = "max",
                  nodata: float | None = 0, compress: str = "lzw") -> Path:
    """Mosaic all .tif files inside a single folder and save a single output raster."""
    tif_files = sorted(folder.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in folder: {folder}")

    srcs = [rasterio.open(fp) for fp in tif_files]
    try:
        if srcs[0].crs is not None and srcs[0].crs.is_geographic:
            # Southeast Asia spans multiple UTM zones; using one UTM projection would
            # be invalid for the region as a whole. A continuous projected CRS keeps the
            # mosaic coherent without pretending the entire extent is in a single UTM zone.
            bounds = (
                min(src.bounds.left for src in srcs),
                min(src.bounds.bottom for src in srcs),
                max(src.bounds.right for src in srcs),
                max(src.bounds.top for src in srcs),
            )
            target_crs = choose_target_crs_for_extent(bounds)
            log.info("Detected geographic CRS for %s; choosing %s for regional mosaic", folder, target_crs)
            reproj_dir = output_dir / "reprojected"
            reproj_dir.mkdir(parents=True, exist_ok=True)
            reproj_paths = []
            for fp in tif_files:
                out_fp = reproj_dir / f"{fp.stem}_reproj.tif"
                reproj_paths.append(reproject_raster_to_crs(fp, out_fp, target_crs, target_resolution=100.0, nodata=nodata, compress=compress))
            srcs = [rasterio.open(fp) for fp in reproj_paths]
            mosaic_arr, out_transform = merge(srcs, method=method, nodata=nodata)
            profile = srcs[0].profile.copy()
            profile.update(
                driver="GTiff",
                height=mosaic_arr.shape[1],
                width=mosaic_arr.shape[2],
                transform=out_transform,
                nodata=nodata,
                compress=compress,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"Mosaic_{folder.name}.tif"
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(mosaic_arr)
            return out_path

        mosaic_arr, out_transform = merge(srcs, method=method, nodata=nodata)
        profile = srcs[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=mosaic_arr.shape[1],
            width=mosaic_arr.shape[2],
            transform=out_transform,
            nodata=nodata,
            compress=compress,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"Mosaic_{folder.name}.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mosaic_arr)
        return out_path
    finally:
        for src in srcs:
            src.close()


def clip_raster_to_aoi(raster_path: Path, aoi_path: Path, output_path: Path,
                      nodata: float | None = 0) -> Path:
    """Clip a raster to one AOI shapefile and save the clipped result."""
    aoi = gpd.read_file(aoi_path)

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        aoi_reproj = aoi.to_crs(raster_crs)
        shapes = [geom for geom in aoi_reproj.geometry if not geom.is_empty]
        if not shapes:
            raise ValueError(f"No valid geometries found in AOI: {aoi_path}")

        clipped, out_transform = mask(
            src,
            shapes=shapes,
            crop=True,
            nodata=nodata,
            all_touched=False,
        )

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": clipped.shape[1],
            "width": clipped.shape[2],
            "transform": out_transform,
            "nodata": nodata,
            "compress": "lzw",
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(clipped)

    return output_path


def process_folders(folders: list[Path] | tuple[Path, ...],
                   aoi_paths: list[Path] | tuple[Path, ...],
                   output_root: Path,
                   method: str = "max",
                   nodata: float | None = 0,
                   compress: str = "lzw") -> list[Path]:
    """Run the full workflow for multiple folders and AOI shapefiles."""
    output_root.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for folder in folders:
        if not folder.exists() or not folder.is_dir():
            log.warning("Skipping invalid folder: %s", folder)
            continue

        try:
            mosaic_path = mosaic_folder(folder, output_root, method=method, nodata=nodata, compress=compress)
            created.append(mosaic_path)
            log.info("Created mosaic: %s", mosaic_path)
        except FileNotFoundError:
            log.warning("No raster files found in %s; skipping", folder)
            continue

        for aoi_path in aoi_paths:
            if not aoi_path.exists():
                log.warning("AOI shapefile not found, skipping: %s", aoi_path)
                continue

            clipped_path = output_root / f"{mosaic_path.stem}_{aoi_path.stem}_clipped.tif"
            clipped_result = clip_raster_to_aoi(mosaic_path, aoi_path, clipped_path, nodata=nodata)
            created.append(clipped_result)
            log.info("Created clipped raster: %s", clipped_result)

    return created


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", required=True,
                     help="Input directory containing tiles (globbed with --glob), "
                          "or a single glob pattern with wildcards.")
    ap.add_argument("-o", "--output", required=True,
                     help="Output directory (one mosaic per group) or output .tif path "
                          "(single group only).")
    ap.add_argument("--glob", default="*.tif", help="Glob pattern used when --input is a directory (default: *.tif)")
    ap.add_argument("--method", default="first", choices=["first", "last", "min", "max", "sum", "count"],
                     help="How to resolve overlapping pixels (default: first)")
    ap.add_argument("--resampling", default="nearest",
                     choices=[r.name for r in Resampling],
                     help="Resampling for reprojection during merge (default: nearest; use "
                          "'bilinear'/'cubic' for continuous data like probability rasters)")
    ap.add_argument("--nodata", type=float, default=None, help="Override nodata value")
    ap.add_argument("--compress", default="deflate", help="Output compression (default: deflate)")
    ap.add_argument("--dtype", default=None, help="Force output dtype (default: keep source dtype)")
    args = ap.parse_args()

    resampling = Resampling[args.resampling]
    in_path = Path(args.input)
    out_path = Path(args.output)

    if in_path.is_dir():
        groups = discover_groups(in_path, args.glob)
        if not groups:
            log.error("No files matched %s in %s", args.glob, in_path)
            sys.exit(1)
        log.info("Found %d group(s): %s", len(groups), ", ".join(groups))
        for name, files in groups.items():
            out_file = out_path / f"{name}.tif" if out_path.suffix == "" else out_path
            mosaic_group(files, out_file, args.method, resampling, args.nodata, args.compress, args.dtype)
    else:
        # single explicit glob pattern -> one mosaic
        parent = in_path.parent
        pattern = in_path.name
        files = sorted(parent.glob(pattern))
        if not files:
            log.error("No files matched pattern %s", in_path)
            sys.exit(1)
        mosaic_group(files, out_path, args.method, resampling, args.nodata, args.compress, args.dtype)

    log.info("Done.")


if __name__ == "__main__":
    main()


import glob
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.transform import Affine

file_list = glob.glob(r"C:\Data Spasial\FAO_Binary_Commodities_FPDP\commodities_country_processed\2025_data_process\rice_clip\rice_process_2025\*.tif")

scale_factor = 10 / 100  # 10m -> 100m

downsampled_srcs = []
for fp in file_list:
    src = rasterio.open(fp)
    out_h = max(1, round(src.height * scale_factor))
    out_w = max(1, round(src.width * scale_factor))

    data = src.read(
        out_shape=(src.count, out_h, out_w),
        resampling=Resampling.average  # fraction of 10m presence per 100m cell
    )

    new_transform = src.transform * Affine.scale(
        src.width / out_w, src.height / out_h
    )

    # Wrap as an in-memory dataset so rasterio.merge can use it directly
    memfile = rasterio.io.MemoryFile()
    profile = src.profile.copy()
    profile.update(
        height=out_h, width=out_w, transform=new_transform,
        dtype=data.dtype
    )
    mem_ds = memfile.open(**profile)
    mem_ds.write(data)
    downsampled_srcs.append(mem_ds)
    src.close()

# Now merge — arrays are ~100x smaller, should fit comfortably in memory
mosaic, out_transform = merge(downsampled_srcs, method='max', nodata=0)

out_meta = downsampled_srcs[0].meta.copy()
out_meta.update({
    "driver": "GTiff",
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": out_transform,
    "nodata": 0,
    "compress": "lzw"
})

with rasterio.open("SEA_rice_mosaic_100m_bhtan.tif", "w", **out_meta) as dst:
    dst.write(mosaic)

for ds in downsampled_srcs:
    ds.close()