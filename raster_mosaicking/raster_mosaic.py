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
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.merge import merge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# GEE sharded exports typically look like: <name>-<row>-<col>.tif or <name>0000000000-0000000000.tif
TILE_SUFFIX_RE = re.compile(r"[-_]?\d{4,}[-_]\d{4,}$")


def group_key(path: Path) -> str:
    """Derive a region/group name by stripping trailing tile-index suffixes."""
    stem = path.stem
    key = TILE_SUFFIX_RE.sub("", stem)
    return key or stem


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