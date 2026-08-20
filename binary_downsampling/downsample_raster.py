"""
downsample_rasters.py

Batch downsample binary presence/absence rasters (e.g. rice 10m -> 100m)
using proportional (fractional) aggregation, which is the statistically
correct way to resample a categorical/binary variable to a coarser grid.

Default behavior: outputs a float32 "fractional presence" raster per 100m
cell (0.0-1.0), computed via Resampling.average over the 10x10 block of
10m pixels. Optionally thresholds back to a binary 0/1 product.

Usage:
    python downsample_rasters.py -i /path/to/10m_folder -o /path/to/100m_folder
    python downsample_rasters.py -i in/ -o out/ --factor 10
    python downsample_rasters.py -i in/ -o out/ --threshold 0.5   # also write binary version
"""

import argparse
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling

def downsample_raster(
    src_path: Path,
    dst_path: Path,
    factor: int = 10,
    resampling: Resampling = Resampling.average,
    nodata: float | None = None,
) -> np.ndarray:
    """
    Downsample a single raster by an integer factor using a decimated read.

    Using out_shape in src.read() lets GDAL do the aggregation during read,
    which is far more memory-efficient than reading the full 10m array into
    memory and reducing it manually with numpy/skimage.

    Returns the output array (so it can optionally be reused, e.g. for
    thresholding to binary, without re-reading from disk).
    """
    with rasterio.open(src_path) as src:
        new_width = max(1, src.width // factor)
        new_height = max(1, src.height // factor)

        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=resampling,
        )

        new_transform = src.transform * src.transform.scale(
            src.width / new_width, src.height / new_height
        )

        profile = src.profile.copy()
        profile.update(
            {
                "height": new_height,
                "width": new_width,
                "transform": new_transform,
                "dtype": data.dtype,
                "nodata": nodata if nodata is not None else src.nodata,
                "compress": "lzw",
            }
        )

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data)

    return data


def process_folder(
    in_folder: Path,
    out_folder: Path,
    factor: int = 10,
    threshold: float | None = None,
    pattern: str = "*.tif",
) -> None:
    files = sorted(in_folder.glob(pattern))
    if not files:
        print(f"No files matching '{pattern}' found in {in_folder}")
        return

    for f in files:
        out_path = out_folder / f.name
        print(f"[{f.name}] downsampling {factor}x -> fractional cover ...")

        try:
            frac = downsample_raster(
                f, out_path, factor=factor, resampling=Resampling.average
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        if threshold is not None:
            bin_path = out_folder / f"{f.stem}_binary{f.suffix}"
            with rasterio.open(out_path) as ref:
                profile = ref.profile.copy()

            binary = (frac >= threshold).astype("uint8")
            profile.update(dtype="uint8", nodata=None, compress="lzw")
            with rasterio.open(bin_path, "w", **profile) as dst:
                dst.write(binary)
            print(f"  -> {out_path.name} (fractional), {bin_path.name} (binary @ {threshold})")
        else:
            print(f"  -> {out_path.name} (fractional)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", required=True, type=Path, help="Folder of 10m rasters")
    ap.add_argument("-o", "--output", required=True, type=Path, help="Output folder for 100m rasters")
    ap.add_argument("--factor", type=int, default=10, help="Downsample factor (10m*10 = 100m). Default 10.")
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="If set, also write a binary 0/1 raster using fraction >= threshold (e.g. 0.5).",
    )
    ap.add_argument("--pattern", type=str, default="*.tif", help="Glob pattern for input files. Default *.tif")
    args = ap.parse_args()

    process_folder(
        in_folder=args.input,
        out_folder=args.output,
        factor=args.factor,
        threshold=args.threshold,
        pattern=args.pattern,
    )


if __name__ == "__main__":
    main()