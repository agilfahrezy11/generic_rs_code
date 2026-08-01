"""
remap_classes.py

Generic tool to remap training-data shapefile ID/class fields by combining
multiple original classes into a single new class.

Example use case:
    Original:  1 = Primary Forest, 2 = Secondary Forest
    Remapped:  1 = Forest  (both original IDs merged into one new class)

Can be used two ways:

1. Standalone from the command line:
       python remap_classes.py --input training.shp --output training_remapped.shp

   Edit the REMAP_RULES / ID_COL / CLASS_COL constants below first, or pass
   a JSON rules file with --rules rules.json (see build_mapping_from_rules
   docstring for the expected format).

2. Imported in a Jupyter notebook:
       from remap_classes import remap_shapefile, remap_geodataframe

       gdf_new = remap_geodataframe(gdf, id_col="ID", class_col="ClassName",
                                     rules=REMAP_RULES)

A report of the remapping process (rules applied, original vs. new class
distribution, any unmapped features) is generated automatically. Pass
report_path=... (or --report on the CLI) to save it — use a .md/.txt
extension for a readable Markdown summary, or .csv for a flat mapping
table. It's also always available via gdf_new.attrs["remap_report"] after
calling remap_shapefile(), and can be built directly with
generate_remap_report() for full control (e.g. inside a notebook).

Requires: geopandas, pandas
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import geopandas as gpd
import pandas as pd

# --------------------------------------------------------------------------
# USER CONFIGURATION
# Edit these to match your shapefile and desired remapping.
# These are only used when running this file directly (python remap_classes.py
# with no --rules/--id-col/--class-col overrides). When importing functions
# into a notebook, you can just pass your own values as arguments instead.
# --------------------------------------------------------------------------

# Column name in the shapefile that holds the original class ID (int)
ID_COL = "ID"

# Column name in the shapefile that holds the original class name (str)
CLASS_COL = "ClassName"

# Column names to write the remapped ID/class into.
# Set these equal to ID_COL / CLASS_COL if you want to overwrite in place.
NEW_ID_COL = "new_id"
NEW_CLASS_COL = "new_class"

# Define your remap rules here. Each rule is one *new* (merged) class.
# "old_ids" and/or "old_names" list which original records get folded into
# this new class. You can use either or both — a record matches a rule if
# its ID is in old_ids OR its class name is in old_names.
#
# Example (matches the Forest example in the prompt):
REMAP_RULES: List[Dict[str, Any]] = [
    {
        "new_id": 1,
        "new_class": "Forest",
        "old_ids": [1, 2],
        "old_names": ["Primary Forest", "Secondary Forest"],
    },
    # Add more rules below, e.g.:
    # {
    #     "new_id": 2,
    #     "new_class": "Agriculture",
    #     "old_ids": [3, 4, 5],
    #     "old_names": ["Paddy Field", "Plantation", "Cropland"],
    # },
]


# --------------------------------------------------------------------------
# CORE FUNCTIONS
# --------------------------------------------------------------------------

def build_mapping_from_rules(
    rules: List[Dict[str, Any]]
) -> Dict[str, tuple]:
    """
    Flatten a list of remap rules into two lookup dicts keyed by the
    original ID and original class name.

    Each rule looks like:
        {
            "new_id": 1,
            "new_class": "Forest",
            "old_ids": [1, 2],          # optional
            "old_names": ["Primary Forest", "Secondary Forest"],  # optional
        }

    Returns
    -------
    dict with two keys:
        "by_id":   {old_id: (new_id, new_class)}
        "by_name": {old_class_name: (new_id, new_class)}
    """
    by_id: Dict[Any, tuple] = {}
    by_name: Dict[str, tuple] = {}

    for rule in rules:
        if "new_id" not in rule or "new_class" not in rule:
            raise ValueError(
                f"Each rule needs 'new_id' and 'new_class'. Bad rule: {rule}"
            )
        new_id = rule["new_id"]
        new_class = rule["new_class"]

        for old_id in rule.get("old_ids", []):
            if old_id in by_id:
                raise ValueError(
                    f"old_id {old_id} is mapped by more than one rule "
                    f"(conflict with new_class '{by_id[old_id][1]}' vs '{new_class}')."
                )
            by_id[old_id] = (new_id, new_class)

        for old_name in rule.get("old_names", []):
            if old_name in by_name:
                raise ValueError(
                    f"old_name '{old_name}' is mapped by more than one rule "
                    f"(conflict with new_class '{by_name[old_name][1]}' vs '{new_class}')."
                )
            by_name[old_name] = (new_id, new_class)

    return {"by_id": by_id, "by_name": by_name}


def remap_geodataframe(
    gdf: gpd.GeoDataFrame,
    id_col: str,
    class_col: str,
    rules: List[Dict[str, Any]],
    new_id_col: str = "new_id",
    new_class_col: str = "new_class",
    drop_unmapped: bool = False,
) -> gpd.GeoDataFrame:
    """
    Apply remap rules to a GeoDataFrame and return a copy with new
    ID/class columns added.

    Matching priority: a feature is matched by old_id first; if no old_id
    match is found, it falls back to matching by old_name.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input training data with an ID column and a class name column.
    id_col : str
        Name of the existing ID column in gdf.
    class_col : str
        Name of the existing class name column in gdf.
    rules : list of dict
        See build_mapping_from_rules() for the expected rule format.
    new_id_col, new_class_col : str
        Names for the newly created columns.
    drop_unmapped : bool
        If True, rows that don't match any rule are dropped. If False
        (default), they are kept with new_id_col/new_class_col set to
        NaN/None, and a warning listing the unmapped values is printed.

    Returns
    -------
    GeoDataFrame
        Copy of the input with new_id_col and new_class_col added.
    """
    if id_col not in gdf.columns:
        raise KeyError(f"id_col '{id_col}' not found in GeoDataFrame columns: {list(gdf.columns)}")
    if class_col not in gdf.columns:
        raise KeyError(f"class_col '{class_col}' not found in GeoDataFrame columns: {list(gdf.columns)}")

    mapping = build_mapping_from_rules(rules)
    by_id, by_name = mapping["by_id"], mapping["by_name"]

    out = gdf.copy()
    new_ids = []
    new_classes = []
    unmapped_pairs = set()

    for old_id, old_name in zip(out[id_col], out[class_col]):
        match = by_id.get(old_id)
        if match is None:
            match = by_name.get(old_name)

        if match is None:
            new_ids.append(pd.NA)
            new_classes.append(None)
            unmapped_pairs.add((old_id, old_name))
        else:
            new_ids.append(match[0])
            new_classes.append(match[1])

    out[new_id_col] = new_ids
    out[new_class_col] = new_classes

    if unmapped_pairs:
        print(
            f"[remap_classes] WARNING: {len(unmapped_pairs)} original "
            f"(ID, class) pair(s) had no matching rule and were left unmapped:"
        )
        for pair in sorted(unmapped_pairs, key=lambda p: str(p[0])):
            print(f"    {pair}")

    if drop_unmapped:
        before = len(out)
        out = out[out[new_id_col].notna()].copy()
        dropped = before - len(out)
        if dropped:
            print(f"[remap_classes] Dropped {dropped} unmapped row(s).")

    return out


def build_mapping_table(
    gdf_after: gpd.GeoDataFrame,
    id_col: str,
    class_col: str,
    new_id_col: str,
    new_class_col: str,
) -> pd.DataFrame:
    """
    Build a flat mapping table: one row per unique (old_id, old_class) ->
    (new_id, new_class) combination actually present in the data, with a
    count of how many features fall in that combination.

    Useful as a CSV export for documentation / QA (e.g. "these are the
    exact rows that got merged into which new class").
    """
    table = (
        gdf_after.groupby([id_col, class_col, new_id_col, new_class_col], dropna=False)
        .size()
        .reset_index(name="count")
        .rename(columns={id_col: "old_id", class_col: "old_class",
                         new_id_col: "new_id", new_class_col: "new_class"})
        .sort_values(["new_id", "old_id"], na_position="last")
        .reset_index(drop=True)
    )
    return table


def generate_remap_report(
    gdf_before: gpd.GeoDataFrame,
    gdf_after: gpd.GeoDataFrame,
    id_col: str,
    class_col: str,
    new_id_col: str,
    new_class_col: str,
    rules: List[Dict[str, Any]],
    input_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Build a human-readable Markdown report documenting a remap run:
    rules applied, original class distribution, new class distribution,
    and any unmapped (ID, class) pairs.

    Returns the report as a string (does not write to disk — use
    save_report() or pass report_path to remap_shapefile() for that).
    """
    original_counts = (
        gdf_before.groupby([id_col, class_col]).size()
        .reset_index(name="count").sort_values(id_col)
    )
    new_counts = (
        gdf_after.groupby([new_id_col, new_class_col], dropna=False).size()
        .reset_index(name="count").sort_values(new_id_col, na_position="last")
    )
    unmapped = (
        gdf_after.loc[gdf_after[new_id_col].isna(), [id_col, class_col]]
        .drop_duplicates()
        .sort_values(id_col)
    )

    total_before = len(gdf_before)
    total_after = len(gdf_after)
    dropped = total_before - total_after

    lines: List[str] = []
    lines.append("# Class Remapping Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    if input_path is not None:
        lines.append(f"- Input file: `{input_path}`")
    if output_path is not None:
        lines.append(f"- Output file: `{output_path}`")
    lines.append(f"- ID column -> new ID column: `{id_col}` -> `{new_id_col}`")
    lines.append(f"- Class column -> new class column: `{class_col}` -> `{new_class_col}`")
    lines.append(f"- Total input features: {total_before}")
    lines.append(f"- Total output features: {total_after}")
    if dropped:
        lines.append(f"- Dropped (unmapped) features: {dropped}")
    lines.append(f"- Original unique (ID, class) combinations: {len(original_counts)}")
    lines.append(f"- New unique classes: {gdf_after[new_class_col].nunique(dropna=True)}")
    lines.append("")

    lines.append("## Remap Rules Applied")
    lines.append("")
    lines.append("| new_id | new_class | old_ids | old_names |")
    lines.append("|---|---|---|---|")
    for rule in rules:
        old_ids = ", ".join(str(x) for x in rule.get("old_ids", [])) or "-"
        old_names = ", ".join(rule.get("old_names", [])) or "-"
        lines.append(f"| {rule['new_id']} | {rule['new_class']} | {old_ids} | {old_names} |")
    lines.append("")

    lines.append("## Original Class Distribution")
    lines.append("")
    lines.append(f"| {id_col} | {class_col} | count |")
    lines.append("|---|---|---|")
    for _, row in original_counts.iterrows():
        lines.append(f"| {row[id_col]} | {row[class_col]} | {row['count']} |")
    lines.append("")

    lines.append("## New (Remapped) Class Distribution")
    lines.append("")
    lines.append(f"| {new_id_col} | {new_class_col} | count |")
    lines.append("|---|---|---|")
    for _, row in new_counts.iterrows():
        nid = row[new_id_col]
        ncls = row[new_class_col]
        if pd.isna(nid):
            nid_str = "unmapped"
        else:
            nid_str = str(int(nid)) if float(nid).is_integer() else str(nid)
        ncls_str = "unmapped" if pd.isna(ncls) else str(ncls)
        lines.append(f"| {nid_str} | {ncls_str} | {row['count']} |")
    lines.append("")

    lines.append("## Unmapped Original (ID, Class) Pairs")
    lines.append("")
    if len(unmapped) > 0:
        lines.append(
            "These did not match any remap rule. Add a rule to cover them, "
            "or run with `drop_unmapped=True` / `--drop-unmapped` to discard them."
        )
        lines.append("")
        lines.append(f"| {id_col} | {class_col} |")
        lines.append("|---|---|")
        for _, row in unmapped.iterrows():
            lines.append(f"| {row[id_col]} | {row[class_col]} |")
    else:
        lines.append("None — all input features matched a rule.")
    lines.append("")

    return "\n".join(lines)


def save_report(
    gdf_after: gpd.GeoDataFrame,
    report_text: str,
    report_path: Union[str, Path],
    id_col: Optional[str] = None,
    class_col: Optional[str] = None,
    new_id_col: Optional[str] = None,
    new_class_col: Optional[str] = None,
) -> Path:
    """
    Write a report to disk. If report_path ends in .csv, writes the flat
    mapping table (build_mapping_table) instead of the Markdown text —
    id_col/class_col/new_id_col/new_class_col are required in that case.
    Otherwise writes report_text as-is (.md/.txt/etc).
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if report_path.suffix.lower() == ".csv":
        missing = [n for n, v in [("id_col", id_col), ("class_col", class_col),
                                   ("new_id_col", new_id_col), ("new_class_col", new_class_col)] if v is None]
        if missing:
            raise ValueError(f"CSV report requires {missing} to be provided.")
        table = build_mapping_table(gdf_after, id_col, class_col, new_id_col, new_class_col)
        table.to_csv(report_path, index=False)
    else:
        report_path.write_text(report_text)

    print(f"[remap_classes] Wrote report to: {report_path}")
    return report_path


def remap_shapefile(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    id_col: str = ID_COL,
    class_col: str = CLASS_COL,
    rules: Optional[List[Dict[str, Any]]] = None,
    new_id_col: str = NEW_ID_COL,
    new_class_col: str = NEW_CLASS_COL,
    drop_unmapped: bool = False,
    report_path: Optional[Union[str, Path]] = None,
) -> gpd.GeoDataFrame:
    """
    Read a shapefile (or any OGR-readable vector file), remap its ID/class
    columns, and write the result out.

    Output format is inferred from the output_path extension
    (.shp, .gpkg, .geojson, etc. — anything Fiona/pyogrio can write).

    Returns the remapped GeoDataFrame as well, so it can be inspected
    immediately after calling this from a notebook.
    """
    if rules is None:
        rules = REMAP_RULES

    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"[remap_classes] Reading: {input_path}")
    gdf = gpd.read_file(input_path)
    print(f"[remap_classes] Loaded {len(gdf)} features. Columns: {list(gdf.columns)}")

    remapped = remap_geodataframe(
        gdf,
        id_col=id_col,
        class_col=class_col,
        rules=rules,
        new_id_col=new_id_col,
        new_class_col=new_class_col,
        drop_unmapped=drop_unmapped,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    remapped.to_file(output_path)
    print(f"[remap_classes] Wrote {len(remapped)} features to: {output_path}")

    report_text = generate_remap_report(
        gdf_before=gdf,
        gdf_after=remapped,
        id_col=id_col,
        class_col=class_col,
        new_id_col=new_id_col,
        new_class_col=new_class_col,
        rules=rules,
        input_path=input_path,
        output_path=output_path,
    )

    # quick summary printed to console regardless of report_path
    summary = (
        remapped.groupby([new_id_col, new_class_col], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(new_id_col)
    )
    print("[remap_classes] New class summary:")
    print(summary.to_string(index=False))

    if report_path is not None:
        save_report(
            remapped, report_text, report_path,
            id_col=id_col, class_col=class_col,
            new_id_col=new_id_col, new_class_col=new_class_col,
        )

    remapped.attrs["remap_report"] = report_text
    return remapped


def load_rules_from_json(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load remap rules from a JSON file, e.g.:

    [
      {"new_id": 1, "new_class": "Forest", "old_ids": [1, 2]},
      {"new_id": 2, "new_class": "Agriculture", "old_ids": [3, 4, 5]}
    ]
    """
    with open(path, "r") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Remap a shapefile's ID/class columns by merging classes."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input shapefile/vector file")
    parser.add_argument("--output", "-o", required=True, help="Path to output vector file")
    parser.add_argument("--id-col", default=ID_COL, help=f"Original ID column name (default: {ID_COL})")
    parser.add_argument("--class-col", default=CLASS_COL, help=f"Original class name column (default: {CLASS_COL})")
    parser.add_argument("--new-id-col", default=NEW_ID_COL, help=f"Output new ID column name (default: {NEW_ID_COL})")
    parser.add_argument("--new-class-col", default=NEW_CLASS_COL, help=f"Output new class column name (default: {NEW_CLASS_COL})")
    parser.add_argument("--rules", help="Path to a JSON file defining remap rules (overrides REMAP_RULES in this file)")
    parser.add_argument("--drop-unmapped", action="store_true", help="Drop rows that don't match any rule instead of keeping them as NaN")
    parser.add_argument("--report", help="Path to write a remap report. Use a .csv extension for a flat mapping table, or .md/.txt for a readable Markdown summary.")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    rules = load_rules_from_json(args.rules) if args.rules else REMAP_RULES

    remap_shapefile(
        input_path=args.input,
        output_path=args.output,
        id_col=args.id_col,
        class_col=args.class_col,
        rules=rules,
        new_id_col=args.new_id_col,
        new_class_col=args.new_class_col,
        drop_unmapped=args.drop_unmapped,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()