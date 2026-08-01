# remap-classes

Remap a shapefile's ID/class fields by merging multiple original classes into new combined classes
(e.g. `1=Primary Forest` + `2=Secondary Forest` -> `1=Forest`).

## Install

```bash
pip install -e .
```

## Usage

CLI (edit `REMAP_RULES` in `remap_classes.py` first, or use `--rules rules.json`):

```bash
remap-classes --input training.shp --output training_remapped.shp
# or, without installing:
python remap_classes.py --input training.shp --output training_remapped.shp
```

Jupyter / import:

```python
from remap_classes import remap_shapefile, REMAP_RULES

gdf_new = remap_shapefile(
    "training.shp", "training_remapped.shp",
    id_col="ID", class_col="ClassName", rules=REMAP_RULES,
)
```

## Report

Every run automatically builds a report documenting what happened: rules applied, original vs.
new class distribution, and any unmapped features. Save it with `--report` (CLI) or
`report_path=` (Python):

```bash
remap-classes --input training.shp --output training_remapped.shp --report report.md
# or a flat CSV mapping table instead:
remap-classes --input training.shp --output training_remapped.shp --report mapping.csv
```

```python
gdf_new = remap_shapefile(..., report_path="report.md")
print(gdf_new.attrs["remap_report"])  # report text is always available here too
```

See `remap_classes.py` for full documentation of the rule format and all function signatures.