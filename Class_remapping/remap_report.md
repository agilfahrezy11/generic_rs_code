# Class Remapping Report

- Generated: 2026-08-01T21:33:05
- Input file: `C:\Users\AFahrezi\Documents\GitHub\generic_rs_code\Class_remapping\Data\TrainingData_2016_OganIlir.shp`
- Output file: `Remap_TD_2016_OganIlir.shp`
- ID column -> new ID column: `ID_epistem` -> `new_id`
- Class column -> new class column: `NameID` -> `new_class`
- Total input features: 766
- Total output features: 742
- Dropped (unmapped) features: 24
- Original unique (ID, class) combinations: 12
- New unique classes: 10

## Remap Rules Applied

| new_id | new_class | old_ids | old_names |
|---|---|---|---|
| 1 | Hutan Lahan Kering Sekunder | 1 | - |
| 2 | Hutan Rawa Sekunder | 2 | - |
| 3 | Perkebunan Karet | 3, 6 | Karet Monokultur, Agroforestri Karet |
| 4 | Sawit Monokultur | 4 | - |
| 5 | Pertanian Lahan Kering | 5 | - |
| 6 | Sawah | 8 | - |
| 7 | Padang Rumput | 9 | - |
| 8 | Permukiman | 10 | - |
| 9 | Lahan Terbuka | 11 | - |
| 10 | Badan Air | 12 | - |

## Original Class Distribution

| ID_epistem | NameID | count |
|---|---|---|
| 1 | Hutan Lahan Kering Sekunder | 51 |
| 2 | Hutan Rawa Sekunder | 82 |
| 3 | Karet Monokultur | 71 |
| 4 | Sawit Monokultur | 106 |
| 5 | Pertanian Lahan Kering | 56 |
| 6 | Agroforestri Karet | 35 |
| 7 | Kebun Campuran | 24 |
| 8 | Sawah | 37 |
| 9 | Padang Rumput | 24 |
| 10 | Permukiman | 100 |
| 11 | Lahan Terbuka | 80 |
| 12 | Badan Air | 100 |

## New (Remapped) Class Distribution

| new_id | new_class | count |
|---|---|---|
| 1 | Hutan Lahan Kering Sekunder | 51 |
| 2 | Hutan Rawa Sekunder | 82 |
| 3 | Perkebunan Karet | 106 |
| 4 | Sawit Monokultur | 106 |
| 5 | Pertanian Lahan Kering | 56 |
| 6 | Sawah | 37 |
| 7 | Padang Rumput | 24 |
| 8 | Permukiman | 100 |
| 9 | Lahan Terbuka | 80 |
| 10 | Badan Air | 100 |

## Unmapped Original (ID, Class) Pairs

None — all input features matched a rule.
