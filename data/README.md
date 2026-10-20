# Data

Station observations belong to the National Meteorological Agency of Burkina Faso (ANAM-BF).
They are not distributed with this repository.

Expected input files in this folder:

| File | Content |
|---|---|
| `gsod_bf.csv` | Daily station observations (NAME, YEARMODA, YEAR, MONTH, YDAY, LATITUDE, LONGITUDE, ELEVATION, MAX, MIN, TEMP, DEWP, RH, WDSP, PRCP, SLP, STP, VISIB, MXSPD) |
| `oni` | Monthly Oceanic Niño Index, one line per year (year + 12 values) |
| `nao` | Monthly North Atlantic Oscillation index, one line per year (year + 12 values) |
| `anam_bf.csv` | Optional. ANAM observations used to cross-validate GSOD |
| `shapefiles/` | Optional. National boundary from OCHA HDX, for maps |

To test the pipeline without the real data, generate a synthetic set with the same format:

```bash
python scripts/make_synthetic_data.py --out data/sample
```
