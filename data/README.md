# Data

Data files are not tracked in this repository.

Daily station observations come from the NOAA Global Surface Summary of the Day (GSOD), which is freely available from NOAA.
ANAM-BF observations were used only to cross-validate GSOD and are not distributed.

Expected input files in this folder:

| File | Content |
|---|---|
| `gsod_bf.csv` | Daily GSOD observations for Burkina Faso (NAME, YEARMODA, YEAR, MONTH, YDAY, LATITUDE, LONGITUDE, ELEVATION, MAX, MIN, TEMP, DEWP, RH, WDSP, PRCP, SLP, STP, VISIB, MXSPD), temperatures in °C |
| `oni` | Monthly Oceanic Niño Index, one line per year (year + 12 values) |
| `nao` | Monthly North Atlantic Oscillation index, one line per year (year + 12 values) |
| `anam_bf.csv` | Optional. ANAM-BF observations used to cross-validate GSOD |
| `shapefiles/` | Optional. National boundary from OCHA HDX, for maps |

To test the pipeline without the real files, generate a synthetic set with the same format:

```bash
python scripts/make_synthetic_data.py --out data/sample
export DIGNN_DATA_DIR=data/sample
```
