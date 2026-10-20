"""
Generate a synthetic dataset with the same format as the real inputs.

The real station data belong to ANAM-BF and cannot be shared.
This script writes three files with identical structure so the pipeline can be run end to end:
    gsod_bf.csv  daily station observations, 2010-2024
    oni          monthly Oceanic Nino Index (year + 12 columns)
    nao          monthly North Atlantic Oscillation index (year + 12 columns)

Values are random draws around a plausible Sahelian seasonal cycle.
They carry no scientific meaning.

Usage
-----
    python scripts/make_synthetic_data.py --out data/sample
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Approximate public coordinates of the synoptic stations (lat, lon, elevation in m)
STATIONS = {
    "BOBO DIOULASSO": (11.17, -4.32, 460),
    "BOROMO":         (11.75, -2.93, 271),
    "DEDOUGOU":       (12.47, -3.48, 300),
    "DORI":           (14.03, -0.03, 277),
    "FADA N'GOURMA":  (12.03,  0.37, 309),
    "GAOUA":          (10.33, -3.18, 333),
    "OUAGADOUGOU":    (12.35, -1.52, 306),
    "OUAHIGOUYA":     (13.57, -2.42, 336),
    "PO":             (11.15, -1.15, 322),
}


def ar1_noise(n, phi, sigma, rng):
    """AR(1) noise, so that warm anomalies persist over several days."""
    eps = rng.normal(0, sigma, n)
    out = np.zeros(n)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + eps[t]
    return out


def make_station(name, lat, lon, elev, dates, rng):
    n = len(dates)
    doy = dates.dayofyear.values
    north = (lat - 10.0) / 4.0

    # Tmax: main peak in April, secondary peak in October, cooler during the monsoon
    tmax = (34.0 + 1.5 * north
            + 3.5 * np.cos(2 * np.pi * (doy - 105) / 365)
            + 1.2 * np.cos(4 * np.pi * (doy - 290) / 365)
            + ar1_noise(n, 0.8, 1.0, rng))
    tmin = tmax - 11.0 - 2.0 * np.cos(2 * np.pi * (doy - 15) / 365) + rng.normal(0, 1.0, n)
    temp = (tmax + tmin) / 2

    monsoon = np.clip(np.cos(2 * np.pi * (doy - 225) / 365), 0, None)
    rh = np.clip(25 + 50 * monsoon + rng.normal(0, 6, n), 5, 100)
    dewp = temp - (100 - rh) / 5.0
    prcp = np.where(rng.random(n) < 0.05 + 0.45 * monsoon, rng.gamma(0.8, 12, n), 0.0)
    wdsp = np.clip(2.0 + rng.normal(0, 0.8, n), 0, None)
    slp = 1011 - 3 * monsoon + rng.normal(0, 1.5, n)

    df = pd.DataFrame({
        "NAME": name,
        "YEARMODA": dates.strftime("%Y-%m-%d"),
        "YEAR": dates.year,
        "MONTH": dates.month,
        "YDAY": doy,
        "LATITUDE": lat,
        "LONGITUDE": lon,
        "ELEVATION": elev,
        "MAX": tmax.round(1),
        "MIN": tmin.round(1),
        "TEMP": temp.round(1),
        "DEWP": dewp.round(1),
        "RH": rh.round(1),
        "WDSP": wdsp.round(1),
        "PRCP": prcp.round(1),
        "SLP": slp.round(1),
        "STP": (slp - elev / 8.3).round(1),
        "VISIB": np.clip(rng.normal(8, 2, n), 0.5, None).round(1),
        "MXSPD": (wdsp + np.abs(rng.normal(2, 1, n))).round(1),
    })

    # About 1 % of scattered missing values
    for col in ["MAX", "MIN", "DEWP", "RH", "WDSP", "PRCP", "SLP"]:
        df.loc[rng.random(n) < 0.01, col] = np.nan

    # Long gap at Dori, so that the quality filter excludes it as in the real data
    if name == "DORI":
        gap = (df["YEARMODA"] >= "2016-03-01") & (df["YEARMODA"] < "2016-09-01")
        df.loc[gap, ["MAX", "MIN"]] = np.nan

    return df


def write_index(path, header, years, rng, scale):
    lines = [header]
    for y in years:
        vals = " ".join(f"{v:.2f}" for v in rng.normal(0, scale, 12))
        lines.append(f"{y} {vals}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out", default="data/sample", help="output folder")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    dates = pd.date_range("2010-01-01", "2024-12-31", freq="D")
    gsod = pd.concat(
        [make_station(n, *meta, dates, rng) for n, meta in STATIONS.items()],
        ignore_index=True)
    gsod.to_csv(out / "gsod_bf.csv", index=False)

    years = range(2009, 2025)
    write_index(out / "oni", "YEAR DJF JFM FMA MAM AMJ MJJ JJA JAS ASO SON OND NDJ", years, rng, 0.8)
    write_index(out / "nao", "Year Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec", years, rng, 1.0)

    print(f"Wrote {len(gsod):,} rows for {gsod['NAME'].nunique()} stations to {out}")


if __name__ == "__main__":
    main()
