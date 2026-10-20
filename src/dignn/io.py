"""Loading of station observations (GSOD) and climate indices (ONI, NAO)."""

import numpy as np
import pandas as pd

from .config import GSOD_FILE, ONI_FILE, NAO_FILE


def load_gsod_data():
    """Load GSOD data for Burkina Faso."""
    print("\n1. Loading GSOD data...")
    
    df = pd.read_csv(GSOD_FILE)
    print(f"   Loaded {len(df):,} records")
    print(f"   Stations: {df['NAME'].nunique()}")
    print(f"   Years: {df['YEAR'].min()}-{df['YEAR'].max()}")
    
    return df


def load_oni_data():
    """Load Oceanic Niño Index data."""
    print("\n2. Loading ONI data...")
    
    oni_data = {}
    
    if ONI_FILE.exists():
        with open(ONI_FILE, 'r') as f:
            lines = f.readlines()
        
        for line in lines[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 13:
                year = int(parts[0])
                # Monthly ONI values (use 3-month running mean columns)
                months = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 
                         'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']
                for i, month in enumerate(months):
                    try:
                        oni_data[(year, i+1)] = float(parts[i+1])
                    except (ValueError, IndexError):
                        pass
        
        print(f"   Loaded ONI for {len(oni_data)} year-month pairs")
    else:
        print("   WARNING: ONI file not found, using zeros")
    
    return oni_data


def load_nao_data():
    """
    Load North Atlantic Oscillation (NAO) index data.

    Format attendu : colonnes Year + 12 mois (Jan-Dec), valeurs mensuelles.
    Source : NOAA Climate Prediction Center.

    Returns
    -------
    dict
        Dictionnaire {(year, month): nao_value}
    """
    print("\n2b. Loading NAO data...")

    nao_data = {}

    if NAO_FILE.exists():
        with open(NAO_FILE, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 13:
                year = int(parts[0])
                for month in range(1, 13):
                    try:
                        val = float(parts[month])
                        if val > -99:  # Valeurs manquantes codees -99.99
                            nao_data[(year, month)] = val
                    except (ValueError, IndexError):
                        pass

        print(f"   Loaded NAO for {len(nao_data)} year-month pairs")
    else:
        print("   WARNING: NAO file not found, using zeros")

    return nao_data
