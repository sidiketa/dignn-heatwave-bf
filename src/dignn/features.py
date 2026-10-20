"""Feature engineering, normalisation and sequence construction."""

import numpy as np
import pandas as pd

from .config import TRAIN_YEARS, C_IN, C_OUT


def add_features(df, oni_data, nao_data):
    """Add features following Li et al. (2023) extended with NAO teleconnection."""
    print("\n6. Adding features...")

    df = df.copy()

    # Day of year — encodage cyclique (preserve continuity 31-Dec -> 1-Jan)
    df['DOY_sin'] = np.sin(2 * np.pi * df['YDAY'] / 365)
    df['DOY_cos'] = np.cos(2 * np.pi * df['YDAY'] / 365)

    # ONI — indice ENSO (moyenne glissante 3 mois)
    df['ONI'] = df.apply(
        lambda row: oni_data.get((row['YEAR'], row['MONTH']), 0),
        axis=1
    )

    # NAO — indice mensuel de l'Oscillation Nord-Atlantique
    df['NAO'] = df.apply(
        lambda row: nao_data.get((row['YEAR'], row['MONTH']), 0),
        axis=1
    )

    print(f"   Added DOY_sin, DOY_cos, ONI, NAO")

    return df


def normalize_features(df, feature_cols):
    """Normalize features using training set statistics."""
    print("\n10. Normalizing features...")
    
    df = df.copy()
    train_mask = df['YEAR'].isin(TRAIN_YEARS)
    
    norm_params = {}
    
    for col in feature_cols:
        if col in df.columns and col not in ['beta', 'NAME', 'YEARMODA', 'YEAR']:
            mean = df.loc[train_mask, col].mean()
            std = df.loc[train_mask, col].std()
            
            if std == 0 or np.isnan(std):
                std = 1
            
            df[col] = (df[col] - mean) / std
            norm_params[col] = {'mean': mean, 'std': std}
    
    print(f"   Normalized {len(norm_params)} features")
    
    return df, norm_params


def create_sequences(df, stations, c_in=C_IN, c_out=C_OUT):
    """Create sequences for training."""
    print(f"\n11. Creating sequences (C_in={c_in}, C_out={c_out})...")
    
    # Feature columns
    feature_cols = [
        'beta', 'MAX', 'MIN', 'DEWP', 'RH', 'WDSP', 'PRCP', 'SLP',
        'LONGITUDE', 'LATITUDE', 'DOY_sin', 'DOY_cos', 'ONI', 'NAO', 'Tthresh',
        'GPD_xi', 'GPD_sigma', 'GPD_mu', 'GPD_var', 'GPD_q95'
    ]
    
    # Filter to available columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"   Using {len(feature_cols)} features")
    
    # Get unique dates
    dates = sorted(df['YEARMODA'].unique())
    
    X_list, y_list, years_list = [], [], []
    
    for i in range(len(dates) - c_in - c_out + 1):
        input_dates = dates[i:i+c_in]
        output_dates = dates[i+c_in:i+c_in+c_out]
        
        # Check all stations have data
        input_df = df[df['YEARMODA'].isin(input_dates)]
        output_df = df[df['YEARMODA'].isin(output_dates)]
        
        if len(input_df['NAME'].unique()) != len(stations):
            continue
        if len(output_df['NAME'].unique()) != len(stations):
            continue
        
        # Build tensors
        X_seq = np.zeros((len(stations), c_in, len(feature_cols)))
        y_seq = np.zeros((len(stations), c_out))
        
        valid = True
        for s_idx, station in enumerate(stations):
            station_input = input_df[input_df['NAME'] == station].sort_values('YEARMODA')
            station_output = output_df[output_df['NAME'] == station].sort_values('YEARMODA')
            
            if len(station_input) != c_in or len(station_output) != c_out:
                valid = False
                break
            
            X_seq[s_idx] = station_input[feature_cols].values
            y_seq[s_idx] = station_output['beta'].values
        
        if valid:
            X_list.append(X_seq)
            y_list.append(y_seq)
            years_list.append(int(input_dates[-1][:4]))
    
    X = np.array(X_list)
    y = np.array(y_list)
    years = np.array(years_list)
    
    print(f"   Created {len(X):,} sequences")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Positive labels: {y.sum():.0f} ({y.sum()/y.size*100:.2f}%)")
    
    return X, y, years, feature_cols
