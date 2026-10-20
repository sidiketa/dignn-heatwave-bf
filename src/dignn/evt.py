"""Extreme value theory: GPD fitting with a Peaks-Over-Threshold approach."""

import numpy as np
from scipy.stats import genpareto

from .config import TRAIN_YEARS


def fit_gpd_parameters(df):
    """Fit GPD parameters for each station (DI-GNN)."""
    print("\n7. Fitting GPD parameters...")
    
    gpd_params = {}
    train_df = df[df['YEAR'].isin(TRAIN_YEARS)]
    
    for station in df['NAME'].unique():
        station_data = train_df[train_df['NAME'] == station]['MAX'].dropna()
        
        if len(station_data) < 100:
            print(f"   WARNING: {station} has only {len(station_data)} samples")
            continue
        
        # Threshold: 90th percentile
        threshold = np.percentile(station_data, 90)
        exceedances = station_data[station_data > threshold] - threshold
        
        if len(exceedances) < 10:
            print(f"   WARNING: {station} has only {len(exceedances)} exceedances")
            # Use default values
            gpd_params[station] = {
                'xi': 0.1,
                'sigma': 2.0,
                'mu': station_data.mean(),
                'var': station_data.var(),
                'q95': np.percentile(station_data, 95)
            }
            continue
        
        try:
            # Fit GPD
            shape, loc, scale = genpareto.fit(exceedances, floc=0)
            
            gpd_params[station] = {
                'xi': shape,           # Shape parameter
                'sigma': scale,        # Scale parameter
                'mu': station_data.mean(),
                'var': station_data.var(),
                'q95': np.percentile(station_data, 95)
            }
            
            print(f"   {station}: ξ={shape:.4f}, σ={scale:.2f}, μ={station_data.mean():.1f}")
            
        except Exception as e:
            print(f"   WARNING: GPD fit failed for {station}: {e}")
            gpd_params[station] = {
                'xi': 0.1,
                'sigma': 2.0,
                'mu': station_data.mean(),
                'var': station_data.var(),
                'q95': np.percentile(station_data, 95)
            }
    
    return gpd_params


def add_gpd_features(df, gpd_params):
    """Add GPD parameters as features (DI-GNN)."""
    print("\n8. Adding GPD features...")
    
    df = df.copy()
    
    for feature in ['xi', 'sigma', 'mu', 'var', 'q95']:
        df[f'GPD_{feature}'] = df['NAME'].map(
            lambda x: gpd_params.get(x, {}).get(feature, 0)
        )
    
    print(f"   Added GPD_xi, GPD_sigma, GPD_mu, GPD_var, GPD_q95")
    
    return df
