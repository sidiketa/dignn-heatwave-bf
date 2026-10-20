"""Heatwave labelling (Perkins-Kirkpatrick-Lewis definition) and event statistics."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import TRAIN_YEARS, VAL_YEARS
from .plotting.data_figures import create_heatwave_figures, create_station_network_map


def compute_heatwave_labels(df, percentile=90, min_duration=3):
    """Compute heatwave labels using PKL criteria."""
    print("\n5. Computing heatwave labels...")
    
    df = df.copy()
    df['beta'] = 0
    
    # Compute threshold per station (from training period)
    train_df = df[df['YEAR'].isin(TRAIN_YEARS)]
    
    thresholds = {}
    for station in df['NAME'].unique():
        station_train = train_df[train_df['NAME'] == station]['MAX']
        thresholds[station] = np.nanpercentile(station_train, percentile)
    
    df['Tthresh'] = df['NAME'].map(thresholds)
    df['above_thresh'] = (df['MAX'] > df['Tthresh']).astype(int)
    
    # Find consecutive days above threshold
    total_hw_days = 0
    
    for station in df['NAME'].unique():
        mask = df['NAME'] == station
        station_idx = df[mask].index
        above = df.loc[station_idx, 'above_thresh'].values
        
        # Find runs of consecutive days
        runs = []
        current_run = []
        
        for i, val in enumerate(above):
            if val == 1:
                current_run.append(i)
            else:
                if len(current_run) >= min_duration:
                    runs.extend(current_run)
                current_run = []
        
        # Don't forget last run
        if len(current_run) >= min_duration:
            runs.extend(current_run)
        
        # Mark heatwave days
        if runs:
            hw_indices = [station_idx[i] for i in runs]
            df.loc[hw_indices, 'beta'] = 1
            total_hw_days += len(runs)
    
    hw_pct = df['beta'].sum() / len(df) * 100
    print(f"   Total heatwave days: {df['beta'].sum():,} ({hw_pct:.2f}%)")

    # -------------------------------------------------------------------------
    # Statistiques de déséquilibre de classes
    # -------------------------------------------------------------------------
    print("\n   --- Statistiques de déséquilibre de classes ---")

    n_total    = len(df)
    n_hw       = int(df['beta'].sum())
    n_non_hw   = n_total - n_hw
    ratio      = n_non_hw / n_hw if n_hw > 0 else float('inf')

    print(f"   Observations totales      : {n_total:,}")
    print(f"   Jours de vague de chaleur : {n_hw:,}  ({n_hw/n_total*100:.2f}%)")
    print(f"   Jours ordinaires          : {n_non_hw:,}  ({n_non_hw/n_total*100:.2f}%)")
    print(f"   Ratio déséquilibre        : 1 : {ratio:.1f}")
    print(f"   Pos_weight recommandé     : {ratio:.2f}  (pour BCE pondérée)")

    # Par station
    print("\n   Déséquilibre par station :")
    print(f"   {'Station':<20} {'N total':>9} {'HW jours':>10} {'HW %':>8} {'Ratio':>8}")
    print(f"   {'-'*57}")
    for station in sorted(df['NAME'].unique()):
        s      = df[df['NAME'] == station]
        n_s    = len(s)
        n_hw_s = int(s['beta'].sum())
        pct_s  = n_hw_s / n_s * 100 if n_s > 0 else 0
        rat_s  = (n_s - n_hw_s) / n_hw_s if n_hw_s > 0 else float('inf')
        print(f"   {station:<20} {n_s:>9,} {n_hw_s:>10,} {pct_s:>7.2f}% {rat_s:>7.1f}:1")

    # Par année (train vs validation)
    print("\n   Déséquilibre par période :")
    for period, years, label in [
        ('Entraînement', TRAIN_YEARS, 'train'),
        ('Validation',   VAL_YEARS,   'val')
    ]:
        sub    = df[df['YEAR'].isin(years)]
        n_p    = len(sub)
        n_hw_p = int(sub['beta'].sum())
        pct_p  = n_hw_p / n_p * 100 if n_p > 0 else 0
        print(f"   {period:<15} : {n_hw_p:,} HW / {n_p:,} jours ({pct_p:.2f}%)")

    return df


def identify_heatwave_events(df, percentile=90, min_duration=3):
    """
    Identify heatwave events and compute their characteristics.
    
    Parameters
    ----------
    df : pd.DataFrame
        GSOD data with DATE, NAME, MAX columns
    percentile : int
        Percentile threshold for defining hot days (default: 90)
    min_duration : int
        Minimum consecutive days to qualify as heatwave (default: 3)
        
    Returns
    -------
    pd.DataFrame
        DataFrame of individual events with characteristics:
        - Station, Start_Date, Year, Month
        - Duration (days)
        - Is_Heatwave (True if Duration >= min_duration)
        - Threshold, Peak_Temp, Mean_Temp
        - Cumulative_Excess, Mean_Excess
    """
    events = []
    
    # Ensure DATE column exists
    df = df.copy()
    if 'DATE' not in df.columns:
        df['DATE'] = pd.to_datetime(df['YEARMODA'])
    
    for station in df['NAME'].unique():
        sdf = df[df['NAME'] == station].sort_values('DATE').copy()
        
        # Compute threshold (percentile of MAX temperature)
        threshold = np.percentile(sdf['MAX'].dropna(), percentile)
        
        # Identify days above threshold
        sdf['above_thresh'] = sdf['MAX'] > threshold
        sdf['excess'] = np.maximum(0, sdf['MAX'] - threshold)
        
        # Find consecutive runs
        sdf['run_id'] = (sdf['above_thresh'] != sdf['above_thresh'].shift()).cumsum()
        
        # Group by runs and extract events
        for run_id, group in sdf[sdf['above_thresh']].groupby('run_id'):
            duration = len(group)
            
            event = {
                'Station': station,
                'Start_Date': group['DATE'].min(),
                'End_Date': group['DATE'].max(),
                'Year': group['DATE'].min().year,
                'Month': group['DATE'].min().month,
                'Duration': duration,
                'Is_Heatwave': duration >= min_duration,
                'Threshold': threshold,
                'Peak_Temp': group['MAX'].max(),
                'Mean_Temp': group['MAX'].mean(),
                'Cumulative_Excess': group['excess'].sum(),
                'Mean_Excess': group['excess'].mean(),
            }
            events.append(event)
    
    return pd.DataFrame(events)


def identify_heatwave_events_mam(df, percentile=90, min_duration=3, warm_months=[3, 4, 5]):
    """
    Identify heatwave events using Sahel-specific definition.
    
    Threshold is computed as the percentile of Tmax during the warm season
    (March-April-May by default), giving a single fixed threshold per station.
    This definition is more appropriate for Sahel regions.
    
    Parameters
    ----------
    df : pd.DataFrame
        GSOD data with DATE, NAME, MAX columns
    percentile : int
        Percentile threshold (default: 90)
    min_duration : int
        Minimum consecutive days to qualify as heatwave (default: 3)
    warm_months : list
        Months to use for threshold computation (default: [3, 4, 5] for MAM)
        
    Returns
    -------
    tuple
        (events_df, thresholds_dict)
        - events_df: DataFrame of individual events with characteristics
        - thresholds_dict: Dictionary of station -> threshold value
    """
    events = []
    thresholds = {}
    
    df = df.copy()
    if 'DATE' not in df.columns:
        df['DATE'] = pd.to_datetime(df['YEARMODA'])
    if 'MONTH' not in df.columns:
        df['MONTH'] = df['DATE'].dt.month
    
    for station in df['NAME'].unique():
        sdf = df[df['NAME'] == station].sort_values('DATE').copy()
        
        # Compute threshold from warm season data only
        mam_data = sdf[sdf['MONTH'].isin(warm_months)]['MAX']
        threshold = np.percentile(mam_data.dropna(), percentile)
        thresholds[station] = threshold
        
        # Apply threshold to ALL days (year-round)
        sdf['above_thresh'] = sdf['MAX'] > threshold
        sdf['excess'] = np.maximum(0, sdf['MAX'] - threshold)
        sdf['run_id'] = (sdf['above_thresh'] != sdf['above_thresh'].shift()).cumsum()
        
        for run_id, group in sdf[sdf['above_thresh']].groupby('run_id'):
            duration = len(group)
            event = {
                'Station': station,
                'Start_Date': group['DATE'].min(),
                'End_Date': group['DATE'].max(),
                'Year': group['DATE'].min().year,
                'Month': group['DATE'].min().month,
                'Duration': duration,
                'Is_Heatwave': duration >= min_duration,
                'Threshold': threshold,
                'Peak_Temp': group['MAX'].max(),
                'Mean_Temp': group['MAX'].mean(),
                'Cumulative_Excess': group['excess'].sum(),
                'Mean_Excess': group['excess'].mean(),
            }
            events.append(event)
    
    return pd.DataFrame(events), thresholds


def compute_heatwave_statistics(events_df, output_dir=None, suffix=''):
    """
    Compute comprehensive heatwave statistics.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Output from identify_heatwave_events()
    output_dir : Path, optional
        Directory to save statistics CSV files
        
    Returns
    -------
    dict
        Dictionary containing summary, station, and annual statistics
    """
    hw_events = events_df[events_df['Is_Heatwave']]
    hot_days = events_df[~events_df['Is_Heatwave']]
    
    # Overall summary
    summary = pd.DataFrame({
        'Metric': ['Duration (days)', 'Peak Temperature (°C)', 
                   'Mean Excess (°C)', 'Cumulative Excess (°C·days)'],
        'Min': [hw_events['Duration'].min(), hw_events['Peak_Temp'].min(),
                hw_events['Mean_Excess'].min(), hw_events['Cumulative_Excess'].min()],
        'Max': [hw_events['Duration'].max(), hw_events['Peak_Temp'].max(),
                hw_events['Mean_Excess'].max(), hw_events['Cumulative_Excess'].max()],
        'Mean': [hw_events['Duration'].mean(), hw_events['Peak_Temp'].mean(),
                 hw_events['Mean_Excess'].mean(), hw_events['Cumulative_Excess'].mean()],
        'Median': [hw_events['Duration'].median(), hw_events['Peak_Temp'].median(),
                   hw_events['Mean_Excess'].median(), hw_events['Cumulative_Excess'].median()],
        'Std': [hw_events['Duration'].std(), hw_events['Peak_Temp'].std(),
                hw_events['Mean_Excess'].std(), hw_events['Cumulative_Excess'].std()]
    }).round(2)
    
    # Station-level statistics
    station_stats = hw_events.groupby('Station').agg({
        'Duration': ['count', 'mean', 'std', 'max'],
        'Peak_Temp': ['mean', 'max'],
        'Mean_Excess': 'mean',
        'Cumulative_Excess': 'mean'
    }).round(2)
    station_stats.columns = ['Count', 'Dur_Mean', 'Dur_Std', 'Dur_Max',
                              'Peak_Mean', 'Peak_Max', 'Excess_Mean', 'CumExcess_Mean']
    station_stats = station_stats.reset_index()
    
    # Annual statistics
    annual_stats = hw_events.groupby('Year').agg({
        'Duration': ['count', 'mean', 'sum'],
        'Peak_Temp': ['mean', 'max'],
        'Cumulative_Excess': 'sum'
    }).round(2)
    annual_stats.columns = ['HW_Count', 'Dur_Mean', 'Dur_Total',
                            'Peak_Mean', 'Peak_Max', 'CumExcess_Total']
    annual_stats = annual_stats.reset_index()
    
    # Save if output directory provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use suffix for filenames (e.g., '_mam' for MAM definition)
        events_df.to_csv(output_dir / f"heatwave_events{suffix}.csv", index=False)
        summary.to_csv(output_dir / f"heatwave_statistics_summary{suffix}.csv", index=False)
        station_stats.to_csv(output_dir / f"heatwave_statistics_by_station{suffix}.csv", index=False)
        annual_stats.to_csv(output_dir / f"heatwave_statistics_by_year{suffix}.csv", index=False)
        
        print(f"   Saved: heatwave_events{suffix}.csv")
        print(f"   Saved: heatwave_statistics_summary{suffix}.csv")
    
    return {
        'summary': summary,
        'station_stats': station_stats,
        'annual_stats': annual_stats,
        'total_heatwaves': len(hw_events),
        'total_hot_days': len(hot_days)
    }


def analyze_heatwave_characteristics(df, output_dir=None, percentile=90, min_duration=3, 
                                      shapefile_path=None):
    """
    Complete heatwave characteristics analysis with statistics and figures.
    
    Uses PKL (Perkins-Kirkpatrick-Lewis) definition:
    - Threshold: 90th percentile of all Tmax data per station
    - Heatwave: ≥3 consecutive days above threshold
    
    Note: MAM (Sahel-specific) definition was evaluated but excluded due to
    insufficient sample size (92 events vs 553 for PKL).
    
    Parameters
    ----------
    df : pd.DataFrame
        GSOD data
    output_dir : Path, optional
        Directory to save outputs
    percentile : int
        Percentile threshold (default: 90)
    min_duration : int
        Minimum duration for heatwave (default: 3)
    shapefile_path : str or Path, optional
        Path to Burkina Faso admin level 0 shapefile for network map
        
    Returns
    -------
    dict
        Dictionary with events_df, statistics, and summary info
    """
    print("\n   Analyzing heatwave characteristics...")
    print(f"   Definition: PKL (Perkins-Kirkpatrick-Lewis)")
    print(f"   Threshold: Tmax > {percentile}th percentile for ≥{min_duration} consecutive days")
    
    # Identify events
    events_df = identify_heatwave_events(df, percentile=percentile, min_duration=min_duration)
    
    hw_events    = events_df[events_df['Is_Heatwave']]
    short_events = events_df[~events_df['Is_Heatwave']]

    # Jours chauds = chaque jour T_max > P90 compte pour 1.
    # On somme les durées des épisodes (Duration = nombre de jours de l'épisode).
    total_hot_days     = int(events_df['Duration'].sum())          # tous les jours > P90
    hw_days            = int(hw_events['Duration'].sum())          # jours en vague de chaleur
    short_days         = int(short_events['Duration'].sum())       # jours hors vague (épisodes < 3 j)

    print(f"\n   Résultats :")
    print(f"   - Jours chauds (T_max > P{percentile}) : {total_hot_days}")
    print(f"       dont jours en vague de chaleur (épisode ≥ {min_duration} j) : {hw_days}")
    print(f"       dont jours hors vague (épisode < {min_duration} j)          : {short_days}")
    print(f"   - Épisodes au total : {len(events_df)}")
    print(f"       dont épisodes courts (< {min_duration} j) : {len(short_events)}")
    print(f"       dont vagues de chaleur (≥ {min_duration} j) : {len(hw_events)}")
    print(f"   - Durée moyenne d'une vague : {hw_events['Duration'].mean():.1f} jours")
    print(f"   - Durée maximale : {hw_events['Duration'].max()} jours")
    print(f"   - Plage des pics de température : {hw_events['Peak_Temp'].min():.1f} - {hw_events['Peak_Temp'].max():.1f}°C")
    
    # Compute statistics and create figures
    stats = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stats = compute_heatwave_statistics(events_df, output_dir)
        create_heatwave_figures(df, events_df, output_dir, percentile=percentile)
        create_station_network_map(df, events_df, output_dir, shapefile_path=shapefile_path)
    
    return {
        'events_df': events_df,
        'statistics': stats,
        'total_heatwaves': len(hw_events),   # nombre de vagues (épisodes ≥ 3 j)
        'total_hot_days': total_hot_days,    # nombre de jours chauds (T_max > P90)
        'hw_days': hw_days,                  # jours appartenant à une vague de chaleur
        'short_events': len(short_events),   # épisodes courts (< 3 j)
    }
