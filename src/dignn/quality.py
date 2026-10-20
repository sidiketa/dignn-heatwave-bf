"""Data quality: missing-data analysis, ANAM cross-validation, station filtering, imputation."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .plotting.data_figures import create_crossval_heatmap


def compute_missing_data_table(df, output_dir=None):
    """
    Analyze missing data patterns before imputation.
    
    Creates a Station × Parameter matrix showing missing observation counts.
    This should be called BEFORE imputation for accurate reporting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw GSOD data (before imputation)
    output_dir : Path, optional
        Directory to save output files
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'matrix': Station × Parameter missing count matrix
        - 'pct_matrix': Station × Parameter missing percentage matrix
        - 'summary': Overall summary statistics
    """
    print("\n   Analyzing missing data patterns...")
    
    # Define meteorological parameters to analyze (exclude GUST - mostly missing)
    parameters = ['MAX', 'MIN', 'TEMP', 'DEWP', 'RH', 'WDSP', 'PRCP', 
                  'SLP', 'STP', 'VISIB', 'MXSPD']
    
    # Filter to columns that exist in the dataframe
    parameters = [col for col in parameters if col in df.columns]
    
    stations = sorted(df['NAME'].unique())
    
    # ==========================================================================
    # Build Missing Observations Matrix (counts)
    # Rows: Stations | Columns: Parameters | Values: Missing observation counts
    # ==========================================================================
    missing_matrix = pd.DataFrame(
        index=stations, 
        columns=['N'] + parameters + ['Total'], 
        dtype=int
    )
    
    for station in stations:
        sdf = df[df['NAME'] == station]
        missing_matrix.loc[station, 'N'] = len(sdf)
        for param in parameters:
            missing_matrix.loc[station, param] = int(sdf[param].isna().sum())
        missing_matrix.loc[station, 'Total'] = missing_matrix.loc[station, parameters].sum()
    
    # Add Total row
    missing_matrix.loc['Total', 'N'] = len(df)
    for param in parameters:
        missing_matrix.loc['Total', param] = missing_matrix.loc[stations, param].sum()
    missing_matrix.loc['Total', 'Total'] = missing_matrix.loc[stations, 'Total'].sum()
    
    # ==========================================================================
    # Build Missing Observations Matrix (percentages)
    # ==========================================================================
    pct_matrix = pd.DataFrame(
        index=stations, 
        columns=['N'] + parameters + ['Avg'], 
        dtype=float
    )
    
    for station in stations:
        n = missing_matrix.loc[station, 'N']
        pct_matrix.loc[station, 'N'] = n
        for param in parameters:
            pct_matrix.loc[station, param] = (missing_matrix.loc[station, param] / n) * 100
        pct_matrix.loc[station, 'Avg'] = pct_matrix.loc[station, parameters].mean()
    
    # Add Total row
    total_n = len(df)
    pct_matrix.loc['Total', 'N'] = total_n
    for param in parameters:
        pct_matrix.loc['Total', param] = (missing_matrix.loc['Total', param] / total_n) * 100
    pct_matrix.loc['Total', 'Avg'] = pct_matrix.loc['Total', parameters].mean()
    
    # Round percentages
    pct_matrix = pct_matrix.round(2)
    
    # ==========================================================================
    # Compute summary statistics
    # ==========================================================================
    total_cells = len(df) * len(parameters)
    total_missing = int(missing_matrix.loc['Total', 'Total'])
    overall_pct = (total_missing / total_cells) * 100
    
    # ==========================================================================
    # Print results
    # ==========================================================================
    print(f"\n   MISSING OBSERVATIONS MATRIX (Station × Parameter)")
    print(f"   Rows: Stations | Columns: Parameters | Values: Missing counts")
    print(f"   N = Total records per station")
    print()
    
    # Format and print the matrix
    print("   " + "-" * 100)
    header = f"   {'Station':<18} {'N':>6}"
    for param in parameters:
        header += f" {param:>6}"
    header += f" {'Total':>7}"
    print(header)
    print("   " + "-" * 100)
    
    for station in stations:
        row = f"   {station:<18} {int(missing_matrix.loc[station, 'N']):>6}"
        for param in parameters:
            val = int(missing_matrix.loc[station, param])
            row += f" {val:>6}"
        row += f" {int(missing_matrix.loc[station, 'Total']):>7}"
        print(row)
    
    print("   " + "-" * 100)
    row = f"   {'Total':<18} {int(missing_matrix.loc['Total', 'N']):>6}"
    for param in parameters:
        val = int(missing_matrix.loc['Total', param])
        row += f" {val:>6}"
    row += f" {int(missing_matrix.loc['Total', 'Total']):>7}"
    print(row)
    print("   " + "-" * 100)
    
    # Print summary
    print(f"\n   Summary:")
    print(f"   - Total observations: {len(df):,}")
    print(f"   - Total cells (obs × params): {total_cells:,}")
    print(f"   - Total missing: {total_missing:,} ({overall_pct:.2f}%)")
    
    # Flag problematic parameters
    problem_params = [p for p in parameters if pct_matrix.loc['Total', p] > 5]
    if problem_params:
        print(f"\n   ⚠️ Parameters with >5% missing:")
        for param in problem_params:
            print(f"      - {param}: {pct_matrix.loc['Total', param]:.2f}%")
    
    # Flag problematic stations
    problem_stations = [s for s in stations if pct_matrix.loc[s, 'Avg'] > 5]
    if problem_stations:
        print(f"\n   ⚠️ Stations with >5% missing (avg across params):")
        for station in problem_stations:
            print(f"      - {station}: {pct_matrix.loc[station, 'Avg']:.2f}%")
    
    # ==========================================================================
    # Save outputs if directory provided
    # ==========================================================================
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV files
        missing_matrix.to_csv(output_dir / "missing_observations_matrix.csv")
        pct_matrix.to_csv(output_dir / "missing_observations_pct.csv")
        
        print(f"\n   Saved: {output_dir / 'missing_observations_matrix.csv'}")
        print(f"   Saved: {output_dir / 'missing_observations_pct.csv'}")
    
    return {
        'matrix': missing_matrix,
        'pct_matrix': pct_matrix,
        'parameters': parameters,
        'stations': stations,
        'summary': {
            'total_obs': len(df),
            'total_cells': total_cells,
            'total_missing': total_missing,
            'overall_pct': overall_pct
        }
    }


def cross_validate_with_anam(df_gsod, anam_file=None, output_dir=None):
    """
    Cross-validate GSOD data against ANAM (Burkina Faso Met Agency) data.
    
    Parameters
    ----------
    df_gsod : pd.DataFrame
        GSOD data
    anam_file : str or Path, optional
        Path to ANAM CSV file
    output_dir : Path, optional
        Directory to save output files
        
    Returns
    -------
    dict
        Cross-validation results by station and variable
    """
    print("\n   Cross-validating with ANAM data...")
    
    if anam_file is None:
        print("   WARNING: No ANAM file provided, skipping cross-validation")
        return None
    
    anam_file = Path(anam_file)
    if not anam_file.exists():
        print(f"   WARNING: ANAM file not found: {anam_file}")
        return None
    
    # Load ANAM data
    df_anam = pd.read_csv(anam_file)
    print(f"   Loaded ANAM data: {len(df_anam):,} records")
    
    # Prepare dates
    df_gsod = df_gsod.copy()
    df_gsod['DATE'] = pd.to_datetime(df_gsod['YEARMODA'])
    df_anam['DATE'] = pd.to_datetime(df_anam[['YEAR', 'MONTH', 'DAY']])
    
    # Merge datasets
    df_merged = pd.merge(
        df_gsod[['NAME', 'DATE', 'MAX', 'MIN', 'SLP', 'PRCP', 'WDSP', 'RH']],
        df_anam[['NAME', 'DATE', 'MAX', 'MIN', 'SLP', 'PRCP', 'WDSP', 'RHMAX', 'RHMIN']],
        on=['NAME', 'DATE'],
        suffixes=('_gsod', '_anam')
    )
    
    # Calculate ANAM RH mean for fair comparison
    df_merged['RH_anam_mean'] = (df_merged['RHMAX'] + df_merged['RHMIN']) / 2
    
    print(f"   Merged records: {len(df_merged):,}")
    
    # Variables to compare
    variables = [
        ('MAX', 'MAX_gsod', 'MAX_anam', '°C'),
        ('MIN', 'MIN_gsod', 'MIN_anam', '°C'),
        ('SLP', 'SLP_gsod', 'SLP_anam', 'hPa'),
        ('PRCP', 'PRCP_gsod', 'PRCP_anam', 'mm'),
        ('WDSP', 'WDSP_gsod', 'WDSP_anam', 'm/s'),
        ('RH', 'RH', 'RH_anam_mean', '%')
    ]
    
    results = []
    stations = sorted(df_merged['NAME'].unique())
    
    # Overall statistics
    print(f"\n   Cross-Validation Results (Overall):")
    print(f"   {'Variable':<8} {'N pairs':>10} {'Corr':>10} {'Bias':>12} {'RMSE':>10}")
    print(f"   {'-'*55}")
    
    for var_name, gsod_col, anam_col, unit in variables:
        mask = df_merged[gsod_col].notna() & df_merged[anam_col].notna()
        gsod_vals = df_merged.loc[mask, gsod_col].values
        anam_vals = df_merged.loc[mask, anam_col].values
        
        n_pairs = len(gsod_vals)
        
        if n_pairs > 50:
            corr, _ = stats.pearsonr(gsod_vals, anam_vals)
            diff = gsod_vals - anam_vals
            bias = np.mean(diff)
            rmse = np.sqrt(np.mean(diff**2))
            mae = np.mean(np.abs(diff))
            
            results.append({
                'Variable': var_name,
                'Station': 'ALL',
                'N': n_pairs,
                'Correlation': corr,
                'Bias': bias,
                'RMSE': rmse,
                'MAE': mae,
                'Unit': unit
            })
            
            flag = " ⚠️" if corr < 0.90 else ""
            print(f"   {var_name:<8} {n_pairs:>10,} {corr:>10.4f} {bias:>+12.2f} {rmse:>10.2f}{flag}")
    
    # By station statistics
    for station in stations:
        sdf = df_merged[df_merged['NAME'] == station]
        
        for var_name, gsod_col, anam_col, unit in variables:
            mask = sdf[gsod_col].notna() & sdf[anam_col].notna()
            gsod_vals = sdf.loc[mask, gsod_col].values
            anam_vals = sdf.loc[mask, anam_col].values
            
            n_pairs = len(gsod_vals)
            
            if n_pairs > 50:
                corr, _ = stats.pearsonr(gsod_vals, anam_vals)
                diff = gsod_vals - anam_vals
                bias = np.mean(diff)
                rmse = np.sqrt(np.mean(diff**2))
                mae = np.mean(np.abs(diff))
                
                results.append({
                    'Variable': var_name,
                    'Station': station,
                    'N': n_pairs,
                    'Correlation': corr,
                    'Bias': bias,
                    'RMSE': rmse,
                    'MAE': mae,
                    'Unit': unit
                })
    
    results_df = pd.DataFrame(results)
    
    # Save outputs and create visualization if directory provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        results_df.to_csv(output_dir / "cross_validation_by_station.csv", index=False)
        print(f"\n   Saved: {output_dir / 'cross_validation_by_station.csv'}")
        
        # Create heatmap visualization
        create_crossval_heatmap(results_df, variables, output_dir)
    
    return {
        'results': results_df,
        'merged_data': df_merged
    }


def filter_stations(df, max_gap_days=60, max_missing_pct=10, key_vars=['MAX', 'MIN']):
    """
    Filter stations based on data quality criteria.
    
    This function checks both:
    1. Temporal gaps (missing dates)
    2. Missing observations on existing dates
    3. Consecutive vs random missing patterns
    
    Parameters
    ----------
    df : pd.DataFrame
        GSOD data with YEARMODA and NAME columns
    max_gap_days : int
        Maximum allowed consecutive days with missing data (default: 60)
    max_missing_pct : float
        Maximum allowed percentage of missing data for key variables (default: 10%)
    key_vars : list
        Key variables to check for missing data (default: ['MAX', 'MIN'])
        
    Returns
    -------
    df_filtered : pd.DataFrame
        Filtered dataframe with only valid stations
    valid_stations : list
        List of valid station names
    """
    print("\n3. Filtering stations (enhanced quality check)...")
    
    valid_stations = []
    quality_report = {}
    
    # Get the full date range
    df = df.copy()
    df['DATE'] = pd.to_datetime(df['YEARMODA'])
    date_min = df['DATE'].min()
    date_max = df['DATE'].max()
    full_date_range = pd.date_range(start=date_min, end=date_max, freq='D')
    n_expected_days = len(full_date_range)
    
    print(f"   Expected date range: {date_min.date()} to {date_max.date()} ({n_expected_days:,} days)")
    print(f"   Key variables checked: {key_vars}")
    print(f"   Thresholds: max_gap={max_gap_days} days, max_missing={max_missing_pct}%")
    print()
    
    for station in sorted(df['NAME'].unique()):
        station_df = df[df['NAME'] == station].copy()
        station_df = station_df.sort_values('DATE')
        
        # === 1. Check for missing dates ===
        station_dates = set(station_df['DATE'])
        missing_dates = set(full_date_range) - station_dates
        n_missing_dates = len(missing_dates)
        
        # === 2. Check for missing observations on existing dates ===
        # Create a combined "data available" flag for key variables
        available_key_vars = [v for v in key_vars if v in station_df.columns]
        if available_key_vars:
            station_df['has_data'] = station_df[available_key_vars].notna().all(axis=1)
        else:
            station_df['has_data'] = True
        
        # === 3. Create a complete timeline with data availability ===
        # Reindex to full date range
        station_full = station_df.set_index('DATE').reindex(full_date_range)
        station_full['has_data'] = station_full['has_data'].fillna(False).infer_objects(copy=False)
        
        # === 4. Find consecutive missing periods ===
        # A day is "missing" if either the date is absent OR key vars are NaN
        station_full['is_missing'] = ~station_full['has_data']
        
        # Find runs of consecutive missing days
        station_full['run_id'] = (station_full['is_missing'] != station_full['is_missing'].shift()).cumsum()
        missing_runs = station_full[station_full['is_missing']].groupby('run_id').size()
        
        max_consecutive_missing = int(missing_runs.max()) if len(missing_runs) > 0 else 0
        n_missing_runs = len(missing_runs)
        
        # === 5. Calculate total missing ===
        total_missing_days = int(station_full['is_missing'].sum())
        missing_pct = (total_missing_days / n_expected_days) * 100
        
        # === 6. Classify missing pattern ===
        if n_missing_runs > 0:
            avg_run_length = missing_runs.mean()
            pattern = "clustered" if avg_run_length > 5 else "scattered"
        else:
            pattern = "none"
        
        # === 7. Quality decision ===
        is_valid = (max_consecutive_missing <= max_gap_days) and (missing_pct <= max_missing_pct)
        
        # Store report
        quality_report[station] = {
            'n_records': len(station_df),
            'n_expected': n_expected_days,
            'missing_dates': n_missing_dates,
            'missing_obs': int(total_missing_days - n_missing_dates),
            'total_missing': total_missing_days,
            'missing_pct': missing_pct,
            'max_consecutive': max_consecutive_missing,
            'n_gaps': n_missing_runs,
            'pattern': pattern,
            'is_valid': is_valid
        }
        
        # Print status
        status = "✓" if is_valid else "✗"
        reason = ""
        if not is_valid:
            if max_consecutive_missing > max_gap_days:
                reason = f"max gap {max_consecutive_missing} > {max_gap_days}"
            if missing_pct > max_missing_pct:
                reason += f" missing {missing_pct:.1f}% > {max_missing_pct}%"
        
        print(f"   {status} {station}:")
        print(f"      Records: {len(station_df):,}/{n_expected_days:,} | "
              f"Missing: {total_missing_days:,} ({missing_pct:.2f}%) | "
              f"Max gap: {max_consecutive_missing} days | "
              f"Pattern: {pattern}")
        if not is_valid:
            print(f"      → Excluded: {reason}")
        
        if is_valid:
            valid_stations.append(station)
    
    # Filter dataframe
    df_filtered = df[df['NAME'].isin(valid_stations)].copy()
    
    # Remove temporary columns
    if 'DATE' in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=['DATE'])
    
    print(f"\n   Summary: Kept {len(valid_stations)}/{len(df['NAME'].unique())} stations, "
          f"{len(df_filtered):,} records")
    
    return df_filtered, valid_stations


def impute_missing_data(df):
    """Impute missing values using meteorological methods."""
    print("\n4. Imputing missing data...")
    
    df = df.copy()
    
    # Temperature: interpolate then seasonal mean
    for col in ['MAX', 'MIN', 'TEMP', 'DEWP']:
        if col in df.columns:
            missing_before = df[col].isna().sum()
            
            # Linear interpolation (max 3 days)
            df[col] = df.groupby('NAME')[col].transform(
                lambda x: x.interpolate(method='linear', limit=3)
            )
            
            # Fill remaining with station mean
            df[col] = df.groupby('NAME')[col].transform(
                lambda x: x.fillna(x.mean())
            )
            
            missing_after = df[col].isna().sum()
            if missing_before > 0:
                print(f"   {col}: {missing_before} -> {missing_after} missing")
    
    # Pressure: interpolate then station mean
    for col in ['SLP', 'STP']:
        if col in df.columns:
            missing_before = df[col].isna().sum()
            df[col] = df.groupby('NAME')[col].transform(
                lambda x: x.interpolate(method='linear', limit=5)
            )
            df[col] = df.groupby('NAME')[col].transform(
                lambda x: x.fillna(x.mean())
            )
            missing_after = df[col].isna().sum()
            if missing_before > 0:
                print(f"   {col}: {missing_before} -> {missing_after} missing")
    
    # Precipitation: fill with 0 (conservative)
    if 'PRCP' in df.columns:
        missing_before = df['PRCP'].isna().sum()
        df['PRCP'] = df['PRCP'].fillna(0)
        if missing_before > 0:
            print(f"   PRCP: {missing_before} -> 0 missing (filled with 0)")
    
    # Wind: interpolate then station mean
    for col in ['WDSP', 'MXSPD', 'GUST']:
        if col in df.columns:
            df[col] = df.groupby('NAME')[col].transform(
                lambda x: x.interpolate(method='linear', limit=3)
            )
            df[col] = df.groupby('NAME')[col].transform(
                lambda x: x.fillna(x.mean())
            )
    
    # RH: compute from temperature if possible, else interpolate
    if 'RH' in df.columns:
        missing_before = df['RH'].isna().sum()
        
        # Try to compute from DEWP and TEMP
        if 'DEWP' in df.columns and 'TEMP' in df.columns:
            mask = df['RH'].isna() & df['DEWP'].notna() & df['TEMP'].notna()
            if mask.any():
                T = df.loc[mask, 'TEMP']
                Td = df.loc[mask, 'DEWP']
                # Magnus formula approximation
                rh = 100 * np.exp((17.625 * Td) / (243.04 + Td)) / \
                         np.exp((17.625 * T) / (243.04 + T))
                df.loc[mask, 'RH'] = rh.clip(0, 100)
        
        # Interpolate remaining
        df['RH'] = df.groupby('NAME')['RH'].transform(
            lambda x: x.interpolate(method='linear', limit=3)
        )
        df['RH'] = df.groupby('NAME')['RH'].transform(
            lambda x: x.fillna(x.mean())
        )
        
        missing_after = df['RH'].isna().sum()
        if missing_before > 0:
            print(f"   RH: {missing_before} -> {missing_after} missing")
    
    # Final check - fill any remaining with column mean
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
    
    return df
