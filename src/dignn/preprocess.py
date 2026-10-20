"""
Preprocessing pipeline for DI-GNN heatwave forecasting.

Usage
-----
    python -m dignn.preprocess                      # reference configuration CIN10_COUT3
    python -m dignn.preprocess --c_in 10 --c_out 7  # ablation
    python -m dignn.preprocess --sequences_only --c_in 14 --c_out 3

Set DIGNN_DATA_DIR to run on another data folder, for example the synthetic sample:
    DIGNN_DATA_DIR=data/sample python -m dignn.preprocess
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (DATA_DIR, ANAM_FILE, BF_SHAPEFILE, C_IN, C_OUT,
                     C_IN_OPTIONS, C_OUT_OPTIONS, TRAIN_YEARS, VAL_YEARS,
                     HW_PERCENTILE, HW_MIN_DURATION)
from .io import load_gsod_data, load_oni_data, load_nao_data
from .quality import (compute_missing_data_table, cross_validate_with_anam,
                      filter_stations, impute_missing_data)
from .labels import compute_heatwave_labels, analyze_heatwave_characteristics
from .features import add_features, normalize_features, create_sequences
from .evt import fit_gpd_parameters, add_gpd_features
from .graph import compute_adjacency_matrices
from .plotting.data_figures import create_thesis_figures

warnings.filterwarnings('ignore')


def save_outputs(df, X, y, years, feature_cols, adj_pearson, adj_gpd, 
                 stations, gpd_params, norm_params,
                 c_in=C_IN, c_out=C_OUT, missing_data_results=None):
    """Save all preprocessed data."""
    print("\n12. Saving outputs...")
    
    # Save preprocessed CSV
    csv_path = DATA_DIR / "gsod_bf_preprocessed_v2.csv"
    df.to_csv(csv_path, index=False)
    print(f"   Saved: {csv_path}")
    
    # Save sequences — nommage par (c_in, c_out) pour l'étude d'ablation
    seq_path = DATA_DIR / f"sequences_cin{c_in}_cout{c_out}_v2.npz"
    np.savez(seq_path, X=X, y=y, years=years, features=feature_cols)
    print(f"   Saved: {seq_path}")
    
    # Save adjacency matrices
    adj_path = DATA_DIR / "adjacency_v2.npz"
    np.savez(adj_path, 
             pearson=adj_pearson, 
             gpd_weighted=adj_gpd, 
             stations=stations)
    print(f"   Saved: {adj_path}")
    
    # Save GPD parameters
    gpd_path = DATA_DIR / "gpd_params_v2.pkl"
    with open(gpd_path, 'wb') as f:
        pickle.dump(gpd_params, f)
    print(f"   Saved: {gpd_path}")
    
    # Save normalization parameters
    norm_path = DATA_DIR / "norm_params_v2.pkl"
    with open(norm_path, 'wb') as f:
        pickle.dump(norm_params, f)
    print(f"   Saved: {norm_path}")
    
    # Save missing data results if available
    if missing_data_results is not None:
        missing_path = DATA_DIR / "missing_data_analysis.pkl"
        with open(missing_path, 'wb') as f:
            pickle.dump(missing_data_results, f)
        print(f"   Saved: {missing_path}")
    
    # Save config
    config = {
        'c_in': c_in,
        'c_out': c_out,
        'n_features': len(feature_cols),
        'n_stations': len(stations),
        'features': feature_cols,
        'stations': list(stations),
        'hw_percentile': HW_PERCENTILE,
        'hw_min_duration': HW_MIN_DURATION
    }
    config_path = DATA_DIR / "config_v2.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"   Saved: {config_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Preprocessing Pipeline V2 — Heatwave Prediction (BF DI-GNN)"
    )
    parser.add_argument('--c_in', type=int, default=C_IN,
        help=f"Fenêtre d'entrée en jours (défaut: {C_IN}). Ablation: {C_IN_OPTIONS}")
    parser.add_argument('--c_out', type=int, default=C_OUT, choices=C_OUT_OPTIONS,
        help=f"Horizon de prévision en jours (défaut: {C_OUT}). Ablation: {C_OUT_OPTIONS}")
    parser.add_argument('--sequences_only', action='store_true',
        help="Recrée seulement les séquences depuis le CSV prétraité existant "
             "(rapide pour l'ablation — saute GPD/figures).")
    args = parser.parse_args()

    print("=" * 60)
    print("PREPROCESSING PIPELINE V2")
    print(f"  C_in={args.c_in}, C_out={args.c_out}")
    print("=" * 60)

    # =========================================================================
    # MODE --sequences_only : charge le CSV prétraité et recrée seulement
    # les séquences avec les nouveaux (c_in, c_out). Pour l'étude d'ablation.
    # =========================================================================
    if args.sequences_only:
        print("\n[MODE --sequences_only]")
        csv_path = DATA_DIR / "gsod_bf_preprocessed_v2.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} introuvable. Lancer d'abord le preprocessing complet.")
        df = pd.read_csv(csv_path)
        adj_data = np.load(DATA_DIR / "adjacency_v2.npz", allow_pickle=True)
        adj_pearson = adj_data['pearson']; adj_gpd = adj_data['gpd_weighted']
        stations = list(adj_data['stations'])
        with open(DATA_DIR / "gpd_params_v2.pkl", 'rb') as f:
            gpd_params = pickle.load(f)
        with open(DATA_DIR / "norm_params_v2.pkl", 'rb') as f:
            norm_params = pickle.load(f)
        feature_cols = [
            'beta', 'MAX', 'MIN', 'DEWP', 'RH', 'WDSP', 'PRCP', 'SLP',
            'LONGITUDE', 'LATITUDE', 'DOY_sin', 'DOY_cos', 'ONI', 'NAO', 'Tthresh',
            'GPD_xi', 'GPD_sigma', 'GPD_mu', 'GPD_var', 'GPD_q95']
        feature_cols = [c for c in feature_cols if c in df.columns]
        X, y, years, feature_cols = create_sequences(
            df, stations, c_in=args.c_in, c_out=args.c_out)
        save_outputs(df, X, y, years, feature_cols, adj_pearson, adj_gpd,
                     stations, gpd_params, norm_params,
                     c_in=args.c_in, c_out=args.c_out, missing_data_results=None)
        print(f"\n[sequences_only] Terminé : "
              f"sequences_cin{args.c_in}_cout{args.c_out}_v2.npz")
        return

    # Load data
    df = load_gsod_data()
    oni_data = load_oni_data()
    nao_data = load_nao_data()
    
    # =========================================================================
    # Analyze missing data BEFORE imputation
    # =========================================================================
    print("\n" + "=" * 60)
    print("MISSING DATA ANALYSIS (before imputation)")
    print("=" * 60)
    missing_data_results = compute_missing_data_table(
        df, 
        output_dir=DATA_DIR / "quality_analysis"
    )
    
    # =========================================================================
    # Cross-validate with ANAM (optional - set ANAM_FILE in config)
    # =========================================================================
    crossval_results = None
    anam_file = ANAM_FILE
    if anam_file is not None and Path(anam_file).exists():
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION WITH ANAM")
        print("=" * 60)
        crossval_results = cross_validate_with_anam(
            df,
            anam_file=anam_file,
            output_dir=DATA_DIR / "quality_analysis"
        )
    
    # Filter stations
    df, stations = filter_stations(
        df, 
        max_gap_days=60, 
        max_missing_pct=10, 
        key_vars=['MAX', 'MIN']
    )
    
    # Impute missing data
    df = impute_missing_data(df)
    
    # Compute heatwave labels
    df = compute_heatwave_labels(df, percentile=HW_PERCENTILE, 
                                  min_duration=HW_MIN_DURATION)
    
    # =========================================================================
    # Analyze heatwave characteristics and create publication figures
    # =========================================================================
    print("\n" + "=" * 60)
    print("HEATWAVE CHARACTERISTICS ANALYSIS")
    print("=" * 60)
    print(f"   Using filtered data: {len(stations)} stations")
    print(f"   Stations: {', '.join(stations)}")
    hw_analysis = analyze_heatwave_characteristics(
        df,
        output_dir=DATA_DIR / "heatwave_analysis",
        percentile=HW_PERCENTILE,
        min_duration=HW_MIN_DURATION,
        shapefile_path=BF_SHAPEFILE
    )
    
    # Add features
    df = add_features(df, oni_data, nao_data)
    
    # Fit GPD parameters
    gpd_params = fit_gpd_parameters(df)
    
    # Add GPD features
    df = add_gpd_features(df, gpd_params)
    
    # Compute adjacency matrices
    adj_pearson, adj_gpd, stations = compute_adjacency_matrices(df, gpd_params)
    

    # =========================================================================
    # Figures de these (serie Tmax, heatmap, graphe, GPD, timeline)
    # =========================================================================
    create_thesis_figures(
        df            = df,
        events_df     = hw_analysis["events_df"],
        gpd_params    = gpd_params,
        adj_pearson   = adj_pearson,
        adj_gpd       = adj_gpd,
        stations_list = list(stations),
        output_dir    = DATA_DIR / "heatwave_analysis",
        percentile    = HW_PERCENTILE,
        train_years   = list(TRAIN_YEARS),
        val_years     = list(VAL_YEARS),
    )
    # Define feature columns for normalization
    feature_cols = [
        'MAX', 'MIN', 'DEWP', 'RH', 'WDSP', 'PRCP', 'SLP',
        'DOY_sin', 'DOY_cos', 'ONI', 'NAO', 'Tthresh',
        'GPD_xi', 'GPD_sigma', 'GPD_mu', 'GPD_var', 'GPD_q95'
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    # Normalize features
    df, norm_params = normalize_features(df, feature_cols)
    
    # Create sequences
    X, y, years, feature_cols = create_sequences(
        df, stations, c_in=args.c_in, c_out=args.c_out)
    
    # Save outputs
    save_outputs(df, X, y, years, feature_cols, adj_pearson, adj_gpd,
                 stations, gpd_params, norm_params,
                 c_in=args.c_in, c_out=args.c_out,
                 missing_data_results=missing_data_results)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print(f"  Séquences : sequences_cin{args.c_in}_cout{args.c_out}_v2.npz")
    print("=" * 60)
    
    # Summary
    print(f"\nSummary:")
    print(f"  Stations: {len(stations)}")
    print(f"  Sequences: {len(X):,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Heatwave %: {y.sum()/y.size*100:.2f}%")
    print(f"  Missing data: {missing_data_results['summary']['overall_pct']:.2f}% (before imputation)")
    print(f"  Total heatwaves: {hw_analysis['total_heatwaves']}")
    print(f"  Jours chauds (T_max > P90): {hw_analysis['total_hot_days']}")
    print(f"  Jours en vague de chaleur: {hw_analysis['hw_days']}")


if __name__ == "__main__":
    main()
