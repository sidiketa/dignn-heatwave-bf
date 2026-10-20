"""Station graph: Pearson adjacency and GPD-weighted adjacency."""

import numpy as np


def compute_adjacency_matrices(df, gpd_params):
    """Compute adjacency matrices (Pearson + GPD-weighted)."""
    print("\n9. Computing adjacency matrices...")
    
    stations = sorted(df['NAME'].unique())
    n_stations = len(stations)
    
    # Pivot to get time series per station
    pivot_tmax = df.pivot_table(
        index='YEARMODA', columns='NAME', values='MAX', aggfunc='first'
    )[stations]
    
    pivot_prcp = df.pivot_table(
        index='YEARMODA', columns='NAME', values='PRCP', aggfunc='first'
    )[stations]
    
    # Pearson correlation (Li et al. 2023: Tmax + PRCP)
    corr_tmax = pivot_tmax.corr().fillna(0).values
    corr_prcp = pivot_prcp.corr().fillna(0).values
    
    # Combined correlation (average)
    adj_pearson = (np.abs(corr_tmax) + np.abs(corr_prcp)) / 2
    np.fill_diagonal(adj_pearson, 0)
    
    # Threshold small correlations
    adj_pearson[adj_pearson < 0.3] = 0
    
    print(f"   Pearson adjacency: {(adj_pearson > 0).sum()} edges")
    
    # GPD-weighted adjacency (DI-GNN)
    weights = np.zeros(n_stations)
    for i, station in enumerate(stations):
        params = gpd_params.get(station, {'xi': 0, 'sigma': 1})
        xi = params.get('xi', 0)
        sigma = params.get('sigma', 1)
        sigma_max = max(p.get('sigma', 1) for p in gpd_params.values())
        weights[i] = 1 + abs(xi) + sigma / sigma_max
    
    # Outer product of weights
    weight_matrix = np.outer(weights, weights)
    adj_gpd = adj_pearson * weight_matrix
    
    # Normalize
    if adj_gpd.max() > 0:
        adj_gpd = adj_gpd / adj_gpd.max()
    
    print(f"   GPD-weighted adjacency: computed")
    
    return adj_pearson, adj_gpd, stations
