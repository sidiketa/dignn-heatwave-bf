"""Publication and thesis figures."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.cm as cm

from ..config import DATA_DIR, BF_SHAPEFILE

PLOT_PARAMS = {
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
}

NAVY   = '#1A5276'
GRAY   = '#5D6D7E'
TERRA  = '#CA6F1E'
RED_HW = '#C0392B'
GREEN  = '#1E8449'
LIGHT  = '#EBF5FB'

PLOT_PARAMS_THESE = {
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.4,
    'axes.spines.top': False,
    'axes.spines.right': False,
}


def create_crossval_heatmap(results_df, variables, output_dir):
    """
    Create cross-validation heatmap showing correlation and bias by station.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Cross-validation results from cross_validate_with_anam()
    variables : list
        List of (var_name, gsod_col, anam_col, unit) tuples
    output_dir : Path
        Directory to save output files
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("   WARNING: matplotlib not available, skipping heatmap creation")
        return
    
    print("\n   Creating cross-validation heatmap...")
    
    # Filter to station-level results (exclude 'ALL')
    station_results = results_df[results_df['Station'] != 'ALL'].copy()
    
    if len(station_results) == 0:
        print("   WARNING: No station-level results, skipping heatmap")
        return
    
    # Apply publication settings
    plt.rcParams.update(PLOT_PARAMS)
    
    # Create pivot tables for correlation and bias
    var_order = [v[0] for v in variables]
    
    corr_pivot = station_results.pivot(
        index='Station', columns='Variable', values='Correlation'
    ).reindex(columns=var_order)
    
    bias_pivot = station_results.pivot(
        index='Station', columns='Variable', values='Bias'
    ).reindex(columns=var_order)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Correlation heatmap
    ax = axes[0]
    corr_data = corr_pivot.values
    im = ax.imshow(corr_data, cmap='RdYlGn', aspect='auto', vmin=0.2, vmax=1.0)
    
    ax.set_xticks(np.arange(len(corr_pivot.columns)))
    ax.set_yticks(np.arange(len(corr_pivot.index)))
    ax.set_xticklabels(corr_pivot.columns, fontsize=10)
    ax.set_yticklabels(corr_pivot.index, fontsize=9)
    
    # Add text annotations
    for i in range(len(corr_pivot.index)):
        for j in range(len(corr_pivot.columns)):
            val = corr_data[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       color=color, fontsize=8)
    
    ax.set_title('(a) Correlation (r)', fontsize=11, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation', fontsize=10)
    
    # Bias heatmap
    ax = axes[1]
    bias_data = bias_pivot.values
    max_abs = np.nanmax(np.abs(bias_data))
    im = ax.imshow(bias_data, cmap='RdBu_r', aspect='auto', 
                   vmin=-max_abs, vmax=max_abs)
    
    ax.set_xticks(np.arange(len(bias_pivot.columns)))
    ax.set_yticks(np.arange(len(bias_pivot.index)))
    ax.set_xticklabels(bias_pivot.columns, fontsize=10)
    ax.set_yticklabels(bias_pivot.index, fontsize=9)
    
    # Add text annotations
    for i in range(len(bias_pivot.index)):
        for j in range(len(bias_pivot.columns)):
            val = bias_data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:+.1f}', ha='center', va='center', 
                       color='black', fontsize=8)
    
    ax.set_title('(b) Bias (GSOD − ANAM)', fontsize=11, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Bias', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure (PNG + PDF)
    fig_path = output_dir / "fig_cross_validation_heatmap"
    plt.savefig(f"{fig_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   Saved: {fig_path}.png")
    
    # Also save the pivot tables as CSV for reference
    corr_pivot.to_csv(output_dir / "cross_validation_correlation_matrix.csv")
    bias_pivot.to_csv(output_dir / "cross_validation_bias_matrix.csv")
    
    print(f"   Saved: {output_dir / 'cross_validation_correlation_matrix.csv'}")
    print(f"   Saved: {output_dir / 'cross_validation_bias_matrix.csv'}")


def create_heatwave_figures(df, events_df, output_dir, percentile=90):
    """
    Create publication-quality figures for heatwave characteristics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original GSOD data with DATE, NAME, MAX columns
    events_df : pd.DataFrame
        Output from identify_heatwave_events()
    output_dir : Path
        Directory to save figures
    percentile : int
        Percentile used for threshold (default: 90)
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("   WARNING: matplotlib not available, skipping figure creation")
        return
    
    # Apply publication settings
    plt.rcParams.update(PLOT_PARAMS)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    hw_events = events_df[events_df['Is_Heatwave']]
    hot_days = events_df[~events_df['Is_Heatwave']]
    
    print("\n   Creating publication figures...")
    
    # =========================================================================
    # FIGURE 1: Duration Distribution (Boxplot)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(6, 4))
    
    duration_range = range(1, 16)
    hot_data = []
    hw_data = []
    
    years = sorted(events_df['Year'].unique())
    
    for d in duration_range:
        hot_counts = hot_days[hot_days['Duration'] == d].groupby('Year').size()
        hw_counts = hw_events[hw_events['Duration'] == d].groupby('Year').size()
        
        hot_counts = hot_counts.reindex(years, fill_value=0)
        hw_counts = hw_counts.reindex(years, fill_value=0)
        
        hot_data.append(hot_counts.values)
        hw_data.append(hw_counts.values)
    
    positions = np.arange(1, 16)
    width = 0.35
    
    bp1 = ax.boxplot(hot_data, positions=positions-width/2, widths=width,
                      patch_artist=True, showfliers=True,
                      flierprops=dict(marker='+', markersize=4, markeredgecolor='gray'))
    bp2 = ax.boxplot(hw_data, positions=positions+width/2, widths=width,
                      patch_artist=True, showfliers=True,
                      flierprops=dict(marker='+', markersize=4, markeredgecolor='gray'))
    
    for patch in bp1['boxes']:
        patch.set_facecolor('#808080')
        patch.set_alpha(0.7)
    for patch in bp2['boxes']:
        patch.set_facecolor('#6BAED6')
        patch.set_alpha(0.7)
    
    for element in ['whiskers', 'caps']:
        for line in bp1[element] + bp2[element]:
            line.set_color('black')
            line.set_linewidth(0.8)
    
    for median in bp1['medians'] + bp2['medians']:
        median.set_color('#E74C3C')
        median.set_linewidth(1.2)
    
    ax.set_xlabel('Duration (day)')
    ax.set_ylabel('No. of event')
    ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
    ax.set_xticklabels(['1', '3', '5', '7', '9', '11', '13', '15'])
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, None)
    ax.legend([Patch(facecolor='#808080', alpha=0.7),
               Patch(facecolor='#6BAED6', alpha=0.7)],
              ['Épisode court (< 3 j)', 'Vague de chaleur (≥ 3 j)'], loc='upper right', frameon=True)
    ax.grid(axis='y', alpha=0.3, linestyle='-')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_duration_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: fig_duration_distribution.png")
    
    # =========================================================================
    # FIGURE 2: Seasonality + Annual Frequency (2-panel)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # (a) Monthly Distribution
    ax = axes[0]
    monthly_hw = hw_events.groupby('Month').size()
    months = range(1, 13)
    month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    monthly_hw = monthly_hw.reindex(months, fill_value=0)
    
    ax.bar(months, monthly_hw, color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of heatwave events')
    ax.set_xticks(months)
    ax.set_xticklabels(month_names)
    ax.set_title('(a)', loc='left', fontweight='bold')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis='y', alpha=0.3)
    
    # Annotate peak months
    for m, v in zip(months, monthly_hw):
        if v > 50:
            ax.annotate(f'{v}', (m, v + 5), ha='center', fontsize=9)
    
    # (b) Annual Frequency with trend
    ax = axes[1]
    yearly_hw = hw_events.groupby('Year').size()
    yearly_hw = yearly_hw.reindex(years, fill_value=0)
    
    ax.bar(years, yearly_hw, color='#3498DB', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Trend line
    z = np.polyfit(list(years), yearly_hw.values, 1)
    p = np.poly1d(z)
    ax.plot(years, p(list(years)), 'k--', linewidth=1.5, label=f'Trend: {z[0]:+.1f} events/year')
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of heatwave events')
    ax.set_xticks(years[::2])
    ax.set_xticklabels([str(y) for y in years[::2]], rotation=45)
    ax.set_title('(b)', loc='left', fontweight='bold')
    ax.legend(loc='upper left', frameon=True)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_seasonality_frequency.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: fig_seasonality_frequency.png")
    
    # =========================================================================
    # FIGURE 3: Threshold Annual Cycle by Station
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Ensure DATE and MONTH columns exist
    df_plot = df.copy()
    if 'DATE' not in df_plot.columns:
        df_plot['DATE'] = pd.to_datetime(df_plot['YEARMODA'])
    df_plot['MONTH'] = df_plot['DATE'].dt.month
    
    stations = sorted(df_plot['NAME'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(stations)))
    month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    
    for i, station in enumerate(stations):
        sdf = df_plot[df_plot['NAME'] == station]
        
        # Compute monthly threshold (percentile)
        monthly_thresh = sdf.groupby('MONTH')['MAX'].quantile(percentile / 100)
        
        short_name = station.split()[0][:8]
        ax.plot(monthly_thresh.index, monthly_thresh.values,
                'o-', color=colors[i], linewidth=1.5, markersize=4,
                label=short_name)
    
    ax.set_xlabel('Month')
    ax.set_ylabel(f'{percentile}th percentile threshold (°C)')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.legend(loc='lower right', ncol=3, frameon=True)
    ax.grid(alpha=0.3)
    ax.set_xlim(0.5, 12.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_threshold_annual_cycle.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: fig_threshold_annual_cycle.png")
    
    # =========================================================================
    # FIGURE 4: Station Comparison (2-panel boxplots)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    station_order = hw_events.groupby('Station')['Peak_Temp'].mean().sort_values(ascending=False).index
    short_names = [s.split()[0][:8] for s in station_order]
    
    # (a) Duration by station
    ax = axes[0]
    data_dur = [hw_events[hw_events['Station'] == s]['Duration'].values for s in station_order]
    bp = ax.boxplot(data_dur, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=4))
    
    for patch in bp['boxes']:
        patch.set_facecolor('#3498DB')
        patch.set_alpha(0.6)
    
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    ax.set_ylabel('Duration (days)')
    ax.set_title('(a)', loc='left', fontweight='bold')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis='y', alpha=0.3)
    
    # (b) Peak temperature by station
    ax = axes[1]
    data_temp = [hw_events[hw_events['Station'] == s]['Peak_Temp'].values for s in station_order]
    bp = ax.boxplot(data_temp, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=4))
    
    import matplotlib.cm as cm
    colors = cm.YlOrRd(np.linspace(0.3, 0.8, len(station_order)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    ax.set_ylabel('Peak temperature (°C)')
    ax.set_title('(b)', loc='left', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_station_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: fig_station_comparison.png")


def create_station_network_map(df, events_df, output_dir, shapefile_path=None):
    """
    Create publication-quality station network map showing total heatwave events.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original GSOD data with NAME, LATITUDE, LONGITUDE columns
    events_df : pd.DataFrame
        Output from identify_heatwave_events()
    output_dir : Path
        Directory to save figures
    shapefile_path : str or Path, optional
        Path to Burkina Faso admin level 0 shapefile
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize
        from matplotlib.patches import Polygon as MplPolygon, FancyArrow
        from itertools import combinations
    except ImportError:
        print("   WARNING: matplotlib not available, skipping map creation")
        return
    
    # Contour national : chargeur commun (shapefile officiel, repli embarqué)
    bf_coords = load_bf_boundary_coords(simplify_tol=0.01)
    
    plt.rcParams.update(PLOT_PARAMS)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get station coordinates
    stations_info = df.groupby('NAME').agg({
        'LATITUDE': 'first',
        'LONGITUDE': 'first',
    }).reset_index()
    
    # Compute TOTAL HW per station
    hw_events = events_df[events_df['Is_Heatwave']]
    hw_per_station = hw_events.groupby('Station').size()
    hw_per_station = hw_per_station.reset_index()
    hw_per_station.columns = ['NAME', 'Total_HW']
    
    # Merge with coordinates
    stations_info = stations_info.merge(hw_per_station, on='NAME', how='left')
    stations_info['Total_HW'] = stations_info['Total_HW'].fillna(0).astype(int)
    
    print(f"   Creating station network map...")
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set map extent
    ax.set_xlim(-6.0, 3.0)
    ax.set_ylim(9.0, 15.6)
    
    # Background color (surrounding area)
    ax.set_facecolor('#e0e0e0')
    
    # Draw Burkina Faso boundary (bf_coords fourni par le chargeur commun)
    bf_polygon = MplPolygon(bf_coords, closed=True, 
                            facecolor='#f5f5dc',
                            edgecolor='#1a1a1a',
                            linewidth=1.8, zorder=2)
    ax.add_patch(bf_polygon)
    
    # Draw edges (connections between all stations) - graph structure
    lons = stations_info['LONGITUDE'].values
    lats = stations_info['LATITUDE'].values
    
    for i, j in combinations(range(len(stations_info)), 2):
        ax.plot([lons[i], lons[j]], [lats[i], lats[j]], 
                color='#404040', linewidth=1.0, alpha=0.6, zorder=3)
    
    # Color scale for HW events (total counts)
    vmin = stations_info['Total_HW'].min() - 2
    vmax = stations_info['Total_HW'].max() + 2
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps['RdYlBu_r']
    
    # Plot stations as nodes with total HW count inside
    node_size = 0.32
    for idx, row in stations_info.iterrows():
        color = cmap(norm(row['Total_HW']))
        
        # Draw node circle
        circle = plt.Circle((row['LONGITUDE'], row['LATITUDE']), node_size, 
                             color=color, ec='black', linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        
        # Add total HW count inside node - BLACK TEXT
        hw_label = str(int(row['Total_HW']))
        ax.text(row['LONGITUDE'], row['LATITUDE'], hw_label,
                ha='center', va='center', fontsize=10, fontweight='bold',
                color='black', zorder=6)
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label('Total HW events (2010-2024)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    
    # Set tick formatting
    ax.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2])
    ax.set_yticks([9, 10, 11, 12, 13, 14, 15])
    ax.set_xticklabels(['5°W', '4°W', '3°W', '2°W', '1°W', '0°', '1°E', '2°E'])
    ax.set_yticklabels(['9°N', '10°N', '11°N', '12°N', '13°N', '14°N', '15°N'])
    
    # Labels
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    
    # Add scale bar (100 km)
    # At ~12°N latitude, 1° longitude ≈ 109 km, so 100 km ≈ 0.917°
    scale_lon = 1.3
    scale_lat = 9.5
    scale_length = 100 / 109  # Convert 100 km to degrees
    
    # Scale bar with end caps
    ax.plot([scale_lon, scale_lon + scale_length], [scale_lat, scale_lat], 'k-', linewidth=3, zorder=7)
    ax.plot([scale_lon, scale_lon], [scale_lat - 0.08, scale_lat + 0.08], 'k-', linewidth=2, zorder=7)
    ax.plot([scale_lon + scale_length, scale_lon + scale_length], [scale_lat - 0.08, scale_lat + 0.08], 'k-', linewidth=2, zorder=7)
    ax.text(scale_lon + scale_length/2, scale_lat + 0.2, '100 km', ha='center', fontsize=9, fontweight='bold', zorder=7)
    
    # Professional North Arrow
    arrow_x = 2.4
    arrow_y = 14.3
    arrow_length = 0.6
    
    # Draw the north arrow (professional style with filled triangle)
    arrow = FancyArrow(arrow_x, arrow_y, 0, arrow_length, 
                       head_width=0.15, head_length=0.15, 
                       fc='black', ec='black', linewidth=1, zorder=8)
    ax.add_patch(arrow)
    
    # Add "N" label
    ax.text(arrow_x, arrow_y + arrow_length + 0.2, 'N', 
            ha='center', va='bottom', fontsize=14, fontweight='bold', zorder=8)
    
    # Add small circle at base of arrow
    base_circle = plt.Circle((arrow_x, arrow_y), 0.06, fc='black', ec='black', zorder=8)
    ax.add_patch(base_circle)
    
    # Set aspect ratio
    ax.set_aspect(1.0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_station_network_map.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: fig_station_network_map.png")


def create_heatwave_figures_mam(df, events_df, thresholds, output_dir, percentile=90):
    """
    Create publication-quality figures for MAM-based heatwave characteristics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original GSOD data
    events_df : pd.DataFrame
        Output from identify_heatwave_events_mam()
    thresholds : dict
        Station -> threshold mapping from identify_heatwave_events_mam()
    output_dir : Path
        Directory to save figures
    percentile : int
        Percentile used for threshold (default: 90)
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.ticker import MaxNLocator
        import matplotlib.cm as cm
    except ImportError:
        print("   WARNING: matplotlib not available, skipping figure creation")
        return
    
    plt.rcParams.update(PLOT_PARAMS)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    hw_events = events_df[events_df['Is_Heatwave']]
    hot_days = events_df[~events_df['Is_Heatwave']]
    years = sorted(events_df['Year'].unique())
    stations = sorted(df['NAME'].unique())
    
    print("\n   Creating publication figures (MAM definition)...")
    
    # =========================================================================
    # FIGURE 1: Duration Distribution
    # =========================================================================
    fig, ax = plt.subplots(figsize=(6, 4))
    
    duration_range = range(1, 16)
    hot_data = []
    hw_data = []
    
    for d in duration_range:
        hot_counts = hot_days[hot_days['Duration'] == d].groupby('Year').size()
        hw_counts = hw_events[hw_events['Duration'] == d].groupby('Year').size()
        hot_counts = hot_counts.reindex(years, fill_value=0)
        hw_counts = hw_counts.reindex(years, fill_value=0)
        hot_data.append(hot_counts.values)
        hw_data.append(hw_counts.values)
    
    positions = np.arange(1, 16)
    width = 0.35
    
    bp1 = ax.boxplot(hot_data, positions=positions-width/2, widths=width,
                      patch_artist=True, showfliers=True,
                      flierprops=dict(marker='+', markersize=4, markeredgecolor='gray'))
    bp2 = ax.boxplot(hw_data, positions=positions+width/2, widths=width,
                      patch_artist=True, showfliers=True,
                      flierprops=dict(marker='+', markersize=4, markeredgecolor='gray'))
    
    for patch in bp1['boxes']:
        patch.set_facecolor('#808080')
        patch.set_alpha(0.7)
    for patch in bp2['boxes']:
        patch.set_facecolor('#6BAED6')
        patch.set_alpha(0.7)
    
    for element in ['whiskers', 'caps']:
        for line in bp1[element] + bp2[element]:
            line.set_color('black')
            line.set_linewidth(0.8)
    
    for median in bp1['medians'] + bp2['medians']:
        median.set_color('#E74C3C')
        median.set_linewidth(1.2)
    
    ax.set_xlabel('Duration (day)')
    ax.set_ylabel('No. of event')
    ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
    ax.set_xticklabels(['1', '3', '5', '7', '9', '11', '13', '15'])
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, None)
    ax.legend([Patch(facecolor='#808080', alpha=0.7),
               Patch(facecolor='#6BAED6', alpha=0.7)],
              ['Épisode court (< 3 j)', 'Vague de chaleur (≥ 3 j)'], loc='upper right', frameon=True)
    ax.grid(axis='y', alpha=0.3, linestyle='-')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_duration_distribution_mam.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: fig_duration_distribution_mam.png")
    
    # =========================================================================
    # FIGURE 2: Seasonality + Annual Frequency
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # (a) Monthly
    ax = axes[0]
    monthly_hw = hw_events.groupby('Month').size()
    months = range(1, 13)
    month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    monthly_hw = monthly_hw.reindex(months, fill_value=0)
    
    ax.bar(months, monthly_hw, color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of heatwave events')
    ax.set_xticks(months)
    ax.set_xticklabels(month_names)
    ax.set_title('(a)', loc='left', fontweight='bold')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis='y', alpha=0.3)
    
    for m, v in zip(months, monthly_hw):
        if v > 20:
            ax.annotate(f'{v}', (m, v + 1), ha='center', fontsize=9)
    
    # (b) Annual
    ax = axes[1]
    yearly_hw = hw_events.groupby('Year').size()
    yearly_hw = yearly_hw.reindex(years, fill_value=0)
    
    ax.bar(years, yearly_hw, color='#3498DB', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    z = np.polyfit(list(years), yearly_hw.values, 1)
    p = np.poly1d(z)
    ax.plot(years, p(list(years)), 'k--', linewidth=1.5, label=f'Trend: {z[0]:+.2f} events/year')
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of heatwave events')
    ax.set_xticks(years[::2])
    ax.set_xticklabels([str(y) for y in years[::2]], rotation=45)
    ax.set_title('(b)', loc='left', fontweight='bold')
    ax.legend(loc='upper left', frameon=True)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_seasonality_frequency_mam.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: fig_seasonality_frequency_mam.png")
    
    # =========================================================================
    # FIGURE 3: Fixed Threshold by Station (horizontal bar chart)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sorted_stations = sorted(thresholds.keys(), key=lambda x: thresholds[x], reverse=True)
    short_names = [s.split()[0][:10] for s in sorted_stations]
    thresh_values = [thresholds[s] for s in sorted_stations]
    
    colors = cm.YlOrRd(np.linspace(0.3, 0.8, len(sorted_stations)))
    bars = ax.barh(range(len(sorted_stations)), thresh_values, color=colors, 
                   edgecolor='black', linewidth=0.5)
    
    for i, (bar, val) in enumerate(zip(bars, thresh_values)):
        ax.text(val + 0.2, i, f'{val:.1f}°C', va='center', fontsize=9)
    
    ax.set_yticks(range(len(sorted_stations)))
    ax.set_yticklabels(short_names)
    ax.set_xlabel(f'{percentile}th percentile threshold (°C)')
    ax.set_xlim(38, 45)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_threshold_by_station_mam.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: fig_threshold_by_station_mam.png")
    
    # =========================================================================
    # FIGURE 4: Station Comparison
    # =========================================================================
    if len(hw_events) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        station_order = hw_events.groupby('Station')['Peak_Temp'].mean().sort_values(ascending=False).index
        short_names = [s.split()[0][:8] for s in station_order]
        
        # (a) Duration
        ax = axes[0]
        data_dur = [hw_events[hw_events['Station'] == s]['Duration'].values for s in station_order]
        bp = ax.boxplot(data_dur, patch_artist=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=4))
        
        for patch in bp['boxes']:
            patch.set_facecolor('#3498DB')
            patch.set_alpha(0.6)
        
        ax.set_xticklabels(short_names, rotation=45, ha='right')
        ax.set_ylabel('Duration (days)')
        ax.set_title('(a)', loc='left', fontweight='bold')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis='y', alpha=0.3)
        
        # (b) Peak temp
        ax = axes[1]
        data_temp = [hw_events[hw_events['Station'] == s]['Peak_Temp'].values for s in station_order]
        bp = ax.boxplot(data_temp, patch_artist=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=4))
        
        colors = cm.YlOrRd(np.linspace(0.3, 0.8, len(station_order)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xticklabels(short_names, rotation=45, ha='right')
        ax.set_ylabel('Peak temperature (°C)')
        ax.set_title('(b)', loc='left', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'fig_station_comparison_mam.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   Saved: fig_station_comparison_mam.png")


def fig_tmax_timeseries(df, events_df, output_dir, percentile=90):
    """
    Figure thèse : Série temporelle Tmax + épisodes HW par station.
    - Courbe Tmax journalière (gris clair)
    - Moyenne mobile 30 jours (bleu marine)
    - Seuil PKL (pointillé rouge)
    - Épisodes HW (zones ombrées rouges)
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update(PLOT_PARAMS_THESE)

    df = df.copy()
    if 'DATE' not in df.columns:
        df['DATE'] = pd.to_datetime(df['YEARMODA'])

    hw_events = events_df[events_df['Is_Heatwave']]
    stations  = sorted(df['NAME'].unique())
    n         = len(stations)
    ncols     = 2
    nrows     = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(14, 3.5 * nrows),
                              sharex=False)
    axes = np.array(axes).flatten()

    for idx, station in enumerate(stations):
        ax  = axes[idx]
        sdf = df[df['NAME'] == station].sort_values('DATE')

        if 'Tthresh' in sdf.columns:
            thresh = sdf['Tthresh'].iloc[0]
        else:
            thresh = np.percentile(sdf['MAX'].dropna(), percentile)

        ax.plot(sdf['DATE'], sdf['MAX'],
                color='#BDC3C7', linewidth=0.5, alpha=0.8, zorder=1)

        rolling = sdf.set_index('DATE')['MAX'].rolling('30D', min_periods=15).mean()
        ax.plot(rolling.index, rolling.values,
                color=NAVY, linewidth=1.4, zorder=3, label='Moy. mobile 30 j')

        ax.axhline(thresh, color=RED_HW, linewidth=1.0,
                   linestyle='--', zorder=2,
                   label=f'Seuil PKL ({percentile}e pct.)')

        station_hw = hw_events[hw_events['Station'] == station]
        first_hw   = True
        for _, evt in station_hw.iterrows():
            label = 'Épisode HW' if first_hw else None
            ax.axvspan(evt['Start_Date'], evt['End_Date'],
                       alpha=0.25, color=RED_HW, zorder=2, label=label)
            first_hw = False

        ax.set_title(station, fontsize=10, fontweight='bold',
                     color=NAVY, pad=4)
        ax.set_ylabel('$T_{\\max}$ (°C)', fontsize=9)
        ax.set_ylim(20, 50)
        ax.grid(axis='y', alpha=0.3, linestyle=':')
        ax.tick_params(axis='x', rotation=30, labelsize=8)

        if idx == 0:
            ax.legend(loc='upper left', fontsize=8,
                      framealpha=0.85, ncol=1)

        n_hw = len(station_hw)
        ax.text(0.98, 0.96, f'n = {n_hw} épisodes',
                transform=ax.transAxes,
                ha='right', va='top', fontsize=8,
                color=RED_HW, fontstyle='italic')

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        '$T_{\\max}$ journalière et épisodes de vagues de chaleur\n'
        'par station synoptique — Burkina Faso 2010–2024',
        fontsize=12, fontweight='bold', color=NAVY, y=1.01)

    plt.tight_layout(pad=1.5)
    out = Path(output_dir) / 'fig_tmax_timeseries.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: fig_tmax_timeseries.png")


def fig_heatmap_seasonality(events_df, output_dir):
    """
    Figure thèse : Heatmap distribution mensuelle des épisodes HW par station.
    Justifie le choix du critère PKL annuel vs MAM.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update(PLOT_PARAMS_THESE)

    hw_events = events_df[events_df['Is_Heatwave']].copy()
    stations  = sorted(hw_events['Station'].unique())
    months    = range(1, 13)
    month_lbl = ['Jan','Fév','Mar','Avr','Mai','Jun',
                  'Jul','Aoû','Sep','Oct','Nov','Déc']

    pivot = pd.pivot_table(hw_events, values='Duration',
                            index='Station', columns='Month',
                            aggfunc='count', fill_value=0)
    pivot = pivot.reindex(columns=months, fill_value=0)
    pivot = pivot.reindex(stations)
    short_names = [s.split()[0] for s in stations]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    cmap = LinearSegmentedColormap.from_list(
        'hw_cmap', ['#FFFFFF', '#FADBD8', '#E74C3C', '#922B21'])
    im   = ax.imshow(pivot.values, cmap=cmap, aspect='auto',
                     vmin=0, vmax=max(1, pivot.values.max()))

    for i in range(len(stations)):
        for j in range(len(months)):
            val = pivot.values[i, j]
            color = 'white' if val > pivot.values.max() * 0.6 else '#2C3E50'
            if val > 0:
                ax.text(j, i, str(val),
                        ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_lbl, fontsize=10)
    ax.set_yticks(range(len(stations)))
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_xlabel('Mois', fontsize=11)
    ax.set_ylabel('Station', fontsize=11)

    import matplotlib.patches as mpatches
    for j in [2, 3, 4]:
        rect = mpatches.Rectangle(
            (j - 0.5, -0.5), 1, len(stations),
            linewidth=2, edgecolor=NAVY,
            facecolor='none', zorder=5)
        ax.add_patch(rect)
    ax.text(3, -0.85, 'MAM', ha='center', fontsize=9,
            color=NAVY, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Nombre d'épisodes HW", fontsize=10)

    ax.set_title(
        'Distribution mensuelle des épisodes de vagues de chaleur par station\n'
        'Burkina Faso — 2010–2024 (critère PKL annuel)',
        fontsize=11, fontweight='bold', color=NAVY, pad=8)

    plt.tight_layout()
    out = Path(output_dir) / 'fig_heatmap_seasonality.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: fig_heatmap_seasonality.png")


def load_bf_boundary_coords(simplify_tol=0.01):
    """
    Charge le contour national du Burkina Faso depuis le shapefile officiel.

    Cherche dans cet ordre :
      1. BF_SHAPEFILE (config.py → data/shapefiles/...)
      2. tout .shp présent dans data/shapefiles/
      3. repli : contour approximatif embarqué

    Retourne une liste de (lon, lat).
    """
    import os
    os.environ['SHAPE_RESTORE_SHX'] = 'YES'   # tolère l'absence de .shx

    candidates = []
    try:
        candidates.append(Path(BF_SHAPEFILE))
    except NameError:
        pass
    shp_dir = DATA_DIR / "shapefiles"
    if shp_dir.exists():
        candidates += sorted(shp_dir.glob("*.shp"))

    for shp in candidates:
        if shp and shp.exists():
            try:
                import geopandas as gpd
                gdf = gpd.read_file(shp)
                geom = gdf.geometry.iloc[0]
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                main = max(polys, key=lambda p: p.area)
                simp = main.simplify(simplify_tol, preserve_topology=True)
                coords = list(simp.exterior.coords)
                print(f"   Contour BF chargé depuis {shp.name} ({len(coords)} points)")
                return coords
            except Exception as e:
                print(f"   WARNING: lecture {shp.name} échouée ({str(e)[:50]})")

    # Repli embarqué
    print("   Contour BF : repli approximatif embarqué")
    bf_lon = [-5.52, -4.75, -3.50, -2.10, -0.24, 0.63, 1.24, 1.97,
               2.40, 1.85, 0.50, -0.07, -0.64, -1.62, -2.68, -3.48,
              -3.68, -4.66, -5.10, -5.46, -5.47, -5.22, -5.23, -5.52]
    bf_lat = [14.98, 15.10, 15.18, 14.93, 15.07, 14.42, 12.83, 12.07,
              11.43, 11.00, 10.57, 10.63, 11.08, 10.57, 10.58, 10.15,
              10.65, 10.82, 11.37, 11.44, 12.38, 12.63, 13.70, 14.98]
    return list(zip(bf_lon, bf_lat))


def fig_graph_pearson(df, adj_pearson, adj_gpd, stations_list, output_dir):
    """
    Figure thèse : Graphe de stations avec poids Pearson et GPD.
    Nœuds positionnés selon les coordonnées géographiques réelles.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.patheffects as pe
    plt.rcParams.update(PLOT_PARAMS_THESE)

    coords = df.groupby('NAME')[['LATITUDE', 'LONGITUDE']].first()
    coords = coords.reindex(stations_list)
    lats   = coords['LATITUDE'].values
    lons   = coords['LONGITUDE'].values
    n      = len(stations_list)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Contour national depuis le shapefile officiel (repli embarqué sinon)
    bf_coords = load_bf_boundary_coords(simplify_tol=0.01)
    bf_lon = [c[0] for c in bf_coords]
    bf_lat = [c[1] for c in bf_coords]

    for ax_idx, (matrix, title, cmap_name) in enumerate([
        (adj_pearson, '(a) Adjacence Pearson ($\\rho_{ij}$)', 'Blues'),
        (adj_gpd,     '(b) Adjacence pondérée GPD ($A_{ij}$)', 'YlOrRd'),
    ]):
        ax = axes[ax_idx]
        ax.set_facecolor(LIGHT)

        bf_poly = mpatches.Polygon(
            list(zip(bf_lon, bf_lat)), closed=True,
            facecolor='#FDFBE4', edgecolor='#2C3E50',
            linewidth=1.5, zorder=1)
        ax.add_patch(bf_poly)

        vmax_edge = matrix.max() if matrix.max() > 0 else 1
        cmap_edge = plt.get_cmap(cmap_name)
        norm_edge = Normalize(vmin=0, vmax=vmax_edge)

        for i in range(n):
            for j in range(i + 1, n):
                w = matrix[i, j]
                if w > 0:
                    color = cmap_edge(norm_edge(w))
                    lw    = 0.5 + 4.0 * (w / vmax_edge)
                    ax.plot([lons[i], lons[j]],
                            [lats[i], lats[j]],
                            color=color, linewidth=lw,
                            alpha=0.75, zorder=2)

        ax.scatter(lons, lats, s=220, c=NAVY,
                   edgecolors='white', linewidths=1.5,
                   zorder=5, marker='o')

        for i, st in enumerate(stations_list):
            short = st.split()[0]
            txt = ax.text(lons[i], lats[i] + 0.18, short,
                          ha='center', va='bottom',
                          fontsize=8, color='#1A2951',
                          fontweight='bold', zorder=6)
            txt.set_path_effects([
                pe.withStroke(linewidth=2, foreground='white')])

        sm = cm.ScalarMappable(cmap=cmap_edge, norm=norm_edge)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.65, pad=0.02)
        cbar.set_label('Poids de l\'arête', fontsize=9)

        ax.set_xlim(-6.0, 2.8)
        ax.set_ylim(9.2, 15.8)
        ax.set_xlabel('Longitude (°)', fontsize=10)
        ax.set_ylabel('Latitude (°N)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold',
                     color=NAVY, pad=6)

        for lg in np.arange(-6, 3, 2):
            ax.axvline(lg, color='#CCCCCC', lw=0.3, ls=':', zorder=0)
        for la in np.arange(10, 16, 1):
            ax.axhline(la, color='#CCCCCC', lw=0.3, ls=':', zorder=0)

    fig.suptitle(
        'Structure du graphe de stations — Burkina Faso\n'
        'Arêtes : corrélation de Pearson (gauche) et pondération GPD (droite)',
        fontsize=12, fontweight='bold', color=NAVY)

    plt.tight_layout()
    out = Path(output_dir) / 'fig_graph_pearson.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: fig_graph_pearson.png")


def fig_gpd_parameters(gpd_params, output_dir):
    """
    Figure thèse : Paramètres GPD (ξ et σ) estimés par station.
    Valeurs ξ < 0 = queues bornées, caractéristique sahélienne.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update(PLOT_PARAMS_THESE)

    stations = sorted(gpd_params.keys())
    short    = [s.split()[0] for s in stations]
    xi_vals  = [gpd_params[s]['xi']    for s in stations]
    sig_vals = [gpd_params[s]['sigma'] for s in stations]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    colors_xi = [RED_HW if x < 0 else GREEN for x in xi_vals]
    bars = ax.bar(range(len(stations)), xi_vals,
                  color=colors_xi, alpha=0.8,
                  edgecolor='white', linewidth=0.8)
    ax.axhline(0, color='#2C3E50', linewidth=1.2)
    ax.axhline(-0.1,  color=GRAY, lw=0.8, ls='--', alpha=0.6,
               label='ξ = −0,1')
    ax.axhline(-0.34, color=GRAY, lw=0.8, ls=':', alpha=0.6,
               label='ξ = −0,34')
    for i, (bar, val) in enumerate(zip(bars, xi_vals)):
        ypos = val - 0.015 if val < 0 else val + 0.005
        va   = 'top'       if val < 0 else 'bottom'
        ax.text(i, ypos, f'{val:.3f}', ha='center', va=va,
                fontsize=8.5, fontweight='bold', color='#1A2951')
    ax.set_xticks(range(len(stations)))
    ax.set_xticklabels(short, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Paramètre de forme $\\xi$', fontsize=11)
    ax.set_title('(a) Paramètre de forme $\\xi$ — queues bornées ($\\xi < 0$)',
                 fontsize=10, fontweight='bold', color=NAVY, pad=6)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    ax.set_ylim(min(xi_vals) - 0.06, 0.10)
    ax.axhspan(min(xi_vals) - 0.06, 0, alpha=0.04, color=RED_HW, zorder=0)

    ax = axes[1]
    colors_sig = cm.YlOrRd(
        np.array(sig_vals) / max(sig_vals) * 0.8 + 0.1)
    bars2 = ax.bar(range(len(stations)), sig_vals,
                   color=colors_sig, alpha=0.85,
                   edgecolor='white', linewidth=0.8)
    for i, (bar, val) in enumerate(zip(bars2, sig_vals)):
        ax.text(i, val + 0.03, f'{val:.2f}', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold', color='#1A2951')
    ax.set_xticks(range(len(stations)))
    ax.set_xticklabels(short, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel("Paramètre d'échelle $\\sigma$ (°C)", fontsize=11)
    ax.set_title("(b) Paramètre d'échelle $\\sigma$",
                 fontsize=10, fontweight='bold', color=NAVY, pad=6)
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    ax.set_ylim(0, max(sig_vals) * 1.18)

    fig.suptitle(
        'Paramètres de la distribution de Pareto généralisée (GPD) par station\n'
        'Estimés sur la période d\'entraînement 2010–2022',
        fontsize=11, fontweight='bold', color=NAVY)

    plt.tight_layout()
    out = Path(output_dir) / 'fig_gpd_parameters.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: fig_gpd_parameters.png")


def fig_validation_timeline(output_dir, train_years=None, val_years=None):
    """
    Figure thèse : Schéma de la division temporelle hold-out chronologique.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update(PLOT_PARAMS_THESE)

    if train_years is None:
        train_years = list(range(2010, 2023))
    if val_years is None:
        val_years   = [2023, 2024]

    y_start = min(train_years)
    y_end   = max(val_years)
    y_split = max(train_years) + 0.5

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_facecolor('white')
    ax.set_xlim(y_start - 0.5, y_end + 0.5)
    ax.set_ylim(-1.2, 2.5)
    ax.axis('off')

    bar_y = 0.8
    bar_h = 0.55

    ax.barh(bar_y, width=y_split - y_start,
            left=y_start, height=bar_h,
            color=NAVY, alpha=0.85, zorder=2)
    ax.text((y_start + y_split) / 2, bar_y,
            f'Entraînement\n{y_start}–{max(train_years)}\n'
            f'({len(train_years)} ans — ~80 %)',
            ha='center', va='center', fontsize=10,
            color='white', fontweight='bold')

    ax.barh(bar_y, width=y_end - y_split + 0.5,
            left=y_split, height=bar_h,
            color=TERRA, alpha=0.85, zorder=2)
    ax.text((y_split + y_end + 0.5) / 2, bar_y,
            f'Validation\n{min(val_years)}–{max(val_years)}\n'
            f'({len(val_years)} ans — ~20 %)',
            ha='center', va='center', fontsize=10,
            color='white', fontweight='bold')

    ax.axvline(y_split, color='#E74C3C', linewidth=2.5,
               linestyle='--', zorder=3, ymin=0.1, ymax=0.95)
    ax.text(y_split, 1.55, 'Coupure temporelle\n(jan. 2023)',
            ha='center', va='bottom', fontsize=9,
            color='#E74C3C', fontweight='bold')

    ax.annotate('', xy=(y_end + 0.5, -0.35),
                xytext=(y_start - 0.5, -0.35),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.8))

    for yr in range(y_start, y_end + 1):
        ax.plot([yr, yr], [-0.45, -0.25], color='#2C3E50', lw=1.0)
        if yr % 2 == 0 or yr in val_years:
            ax.text(yr, -0.62, str(yr),
                    ha='center', va='top', fontsize=8.5, color='#2C3E50')

    ax.annotate(
        'Seuil $T_{seuil,i}$ et paramètres\nGPD calculés ici uniquement',
        xy=((y_start + y_split) / 2, bar_y - bar_h / 2 - 0.02),
        xytext=((y_start + y_split) / 2, -1.05),
        ha='center', fontsize=8.5, color=NAVY,
        arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.0))

    ax.annotate(
        'Ensemble strictement\ninédit lors de l\'évaluation',
        xy=((y_split + y_end + 0.5) / 2, bar_y - bar_h / 2 - 0.02),
        xytext=((y_split + y_end + 0.5) / 2, -1.05),
        ha='center', fontsize=8.5, color=TERRA,
        arrowprops=dict(arrowstyle='->', color=TERRA, lw=1.0))

    ax.set_title(
        'Stratégie de validation hold-out chronologique\n'
        'Burkina Faso — DI-GNN Prévision des vagues de chaleur',
        fontsize=11, fontweight='bold', color=NAVY,
        x=0.5, y=1.02, ha='center')

    plt.tight_layout()
    out = Path(output_dir) / 'fig_validation_timeline.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: fig_validation_timeline.png")


def create_thesis_figures(df, events_df, gpd_params=None,
                           adj_pearson=None, adj_gpd=None,
                           stations_list=None, output_dir=None,
                           percentile=90,
                           train_years=None, val_years=None):
    """
    Génère toutes les figures de thèse en un seul appel.
    À appeler dans main() après compute_adjacency_matrices().

    Paramètres
    ----------
    df : pd.DataFrame           Données GSOD prétraitées
    events_df : pd.DataFrame    Sorties de identify_heatwave_events()
    gpd_params : dict           Sorties de fit_gpd_parameters()
    adj_pearson : np.ndarray    Matrice Pearson (n×n)
    adj_gpd : np.ndarray        Matrice GPD pondérée (n×n)
    stations_list : list        Liste ordonnée des stations
    output_dir : Path           Répertoire de sauvegarde
    percentile : int            Percentile PKL (défaut 90)
    train_years : list          Années entraînement (défaut 2010–2022)
    val_years : list            Années validation (défaut 2023–2024)
    """
    if output_dir is None:
        output_dir = Path('.') / 'thesis_figures'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if train_years is None:
        train_years = list(range(2010, 2023))
    if val_years is None:
        val_years   = [2023, 2024]

    print("\n" + "=" * 60)
    print("FIGURES DE THÈSE")
    print("=" * 60)

    print("\n   [1/5] Série temporelle Tmax + épisodes HW...")
    try:
        fig_tmax_timeseries(df, events_df, output_dir, percentile)
    except Exception as e:
        print(f"   ERREUR figure 1 : {e}")

    print("\n   [2/5] Heatmap saisonnalité HW (mois × station)...")
    try:
        fig_heatmap_seasonality(events_df, output_dir)
    except Exception as e:
        print(f"   ERREUR figure 2 : {e}")

    if adj_pearson is not None and adj_gpd is not None \
            and stations_list is not None:
        print("\n   [3/5] Graphe de stations (Pearson + GPD)...")
        try:
            fig_graph_pearson(df, adj_pearson, adj_gpd,
                              stations_list, output_dir)
        except Exception as e:
            print(f"   ERREUR figure 3 : {e}")
    else:
        print("\n   [3/5] Graphe — ignoré (matrices non fournies)")

    if gpd_params is not None:
        print("\n   [4/5] Paramètres GPD par station...")
        try:
            fig_gpd_parameters(gpd_params, output_dir)
        except Exception as e:
            print(f"   ERREUR figure 4 : {e}")
    else:
        print("\n   [4/5] Paramètres GPD — ignoré (gpd_params non fourni)")

    print("\n   [5/5] Timeline de la division temporelle...")
    try:
        fig_validation_timeline(output_dir, train_years, val_years)
    except Exception as e:
        print(f"   ERREUR figure 5 : {e}")

    print(f"\n   Figures de thèse sauvegardées dans : {output_dir}")
