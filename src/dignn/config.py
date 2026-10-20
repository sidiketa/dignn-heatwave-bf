"""
Configuration for DI-GNN heatwave forecasting in Burkina Faso.
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
# Project root: two levels above src/dignn/config.py.
# DIGNN_DATA_DIR overrides the data folder (e.g. data/sample for the synthetic set).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("DIGNN_DATA_DIR", PROJECT_ROOT / "data")).resolve()
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA FILES
# =============================================================================
GSOD_FILE = DATA_DIR / "gsod_bf.csv"
ONI_FILE  = DATA_DIR / "oni"
NAO_FILE  = DATA_DIR / "nao"
ANAM_FILE = DATA_DIR / "anam_bf.csv"

# Burkina Faso admin boundary shapefile (for station network map)
# Source: OCHA Humanitarian Data Exchange (HDX) - bfa_admbnda_adm0_igb_20200323.shp
BF_SHAPEFILE = DATA_DIR / "shapefiles" / "bfa_admbnda_adm0_igb_20200323.shp"

# =============================================================================
# TEMPORAL CONFIGURATION
# =============================================================================
C_IN = 10   # Default input window: 10 days of history (config de référence CIN10_COUT3)
C_OUT = 3   # Default forecast horizon: 3 days ahead (config de référence CIN10_COUT3)

# =============================================================================
# ABLATION STUDY: INPUT/OUTPUT WINDOW ANALYSIS
# =============================================================================
# Experiments to evaluate:
#   1. Skill degradation with forecast horizon (fixed C_IN=10)
#   2. Impact of input context length (fixed C_OUT=3)

C_IN_OPTIONS = [7, 10, 14]    # Input window lengths to test
C_OUT_OPTIONS = [1, 3, 5, 7]  # Forecast horizons to test

# Full experimental grid for ablation study
ABLATION_EXPERIMENTS = [
    # Vary C_OUT with fixed C_IN=10 (forecast horizon analysis)
    {'c_in': 10, 'c_out': 1, 'name': 'CIN10_COUT1', 'focus': 'horizon'},
    {'c_in': 10, 'c_out': 3, 'name': 'CIN10_COUT3', 'focus': 'horizon'},
    {'c_in': 10, 'c_out': 5, 'name': 'CIN10_COUT5', 'focus': 'horizon'},
    {'c_in': 10, 'c_out': 7, 'name': 'CIN10_COUT7', 'focus': 'horizon'},
    # Vary C_IN with fixed C_OUT=3 (input context analysis)
    {'c_in': 7, 'c_out': 3, 'name': 'CIN7_COUT3', 'focus': 'context'},
    {'c_in': 14, 'c_out': 3, 'name': 'CIN14_COUT3', 'focus': 'context'},
]

# Note: CIN10_COUT3 appears in both analyses (baseline configuration)

# =============================================================================
# HEATWAVE DEFINITION (Perkins-Kirkpatrick-Lewis)
# =============================================================================
# PKL Definition: 90th percentile of Tmax for ≥3 consecutive days
# Reference: Perkins & Alexander (2013), Perkins-Kirkpatrick & Lewis (2020)
#
# Note: MAM (Sahel-specific) definition was considered but excluded due to
# insufficient sample size (92 events vs 553 for PKL). The MAM definition
# uses the 90th percentile of March-April-May Tmax only, resulting in higher
# thresholds (39.5-42.7°C) that are rarely exceeded.

HW_PERCENTILE = 90      # 90th percentile threshold
HW_MIN_DURATION = 3     # Minimum 3 consecutive days

# Analysis months (when heatwaves can occur)
# In Sahel, heatwaves can occur year-round but peak in Feb-May
ANALYSIS_MONTHS = list(range(1, 13))  # All year

# =============================================================================
# EXPERIMENTAL CONFIGURATIONS
# =============================================================================
# Primary experiments for thesis (subset of ablation grid)
# Selected to answer two key research questions:
#   Q1: How does forecast skill degrade with lead time?
#   Q2: How much historical context does DI-GNN need?

EXPERIMENTS = [
    # Q1: Forecast horizon analysis (C_IN=10 fixed)
    {'c_in': 10, 'c_out': 1, 'name': 'CIN10_COUT1', 'description': '1-day ahead (baseline)'},
    {'c_in': 10, 'c_out': 3, 'name': 'CIN10_COUT3', 'description': '3-day ahead (short-range)'},
    {'c_in': 10, 'c_out': 5, 'name': 'CIN10_COUT5', 'description': '5-day ahead (medium-range)'},
    {'c_in': 10, 'c_out': 7, 'name': 'CIN10_COUT7', 'description': '7-day ahead (extended)'},
    # Q2: Input context analysis (C_OUT=3 fixed)
    {'c_in': 7, 'c_out': 3, 'name': 'CIN7_COUT3', 'description': '7-day input window'},
    {'c_in': 14, 'c_out': 3, 'name': 'CIN14_COUT3', 'description': '14-day input window'},
]

# =============================================================================
# DATA SPLIT
# =============================================================================
TRAIN_YEARS = list(range(2010, 2023))  # 2010-2022 for training (13 years)
VAL_YEARS = [2023, 2024]               # 2023-2024 for validation (2 years)

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
HIDDEN_DIM = 64
NUM_LAYERS = 2
NUM_HEADS = 4
DROPOUT = 0.3

# =============================================================================
# TRAINING
# =============================================================================
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5
PATIENCE = 20  # Early stopping patience

# =============================================================================
# FEATURES
# =============================================================================
# Base meteorological features
BASE_FEATURES = [
    'MAX',       # Maximum temperature (Tmax)
    'MIN',       # Minimum temperature (Tmin)
    'DEWP',      # Dew point temperature
    'RH',        # Relative humidity
    'WDSP',      # Wind speed
    'PRCP',      # Precipitation
    'SLP',       # Sea level pressure
]

# Temporal features
TEMPORAL_FEATURES = [
    'DOY_sin',   # Day of year (sine encoding)
    'DOY_cos',   # Day of year (cosine encoding)
]

# Climate indices
CLIMATE_FEATURES = [
    'ONI',       # Oceanic Niño Index (ENSO teleconnection)
    'NAO',       # North Atlantic Oscillation index (West African teleconnection)
]

# Derived features
DERIVED_FEATURES = [
    'Tthresh',   # 90th percentile threshold
]

# GPD features for DI-GNN (Extreme Value Theory)
GPD_FEATURES = [
    'GPD_xi',      # Shape parameter
    'GPD_sigma',   # Scale parameter
    'GPD_mu',      # Mean of exceedances
    'GPD_var',     # Variance of exceedances
    'GPD_q95',     # 95th percentile
]

# Static features (per station, not time-varying)
STATIC_FEATURES = [
    'LATITUDE',
    'LONGITUDE',
    'ELEVATION',
]

# =============================================================================
# STATIONS
# =============================================================================
# Retained stations after quality filtering (Dori excluded due to data gap)
STATIONS = [
    'BOBO DIOULASSO',
    'BOROMO',
    'DEDOUGOU',
    "FADA N'GOURMA",
    'GAOUA',
    'OUAGADOUGOU',
    'OUAHIGOUYA',
    'PO',
]

# =============================================================================
# MONTE CARLO DROPOUT
# =============================================================================
MC_SAMPLES   = 50    # Nombre de passages forward pour l'inférence MC
MC_THRESHOLD = 0.5   # Seuil de décision sur la probabilité moyenne

# =============================================================================
# RANDOM SEED
# =============================================================================
SEED = 42

# Autoregressive feature (past heatwave label — intentional design choice)
# beta_t = 1 if day t was a heatwave day. Including past heatwave labels in the
# input window gives the model a persistence signal. Analogous to AR(p) models.
# NOTE: beta is used in feature_cols inside create_sequences() in preprocessing.
AUTOREGRESSIVE_FEATURES = ['beta']

# =============================================================================
# EVALUATION METRICS
# =============================================================================
# Primary metrics for imbalanced classification
PRIMARY_METRICS = [
    'balanced_accuracy',
    'f1_score',
    'recall',      # Probability of Detection (POD)
    'precision',   # Success Rate
]

# Secondary metrics
SECONDARY_METRICS = [
    'auc_roc',
    'auc_pr',      # Average Precision
    'accuracy',    # Standard accuracy (for reference only)
]