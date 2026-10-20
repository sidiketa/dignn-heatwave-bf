"""
Distribution-Informed GNN (DI-GNN) for Heatwave Prediction
==========================================================

Implementation based on Chishtie et al. (2024):
- Integrates Extreme Value Theory (EVT) into GNN architecture
- GPD parameters embedded in feature space
- GPD-weighted adjacency matrix
- GPD-weighted loss function for rare event detection

Key innovations:
1. GPD-derived features (xi, sigma, mu, var, q95) augment input
2. Adjacency reweighted by station extremity (heavy tails get more influence)
3. Custom loss function prioritizes rare heatwave events

Reference:
Chishtie, F.A., et al. (2024). Advancing Heatwave Forecasting via Distribution
Informed-Graph Neural Networks (DI-GNNs). arXiv:2411.13496v1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .baseline_gnn import TemporalEncoder, MultiHeadGAT, count_parameters


class DIGNN(nn.Module):
    """
    Distribution-Informed Graph Neural Network.
    
    Extends baseline GNN with:
    - GPD-weighted adjacency matrix
    - Station-specific GPD weights for loss computation
    
    Architecture:
    1. Temporal Encoder (GRU): Process C_in time steps
    2. GAT Layers with GPD-weighted adjacency
    3. Output Layer: Predict C_out days of heatwave probability
    
    The GPD parameters should be included in the input features
    (done during preprocessing).
    
    Args:
        n_features: Number of input features (including GPD features)
        hidden_dim: Hidden dimension for GRU and GAT
        n_stations: Number of weather stations (nodes)
        c_out: Number of days to predict
        num_gat_layers: Number of GAT layers
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        n_features,
        hidden_dim=64,
        n_stations=8,
        c_out=3,
        num_gat_layers=2,
        num_heads=4,
        dropout=0.3
    ):
        super().__init__()
        
        self.n_stations = n_stations
        self.c_out = c_out
        self.hidden_dim = hidden_dim
        
        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            num_layers=1,
            dropout=dropout
        )
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            self.gat_layers.append(
                MultiHeadGAT(
                    in_features=in_dim,
                    out_features=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
            )
        
        # Output layer: predict c_out days
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, c_out)
        )
        
        # GPD-weighted adjacency matrix
        self.adj_gpd = None
        
        # GPD weights for loss computation (per station)
        self.gpd_weights = None
    
    def set_adjacency(self, adj_gpd):
        """
        Set GPD-weighted adjacency matrix.
        
        Args:
            adj_gpd: GPD-weighted adjacency matrix (n_stations, n_stations)
        """
        if isinstance(adj_gpd, np.ndarray):
            adj_gpd = torch.FloatTensor(adj_gpd)
        self.adj_gpd = adj_gpd
    
    def set_gpd_weights(self, gpd_weights):
        """
        Set GPD weights for loss computation.
        
        Weights are computed as: w_i = 1 + |xi_i| + sigma_i / max(sigma)
        
        Args:
            gpd_weights: Array of weights per station (n_stations,)
        """
        if isinstance(gpd_weights, np.ndarray):
            gpd_weights = torch.FloatTensor(gpd_weights)
        self.gpd_weights = gpd_weights
    
    def forward(self, x, edge_index=None, edge_weight=None):
        """
        Forward pass using GPD-weighted adjacency.
        
        Args:
            x: Input tensor (batch, n_stations, c_in, n_features)
            edge_index: Edge indices (2, num_edges) - optional
            edge_weight: Edge weights (num_edges,) - optional
        
        Returns:
            out: Predictions (batch, n_stations, c_out)
        """
        if edge_index is not None:
            # Reconstruct adjacency from edge_index
            n_nodes = x.shape[1]
            adj = torch.zeros(n_nodes, n_nodes, device=x.device)
            if edge_weight is not None:
                adj[edge_index[0], edge_index[1]] = edge_weight
            else:
                adj[edge_index[0], edge_index[1]] = 1.0
        elif self.adj_gpd is not None:
            adj = self.adj_gpd.to(x.device)
        else:
            raise ValueError("Provide edge_index or call set_adjacency() first.")
        
        # 1. Temporal encoding
        h = self.temporal_encoder(x)  # (batch, n_stations, hidden_dim)
        
        # 2. Graph attention layers with GPD-weighted adjacency
        for gat in self.gat_layers:
            h = gat(h, adj)  # (batch, n_stations, hidden_dim)
        
        # 3. Output prediction
        out = self.output_layer(h)  # (batch, n_stations, c_out)
        
        return out
    
    def predict_proba(self, x):
        """Get probability predictions."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def predict(self, x, threshold=0.5):
        """Get binary predictions."""
        proba = self.predict_proba(x)
        return (proba > threshold).float()
    

    def enable_dropout(self):
        """
        Active le dropout pendant l'inférence pour Monte Carlo Dropout.

        Appelée après model.eval() pour réactiver uniquement les couches
        Dropout, laissant BatchNorm et autres couches en mode évaluation.
        """
        for module in self.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

    def mc_predict(self, x, edge_index=None, edge_weight=None, n_samples=50):
        """
        Prédiction probabiliste par Monte Carlo Dropout.

        Effectue n_samples passages forward avec dropout actif, produisant
        une distribution de prédictions par station et par horizon.
        La moyenne donne la probabilité estimée ; l'écart-type quantifie
        l'incertitude épistémique du modèle.

        Args:
            x           : Tenseur d'entrée (batch, n_stations, c_in, n_features)
            edge_index  : Indices des arêtes — optionnel
            edge_weight : Poids des arêtes — optionnel
            n_samples   : Nombre de passages forward (défaut : 50)

        Returns:
            mean_proba  : Probabilité moyenne MC (batch, n_stations, c_out)
            std_proba   : Écart-type / incertitude (batch, n_stations, c_out)
            all_probas  : Toutes les prédictions (n_samples, batch, n_stations, c_out)

        Exemple d'utilisation :
            model.eval()
            mean_p, std_p, _ = model.mc_predict(X_val, edge_index, edge_weight)
            preds = (mean_p > 0.5).float()
        """
        self.eval()             # Désactive BatchNorm etc.
        self.enable_dropout()   # Réactive uniquement le dropout

        all_probas = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x, edge_index, edge_weight)
                probas = torch.sigmoid(logits)
                all_probas.append(probas.unsqueeze(0))

        # Stack: (n_samples, batch, n_stations, c_out)
        all_probas  = torch.cat(all_probas, dim=0)
        mean_proba  = all_probas.mean(dim=0)
        std_proba   = all_probas.std(dim=0)

        return mean_proba, std_proba, all_probas

    def get_gpd_weights_for_batch(self, batch_size, device):
        """
        Get GPD weights expanded for batch computation.
        
        Returns weights of shape (batch, n_stations, 1) for broadcasting
        with predictions of shape (batch, n_stations, c_out).
        """
        if self.gpd_weights is None:
            # Default to uniform weights
            return torch.ones(batch_size, self.n_stations, 1, device=device)
        
        weights = self.gpd_weights.to(device)
        # Expand: (n_stations,) -> (1, n_stations, 1) -> (batch, n_stations, 1)
        weights = weights.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
        return weights


class DIGNNWithAttention(DIGNN):
    """
    DI-GNN with attention visualization support.
    
    Extends DIGNN to store and return attention weights for interpretability.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_weights = []
    
    def forward(self, x, return_attention=False):
        """
        Forward pass with optional attention weight extraction.
        
        Args:
            x: Input tensor
            return_attention: If True, return attention weights
        
        Returns:
            out: Predictions
            attention: List of attention weight matrices (if return_attention=True)
        """
        self.attention_weights = []
        out = super().forward(x)
        
        if return_attention:
            return out, self.attention_weights
        return out


def create_model(model_type, n_features, n_stations, c_out, hidden_dim=64,
                 num_layers=2, num_heads=4, dropout=0.3):
    """
    Factory function to create GNN models.
    
    Args:
        model_type: 'baseline' or 'dignn'
        n_features: Number of input features
        n_stations: Number of stations
        c_out: Prediction horizon
        hidden_dim: Hidden dimension
        num_layers: Number of GAT layers
        num_heads: Number of attention heads
        dropout: Dropout rate
    
    Returns:
        model: GNN model instance
    """
    if model_type.lower() == 'baseline':
        from .baseline_gnn import BaselineGNN
        return BaselineGNN(
            n_features=n_features,
            hidden_dim=hidden_dim,
            n_stations=n_stations,
            c_out=c_out,
            num_gat_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
    elif model_type.lower() == 'dignn':
        return DIGNN(
            n_features=n_features,
            hidden_dim=hidden_dim,
            n_stations=n_stations,
            c_out=c_out,
            num_gat_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'baseline' or 'dignn'.")


if __name__ == "__main__":
    # Test model
    print("Testing DIGNN...")
    
    # Model parameters
    batch_size = 16
    n_stations = 8
    c_in = 10
    c_out = 3
    n_features = 20  # Including GPD features
    hidden_dim = 64
    
    # Create model
    model = DIGNN(
        n_features=n_features,
        hidden_dim=hidden_dim,
        n_stations=n_stations,
        c_out=c_out,
        num_gat_layers=2,
        num_heads=4,
        dropout=0.3
    )
    
    # Create dummy GPD-weighted adjacency matrix
    adj = torch.rand(n_stations, n_stations)
    adj = (adj + adj.T) / 2  # Symmetric
    model.set_adjacency(adj)
    
    # Create dummy GPD weights
    gpd_weights = torch.rand(n_stations) + 1.0  # Range [1, 2]
    model.set_gpd_weights(gpd_weights)
    
    # Create dummy input
    x = torch.randn(batch_size, n_stations, c_in, n_features)
    
    # Forward pass
    out = model(x)
    proba = model.predict_proba(x)
    weights = model.get_gpd_weights_for_batch(batch_size, x.device)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Probability shape: {proba.shape}")
    print(f"  GPD weights shape: {weights.shape}")
    print(f"  Parameters: {count_parameters(model):,}")
    print("  Test passed!")
    
    # Test factory function
    print("\nTesting factory function...")
    baseline = create_model('baseline', n_features, n_stations, c_out)
    dignn = create_model('dignn', n_features, n_stations, c_out)
    print(f"  Baseline params: {count_parameters(baseline):,}")
    print(f"  DIGNN params: {count_parameters(dignn):,}")
    print("  Factory test passed!")
