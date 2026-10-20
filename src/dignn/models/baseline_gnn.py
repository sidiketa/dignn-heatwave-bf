"""
Baseline GNN Model for Heatwave Prediction
==========================================

Implementation based on Li et al. (2023) approach:
- Temporal encoding using LSTM/GRU
- Graph Attention Networks (GAT) for spatial relationships
- Multi-step prediction for C_out days

Reference:
Li, P., Wang, J., Wang, X., & Lee, K. (2023). Regional Heatwave Prediction
Using Graph Neural Network and Weather Station Data. Geophysical Research Letters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemporalEncoder(nn.Module):
    """
    Encode temporal sequences using GRU.
    
    Input: (batch, n_stations, c_in, n_features)
    Output: (batch, n_stations, hidden_dim)
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, n_stations, c_in, n_features)
        Returns:
            h: (batch, n_stations, hidden_dim)
        """
        batch_size, n_stations, c_in, n_features = x.shape
        
        # Reshape to process all stations together
        x = x.view(batch_size * n_stations, c_in, n_features)
        
        # GRU encoding
        _, h = self.gru(x)  # h: (num_layers, batch*n_stations, hidden_dim)
        
        # Take last layer hidden state
        h = h[-1]  # (batch*n_stations, hidden_dim)
        
        # Reshape back
        h = h.view(batch_size, n_stations, -1)
        
        return self.dropout(h)


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT).
    
    Implements attention mechanism to weigh neighbor contributions.
    """
    
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.alpha = alpha
        
        # Linear transformation
        self.W = nn.Linear(in_features, out_features, bias=False)
        
        # Attention mechanism
        self.a = nn.Parameter(torch.zeros(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a.view(2, out_features))
        
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h, adj):
        """
        Args:
            h: Node features (batch, n_nodes, in_features)
            adj: Adjacency matrix (n_nodes, n_nodes)
        Returns:
            h': Updated node features (batch, n_nodes, out_features)
        """
        batch_size, n_nodes, _ = h.shape
        
        # Linear transformation
        Wh = self.W(h)  # (batch, n_nodes, out_features)
        
        # Compute attention coefficients
        # For each pair (i, j), compute attention
        Wh_i = Wh.unsqueeze(2).repeat(1, 1, n_nodes, 1)  # (batch, n, n, out)
        Wh_j = Wh.unsqueeze(1).repeat(1, n_nodes, 1, 1)  # (batch, n, n, out)
        
        # Concatenate and compute attention logits
        a_input = torch.cat([Wh_i, Wh_j], dim=-1)  # (batch, n, n, 2*out)
        e = self.leaky_relu(torch.matmul(a_input, self.a).squeeze(-1))  # (batch, n, n)
        
        # Mask with adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)
        adj_expanded = adj.unsqueeze(0).expand(batch_size, -1, -1)
        attention = torch.where(adj_expanded > 0, e, zero_vec)
        
        # Softmax normalization
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        # Aggregate neighbor features
        h_prime = torch.bmm(attention, Wh)  # (batch, n_nodes, out_features)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class MultiHeadGAT(nn.Module):
    """
    Multi-head Graph Attention Layer.
    """
    
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1, concat=True):
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat
        
        if concat:
            assert out_features % num_heads == 0
            head_dim = out_features // num_heads
        else:
            head_dim = out_features
        
        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_features, head_dim, dropout=dropout, concat=concat)
            for _ in range(num_heads)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h, adj):
        """
        Args:
            h: (batch, n_nodes, in_features)
            adj: (n_nodes, n_nodes)
        Returns:
            h': (batch, n_nodes, out_features)
        """
        if self.concat:
            # Concatenate all heads
            h = torch.cat([head(h, adj) for head in self.heads], dim=-1)
        else:
            # Average all heads
            h = torch.mean(torch.stack([head(h, adj) for head in self.heads]), dim=0)
        
        return self.dropout(h)


class BaselineGNN(nn.Module):
    """
    Baseline GNN for Heatwave Prediction.
    
    Architecture:
    1. Temporal Encoder (GRU): Process C_in time steps
    2. GAT Layers: Graph attention for spatial relationships
    3. Output Layer: Predict C_out days of heatwave probability
    
    Args:
        n_features: Number of input features per time step
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
        
        # Store adjacency matrix
        self.adj = None
    
    def set_adjacency(self, adj):
        """Set adjacency matrix (call before forward pass)."""
        if isinstance(adj, np.ndarray):
            adj = torch.FloatTensor(adj)
        self.adj = adj
    
    def forward(self, x, edge_index=None, edge_weight=None):
        """
        Forward pass.
        
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
        elif self.adj is not None:
            adj = self.adj.to(x.device)
        else:
            raise ValueError("Provide edge_index or call set_adjacency() first.")
        
        # 1. Temporal encoding
        h = self.temporal_encoder(x)  # (batch, n_stations, hidden_dim)
        
        # 2. Graph attention layers
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


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    print("Testing BaselineGNN...")
    
    # Model parameters
    batch_size = 16
    n_stations = 8
    c_in = 10
    c_out = 3
    n_features = 20
    hidden_dim = 64
    
    # Create model
    model = BaselineGNN(
        n_features=n_features,
        hidden_dim=hidden_dim,
        n_stations=n_stations,
        c_out=c_out,
        num_gat_layers=2,
        num_heads=4,
        dropout=0.3
    )
    
    # Create dummy adjacency matrix
    adj = torch.rand(n_stations, n_stations)
    adj = (adj + adj.T) / 2  # Symmetric
    model.set_adjacency(adj)
    
    # Create dummy input
    x = torch.randn(batch_size, n_stations, c_in, n_features)
    
    # Forward pass
    out = model(x)
    proba = model.predict_proba(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Probability shape: {proba.shape}")
    print(f"  Parameters: {count_parameters(model):,}")
    print("  Test passed!")
