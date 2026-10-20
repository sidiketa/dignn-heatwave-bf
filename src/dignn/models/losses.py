"""
Loss Functions for Heatwave Prediction
======================================

Implements:
1. StandardLoss: Binary cross-entropy with optional class weights
2. GPDWeightedLoss: GPD-weighted F1 loss (DI-GNN paper)
3. FocalLoss: For handling class imbalance

The GPD-weighted loss is key to DI-GNN's improved performance on rare events.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardLoss(nn.Module):
    """
    Standard Binary Cross-Entropy Loss with optional class weighting.
    
    Args:
        pos_weight: Weight for positive class (heatwave days)
                   Higher values increase recall at cost of precision
    """
    
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, logits, targets):
        """
        Compute BCE loss.
        
        Args:
            logits: Model outputs before sigmoid (batch, n_stations, c_out)
            targets: Binary labels (batch, n_stations, c_out)
        
        Returns:
            loss: Scalar loss value
        """
        if self.pos_weight is not None:
            pos_weight = torch.tensor([self.pos_weight], device=logits.device)
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight
            )
        else:
            loss = F.binary_cross_entropy_with_logits(logits, targets)
        
        return loss


class GPDWeightedLoss(nn.Module):
    """
    GPD-Weighted F1 Loss for Rare Event Detection.
    
    From DI-GNN paper (Chishtie et al. 2024):
    
    Loss = 1 - F1_weighted
    
    where F1_weighted uses GPD-derived weights to prioritize
    correct classification of rare and extreme events.
    
    The weights are: w_i = 1 + |xi_i| + sigma_i / max(sigma)
    
    Args:
        beta: F-beta score parameter
              beta > 1 prioritizes recall (catching heatwaves)
              beta < 1 prioritizes precision (avoiding false alarms)
              beta = 1 is standard F1 (balanced)
        smooth: Smoothing factor to avoid division by zero
    """
    
    def __init__(self, beta=1.0, smooth=1e-7):
        super().__init__()
        self.beta = beta
        self.beta_sq = beta ** 2
        self.smooth = smooth
    
    def forward(self, logits, targets, gpd_weights=None):
        """
        Compute GPD-weighted F1 loss.
        
        Args:
            logits: Model outputs before sigmoid (batch, n_stations, c_out)
            targets: Binary labels (batch, n_stations, c_out)
            gpd_weights: Station weights (batch, n_stations, 1) or None
        
        Returns:
            loss: Scalar loss value (1 - weighted_F1)
        """
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # Default weights if not provided
        if gpd_weights is None:
            gpd_weights = torch.ones_like(targets)
        
        # Expand weights to match targets shape if needed
        if gpd_weights.dim() == 3 and gpd_weights.shape[-1] == 1:
            gpd_weights = gpd_weights.expand_as(targets)
        
        # Weighted true positives, false positives, false negatives
        # TP: predicted positive AND actually positive
        tp_weighted = (gpd_weights * probs * targets).sum()
        
        # FP: predicted positive AND actually negative
        fp_weighted = (gpd_weights * probs * (1 - targets)).sum()
        
        # FN: predicted negative AND actually positive
        fn_weighted = (gpd_weights * (1 - probs) * targets).sum()
        
        # Weighted precision and recall
        precision = tp_weighted / (tp_weighted + fp_weighted + self.smooth)
        recall = tp_weighted / (tp_weighted + fn_weighted + self.smooth)
        
        # Weighted F-beta score
        f_beta = (1 + self.beta_sq) * precision * recall / (
            self.beta_sq * precision + recall + self.smooth
        )
        
        # Loss is 1 - F_beta
        loss = 1 - f_beta
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    From "Focal Loss for Dense Object Detection" (Lin et al. 2017):
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Reduces loss for well-classified examples, focusing on hard cases.
    
    Args:
        alpha: Weighting factor for positive class (0.25 default)
        gamma: Focusing parameter (2.0 default)
               Higher gamma = more focus on hard examples
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        """
        Compute focal loss.
        
        Args:
            logits: Model outputs before sigmoid
            targets: Binary labels
        
        Returns:
            loss: Scalar focal loss
        """
        probs = torch.sigmoid(logits)
        
        # Binary cross-entropy
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal weight: (1 - p_t)^gamma
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal loss
        loss = alpha_t * focal_weight * bce
        
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss: BCE + GPD-weighted F1.
    
    Allows balancing between standard classification loss
    and the F1-based loss that prioritizes rare events.
    
    Args:
        bce_weight: Weight for BCE loss component
        f1_weight: Weight for GPD-weighted F1 loss component
        beta: F-beta parameter for F1 loss
        pos_weight: Positive class weight for BCE
    """
    
    def __init__(self, bce_weight=0.5, f1_weight=0.5, beta=1.0, pos_weight=None):
        super().__init__()
        self.bce_weight = bce_weight
        self.f1_weight = f1_weight
        self.bce_loss = StandardLoss(pos_weight=pos_weight)
        self.f1_loss = GPDWeightedLoss(beta=beta)
    
    def forward(self, logits, targets, gpd_weights=None):
        """
        Compute combined loss.
        
        Args:
            logits: Model outputs
            targets: Binary labels
            gpd_weights: GPD weights for F1 loss
        
        Returns:
            loss: Weighted sum of BCE and F1 losses
        """
        bce = self.bce_loss(logits, targets)
        f1 = self.f1_loss(logits, targets, gpd_weights)
        
        return self.bce_weight * bce + self.f1_weight * f1


def create_loss(loss_type, **kwargs):
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: 'bce', 'gpd_weighted', 'focal', or 'combined'
        **kwargs: Loss-specific arguments
    
    Returns:
        loss_fn: Loss function instance
    """
    if loss_type == 'bce':
        return StandardLoss(pos_weight=kwargs.get('pos_weight', None))
    elif loss_type == 'gpd_weighted':
        return GPDWeightedLoss(beta=kwargs.get('beta', 1.0))
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=kwargs.get('alpha', 0.25),
            gamma=kwargs.get('gamma', 2.0)
        )
    elif loss_type == 'combined':
        return CombinedLoss(
            bce_weight=kwargs.get('bce_weight', 0.5),
            f1_weight=kwargs.get('f1_weight', 0.5),
            beta=kwargs.get('beta', 1.0),
            pos_weight=kwargs.get('pos_weight', None)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    batch_size = 16
    n_stations = 8
    c_out = 3
    
    # Create dummy data
    logits = torch.randn(batch_size, n_stations, c_out)
    targets = torch.randint(0, 2, (batch_size, n_stations, c_out)).float()
    gpd_weights = torch.rand(batch_size, n_stations, 1) + 1.0
    
    # Test StandardLoss
    bce_loss = StandardLoss(pos_weight=2.0)
    loss1 = bce_loss(logits, targets)
    print(f"  BCE Loss: {loss1.item():.4f}")
    
    # Test GPDWeightedLoss
    gpd_loss = GPDWeightedLoss(beta=1.0)
    loss2 = gpd_loss(logits, targets, gpd_weights)
    print(f"  GPD-Weighted F1 Loss: {loss2.item():.4f}")
    
    # Test FocalLoss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss3 = focal_loss(logits, targets)
    print(f"  Focal Loss: {loss3.item():.4f}")
    
    # Test CombinedLoss
    combined_loss = CombinedLoss(bce_weight=0.5, f1_weight=0.5)
    loss4 = combined_loss(logits, targets, gpd_weights)
    print(f"  Combined Loss: {loss4.item():.4f}")
    
    # Test factory
    print("\nTesting factory function...")
    for loss_type in ['bce', 'gpd_weighted', 'focal', 'combined']:
        loss_fn = create_loss(loss_type)
        print(f"  Created {loss_type}: {loss_fn.__class__.__name__}")
    
    print("\nAll tests passed!")
