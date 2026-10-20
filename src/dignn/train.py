"""
Training Script V2 for Heatwave Prediction Models
==================================================

Usage:
    python -m dignn.train --model baseline --definition 1 --loss standard
    python -m dignn.train --model dignn --definition 1 --loss combined
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


from .config import *
from .models.baseline_gnn import BaselineGNN
from .models.di_gnn import DIGNN
from .models.losses import GPDWeightedLoss, FocalLoss, CombinedLoss


def load_data(c_in=C_IN, c_out=C_OUT, definition=1):
    """Load preprocessed data."""
    print("\n1. Loading data...")

    # Fichier séquences nommé par (c_in, c_out) — convention ablation
    seq_file = DATA_DIR / f"sequences_cin{c_in}_cout{c_out}_v2.npz"
    if not seq_file.exists():
        # Compatibilité ascendante avec l'ancien nommage
        seq_file = DATA_DIR / f"sequences_def{definition}_v2.npz"
        if not seq_file.exists():
            seq_file = DATA_DIR / "sequences_def1_v2.npz"
        print(f"   WARNING: fichier cin{c_in}_cout{c_out} introuvable, "
              f"utilisation de {seq_file.name}")
    else:
        print(f"   Séquences : {seq_file.name}")

    data = np.load(seq_file, allow_pickle=True)
    X = data['X']
    y = data['y']
    years = data['years']
    features = list(data['features'])

    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Features: {len(features)}")

    # Load adjacency matrices
    adj_file = DATA_DIR / "adjacency_v2.npz"
    adj_data = np.load(adj_file, allow_pickle=True)
    adj_pearson = adj_data['pearson']
    adj_gpd = adj_data['gpd_weighted']
    stations = list(adj_data['stations'])

    print(f"   Stations: {len(stations)}")

    # Split chronologique calé sur VAL_YEARS (config.py) — pas de fuite temporelle
    unique_years = sorted(np.unique(years))
    val_years   = [y for y in unique_years if y in VAL_YEARS]
    train_years = [y for y in unique_years if y not in VAL_YEARS]

    if not val_years:
        print("   WARNING: aucune année de validation dans les données. "
              "Vérifier VAL_YEARS dans config.py.")
        val_years   = unique_years[-2:]
        train_years = unique_years[:-2]

    print(f"   Train years: {train_years}")
    print(f"   Val years: {val_years}")
    
    train_mask = np.isin(years, train_years)
    val_mask = np.isin(years, val_years)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    
    # Handle case with no validation data
    if len(X_val) == 0:
        print("   WARNING: No validation data! Using last 20% of data.")
        n_val = max(1, int(len(X) * 0.2))
        X_train, y_train = X[:-n_val], y[:-n_val]
        X_val, y_val = X[-n_val:], y[-n_val:]
    
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Val: {len(X_val):,} samples")
    print(f"   Train HW%: {y_train.sum()/y_train.size*100:.2f}%")
    print(f"   Val HW%: {y_val.sum()/y_val.size*100:.2f}%")
    
    return (X_train, y_train, X_val, y_val, 
            adj_pearson, adj_gpd, stations, features)


def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE):
    """Create PyTorch dataloaders."""
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def build_model(model_type, n_stations, n_features, c_out, adj_matrix, device):
    """Build model."""
    print(f"\n2. Building {model_type} model...")
    
    # Convert adjacency to edge_index format
    adj = torch.FloatTensor(adj_matrix)
    edge_index = (adj > 0).nonzero().t().contiguous()
    edge_weight = adj[adj > 0]
    
    if model_type == 'baseline':
        model = BaselineGNN(
            n_features=n_features,
            hidden_dim=HIDDEN_DIM,
            n_stations=n_stations,
            c_out=c_out,
            num_gat_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            dropout=DROPOUT
        )
    else:  # dignn
        model = DIGNN(
            n_features=n_features,
            hidden_dim=HIDDEN_DIM,
            n_stations=n_stations,
            c_out=c_out,
            num_gat_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            dropout=DROPOUT
        )
    
    model = model.to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {n_params:,}")
    
    return model, edge_index, edge_weight


def get_loss_function(loss_type, pos_weight, device):
    """Get loss function."""
    
    if loss_type == 'standard':
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    elif loss_type == 'focal':
        return FocalLoss(alpha=0.75, gamma=2.0)
    elif loss_type == 'gpd':
        return GPDWeightedLoss(beta=2.0)
    elif loss_type == 'combined':
        return CombinedLoss(bce_weight=0.5, f1_weight=0.5, beta=2.0, 
                           pos_weight=pos_weight)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_epoch(model, train_loader, criterion, optimizer, edge_index, edge_weight, device):
    """Train one epoch."""
    model.train()
    total_loss = 0
    all_preds, all_labels, all_probas = [], [], []
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(X_batch, edge_index, edge_weight)
        
        # Compute loss
        loss = criterion(logits, y_batch)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Store predictions and probabilities
        probas = torch.sigmoid(logits)
        preds  = (probas > 0.5).float()
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())
        all_probas.append(probas.detach().cpu().numpy())

    # Compute metrics
    all_preds  = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    all_probas = np.concatenate(all_probas).flatten()

    # AUC-ROC : nécessite au moins une classe positive
    try:
        auc_roc = roc_auc_score(all_labels, all_probas)                   if all_labels.sum() > 0 else float('nan')
    except ValueError:
        auc_roc = float('nan')

    metrics = {
        'loss':         total_loss / len(train_loader),
        'balanced_acc': balanced_accuracy_score(all_labels, all_preds),
        'f1':           f1_score(all_labels, all_preds, zero_division=0),
        'precision':    precision_score(all_labels, all_preds, zero_division=0),
        'recall':       recall_score(all_labels, all_preds, zero_division=0),
        'auc_roc':      auc_roc,
    }

    return metrics


def validate(model, val_loader, criterion, edge_index, edge_weight, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probas = [], [], []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            logits = model(X_batch, edge_index, edge_weight)
            loss = criterion(logits, y_batch)
            
            total_loss += loss.item()
            
            probas = torch.sigmoid(logits)
            preds  = (probas > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            all_probas.append(probas.cpu().numpy())

    all_preds  = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    all_probas = np.concatenate(all_probas).flatten()

    try:
        auc_roc = roc_auc_score(all_labels, all_probas)                   if all_labels.sum() > 0 else float('nan')
    except ValueError:
        auc_roc = float('nan')

    metrics = {
        'loss':         total_loss / len(val_loader),
        'balanced_acc': balanced_accuracy_score(all_labels, all_preds),
        'f1':           f1_score(all_labels, all_preds, zero_division=0),
        'precision':    precision_score(all_labels, all_preds, zero_division=0),
        'recall':       recall_score(all_labels, all_preds, zero_division=0),
        'auc_roc':      auc_roc,
    }

    return metrics


def set_seed(seed):
    """Applique la graine à toutes les sources d'aléatoire pour la reproductibilité."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Déterminisme (léger coût de vitesse, gain de reproductibilité)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    """Main training function."""
    print("=" * 60)
    print(f"TRAINING {args.model.upper()} MODEL")
    print("=" * 60)

    # Reproductibilité : appliquer la graine AVANT toute opération aléatoire
    set_seed(args.seed)
    print(f"\nGraine fixée : {args.seed}")

    # Nom de sortie déterministe : {model}_cin{c_in}_cout{c_out}_seed{seed}
    # Re-lancer la même config + graine ECRASE proprement (pas d'accumulation).
    model_name = f"{args.model}_cin{args.c_in}_cout{args.c_out}_seed{args.seed}"
    model_path = OUTPUT_DIR / f"{model_name}.pth"
    if model_path.exists():
        print(f"  (écrasement du modèle existant : {model_path.name})")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    (X_train, y_train, X_val, y_val, 
     adj_pearson, adj_gpd, stations, features) = load_data(
         c_in=args.c_in, c_out=args.c_out, definition=args.definition
    )
    
    # Get dimensions
    n_stations = X_train.shape[1]
    n_features = X_train.shape[3]
    c_out = y_train.shape[2] if len(y_train.shape) > 2 else y_train.shape[1]
    
    # Select adjacency matrix
    adj_matrix = adj_gpd if args.model == 'dignn' else adj_pearson
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, args.batch_size
    )
    
    # Build model
    model, edge_index, edge_weight = build_model(
        args.model, n_stations, n_features, c_out, adj_matrix, device
    )
    
    # Compute class weight
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"\n   Positive weight: {pos_weight:.2f}")
    
    # Loss function
    criterion = get_loss_function(args.loss, pos_weight, device)
    print(f"   Loss: {args.loss}")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_f1': [], 'train_balanced_acc': [],
        'train_precision': [], 'train_recall': [], 'train_auc_roc': [],
        'val_loss': [], 'val_f1': [], 'val_balanced_acc': [],
        'val_precision': [], 'val_recall': [], 'val_auc_roc': []
    }
    
    best_f1 = 0
    best_epoch = 0
    patience_counter = 0
    
    print("\n3. Training...")
    print("-" * 60)
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, 
            edge_index, edge_weight, device
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, 
            edge_index, edge_weight, device
        )
        
        # Update scheduler
        scheduler.step(val_metrics['f1'])
        
        # Store history
        for key in ['loss', 'f1', 'balanced_acc', 'precision', 'recall', 'auc_roc']:
            history[f'train_{key}'].append(train_metrics[key])
            history[f'val_{key}'].append(val_metrics[key])
        
        # Print progress
        auc_str = f"{val_metrics['auc_roc']:.4f}" if not (isinstance(val_metrics['auc_roc'], float) and val_metrics['auc_roc'] != val_metrics['auc_roc']) else "N/A"
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} F1: {train_metrics['f1']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} F1: {val_metrics['f1']:.4f} "
              f"BA: {val_metrics['balanced_acc']:.4f} AUC: {auc_str}")
        
        # Check for improvement
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model (nom déterministe défini en début de train())
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_f1': best_f1,
                'config': {
                    'model': args.model,
                    'definition': args.definition,
                    'n_features': n_features,
                    'n_stations': n_stations,
                    'c_in': args.c_in,
                    'c_out': c_out,
                    'seed': args.seed,
                    'hidden_dim': HIDDEN_DIM,
                    'num_layers': NUM_LAYERS,
                    'num_heads': NUM_HEADS,
                    'dropout': DROPOUT
                }
            }, model_path)
            
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Save final outputs
    print("\n4. Saving outputs...")
    
    # Save history
    history_path = OUTPUT_DIR / f"{model_name}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"   History: {history_path}")
    
    # Save config
    config = {
        'model': args.model,
        'definition': args.definition,
        'loss': args.loss,
        'n_features': n_features,
        'n_stations': n_stations,
        'c_out': c_out,
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS,
        'num_heads': NUM_HEADS,
        'dropout': DROPOUT,
        'epochs_trained': epoch + 1,
        'best_epoch': best_epoch,
        'best_f1': best_f1
    }
    config_path = OUTPUT_DIR / f"{model_name}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"   Config: {config_path}")
    
    print(f"   Model: {model_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nBest Results (Epoch {best_epoch}):")
    print(f"  F1 Score: {best_f1:.4f}")
    print(f"  Balanced Accuracy: {history['val_balanced_acc'][best_epoch-1]:.4f}")
    print(f"  Precision: {history['val_precision'][best_epoch-1]:.4f}")
    print(f"  Recall: {history['val_recall'][best_epoch-1]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train heatwave prediction models")
    parser.add_argument('--c_in', type=int, default=C_IN,
                        help=f"Fenêtre d'entrée en jours (défaut: {C_IN})")
    parser.add_argument('--c_out', type=int, default=C_OUT,
                        choices=C_OUT_OPTIONS,
                        help=f"Horizon de prévision en jours (défaut: {C_OUT})")
    parser.add_argument('--model', type=str, default='baseline', 
                        choices=['baseline', 'dignn'],
                        help="Model type")
    parser.add_argument('--definition', type=int, default=1, 
                        choices=[1, 2],
                        help="Heatwave definition (1=90th, 2=95th percentile)")
    parser.add_argument('--loss', type=str, default='standard',
                        choices=['standard', 'focal', 'gpd', 'combined'],
                        help="Loss function")
    parser.add_argument('--seed', type=int, default=SEED,
                        help=f"Graine aléatoire (défaut: {SEED}). "
                             f"Varier pour produire un ensemble multi-graines.")
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help="Batch size")
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help="Learning rate")
    
    args = parser.parse_args()
    train(args)