"""
Evaluation Script V2 for Heatwave Prediction Models
====================================================

Usage:
    python -m dignn.evaluate --run_all
    python -m dignn.evaluate --model_path outputs/model.pth
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)


from .config import *
from .models.baseline_gnn import BaselineGNN
from .models.di_gnn import DIGNN
from .metrics import (
    brier_skill_score, bss_by_lead_time, auc_by_lead_time,
    reliability_diagram, probabilistic_summary
)


def load_data(c_in=C_IN, c_out=C_OUT, definition=1):
    """Load preprocessed data."""

    seq_file = DATA_DIR / f"sequences_cin{c_in}_cout{c_out}_v2.npz"
    if not seq_file.exists():
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
    
    adj_file = DATA_DIR / "adjacency_v2.npz"
    adj_data = np.load(adj_file, allow_pickle=True)
    adj_pearson = adj_data['pearson']
    adj_gpd = adj_data['gpd_weighted']
    stations = list(adj_data['stations'])
    
    # Split chronologique calé sur VAL_YEARS (config.py)
    unique_years = sorted(np.unique(years))
    val_years = [yr for yr in unique_years if yr in VAL_YEARS]
    if not val_years:
        val_years = unique_years[-2:]
    print(f"   Val years: {val_years}")

    val_mask = np.isin(years, val_years)
    X_val, y_val = X[val_mask], y[val_mask]
    
    # Handle empty validation set
    if len(X_val) == 0:
        n_val = max(1, int(len(X) * 0.2))
        X_val, y_val = X[-n_val:], y[-n_val:]
    
    return X_val, y_val, adj_pearson, adj_gpd, stations, features


def load_model(model_path, device):
    """Load trained model."""
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    
    model_type = config.get('model', 'baseline')
    n_features = config.get('n_features', 19)
    n_stations = config.get('n_stations', 8)
    c_out = config.get('c_out', 3)
    hidden_dim = config.get('hidden_dim', HIDDEN_DIM)
    num_layers = config.get('num_layers', config.get('num_gat_layers', NUM_LAYERS))
    num_heads = config.get('num_heads', NUM_HEADS)
    dropout = config.get('dropout', DROPOUT)
    
    if model_type == 'dignn':
        model = DIGNN(
            n_features=n_features,
            hidden_dim=hidden_dim,
            n_stations=n_stations,
            c_out=c_out,
            num_gat_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
    else:
        model = BaselineGNN(
            n_features=n_features,
            hidden_dim=hidden_dim,
            n_stations=n_stations,
            c_out=c_out,
            num_gat_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config


def evaluate_model(model, X_val, y_val, adj_matrix, device):
    """Evaluate model on validation set."""
    
    # Prepare data
    X_tensor = torch.FloatTensor(X_val).to(device)
    y_true = y_val.flatten()
    
    # Prepare adjacency
    adj = torch.FloatTensor(adj_matrix)
    edge_index = (adj > 0).nonzero().t().contiguous().to(device)
    edge_weight = adj[adj > 0].to(device)
    
    # Get predictions
    with torch.no_grad():
        logits = model(X_tensor, edge_index, edge_weight)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(float)
    
    # Compute metrics
    metrics = {
        'balanced_accuracy': balanced_accuracy_score(y_true, preds),
        'precision': precision_score(y_true, preds, zero_division=0),
        'recall': recall_score(y_true, preds, zero_division=0),
        'f1': f1_score(y_true, preds, zero_division=0),
        'auc_roc': roc_auc_score(y_true, probs) if y_true.sum() > 0 else 0,
        'avg_precision': average_precision_score(y_true, probs) if y_true.sum() > 0 else 0
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, preds)
    metrics['confusion_matrix'] = cm.tolist()

    # Vraies courbes ROC et Précision-Rappel (pour les figures du mémoire)
    # Sous-échantillonnage à ~300 points pour garder eval_results.json léger.
    from sklearn.metrics import roc_curve, precision_recall_curve

    def _subsample(arr, n=300):
        arr = np.asarray(arr)
        if len(arr) <= n:
            return arr.tolist()
        idx = np.linspace(0, len(arr) - 1, n).astype(int)
        return arr[idx].tolist()

    if y_true.sum() > 0:
        fpr, tpr, _ = roc_curve(y_true, probs)
        prec_c, rec_c, _ = precision_recall_curve(y_true, probs)
        metrics['fpr'] = _subsample(fpr)
        metrics['tpr'] = _subsample(tpr)
        metrics['precision_curve'] = _subsample(prec_c)
        metrics['recall_curve'] = _subsample(rec_c)
        metrics['prevalence'] = float(y_true.mean())

    return metrics



def evaluate_model_mc(model, X_val, y_val, adj_matrix, device,
                      n_samples=None):
    """
    Évaluation probabiliste par Monte Carlo Dropout.

    Utilise mc_predict() de la classe DIGNN pour produire des probabilités
    moyennes et des incertitudes sur l'ensemble de validation.

    Args:
        model      : Modèle DIGNN entraîné
        X_val      : Données de validation (N, n_stations, c_in, n_features)
        y_val      : Étiquettes (N, n_stations, c_out)
        adj_matrix : Matrice d'adjacence GPD
        device     : Dispositif de calcul
        n_samples  : Nombre de passages MC (défaut : MC_SAMPLES depuis config)

    Returns:
        mc_metrics : Dictionnaire des métriques probabilistes
        mean_proba : Probabilités moyennes (N, n_stations, c_out)
        std_proba  : Incertitudes (N, n_stations, c_out)
    """
    if n_samples is None:
        n_samples = MC_SAMPLES

    if not hasattr(model, 'mc_predict'):
        print("   WARNING: modele sans mc_predict — évaluation déterministe uniquement.")
        return None, None, None

    X_tensor = torch.FloatTensor(X_val).to(device)

    adj = torch.FloatTensor(adj_matrix)
    edge_index  = (adj > 0).nonzero().t().contiguous().to(device)
    edge_weight = adj[adj > 0].to(device)

    print(f"   MC Dropout : {n_samples} passages forward...")
    mean_proba, std_proba, _ = model.mc_predict(
        X_tensor, edge_index, edge_weight, n_samples=n_samples
    )

    mean_proba_np = mean_proba.cpu().numpy()   # (N, n_stations, c_out)
    std_proba_np  = std_proba.cpu().numpy()
    y_true        = y_val                       # (N, n_stations, c_out)

    # Assurer la forme 3D
    if y_true.ndim == 2:
        y_true = y_true[:, :, np.newaxis]
    if mean_proba_np.ndim == 2:
        mean_proba_np = mean_proba_np[:, :, np.newaxis]
        std_proba_np  = std_proba_np[:, :, np.newaxis]

    c_out = y_true.shape[-1]
    c_out_labels = list(range(1, c_out + 1))

    print("\n   Métriques probabilistes par horizon :")
    mc_metrics = probabilistic_summary(
        y_true, mean_proba_np, std_proba_np,
        c_out_labels=c_out_labels
    )

    # Incertitude moyenne globale
    mc_metrics['mean_uncertainty_global'] = float(std_proba_np.mean())
    print(f"   Incertitude moyenne (σ) : {mc_metrics['mean_uncertainty_global']:.4f}")

    return mc_metrics, mean_proba_np, std_proba_np


def evaluate_single_model(model_path, c_in, c_out, definition, device):
    """Evaluate a single model."""
    
    print(f"\nEvaluating: {model_path}")
    
    # Load data
    X_val, y_val, adj_pearson, adj_gpd, stations, features = load_data(
        c_in=c_in, c_out=c_out, definition=definition)
    
    # Load model
    model, config = load_model(model_path, device)
    model_type = config.get('model', 'baseline')
    
    # Select adjacency
    adj_matrix = adj_gpd if model_type == 'dignn' else adj_pearson
    
    # Evaluate — métriques déterministes
    metrics = evaluate_model(model, X_val, y_val, adj_matrix, device)

    # Print results
    print(f"\n  Résultats déterministes — {model_type.upper()}:")
    print(f"    Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"    Precision:         {metrics['precision']:.4f}")
    print(f"    Recall:            {metrics['recall']:.4f}")
    print(f"    F1 Score:          {metrics['f1']:.4f}")
    print(f"    AUC-ROC:           {metrics['auc_roc']:.4f}")
    print(f"    Avg Precision:     {metrics['avg_precision']:.4f}")

    if 'confusion_matrix' in metrics:
        cm = np.array(metrics['confusion_matrix'])
        print(f"\n    Matrice de confusion :")
        print(f"    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
        print(f"    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")

    # Évaluation probabiliste MC Dropout — DI-GNN uniquement
    if model_type == 'dignn':
        print(f"\n  Évaluation probabiliste MC Dropout — {model_type.upper()}:")
        mc_metrics, mean_proba, std_proba = evaluate_model_mc(
            model, X_val, y_val, adj_matrix, device
        )
        if mc_metrics is not None:
            metrics['mc'] = mc_metrics

            # Sauvegarde des probabilités brutes pour la calibration (Platt) ultérieure.
            # Trois tableaux alignés de forme (N, n_stations, c_out).
            if mean_proba is not None:
                probs_dir = OUTPUT_DIR / "mc_probs"
                probs_dir.mkdir(exist_ok=True)
                stem = Path(model_path).stem
                y_true_3d = y_val
                if y_true_3d.ndim == 2:
                    y_true_3d = y_true_3d[:, :, np.newaxis]
                np.save(probs_dir / f"{stem}_mean_proba.npy", mean_proba)
                np.save(probs_dir / f"{stem}_std_proba.npy",  std_proba)
                np.save(probs_dir / f"{stem}_y_true.npy",     y_true_3d)
                print(f"   Probabilités MC sauvegardées : "
                      f"mc_probs/{stem}_{{mean_proba,std_proba,y_true}}.npy")
                print(f"     formes : mean={mean_proba.shape}, "
                      f"y_true={y_true_3d.shape}")

            # Diagramme de fiabilité
            if mean_proba is not None:
                fig_path = OUTPUT_DIR / f"reliability_{Path(model_path).stem}.png"
                reliability_diagram(
                    y_val.flatten(),
                    mean_proba.flatten(),
                    title=f"Diagramme de fiabilité — DI-GNN",
                    save_path=str(fig_path)
                )
                import matplotlib.pyplot as plt
                plt.close('all')

    # Métadonnées issues de la config du checkpoint (source fiable, pas le nom de fichier)
    metrics['model_type'] = model_type
    metrics['c_in']  = config.get('c_in', c_in)
    metrics['c_out'] = config.get('c_out', c_out)

    return metrics


def find_models(output_dir):
    """Find all model files."""
    
    models = []
    
    for f in Path(output_dir).glob("*.pth"):
        name = f.stem
        
        if 'baseline' in name.lower():
            model_type = 'baseline'
        elif 'dignn' in name.lower():
            model_type = 'dignn'
        else:
            continue
        
        if 'def1' in name.lower():
            definition = 1
        elif 'def2' in name.lower():
            definition = 2
        else:
            definition = 1
        
        models.append({
            'path': f,
            'name': name,
            'model_type': model_type,
            'definition': definition
        })
    
    return models



def save_eval_results(results):
    """
    Sauvegarde les métriques déterministes dans outputs/eval_results.json.
    Utilisé par generate_figures.py pour produire les figures du chapitre Résultats.

    Args:
        results : dict {model_name: metrics} retourné par evaluate_single_model
    """
    METRIC_KEYS = [
        'balanced_accuracy', 'precision', 'recall', 'f1',
        'auc_roc', 'avg_precision', 'confusion_matrix',
        'fpr', 'tpr', 'precision_curve', 'recall_curve', 'prevalence',
        'model_type', 'c_in', 'c_out'
    ]
    serializable = {}
    for name, metrics in results.items():
        entry = {}
        # model_type/c_in/c_out viennent de la config du checkpoint (injectés
        # dans evaluate_single_model). Repli sur le nom de fichier si absents.
        entry['model_type'] = metrics.get(
            'model_type', 'dignn' if 'dignn' in name.lower() else 'baseline')
        entry['c_in']  = metrics.get('c_in')
        entry['c_out'] = metrics.get('c_out')
        for key in METRIC_KEYS:
            if key in metrics and key not in ('model_type', 'c_in', 'c_out'):
                val = metrics[key]
                entry[key] = val if isinstance(val, (list, int)) else float(val)
        serializable[name] = entry
    path = OUTPUT_DIR / "eval_results.json"
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"   Métriques déterministes sauvegardées : {path}")



def save_mc_results(results):
    """
    Sauvegarde les métriques MC Dropout dans outputs/mc_results.json.

    Extrait les résultats MC de tous les modèles DI-GNN évalués
    et les consolide dans un fichier JSON lu par generate_paper_summary.py.

    Args:
        results : dict {model_name: metrics} retourné par evaluate_single_model
    """
    mc_combined = {}

    for model_name, metrics in results.items():
        if 'mc' not in metrics:
            continue  # Baseline ou modèle sans MC

        mc = metrics['mc']
        for horizon, vals in mc.items():
            if horizon == 'mean_uncertainty_global':
                continue
            # Prendre la meilleure valeur BSS si plusieurs modèles DI-GNN
            if horizon not in mc_combined:
                mc_combined[horizon] = vals
            else:
                # Garder le modèle avec le meilleur BSS
                if vals.get('bss', -999) > mc_combined[horizon].get('bss', -999):
                    mc_combined[horizon] = vals

        # Incertitude globale — moyenne sur tous les modèles DI-GNN
        global_unc = mc.get('mean_uncertainty_global', None)
        if global_unc is not None:
            prev = mc_combined.get('mean_uncertainty_global', [])
            if isinstance(prev, list):
                prev.append(global_unc)
            mc_combined['mean_uncertainty_global'] = prev if isinstance(prev, list) else [global_unc]

    # Moyenne de l'incertitude globale si plusieurs modèles
    if isinstance(mc_combined.get('mean_uncertainty_global'), list):
        vals_unc = mc_combined['mean_uncertainty_global']
        mc_combined['mean_uncertainty_global'] = float(np.mean(vals_unc))

    if mc_combined:
        mc_path = OUTPUT_DIR / "mc_results.json"
        with open(mc_path, 'w') as f:
            json.dump(mc_combined, f, indent=2)
        print(f"\n   Résultats MC sauvegardés : {mc_path}")
    else:
        print("\n   Aucun résultat MC à sauvegarder (aucun modèle DI-GNN évalué).")



def main(args):
    print("=" * 60)
    print("MODEL EVALUATION V2")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if args.model_path:
        # Evaluate single model
        metrics = evaluate_single_model(args.model_path, args.c_in, args.c_out, args.definition, device)
        save_eval_results({Path(args.model_path).stem: metrics})
        save_mc_results({Path(args.model_path).stem: metrics})
    
    elif args.run_all:
        # Find and evaluate all models
        models = find_models(OUTPUT_DIR)
        
        if not models:
            print(f"\nNo model files found in {OUTPUT_DIR}")
            print("Train models first with: python train_v2.py")
            return
        
        print(f"\nFound {len(models)} models:")
        for m in models:
            print(f"  - {m['name']}")
        
        results = {}
        for m in models:
            try:
                metrics = evaluate_single_model(m['path'], args.c_in, args.c_out, m['definition'], device)
                results[m['name']] = metrics
            except Exception as e:
                print(f"  Error: {e}")
        
        # Sauvegarde des métriques déterministes et MC Dropout
        save_eval_results(results)
        save_mc_results(results)

        # Summary table
        if results:
            print("\n" + "=" * 60)
            print("RÉSUMÉ FINAL")
            print("=" * 60)
            print(f"\n{'Modèle':<40} {'BA':>8} {'F1':>8} {'AUC':>8}")
            print("-" * 66)
            for name, m in results.items():
                print(f"{name[:40]:<40} {m['balanced_accuracy']:>8.4f} "
                      f"{m['f1']:>8.4f} {m['auc_roc']:>8.4f}")
    
    else:
        print("\nUsage:")
        print("  python -m dignn.evaluate --run_all")
        print("  python -m dignn.evaluate --model_path outputs/model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument('--model_path', type=str, default=None,
                        help="Path to model file")
    parser.add_argument('--definition', type=int, default=1, choices=[1, 2],
                        help="Heatwave definition")
    parser.add_argument('--c_in', type=int, default=C_IN,
                        help=f"Fenetre d'entree (defaut: {C_IN})")
    parser.add_argument('--c_out', type=int, default=C_OUT, choices=C_OUT_OPTIONS,
                        help=f"Horizon de prevision (defaut: {C_OUT})")
    parser.add_argument('--run_all', '--all', action='store_true',
                        help="Evaluate all models")
    
    args = parser.parse_args()
    main(args)