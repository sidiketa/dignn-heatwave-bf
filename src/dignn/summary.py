"""
Génération du Résumé des Résultats — Thèse DI-GNN Burkina Faso
===============================================================
Usage:
    python -m dignn.summary
"""

import json
import csv
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np


from .config import DATA_DIR, OUTPUT_DIR, TRAIN_YEARS, VAL_YEARS, C_IN, C_OUT


def load_sequences():
    seq_file = DATA_DIR / f"sequences_cin{C_IN}_cout{C_OUT}_v2.npz"
    if not seq_file.exists():
        seq_file = DATA_DIR / "sequences_def1_v2.npz"   # compat ascendante
    if not seq_file.exists():
        return None
    data = np.load(seq_file, allow_pickle=True)
    return {'X': data['X'], 'y': data['y'],
            'years': data['years'], 'features': list(data['features'])}


def load_eval_results():
    """Métriques déterministes officielles produites par evaluate_v2.py."""
    f = OUTPUT_DIR / "eval_results.json"
    if f.exists():
        with open(f) as fh:
            return json.load(fh)
    return None


def load_gpd_params():
    gpd_file = DATA_DIR / "gpd_params_v2.pkl"
    if not gpd_file.exists():
        return None
    with open(gpd_file, 'rb') as f:
        return pickle.load(f)


def load_mc_results():
    mc_file = OUTPUT_DIR / "mc_results.json"
    if mc_file.exists():
        with open(mc_file) as f:
            return json.load(f)
    return None


def section_dataset(seq):
    lines = ["## 1. Statistiques du jeu de données\n"]
    if seq is None:
        lines.append("*Données non trouvées. Lancer preprocessing_v2.py d'abord.*\n")
        return lines

    X, y, years, features = seq['X'], seq['y'], seq['years'], seq['features']

    lines.append(f"- **Séquences totales :** {len(X):,}")
    lines.append(f"- **Stations :** {X.shape[1]}")
    lines.append(f"- **Fenêtre d'entrée (C_in) :** {X.shape[2]} jours")
    lines.append(f"- **Nombre de variables :** {len(features)}")
    lines.append(f"- **Variables :** {', '.join(features)}")
    lines.append("")

    n_total  = y.size
    n_hw     = int(y.sum())
    n_non_hw = n_total - n_hw
    ratio    = n_non_hw / n_hw if n_hw > 0 else float('inf')

    lines.append("### Déséquilibre de classes\n")
    lines.append("| Métrique | Valeur |")
    lines.append("|---|---|")
    lines.append(f"| Étiquettes totales | {n_total:,} |")
    lines.append(f"| Jours de vague de chaleur | {n_hw:,} ({n_hw/n_total*100:.2f}%) |")
    lines.append(f"| Jours ordinaires | {n_non_hw:,} ({n_non_hw/n_total*100:.2f}%) |")
    lines.append(f"| Ratio déséquilibre | 1 : {ratio:.1f} |")
    lines.append(f"| pos_weight recommandé (BCE) | {ratio:.2f} |")
    lines.append("")

    train_mask = np.isin(years, TRAIN_YEARS)
    val_mask   = np.isin(years, VAL_YEARS)

    lines.append("### Déséquilibre par période\n")
    lines.append("| Période | Séquences | HW (%) |")
    lines.append("|---|---|---|")
    for label, mask in [("Entraînement (2010-2022)", train_mask),
                         ("Validation (2023-2024)",   val_mask)]:
        y_sub  = y[mask]
        n_s    = y_sub.size
        n_hw_s = int(y_sub.sum())
        pct    = n_hw_s / n_s * 100 if n_s > 0 else 0
        lines.append(f"| {label} | {n_s:,} | {n_hw_s:,} ({pct:.2f}%) |")
    lines.append("")
    return lines


def section_gpd(gpd_params):
    lines = ["## 2. Paramètres GPD par station\n"]
    if gpd_params is None:
        lines.append("*Paramètres GPD non trouvés.*\n")
        return lines
    lines.append("| Station | xi (Forme) | sigma (Echelle) | mu (C) | q95 (C) |")
    lines.append("|---|---|---|---|---|")
    for station, p in sorted(gpd_params.items()):
        lines.append(
            f"| {station} | {p['xi']:.4f} | {p['sigma']:.3f} | "
            f"{p['mu']:.1f} | {p.get('q95', float('nan')):.1f} |"
        )
    lines.append("")
    return lines


def section_model_results(eval_results):
    lines = ["## 3. Résultats déterministes des modèles\n"]

    # --- Source prioritaire : eval_results.json (évaluation officielle) ---
    if eval_results:
        lines.append("*Source : `eval_results.json` (évaluation sur 2023–2024).*\n")
        lines.append("| Modèle | BA | Précision | Rappel | F1 | AUC-ROC | AP |")
        lines.append("|---|---|---|---|---|---|---|")
        for name, m in sorted(eval_results.items()):
            def g(k):
                v = m.get(k, None)
                return f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"
            lines.append(
                f"| {name[:35]} | {g('balanced_accuracy')} | {g('precision')} | "
                f"{g('recall')} | {g('f1')} | {g('auc_roc')} | {g('avg_precision')} |"
            )
        lines.append("")
        return lines

    # --- Repli : historique d'entraînement (validation à la meilleure époque) ---
    history_files = list(OUTPUT_DIR.glob("*_history.json"))
    if not history_files:
        lines.append("*Aucun modèle entraîné. Lancer train_v2.py puis evaluate_v2.py.*\n")
        lines.append("```bash")
        lines.append("python -m dignn.train --model baseline --loss standard")
        lines.append("python -m dignn.train --model dignn --loss combined")
        lines.append("python -m dignn.evaluate --run_all")
        lines.append("```\n")
        return lines

    lines.append("*Source : historique d'entraînement (eval_results.json absent).*\n")
    lines.append("| Modèle | BA | Précision | Rappel | F1 | AUC-ROC |")
    lines.append("|---|---|---|---|---|---|")

    for hf in sorted(history_files):
        with open(hf) as f:
            history = json.load(f)
        name        = hf.stem.replace('_history', '')
        val_f1_list = history.get('val_f1', [0])
        best_idx    = int(np.argmax(val_f1_list)) if val_f1_list else 0

        def safe_get(key, default=float('nan')):
            lst = history.get(key, [])
            if not lst or best_idx >= len(lst):
                return default
            return lst[best_idx]

        ba   = safe_get('val_balanced_acc')
        prec = safe_get('val_precision')
        rec  = safe_get('val_recall')
        f1   = safe_get('val_f1')
        auc  = safe_get('val_auc_roc', None)

        def fmt(v):
            return f"{v:.4f}" if (v is not None and not (isinstance(v, float) and np.isnan(v))) else "N/A"

        lines.append(
            f"| {name[:35]} | {fmt(ba)} | {fmt(prec)} | "
            f"{fmt(rec)} | {fmt(f1)} | {fmt(auc)} |"
        )
    lines.append("")
    return lines


def export_csv_tables(eval_results, gpd_params):
    """Exporte table_performance.csv et table_gpd_params.csv pour LaTeX."""
    # Tableau de performance
    if eval_results:
        path = OUTPUT_DIR / "table_performance.csv"
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['modele', 'balanced_accuracy', 'precision', 'recall',
                        'f1', 'auc_roc', 'avg_precision'])
            for name, m in sorted(eval_results.items()):
                w.writerow([name,
                            m.get('balanced_accuracy', ''),
                            m.get('precision', ''),
                            m.get('recall', ''),
                            m.get('f1', ''),
                            m.get('auc_roc', ''),
                            m.get('avg_precision', '')])
        print(f"  CSV exporté : {path.name}")

    # Tableau GPD
    if gpd_params:
        path = OUTPUT_DIR / "table_gpd_params.csv"
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['station', 'xi', 'sigma', 'mu', 'var', 'q95'])
            for station, p in sorted(gpd_params.items()):
                w.writerow([station, p.get('xi', ''), p.get('sigma', ''),
                            p.get('mu', ''), p.get('var', ''), p.get('q95', '')])
        print(f"  CSV exporté : {path.name}")


def section_mc_results(mc_results):
    lines = ["## 4. Métriques probabilistes — MC Dropout (DI-GNN)\n"]
    if mc_results is None:
        lines.append("*Résultats MC Dropout non trouvés.*")
        lines.append("*Lancer `python -m dignn.evaluate --run_all` après entraînement.*\n")
        return lines

    lines.append("| Horizon (C_out) | BSS | AUC-ROC | BS | Incertitude sigma |")
    lines.append("|---|---|---|---|---|")
    for horizon, metrics in mc_results.items():
        if horizon == 'mean_uncertainty_global':
            continue
        bss = metrics.get('bss',              float('nan'))
        auc = metrics.get('auc',              float('nan'))
        bs  = metrics.get('bs',               float('nan'))
        unc = metrics.get('mean_uncertainty', float('nan'))
        lines.append(f"| {horizon} | {bss:.4f} | {auc:.4f} | {bs:.4f} | {unc:.4f} |")

    global_unc = mc_results.get('mean_uncertainty_global', None)
    if global_unc is not None:
        lines.append(f"\n**Incertitude moyenne globale (sigma) :** {global_unc:.4f}")
    lines.append("")
    return lines


def main():
    print("=" * 60)
    print("GENERATION DU RESUME — THESE DI-GNN BURKINA FASO")
    print("=" * 60)

    seq          = load_sequences()
    gpd_params   = load_gpd_params()
    mc_results   = load_mc_results()
    eval_results = load_eval_results()

    lines = []
    lines.append("# Résumé des résultats — Prévision des vagues de chaleur (Burkina Faso)")
    lines.append(f"\n*Généré le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    lines.append("---\n")

    lines += section_dataset(seq)
    lines += section_gpd(gpd_params)
    lines += section_model_results(eval_results)
    lines += section_mc_results(mc_results)

    output_path = OUTPUT_DIR / "paper_summary.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    # Export des tableaux CSV pour LaTeX
    export_csv_tables(eval_results, gpd_params)

    print(f"\nRésumé sauvegardé : {output_path}")
    print("\n" + '\n'.join(lines[:40]))


if __name__ == "__main__":
    main()
