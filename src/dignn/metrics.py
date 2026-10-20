"""
Évaluation Probabiliste du DI-GNN
===================================
Métriques probabilistes pour la prévision des vagues de chaleur :
  - Brier Score (BS) et Brier Skill Score (BSS)
  - Diagramme de fiabilité (Reliability Diagram)
  - AUC-ROC par horizon de prévision (C_out)

Référence :
  Wilks, D.S. (2011). Statistical Methods in the Atmospheric Sciences.
  Academic Press.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve


# =============================================================================
# BRIER SCORE ET BRIER SKILL SCORE
# =============================================================================

def brier_score(y_true, y_prob):
    """
    Calcule le Brier Score.

    BS = (1/N) * sum((p_i - o_i)^2)

    Un BS parfait vaut 0. Plus il est faible, meilleure est la prévision.

    Args:
        y_true : Observations binaires (N,)
        y_prob : Probabilités prédites (N,)

    Returns:
        bs : Brier Score (float)
    """
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()
    return np.mean((y_prob - y_true) ** 2)


def brier_skill_score(y_true, y_prob):
    """
    Calcule le Brier Skill Score (BSS).

    BSS = 1 - BS_model / BS_climatologie

    La prévision de référence (climatologie) est la fréquence
    de base des vagues de chaleur dans l'ensemble de validation.

    BSS > 0 : le modèle bat la climatologie
    BSS = 0 : équivalent à la climatologie
    BSS < 0 : moins bon que la climatologie

    Args:
        y_true : Observations binaires (N,)
        y_prob : Probabilités prédites (N,)

    Returns:
        bss        : Brier Skill Score (float)
        bs_model   : Brier Score du modèle (float)
        bs_ref     : Brier Score de référence climatologique (float)
    """
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()

    # Prévision de référence = fréquence climatologique
    clim_prob = y_true.mean()
    bs_ref    = brier_score(y_true, np.full_like(y_prob, clim_prob))
    bs_model  = brier_score(y_true, y_prob)

    if bs_ref == 0:
        return 0.0, bs_model, bs_ref

    bss = 1.0 - bs_model / bs_ref
    return bss, bs_model, bs_ref


def bss_by_lead_time(y_true_all, y_prob_all):
    """
    Calcule le BSS pour chaque horizon de prévision C_out.

    Args:
        y_true_all : Observations (N, n_stations, c_out)
        y_prob_all : Probabilités (N, n_stations, c_out)

    Returns:
        bss_scores : Liste de BSS pour chaque horizon (c_out,)
    """
    c_out = y_true_all.shape[-1]
    bss_scores = []

    for t in range(c_out):
        y_t = y_true_all[:, :, t].flatten()
        p_t = y_prob_all[:, :, t].flatten()
        bss, _, _ = brier_skill_score(y_t, p_t)
        bss_scores.append(bss)

    return bss_scores


# =============================================================================
# AUC-ROC PAR HORIZON DE PRÉVISION
# =============================================================================

def auc_by_lead_time(y_true_all, y_prob_all):
    """
    Calcule l'AUC-ROC pour chaque horizon de prévision.

    Args:
        y_true_all : Observations (N, n_stations, c_out)
        y_prob_all : Probabilités (N, n_stations, c_out)

    Returns:
        auc_scores : Liste d'AUC pour chaque horizon (c_out,)
    """
    c_out = y_true_all.shape[-1]
    auc_scores = []

    for t in range(c_out):
        y_t = y_true_all[:, :, t].flatten()
        p_t = y_prob_all[:, :, t].flatten()
        try:
            auc = roc_auc_score(y_t, p_t)
        except ValueError:
            auc = float('nan')
        auc_scores.append(auc)

    return auc_scores


# =============================================================================
# DIAGRAMME DE FIABILITÉ
# =============================================================================

def reliability_diagram(y_true, y_prob, n_bins=10, title="Diagramme de fiabilité",
                         save_path=None):
    """
    Trace le diagramme de fiabilité (reliability diagram).

    Évalue la calibration probabiliste du modèle : les prédictions de
    probabilité p doivent correspondre à une fréquence observée ~p.

    Args:
        y_true    : Observations binaires (N,)
        y_prob    : Probabilités prédites (N,)
        n_bins    : Nombre de bins de probabilité (défaut : 10)
        title     : Titre du graphique
        save_path : Chemin de sauvegarde (optionnel)

    Returns:
        fig : Figure matplotlib
    """
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob,
                                              n_bins=n_bins, strategy='uniform')

    # Brier Skill Score pour annotation
    bss, bs_model, bs_ref = brier_skill_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Diagramme de fiabilité ---
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Calibration parfaite')
    ax1.plot(prob_pred, prob_true, 'o-', color='steelblue',
             linewidth=2, markersize=6, label='DI-GNN (MC Dropout)')
    ax1.fill_between(prob_pred, prob_pred, prob_true,
                     alpha=0.15, color='steelblue')

    ax1.set_xlabel('Probabilité prédite moyenne', fontsize=12)
    ax1.set_ylabel('Fréquence observée', fontsize=12)
    ax1.set_title(title, fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.text(0.05, 0.90,
             f'BSS = {bss:.3f}\nBS  = {bs_model:.4f}',
             transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax1.grid(True, alpha=0.3)

    # --- Histogramme des probabilités prédites ---
    ax2 = axes[1]
    ax2.hist(y_prob[y_true == 0], bins=20, alpha=0.6,
             color='steelblue', label='Non-vague de chaleur', density=True)
    ax2.hist(y_prob[y_true == 1], bins=20, alpha=0.6,
             color='tomato', label='Vague de chaleur', density=True)
    ax2.set_xlabel('Probabilité prédite', fontsize=12)
    ax2.set_ylabel('Densité', fontsize=12)
    ax2.set_title('Distribution des probabilités prédites', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée : {save_path}")

    return fig


# =============================================================================
# SYNTHÈSE DES MÉTRIQUES PROBABILISTES
# =============================================================================

def probabilistic_summary(y_true_all, y_prob_all, y_std_all=None,
                           c_out_labels=None):
    """
    Rapport de synthèse des métriques probabilistes.

    Args:
        y_true_all   : Observations (N, n_stations, c_out)
        y_prob_all   : Probabilités moyennes MC (N, n_stations, c_out)
        y_std_all    : Écarts-types MC (N, n_stations, c_out) — optionnel
        c_out_labels : Étiquettes des horizons (ex. [1,3,5,7])

    Returns:
        summary : Dictionnaire des métriques par horizon
    """
    c_out = y_true_all.shape[-1]
    if c_out_labels is None:
        c_out_labels = list(range(1, c_out + 1))

    bss_scores = bss_by_lead_time(y_true_all, y_prob_all)
    auc_scores = auc_by_lead_time(y_true_all, y_prob_all)

    summary = {}
    print(f"\n{'Horizon':<12} {'BSS':>8} {'AUC-ROC':>10} {'BS':>10}")
    print("-" * 42)

    for i, label in enumerate(c_out_labels):
        y_t = y_true_all[:, :, i].flatten()
        p_t = y_prob_all[:, :, i].flatten()
        _, bs_model, _ = brier_skill_score(y_t, p_t)

        summary[f'C_out={label}'] = {
            'bss'    : bss_scores[i],
            'auc'    : auc_scores[i],
            'bs'     : bs_model
        }
        if y_std_all is not None:
            mean_unc = y_std_all[:, :, i].mean()
            summary[f'C_out={label}']['mean_uncertainty'] = float(mean_unc)

        print(f"C_out={label:<6} {bss_scores[i]:>8.4f} {auc_scores[i]:>10.4f} "
              f"{bs_model:>10.4f}")

    return summary


# =============================================================================
# TEST RAPIDE
# =============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    N, S, T = 500, 8, 3

    y_true = (np.random.rand(N, S, T) > 0.95).astype(float)
    y_prob = np.clip(y_true + np.random.randn(N, S, T) * 0.2, 0, 1)
    y_std  = np.abs(np.random.randn(N, S, T) * 0.1)

    print("=== Test évaluation probabiliste ===")
    summary = probabilistic_summary(y_true, y_prob, y_std,
                                    c_out_labels=[1, 3, 5])

    fig = reliability_diagram(y_true.flatten(), y_prob.flatten(),
                               title="Diagramme de fiabilité — DI-GNN (test)")
    plt.show()
    print("\nTest réussi !")
