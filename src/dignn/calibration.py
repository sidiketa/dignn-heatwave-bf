"""
Calibration de Platt par horizon — DI-GNN (Burkina Faso)
=========================================================
Recalibre les probabilités MC Dropout du modèle DI-GNN, séparément
pour chaque horizon de prévision (J+1 ... J+7).

Principe scientifique :
  - Le modèle discrimine bien (AUC élevée) mais ses probabilités sont
    sur-confiantes (BSS négatif). Platt apprend une transformation
    sigmoïde monotone qui préserve l'ordre (donc l'AUC) tout en
    corrigeant l'échelle des probabilités (donc le BSS).
  - Découpage HONNÊTE : le calibrateur est ajusté sur une moitié de la
    validation, le gain est mesuré sur l'autre moitié. Sans cela, le
    gain serait un mirage (sur-ajustement du calibrateur).

Entrées (outputs/mc_probs/) :
  {stem}_mean_proba.npy   (N, n_stations, c_out)
  {stem}_y_true.npy       (N, n_stations, c_out)

Sortie :
  outputs/calibration_results.json   — BSS avant/après par horizon
  figures/resultats/fig_calibration.png
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve

# Palette thèse
NAVY, GRAY, TERRA = '#1A5276', '#5D6D7E', '#CA6F1E'
plt.rcParams.update({'font.family': 'serif', 'savefig.dpi': 300,
                     'figure.dpi': 120, 'axes.grid': True, 'grid.alpha': 0.3})

from .config import OUTPUT_DIR, FIGURES_DIR

PROBS_DIR  = OUTPUT_DIR / "mc_probs"
OUT_DIR    = OUTPUT_DIR
FIG_DIR    = FIGURES_DIR

SEED = 42


# =============================================================================
# Métriques
# =============================================================================
def brier_score(y, p):
    return np.mean((p - y) ** 2)


def brier_skill_score(y, p):
    """BSS vs climatologie (fréquence de base observée)."""
    clim = np.full_like(p, y.mean())
    bs_ref = brier_score(y, clim)
    if bs_ref == 0:
        return 0.0
    return 1.0 - brier_score(y, p) / bs_ref


# =============================================================================
# Calibration de Platt (régression logistique sur le logit)
# =============================================================================
def platt_fit(p_train, y_train):
    """
    Ajuste une sigmoïde de Platt. On régresse y sur le logit de p
    (et non sur p brut) : c'est la formulation standard, plus stable.
    """
    eps = 1e-6
    p = np.clip(p_train, eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver='lbfgs')  # quasi sans régularisation
    lr.fit(logit, y_train)
    return lr


def platt_apply(lr, p):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    return lr.predict_proba(logit)[:, 1]


# =============================================================================
# Pipeline
# =============================================================================
def main(stem):
    mean = np.load(PROBS_DIR / f"{stem}_mean_proba.npy")   # (N, S, T)
    y    = np.load(PROBS_DIR / f"{stem}_y_true.npy")        # (N, S, T)
    if y.ndim == 2:
        y = y[:, :, None]
    if mean.ndim == 2:
        mean = mean[:, :, None]

    N, S, T = mean.shape
    print(f"Modèle : {stem}")
    print(f"Formes : {mean.shape}  (N={N}, stations={S}, horizons={T})\n")

    # Découpage honnête : indices d'échantillons (axe temporel N), 50/50.
    # On découpe sur N (pas sur les stations) pour garder l'intégrité spatiale.
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(N)
    cut = N // 2
    fit_idx, test_idx = idx[:cut], idx[cut:]
    print(f"Découpage : {len(fit_idx)} échantillons pour l'ajustement, "
          f"{len(test_idx)} pour la mesure\n")

    results = {}
    print(f"{'Horizon':<9} {'BSS avant':>11} {'BSS après':>11} "
          f"{'AUC avant':>10} {'AUC après':>10}")
    print("-" * 54)

    rows = []
    for t in range(T):
        # Aplatir (échantillons × stations) sur le sous-ensemble correspondant
        p_fit  = mean[fit_idx,  :, t].ravel()
        y_fit  = y[fit_idx,     :, t].ravel()
        p_test = mean[test_idx, :, t].ravel()
        y_test = y[test_idx,    :, t].ravel()

        # Ajuster Platt sur fit, appliquer sur test
        lr = platt_fit(p_fit, y_fit)
        p_test_cal = platt_apply(lr, p_test)

        # Métriques sur le test (jamais vu par le calibrateur)
        bss_before = brier_skill_score(y_test, p_test)
        bss_after  = brier_skill_score(y_test, p_test_cal)
        try:
            auc_before = roc_auc_score(y_test, p_test)
            auc_after  = roc_auc_score(y_test, p_test_cal)
        except ValueError:
            auc_before = auc_after = float('nan')

        results[f"C_out={t+1}"] = {
            'bss_before': float(bss_before),
            'bss_after':  float(bss_after),
            'auc_before': float(auc_before),
            'auc_after':  float(auc_after),
            'platt_a': float(lr.coef_[0][0]),
            'platt_b': float(lr.intercept_[0]),
        }
        rows.append((t+1, p_test, y_test, p_test_cal, bss_before, bss_after))
        print(f"J+{t+1:<7} {bss_before:>11.4f} {bss_after:>11.4f} "
              f"{auc_before:>10.4f} {auc_after:>10.4f}")

    # Sauvegarde JSON
    tag = stem.replace("dignn_", "").replace("_seed42", "")
    with open(OUT_DIR / f"calibration_results_{tag}.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nRésultats sauvegardés : calibration_results_{tag}.json")

    # =========================================================================
    # FIGURE 1 — BSS avant/après par horizon
    # =========================================================================
    horizons = list(range(1, T + 1))
    bss_b = [results[f"C_out={t}"]['bss_before'] for t in horizons]
    bss_a = [results[f"C_out={t}"]['bss_after']  for t in horizons]

    # Extraire c_in/c_out du stem pour un titre exact
    import re as _re
    _cin = _re.search(r'cin(\d+)', stem)
    _cout = _re.search(r'cout(\d+)', stem)
    _cin = _cin.group(1) if _cin else '?'
    _cout = _cout.group(1) if _cout else '?'

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color=TERRA, linestyle='--', linewidth=1.2,
               label='BSS = 0 (climatologie)')
    ax.plot(horizons, bss_b, 'o--', color=GRAY, linewidth=2, markersize=8,
            label='Avant calibration')
    ax.plot(horizons, bss_a, 'o-', color=NAVY, linewidth=2.5, markersize=8,
            label='Après calibration de Platt')
    ax.set_xlabel("Horizon de prévision")
    ax.set_ylabel("Brier Skill Score (BSS)")
    ax.set_xticks(horizons)
    ax.set_xticklabels([f'J+{t}' for t in horizons])
    ax.set_title("Effet de la calibration de Platt sur le BSS\n"
             f"DI-GNN (C_in={_cin}, C_out={_cout})",
             fontweight='bold', pad=10)
    ax.legend(loc='center right', framealpha=0.9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"fig_calibration_bss_{tag}.png", dpi=300, bbox_inches='tight')
    print(f"Figure : fig_calibration_bss_{tag}.png")
    plt.close(fig)

    # =========================================================================
    # FIGURE 2 — Diagrammes de fiabilité avant/après (J+1, J+3, J+7)
    # =========================================================================
    show = [1, 3, 7] if T >= 7 else horizons[:3]
    fig, axes = plt.subplots(1, len(show), figsize=(5 * len(show), 4.5))
    if len(show) == 1:
        axes = [axes]
    for ax, h in zip(axes, show):
        _, p_test, y_test, p_cal, _, _ = rows[h - 1]
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Parfait')
        for p_data, lab, col in [(p_test, 'Avant', GRAY), (p_cal, 'Après', NAVY)]:
            try:
                frac, mean_pred = calibration_curve(y_test, p_data, n_bins=8,
                                                    strategy='quantile')
                ax.plot(mean_pred, frac, 'o-', color=col, label=lab, markersize=5)
            except Exception:
                pass
        ax.set_title(f'J+{h}', fontweight='bold')
        ax.set_xlabel('Probabilité prédite')
        ax.set_ylabel('Fréquence observée')
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.legend(fontsize=8)
    fig.suptitle("Diagrammes de fiabilité du DI-GNN avant et après calibration de Platt",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"fig_calibration_reliability_{tag}.png", dpi=300,
                bbox_inches='tight')
    print(f"Figure : fig_calibration_reliability_{tag}.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibration de Platt par horizon")
    parser.add_argument("--stem", type=str, default="dignn_cin10_cout7_seed42",
                        help="Préfixe du modèle dans outputs/mc_probs/ "
                             "(ex: dignn_cin10_cout3_seed42)")
    args = parser.parse_args()
    main(args.stem)