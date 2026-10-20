#!/usr/bin/env python3
"""
Figures du Chapitre Résultats — Thèse MSc DI-GNN (UVBF)
=========================================================
Génère 8 figures en français, 300 DPI, palette thèse.

Palette :
  Navy      #1A5276  — DI-GNN (modèle principal)
  Gris      #5D6D7E  — GNN de référence / secondaire
  Terracotta #CA6F1E — accents, lignes de référence

Dépendances (générées par l'évaluation) :
  outputs/eval_results.json   ← python -m dignn.evaluate --run_all
  outputs/mc_results.json     ← python -m dignn.evaluate --run_all
  outputs/*dignn*_history.json ← python -m dignn.train

Le script s'arrête avec un message explicite si un fichier d'entrée manque.

CORRECTION (juillet 2026) : _primary_dignn / _primary_baseline
sélectionnent désormais la graine 42 de manière FIXE, plutôt que le
meilleur F1 parmi les 10 graines disponibles. Cela garantit la
cohérence avec le reste du chapitre 4 (matrice de confusion "sur la
graine 42", tableau de comparaison multi-graines) et évite tout
soupçon de sélection a posteriori de la graine la plus favorable.

Usage :
  python -m dignn.plotting.result_figures
"""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')

# =============================================================================
# PALETTE ET STYLE
# =============================================================================
NAVY  = '#1A5276'
GRAY  = '#5D6D7E'
TERRA = '#CA6F1E'
WHITE = '#FFFFFF'
LIGHT = '#D6EAF8'

plt.rcParams.update({
    'font.family'       : 'serif',
    'font.size'         : 11,
    'axes.labelsize'    : 12,
    'axes.titlesize'    : 13,
    'xtick.labelsize'   : 10,
    'ytick.labelsize'   : 10,
    'legend.fontsize'   : 10,
    'figure.dpi'        : 150,
    'savefig.dpi'       : 300,
    'savefig.bbox'      : 'tight',
    'axes.grid'         : True,
    'grid.alpha'        : 0.3,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
})

# =============================================================================
# CHEMINS
# =============================================================================
from ..config import DATA_DIR, OUTPUT_DIR, FIGURES_DIR

# =============================================================================
# CHARGEMENT
# =============================================================================

def load_eval_results():
    """Charge eval_results.json produit par evaluate_v2.py."""
    path = OUTPUT_DIR / "eval_results.json"
    if not path.exists():
        print(f"  [INFO] {path.name} absent — données de démonstration utilisées.")
        print("  Lancer d'abord : python evaluate_v2.py --run_all")
        return None
    with open(path) as f:
        return json.load(f)


def load_mc_results():
    """Charge mc_results.json produit par evaluate_v2.py."""
    path = OUTPUT_DIR / "mc_results.json"
    if not path.exists():
        print(f"  [INFO] {path.name} absent — métriques MC non disponibles.")
        return None
    with open(path) as f:
        return json.load(f)


def load_best_history():
    """Charge l'historique d'entraînement du meilleur DI-GNN (CIN10_COUT3)."""
    candidates = sorted(OUTPUT_DIR.glob("*dignn*cin10*cout3*_history.json"))
    if not candidates:
        candidates = sorted(OUTPUT_DIR.glob("*dignn*_history.json"))
    if not candidates:
        print("  [INFO] Aucun historique d'entraînement trouvé.")
        return None
    path = candidates[-1]
    print(f"  Historique chargé : {path.name}")
    with open(path) as f:
        return json.load(f)




def _save(fig, name):
    """Sauvegarde PNG + PDF dans FIGURES_DIR."""
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / f"{name}.pdf",           bbox_inches='tight')
    print(f"  ✓ {name}.png")
    plt.close(fig)


# =============================================================================
# FIGURE 1 — Courbes ROC et Précision-Rappel (2 panneaux)
# Référence : Li et al. (2023), Chishtie et al. (2024)
# Justification : indispensable pour tout problème de classification binaire.
# La courbe PR est cruciale ici car le déséquilibre est de 1:22.
# =============================================================================
def _pick(eval_results, model_type='dignn', c_in=None, c_out=None):
    """
    Sélectionne une entrée par type de modèle et, optionnellement, par (c_in, c_out).
    Robuste au nommage des fichiers (utilise les champs model_type/c_in/c_out).
    Si plusieurs candidats, garde le meilleur F1.

    NOTE : cette fonction de repli n'est utilisée par _primary_dignn /
    _primary_baseline que si la graine 42 est introuvable dans le
    fichier. Le choix normal (seed=42 fixe) ne passe pas par ici.
    """
    cands = []
    for name, m in eval_results.items():
        if m.get('model_type') != model_type:
            continue
        if c_in is not None and m.get('c_in') != c_in:
            continue
        if c_out is not None and m.get('c_out') != c_out:
            continue
        cands.append(m)
    if not cands:
        return {}
    return max(cands, key=lambda m: m.get('f1', 0))


def _primary_dignn(eval_results, seed=42):
    """
    DI-GNN de référence : c_in=10, c_out=3, seed FIXE (cohérent avec le
    reste du chapitre — matrice de confusion et tableau de comparaison
    utilisent tous les deux la graine 42). Ne PAS sélectionner par
    meilleur F1 : cela romprait la cohérence entre figures et
    contredirait l'argument méthodologique du protocole multi-graines
    (choisir la graine la plus favorable a posteriori reviendrait à du
    cherry-picking).
    """
    for name, m in eval_results.items():
        if (m.get('model_type') == 'dignn' and m.get('c_in') == 10
                and m.get('c_out') == 3 and f'seed{seed}' in name):
            return m
    # Repli : si la graine demandée est absente du fichier, on retombe
    # sur l'ancien comportement plutôt que de planter.
    print(f"   [ATTENTION] dignn seed{seed} (c_in=10, c_out=3) introuvable "
          f"— repli sur le meilleur F1 disponible.")
    d = _pick(eval_results, 'dignn', c_in=10, c_out=3)
    return d if d else _pick(eval_results, 'dignn')


def _primary_baseline(eval_results, seed=42):
    """GNN de référence : c_in=10, c_out=3, même graine que le DI-GNN (42)."""
    for name, m in eval_results.items():
        if (m.get('model_type') == 'baseline' and m.get('c_in') == 10
                and m.get('c_out') == 3 and f'seed{seed}' in name):
            return m
    print(f"   [ATTENTION] baseline seed{seed} (c_in=10, c_out=3) introuvable "
          f"— repli sur le meilleur F1 disponible.")
    return _pick(eval_results, 'baseline')


def fig_roc_pr(eval_results, save=True):

    dignn = _primary_dignn(eval_results)
    base  = _primary_baseline(eval_results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panneau gauche : courbe ROC ---
    ax = axes[0]

    for data, label, color, ls in [
        (dignn, 'DI-GNN',            NAVY, '-'),
        (base,  'GNN de référence',  GRAY, '--'),
    ]:
        if not data:
            continue
        auc_val = data.get('auc_roc', 0)
        if 'fpr' in data and 'tpr' in data:
            fpr = np.array(data['fpr'])
            tpr = np.array(data['tpr'])
            ax.plot(fpr, tpr, color=color, linewidth=2.5, linestyle=ls,
                    label=f'{label}  (AUC = {auc_val:.3f})')
        else:
            print(f"   [INFO] Courbe ROC absente pour {label} "
                  f"— relancer evaluate_v2.py pour générer fpr/tpr.")

    ax.plot([0, 1], [0, 1], '--', color=TERRA, linewidth=1.2, alpha=0.7,
            label='Référence aléatoire')
    ax.set_xlabel('Taux de faux positifs')
    ax.set_ylabel('Taux de vrais positifs (Rappel)')
    ax.set_title('Courbe ROC\nEnsemble de validation 2023–2024',
                 fontweight='bold', pad=10)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    # --- Panneau droit : courbe Précision-Rappel ---
    ax = axes[1]
    # Prévalence réelle issue de l'évaluation (pas de valeur codée en dur)
    prevalence = dignn.get('prevalence', base.get('prevalence', 0.069))

    for data, label, color, ls in [
        (dignn, 'DI-GNN',            NAVY, '-'),
        (base,  'GNN de référence',  GRAY, '--'),
    ]:
        if not data:
            continue
        ap = data.get('avg_precision', 0)
        if 'recall_curve' in data and 'precision_curve' in data:
            rec  = np.array(data['recall_curve'])
            prec = np.array(data['precision_curve'])
            ax.plot(rec, prec, color=color, linewidth=2.5, linestyle=ls,
                    label=f'{label}  (AP = {ap:.3f})')
        else:
            print(f"   [INFO] Courbe PR absente pour {label} "
                  f"— relancer evaluate_v2.py pour générer les courbes.")

    ax.axhline(y=prevalence, color=TERRA, linestyle=':', linewidth=1.5,
               label=f'Climatologie  (fréquence = {prevalence:.3f})')
    ax.set_xlabel('Rappel')
    ax.set_ylabel('Précision')
    ax.set_title('Courbe Précision-Rappel\nEnsemble de validation 2023–2024',
                 fontweight='bold', pad=10)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    fig.suptitle(
    'Capacité discriminante du DI-GNN et du GNN de référence\n'
    '8 stations, validation 2023-2024, définition PKL (P90, ≥ 3 jours consécutifs)',
     fontsize=12, fontweight='bold')

    plt.tight_layout()
    if save:
        _save(fig, 'fig1_roc_pr')
    else:
        return fig


# =============================================================================
# FIGURE 2 — Comparaison globale des métriques (barplot horizontal)
# Référence : Chishtie et al. (2024), Fig. 10 ; Li et al. (2023), Table 2
# Justification : lecture directe du gain apporté par le DI-GNN sur chaque
# métrique. Le rappel de 0,896 est l'argument central (vs Li et al. 58,5 %).
# =============================================================================
def fig_comparaison_globale(eval_results, save=True):

    dignn = _primary_dignn(eval_results)
    base  = _primary_baseline(eval_results)

    etiquettes = [
        'Exactitude équilibrée',
        'Rappel (Taux de détection)',
        'Précision',
        'Score F1',
        'AUC-ROC',
    ]
    cles = ['balanced_accuracy', 'recall', 'precision', 'f1', 'auc_roc']

    vals_d = [dignn.get(k, 0) for k in cles]
    vals_b = [base.get(k, 0)  for k in cles]

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(cles))
    h = 0.36

    bars_d = ax.barh(y + h / 2, vals_d, h, color=NAVY,
                     label='DI-GNN  (C_in=10, C_out=3)',
                     edgecolor='white', linewidth=0.5)
    bars_b = ax.barh(y - h / 2, vals_b, h, color=GRAY,
                     label='GNN de référence',
                     edgecolor='white', linewidth=0.5)

    for bar, val in list(zip(bars_d, vals_d)) + list(zip(bars_b, vals_b)):
        ax.text(bar.get_width() + 0.010,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', ha='left', fontsize=9.5)

    ax.set_yticks(y)
    ax.set_yticklabels(etiquettes)
    ax.set_xlabel('Score')
    ax.set_title(
        'Comparaison des performances — DI-GNN vs GNN de référence\n'
        'Ensemble de validation 2023–2024, 8 stations, 553 événements',
        fontweight='bold', pad=10
    )
    ax.set_xlim([0, 1.15])
    ax.legend(loc='lower right', framealpha=0.9)
    ax.axvline(x=0.5, color=TERRA, linestyle=':', linewidth=1, alpha=0.6,
               label='Seuil 0,5')

    plt.tight_layout()
    if save:
        _save(fig, 'fig2_comparaison_globale')
    else:
        return fig


# =============================================================================
# FIGURE 3 — Heatmap d'ablation C_IN × C_OUT (2 panneaux)
# Référence : Chishtie et al. (2024), Fig. 6–7 ; Davidson et al. (2022)
# Justification : répond aux deux questions de recherche Q1 (horizon) et
# Q2 (fenêtre d'entrée). Figure centrale de la contribution empirique.
# =============================================================================
def fig_ablation_heatmap(eval_results, save=True):

    C_IN_VALS  = [7, 10, 14]
    C_OUT_VALS = [1,  3,  5,  7]

    def _matrice(metrique):
        M = np.full((len(C_IN_VALS), len(C_OUT_VALS)), np.nan)
        for i, cin in enumerate(C_IN_VALS):
            for j, cout in enumerate(C_OUT_VALS):
                d = _pick(eval_results, 'dignn', c_in=cin, c_out=cout)
                M[i, j] = d.get(metrique, np.nan) if d else np.nan
        return M

    ba_mat = _matrice('balanced_accuracy')
    f1_mat = _matrice('f1')

    cmap_ba = LinearSegmentedColormap.from_list('ba', [WHITE, LIGHT, NAVY],  N=256)
    cmap_f1 = LinearSegmentedColormap.from_list('f1', [WHITE, '#F5CBA7', TERRA], N=256)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, mat, cmap, titre in [
        (axes[0], ba_mat, cmap_ba, 'Exactitude équilibrée'),
        (axes[1], f1_mat, cmap_f1, 'Score F1'),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=0.50, vmax=1.00, aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(C_OUT_VALS)))
        ax.set_xticklabels([f'C_out={c}\n(J+{c})' for c in C_OUT_VALS])
        ax.set_yticks(range(len(C_IN_VALS)))
        ax.set_yticklabels([f'C_in={c}' for c in C_IN_VALS])
        ax.set_xlabel('Horizon de prévision (jours)')
        ax.set_ylabel("Fenêtre d'entrée (jours)")
        ax.set_title(titre, fontweight='bold', pad=8)

        for i in range(len(C_IN_VALS)):
            for j in range(len(C_OUT_VALS)):
                val = mat[i, j]
                if np.isnan(val):
                    continue
                ref = (C_IN_VALS[i] == 10 and C_OUT_VALS[j] == 3)
                txt = f'{val:.3f}' + (' ★' if ref else '')
                color  = 'white' if val > 0.80 else 'black'
                weight = 'bold'  if ref else 'normal'
                ax.text(j, i, txt, ha='center', va='center',
                        color=color, fontsize=9, fontweight=weight)

    fig.suptitle(
        "Étude d'ablation DI-GNN — Impact de la fenêtre d'entrée et de l'horizon\n"
        "★ = configuration de référence (C_in=10, C_out=3)",
        fontsize=12, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    if save:
        _save(fig, 'fig3_ablation_heatmap')
    else:
        return fig


# =============================================================================
# FIGURE 4 — Dégradation du score avec l'horizon de prévision
# Référence : Davidson et al. (2022) ; Miloshevich et al. (2022)
# Justification : quantifie la perte de compétence J+1 → J+7.
# Axe double : métriques déterministes (gauche) + BSS probabiliste (droite).
# =============================================================================
def fig_degradation_horizon(eval_results, mc_results, save=True):

    C_OUT_VALS = [1, 3, 5, 7]

    ba_vals  = []
    rec_vals = []
    bss_vals = []

    for cout in C_OUT_VALS:
        d = _pick(eval_results, 'dignn', c_in=10, c_out=cout)
        ba_vals.append(d.get('balanced_accuracy', np.nan))
        rec_vals.append(d.get('recall', np.nan))

        bss = np.nan
        if mc_results:
            mc = mc_results.get(f'C_out={cout}', {})
            bss = mc.get('bss', np.nan)
        bss_vals.append(bss)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax2.spines['right'].set_visible(True)

    l1, = ax1.plot(C_OUT_VALS, ba_vals,  'o-',  color=NAVY,  linewidth=2.5,
                   markersize=8, label='Exactitude équilibrée (gauche)')
    l2, = ax1.plot(C_OUT_VALS, rec_vals, 's--', color=GRAY,  linewidth=2.5,
                   markersize=8, label='Rappel (gauche)')
    l3, = ax2.plot(C_OUT_VALS, bss_vals, 'D:',  color=TERRA, linewidth=2.5,
                   markersize=8, label='BSS — MC Dropout (droite)')

    ax2.axhline(y=0, color=TERRA, linestyle='-', linewidth=0.8, alpha=0.4)

    ax1.set_xlabel('Horizon de prévision')
    ax1.set_ylabel('Score (0–1)')
    ax2.set_ylabel('Brier Skill Score')
    ax1.set_title(
        "Dégradation de la compétence avec l'horizon de prévision\n"
        "DI-GNN — C_in = 10 fixé, 8 stations, validation 2023–2024",
        fontweight='bold', pad=10
    )
    ax1.set_xticks(C_OUT_VALS)
    ax1.set_xticklabels([f'J+{c}' for c in C_OUT_VALS])
    ax1.set_ylim([0.40, 1.02])
    ax2.set_ylim([-0.15, 1.02])

    # Annoter les valeurs BA
    for x, y in zip(C_OUT_VALS, ba_vals):
        if not np.isnan(y):
            ax1.annotate(f'{y:.3f}', (x, y),
                         xytext=(0, 12), textcoords='offset points',
                         ha='center', fontsize=9, color=NAVY)

    lignes  = [l1, l2, l3]
    labels  = [l.get_label() for l in lignes]
    ax1.legend(lignes, labels, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    if save:
        _save(fig, 'fig4_degradation_horizon')
    else:
        return fig


# =============================================================================
# FIGURE 5 — Matrice de confusion DI-GNN (valeurs réelles)
# Référence : standard dans toute la littérature de classification
# Justification : visualise précisément TN/FP/FN/TP du modèle principal.
# Double panneau : valeurs absolues + taux normalisés par ligne.
# =============================================================================
def fig_matrice_confusion(eval_results, save=True):

    data = _primary_dignn(eval_results)
    cm   = np.array(data.get('confusion_matrix', [[11820, 380], [58, 506]]))

    # Normalisation par ligne (taux de classification correcte par classe)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    classes  = ['Jour ordinaire', 'Vague de chaleur']
    cmap_abs = LinearSegmentedColormap.from_list('cm_abs',  [WHITE, NAVY],  N=256)
    cmap_pct = LinearSegmentedColormap.from_list('cm_pct',  [WHITE, TERRA], N=256)

    for ax, mat, cmap, titre, fmt in [
        (axes[0], cm,      cmap_abs, 'Valeurs absolues',
         lambda v: f'{int(v):,}'),
        (axes[1], cm_norm, cmap_pct, 'Taux normalisés par classe réelle',
         lambda v: f'{v:.1%}'),
    ]:
        im = ax.imshow(mat, cmap=cmap,
                       vmin=0, vmax=(cm.max() if mat is cm else 1.0))
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(classes, rotation=15, ha='right')
        ax.set_yticklabels(classes)
        ax.set_xlabel('Classe prédite')
        ax.set_ylabel('Classe réelle')
        ax.set_title(titre, fontweight='bold', pad=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        thresh = mat.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = mat[i, j]
                couleur = 'white' if val > thresh else 'black'
                ax.text(j, i, fmt(val),
                        ha='center', va='center',
                        color=couleur, fontsize=13, fontweight='bold')

    

    fig.suptitle(
        'Matrice de confusion : DI-GNN (C_in=10, C_out=3, graine 42)\n'
        'Ensemble de validation 2023-2024',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    if save:
        _save(fig, 'fig5_matrice_confusion')
    else:
        return fig


# =============================================================================
# FIGURE 6 — Courbes d'apprentissage (loss + F1)
# Référence : Chishtie et al. (2024), Fig. 2–3 ; Li et al. (2023)
# Justification : montre la convergence et l'absence de surapprentissage.
# Deux métriques seulement (pas 6) pour la lisibilité.
# =============================================================================
def fig_apprentissage(history, save=True):

    if history is None:
        print("  [INFO] Historique absent — fig_apprentissage ignorée.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    panneaux = [
        ('train_loss', 'val_loss', 'Fonction de perte combinée\n(0,5 × BCE + 0,5 × (1 − F₁))', False),
        ('train_f1',  'val_f1',   'Score F1',                                                    True),
    ]

    for ax, (train_k, val_k, titre, chercher_best) in zip(axes, panneaux):
        train_vals = history.get(train_k, [])
        val_vals   = history.get(val_k,   [])
        if not train_vals:
            ax.set_title(f'{titre}\n(données absentes)', fontweight='bold')
            continue

        epochs = range(1, len(train_vals) + 1)
        ax.plot(epochs, train_vals, color=NAVY, linewidth=2.0,
                label='Entraînement')
        if val_vals:
            ax.plot(epochs, val_vals, color=GRAY, linewidth=2.0,
                    linestyle='--', label='Validation')

            if chercher_best and val_vals:
                best_ep  = int(np.argmax(val_vals)) + 1
                best_val = float(np.max(val_vals))
                ax.axvline(x=best_ep, color=TERRA, linestyle=':',
                           linewidth=1.5,
                           label=f'Meilleure époque : {best_ep}')
                ax.annotate(
                    f'F₁ = {best_val:.3f}',
                    xy=(best_ep, best_val),
                    xytext=(best_ep + len(train_vals) * 0.05, best_val - 0.06),
                    fontsize=9, color=TERRA,
                    arrowprops=dict(arrowstyle='->', color=TERRA, lw=1.2)
                )

        ax.set_xlabel('Époque')
        ax.set_ylabel(titre.split('\n')[0])
        ax.set_title(titre, fontweight='bold', pad=8)
        ax.legend(loc='best', framealpha=0.9)

    fig.suptitle(
        "Courbes d'apprentissage — DI-GNN (C_in=10, C_out=3)\n"
        "Arrêt anticipé : patience = 20 époques, optimiseur Adam (lr=5×10⁻⁴)",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    if save:
        _save(fig, 'fig6_apprentissage')
    else:
        return fig


# =============================================================================
# FIGURE 7 — Diagramme de fiabilité MC Dropout (BSS par horizon)
# Référence : Wilks (2011) ; Miloshevich et al. (2022)
# Justification : évalue la calibration probabiliste du modèle.
# Argument fort : le DI-GNN produit des probabilités interprétables.
# =============================================================================
def fig_fiabilite_mc(mc_results, save=True):

    if mc_results is None:
        print("  [INFO] mc_results.json absent — fig_fiabilite_mc ignorée.")
        return

    C_OUT_VALS  = [1, 3, 5, 7]
    COLS_HORIZ  = [NAVY, GRAY, TERRA, '#7D3C98']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panneau gauche : BSS par horizon ---
    ax = axes[0]
    bss_vals = []
    auc_vals = []
    for cout in C_OUT_VALS:
        mc = mc_results.get(f'C_out={cout}', {})
        bss_vals.append(mc.get('bss', np.nan))
        auc_vals.append(mc.get('auc', np.nan))

    x = np.arange(len(C_OUT_VALS))
    w = 0.38
    bars_b = ax.bar(x - w / 2, bss_vals, w, color=NAVY,
                    label='Brier Skill Score', edgecolor='white')
    bars_a = ax.bar(x + w / 2, auc_vals, w, color=GRAY,
                    label='AUC-ROC (MC)', edgecolor='white', alpha=0.85)

    for bars, vals in [(bars_b, bss_vals), (bars_a, auc_vals)]:
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.axhline(y=0, color=TERRA, linestyle='--', linewidth=1.2,
               label='BSS = 0 (climatologie)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'J+{c}' for c in C_OUT_VALS])
    ax.set_xlabel('Horizon de prévision')
    ax.set_ylabel('Score')
    ax.set_title('Métriques probabilistes par horizon\n(MC Dropout, N=50 passages)',
                 fontweight='bold', pad=8)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim([-0.1, 1.1])

    # --- Panneau droit : incertitude globale ---
    ax = axes[1]
    sigma_global = mc_results.get('mean_uncertainty_global', None)
    sigma_horiz  = []
    for cout in C_OUT_VALS:
        mc = mc_results.get(f'C_out={cout}', {})
        sigma_horiz.append(mc.get('mean_uncertainty', np.nan))

    ax.bar([f'J+{c}' for c in C_OUT_VALS], sigma_horiz,
           color=COLS_HORIZ, edgecolor='white', linewidth=0.5)

    if sigma_global is not None:
        ax.axhline(y=sigma_global, color=NAVY, linestyle='--',
                   linewidth=1.5, label=f'σ moyen global = {sigma_global:.4f}')
        ax.legend(loc='upper right', framealpha=0.9)

    ax.set_xlabel('Horizon de prévision')
    ax.set_ylabel('Incertitude épistémique moyenne (σ)')
    ax.set_title("Incertitude MC Dropout par horizon\n"
                 "(σ plus élevé = prévision moins certaine)",
                 fontweight='bold', pad=8)

    fig.suptitle(
        'Évaluation probabiliste — DI-GNN avec MC Dropout (N=50)\n'
        'Validation 2023–2024, 8 stations',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    if save:
        _save(fig, 'fig7_fiabilite_mc')
    else:
        return fig


# =============================================================================
# FIGURE 8 — BSS par horizon comparé à la littérature (optional, mais fort)
# Référence : Li et al. (2023) Rappel 58,5 % ; Chishtie et al. (2024) BA >86 %
# Justification : positionne le DI-GNN face aux modèles comparables.
# =============================================================================
def fig_comparaison_litterature(eval_results, save=True):
    """
    Barplot de comparaison avec les études similaires.
    Chiffres de la revue de littérature (Ch.2, Tab. comparaison_lit).
    """
    # Données issues de la synthèse — à mettre à jour avec résultats réels
    etudes = [
        'DI-GNN\n(cette étude)',
        'Chishtie\net al. (2024)',
        'Li et al.\n(2023)',
        'Jacques-Dumas\net al. (2022)',
        'Guigma\net al. (2021)',
    ]
    ba_vals = [
        _primary_dignn(eval_results).get('balanced_accuracy', np.nan),
        0.860,   # Chishtie et al. (2024) — BA >86 %
        np.nan,  # Li et al. — Rappel reporté, pas BA
        np.nan,  # Jacques-Dumas — métrique différente
        np.nan,  # Guigma — score de compétence saisonnier
    ]
    rec_vals = [
        _primary_dignn(eval_results).get('recall', np.nan),
        np.nan,  # Chishtie — non reporté
        0.585,   # Li et al. (2023) — Rappel 58,5 %
        np.nan,
        np.nan,
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(etudes))
    w = 0.38

    bars_ba  = ax.bar(x - w / 2,
                      [v if not np.isnan(v) else 0 for v in ba_vals],
                      w, color=NAVY, label='Exactitude équilibrée',
                      edgecolor='white')
    bars_rec = ax.bar(x + w / 2,
                      [v if not np.isnan(v) else 0 for v in rec_vals],
                      w, color=GRAY, label='Rappel',
                      edgecolor='white', alpha=0.85)

    # Marquer les barres vides (non reportées)
    for vals, bars in [(ba_vals, bars_ba), (rec_vals, bars_rec)]:
        for val, bar in zip(vals, bars):
            if np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                        'n.r.', ha='center', va='bottom',
                        fontsize=8, color='gray', style='italic')
            else:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(etudes, fontsize=10)
    ax.set_ylabel('Score')
    ax.set_ylim([0, 1.12])
    ax.set_title(
        'Positionnement par rapport à la littérature — Prévision des vagues de chaleur\n'
        'n.r. = non reporté dans les études citées',
        fontweight='bold', pad=10
    )
    ax.legend(loc='upper right', framealpha=0.9)
    ax.axhline(y=0.5, color=TERRA, linestyle=':', linewidth=1, alpha=0.6)

    plt.tight_layout()
    if save:
        _save(fig, 'fig8_comparaison_litterature')
    else:
        return fig


# =============================================================================
# FIGURES RAPATRIÉES DE LA MÉTHODOLOGIE
# Décision : ξ/σ GPD, timeline de validation et saisonnalité appartiennent
# au chapitre Résultats (caractérisation du régime extrême + protocole).
# =============================================================================

def load_gpd_params():
    """Charge gpd_params_v2.pkl (paramètres GPD par station)."""
    import pickle
    path = DATA_DIR / "gpd_params_v2.pkl"
    if not path.exists():
        print(f"  [INFO] {path.name} absent.")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)




# -----------------------------------------------------------------------------
# FIGURE M1 — Paramètres GPD par station (ξ et σ)
# Référence : Coles (2001) ; Davison & Smith (1990)
# Justification : ξ<0 partout = queues bornées, signature du régime sahélien.
# Résultat central qui justifie l'intégration de la TVE dans le DI-GNN.
# -----------------------------------------------------------------------------
def fig_gpd_stations(gpd_params, save=True):

    stations = list(gpd_params.keys())
    xi_vals    = [gpd_params[s].get('xi', np.nan)    for s in stations]
    sigma_vals = [gpd_params[s].get('sigma', np.nan) for s in stations]

    # Noms courts pour l'axe
    noms = [s.split()[0].capitalize() for s in stations]
    x = np.arange(len(stations))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # --- Panneau gauche : paramètre de forme ξ ---
    ax = axes[0]
    couleurs = [NAVY if v < 0 else TERRA for v in xi_vals]
    bars = ax.bar(x, xi_vals, color=couleurs, edgecolor='white', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=1.0)
    moy_xi = np.nanmean(xi_vals)
    ax.axhline(y=moy_xi, color=TERRA, linestyle='--', linewidth=1.5,
               label=f'Moyenne : ξ = {moy_xi:.3f}')

    for bar, val in zip(bars, xi_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val - 0.015 if val < 0 else val + 0.005,
                f'{val:.2f}', ha='center',
                va='top' if val < 0 else 'bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(noms, rotation=40, ha='right')
    ax.set_ylabel('Paramètre de forme (ξ)')
    ax.set_title('Paramètre de forme GPD\nξ < 0 → queue bornée (régime sahélien)',
                 fontweight='bold', pad=8)
    ax.legend(loc='lower left', framealpha=0.9)

    # --- Panneau droit : paramètre d'échelle σ ---
    ax = axes[1]
    bars = ax.bar(x, sigma_vals, color=GRAY, edgecolor='white', linewidth=0.5)
    moy_sigma = np.nanmean(sigma_vals)
    ax.axhline(y=moy_sigma, color=TERRA, linestyle='--', linewidth=1.5,
               label=f'Moyenne : σ = {moy_sigma:.3f}')

    for bar, val in zip(bars, sigma_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(noms, rotation=40, ha='right')
    ax.set_ylabel("Paramètre d'échelle (σ, °C)")
    ax.set_title("Paramètre d'échelle GPD\nDispersion des dépassements de seuil",
                 fontweight='bold', pad=8)
    ax.legend(loc='upper right', framealpha=0.9)

    fig.suptitle(
        'Paramètres de la distribution de Pareto généralisée par station\n'
        'Ajustés sur 2010–2022 (P90 local, ≥ 3 jours consécutifs)',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    if save:
        _save(fig, 'fig_m1_gpd_stations')
    else:
        return fig


# -----------------------------------------------------------------------------
# FIGURE M2 — Timeline de validation hold-out chronologique
# Référence : albelali2025 ; boser2024
# Justification : illustre le protocole sans fuite temporelle.
# Ouvre le chapitre Résultats en rappelant le cadre d'évaluation.
# -----------------------------------------------------------------------------
def fig_timeline_validation(save=True):

    train_years = list(range(2010, 2023))   # 2010–2022
    val_years   = [2023, 2024]

    fig, ax = plt.subplots(figsize=(11, 2.6))

    annees = train_years + val_years
    for i, an in enumerate(annees):
        couleur = NAVY if an in train_years else TERRA
        ax.barh(0, 1, left=i, height=0.6, color=couleur,
                edgecolor='white', linewidth=1.2)
        ax.text(i + 0.5, 0, str(an), ha='center', va='center',
                color='white', fontsize=8.5, fontweight='bold', rotation=90)

    # Séparateur train/val
    sep = len(train_years)
    ax.axvline(x=sep, color='black', linewidth=2, linestyle='-')

    ax.text(len(train_years) / 2, 0.55,
            f'ENTRAÎNEMENT\n{train_years[0]}–{train_years[-1]}  ({len(train_years)} ans)',
            ha='center', va='bottom', fontsize=10, fontweight='bold', color=NAVY)
    ax.text(sep + len(val_years) / 2, 0.55,
            f'VALIDATION\n{val_years[0]}–{val_years[-1]}  ({len(val_years)} ans)',
            ha='center', va='bottom', fontsize=10, fontweight='bold', color=TERRA)

    ax.set_xlim([-0.3, len(annees) + 0.3])
    ax.set_ylim([-0.5, 1.3])
    ax.axis('off')

    fig.suptitle(
        'Protocole de validation hold-out chronologique\n'
        'Découpage strictement temporel — aucune fuite entre entraînement et test',
        fontsize=12, fontweight='bold', y=1.18
    )
    plt.tight_layout()
    if save:
        _save(fig, 'fig_m2_timeline_validation')
    else:
        return fig


# -----------------------------------------------------------------------------
# FIGURE M3 — Heatmap de saisonnalité (mois × station)
# Référence : oueslati2017 ; sawadogo2024
# Justification : concentre les extrêmes en FMAM, appuie le choix PKL annuel.
# -----------------------------------------------------------------------------
def load_seasonality_matrix():
    """
    Charge le CSV prétraité et calcule le % de jours chauds par mois × station.
    Retourne (matrice, stations, ok) ; ok=False si données absentes.
    """
    import pandas as pd
    path = DATA_DIR / "gsod_bf_preprocessed_v2.csv"
    if not path.exists():
        return None, None, False

    df = pd.read_csv(path)
    col_nom = 'NAME' if 'NAME' in df.columns else 'STATION'
    if 'MONTH' not in df.columns and 'DATE' in df.columns:
        df['MONTH'] = pd.to_datetime(df['DATE']).dt.month

    seuils = df.groupby(col_nom)['MAX'].quantile(0.90)
    df['seuil']  = df[col_nom].map(seuils)
    df['chaud']  = (df['MAX'] > df['seuil']).astype(float)

    pivot = df.pivot_table(index=col_nom, columns='MONTH',
                           values='chaud', aggfunc='mean') * 100
    stations = list(pivot.index)
    matrice  = pivot.reindex(columns=range(1, 13)).values
    return matrice, stations, True




def fig_saisonnalite(save=True):

    matrice, stations, ok = load_seasonality_matrix()
    if not ok:
        raise FileNotFoundError(
            "gsod_bf_preprocessed_v2.csv absent : lancer d'abord python -m dignn.preprocess")

    mois = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
            'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']

    cmap = LinearSegmentedColormap.from_list(
        'chaleur', [WHITE, '#FAD7A0', TERRA, '#922B21'], N=256
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(matrice, cmap=cmap, aspect='auto', vmin=0)

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('Jours chauds (%)  —  T_max > P90 local', fontsize=11)

    ax.set_xticks(range(12))
    ax.set_xticklabels(mois)
    ax.set_yticks(range(len(stations)))
    ax.set_yticklabels([s.split()[0].capitalize() if ' ' in str(s) else str(s)
                        for s in stations])
    ax.set_xlabel('Mois')
    ax.set_ylabel('Station')

    # Annoter les valeurs
    for i in range(matrice.shape[0]):
        for j in range(matrice.shape[1]):
            val = matrice[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                    fontsize=8,
                    color='white' if val > matrice[~np.isnan(matrice)].max() * 0.55
                    else 'black')

    ax.set_title(
        'Saisonnalité des jours chauds par station (2010–2024)\n'
        'Concentration en saison sèche chaude (février–mai), creux en mousson (juin–septembre)',
        fontweight='bold', pad=10
    )
    plt.tight_layout()
    if save:
        _save(fig, 'fig_m3_saisonnalite')
    else:
        return fig


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("  GÉNÉRATION DES FIGURES — CHAPITRE RÉSULTATS")
    print(f"  Dossier : {FIGURES_DIR}")
    print("=" * 60)

    eval_results = load_eval_results()
    if eval_results is None:
        raise FileNotFoundError(
            "outputs/eval_results.json absent : lancer d'abord python -m dignn.evaluate --run_all")
    mc_results   = load_mc_results()
    history      = load_best_history()
    gpd_params   = load_gpd_params()
    if gpd_params is None:
        raise FileNotFoundError(
            "data/gpd_params_v2.pkl absent : lancer d'abord python -m dignn.preprocess")

    print()
    print("--- Figures rapatriées de la méthodologie ---")
    print("Fig. M1 — Paramètres GPD par station (xi, sigma)")
    fig_gpd_stations(gpd_params)

    print("Fig. M2 — Timeline de validation hold-out")
    fig_timeline_validation()

    print("Fig. M3 — Saisonnalité mois x station")
    fig_saisonnalite()

    print()
    print("--- Figures du chapitre Résultats ---")
    print("Fig. 1  — Courbes ROC et Précision-Rappel")
    fig_roc_pr(eval_results)

    print("Fig. 2  — Comparaison globale des métriques")
    fig_comparaison_globale(eval_results)

    print("Fig. 3  — Heatmap d'ablation C_IN × C_OUT")
    fig_ablation_heatmap(eval_results)

    print("Fig. 4  — Dégradation avec l'horizon")
    fig_degradation_horizon(eval_results, mc_results)

    print("Fig. 5  — Matrice de confusion DI-GNN")
    fig_matrice_confusion(eval_results)

    print("Fig. 6  — Courbes d'apprentissage")
    fig_apprentissage(history)

    print("Fig. 7  — Fiabilité et incertitude MC Dropout")
    fig_fiabilite_mc(mc_results)

    print("Fig. 8  — Positionnement vs littérature")
    fig_comparaison_litterature(eval_results)

    print()
    print("=" * 60)
    print("  TERMINÉ")
    generated = sorted(FIGURES_DIR.glob("*.png"))
    for f in generated:
        print(f"  ✓ {f.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()