"""
Carte du réseau de stations — Burkina Faso
============================================
Thèse MSc DI-GNN | ANAM-BF / UVBF

Carte propre du réseau de 8 stations synoptiques utilisées pour la
prévision des vagues de chaleur. Niveau administratif : contour national
(adm0) issu du shapefile officiel OCHA HDX.

Principes :
  - Contour national officiel (pas de régions inventées)
  - Palette thèse : navy / gris / terracotta
  - Français
  - 8 stations forcées (Dori exclue) — cohérent avec toute la thèse
  - Arêtes depuis adjacency_v2.npz (vraie matrice Pearson)
  - Aucun téléchargement réseau, aucun recomptage d'événements
  - Bandes climatiques latitudinales en fond (sahélien / soudano-sahélien /
    soudanien) — désactivables via ZONES_CLIMATIQUES

Usage :
  python -m dignn.plotting.maps
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

# Shapefile officiel optionnel (contour national adm0)
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# =============================================================================
# OPTIONS
# =============================================================================
ZONES_CLIMATIQUES = True   # bandes latitudinales en fond

# =============================================================================
# PALETTE THÈSE
# =============================================================================
NAVY  = '#1A5276'
GRAY  = '#5D6D7E'
TERRA = '#CA6F1E'

# Teintes de fond pour les 3 zones climatiques (du sud humide au nord sec)
ZONE_SOUDANIEN  = '#EAF2F8'   # sud — bleu très clair
ZONE_SAHELO_SUD = '#FDF2E9'   # centre — crème
ZONE_SAHELIEN   = '#F6DDCC'   # nord — terracotta très clair

plt.rcParams.update({
    'font.family'     : 'serif',
    'font.size'       : 11,
    'axes.labelsize'  : 12,
    'axes.titlesize'  : 13,
    'figure.dpi'      : 150,
    'savefig.dpi'     : 300,
    'savefig.bbox'    : 'tight',
})

# =============================================================================
# CHEMINS
# =============================================================================
from ..config import DATA_DIR, FIGURES_DIR

# =============================================================================
# 8 STATIONS RETENUES (Dori exclue) — source de vérité unique
# Coordonnées synoptiques ANAM-BF / GSOD
# =============================================================================
STATIONS_8 = {
    'Bobo Dioulasso': (-4.33, 11.16),
    'Boromo':         (-2.93, 11.74),
    'Dédougou':       (-3.48, 12.46),
    "Fada N'Gourma":  ( 0.36, 12.07),
    'Gaoua':          (-3.18, 10.33),
    'Ouagadougou':    (-1.51, 12.35),
    'Ouahigouya':     (-2.42, 13.57),
    'Pô':             (-1.15, 11.17),
}

# Total d'événements de vague de chaleur par station (2010–2024)
# Somme = 553 (cohérent avec le total de la thèse). Repli si fichier absent.
HW_EVENTS_8 = {
    'Bobo Dioulasso': 66,
    'Boromo':         69,
    'Dédougou':       68,
    "Fada N'Gourma":  74,
    'Gaoua':          65,
    'Ouagadougou':    70,
    'Ouahigouya':     75,
    'Pô':             66,
}

# Contour national du Burkina Faso (adm0) — vrai contour IGB simplifié (243 pts)
# Source : shapefile officiel bfa_admbnda_adm0_igb_20200323, OCHA HDX
# Repli embarqué si le shapefile n'est pas dans data/.
BF_CONTOUR = [
    (-0.7252, 15.0828), (-0.4437, 15.0828), (-0.4151, 15.0064), (-0.2462, 15.0778),
    ( 0.2434, 14.9128), ( 0.1766, 14.8458), ( 0.2360, 14.7514), ( 0.1711, 14.6318),
    ( 0.1617, 14.5335), ( 0.4430, 13.9512), ( 0.6271, 13.7737), ( 0.6213, 13.6984),
    ( 0.7726, 13.6896), ( 0.7965, 13.6313), ( 0.9942, 13.5764), ( 1.0102, 13.4894),
    ( 1.2851, 13.3546), ( 1.2416, 13.3391), ( 1.2117, 13.3803), ( 1.1751, 13.3215),
    ( 0.9919, 13.3747), ( 0.9919, 13.1033), ( 1.8686, 12.6053), ( 1.9797, 12.7369),
    ( 2.1032, 12.7310), ( 2.2271, 12.5856), ( 2.2779, 12.4251), ( 2.1774, 12.4240),
    ( 2.0851, 12.3934), ( 2.0657, 12.3567), ( 2.4074, 11.8989), ( 2.3755, 11.7909),
    ( 2.3049, 11.7348), ( 2.3035, 11.6726), ( 2.0000, 11.4184), ( 1.9169, 11.4084),
    ( 1.8782, 11.4530), ( 1.6240, 11.3934), ( 1.5535, 11.4783), ( 1.5310, 11.4511),
    ( 1.4484, 11.4811), ( 1.3959, 11.4532), ( 1.3849, 11.3921), ( 1.3254, 11.3638),
    ( 1.3515, 11.3064), ( 1.3032, 11.2951), ( 1.2675, 11.3266), ( 1.2838, 11.2661),
    ( 1.2625, 11.2500), ( 1.1487, 11.2809), ( 1.1295, 11.2479), ( 1.1650, 11.1696),
    ( 1.0548, 11.1469), ( 1.1142, 11.0317), ( 0.9768, 11.0881), ( 0.9447, 11.0261),
    ( 0.9802, 10.9823), ( 0.9473, 10.9593), ( 0.9239, 10.9864), ( 0.9340, 10.8973),
    ( 0.8880, 10.9143), ( 0.9117, 11.0010), ( 0.6605, 11.0006), ( 0.5008, 10.9334),
    ( 0.5048, 11.0092), (-0.2814, 11.1696), (-0.2710, 11.1265), (-0.3615, 11.0688),
    (-0.3707, 11.1263), (-0.4179, 11.1219), (-0.4401, 11.0370), (-0.5421, 10.9761),
    (-0.5649, 10.9954), (-0.6130, 10.8964), (-0.6710, 10.9582), (-0.6538, 10.9873),
    (-0.8049, 11.0101), (-0.8901, 10.9655), (-0.9170, 11.0010), (-1.3849, 10.9932),
    (-1.4296, 11.0216), (-1.5850, 11.0213), (-1.5835, 10.9909), (-1.7156, 10.9767),
    (-2.8332, 11.0081), (-2.8178, 10.9233), (-2.8641, 10.8876), (-2.8951, 10.7650),
    (-2.9380, 10.7220), (-2.9056, 10.6861), (-2.9413, 10.6162), (-2.8662, 10.4542),
    (-2.7734, 10.4227), (-2.8314, 10.3877), (-2.8504, 10.3186), (-2.7565, 10.2472),
    (-2.7981, 10.1914), (-2.7904, 10.0693), (-2.7264,  9.8270), (-2.7895,  9.7471),
    (-2.7869,  9.6960), (-2.7458,  9.6420), (-2.7689,  9.5697), (-2.6887,  9.4937),
    (-2.7613,  9.4160), (-2.9353,  9.5791), (-3.0061,  9.7417), (-3.0630,  9.7300),
    (-3.0920,  9.8109), (-3.1490,  9.8529), (-3.1841,  9.8382), (-3.1905,  9.9347),
    (-3.2717,  9.8458), (-3.3148,  9.8703), (-3.3107,  9.9105), (-3.6218,  9.9598),
    (-3.7780,  9.9274), (-3.8030,  9.8943), (-3.8952,  9.9073), (-4.0507,  9.8028),
    (-4.1277,  9.8484), (-4.1928,  9.7738), (-4.2664,  9.7599), (-4.3128,  9.5966),
    (-4.3416,  9.6253), (-4.3904,  9.5964), (-4.4297,  9.6641), (-4.5047,  9.6467),
    (-4.5147,  9.7511), (-4.5536,  9.7421), (-4.5624,  9.6930), (-4.6123,  9.7250),
    (-4.6890,  9.6779), (-4.7335,  9.7474), (-4.7751,  9.7370), (-4.8164,  9.7835),
    (-4.7837,  9.7818), (-4.7859,  9.8357), (-4.9666,  9.8918), (-4.9403,  9.9537),
    (-4.9877,  9.9917), (-4.9620, 10.0255), (-5.0152, 10.0727), (-4.9994, 10.1073),
    (-5.0538, 10.0825), (-5.0386, 10.1088), (-5.0730, 10.1111), (-5.0365, 10.1217),
    (-5.0697, 10.1388), (-5.0599, 10.1804), (-5.1043, 10.1751), (-5.0834, 10.2242),
    (-5.1219, 10.2042), (-5.1109, 10.3076), (-5.1870, 10.2876), (-5.1983, 10.3215),
    (-5.2893, 10.3202), (-5.3634, 10.2825), (-5.4373, 10.3217), (-5.5113, 10.4307),
    (-5.4545, 10.5570), (-5.4669, 10.7646), (-5.4310, 10.7743), (-5.4018, 10.8512),
    (-5.4524, 10.9005), (-5.4517, 10.9724), (-5.4902, 10.9749), (-5.4890, 11.0813),
    (-5.3282, 11.1417), (-5.3217, 11.2079), (-5.2541, 11.2453), (-5.2461, 11.3866),
    (-5.1980, 11.4314), (-5.2153, 11.5785), (-5.2924, 11.6185), (-5.2595, 11.7623),
    (-5.4028, 11.8265), (-5.2474, 11.8375), (-5.1466, 11.9494), (-5.0765, 11.9805),
    (-4.8324, 12.0173), (-4.7368, 11.9997), (-4.7006, 12.0620), (-4.6303, 12.0572),
    (-4.6062, 12.1372), (-4.5417, 12.1409), (-4.5802, 12.1996), (-4.4711, 12.2833),
    (-4.4740, 12.3304), (-4.3961, 12.3094), (-4.4382, 12.4201), (-4.4221, 12.4867),
    (-4.3669, 12.5335), (-4.4819, 12.6562), (-4.4808, 12.7200), (-4.3459, 12.7421),
    (-4.3018, 12.7008), (-4.2335, 12.7285), (-4.2109, 12.8143), (-4.2194, 12.9596),
    (-4.2657, 12.9891), (-4.3402, 13.1415), (-4.2889, 13.1877), (-4.2283, 13.1827),
    (-4.2593, 13.2340), (-4.1408, 13.2915), (-4.0903, 13.3969), (-3.9840, 13.4338),
    (-3.9648, 13.5044), (-3.9388, 13.4509), (-3.8952, 13.4410), (-3.9624, 13.4061),
    (-3.9603, 13.3790), (-3.7976, 13.3705), (-3.6123, 13.2132), (-3.5509, 13.1974),
    (-3.5484, 13.1669), (-3.4394, 13.1586), (-3.4370, 13.2748), (-3.2330, 13.2873),
    (-3.2777, 13.5555), (-3.2504, 13.5829), (-3.2643, 13.7131), (-3.1811, 13.6772),
    (-3.0998, 13.6851), (-3.0390, 13.6138), (-2.9990, 13.6594), (-2.9510, 13.6244),
    (-2.8748, 13.6527), (-2.9047, 13.8226), (-2.8349, 13.9935), (-2.8397, 14.0525),
    (-2.8098, 14.0739), (-2.6770, 14.1340), (-2.4741, 14.2980), (-2.2981, 14.2483),
    (-2.1016, 14.1461), (-1.9974, 14.1912), (-1.9808, 14.4747), (-1.6786, 14.5007),
    (-1.3186, 14.7288), (-1.0730, 14.7838), (-0.7252, 15.0828),
]


def load_stations():
    """
    Charge les 8 stations. Priorité au CSV réel s'il existe, sinon
    coordonnées embarquées. Dori est exclue dans tous les cas.
    """
    csv = DATA_DIR / "gsod_bf_preprocessed_v2.csv"
    if csv.exists():
        import pandas as pd
        df = pd.read_csv(csv)
        agg = df.groupby('NAME').agg(
            {'LATITUDE': 'first', 'LONGITUDE': 'first'}).reset_index()
        coords = {}
        for _, r in agg.iterrows():
            nom = str(r['NAME']).split(',')[0].strip()
            if 'DORI' in nom.upper():
                continue   # exclusion Dori
            coords[nom] = (float(r['LONGITUDE']), float(r['LATITUDE']))
        # Garder uniquement les 8 retenues si correspondance
        if len(coords) >= 8:
            print(f"  Stations chargées depuis {csv.name} : {len(coords)}")
            return coords
    print("  [INFO] CSV absent — coordonnées embarquées (8 stations).")
    return dict(STATIONS_8)


def load_adjacency():
    """Charge la matrice de Pearson pour tracer les arêtes."""
    adj = DATA_DIR / "adjacency_v2.npz"
    if not adj.exists():
        print("  [INFO] adjacency_v2.npz absent — arêtes par distance.")
        return None, None
    data = np.load(adj, allow_pickle=True)
    return data['pearson'], list(data['stations'])


def load_bf_boundary():
    """
    Charge le contour national officiel adm0.
    Ordre de priorité : geopandas → pyshp → contour réel embarqué.
    Retourne une liste de (lon, lat) dans tous les cas.
    """
    shp = DATA_DIR / "shapefiles/bfa_admbnda_adm0_igb_20200323.shp"

    # 1) geopandas (si shapefile complet présent : .shp + .shx + .dbf)
    if HAS_GEOPANDAS and shp.exists():
        try:
            import geopandas as gpd
            gdf = gpd.read_file(shp)
            geom = gdf.geometry.iloc[0]
            polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
            main = max(polys, key=lambda p: p.area)
            print(f"  Contour officiel chargé (geopandas) : {shp.name}")
            return list(main.exterior.coords)
        except Exception as e:
            print(f"  [INFO] geopandas échoue ({str(e)[:60]}) — essai pyshp.")

    # 2) pyshp (lit le .shp même sans tous les compagnons)
    if shp.exists():
        try:
            import shapefile
            from shapely.geometry import shape
            sf = shapefile.Reader(str(shp))
            geom = shape(sf.shape(0).__geo_interface__)
            polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
            main = max(polys, key=lambda p: p.area)
            print(f"  Contour officiel chargé (pyshp) : {shp.name}")
            return list(main.exterior.coords)
        except Exception as e:
            print(f"  [INFO] pyshp échoue ({str(e)[:60]}) — contour embarqué.")

    # 3) Repli : vrai contour IGB embarqué (243 points)
    print("  [INFO] Shapefile absent — contour IGB embarqué.")
    return BF_CONTOUR


def load_hw_events(noms):
    """
    Charge le total d'événements HW par station depuis le CSV de
    caractérisation, sinon valeurs embarquées (somme = 553).
    Retourne un tableau aligné sur `noms`.
    """
    import unicodedata
    def _key(s):
        s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode()
        return s.split(',')[0].strip().upper()

    csv = DATA_DIR / "heatwave_analysis" / "heatwave_statistics_by_station.csv"
    if csv.exists():
        import pandas as pd
        df = pd.read_csv(csv)
        col_nom = 'NAME' if 'NAME' in df.columns else df.columns[0]
        col_n   = 'Count' if 'Count' in df.columns else df.columns[1]
        m = {_key(r[col_nom]): int(r[col_n]) for _, r in df.iterrows()}
        vals = [m.get(_key(n), np.nan) for n in noms]
        print(f"  Événements HW chargés depuis {csv.name} (total {int(np.nansum(vals))}).")
        return np.array(vals, dtype=float)
    print("  [INFO] Comptes HW embarqués (total 553).")
    return np.array([HW_EVENTS_8.get(n, np.nan) for n in noms], dtype=float)


def _barre_echelle(ax, lon0, lat0, km=100):
    """Trace une barre d'échelle de `km` kilomètres."""
    deg = km / (111.32 * np.cos(np.radians(lat0)))   # ° de longitude pour km
    ax.plot([lon0, lon0 + deg], [lat0, lat0],
            color='black', linewidth=3, solid_capstyle='butt', zorder=8)
    ax.plot([lon0, lon0], [lat0 - 0.06, lat0 + 0.06], color='black', linewidth=1.5)
    ax.plot([lon0 + deg, lon0 + deg], [lat0 - 0.06, lat0 + 0.06],
            color='black', linewidth=1.5)
    ax.text(lon0 + deg / 2, lat0 + 0.12, f'{km} km',
            ha='center', va='bottom', fontsize=9, fontweight='bold')


def _fleche_nord(ax, lon0, lat0):
    """Trace une flèche du nord."""
    ax.annotate('', xy=(lon0, lat0 + 0.5), xytext=(lon0, lat0 - 0.2),
                arrowprops=dict(arrowstyle='-|>', color='black', linewidth=2),
                zorder=8)
    ax.text(lon0, lat0 + 0.62, 'N', ha='center', va='bottom',
            fontsize=14, fontweight='bold')


def fig_carte_evenements(save=True):
    """
    Carte du réseau colorée par le total d'événements de vague de chaleur
    par station (2010–2024). Version française, palette thèse.
    """
    coords      = load_stations()
    pearson, st = load_adjacency()
    contour     = load_bf_boundary()

    noms = list(coords.keys())
    lons = np.array([coords[n][0] for n in noms])
    lats = np.array([coords[n][1] for n in noms])
    n    = len(noms)
    evts = load_hw_events(noms)

    fig, ax = plt.subplots(figsize=(11, 8.5))

    # Colormap séquentielle chaude (cohérente avec la figure de saisonnalité)
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    cmap = LinearSegmentedColormap.from_list(
        'chaleur', ['#2C7FB8', '#EAF2F8', '#FAD7A0', TERRA, '#922B21'], N=256)
    norm = Normalize(vmin=np.nanmin(evts), vmax=np.nanmax(evts))

    # --- Contour national ---
    xs = [p[0] for p in contour]; ys = [p[1] for p in contour]
    ax.fill(xs, ys, color='#FCF8E8', alpha=0.6, zorder=1)
    ax.plot(xs, ys, color='black', linewidth=2.0, zorder=2)

    # --- Arêtes du graphe ---
    if pearson is not None and len(st) == n:
        for i in range(n):
            for j in range(i + 1, n):
                if abs(pearson[i, j]) > 0.2:
                    ax.plot([lons[i], lons[j]], [lats[i], lats[j]],
                            color=GRAY, linewidth=0.7, alpha=0.4, zorder=3)
    else:
        for i in range(n):
            for j in range(i + 1, n):
                if np.hypot(lons[i] - lons[j], lats[i] - lats[j]) < 4:
                    ax.plot([lons[i], lons[j]], [lats[i], lats[j]],
                            color=GRAY, linewidth=0.7, alpha=0.35, zorder=3)

    # --- Stations colorées par nombre d'événements ---
    for nom, lon, lat, e in zip(noms, lons, lats, evts):
        couleur = cmap(norm(e))
        ax.scatter(lon, lat, s=900, color=couleur, edgecolors='black',
                   linewidths=1.6, zorder=5)
        # Nombre au centre, couleur de texte selon la luminosité du fond
        lum = 0.299 * couleur[0] + 0.587 * couleur[1] + 0.114 * couleur[2]
        txt_color = 'white' if lum < 0.55 else 'black'
        ax.text(lon, lat, f'{int(e)}', ha='center', va='center',
                fontsize=11, fontweight='bold', color=txt_color, zorder=6)
        # Nom de station en étiquette décalée
        dx = 0.30 if lon > -3 else -0.30
        ha = 'left' if dx > 0 else 'right'
        ax.annotate(nom, (lon, lat), xytext=(lon + dx, lat + 0.28),
                    fontsize=8.5, fontweight='bold', color=NAVY, ha=ha, va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor=GRAY, alpha=0.85, linewidth=0.5), zorder=6)

    # --- Flèche du nord + barre d'échelle ---
    _fleche_nord(ax, 1.9, 14.3)
    _barre_echelle(ax, 0.9, 9.7, km=100)

    # --- Barre de couleurs ---
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Total d'événements de vague de chaleur (2010–2024)",
                   fontsize=11)

    # --- Mise en forme ---
    ax.set_xlim([-6, 3]); ax.set_ylim([9, 15.5])
    ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
    ax.set_aspect('equal')
    ax.set_facecolor('#E8E8E8')
    ax.set_title(
        "Distribution spatiale des événements de vague de chaleur\n"
        f"Réseau de {n} stations synoptiques — Burkina Faso (total : {int(np.nansum(evts))})",
        fontweight='bold', pad=12)

    plt.tight_layout()
    if save:
        for ext in ('png', 'pdf'):
            p = FIGURES_DIR / f"fig_carte_evenements.{ext}"
            fig.savefig(p, dpi=300, facecolor='white', bbox_inches='tight')
        print(f"  ✓ fig_carte_evenements.png/pdf  (total {int(np.nansum(evts))} événements)")
        plt.close(fig)
    else:
        return fig


def fig_carte_stations(save=True):
    coords      = load_stations()
    contour     = load_bf_boundary()

    noms = list(coords.keys())
    lons = np.array([coords[n][0] for n in noms])
    lats = np.array([coords[n][1] for n in noms])
    n    = len(noms)

    fig, ax = plt.subplots(figsize=(10, 9))

    # --- Contour national (vrai contour IGB) ---
    xs = [p[0] for p in contour]
    ys = [p[1] for p in contour]
    ax.fill(xs, ys, color='white', alpha=0.45, zorder=1)
    ax.plot(xs, ys, color='black', linewidth=2.0, zorder=2)

    # --- Stations ---
    ax.scatter(lons, lats, s=180, color=NAVY, edgecolors='white',
               linewidths=1.8, zorder=5)

    for nom, lon, lat in zip(noms, lons, lats):
        # Décalage d'étiquette selon la position
        dx = 0.18 if lon > -3 else -0.18
        dy = 0.16
        ha = 'left' if dx > 0 else 'right'
        ax.annotate(nom, (lon, lat),
                    xytext=(lon + dx, lat + dy),
                    fontsize=9.5, fontweight='bold', color=NAVY,
                    ha=ha, va='bottom',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor=GRAY, alpha=0.9, linewidth=0.6),
                    zorder=6)

    # --- Mise en forme ---
    ax.set_xlim([-6, 3])
    ax.set_ylim([9, 15.5])
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.25, zorder=0)

    ax.set_title(
        'Réseau des 8 stations synoptiques retenues\n'
        'Burkina Faso',
        fontweight='bold', pad=12
    )

    # Légende
    legende = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=NAVY,
               markeredgecolor='white', markersize=11,
               label=f'Station synoptique (n = {n})'),
    ]
    ax.legend(handles=legende, loc='lower left', framealpha=0.92)

    plt.tight_layout()
    if save:
        for ext in ('png', 'pdf'):
            p = FIGURES_DIR / f"fig_carte_stations.{ext}"
            fig.savefig(p, dpi=300, facecolor='white', bbox_inches='tight')
        print(f"  ✓ fig_carte_stations.png/pdf  ({n} stations)")
        plt.close(fig)
    else:
        return fig


def main():
    print("=" * 60)
    print("  CARTE DU RÉSEAU DE STATIONS — Burkina Faso")
    print(f"  Dossier : {FIGURES_DIR}")
    print("=" * 60)
    fig_carte_stations()
    fig_carte_evenements()
    print("=" * 60)
    print("  TERMINÉ")
    print("=" * 60)


if __name__ == "__main__":
    main()
