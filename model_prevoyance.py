"""
============================================================
MODÈLE DE TARIFICATION — PRÉVOYANCE COLLECTIVE ENTREPRISE
Paméla Fagla | M1 Actuariat ISUP, Sorbonne Université
============================================================

Couvertures modélisées :
  - Garantie Décès : capital = k × salaire annuel brut
  - Garantie IJ (Incapacité Temporaire de Travail) : indemnités journalières
  - Garantie Invalidité Permanente Totale : rente jusqu'à 65 ans

Tables actuarielles :
  - Mortalité : TD 88-90 (Tables de Décès françaises, référence réglementaire)
  - Morbidité : hypothèses simplifiées inspirées des données BCAC/CNAM

Méthodologie :
  - Prime pure = espérance des prestations futures
  - Prime commerciale = prime pure / (1 - taux de chargement)
  - Compte de résultat = primes acquises - sinistres réglés
  - Ratio S/P = indicateur de rentabilité du régime
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# ============================================================
# 1. TABLES ACTUARIELLES
# ============================================================

# --- 1.1 Table de mortalité TD 88-90 ---
# Source : Institut des Actuaires Français / BCAC
# qx = probabilité de décès dans l'année à l'âge exact x
# Valeurs aux âges pivots, interpolation spline cubique entre les âges

_AGES_PIVOTS = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

_QX_H_PIVOTS = [
    0.00113, 0.00121, 0.00135, 0.00177, 0.00260,
    0.00393, 0.00620, 0.00982, 0.01573, 0.02518, 0.04045
]

_QX_F_PIVOTS = [
    0.00043, 0.00047, 0.00057, 0.00079, 0.00120,
    0.00189, 0.00314, 0.00500, 0.00812, 0.01325, 0.02168
]

def build_mortality_table(age_min=20, age_max=70):
    """
    Construit la table de mortalité TD 88-90 pour les âges [age_min, age_max].
    Interpolation par spline cubique entre les valeurs aux âges pivots.
    
    Retourne un DataFrame avec colonnes : age, qx_homme, qx_femme
    """
    spline_h = CubicSpline(_AGES_PIVOTS, _QX_H_PIVOTS)
    spline_f = CubicSpline(_AGES_PIVOTS, _QX_F_PIVOTS)
    
    ages = np.arange(age_min, age_max + 1)
    return pd.DataFrame({
        'age':      ages,
        'qx_homme': np.clip(spline_h(ages), 1e-5, 0.5),
        'qx_femme': np.clip(spline_f(ages), 1e-5, 0.5)
    }).set_index('age')


# --- 1.2 Tables de morbidité : incapacité et invalidité ---
# 
# Hypothèses par tranche d'âge (inspirées des statistiques DREES/CNAM) :
#
# p_arret   : probabilité d'au moins un arrêt de travail dans l'année
#             (après franchise contractuelle de 8 jours)
# d_ij      : durée moyenne nette d'un arrêt indemnisé (jours, hors franchise)
# p_inv     : probabilité de basculer en invalidité 2ème/3ème catégorie
#
# Note : ces taux sont des hypothèses simplifiées.
# En pratique, les assureurs calibrent sur leur propre expérience de portefeuille.

_MORBIDITE = [
    # (age_min, age_max, p_arret, d_ij_jours, p_inv)
    #
    # p_arret : P(arrêt indemnisé dans l'année, après franchise 8j)
    #   Calibré sur les statistiques CNAM (Tableaux de bord arrêts de travail 2022)
    #   pour la population salariée du secteur privé, après exclusion des très courts arrêts
    #
    # d_ij    : durée moyenne nette d'arrêt indemnisé (jours) — au-delà de la franchise
    #   Reflète les durées moyennes observées par tranche d'âge dans les statistiques DREES
    #
    # p_inv   : P(nouveaux cas d'invalidité 2ème/3ème catégorie dans l'année)
    #   Calibré sur les données CNAM — taux d'incidence de nouvelles rentes d'invalidité
    #   par tranche d'âge pour 1 000 assurés
    #
    (22, 29, 0.08,  10, 0.0005),   # Jeunes actifs : peu d'arrêts longs, invalidité rare
    (30, 39, 0.10,  14, 0.0010),   # Montée progressive des TMS et pathologies
    (40, 49, 0.14,  20, 0.0025),   # Pic des pathologies de l'appareil locomoteur
    (50, 59, 0.19,  28, 0.0060),   # Forte hausse des maladies chroniques
    (60, 65, 0.24,  38, 0.0085),   # Maladies cardiovasculaires, cancers, multi-pathologies
]

def get_morbidite(age):
    """
    Retourne (p_arret, d_ij, p_inv) pour un âge donné.
    """
    for amin, amax, p_arret, d_ij, p_inv in _MORBIDITE:
        if amin <= age <= amax:
            return p_arret, d_ij, p_inv
    return 0.25, 30, 0.005  # valeur par défaut


def annuite_certaine(n, i=0.035):
    """
    Valeur actualisée d'une rente certaine de 1€/an pendant n années.
    Formule : ä_n = (1 - v^n) / i,  v = 1 / (1 + i)
    
    Utilisée pour valoriser la rente d'invalidité jusqu'à 65 ans.
    Taux technique i = 3.5% (proche du RFR EIOPA sur la zone euro).
    
    Paramètres
    ----------
    n : int - durée en années
    i : float - taux d'actualisation (défaut 3.5%)
    """
    if n <= 0:
        return 0.5  # engagement résiduel minimal pour les proches de 65 ans
    v = 1.0 / (1.0 + i)
    return (1.0 - v ** n) / i


# ============================================================
# 2. SIMULATION DU PORTEFEUILLE ENTREPRISE
# ============================================================

def simulate_portfolio(
    n_salaries=500,
    age_mean=40, age_std=10, age_min=22, age_max=62,
    pct_hommes=0.60,
    salaire_median=3200, salaire_sigma=0.35,
    salaire_min=1800, salaire_max=12000,
    seed=42
):
    """
    Simule un portefeuille d'entreprise pour la tarification prévoyance.
    
    Distribution des âges    : normale tronquée (μ=40, σ=10, [22, 62])
    Distribution des salaires : log-normale (médiane ≈ 3 200 €/mois brut)
    Sexe                      : 60% hommes / 40% femmes (ratio secteur services)
    
    Retourne un DataFrame avec : id, age, sexe, salaire_mensuel, salaire_annuel,
                                  salaire_journalier
    """
    rng = np.random.default_rng(seed)
    
    ages = rng.normal(age_mean, age_std, n_salaries)
    ages = np.clip(ages, age_min, age_max).astype(int)
    
    sexes = rng.choice(['H', 'F'], size=n_salaries, p=[pct_hommes, 1 - pct_hommes])
    
    salaires_m = rng.lognormal(mean=np.log(salaire_median), sigma=salaire_sigma, size=n_salaries)
    salaires_m = np.clip(salaires_m, salaire_min, salaire_max)
    
    return pd.DataFrame({
        'id':                range(1, n_salaries + 1),
        'age':               ages,
        'sexe':              sexes,
        'salaire_mensuel':   salaires_m,
        'salaire_annuel':    salaires_m * 12,
        'salaire_journalier': salaires_m * 12 / 365,
    })


# ============================================================
# 3. TARIFICATION
# ============================================================

def tarifier_portefeuille(
    df_port,
    table_mortalite=None,
    capital_mult=3,           # Capital décès = 3 × salaire annuel
    taux_remplacement_ij=0.70,# IJ = 70% du salaire journalier brut
    rente_invalidite_mult=0.80,# Rente = 80% du salaire annuel
    taux_tech=0.035,
    loading_deces=0.12,       # Chargement garantie décès : 12%
    loading_ij=0.18,          # Chargement IJ : 18%
    loading_inv=0.20          # Chargement invalidité : 20%
):
    """
    Calcule les primes pures et commerciales pour chaque salarié du portefeuille.
    
    GARANTIE DÉCÈS
    --------------
    Capital assuré = capital_mult × salaire_annuel
    Prime pure     = qx × capital_assuré
    Logique : qx = P(décès dans l'année), le coût attendu est qx × capital.
    
    GARANTIE IJ (Incapacité Temporaire de Travail)
    -----------------------------------------------
    IJ journalière = taux_remplacement × salaire_journalier
    Prime pure     = p_arret × durée_nette × IJ_journalière
    Franchise contractuelle : 8 jours (non indemnisés).
    
    GARANTIE INVALIDITÉ (IPT 2ème / 3ème catégorie)
    -------------------------------------------------
    Rente annuelle = rente_mult × salaire_annuel
    Prime pure     = p_inv × rente_annuelle × ä_{65-x}
    où ä_{65-x} = valeur actualisée d'une rente certaine jusqu'à 65 ans
    
    PRIME COMMERCIALE
    -----------------
    pc = pp / (1 - loading)
    Le chargement couvre : frais de gestion, frais d'acquisition, marge technique.
    """
    if table_mortalite is None:
        table_mortalite = build_mortality_table()
    
    df = df_port.copy()
    
    # --- Garantie Décès ---
    def _qx(row):
        col = 'qx_homme' if row['sexe'] == 'H' else 'qx_femme'
        age = max(20, min(70, row['age']))
        return table_mortalite.loc[age, col]
    
    df['qx']          = df.apply(_qx, axis=1)
    df['capital_deces']= capital_mult * df['salaire_annuel']
    df['pp_deces']    = df['qx'] * df['capital_deces']
    df['pc_deces']    = df['pp_deces'] / (1 - loading_deces)
    
    # --- Garantie IJ ---
    morbidite_vals = df['age'].apply(get_morbidite)
    df['p_arret']     = morbidite_vals.apply(lambda x: x[0])
    df['d_ij_jours']  = morbidite_vals.apply(lambda x: x[1])
    df['ij_journaliere'] = taux_remplacement_ij * df['salaire_journalier']
    df['pp_ij']       = df['p_arret'] * df['d_ij_jours'] * df['ij_journaliere']
    df['pc_ij']       = df['pp_ij'] / (1 - loading_ij)
    
    # --- Garantie Invalidité ---
    df['p_inv']       = morbidite_vals.apply(lambda x: x[2])
    df['rente_annuelle'] = rente_invalidite_mult * df['salaire_annuel']
    df['annuite_inv'] = df['age'].apply(lambda a: annuite_certaine(65 - a, i=taux_tech))
    df['pp_invalidite'] = df['p_inv'] * df['rente_annuelle'] * df['annuite_inv']
    df['pc_invalidite'] = df['pp_invalidite'] / (1 - loading_inv)
    
    # --- Totaux ---
    df['pp_totale']   = df['pp_deces'] + df['pp_ij'] + df['pp_invalidite']
    df['pc_totale']   = df['pc_deces'] + df['pc_ij'] + df['pc_invalidite']
    df['taux_cotisation_pct'] = df['pc_totale'] / df['salaire_annuel'] * 100
    
    return df


# ============================================================
# 4. SIMULATION DES SINISTRES (EXERCICE ANNUEL)
# ============================================================

def simulate_sinistres(df_tarifie, table_mortalite=None, seed=2026):
    """
    Simule la sinistralité réelle d'un exercice annuel.
    
    Pour chaque salarié, on tire :
    - Bernoulli(qx)         → décès ou non
    - Bernoulli(p_arret)    → arrêt de travail ou non
    - Exponentielle(d_ij)   → durée de l'arrêt si arrêt
    - Bernoulli(p_inv)      → invalidité ou non
    
    Les sinistres décès et invalidité sont des événements rares —
    sur 500 salariés, on s'attend à 2-4 décès et 0-2 nouvelles invalidi.
    Les sinistres IJ sont plus fréquents (effet de fréquence).
    """
    if table_mortalite is None:
        table_mortalite = build_mortality_table()
    
    rng = np.random.default_rng(seed)
    df = df_tarifie.copy()
    
    # Décès
    df['sinistre_deces'] = (
        rng.binomial(1, df['qx'].values) * df['capital_deces']
    )
    
    # IJ — durée simulée par loi Gamma (shape=3 → CV=0.58, moins de queues lourdes)
    # Cap à 75 jours (au-delà : reclassification en invalidité ou consolidation)
    GAMMA_SHAPE = 3.0
    arrets = rng.binomial(1, df['p_arret'].values).astype(bool)
    scales = df['d_ij_jours'].values / GAMMA_SHAPE   # scale = mean / shape
    durees = np.where(
        arrets,
        np.clip(rng.gamma(shape=GAMMA_SHAPE, scale=scales), 0, 75.0),
        0.0
    )
    df['sinistre_ij'] = durees * df['ij_journaliere']
    
    # Invalidité
    df['sinistre_invalidite'] = (
        rng.binomial(1, df['p_inv'].values) *
        df['rente_annuelle'] *
        df['annuite_inv']
    )
    
    df['sinistre_total'] = (
        df['sinistre_deces'] + df['sinistre_ij'] + df['sinistre_invalidite']
    )
    
    return df


# ============================================================
# 5. COMPTE DE RÉSULTAT DU RÉGIME
# ============================================================

def compte_resultat(df_sinistres):
    """
    Construit le compte de résultat technique annuel du régime.
    
    Structure :
    ┌──────────────────────────────────────────────────────┐
    │  PRODUITS : Primes commerciales acquises             │
    │  CHARGES  : Sinistres réglés                         │
    │             dont chargements (frais de gestion)      │
    ├──────────────────────────────────────────────────────┤
    │  RÉSULTAT TECHNIQUE = Primes - Sinistres             │
    │  RATIO S/P = Sinistres / Primes                      │
    │             Cible saine : S/P < 75-80%               │
    └──────────────────────────────────────────────────────┘
    
    Le ratio S/P est l'indicateur clé de rentabilité du régime.
    Un S/P > 100% signifie que l'assureur paie plus qu'il ne perçoit
    → régime déficitaire, nécessite une revalorisation tarifaire.
    """
    df = df_sinistres
    
    garanties = ['deces', 'ij', 'invalidite']
    labels     = ['Décès', 'IJ', 'Invalidité']
    
    data = {}
    for gar, lab in zip(garanties, labels):
        data[lab] = {
            'primes_pures':      df[f'pp_{gar}'].sum(),
            'primes_commerciales': df[f'pc_{gar}'].sum(),
            'sinistres':         df[f'sinistre_{gar}'].sum(),
        }
        data[lab]['chargements'] = (
            data[lab]['primes_commerciales'] - data[lab]['primes_pures']
        )
        data[lab]['resultat'] = (
            data[lab]['primes_commerciales'] - data[lab]['sinistres']
        )
        data[lab]['ratio_sp'] = (
            data[lab]['sinistres'] / data[lab]['primes_commerciales']
            if data[lab]['primes_commerciales'] > 0 else 0
        )
    
    # Total
    data['Total'] = {
        'primes_pures':        sum(d['primes_pures'] for d in data.values()),
        'primes_commerciales': sum(d['primes_commerciales'] for d in data.values()),
        'sinistres':           sum(d['sinistres'] for d in data.values()),
        'chargements':         sum(d['chargements'] for d in data.values()),
        'resultat':            sum(d['resultat'] for d in data.values()),
    }
    total_pc = data['Total']['primes_commerciales']
    total_sin = data['Total']['sinistres']
    data['Total']['ratio_sp'] = total_sin / total_pc if total_pc > 0 else 0
    
    return data


# ============================================================
# 6. TARIF DE RENOUVELLEMENT
# ============================================================

def tarif_renouvellement(cr, objectif_sp=0.75, marge_securite=0.05):
    """
    Calcule le facteur d'ajustement tarifaire pour le renouvellement.
    
    Logique :
    Si le S/P réel > S/P cible, le tarif doit être relevé.
    Facteur = S/P_réel / S/P_cible × (1 + marge_sécurité)
    
    Paramètres
    ----------
    cr             : dictionnaire du compte de résultat
    objectif_sp    : ratio S/P cible (défaut 75%)
    marge_securite : marge de sécurité supplémentaire (défaut 5%)
    
    Retourne
    --------
    dict avec facteur_renouvellement et nouveau_taux_cotisation_moyen
    """
    sp_reel = cr['Total']['ratio_sp']
    
    if sp_reel <= objectif_sp:
        facteur = 1.0  # Pas d'augmentation nécessaire
        interpretation = "Tarif équilibré — pas de revalorisation requise"
    else:
        facteur = (sp_reel / objectif_sp) * (1 + marge_securite)
        interpretation = (
            f"Revalorisation tarifaire de +{(facteur - 1)*100:.1f}% recommandée "
            f"pour ramener le S/P vers {objectif_sp:.0%}"
        )
    
    return {
        'sp_reel':          sp_reel,
        'sp_cible':         objectif_sp,
        'facteur':          facteur,
        'interpretation':   interpretation,
    }


def compte_resultat_theorique(df_tarifie):
    """
    Compte de résultat THÉORIQUE : utilise les sinistres attendus (= primes pures),
    sans la variabilité d'une simulation.
    
    C'est le compte de résultat "en espérance" — ce que l'assureur espère sur le long terme.
    Il sert de référence pour la tarification et le pilotage multi-annuel.
    
    Différence avec le compte de résultat réalisé :
    - Théorique : sinistres = E[sinistres] = primes pures → S/P = (1 - loading)
    - Réalisé   : sinistres = valeurs simulées → S/P varie (risque de portefeuille)
    
    Sur un petit portefeuille (<500 salariés), l'écart peut être très important,
    notamment sur la garantie invalidité (sinistres rares mais coûteux).
    """
    df = df_tarifie
    garanties = ['deces', 'ij', 'invalidite']
    labels     = ['Décès', 'IJ', 'Invalidité']
    
    data = {}
    for gar, lab in zip(garanties, labels):
        pc  = df[f'pc_{gar}'].sum()
        pp  = df[f'pp_{gar}'].sum()   # sinistres attendus = primes pures
        data[lab] = {
            'primes_pures':        pp,
            'primes_commerciales': pc,
            'sinistres':           pp,   # théorique : sinistres = E[sinistres]
            'chargements':         pc - pp,
            'resultat':            pc - pp,
            'ratio_sp':            pp / pc if pc > 0 else 0
        }
    
    data['Total'] = {
        'primes_pures':        sum(d['primes_pures'] for d in data.values()),
        'primes_commerciales': sum(d['primes_commerciales'] for d in data.values()),
        'sinistres':           sum(d['sinistres'] for d in data.values()),
        'chargements':         sum(d['chargements'] for d in data.values()),
        'resultat':            sum(d['resultat'] for d in data.values()),
    }
    tp = data['Total']['primes_commerciales']
    ts = data['Total']['sinistres']
    data['Total']['ratio_sp'] = ts / tp if tp > 0 else 0
    
    return data


def monte_carlo_sp(df_tarifie, n_simulations=500, seed_base=0, table_mortalite=None):
    """
    Distribution du ratio S/P par simulation Monte Carlo.
    
    Sur un petit portefeuille, la sinistralité réalisée peut s'écarter fortement
    de la sinistralité attendue (loi des grands nombres peu applicable).
    
    Cette fonction simule n_simulations exercices annuels indépendants et retourne
    la distribution du S/P, permettant de quantifier l'incertitude de résultat.
    
    Retourne un tableau avec : seed, sp_total, sp_deces, sp_ij, sp_invalidite
    """
    if table_mortalite is None:
        table_mortalite = build_mortality_table()
    
    results = []
    for i in range(n_simulations):
        df_s = simulate_sinistres(df_tarifie, table_mortalite, seed=seed_base + i)
        cr_s = compte_resultat(df_s)
        results.append({
            'sim':           i,
            'sp_total':      cr_s['Total']['ratio_sp'],
            'sp_deces':      cr_s['Décès']['ratio_sp'],
            'sp_ij':         cr_s['IJ']['ratio_sp'],
            'sp_invalidite': cr_s['Invalidité']['ratio_sp'],
            'resultat':      cr_s['Total']['resultat'],
        })
    return pd.DataFrame(results)
