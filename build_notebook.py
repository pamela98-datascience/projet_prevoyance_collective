"""
Génère le notebook Jupyter prevoyance_collective.ipynb
"""
import nbformat as nbf
import json, os, sys

# On ajoute le répertoire courant au path pour pouvoir importer model_prevoyance
sys.path.insert(0, os.path.dirname(__file__))

nb = nbf.v4.new_notebook()

# ───────────────────────────────────────────────
# Helper pour créer des cellules proprement
# ───────────────────────────────────────────────
def code_cell(src):  return nbf.v4.new_code_cell(src)
def md_cell(src):    return nbf.v4.new_markdown_cell(src)

cells = []

# ════════════════════════════════════════════════════════════
# TITRE & INTRODUCTION
# ════════════════════════════════════════════════════════════
cells.append(md_cell("""# Tarification & Compte de Résultat — Régime de Prévoyance Collective

**Paméla Fagla | M1 Actuariat ISUP, Sorbonne Université**

---

## Contexte

Un régime de **prévoyance collective** est un contrat d'assurance souscrit par une **entreprise** pour l'ensemble de ses salariés. Il est obligatoire depuis l'ANI de 2013 pour les salariés non-cadres (minima imposés) et généralement étendu aux cadres par convention collective.

Ce projet modélise **trois garanties fondamentales** :

| Garantie | Déclencheur | Prestation |
|---|---|---|
| **Décès** | Décès du salarié (toutes causes) | Capital versé aux bénéficiaires |
| **IJ — Incapacité Temporaire** | Arrêt de travail > franchise | Indemnité journalière (complément salaire) |
| **Invalidité IPT** | Invalidité 2ème / 3ème catégorie | Rente mensuelle jusqu'à 65 ans |

### Objectifs du notebook

1. Construire les tables de mortalité et morbidité
2. Simuler un portefeuille de 500 salariés
3. Calculer les primes pures et commerciales par garantie
4. Construire le compte de résultat du régime
5. Analyser le ratio S/P et calculer le tarif de renouvellement
"""))

# ════════════════════════════════════════════════════════════
# CELL 0 : Imports
# ════════════════════════════════════════════════════════════
cells.append(code_cell("""\
import sys, os
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import du module actuariel
from model_prevoyance import (
    build_mortality_table, get_morbidite, annuite_certaine,
    simulate_portfolio, tarifier_portefeuille,
    simulate_sinistres, compte_resultat, tarif_renouvellement
)

# Style graphique
plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi':      120,
})
NAVY   = '#1F3864'
BLUE   = '#2E75B6'
GREEN  = '#70AD47'
ORANGE = '#C55A11'
GREY   = '#7F7F7F'

pd.set_option('display.float_format', '{:,.2f}'.format)
print("✅ Modules chargés avec succès")
"""))

# ════════════════════════════════════════════════════════════
# SECTION 1 : TABLES ACTUARIELLES
# ════════════════════════════════════════════════════════════
cells.append(md_cell("""---
## Section 1 — Tables Actuarielles

### 1.1 Table de mortalité TD 88-90

La **TD 88-90** (Tables de Décès 1988–90) est la table réglementaire de référence pour les engagements de prévoyance en France. Elle donne, pour chaque âge $x$, la probabilité de décès dans l'année :

$$q_x = P(\\text{décès entre } x \\text{ et } x+1)$$

> **Note** : En assurance vie (rentes), on utilise les tables **TGH/TGF 05** (prospectives) car le risque de longévité est crucial. En prévoyance décès court terme, la TD 88-90 reste la référence.
"""))

cells.append(code_cell("""\
# ── Construction de la table TD 88-90 ──────────────────────
table_mort = build_mortality_table(age_min=22, age_max=65)
print(table_mort.head(10).to_string())
"""))

cells.append(code_cell("""\
# ── Visualisation des taux de mortalité ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ages = table_mort.index

ax = axes[0]
ax.semilogy(ages, table_mort['qx_homme'] * 1000, color=NAVY,  lw=2.5, label='Hommes')
ax.semilogy(ages, table_mort['qx_femme'] * 1000, color=ORANGE, lw=2.5, label='Femmes')
ax.set_xlabel('Âge')
ax.set_ylabel('qx (‰)')
ax.set_title('Taux de mortalité TD 88-90 (échelle log)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, which='both')

ax = axes[1]
ratio = table_mort['qx_homme'] / table_mort['qx_femme']
ax.plot(ages, ratio, color=BLUE, lw=2.5)
ax.axhline(y=1, color=GREY, ls='--', alpha=0.5)
ax.set_xlabel('Âge')
ax.set_ylabel('qx_H / qx_F')
ax.set_title('Ratio de mortalité Hommes / Femmes', fontweight='bold')
ax.fill_between(ages, ratio, 1, alpha=0.15, color=BLUE)
ax.text(30, ratio.loc[30]+0.1, f"×{ratio.loc[30]:.1f} à 30 ans", color=BLUE, fontsize=9)
ax.text(55, ratio.loc[55]+0.1, f"×{ratio.loc[55]:.1f} à 55 ans", color=NAVY, fontsize=9)

plt.tight_layout()
plt.savefig('fig_mortalite_td8890.png', bbox_inches='tight', dpi=150)
plt.show()
print("➡  Les hommes ont une mortalité 2 à 2.5× plus élevée que les femmes aux âges actifs.")
"""))

cells.append(md_cell("""### 1.2 Tables de morbidité — Incapacité & Invalidité

Les tables de **morbidité** décrivent la sinistralité liée aux arrêts de travail et à l'invalidité.

**Sources de référence en pratique :**
- **BCAC** (Bureau Commun des Assurances Collectives) : données mutualisées du marché prévoyance
- **CNAM** (Caisse Nationale d'Assurance Maladie) : statistiques nationales sur les arrêts de travail
- **Experience studies** internes aux assureurs

Pour ce projet, des hypothèses simplifiées calibrées sur les ordres de grandeur CNAM sont utilisées.
"""))

cells.append(code_cell("""\
# ── Table de morbidité par tranche d'âge ────────────────────
morb_data = []
for age in range(22, 66):
    p_arret, d_ij, p_inv = get_morbidite(age)
    ann = annuite_certaine(65 - age)
    morb_data.append({
        'age': age,
        'p_arret (%)': p_arret * 100,
        'durée IJ nette (j)': d_ij,
        'p_invalidité (‰)': p_inv * 1000,
        'ä_{65-x} (3.5%)': round(ann, 3)
    })
df_morb = pd.DataFrame(morb_data)
print(df_morb[df_morb['age'].isin([25, 30, 35, 40, 45, 50, 55, 60, 64])].to_string(index=False))
"""))

cells.append(code_cell("""\
# ── Visualisation de la morbidité ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

ax = axes[0]
ax.bar(df_morb['age'], df_morb['p_arret (%)'], color=BLUE, alpha=0.8, width=0.8)
ax.set_xlabel('Âge'); ax.set_ylabel('%')
ax.set_title("Taux d'arrêt annuel", fontweight='bold')
ax.annotate('Pic en fin de carrière\\n(TMS, pathologies chroniques)',
            xy=(58, 30), fontsize=8, color=NAVY)

ax = axes[1]
ax.bar(df_morb['age'], df_morb['durée IJ nette (j)'], color=GREEN, alpha=0.8, width=0.8)
ax.set_xlabel('Âge'); ax.set_ylabel('Jours')
ax.set_title("Durée moyenne d'arrêt (hors franchise)", fontweight='bold')

ax = axes[2]
ax.bar(df_morb['age'], df_morb['p_invalidité (‰)'], color=ORANGE, alpha=0.8, width=0.8)
ax.set_xlabel('Âge'); ax.set_ylabel('‰')
ax.set_title("Taux de passage en invalidité IPT (‰)", fontweight='bold')
ax.annotate('Passage en retraite\\nà 65 ans → taux nul',
            xy=(62, 9), fontsize=8, ha='right')

plt.tight_layout()
plt.savefig('fig_morbidite.png', bbox_inches='tight', dpi=150)
plt.show()
"""))

# ════════════════════════════════════════════════════════════
# SECTION 2 : PORTEFEUILLE
# ════════════════════════════════════════════════════════════
cells.append(md_cell("""---
## Section 2 — Simulation du Portefeuille Entreprise

**Hypothèses de souscription :**
- 500 salariés (secteur services, PME type)
- Distribution des âges : normale tronquée (μ = 40 ans, σ = 10 ans)
- Sexe : 60 % hommes / 40 % femmes
- Salaire mensuel brut : log-normale (médiane ≈ 3 200 €, σ = 0.35)
- Exposition : 1 an (contrat annuel renouvelable)
"""))

cells.append(code_cell("""\
# ── Simulation du portefeuille ──────────────────────────────
df_port = simulate_portfolio(
    n_salaries=500,
    age_mean=40, age_std=10, age_min=22, age_max=62,
    pct_hommes=0.60,
    salaire_median=3200, salaire_sigma=0.35,
    seed=42
)
print(f"Portefeuille : {len(df_port)} salariés")
print(df_port.describe()[['age', 'salaire_mensuel', 'salaire_annuel']].round(0))
"""))

cells.append(code_cell("""\
# ── Visualisation du portefeuille ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

ax = axes[0]
ax.hist(df_port['age'], bins=20, color=NAVY, alpha=0.8, edgecolor='white')
ax.axvline(df_port['age'].mean(), color=ORANGE, lw=2, ls='--',
           label=f"Moyenne : {df_port['age'].mean():.1f} ans")
ax.set_xlabel('Âge'); ax.set_ylabel('Effectif')
ax.set_title('Distribution des âges', fontweight='bold')
ax.legend(fontsize=9)

ax = axes[1]
counts = df_port['sexe'].value_counts()
ax.pie(counts.values, labels=['Hommes', 'Femmes'],
       colors=[NAVY, ORANGE], autopct='%1.0f%%',
       startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
ax.set_title('Répartition H/F', fontweight='bold')

ax = axes[2]
ax.hist(df_port['salaire_mensuel'], bins=30, color=BLUE, alpha=0.8, edgecolor='white')
ax.axvline(df_port['salaire_mensuel'].median(), color=ORANGE, lw=2, ls='--',
           label=f"Médiane : {df_port['salaire_mensuel'].median():,.0f} €")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}€'))
ax.set_xlabel('Salaire mensuel brut'); ax.set_ylabel('Effectif')
ax.set_title('Distribution des salaires', fontweight='bold')
ax.legend(fontsize=9)

plt.suptitle("Portefeuille Entreprise XYZ — 500 salariés", fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig_portefeuille.png', bbox_inches='tight', dpi=150)
plt.show()

print(f"\\nMasse salariale annuelle : {df_port['salaire_annuel'].sum():,.0f} €")
print(f"Salaire annuel moyen     : {df_port['salaire_annuel'].mean():,.0f} €")
"""))

# ════════════════════════════════════════════════════════════
# SECTION 3 : TARIFICATION
# ════════════════════════════════════════════════════════════
cells.append(md_cell("""---
## Section 3 — Tarification des Garanties

### Formules actuarielles

**Garantie Décès**

$$\\text{Prime pure}_{\\text{décès}} = q_x \\times \\underbrace{k \\times S_A}_{\\text{capital assuré}}$$

où $q_x$ est lu dans la TD 88-90, $k = 3$ (multiplicateur de salaire annuel $S_A$).

**Garantie IJ (Incapacité Temporaire)**

$$\\text{Prime pure}_{\\text{IJ}} = p_{\\text{arrêt}}(x) \\times D_{\\text{nette}}(x) \\times \\underbrace{\\alpha \\times \\frac{S_A}{365}}_{\\text{IJ journalière}}$$

où $\\alpha = 70\\%$ est le taux de remplacement et $D_{\\text{nette}}$ la durée nette de franchise.

**Garantie Invalidité IPT**

$$\\text{Prime pure}_{\\text{inv}} = p_{\\text{inv}}(x) \\times \\beta \\times S_A \\times \\ddot{a}_{n}^{3.5\\%}$$

où $\\beta = 80\\%$, $n = 65 - x$ et $\\ddot{a}_n = \\frac{1 - v^n}{i}$ est la valeur actualisée de la rente.

**Prime commerciale**

$$p_c = \\frac{p_p}{1 - \\pi}, \\quad \\pi = \\text{taux de chargement}$$
"""))

cells.append(code_cell("""\
# ── Tarification du portefeuille ────────────────────────────
df_tarifie = tarifier_portefeuille(
    df_port,
    capital_mult=3,
    taux_remplacement_ij=0.70,
    rente_invalidite_mult=0.80,
    loading_deces=0.12,
    loading_ij=0.18,
    loading_inv=0.20
)

# Résumé de la tarification
colonnes_affichees = ['pp_deces', 'pp_ij', 'pp_invalidite', 'pp_totale',
                      'pc_totale', 'taux_cotisation_pct']
print("Primes pures et commerciales — statistiques descriptives (€/an par salarié):")
print(df_tarifie[colonnes_affichees].describe().round(1).to_string())
"""))

cells.append(code_cell("""\
# ── Cotisation totale et décomposition par garantie ─────────
primes_totales = {
    'Décès':      df_tarifie['pc_deces'].sum(),
    'IJ':         df_tarifie['pc_ij'].sum(),
    'Invalidité': df_tarifie['pc_invalidite'].sum(),
}
total_pc = sum(primes_totales.values())
masse_sal = df_tarifie['salaire_annuel'].sum()

print("=" * 55)
print("  PRIMES COMMERCIALES TOTALES — PORTEFEUILLE 500 SALARIÉS")
print("=" * 55)
for gar, prime in primes_totales.items():
    pct = prime / total_pc * 100
    print(f"  {gar:<14} : {prime:>12,.0f} €   ({pct:.1f}%)")
print(f"  {'TOTAL':<14} : {total_pc:>12,.0f} €  (100.0%)")
print(f"\\n  Taux de cotisation global : {total_pc/masse_sal*100:.2f}% de la masse salariale")
print(f"  Cotisation moyenne / salarié : {total_pc/len(df_tarifie):,.0f} €/an")
"""))

cells.append(code_cell("""\
# ── Prime par salarié selon l'âge ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (col, color, title, subtitle) in zip(axes, [
    ('pp_deces',    NAVY,   'Garantie Décès',        'Prime pure = qx × Capital'),
    ('pp_ij',       BLUE,   'Garantie IJ',           'Prime pure = p_arrêt × Durée × IJ'),
    ('pp_invalidite', GREEN,'Garantie Invalidité',   'Prime pure = p_inv × Rente × ä_n'),
]):
    grouped = df_tarifie.groupby('age')[col].mean()
    ax.plot(grouped.index, grouped.values, color=color, lw=2.5)
    ax.fill_between(grouped.index, grouped.values, alpha=0.15, color=color)
    ax.set_xlabel('Âge'); ax.set_ylabel('Prime pure (€/an)')
    ax.set_title(title, fontweight='bold')
    ax.set_subtitle = lambda s: None  # not available, use text
    ax.text(0.05, 0.95, subtitle, transform=ax.transAxes,
            fontsize=8, va='top', color='grey')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}€'))

plt.suptitle("Prime pure moyenne par âge — Portefeuille 500 salariés",
             fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig_primes_par_age.png', bbox_inches='tight', dpi=150)
plt.show()
"""))

# ════════════════════════════════════════════════════════════
# SECTION 4 : COMPTE DE RÉSULTAT
# ════════════════════════════════════════════════════════════
cells.append(md_cell("""---
## Section 4 — Compte de Résultat du Régime

On simule l'exercice annuel : pour chaque salarié, on tire aléatoirement s'il a décédé,
eu un arrêt de travail, ou basculé en invalidité. Ces sinistres simulés sont ensuite
comparés aux primes perçues.

Le **ratio sinistres/primes (S/P)** est l'indicateur clé de rentabilité :

$$S/P = \\frac{\\text{Sinistres réglés}}{\\text{Primes acquises}}$$

- $S/P < 75\\%$ → régime très rentable pour l'assureur
- $S/P \\in [75\\%, 85\\%]$ → zone d'équilibre saine
- $S/P > 90\\%$ → tension → discussion de revalorisation tarifaire
- $S/P > 100\\%$ → régime déficitaire
"""))

cells.append(code_cell("""\
# ── Simulation des sinistres ─────────────────────────────────
df_sin = simulate_sinistres(df_tarifie, seed=2026)

print("─── Sinistres simulés — exercice 2026 ────────────────")
print(f"Décès survenus        : {(df_sin['sinistre_deces'] > 0).sum():>3}  "
      f"({(df_sin['sinistre_deces'] > 0).sum() / len(df_sin)*100:.1f}% du portefeuille)")
print(f"Arrêts de travail     : {(df_sin['sinistre_ij'] > 0).sum():>3}  "
      f"({(df_sin['sinistre_ij'] > 0).sum() / len(df_sin)*100:.1f}% du portefeuille)")
print(f"Invalidités déclarées : {(df_sin['sinistre_invalidite'] > 0).sum():>3}  "
      f"({(df_sin['sinistre_invalidite'] > 0).sum() / len(df_sin)*100:.1f}% du portefeuille)")
print(f"\\nCoût total sinistres  : {df_sin['sinistre_total'].sum():>12,.0f} €")
"""))

cells.append(code_cell("""\
# ── Compte de résultat complet ──────────────────────────────
cr = compte_resultat(df_sin)

gar_order = ['Décès', 'IJ', 'Invalidité', 'Total']

print("\\n" + "=" * 70)
print("   COMPTE DE RÉSULTAT TECHNIQUE — RÉGIME PRÉVOYANCE COLLECTIVE")
print("   Entreprise XYZ | 500 salariés | Exercice 2026")
print("=" * 70)
header = f"{'':35} {'Décès':>10} {'IJ':>10} {'Invalidité':>12} {'TOTAL':>12}"
print(header)
print("─" * 70)

rows = [
    ("Primes commerciales (€)",  'primes_commerciales'),
    ("Primes pures (€)",         'primes_pures'),
    ("Chargements (€)",          'chargements'),
    ("Sinistres réglés (€)",     'sinistres'),
    ("Résultat technique (€)",   'resultat'),
]
for label, key in rows:
    vals = [cr[g][key] for g in ['Décès', 'IJ', 'Invalidité', 'Total']]
    line = f"{label:35}"
    for v in vals:
        line += f" {v:>11,.0f}" if abs(v) >= 1 else f" {'0':>11}"
    print(line)

print("─" * 70)
sp_vals = [cr[g]['ratio_sp'] for g in ['Décès', 'IJ', 'Invalidité', 'Total']]
sp_line = f"{'Ratio S/P':35}"
for v in sp_vals:
    sp_line += f" {v:>11.1%}"
print(sp_line)
print("=" * 70)

sp_global = cr['Total']['ratio_sp']
print(f"\\n  📊 Ratio S/P global : {sp_global:.1%}")
if sp_global < 0.75:
    print("  ✅ Régime rentable — le tarif couvre largement les sinistres.")
elif sp_global < 0.90:
    print("  ⚖️  Régime équilibré — tarif dans la norme de marché.")
else:
    print("  ⚠️  Tension tarifaire — une revalorisation est à étudier.")
"""))

cells.append(code_cell("""\
# ── Visualisation du compte de résultat ─────────────────────
fig = plt.figure(figsize=(15, 5))
gs = GridSpec(1, 3, figure=fig)

# --- Graphique 1 : Primes vs Sinistres par garantie ---
ax1 = fig.add_subplot(gs[0, :2])
garanties = ['Décès', 'IJ', 'Invalidité']
primes_vals   = [cr[g]['primes_commerciales'] / 1000 for g in garanties]
sinistres_vals = [cr[g]['sinistres'] / 1000 for g in garanties]
sp_vals_gar   = [cr[g]['ratio_sp'] for g in garanties]

x = np.arange(len(garanties))
width = 0.35
bars1 = ax1.bar(x - width/2, primes_vals,   width, label='Primes',    color=NAVY,   alpha=0.85)
bars2 = ax1.bar(x + width/2, sinistres_vals, width, label='Sinistres', color=ORANGE, alpha=0.85)

for bar, sp in zip(bars2, sp_vals_gar):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'S/P\n{sp:.0%}', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=ORANGE)

ax1.set_ylabel('Montant (k€)', fontsize=11)
ax1.set_title('Primes vs Sinistres par garantie', fontweight='bold', fontsize=12)
ax1.set_xticks(x); ax1.set_xticklabels(garanties, fontsize=11)
ax1.legend(fontsize=10)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}k€'))

# --- Graphique 2 : Gauge S/P global ---
ax2 = fig.add_subplot(gs[0, 2])
sp_g = cr['Total']['ratio_sp']

# Jauge semi-circulaire
theta = np.linspace(np.pi, 0, 300)
zones = [(0, 0.75, GREEN, 'Sain'), (0.75, 0.90, '#FFC000', 'Équilibré'),
         (0.90, 1.10, ORANGE, 'Tension')]
for sp_min, sp_max, col, lbl in zones:
    t_min = np.pi * (1 - sp_min / 1.10)
    t_max = np.pi * (1 - sp_max / 1.10)
    th = np.linspace(t_min, t_max, 50)
    ax2.fill_between(np.cos(th), np.zeros_like(th), np.sin(th),
                     alpha=0.35, color=col, label=lbl)
    ax2.plot(np.cos(th), np.sin(th), color=col, lw=3)

# Aiguille
angle = np.pi * (1 - sp_g / 1.10)
ax2.annotate('', xy=(0.72 * np.cos(angle), 0.72 * np.sin(angle)),
             xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', color=NAVY, lw=2.5))
ax2.text(0, -0.18, f'S/P = {sp_g:.1%}', ha='center', fontweight='bold',
         fontsize=13, color=NAVY)
ax2.set_xlim(-1.2, 1.2); ax2.set_ylim(-0.3, 1.2)
ax2.set_aspect('equal'); ax2.axis('off')
ax2.set_title('Ratio S/P global', fontweight='bold', fontsize=12)
ax2.legend(loc='lower center', fontsize=8, ncol=3, bbox_to_anchor=(0.5, -0.05))

plt.suptitle('Compte de Résultat Technique — Exercice 2026', fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig_compte_resultat.png', bbox_inches='tight', dpi=150)
plt.show()
"""))

# ════════════════════════════════════════════════════════════
# SECTION 5 : SENSIBILITÉ & RENOUVELLEMENT
# ════════════════════════════════════════════════════════════
cells.append(md_cell("""---
## Section 5 — Analyse de Sensibilité & Tarif de Renouvellement

En pratique, à chaque **renouvellement annuel**, le chargé d'études actuarielles :
1. Calcule le S/P réel de l'exercice écoulé
2. Le compare au S/P technique cible du tarif
3. Propose un **ajustement tarifaire** si nécessaire

Un scénario d'aggravation de sinistralité est également testé pour mesurer la robustesse du régime.
"""))

cells.append(code_cell("""\
# ── Tarif de renouvellement ──────────────────────────────────
renouv = tarif_renouvellement(cr, objectif_sp=0.75, marge_securite=0.05)

print("─── Proposition de renouvellement tarifaire ──────────")
print(f"  S/P réel exercice 2026 : {renouv['sp_reel']:.1%}")
print(f"  S/P cible technique    : {renouv['sp_cible']:.1%}")
print(f"  Facteur d'ajustement   : ×{renouv['facteur']:.3f}")
print(f"  Interprétation         : {renouv['interpretation']}")
"""))

cells.append(code_cell("""\
# ── Analyse de sensibilité : impact d'une dégradation de sinistralité ──
scenarios = {
    'Scenario central': 1.00,
    '+10% sinistres':   1.10,
    '+20% sinistres':   1.20,
    '+30% sinistres':   1.30,
    'Choc extrême +50%':1.50,
}

resultats_scenarios = []
for label, facteur in scenarios.items():
    df_scenario = df_sin.copy()
    for col in ['sinistre_deces', 'sinistre_ij', 'sinistre_invalidite']:
        df_scenario[col] = df_scenario[col] * facteur
    df_scenario['sinistre_total'] = (
        df_scenario['sinistre_deces'] +
        df_scenario['sinistre_ij'] +
        df_scenario['sinistre_invalidite']
    )
    cr_s = compte_resultat(df_scenario)
    sp_s = cr_s['Total']['ratio_sp']
    result_s = cr_s['Total']['resultat'] / 1000
    renouv_s = tarif_renouvellement(cr_s)
    resultats_scenarios.append({
        'Scénario': label,
        'S/P': f"{sp_s:.1%}",
        'Résultat (k€)': f"{result_s:+,.0f}",
        'Revalorisation': f"+{(renouv_s['facteur']-1)*100:.1f}%" if renouv_s['facteur'] > 1 else "0%",
        'Statut': '⚠️ Déficit' if sp_s > 1 else ('⚡ Tension' if sp_s > 0.88 else '✅ Équilibre')
    })

df_scenarios = pd.DataFrame(resultats_scenarios)
print(df_scenarios.to_string(index=False))
"""))

cells.append(code_cell("""\
# ── Graphique de sensibilité ─────────────────────────────────
facteurs_sinistres = np.linspace(0.80, 1.60, 50)
sp_curve = []
for f in facteurs_sinistres:
    df_s = df_sin.copy()
    for col in ['sinistre_deces', 'sinistre_ij', 'sinistre_invalidite']:
        df_s[col] *= f
    df_s['sinistre_total'] = df_s['sinistre_deces'] + df_s['sinistre_ij'] + df_s['sinistre_invalidite']
    cr_s = compte_resultat(df_s)
    sp_curve.append(cr_s['Total']['ratio_sp'])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(facteurs_sinistres, sp_curve, color=NAVY, lw=2.5)
ax.axhline(y=0.75, color=GREEN,  ls='--', alpha=0.7, label='Cible S/P : 75%')
ax.axhline(y=0.90, color='#FFC000', ls='--', alpha=0.7, label='Seuil tension : 90%')
ax.axhline(y=1.00, color=ORANGE, ls='--', alpha=0.7, label='Seuil déficit : 100%')
ax.fill_between(facteurs_sinistres, sp_curve, 0.75, alpha=0.08, color=GREEN)

# Point central
sp_central = cr['Total']['ratio_sp']
ax.scatter([1.0], [sp_central], color=NAVY, s=80, zorder=5, label=f'Scénario central: {sp_central:.1%}')

ax.set_xlabel('Facteur multiplicatif des sinistres', fontsize=11)
ax.set_ylabel('Ratio S/P', fontsize=11)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_title('Sensibilité du ratio S/P à une dégradation de la sinistralité',
             fontweight='bold', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig_sensibilite_sp.png', bbox_inches='tight', dpi=150)
plt.show()
"""))

# ════════════════════════════════════════════════════════════
# SECTION 6 : SYNTHÈSE
# ════════════════════════════════════════════════════════════
cells.append(md_cell("""---
## Section 6 — Synthèse Actuarielle

### Résultats clés

| Indicateur | Valeur |
|---|---|
| Effectif assuré | 500 salariés |
| Masse salariale annuelle | à calculer ci-dessous |
| Primes commerciales totales | à calculer |
| Taux de cotisation global | % masse salariale |
| Ratio S/P global | à calculer |
| Résultat technique | € |

### Limites du modèle

1. **Tables de morbidité simplifiées** : En pratique, les assureurs utilisent leurs propres expériences de portefeuille (données BCAC mutualisées). Les hypothèses CNAM utilisées ici sous-estiment peut-être la sinistralité IJ sur certains secteurs d'activité.

2. **Indépendance des garanties** : Le modèle suppose que décès, IJ et invalidité sont indépendants. En réalité, un long arrêt de travail précède souvent une invalidité — il existe une corrélation positive.

3. **Pas de risque de portefeuille** : Le modèle utilise des hypothèses statiques. Sur un petit portefeuille (<200 salariés), la volatilité des sinistres peut être significative. Un modèle stochastique (bootstrap de sinistres) permettrait de quantifier l'incertitude de réserve.

4. **Pas de rachats / résiliations** : Sur un régime collectif d'entreprise, la résiliation est possible mais rare (changement d'assureur). Le risque anti-sélectif à la souscription (entreprises avec forte sinistralité) est non modélisé.

### Apprentissages clés pour l'entretien

- Le **ratio S/P** est l'indicateur de pilotage central : il déclenche les discussions de renouvellement tarifaire
- La **garantie IJ** est la plus volatile (sinistres fréquents, coût unitaire faible) ; la **garantie Invalidité** est la plus incertaine (sinistres rares mais coût élevé via l'annuité)
- La **valeur actualisée de la rente d'invalidité** (ä_n) est sensible au taux d'actualisation : une hausse des taux réduit les provisions d'invalidité
- Le **taux de cotisation** (~% de la masse salariale) est la métrique que les DRH regardent à la signature et au renouvellement
"""))

cells.append(code_cell("""\
# ── Synthèse finale ──────────────────────────────────────────
masse_sal = df_sin['salaire_annuel'].sum()
total_pc  = df_sin['pc_totale'].sum()
total_sin = df_sin['sinistre_total'].sum()
sp_g      = cr['Total']['ratio_sp']
resultat  = cr['Total']['resultat']

print("=" * 60)
print("   SYNTHÈSE — RÉGIME PRÉVOYANCE COLLECTIVE XYZ 2026")
print("=" * 60)
print(f"  Effectif assuré          : {len(df_sin):>10} salariés")
print(f"  Masse salariale annuelle : {masse_sal:>10,.0f} €")
print(f"  Primes commerciales      : {total_pc:>10,.0f} €")
print(f"  Taux de cotisation       : {total_pc/masse_sal*100:>10.2f} %")
print(f"  Cotisation moy/salarié   : {total_pc/len(df_sin):>10,.0f} €/an")
print("─" * 60)
print(f"  Sinistres réglés         : {total_sin:>10,.0f} €")
print(f"  Ratio S/P global         : {sp_g:>10.1%}")
print(f"  Résultat technique       : {resultat:>+10,.0f} €")
print("=" * 60)

# Décomposition S/P
fig, ax = plt.subplots(figsize=(8, 4))
garanties = ['Décès', 'IJ', 'Invalidité', 'Total']
sp_vals   = [cr[g]['ratio_sp'] * 100 for g in garanties]
colors    = [NAVY, BLUE, GREEN, ORANGE]
bars = ax.bar(garanties, sp_vals, color=colors, alpha=0.85, edgecolor='white')
ax.axhline(y=75, color='green', ls='--', lw=1.5, label='Cible 75%')
ax.axhline(y=90, color='orange', ls='--', lw=1.5, label='Seuil tension 90%')
ax.axhline(y=100, color='red', ls='--', lw=1.5, label='Seuil déficit 100%')

for bar, val in zip(bars, sp_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('Ratio S/P (%)', fontsize=11)
ax.set_title('Ratio S/P par garantie — Exercice 2026', fontweight='bold', fontsize=12)
ax.legend(fontsize=9)
ax.set_ylim(0, 115)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())

plt.tight_layout()
plt.savefig('fig_synthese_sp.png', bbox_inches='tight', dpi=150)
plt.show()
print("\\n📁 Notebook exécuté avec succès — tous les graphiques générés.")
"""))

# ════════════════════════════════════════════════════════════
# ASSEMBLE AND SAVE
# ════════════════════════════════════════════════════════════
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.11.0"
    }
}

out_path = "/home/claude/prevoyance_collective/notebook_prevoyance_collective.ipynb"
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"✅ Notebook généré : {out_path}")
print(f"   {len(nb.cells)} cellules | {sum(1 for c in nb.cells if c.cell_type=='code')} code | {sum(1 for c in nb.cells if c.cell_type=='markdown')} markdown")
