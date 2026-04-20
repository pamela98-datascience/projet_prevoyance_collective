# Projet Actuariel — Tarification & Compte de Résultat Prévoyance Collective
**Paméla Fagla | M1 Actuariat ISUP, Sorbonne Université**

---

## Structure du projet

```
prevoyance_collective/
├── model_prevoyance.py                   ← Module actuariel central (importable)
├── notebook_prevoyance_collective.ipynb  ← Notebook Jupyter commenté (25 cellules)
├── streamlit_app.py                      ← Dashboard interactif
├── build_notebook.py                     ← Génère le .ipynb depuis le modèle
├── build_excel.py                        ← Génère le fichier Excel
└── README.md                             ← Ce fichier
```

---

## Démarrage rapide

### 1. Prérequis
```bash
pip install numpy pandas matplotlib seaborn scipy openpyxl nbformat streamlit
```

### 2. Exécuter le notebook
```bash
jupyter notebook notebook_prevoyance_collective.ipynb
```
Ou dans VS Code : ouvrir le fichier .ipynb directement.

### 3. Lancer le dashboard Streamlit
```bash
streamlit run streamlit_app.py
```
Puis ouvrir : http://localhost:8501

### 4. Régénérer l'Excel
```bash
python build_excel.py
```

---

## Ce que le projet modélise

### 3 garanties prévoyance collective

| Garantie | Déclencheur | Prestation | Prime pure |
|---|---|---|---|
| **Décès** | Décès toutes causes | Capital = 3 × SA | `qx × Capital` |
| **IJ** | Arrêt > 8 jours | 70% salaire journalier | `p_arrêt × D_net × IJ_jour` |
| **Invalidité IPT** | 2ème/3ème catégorie | Rente 80% SA jusqu'à 65 ans | `p_inv × Rente × ä_{65-x}` |

### Tables actuarielles
- **Mortalité** : TD 88-90 (référence réglementaire française), interpolée par spline cubique
- **Morbidité** : Hypothèses simplifiées calibrées CNAM/BCAC par tranche d'âge

### Résultats clés (portefeuille 500 salariés, âge moy. 40 ans)
- Taux de cotisation global : **4.35 % de la masse salariale**
- S/P théorique : **82.0 %** (régime sain)
- S/P réalisé exercice 2026 (seed=12) : **80.4 %**
- Résultat technique réalisé : **+176 k€**

---

## Fichier Excel — 4 feuilles

1. **Compte de résultat** : CR théo vs réalisé, ratio S/P par garantie, renouvellement tarifaire
2. **Tarification** : Primes pures et commerciales pour chacun des 500 salariés
3. **Tables actuarielles** : TD 88-90 + morbidité de 22 à 65 ans
4. **Monte Carlo SP** : Distribution du S/P sur 200 simulations + statistiques

---

## Concepts clés à maîtriser pour l'entretien

### Ratio S/P
```
S/P = Sinistres réglés / Primes acquises
```
- < 75% : régime rentable
- 75-90% : zone d'équilibre saine (loading = marge assureur)
- > 90% : tension, discussion de renouvellement tarifaire
- > 100% : déficitaire

### Prime commerciale
```
pc = pp / (1 - loading)
```
Le loading couvre : frais de gestion, acquisition, marge technique et profit.

### Valeur actualisée de la rente d'invalidité
```
ä_n = (1 - v^n) / i    où v = 1/(1+i), n = 65 - âge
```
Sensibilité : une hausse des taux (↑ i) réduit ä_n → réduit les provisions d'invalidité.

### Renouvellement tarifaire
```
Facteur = (S/P_réel / S/P_cible) × (1 + marge_sécurité)
```

### Pourquoi la volatilité sur petits portefeuilles ?
Sur 500 salariés, un cas d'invalidité = coût ~450 k€ vs primes invalidité totales ~585 k€.
→ Justifie la réassurance stop-loss pour les PME.

---

## Références réglementaires
- **ANI 2013** : Généralisation de la prévoyance collective aux non-cadres
- **Convention collective nationale** : Minima de garanties par branche
- **Article 83 / PER Obligatoire** : Épargne retraite collective (hors scope ici)
- **Directive Solvabilité II** : Provisions techniques BE + SCR pour les assureurs
- **Tables de référence** : TD 88-90 (décès), TGH/TGF 05 (rentes long terme)

---

*Projet réalisé dans le cadre des candidatures à des alternances en études actuarielles — 2026*
