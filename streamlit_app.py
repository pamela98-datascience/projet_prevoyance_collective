"""
Dashboard Streamlit — Tarification & Pilotage Prévoyance Collective
Paméla Fagla | M1 Actuariat ISUP

Usage : streamlit run streamlit_app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from model_prevoyance import (
    build_mortality_table, simulate_portfolio, tarifier_portefeuille,
    simulate_sinistres, compte_resultat, compte_resultat_theorique,
    monte_carlo_sp, tarif_renouvellement, annuite_certaine
)

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Prévoyance Collective — Tarification & Pilotage",
    page_icon="📊",
    layout="wide",
)

NAVY   = '#1F3864'
BLUE   = '#2E75B6'
GREEN  = '#70AD47'
ORANGE = '#C55A11'

# ── CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"]  { font-size: 1.6rem !important; color: #1F3864; font-weight: 700; }
[data-testid="stMetricLabel"]  { font-size: 0.8rem !important; color: #666; }
.section-title                 { background:#1F3864; color:white; padding:8px 16px;
                                  border-radius:6px; font-weight:700; font-size:1rem; margin:12px 0; }
.kpi-box { background:#F0F4F8; border-left:4px solid #1F3864; padding:10px 14px;
            border-radius:6px; margin:4px 0; font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Sorbonne_Universit%C3%A9.svg/200px-Sorbonne_Universit%C3%A9.svg.png",
    width=100
)
st.sidebar.markdown("## 🏢 Paramètres de l'entreprise")

n_sal    = st.sidebar.slider("Effectif assuré",     100, 2000, 500, step=50)
age_moy  = st.sidebar.slider("Âge moyen",           28, 55,   40)
age_std  = st.sidebar.slider("Dispersion des âges", 3,  15,   10)
sal_med  = st.sidebar.slider("Salaire mensuel médian (€)", 1800, 6000, 3200, step=200)
pct_h    = st.sidebar.slider("% Hommes",            30, 80,   60) / 100

st.sidebar.markdown("---")
st.sidebar.markdown("## 🛡️ Paramètres du régime")
cap_deces   = st.sidebar.slider("Capital décès (× salaire annuel)", 1, 5, 3)
tx_ij       = st.sidebar.slider("Taux de remplacement IJ (%)", 50, 100, 70) / 100
rente_inv   = st.sidebar.slider("Rente invalidité (% salaire annuel)", 50, 100, 80) / 100

st.sidebar.markdown("---")
st.sidebar.markdown("## ⚙️ Chargements")
load_dec = st.sidebar.slider("Chargement Décès (%)", 5, 25, 12) / 100
load_ij  = st.sidebar.slider("Chargement IJ (%)",    5, 30, 18) / 100
load_inv = st.sidebar.slider("Chargement Invalidité (%)", 5, 30, 20) / 100

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎲 Simulation")
seed_sim = st.sidebar.number_input("Seed exercice simulé", 0, 9999, 12)
n_mc     = st.sidebar.slider("Simulations Monte Carlo", 100, 500, 200, step=50)

# ── Compute ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute(n_sal, age_moy, age_std, sal_med, pct_h,
            cap_deces, tx_ij, rente_inv, load_dec, load_ij, load_inv,
            seed_sim, n_mc):
    t    = build_mortality_table()
    port = simulate_portfolio(n_salaries=n_sal, age_mean=age_moy, age_std=age_std,
                              salaire_median=sal_med, pct_hommes=pct_h, seed=42)
    tar  = tarifier_portefeuille(port, t,
                                  capital_mult=cap_deces,
                                  taux_remplacement_ij=tx_ij,
                                  rente_invalidite_mult=rente_inv,
                                  loading_deces=load_dec,
                                  loading_ij=load_ij,
                                  loading_inv=load_inv)
    sin  = simulate_sinistres(tar, t, seed=seed_sim)
    cr_r = compte_resultat(sin)
    cr_t = compte_resultat_theorique(tar)
    rv   = tarif_renouvellement(cr_r)
    mc   = monte_carlo_sp(tar, n_simulations=n_mc, table_mortalite=t)
    return t, port, tar, sin, cr_r, cr_t, rv, mc

with st.spinner("Calcul en cours…"):
    t, port, tar, sin, cr_r, cr_t, rv, mc = compute(
        n_sal, age_moy, age_std, sal_med, pct_h,
        cap_deces, tx_ij, rente_inv, load_dec, load_ij, load_inv,
        seed_sim, n_mc
    )

# ── Header ─────────────────────────────────────────────────────
st.markdown(f"""
<div style='background:#1F3864; padding:18px 24px; border-radius:10px; margin-bottom:16px;'>
  <h2 style='color:white; margin:0; font-size:1.5rem;'>
    📊 Tarification & Pilotage — Régime de Prévoyance Collective
  </h2>
  <p style='color:#B8C9E1; margin:4px 0 0 0; font-size:0.85rem;'>
    Paméla Fagla | M1 Actuariat ISUP, Sorbonne Université &nbsp;|&nbsp;
    {n_sal} salariés &nbsp;|&nbsp; Âge moyen {age_moy} ans &nbsp;|&nbsp;
    Salaire médian {sal_med:,} €/mois
  </p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Compte de Résultat",
    "💰 Tarification",
    "🎲 Monte Carlo",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 : COMPTE DE RÉSULTAT
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Indicateurs clés du régime</div>', unsafe_allow_html=True)
    
    sp_real = cr_r['Total']['ratio_sp']
    sp_theo = cr_t['Total']['ratio_sp']
    masse_sal = tar['salaire_annuel'].sum()
    taux_cot = tar['pc_totale'].sum() / masse_sal * 100
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("S/P réalisé",       f"{sp_real:.1%}", f"{(sp_real-sp_theo)*100:+.1f}pt vs théo")
    c2.metric("S/P théorique",     f"{sp_theo:.1%}",  "attendu sur le long terme")
    c3.metric("Résultat technique", f"{cr_r['Total']['resultat']/1000:+,.0f} k€")
    c4.metric("Taux de cotisation", f"{taux_cot:.2f} %", "de la masse salariale")
    c5.metric("Prime moy/salarié",  f"{tar['pc_totale'].mean():,.0f} €/an")
    
    st.markdown("---")
    
    col_l, col_r = st.columns([3, 2])
    
    with col_l:
        st.markdown('<div class="section-title">Compte de résultat technique</div>', unsafe_allow_html=True)
        
        garanties = ["Décès", "IJ", "Invalidité", "TOTAL"]
        cr_data = []
        for g in garanties:
            is_total = g == "TOTAL"
            cr_data.append({
                "Garantie": g,
                "Primes (€)": f"{cr_r.get(g, cr_r.get('Total',{})).get('primes_commerciales', cr_r['Total']['primes_commerciales']):,.0f}",
                "Sinistres réels (€)": f"{cr_r.get(g, cr_r.get('Total',{})).get('sinistres', cr_r['Total']['sinistres']):,.0f}",
                "Sinistres théo (€)": f"{cr_t.get(g, cr_t.get('Total',{})).get('sinistres', cr_t['Total']['sinistres']):,.0f}",
                "S/P réel": f"{cr_r.get(g, cr_r.get('Total',{})).get('ratio_sp', cr_r['Total']['ratio_sp']):.1%}",
                "Résultat (€)": f"{cr_r.get(g, cr_r.get('Total',{})).get('resultat', cr_r['Total']['resultat']):+,.0f}",
            })
        
        for g in ['Décès', 'IJ', 'Invalidité']:
            cr_data[garanties.index(g)]["Primes (€)"] = f"{cr_r[g]['primes_commerciales']:,.0f}"
            cr_data[garanties.index(g)]["Sinistres réels (€)"] = f"{cr_r[g]['sinistres']:,.0f}"
            cr_data[garanties.index(g)]["Sinistres théo (€)"] = f"{cr_t[g]['sinistres']:,.0f}"
            cr_data[garanties.index(g)]["S/P réel"] = f"{cr_r[g]['ratio_sp']:.1%}"
            cr_data[garanties.index(g)]["Résultat (€)"] = f"{cr_r[g]['resultat']:+,.0f}"
        cr_data[3]["Primes (€)"] = f"{cr_r['Total']['primes_commerciales']:,.0f}"
        cr_data[3]["Sinistres réels (€)"] = f"{cr_r['Total']['sinistres']:,.0f}"
        cr_data[3]["Sinistres théo (€)"] = f"{cr_t['Total']['sinistres']:,.0f}"
        cr_data[3]["S/P réel"] = f"{cr_r['Total']['ratio_sp']:.1%}"
        cr_data[3]["Résultat (€)"] = f"{cr_r['Total']['resultat']:+,.0f}"
        
        st.dataframe(pd.DataFrame(cr_data), use_container_width=True, hide_index=True)
        
        # Renouvellement
        st.markdown(f"""
        <div class='kpi-box'>
        <b>🔄 Renouvellement tarifaire :</b> {rv['interpretation']}
        </div>
        """, unsafe_allow_html=True)
    
    with col_r:
        # Gauge S/P
        fig, ax = plt.subplots(figsize=(5, 3.5))
        theta = np.linspace(np.pi, 0, 300)
        zones = [(0, 0.75, GREEN, 'Sain'), (0.75, 0.90, '#FFC000', 'Équilibre'),
                 (0.90, 1.10, ORANGE, 'Tension')]
        for sp_min, sp_max, col, lbl in zones:
            t_a = np.pi * (1 - sp_min / 1.10)
            t_b = np.pi * (1 - sp_max / 1.10)
            th = np.linspace(t_a, t_b, 50)
            ax.fill_between(np.cos(th), 0, np.sin(th), alpha=0.35, color=col)
            ax.plot(np.cos(th), np.sin(th), color=col, lw=3)

        for sp_v, linestyle, lbl_v in [(sp_theo, ':', 'Théo'), (sp_real, '-', 'Réel')]:
            angle = np.pi * (1 - min(sp_v, 1.10) / 1.10)
            arrow_col = BLUE if lbl_v == 'Théo' else NAVY
            ax.annotate('', xy=(0.70*np.cos(angle), 0.70*np.sin(angle)), xytext=(0,0),
                        arrowprops=dict(arrowstyle='->', color=arrow_col, lw=2+int(lbl_v=='Réel')))
            ax.text(0.78*np.cos(angle), 0.80*np.sin(angle), lbl_v, color=arrow_col, fontsize=8, ha='center')
        
        ax.text(0, -0.15, f"S/P = {sp_real:.1%}", ha='center', fontweight='bold', fontsize=14, color=NAVY)
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.3, 1.25)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title("Ratio Sinistres / Primes", fontweight='bold', fontsize=11)
        patches = [mpatches.Patch(color=c, label=l, alpha=0.5) for _, _, c, l in zones]
        ax.legend(handles=patches, loc='lower center', fontsize=8, ncol=3, bbox_to_anchor=(0.5, -0.12))
        st.pyplot(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 2 : TARIFICATION
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Primes par garantie et par âge</div>', unsafe_allow_html=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (col, color, title) in zip(axes, [
        ('pp_deces',    NAVY,   'Garantie Décès\nPrime pure = qx × Capital'),
        ('pp_ij',       BLUE,   'Garantie IJ\nPrime pure = p_arrêt × Durée × IJ_jour'),
        ('pp_invalidite', GREEN,'Garantie Invalidité\nPrime pure = p_inv × Rente × ä_{65-x}'),
    ]):
        grp = tar.groupby('age')[col].mean()
        ax.plot(grp.index, grp.values, color=color, lw=2.5)
        ax.fill_between(grp.index, grp.values, alpha=0.15, color=color)
        ax.set_xlabel('Âge'); ax.set_ylabel('Prime pure (€/an)')
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}€'))
        
        # Annotation pic
        peak_age = grp.idxmax()
        ax.axvline(peak_age, color=color, ls='--', alpha=0.4)
        ax.text(peak_age, grp.max()*0.8, f'  {peak_age} ans', color=color, fontsize=8)
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    
    # Décomposition de la cotisation
    st.markdown('<div class="section-title">Décomposition de la cotisation</div>', unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ages_grp = tar.groupby('age')[['pc_deces', 'pc_ij', 'pc_invalidite']].mean()
        ax2.stackplot(ages_grp.index, ages_grp['pc_deces'], ages_grp['pc_ij'], ages_grp['pc_invalidite'],
                      labels=['Décès', 'IJ', 'Invalidité'], colors=[NAVY, BLUE, GREEN], alpha=0.85)
        ax2.set_xlabel('Âge'); ax2.set_ylabel('Prime commerciale (€/an)')
        ax2.set_title('Décomposition de la cotisation par âge', fontweight='bold')
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}€'))
        ax2.legend(loc='upper left', fontsize=9)
        st.pyplot(fig2, use_container_width=True)
    
    with col_b:
        # Pie chart par garantie
        totaux = {
            'Décès':      tar['pc_deces'].sum(),
            'IJ':         tar['pc_ij'].sum(),
            'Invalidité': tar['pc_invalidite'].sum(),
        }
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        wedges, texts, autos = ax3.pie(
            list(totaux.values()),
            labels=list(totaux.keys()),
            autopct='%1.1f%%',
            colors=[NAVY, BLUE, GREEN],
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        total_pc = sum(totaux.values())
        for w, a in zip(wedges, autos):
            a.set_fontsize(10)
        ax3.set_title(f'Répartition des primes\nTotal : {total_pc/1000:,.0f} k€',
                      fontweight='bold')
        st.pyplot(fig3, use_container_width=True)
    
    # Taux de cotisation
    st.markdown(f"""
    <div class='kpi-box'>
    💡 <b>Taux de cotisation global :</b> {tar['pc_totale'].sum()/tar['salaire_annuel'].sum()*100:.2f}%
    de la masse salariale — dont <b>Décès</b> : {tar['pc_deces'].sum()/tar['salaire_annuel'].sum()*100:.2f}% |
    <b>IJ</b> : {tar['pc_ij'].sum()/tar['salaire_annuel'].sum()*100:.2f}% |
    <b>Invalidité</b> : {tar['pc_invalidite'].sum()/tar['salaire_annuel'].sum()*100:.2f}%
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3 : MONTE CARLO
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Distribution du ratio S/P — volatilité de portefeuille</div>', unsafe_allow_html=True)
    
    st.info("""
    **Pourquoi le Monte Carlo ?** Sur un petit portefeuille (<500 salariés), la sinistralité réalisée
    peut s'écarter fortement de la sinistralité théorique (loi des grands nombres peu applicable).
    Chaque simulation représente un exercice annuel indépendant.
    Cette distribution justifie la constitution d'une **marge de sécurité** et le recours à la
    **réassurance stop-loss** pour les PME.
    """)
    
    sp_vals = mc['sp_total']
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("S/P moyen", f"{sp_vals.mean():.1%}")
    col2.metric("S/P médian", f"{sp_vals.median():.1%}")
    col3.metric("P5 / P95", f"{sp_vals.quantile(0.05):.0%} / {sp_vals.quantile(0.95):.0%}")
    col4.metric("% années déficitaires", f"{(sp_vals > 1.0).mean():.1%}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogramme S/P
    ax = axes[0]
    n_bins = 40
    colors_hist = []
    bins = np.linspace(sp_vals.min(), sp_vals.max(), n_bins+1)
    ax.hist(sp_vals, bins=bins, color=BLUE, alpha=0.7, edgecolor='white')
    ax.axvline(sp_vals.mean(),  color=NAVY,   lw=2.5, ls='-',  label=f"Moyenne : {sp_vals.mean():.1%}")
    ax.axvline(sp_vals.median(),color=BLUE,   lw=2,   ls='--', label=f"Médiane : {sp_vals.median():.1%}")
    ax.axvline(0.75, color=GREEN,  lw=1.5, ls=':', alpha=0.8, label="Cible 75%")
    ax.axvline(0.90, color='#FFC000', lw=1.5, ls=':', alpha=0.8, label="Tension 90%")
    ax.axvline(1.00, color=ORANGE, lw=1.5, ls=':', alpha=0.8, label="Déficit 100%")
    ax.fill_between([1.00, sp_vals.max()], 0, ax.get_ylim()[1] if len(sp_vals) else 10,
                    alpha=0.08, color=ORANGE)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel('Ratio S/P', fontsize=11)
    ax.set_ylabel('Fréquence', fontsize=11)
    ax.set_title(f'Distribution du S/P — {n_mc} simulations\n({n_sal} salariés)', fontweight='bold')
    ax.legend(fontsize=9)
    
    # Impact taille de portefeuille
    ax2 = axes[1]
    sizes = [100, 200, 500, 1000, 2000]
    cv_sp = []
    for sz in sizes:
        p_sz = simulate_portfolio(n_salaries=sz, age_mean=age_moy, age_std=age_std,
                                   salaire_median=sal_med, pct_hommes=pct_h, seed=42)
        t_sz = tarifier_portefeuille(p_sz, t, capital_mult=cap_deces,
                                     taux_remplacement_ij=tx_ij, rente_invalidite_mult=rente_inv,
                                     loading_deces=load_dec, loading_ij=load_ij, loading_inv=load_inv)
        mc_sz = monte_carlo_sp(t_sz, n_simulations=100, table_mortalite=t)
        cv_sp.append(mc_sz['sp_total'].std())
    
    ax2.plot(sizes, cv_sp, color=NAVY, lw=2.5, marker='o', ms=8)
    ax2.axvline(n_sal, color=ORANGE, ls='--', lw=2, label=f"Votre portefeuille : {n_sal}")
    ax2.set_xlabel('Effectif assuré', fontsize=11)
    ax2.set_ylabel('Écart-type du S/P', fontsize=11)
    ax2.set_title("Volatilité du S/P selon la taille du portefeuille\n(Effet loi des grands nombres)",
                  fontweight='bold')
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.legend(fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    
    st.markdown("""
    <div class='kpi-box'>
    📌 <b>Lecture actuarielle :</b> Plus le portefeuille est petit, plus la volatilité du S/P est élevée.
    Sur 100 salariés, 1 cas d'invalidité peut représenter 15-20% des primes annuelles.
    C'est pourquoi les régimes de petites entreprises nécessitent une <b>réassurance stop-loss</b>
    (plafond de sinistres pris en charge par le réassureur au-delà d'un seuil = retention).
    </div>
    """, unsafe_allow_html=True)
    
    for q, a in qas:
        with st.expander(f"❓ {q}"):
            st.markdown(f"**Réponse :**\n\n{a}")
    
    st.markdown("---")
    st.markdown(f"""
    <div style='background:#F0F4F8; padding:16px; border-radius:8px; font-size:0.85rem;'>
    <b>📌 Limites à mentionner spontanément en entretien :</b><br>
    1. Tables de morbidité simplifiées (données BCAC propriétaires en pratique)<br>
    2. Indépendance des garanties supposée (IJ précède souvent l'invalidité)<br>
    3. Pas de modélisation des rachats / résiliations<br>
    4. Taux technique fixe — sensible aux évolutions de taux (impact sur ä_{{65-x}})
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<p style='color:#999; font-size:0.75rem; text-align:center;'>
Paméla Fagla | M1 Actuariat ISUP, Sorbonne Université | Projet Tarification Prévoyance Collective 2026
</p>
""", unsafe_allow_html=True)
