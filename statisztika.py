import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("osszesitett_sebesseg_profilok.csv")
print(f"\nBetöltött galaxisok összesen: {len(df)} db")

LANG = 0
EPS = False

spiral = df[df['kategoria'].str.contains('Spirál', na=False)].copy()
elliptikus = df[df['kategoria'].str.contains('Elliptikus', na=False)].copy()
print(f"  - Ebből spirálgalaxis:    {len(spiral)} db")
print(f"  - Ebből elliptikus:       {len(elliptikus)} db\n")

def szamol_kulso_atlagok(tabla, mod="spiral"):
    v_mert_kulso, v_kep_kulso, v_mod_kulso = [], [], []
    for _, sor in tabla.iterrows():
        vm_lista, vk_lista, vmod_lista = [], [], []
        for i in range(6, 9):
            vg, vs = sor[f'v_gaz_ha_kms_{i}'], sor[f'v_csillag_kms_{i}']
            vk, vmod, sig = sor[f'v_kepler_kms_{i}'], sor[f'v_mod_{i}'], sor[f'sigma_kms_{i}']

            if mod == "spiral":
                jeloltek = [x for x in [vg, vs] if np.isfinite(x) and x > 10]
                vm = max(jeloltek) if jeloltek else np.nan
            else:
                if np.isfinite(sig):
                    v_rot_alap = vs if np.isfinite(vs) else 0.0
                    vm = np.sqrt(v_rot_alap**2 + 2.0 * (sig**2))
                else:
                    vm = np.nan

            if np.isfinite(vm) and np.isfinite(vk) and vk > 0:
                vm_lista.append(vm)
                vk_lista.append(vk)
                if np.isfinite(vmod):
                    vmod_lista.append(vmod)

        v_mert_kulso.append(np.mean(vm_lista) if vm_lista else np.nan)
        v_kep_kulso.append(np.mean(vk_lista) if vk_lista else np.nan)
        v_mod_kulso.append(np.mean(vmod_lista) if vmod_lista else np.nan)

    tabla['V_mert_kulso'] = v_mert_kulso
    tabla['V_kep_kulso'] = v_kep_kulso
    tabla['V_mod_kulso'] = v_mod_kulso
    tabla['Anomalia_mert'] = tabla['V_mert_kulso'] / tabla['V_kep_kulso']
    tabla['Anomalia_mod'] = tabla['V_mod_kulso'] / tabla['V_kep_kulso']
    tabla['Relativ_hiba_pct'] = ((tabla['V_mod_kulso'] - tabla['V_mert_kulso']) / tabla['V_mert_kulso']) * 100.0
    return tabla

spiral = szamol_kulso_atlagok(spiral, mod="spiral")
elliptikus = szamol_kulso_atlagok(elliptikus, mod="elliptikus")

sp_tiszta = spiral.dropna(subset=['V_mert_kulso', 'V_kep_kulso', 'V_mod_kulso', 'csillagtomeg_mnap'])

#el_tiszta = elliptikus.dropna(subset=['V_mert_kulso', 'V_kep_kulso', 'V_mod_kulso', 'csillagtomeg_mnap'])
#el_tiszta = elliptikus.dropna(subset=['V_mert_kulso', 'V_kep_kulso', 'V_mod_kulso', 'csillagtomeg_mnap'])
#el_tiszta = el_tiszta[(el_tiszta['V_mert_kulso'] >= 100.0) & (el_tiszta['csillagtomeg_mnap'] >= 1e10)]

el_tiszta = elliptikus.dropna(subset=['V_mert_kulso', 'V_kep_kulso', 'V_mod_kulso', 'csillagtomeg_mnap'])
# Kizárjuk a műszeres felbontás alatti (<100 km/s) és a kísérőgalaxis által megzavart (>480 km/s) profilokat,
# valamint azokat, ahol a külső diszperzió irreálisan (>20%-kal) nagyobb a 2. gyűrűben mért értéknél:
v_belso_el = np.sqrt(el_tiszta['v_csillag_kms_2'].fillna(0)**2 + 2.0 * el_tiszta['sigma_kms_2']**2)
el_tiszta = el_tiszta[
    (el_tiszta['V_mert_kulso'].between(100.0, 480.0)) &
    (el_tiszta['csillagtomeg_mnap'] >= 1e10) &
    (el_tiszta['V_mert_kulso'] <= 1.20 * v_belso_el)
]


if len(sp_tiszta) > 1:
    korr_sp = np.corrcoef(sp_tiszta['V_mert_kulso'], sp_tiszta['V_mod_kulso'])[0, 1]
    print("=== SPIRÁLGALAXISOK STATISZTIKAI ÖSSZEFOGLALÓJA (KÜLSŐ KORONG) ===")
    print(f"Mért vs. Modell anomália: {sp_tiszta['Anomalia_mert'].mean():.3f} vs. {sp_tiszta['Anomalia_mod'].mean():.3f}")
    print(f"Pearson R = {korr_sp:.4f} | Átl. hiba = {np.abs(sp_tiszta['Relativ_hiba_pct']).mean():.1f} % | Eltolódás = {sp_tiszta['Relativ_hiba_pct'].mean():+.1f} %\n")

if len(el_tiszta) > 1:
    korr_el = np.corrcoef(el_tiszta['V_mert_kulso'], el_tiszta['V_mod_kulso'])[0, 1]
    print("=== ELLIPTIKUS GALAXISOK STATISZTIKAI ÖSSZEFOGLALÓJA (KÜLSŐ ZÓNA) ===")
    print(f"Mért vs. Modell anomália: {el_tiszta['Anomalia_mert'].mean():.3f} vs. {el_tiszta['Anomalia_mod'].mean():.3f}")
    print(f"Pearson R = {korr_el:.4f} | Átl. hiba = {np.abs(el_tiszta['Relativ_hiba_pct']).mean():.1f} % | Eltolódás = {el_tiszta['Relativ_hiba_pct'].mean():+.1f} %\n")

ossz_tiszta = pd.concat([sp_tiszta, el_tiszta], ignore_index=True)
if len(ossz_tiszta) > 1:
    korr_ossz = np.corrcoef(ossz_tiszta['V_mert_kulso'], ossz_tiszta['V_mod_kulso'])[0, 1]
    print(f"=== TELJES EGYESÍTETT MINTA ({len(ossz_tiszta)} DB GALAXIS) ===")
    print(f"Összesített Pearson R = {korr_ossz:.4f} | Átl. hiba = {np.abs(ossz_tiszta['Relativ_hiba_pct']).mean():.1f} % | Eltolódás = {ossz_tiszta['Relativ_hiba_pct'].mean():+.1f} %")


if len(ossz_tiszta) > 1:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))

    # Panel 1: Scaling Relations (Baryonic Tully-Fisher & Faber-Jackson)
    ax = axes[0, 0]
    if len(sp_tiszta) > 0:
        log_m_sp = np.log10(sp_tiszta['csillagtomeg_mnap'])
        txt = [r'Baryonic Keplerian ($V_{\mathrm{Kep}}$)',r'Baryonikus Kepler-féle ($V_{\mathrm{Kep}}$)']
        ax.scatter(np.log10(sp_tiszta['V_kep_kulso']), log_m_sp, alpha=0.35, color='#1f77b4', label=txt[LANG])
        txt = [r'Observed Spiral ($V_{\mathrm{rot}}$)',r'Megfigyelt spirál ($V_{\mathrm{rot}}$)']
        ax.scatter(np.log10(sp_tiszta['V_mert_kulso']), log_m_sp, alpha=0.6, color='#2ca02c', label=txt[LANG])
        txt = [r'Modified Einstein Model ($V_{K+W}$)',r'Módosított Einstein-modell ($V_{K+W}$)']
        ax.scatter(np.log10(sp_tiszta['V_mod_kulso']), log_m_sp, alpha=0.55, color='#9467bd', label=txt[LANG])
        meredekseg, konst = np.polyfit(np.log10(sp_tiszta['V_mert_kulso']), log_m_sp, 1)
        v_grid = np.linspace(np.log10(sp_tiszta['V_mert_kulso']).min(), np.log10(sp_tiszta['V_mert_kulso']).max(), 50)
        txt = [f'BTFR Slope: {meredekseg:.2f}',f'BTFR Esés: {meredekseg:.2f}']
        ax.plot(v_grid, meredekseg * v_grid + konst, 'k--', linewidth=1.4, label=txt[LANG])

    if len(el_tiszta) > 0:
        log_m_el = np.log10(el_tiszta['csillagtomeg_mnap'])
        txt = [r'Observed Elliptical ($V_{\mathrm{eff}}$)',r'Megfigyelt elliptikus ($V_{\mathrm{eff}}$)']
        ax.scatter(np.log10(el_tiszta['V_mert_kulso']), log_m_el, alpha=0.7, color='#d62728', marker='^', label=txt[LANG])
        txt = ['Elliptical Model','Elliptikus Modell']
        ax.scatter(np.log10(el_tiszta['V_mod_kulso']), log_m_el, alpha=0.55, color='#ff7f0e', marker='^', label=txt[LANG])

    txt = ["Dynamical Scaling Relations (Tully–Fisher & Faber–Jackson)","Dinamikus skálázási kapcsolatok (Tully–Fisher & Faber–Jackson)"]
    ax.set_title(txt[LANG])
    ax.set_xlabel(r"$\log_{10}(V_{\mathrm{outer}}\ [\mathrm{km/s}])$")
    ax.set_ylabel(r"$\log_{10}(M_{\star}\ [M_{\odot}])$")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=7.5, loc='lower right')

    # Panel 2: Observed vs. Model Velocity (1:1 Correlation)
    ax = axes[0, 1]
    if len(sp_tiszta) > 0:
        txt = [f'Spirals (N = {len(sp_tiszta)})',f'Spirális (N = {len(sp_tiszta)})']
        ax.scatter(sp_tiszta['V_mert_kulso'], sp_tiszta['V_mod_kulso'], alpha=0.65, color='#2ca02c', edgecolors='k', linewidth=0.3, label=txt[LANG])
    if len(el_tiszta) > 0:
        txt = [f'Ellipticals (N = {len(el_tiszta)})',f'Elliptikus (N = {len(el_tiszta)})']
        ax.scatter(el_tiszta['V_mert_kulso'], el_tiszta['V_mod_kulso'], alpha=0.75, color='#d62728', marker='^', edgecolors='k', linewidth=0.3, label=txt[LANG])

    max_v = max(ossz_tiszta['V_mert_kulso'].max(), ossz_tiszta['V_mod_kulso'].max()) * 1.05
    ax.plot([0, max_v], [0, max_v], 'k--', linewidth=1.5, label='1:1 Identity Line')
    txt = [f"Observed vs. Predicted Outer Velocity (R = {korr_ossz:.4f})",f"Megfigyelt és előrejelzett külső sebesség (R = {korr_ossz:.4f})"]
    ax.set_title(txt[LANG])
    txt = [r"Observed / Virial Outer Velocity, $V_{\mathrm{obs}}$ (km/s)",r"Megfigyelt / virális külső sebesség, $V_{\mathrm{obs}}$ (km/s)"]
    ax.set_xlabel(txt[LANG])
    txt = [r"Modified Einstein Model, $V_{K+W}$ (km/s)",r"Módosított Einstein-modell, $V_{K+W}$ (km/s)"]
    ax.set_ylabel(txt[LANG])
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=8.5)

    # Panel 3: Velocity Anomaly Ratio Distribution
    ax = axes[1, 0]
    bins = np.linspace(0.8, 3.8, 22)
    if len(sp_tiszta) > 0:
        txt = [r'Observed Spirals ($V_{\mathrm{rot}}/V_{\mathrm{Kep}}$)',r'Megfigyelt spirális ($V_{\mathrm{rot}}/V_{\mathrm{Kep}}$)']
        ax.hist(sp_tiszta['Anomalia_mert'], bins=bins, alpha=0.55, color='#2ca02c', edgecolor='black', label=txt[LANG])
    if len(el_tiszta) > 0:
        txt = [r'Observed Ellipticals ($V_{\mathrm{eff}}/V_{\mathrm{Kep}}$)',r'Megfigyelt Elliptikus ($V_{\mathrm{eff}}/V_{\mathrm{Kep}}$)']
        ax.hist(el_tiszta['Anomalia_mert'], bins=bins, alpha=0.55, color='#d62728', edgecolor='black', label=txt[LANG])
    txt = [r'Combined Model ($V_{K+W}/V_{\mathrm{Kep}}$)',r'Combinált Modell ($V_{K+W}/V_{\mathrm{Kep}}$)']
    ax.hist(ossz_tiszta['Anomalia_mod'], bins=bins, alpha=0.45, color='#9467bd', edgecolor='black', label=txt[LANG])
    txt = ['Classical Keplerian (1.0)','Klasszikus Kepler-féle (1.0)']
    ax.axvline(1.0, color='#1f77b4', linestyle='--', linewidth=1.5, label=txt[LANG])
    txt = ["Outer-Disk Velocity Anomaly Distribution","Külső korong sebesség-anomália eloszlása"]
    ax.set_title(txt[LANG])
    txt = [r"Velocity Ratio ($V / V_{\mathrm{Kep}}$)",r"Sebességarány ($V / V_{\mathrm{Kep}}$)"]
    ax.set_xlabel(txt[LANG])
    txt = ["Number of Galaxies","Galaxisok száma"]
    ax.set_ylabel(txt[LANG])
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=8)

    # Panel 4: Relative Residual vs. Stellar Mass
    ax = axes[1, 1]
    if len(sp_tiszta) > 0:
        txt = ['Spirals','Spirális']
        ax.scatter(np.log10(sp_tiszta['csillagtomeg_mnap']), sp_tiszta['Relativ_hiba_pct'], alpha=0.65, color='#2ca02c', edgecolors='k', linewidth=0.3, label=txt[LANG])
    if len(el_tiszta) > 0:
        txt = ['Ellipticals','Elliptikus']
        ax.scatter(np.log10(el_tiszta['csillagtomeg_mnap']), el_tiszta['Relativ_hiba_pct'], alpha=0.75, color='#d62728', marker='^', edgecolors='k', linewidth=0.3, label=txt[LANG])
    ax.axhline(0, color='k', linestyle='--', linewidth=1.5)
    txt = ["Model Relative Residual vs. Stellar Mass","Modell relatív reziduális vs. csillagtömeg"]
    ax.set_title(txt[LANG])
    ax.set_xlabel(r"$\log_{10}(M_{\star}\ [M_{\odot}])$")
    txt = [r"Relative Deviation: $(V_{K+W} - V_{\mathrm{obs}}) / V_{\mathrm{obs}}$ (%)",r"Relatív eltérés: $(V_{K+W} - V_{\mathrm{obs}}) / V_{\mathrm{obs}}$ (%)"]
    ax.set_ylabel(txt[LANG])
    ax.set_ylim(-60, 80)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=8.5)

    plt.tight_layout()
    if EPS :
        txt = ["f01a.eps","f01b.eps"]
        fig.savefig(txt[LANG], format="eps", bbox_inches="tight")
        print(f"\nStatisztikai ábra elmentve: {txt[LANG]}")
    else :
        txt = ["model_statistics_en.png","model_statistics_hu.png"]
        fig.savefig(txt[LANG], dpi=300)
        print(f"\nStatisztikai ábra elmentve (300 DPI): {txt[LANG]}")


