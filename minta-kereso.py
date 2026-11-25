import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("osszesitett_sebesseg_profilok.csv")
LANG = 0
EPS = False

# Külső gyűrűk (6-8.) metrikáinak kiszámítása minden galaxisra
records = []
for idx, row in df.iterrows():
    is_spiral = "Spirál" in str(row['kategoria'])
    vm_list, vk_list, vmod_list = [], [], []
    for i in range(6, 9):
        vg, vs = row[f'v_gaz_ha_kms_{i}'], row[f'v_csillag_kms_{i}']
        vk, vmod, sig = row[f'v_kepler_kms_{i}'], row[f'v_mod_{i}'], row[f'sigma_kms_{i}']
        if is_spiral:
            cands = [x for x in [vg, vs] if np.isfinite(x) and x > 10]
            vm = max(cands) if cands else np.nan
        else:
            vm = np.sqrt((vs if np.isfinite(vs) else 0.0)**2 + 2.0 * (sig**2)) if np.isfinite(sig) else np.nan
        if np.isfinite(vm) and np.isfinite(vk) and vk > 0 and np.isfinite(vmod):
            vm_list.append(vm)
            vk_list.append(vk)
            vmod_list.append(vmod)
    if len(vm_list) == 3:
        vm_mean, vk_mean, vmod_mean = np.mean(vm_list), np.mean(vk_list), np.mean(vmod_list)
        rel_err = abs(vmod_mean - vm_mean) / vm_mean * 100.0
        records.append({
            'df_idx': idx,
            'plateifu': row['plateifu'],
            'name': row['csillagaszati_nev'],
            'is_spiral': is_spiral,
            'ttype': row['ttype'],
            'mass': row['csillagtomeg_mnap'],
            'vm_outer': vm_mean,
            'vk_outer': vk_mean,
            'vmod_outer': vmod_mean,
            'anom_mert': vm_mean / vk_mean,
            'rel_err': rel_err
        })

meta = pd.DataFrame(records)
sp_meta = meta[meta['is_spiral']]
el_meta = meta[~meta['is_spiral']]

# 1. Tipikus közepes tömegű spirálgalaxis (10^10 - 4*10^10 M_sun, kis hiba)
typ_sp = sp_meta[(sp_meta['mass'].between(1e10, 4e10)) & (sp_meta['anom_mert'].between(1.7, 2.3))].sort_values('rel_err').iloc[0]

# 2. Óriás spirálgalaxis (> 6*10^10 M_sun, V_outer > 240 km/s, kis hiba)
giant_sp = sp_meta[(sp_meta['mass'] > 6e10) & (sp_meta['vm_outer'] > 240)].sort_values('rel_err').iloc[0]

# 3. Tipikus magas anomáliájú elliptikus galaxis (anomália 1.7 - 2.2 között, kis hiba)
typ_el = el_meta[el_meta['anom_mert'].between(1.7, 2.2)].sort_values('rel_err').iloc[0]

# 4. Kepler-szerű elliptikus galaxis (legkisebb mért anomália arány az elliptikusok között)
#kep_el = el_meta.sort_values('anom_mert').iloc[0]
# 4. Valódi, nagy tömegű Kepler-szerű elliptikus galaxis (M_* > 3e10 M_sun, anomália 1.0 - 1.35 között)
kep_el = el_meta[(el_meta['mass'] > 3e10) & (el_meta['anom_mert'].between(1.0, 1.35))].sort_values('anom_mert').iloc[0]

kivalasztottak = [
    (["Typical Spiral Galaxy","Tipikus Spirál Galaxis"], typ_sp),
    (["Massive Spiral Galaxy","Hatalmas Spirál Galaxis"], giant_sp),
    (["Standard Elliptical Galaxy","Standard Elliptikus Galaxis"], typ_el),
    (["Kepler-like Elliptical Galaxy","Kepler-féle Elliptikus Galaxis"], kep_el)
]

print("=== KIVÁLASZTOTT MINTAGALAXISOK A CIKKHEZ ===")
for cim, item in kivalasztottak:
    print(f"\n[{cim[LANG]}]")
    print(f"  ID: {item['plateifu']} | Név: {item['name']} | T-type: {item['ttype']}")
    print(f"  Csillagtömeg: {item['mass']:.2e} M_sun")
    print(f"  Külső V_mért: {item['vm_outer']:.1f} km/s | V_Kepler: {item['vk_outer']:.1f} km/s | V_mod: {item['vmod_outer']:.1f} km/s")
    print(f"  Anomália (V_mért / V_Kep): {item['anom_mert']:.2f} | Modell eltérés: {item['rel_err']:.1f} %")
    
fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
axes_flat = axes.flatten()

for ax, (cim, item) in zip(axes_flat, kivalasztottak):
    row = df.iloc[item['df_idx']]
    r = np.array([row[f'r_kpc_{i}'] for i in range(1, 9)], dtype=float)
    vs = np.array([row[f'v_csillag_kms_{i}'] for i in range(1, 9)], dtype=float)
    vg = np.array([row[f'v_gaz_ha_kms_{i}'] for i in range(1, 9)], dtype=float)
    vk = np.array([row[f'v_kepler_kms_{i}'] for i in range(1, 9)], dtype=float)
    sig = np.array([row[f'sigma_kms_{i}'] for i in range(1, 9)], dtype=float)
    vmod = np.array([row[f'v_mod_{i}'] for i in range(1, 9)], dtype=float)

    if item['is_spiral']:
        if np.any(np.isfinite(vg)):
            txt = [r'Ionized Gas ($H\alpha$ rotation)',r'Ionizált Gáz ($H\alpha$ sebesség)']
            ax.plot(r, vg, 'o-', color='#2ca02c', linewidth=2.2, label=txt[LANG])
        if np.any(np.isfinite(vs)):
            txt = [r'Stellar Rotation ($V_{\star}$)',r'Csillag sebesség ($V_{\star}$)']
            ax.plot(r, vs, 's-', color='#d62728', linewidth=1.6, alpha=0.85, label=txt[LANG])
        txt = [r'Velocity Dispersion ($\sigma$)',r'Sebességdiszperzió ($\sigma$)']
        ax.plot(r, sig, '^:', color='#7f7f7f', linewidth=1.6, label=txt[LANG])
    else:
        v_eff = np.sqrt(np.nan_to_num(vs, nan=0.0)**2 + 2.0 * (sig**2))
        txt = [r'Virial Effective Vel. ($\sqrt{V_{\star}^2 + 2\sigma^2}$)',r'Virális effektív sebesség. ($\sqrt{V_{\star}^2 + 2\sigma^2}$)']
        ax.plot(r, v_eff, 'o-', color='#d62728', linewidth=2.2, label=txt[LANG])
        txt = [r'Velocity Dispersion ($\sigma$)',r'Sebességdiszperzió ($\sigma$)']
        ax.plot(r, sig, '^:', color='#7f7f7f', linewidth=1.6, label=txt[LANG])

    txt = [r'Baryonic Keplerian ($V_{\mathrm{Kep}}$)',r'Kepler-féle barionos($V_{\mathrm{Kep}}$)']
    ax.plot(r, vk, '--', color='#1f77b4', linewidth=2.0, label=txt[LANG])
    txt = [r'Modified Einstein Model ($V_{K+W}$)',r'Módosított Einstein-modell ($V_{K+W}$)']
    ax.plot(r, vmod, 'd-.', color='#9467bd', linewidth=2.3, label=txt[LANG])

    clean_name = str(item['name']).replace('_', ' ')
    mass_exp = np.log10(item['mass'])
    ax.set_title(f"{cim[LANG]}\n{clean_name} (MaNGA {item['plateifu']}, $\log M_*/M_\odot = {mass_exp:.2f}$)", fontsize=10.5)
    txt = ["Galactocentric Radius, $r$ (kpc)","Galaktocentrikus sugár, $r$ (kpc)"]
    ax.set_xlabel(txt[LANG], fontsize=10)
    txt = ["Circular / Effective Velocity (km/s)","Körsebesség / effektív sebesség (km/s)"]
    ax.set_ylabel(txt[LANG], fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=8.5, loc='lower right')

plt.tight_layout()
if EPS :
    txt = ["f03a.eps","f03b.eps"]
    fig.savefig(txt[LANG], format="eps", bbox_inches="tight")
    print(f"\nEsettanulmányi ábra elmentve: {txt[LANG]}")
else :
    txt = ["case_studies_en.png","case_studies_hu.png"]
    fig.savefig(txt[LANG], dpi=300)
    print(f"\nEsettanulmányi ábra elmentve (300 DPI): {txt[LANG]}")
