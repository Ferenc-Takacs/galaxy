import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ALPHA_K = 0.800
BETA_W = 0.400
R_REF_FIX = 2.0

LANG = 0

df = pd.read_csv("osszesitett_sebesseg_profilok.csv")
spiral = df[df['kategoria'].str.contains('Spirál', na=False)].copy()
elliptikus = df[df['kategoria'].str.contains('Elliptikus', na=False)].copy()

def gyuru_statisztika(tabla, mod="spiral"):
    sorok = []
    for i in range(1, 9):
        r_vals, vm_vals, vk_vals, vmod_vals = [], [], [], []
        for _, sor in tabla.iterrows():
            r = sor[f'r_kpc_{i}']
            vg = sor[f'v_gaz_ha_kms_{i}']
            vs = sor[f'v_csillag_kms_{i}']
            vk = sor[f'v_kepler_kms_{i}']
            vmod = sor[f'v_mod_{i}']
            sig = sor[f'sigma_kms_{i}']

            if not (np.isfinite(r) and np.isfinite(vk) and vk > 0):
                continue

            if mod == "spiral":
                jeloltek = [x for x in [vg, vs] if np.isfinite(x) and x > 10]
                vm = max(jeloltek) if jeloltek else np.nan
            else:
                if not np.isfinite(sig):
                    continue
                v_rot = vs if np.isfinite(vs) else 0.0
                vm = np.sqrt(v_rot**2 + 2.0 * (sig**2))

            if np.isfinite(vm) and np.isfinite(vmod):
                r_vals.append(r)
                vm_vals.append(vm)
                vk_vals.append(vk)
                vmod_vals.append(vmod)

        r_arr = np.array(r_vals)
        vm_arr = np.array(vm_vals)
        vk_arr = np.array(vk_vals)
        vmod_arr = np.array(vmod_vals)
        rel_hiba = ((vmod_arr - vm_arr) / vm_arr) * 100.0

        sorok.append({
            'gyuru': i,
            'n': len(vm_arr),
            'r_atlag': np.mean(r_arr),
            'v_mert': np.mean(vm_arr),
            'v_kep': np.mean(vk_arr),
            'v_mod': np.mean(vmod_arr),
            'anom_mert': np.mean(vm_arr / vk_arr),
            'anom_mod': np.mean(vmod_arr / vk_arr),
            'eltolodas_pct': np.mean(rel_hiba),
            'abs_rel_hiba_pct': np.mean(np.abs(rel_hiba))
        })
    return pd.DataFrame(sorok)

sp_rad = gyuru_statisztika(spiral, mod="spiral")
el_rad = gyuru_statisztika(elliptikus, mod="elliptikus")

print("=== SPIRÁLGALAXISOK SUGÁRIRÁNYÚ PROFILJA (1-8. GYŰRŰ) ===")
print(sp_rad[['gyuru', 'n', 'r_atlag', 'v_mert', 'v_kep', 'v_mod', 'eltolodas_pct', 'abs_rel_hiba_pct']].to_string(
    index=False, formatters={
        'r_atlag': '{:.2f}'.format, 'v_mert': '{:.1f}'.format,
        'v_kep': '{:.1f}'.format, 'v_mod': '{:.1f}'.format,
        'eltolodas_pct': '{:+.1f}%'.format, 'abs_rel_hiba_pct': '{:.1f}%'.format
    }
))

print("\n=== ELLIPTIKUS GALAXISOK SUGÁRIRÁNYÚ PROFILJA (1-8. GYŰRŰ) ===")
print(el_rad[['gyuru', 'n', 'r_atlag', 'v_mert', 'v_kep', 'v_mod', 'eltolodas_pct', 'abs_rel_hiba_pct']].to_string(
    index=False, formatters={
        'r_atlag': '{:.2f}'.format, 'v_mert': '{:.1f}'.format,
        'v_kep': '{:.1f}'.format, 'v_mod': '{:.1f}'.format,
        'eltolodas_pct': '{:+.1f}%'.format, 'abs_rel_hiba_pct': '{:.1f}%'.format
    }
))

# Skálázódó R_ref tesztelése a teljes sugárprofilra (1-8. gyűrű együtt)
def tesztel_skalazodo_rref(tabla, eta):
    abs_hibak, eltolodasok = [], []
    for _, sor in tabla.iterrows():
        r_osszes = [sor[f'r_kpc_{i}'] for i in range(1, 9) if np.isfinite(sor[f'r_kpc_{i}'])]
        if len(r_osszes) < 2:
            continue
        dr = r_osszes[1] - r_osszes[0]
        r_max = r_osszes[-1] + 0.5 * dr
        r_ref_dinamikus = eta * r_max

        for i in range(1, 9):
            r_mid = sor[f'r_kpc_{i}']
            vg = sor[f'v_gaz_ha_kms_{i}']
            vs = sor[f'v_csillag_kms_{i}']
            vk = sor[f'v_kepler_kms_{i}']
            if not (np.isfinite(r_mid) and np.isfinite(vk) and vk > 0):
                continue
            jeloltek = [x for x in [vg, vs] if np.isfinite(x) and x > 10]
            if not jeloltek:
                continue
            vm = max(jeloltek)
            v_rot = vg if (np.isfinite(vg) and (not np.isfinite(vs) or vg < 3 * vs)) else (vs if np.isfinite(vs) else vk)

            r_kulso = r_mid + 0.5 * dr
            ln_f = max(0.0, np.log(r_kulso / r_ref_dinamikus))
            v_mod = np.sqrt(vk**2 + ALPHA_K * (vk**2) * ln_f + BETA_W * (v_rot * vk) * ln_f)

            rel = ((v_mod - vm) / vm) * 100.0
            abs_hibak.append(abs(rel))
            eltolodasok.append(rel)
    return np.mean(abs_hibak), np.mean(eltolodasok)

print("\n=== BELSŐ MAGSUGÁR (R_REF) SKÁLÁZÁSI TESZT (MINDEN GYŰRŰRE) ===")
fix_abs, fix_elt = [], []
for _, sor in spiral.iterrows():
    for i in range(1, 9):
        vg, vs, vm_mod = sor[f'v_gaz_ha_kms_{i}'], sor[f'v_csillag_kms_{i}'], sor[f'v_mod_{i}']
        jel = [x for x in [vg, vs] if np.isfinite(x) and x > 10]
        if jel and np.isfinite(vm_mod):
            rel = ((vm_mod - max(jel)) / max(jel)) * 100.0
            fix_abs.append(abs(rel))
            fix_elt.append(rel)
print(f"Rögzített R_ref = 2.0 kpc:      Átl. relatív hiba = {np.mean(fix_abs):.2f} % | Eltolódás = {np.mean(fix_elt):+.2f} %")

for eta in [0.10, 0.15, 0.20, 0.25]:
    h_abs, h_elt = tesztel_skalazodo_rref(spiral, eta)
    print(f"Skálázódó R_ref = {eta:.2f} * R_max: Átl. relatív hiba = {h_abs:.2f} % | Eltolódás = {h_elt:+.2f} %")

# Ábrázolás
fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

if LANG == 0 :
    # 1. Spiral Galaxies Mean Radial Profile
    ax = axes[0, 0]
    ax.plot(sp_rad['r_atlag'], sp_rad['v_mert'], 'o-', color='#2ca02c', linewidth=2.2, label=r'Observed Rotation ($V_{\mathrm{obs}}$)')
    ax.plot(sp_rad['r_atlag'], sp_rad['v_mod'], 'd-.', color='#9467bd', linewidth=2.2, label=r'Modified Einstein Model ($V_{K+W}$)')
    ax.plot(sp_rad['r_atlag'], sp_rad['v_kep'], '--', color='#1f77b4', linewidth=2, label=r'Baryonic Keplerian ($V_{\mathrm{Kep}}$)')
    ax.set_title("Mean Radial Velocity Profile: Spiral Galaxies (N = 220)")
    ax.set_xlabel("Galactocentric Radius, $r$ (kpc)")
    ax.set_ylabel("Circular Velocity (km/s)")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()

    # 2. Elliptical Galaxies Mean Radial Profile
    ax = axes[0, 1]
    ax.plot(el_rad['r_atlag'], el_rad['v_mert'], 'o-', color='#d62728', linewidth=2.2, label=r'Observed Virial Vel. ($V_{\mathrm{eff}}$)')
    ax.plot(el_rad['r_atlag'], el_rad['v_mod'], 'd-.', color='#9467bd', linewidth=2.2, label=r'Modified Einstein Model ($V_{K+W}$)')
    ax.plot(el_rad['r_atlag'], el_rad['v_kep'], '--', color='#1f77b4', linewidth=2, label=r'Baryonic Keplerian ($V_{\mathrm{Kep}}$)')
    ax.set_title("Mean Radial Velocity Profile: Elliptical Galaxies (N = 51)")
    ax.set_xlabel("Galactocentric Radius, $r$ (kpc)")
    ax.set_ylabel("Effective Velocity (km/s)")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()

    # 3. Radial Evolution of Mass Discrepancy
    ax = axes[1, 0]
    ax.plot(sp_rad['gyuru'], sp_rad['anom_mert'], 'o-', color='#2ca02c', label=r'Observed Spirals ($V_{\mathrm{rot}}/V_{\mathrm{Kep}}$)')
    ax.plot(sp_rad['gyuru'], sp_rad['anom_mod'], 'd--', color='#2ca02c', alpha=0.7, label='Spiral Model')
    ax.plot(el_rad['gyuru'], el_rad['anom_mert'], 's-', color='#d62728', label=r'Observed Ellipticals ($V_{\mathrm{eff}}/V_{\mathrm{Kep}}$)')
    ax.plot(el_rad['gyuru'], el_rad['anom_mod'], '^--', color='#d62728', alpha=0.7, label='Elliptical Model')
    ax.axhline(1.0, color='#1f77b4', linestyle=':', linewidth=1.5, label='Newtonian Limit (1.0)')
    ax.set_title("Radial Growth of Velocity Discrepancy (Center to Edge)")
    ax.set_xlabel("Radial Annulus (1 = Inner Core, 8 = Outer Zone)")
    ax.set_ylabel(r"Velocity Ratio ($V / V_{\mathrm{Kep}}$)")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=8)

    # 4. Systematic Bias Across Radial Annuli
    ax = axes[1, 1]
    ax.plot(sp_rad['gyuru'], sp_rad['eltolodas_pct'], 'o-', color='#2ca02c', label='Spiral Systematic Bias (%)')
    ax.plot(el_rad['gyuru'], el_rad['eltolodas_pct'], 's-', color='#d62728', label='Elliptical Systematic Bias (%)')
    ax.axhline(0, color='k', linestyle='--', linewidth=1.2)
    ax.set_title("Model Systematic Bias Across Radial Zones")
    ax.set_xlabel("Radial Annulus (1–8)")
    ax.set_ylabel("Mean Relative Deviation (%)")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()

    plt.tight_layout()
    fig.savefig("radial_profiles_en.png", dpi=300)
    print("\nSugárirányú ábra elmentve (300 DPI): radial_profiles_en.png")


if LANG == 1 :
    # 1. Spirálgalaxisok átlagos sebességprofilja
    ax = axes[0, 0]
    ax.plot(sp_rad['r_atlag'], sp_rad['v_mert'], 'o-', color='#2ca02c', linewidth=2.2, label='Mért forgás ($V_{mért}$)')
    ax.plot(sp_rad['r_atlag'], sp_rad['v_mod'], 'd-.', color='#9467bd', linewidth=2.2, label='Modell ($V_{K+W}$)')
    ax.plot(sp_rad['r_atlag'], sp_rad['v_kep'], '--', color='#1f77b4', linewidth=2, label='Klasszikus Kepler ($V_{Kepler}$)')
    ax.set_title("Spirálgalaxisok átlagos sugárprofilja (220 db)")
    ax.set_xlabel("Átlagos sugár, r (kpc)")
    ax.set_ylabel("Sebesség (km/s)")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()

    # 2. Elliptikus galaxisok átlagos sebességprofilja
    ax = axes[0, 1]
    ax.plot(el_rad['r_atlag'], el_rad['v_mert'], 'o-', color='#d62728', linewidth=2.2, label='Mért viriális ($V_{eff}$)')
    ax.plot(el_rad['r_atlag'], el_rad['v_mod'], 'd-.', color='#9467bd', linewidth=2.2, label='Modell ($V_{K+W}$)')
    ax.plot(el_rad['r_atlag'], el_rad['v_kep'], '--', color='#1f77b4', linewidth=2, label='Klasszikus Kepler ($V_{Kepler}$)')
    ax.set_title("Elliptikus galaxisok átlagos sugárprofilja (51 db)")
    ax.set_xlabel("Átlagos sugár, r (kpc)")
    ax.set_ylabel("Effektív sebesség (km/s)")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()

    # 3. Normalizált anomália növekedése sugárral
    ax = axes[1, 0]
    ax.plot(sp_rad['gyuru'], sp_rad['anom_mert'], 'o-', color='#2ca02c', label='Spirál mért ($V/V_{Kep}$)')
    ax.plot(sp_rad['gyuru'], sp_rad['anom_mod'], 'd--', color='#2ca02c', alpha=0.7, label='Spirál modell')
    ax.plot(el_rad['gyuru'], el_rad['anom_mert'], 's-', color='#d62728', label='Elliptikus mért ($V_{eff}/V_{Kep}$)')
    ax.plot(el_rad['gyuru'], el_rad['anom_mod'], '^--', color='#d62728', alpha=0.7, label='Elliptikus modell')
    ax.axhline(1.0, color='#1f77b4', linestyle=':', linewidth=1.5, label='Newtoni határ (1.0)')
    ax.set_title("Tömegdeficit kialakulása a magtól a peremig")
    ax.set_xlabel("Sugárirányú zóna (1 = belső mag, 8 = külső korong)")
    ax.set_ylabel("Sebességarány ($V / V_{Kepler}$)")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=8)

    # 4. Zónánkénti relatív hiba és eltolódás
    ax = axes[1, 1]
    ax.plot(sp_rad['gyuru'], sp_rad['eltolodas_pct'], 'o-', color='#2ca02c', label='Spirál szisztematikus eltolódás (%)')
    ax.plot(el_rad['gyuru'], el_rad['eltolodas_pct'], 's-', color='#d62728', label='Elliptikus szisztematikus eltolódás (%)')
    ax.axhline(0, color='k', linestyle='--', linewidth=1.2)
    ax.set_title("Modell torzítása a galaxison belüli sugár függvényében")
    ax.set_xlabel("Sugárirányú zóna (1–8. gyűrű)")
    ax.set_ylabel("Átlagos relatív eltérés (%)")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()

    plt.tight_layout()
    fig.savefig("radial_profiles_hu.png.png", dpi=150)
    print("\nSugárirányú elemző ábra elmentve: radial_profiles_hu.png")
    
