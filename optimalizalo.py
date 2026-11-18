import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

R_REF = 2.0  # A belső magsugár referenciaértéke (kpc)

# 1. CSV betöltése
df = pd.read_csv("osszesitett_sebesseg_profilok.csv")
print(f"Betöltött galaxisok: {len(df)} db")

# Segédfüggvény: egy galaxis külső gyűrűinek (6-8. gyűrű) újraszámolása adott (alpha_k, beta_w) mellett
def kiertekel_parametereket(tabla, alpha_k, beta_w, mod="spiral"):
    v_mert_lista, v_kep_lista, v_mod_lista = [], [], []

    for _, sor in tabla.iterrows():
        # Gyűrűszélesség (dr) meghatározása az r_kpc oszlopokból
        r_vals = [sor[f'r_kpc_{i}'] for i in range(1, 9) if np.isfinite(sor[f'r_kpc_{i}'])]
        dr = (r_vals[1] - r_vals[0]) if len(r_vals) >= 2 else 0.0

        vm_gyuruk, vk_gyuruk, vmod_gyuruk = [], [], []

        for i in range(6, 9):  # 6., 7. és 8. külső gyűrű
            r_mid = sor[f'r_kpc_{i}']
            vg = sor[f'v_gaz_ha_kms_{i}']
            vs = sor[f'v_csillag_kms_{i}']
            vk = sor[f'v_kepler_kms_{i}']
            sig = sor[f'sigma_kms_{i}']

            if not (np.isfinite(r_mid) and np.isfinite(vk) and vk > 0):
                continue

            r_kulso = r_mid + 0.5 * dr
            ln_faktor = max(0.0, np.log(r_kulso / R_REF))

            if mod == "spiral":
                jeloltek = [x for x in [vg, vs] if np.isfinite(x) and x > 10]
                vm = max(jeloltek) if jeloltek else np.nan
                # A galaxy.py logikája szerinti aktuális forgási sebesség a Weyl-taghoz
                if np.isfinite(vg) and (not np.isfinite(vs) or vg < 3 * vs):
                    v_rot = vg
                else:
                    v_rot = vs if np.isfinite(vs) else vk
            else:
                # Elliptikus galaxisok: viriális effektív sebesség és tiszta csillagrotáció
                if not np.isfinite(sig):
                    continue
                v_rot = vs if np.isfinite(vs) else 0.0
                vm = np.sqrt(v_rot**2 + 2.0 * (sig**2))

            if not np.isfinite(vm):
                continue

            v_k_negyzet = alpha_k * (vk**2) * ln_faktor
            v_w_negyzet = beta_w * (v_rot * vk) * ln_faktor
            v_mod = np.sqrt(vk**2 + v_k_negyzet + v_w_negyzet)

            vm_gyuruk.append(vm)
            vk_gyuruk.append(vk)
            vmod_gyuruk.append(v_mod)

        v_mert_lista.append(np.mean(vm_gyuruk) if vm_gyuruk else np.nan)
        v_kep_lista.append(np.mean(vk_gyuruk) if vk_gyuruk else np.nan)
        v_mod_lista.append(np.mean(vmod_gyuruk) if vmod_gyuruk else np.nan)

    res = pd.DataFrame({
        'V_mert': v_mert_lista,
        'V_kep': v_kep_lista,
        'V_mod': v_mod_lista
    }).dropna()

    rel_hiba = ((res['V_mod'] - res['V_mert']) / res['V_mert']) * 100.0
    abs_rel_hiba = np.abs(rel_hiba)
    rmse = np.sqrt(np.mean((res['V_mod'] - res['V_mert'])**2))
    corr = np.corrcoef(res['V_mert'], res['V_mod'])[0, 1] if len(res) > 1 else np.nan

    return {
        'n': len(res),
        'anomalia_mert': (res['V_mert'] / res['V_kep']).mean(),
        'anomalia_mod': (res['V_mod'] / res['V_kep']).mean(),
        'eltolodas_pct': rel_hiba.mean(),
        'abs_rel_hiba_pct': abs_rel_hiba.mean(),
        'median_rel_hiba_pct': np.median(abs_rel_hiba),
        'rmse': rmse,
        'corr': corr,
        'df': res
    }

spiral = df[df['kategoria'].str.contains('Spirál', na=False)].copy()
elliptikus = df[df['kategoria'].str.contains('Elliptikus', na=False)].copy()

# 2. Alapállapot (ALPHA_K = 1.0, BETA_W = 0.5) kiértékelése mindkét csoporton
print("\n=== 1. JELENLEGI ALAPÁLLAPOT (ALPHA_K = 1.00, BETA_W = 0.50) ===")
sp_alap = kiertekel_parametereket(spiral, 1.0, 0.5, mod="spiral")
el_alap = kiertekel_parametereket(elliptikus, 1.0, 0.5, mod="elliptikus")

print(f"Spirálgalaxisok ({sp_alap['n']} db):")
print(f"  - Mért vs. Modell anomália: {sp_alap['anomalia_mert']:.3f} vs. {sp_alap['anomalia_mod']:.3f}")
print(f"  - Szisztematikus eltolódás: {sp_alap['eltolodas_pct']:+.2f} % | Átl. relatív hiba: {sp_alap['abs_rel_hiba_pct']:.2f} % | R = {sp_alap['corr']:.4f}")

print(f"Elliptikus galaxisok ({el_alap['n']} db):")
print(f"  - Mért (viriális) vs. Modell anomália: {el_alap['anomalia_mert']:.3f} vs. {el_alap['anomalia_mod']:.3f}")
print(f"  - Szisztematikus eltolódás: {el_alap['eltolodas_pct']:+.2f} % | Átl. relatív hiba: {el_alap['abs_rel_hiba_pct']:.2f} % | R = {el_alap['corr']:.4f}")

# 3. Kétlépcsős fizikai kalibráció
# A) ALPHA_K meghatározása az elliptikus galaxisokon (minimális Weyl-járulék mellett)
alpha_racs = np.linspace(0.3, 2.0, 171)
legjobb_alpha = min(alpha_racs, key=lambda a: abs(kiertekel_parametereket(elliptikus, a, 0.0, mod="elliptikus")['eltolodas_pct']))

# B) BETA_W meghatározása a spirálgalaxisokon a rögzített ALPHA_K mellett
beta_racs = np.linspace(0.0, 1.2, 121)
legjobb_beta = min(beta_racs, key=lambda b: abs(kiertekel_parametereket(spiral, legjobb_alpha, b, mod="spiral")['eltolodas_pct']))

print(f"\n=== 2. FIZIKAI KALIBRÁCIÓ (NULLA ELTOLÓDÁSRA HANGOLVA) ===")
print(f"Elliptikusokból adódó ALPHA_K (tiszta Kretschmann-tag): {legjobb_alpha:.3f}")
print(f"Spirálgalaxisokból adódó BETA_W (Weyl-tag):              {legjobb_beta:.3f}")

sp_kal = kiertekel_parametereket(spiral, legjobb_alpha, legjobb_beta, mod="spiral")
el_kal = kiertekel_parametereket(elliptikus, legjobb_alpha, legjobb_beta, mod="elliptikus")

print(f"Kalibrált Spirálgalaxisok:")
print(f"  - Mért vs. Modell anomália: {sp_kal['anomalia_mert']:.3f} vs. {sp_kal['anomalia_mod']:.3f}")
print(f"  - Szisztematikus eltolódás: {sp_kal['eltolodas_pct']:+.2f} % | Átl. relatív hiba: {sp_kal['abs_rel_hiba_pct']:.2f} % (Medián: {sp_kal['median_rel_hiba_pct']:.2f} %) | R = {sp_kal['corr']:.4f}")

print(f"Kalibrált Elliptikus galaxisok:")
print(f"  - Mért vs. Modell anomália: {el_kal['anomalia_mert']:.3f} vs. {el_kal['anomalia_mod']:.3f}")
print(f"  - Szisztematikus eltolódás: {el_kal['eltolodas_pct']:+.2f} % | Átl. relatív hiba: {el_kal['abs_rel_hiba_pct']:.2f} % (Medián: {el_kal['median_rel_hiba_pct']:.2f} %) | R = {el_kal['corr']:.4f}")

# 4. Közös globális optimum keresése (összesített relatív hiba minimalizálása a 271 galaxison)
legjobb_ossz_hiba = np.inf
opt_a, opt_b = legjobb_alpha, legjobb_beta

for a in np.linspace(0.5, 1.6, 56):
    for b in np.linspace(0.0, 1.0, 51):
        s_res = kiertekel_parametereket(spiral, a, b, mod="spiral")
        e_res = kiertekel_parametereket(elliptikus, a, b, mod="elliptikus")
        # Súlyozott átlagos relatív hiba + kis büntetés a szisztematikus eltolódásra
        ossz_hiba = (s_res['abs_rel_hiba_pct'] * s_res['n'] + e_res['abs_rel_hiba_pct'] * e_res['n']) / (s_res['n'] + e_res['n'])
        ossz_hiba += 0.25 * (abs(s_res['eltolodas_pct']) + abs(e_res['eltolodas_pct']))
        if ossz_hiba < legjobb_ossz_hiba:
            legjobb_ossz_hiba = ossz_hiba
            opt_a, opt_b = a, b

print(f"\n=== 3. GLOBÁLIS OPTIMUM A TELJES 271 GALAXISOS MINTÁN ===")
print(f"Optimális paraméterek: ALPHA_K = {opt_a:.3f}, BETA_W = {opt_b:.3f}")
sp_opt = kiertekel_parametereket(spiral, opt_a, opt_b, mod="spiral")
el_opt = kiertekel_parametereket(elliptikus, opt_a, opt_b, mod="elliptikus")
print(f"  - Spirál eltolódás:     {sp_opt['eltolodas_pct']:+.2f} % | Átl. relatív hiba: {sp_opt['abs_rel_hiba_pct']:.2f} % | R = {sp_opt['corr']:.4f}")
print(f"  - Elliptikus eltolódás: {el_opt['eltolodas_pct']:+.2f} % | Átl. relatív hiba: {el_opt['abs_rel_hiba_pct']:.2f} % | R = {el_opt['corr']:.4f}")

# 5. Összehasonlító ábra mentése
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
ax.scatter(sp_opt['df']['V_mert'], sp_opt['df']['V_mod'], alpha=0.65, color='#2ca02c', label=f"Spirál ({sp_opt['n']} db)")
ax.scatter(el_opt['df']['V_mert'], el_opt['df']['V_mod'], alpha=0.75, color='#d62728', marker='^', label=f"Elliptikus ({el_opt['n']} db)")
max_v = max(sp_opt['df']['V_mert'].max(), el_opt['df']['V_mert'].max()) * 1.05
ax.plot([0, max_v], [0, max_v], 'k--', linewidth=1.5, label='1:1 egyezés')
ax.set_title(f"Kalibrált modell (α_K={opt_a:.2f}, β_W={opt_b:.2f})")
ax.set_xlabel("Mért / Viriális sebesség (km/s)")
ax.set_ylabel("Modell sebesség (km/s)")
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend()

ax = axes[1]
kategoriak = ['Spirál\n(Alap)', 'Spirál\n(Kalibrált)', 'Elliptikus\n(Alap)', 'Elliptikus\n(Kalibrált)']
eltolodasok = [sp_alap['eltolodas_pct'], sp_opt['eltolodas_pct'], el_alap['eltolodas_pct'], el_opt['eltolodas_pct']]
szinek = ['#9467bd', '#2ca02c', '#ff7f0e', '#d62728']
ax.bar(kategoriak, eltolodasok, color=szinek, edgecolor='black', alpha=0.8)
ax.axhline(0, color='k', linewidth=1.2)
ax.set_title("Szisztematikus eltolódás csökkenése (%)")
ax.set_ylabel("Átlagos relatív eltérés (%)")
ax.grid(True, axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
fig.savefig("kalibracio_eredmeny.png", dpi=150)
print("\nÖsszehasonlító ábra elmentve: kalibracio_eredmeny.png")
