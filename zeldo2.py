import numpy as np
import matplotlib.pyplot as plt

LANG = 0
EPS = False

# 1. Kozmológiai és kalibrált téregyenlet paraméterek
H0 = 70.0               # km/s/Mpc
OMEGA_B = 0.049         # Megfigyelt barionos anyagsűrűség (Planck)
OMEGA_CDM = 0.266       # Hipotetikus hideg sötét anyag (csak a LCDM referenciához)
OMEGA_L = 0.685         # Kozmológiai tágulási tag (1/2 sqrt(K) háttér)
OMEGA_R = 9.2e-5        # Sugárzási sűrűség

# A 271 MaNGA galaxison kalibrált csatolási állandók
ALPHA_K = 0.800
BETA_W = 0.400

# Filamentum léptékarány (R_fil ~ 2 Mpc vs. maghatár R_ref ~ 50 kpc -> r/R_ref ~ 40)
LN_SCALE_FIL = np.log(40.0)
# Effektív gravitációs erősítés a Weyl-nyírásból és Kretschmann-tagból
G_EFF_RATIO = 1.0 + (ALPHA_K + 1.5 * BETA_W) * LN_SCALE_FIL

def solve_growth(omega_m0, omega_l0, g_eff_factor=1.0, delta_init=1.5e-5, z_start=1100.0, steps=4000):
    """Megoldja a kozmológiai sűrűségingadozás növekedési egyenletét x = ln(a) változóban."""
    a_start = 1.0 / (1.0 + z_start)
    x_grid = np.linspace(np.log(a_start), 0.0, steps)
    dx = x_grid[1] - x_grid[0]

    # Kezdeti feltétel a rekombinációkor (CMB hőmérséklet-ingadozásból: delta ~ 1.5e-5)
    y = np.array([delta_init, delta_init], dtype=float)

    def derivs(x, state):
        a = np.exp(x)
        e2 = omega_m0 * (a**-3) + OMEGA_R * (a**-4) + omega_l0
        dlnh_dx = 0.5 * (-3.0 * omega_m0 * (a**-3) - 4.0 * OMEGA_R * (a**-4)) / e2
        omega_m_a = (omega_m0 * (a**-3)) / e2

        d_val, d_prime = state
        # A Weyl-nyírás (sqrt(C^2)) fokozatosan kapcsol be, ahogy nő az anizotrópia
        if g_eff_factor > 1.0:
            anisotropy_turnon = 1.0 - np.exp(-abs(d_val) / 5e-4)
            eff_g = 1.0 + (g_eff_factor - 1.0) * anisotropy_turnon
        else:
            eff_g = 1.0

        d_double_prime = -(2.0 + dlnh_dx) * d_prime + 1.5 * omega_m_a * eff_g * d_val
        return np.array([d_prime, d_double_prime])

    delta_hist = np.zeros(steps)
    for i, x in enumerate(x_grid):
        delta_hist[i] = y[0]
        k1 = derivs(x, y)
        k2 = derivs(x + 0.5 * dx, y + 0.5 * dx * k1)
        k3 = derivs(x + 0.5 * dx, y + 0.5 * dx * k2)
        k4 = derivs(x + dx, y + dx * k3)
        y = y + (dx / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    a_arr = np.exp(x_grid)
    z_arr = (1.0 / a_arr) - 1.0
    return z_arr, a_arr, delta_hist

# Három modell futtatása z=1100-tól z=0-ig
z, a, d_baryon_gr = solve_growth(OMEGA_B, OMEGA_L, g_eff_factor=1.0, delta_init=1.5e-5)
_, _, d_lcdm = solve_growth(OMEGA_B + OMEGA_CDM, OMEGA_L, g_eff_factor=1.0, delta_init=1.5e-3)
_, _, d_mod_einstein = solve_growth(OMEGA_B, OMEGA_L, g_eff_factor=G_EFF_RATIO, delta_init=1.5e-5)

# Nemlineáris átmenet (delta = 1) vöröseltolódásának megkeresése
z_nl_lcdm = z[np.argmax(d_lcdm >= 1.0)] if np.any(d_lcdm >= 1.0) else np.nan
z_nl_mod = z[np.argmax(d_mod_einstein >= 1.0)] if np.any(d_mod_einstein >= 1.0) else np.nan

print("=== KOZMIKUS HÁLÓ NÖVEKEDÉSI SZIMULÁCIÓ (z = 1100 -> z = 0) ===")
print(f"Kalibrált effektív gravitációs erősítés (G_eff / G): {G_EFF_RATIO:.2f}x")
print(f"Lineáris sűrűségkontraszt ma (z = 0):")
print(f"  1. Tiszta barionos + Klasszikus GR:          delta(0) = {d_baryon_gr[-1]:.4f} (NEM alakul ki szerkezet)")
print(f"  2. Standard Lambda-CDM (sötét anyaggal):     delta(0) = {d_lcdm[-1]:.2f} (Nemlineáris határ: z = {z_nl_lcdm:.1f})")
print(f"  3. Tiszta barionos + Módosított Einstein:    delta(0) = {d_mod_einstein[-1]:.2f} (Nemlineáris határ: z = {z_nl_mod:.1f})")

# 2. Nemlineáris Zeldovics-approximáció (anizotróp sajátértékekkel)
# A kezdeti deformációs tenzor sajátértékei: lambda_1 > lambda_2 > 0 > lambda_3 (filamentum geometria)
D_mod = d_mod_einstein / d_mod_einstein[-1]
D_gr = d_baryon_gr / d_mod_einstein[-1]

lam1, lam2, lam3 = 0.96, 0.80, -0.35  # Folyamatos, törésmentes emelkedés egészen z = 0-ig (rho/rho_bar ~ 93)

def zeldovich_density(D_arr):
    denom = (1.0 - D_arr * lam1) * (1.0 - D_arr * lam2) * (1.0 - D_arr * lam3)
    # Numerikus szingularitás (kaustika) korlátozása fizikai viriális sűrűségre (max 200x)
    rho_ratio = np.where(denom > 0.005, 1.0 / denom, 200.0)
    return np.clip(rho_ratio, 0.1, 200.0)

rho_mod = zeldovich_density(D_mod)
rho_gr = zeldovich_density(D_gr)

# 3. Kétpaneles angol nyelvű ábra készítése
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

# Bal panel: Lineáris növekedés log-log skálán
txt = [r'Standard $\Lambda$CDM (with Dark Matter)',r'Standard $\Lambda$CDM (sötét anyaggal)']
ax1.loglog(1.0 + z, d_lcdm, 'k--', linewidth=2.2, label=txt[LANG])
txt = [r'Pure Baryons + $\Lambda(K, C^2)$ (No Dark Matter)',r'Tiszta barionok + $\Lambda(K, C^2)$ (nincs sötét anyag)']
ax1.loglog(1.0 + z, d_mod_einstein, '-', color='#9467bd', linewidth=2.5, label=txt[LANG])
txt = ['Pure Baryons + Classical GR','Tiszta barionok + klasszikus GR']
ax1.loglog(1.0 + z, d_baryon_gr, ':', color='#1f77b4', linewidth=2.2, label=txt[LANG])
txt = [r'Non-linear Collapse Threshold ($\delta = 1$)',r'Nemlineáris összeomlási küszöbérték ($\delta = 1$)']
ax1.axhline(1.0, color='#d62728', linestyle='-.', linewidth=1.3, label=txt[LANG])

ax1.set_xlim(1.0, 1200.0)
ax1.invert_xaxis()  # Balról jobbra halad az idő: z=1100 -> z=0
txt = ["Growth of Cosmic Density Perturbations (Recombination to Present)","Kozmikus sűrűségzavarok növekedése (rekombináció napjainkig)"]
ax1.set_title(txt[LANG])
txt = [r"Cosmic Expansion Factor, $1 + z$  ($\longleftarrow$ Early Universe | Present $\longrightarrow$)",r"Kozmikus tágulási tényező, $1 + z$  ($\longleftarrow$ Korai Univerzum | Jelen $\longrightarrow$)"]
ax1.set_xlabel(txt[LANG])
txt = [r"Linear Density Contrast, $\delta = \delta\rho / \bar{\rho}$",r"Lineáris sűrűségkontraszt, $\delta = \delta\rho / \bar{\rho}$"]
ax1.set_ylabel(txt[LANG])
ax1.grid(True, which='both', linestyle='--', alpha=0.35)
ax1.legend(fontsize=8.5, loc='upper left')

# Jobb panel: Zeldovics filamentum-sűrűsödés a késői univerzumban (z = 5 -> z = 0)
mask_late = z <= 5.0
txt = [r'Pure Baryons + $\Lambda(K, C^2)$ (Weyl Filament)',r'Tiszta barionok + $\Lambda(K, C^2)$ (Weyl Filament)']
ax2.semilogy(z[mask_late], rho_mod[mask_late], '-', color='#9467bd', linewidth=2.5, label=txt[LANG])
txt = ['Pure Baryons + Classical GR','Tiszta barionok + Klasszikus ÁR']
ax2.semilogy(z[mask_late], rho_gr[mask_late], ':', color='#1f77b4', linewidth=2.2, label=txt[LANG])
txt = [r'Cosmic Mean Density ($\bar{\rho}_{\mathrm{bar}}$)',r'Kozmikus átlagos sűrűség ($\bar{\rho}_{\mathrm{bar}}$)']
ax2.axhline(1.0, color='gray', linestyle='--', linewidth=1.0, label=txt[LANG])

ax2.set_xlim(0.0, 5.0)
ax2.invert_xaxis()
txt = ["Zel'dovich Anisotropic Filament Collapse (Late Universe)","Zel'dovich anizotrop filamentum összeomlása (késői univerzum)"]
ax2.set_title(txt[LANG])
txt = [r"Redshift, $z$",r"Vöröseltolódás, $z$"]
ax2.set_xlabel(txt[LANG])
txt = [r"Baryonic Overdensity, $\rho_{\mathrm{bar}} / \bar{\rho}_{\mathrm{bar}}$",r"Barionikus túlsűrűség, $\rho_{\mathrm{bar}} / \bar{\rho}_{\mathrm{bar}}$"]
ax2.set_ylabel(txt[LANG])
ax2.grid(True, which='both', linestyle='--', alpha=0.35)
ax2.legend(fontsize=8.5, loc='upper left')

plt.tight_layout()
txt = ["en","hu"]
if EPS :
    fig.savefig(f"cosmic_web_growth2_{txt[LANG]}.eps", format="eps", bbox_inches="tight")
    print(f"\nÁbra elmentve: cosmic_web_growth2_{txt[LANG]}.eps")
else :
    fig.savefig(f"cosmic_web_growth2_{txt[LANG]}.png", dpi=300)
    print(f"\nÁbra elmentve: cosmic_web_growth2_{txt[LANG]}.png")
