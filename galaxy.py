import os
import re
import glob
import json
import numpy as np
import pandas as pd
import urllib.request
import urllib.parse
from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Indexkonstansok a FITS csatornákhoz
IDX0 = 0
IDX1 = 1
IDX2 = 2
IDX3 = 3
MINTA_DB = 50

# Fizikai állandók (h = 0.7 kozmológiával)
H_PARAM = 0.7
G_KPC = 4.30091e-6  # Gravitációs állandó: kpc * (km/s)^2 / M_nap

# Kalibrált csatolási paraméterek a módosított Einstein-egyenlethez
ALPHA_K = 0.800       # Kretschmann-tag súlya
BETA_W = 0.400        # Forgásfüggő Weyl-tag súlya
R_REF_RATIO_SPIRAL = 0.15   # Galaxismérettel skálázódó belső referenciasugár (0.15 * R_max)
R_REF_RATIO_ELLIPT = 0.1    # Galaxismérettel skálázódó belső referenciasugár (0.1 * R_max)

# Hálózati kapcsolók (offline futtatáshoz False)
DOWNLOAD_IMAGES = False
DOWNLOAD_FITS = False
USE_SIMBAD = True

URL_MORPH = "https://sdss.org"
LANG = 0 

def tisztit_szoveg(oszlop):
    return [x.decode('utf-8').strip() if isinstance(x, bytes) else str(x).strip() for x in oszlop]

morph_fajl = "manga_visual_morpho-2.0.1.fits"
if not os.path.exists(morph_fajl):
    urllib.request.urlretrieve(URL_MORPH + morph_fajl, morph_fajl)

t_morph = Table.read(morph_fajl)
m_cols = {c.lower(): c for c in t_morph.colnames}
ttype_col = m_cols.get('ttype', m_cols.get('t_type'))

df_morph = pd.DataFrame({
    'plateifu': tisztit_szoveg(t_morph[m_cols['plateifu']]),
    'ttype': t_morph[ttype_col].astype(int)
})

drpall_fajl = "drpall-v3_1_1.fits"
if not os.path.exists(drpall_fajl):
    urllib.request.urlretrieve("https://sdss.org" + drpall_fajl, drpall_fajl)

t_drp = Table.read(drpall_fajl, hdu=IDX1)
d_cols = {c.lower(): c for c in t_drp.colnames}

iauname_col = d_cols.get('nsa_iauname', d_cols['plateifu'])
plateifu_lista = tisztit_szoveg(t_drp[d_cols['plateifu']])
iauname_lista = tisztit_szoveg(t_drp[iauname_col])
tiszta_nevek = [iau if (iau and iau != '-9999') else pifu for iau, pifu in zip(iauname_lista, plateifu_lista)]

mass_elpetro = t_drp[d_cols['nsa_elpetro_mass']].astype(float)
mass_sersic = t_drp[d_cols['nsa_sersic_mass']].astype(float) if 'nsa_sersic_mass' in d_cols else mass_elpetro
mass_nyers = np.where(mass_elpetro > 0, mass_elpetro, mass_sersic)

ba_elpetro = t_drp[d_cols['nsa_elpetro_ba']].astype(float)
ba_sersic = t_drp[d_cols['nsa_sersic_ba']].astype(float) if 'nsa_sersic_ba' in d_cols else ba_elpetro
ba_nyers = np.where(ba_elpetro > 0, ba_elpetro, ba_sersic)
ba_nyers = np.where(ba_nyers > 0, ba_nyers, 0.0)

df_drp = pd.DataFrame({
    'plateifu': plateifu_lista,
    'iauname': tiszta_nevek,
    'ra': t_drp[d_cols['objra']].astype(float),
    'dec': t_drp[d_cols['objdec']].astype(float),
    'ba': ba_nyers,
    'stellar_mass': np.where(mass_nyers > 0, mass_nyers / (H_PARAM ** 2), np.nan)
}).drop_duplicates(subset=['plateifu'])

df = pd.merge(df_morph, df_drp, on='plateifu', how='inner')

nagy_ifu = df['plateifu'].str.split('-').str[IDX1].str.startswith('127')
van_tomeg = df['stellar_mass'].notna() & (df['stellar_mass'] > 0)
jo_dolesszog = df['ba'].between(0.45, 0.55)

elliptikus = df[(df['ttype'] <= -3) & nagy_ifu & van_tomeg]
spiral = df[df['ttype'].between(1, 8) & nagy_ifu & van_tomeg & jo_dolesszog]
print(f"Katalógus jelöltek -> Spirál: {len(spiral)} db | Elliptikus: {len(elliptikus)} db")

def letolt_optikai_kep(plate, ifu, mappa_nev):
    mos_fajl = f"images/manga-{plate}-{ifu}-mos.jpg"
    helyi_mos = os.path.join(mappa_nev, mos_fajl)
    mos_url = f"https://data.sdss.org/sas/dr17/env/MANGA_MORPHOLOGY/manga_visual_morpho/2.0.1/{mos_fajl}"
    try:
        if not os.path.exists(helyi_mos):
            if DOWNLOAD_IMAGES:
                urllib.request.urlretrieve(mos_url, helyi_mos)
            else:
                return None
        img = mpimg.imread(helyi_mos)
        return img[:, :img.shape[1] // 4]
    except Exception:
        return None

def simbad_nev_kereses(ra, dec, galaxis_id, ttype, alapertelmezett_nev, results_mappa):
    meglevo_pngk = glob.glob(os.path.join(results_mappa, f"{galaxis_id}_T{ttype}_*.png"))
    if meglevo_pngk:
        fajlnev = os.path.basename(meglevo_pngk[0])
        elotag = f"{galaxis_id}_T{ttype}_"
        if fajlnev.startswith(elotag) and fajlnev.endswith(".png"):
            return fajlnev[len(elotag):-4]

    if USE_SIMBAD:
        try:
            adql = f"SELECT main_id FROM basic WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, 0.002)) = 1"
            params = urllib.parse.urlencode({'request': 'doQuery', 'lang': 'adql', 'format': 'json', 'query': adql})
            req = urllib.request.urlopen(f"https://u-strasbg.fr?{params}", timeout=5)
            adat = json.loads(req.read().decode('utf-8'))
            if adat.get('data'):
                nev = re.sub(r'\s+', '_', str(adat['data'][IDX0][IDX0]).strip())
                return re.sub(r'[\\/*?:"<>|]', '', nev)
        except Exception:
            pass
    return f"SDSS_{alapertelmezett_nev}"

def szamol_modositott_sebesseg(r_kpc, v_kepler, m_zart, v_rot_becsult, r_ref):
    if not np.isfinite(v_kepler) or r_kpc <= 0 or m_zart <= 0 or r_ref <= 0:
        return np.nan
    ln_faktor = max(0.0, np.log(r_kpc / r_ref))
    v_k_negyzet = ALPHA_K * (G_KPC * m_zart / r_kpc) * ln_faktor
    v_rot = v_rot_becsult if np.isfinite(v_rot_becsult) else v_kepler
    v_w_negyzet = BETA_W * (v_rot * v_kepler) * ln_faktor
    return np.sqrt(v_kepler**2 + v_k_negyzet + v_w_negyzet)

def szamol_sebesseg_profilok(hdul, ba, ossz_csillagtomeg, ttype):
    vel_star = hdul['STELLAR_VEL'].data
    mask_star = hdul['STELLAR_VEL_MASK'].data

    emline_hdr = hdul['EMLINE_GVEL'].header
    ha_csatorna = None
    for k, v in emline_hdr.items():
        if k.startswith('C') and 'Ha-6564' in str(v):
            ha_csatorna = int(k[1:]) - 1
            break

    vel_gas = hdul['EMLINE_GVEL'].data[ha_csatorna] if ha_csatorna is not None else vel_star
    mask_gas = hdul['EMLINE_GVEL_MASK'].data[ha_csatorna] if ha_csatorna is not None else mask_star

    sigma = hdul['STELLAR_SIGMA'].data
    sigma_mask = hdul['STELLAR_SIGMA_MASK'].data
    flux = hdul['SPX_MFLUX'].data
    ellcoo = hdul['SPX_ELLCOO'].data

    r_kpc = ellcoo[IDX2] / H_PARAM
    azimut_rad = np.deg2rad(ellcoo[IDX3])

    q0 = 0.2
    ba_korl = np.clip(ba, q0, 1.0)
    cos2_i = (ba_korl**2 - q0**2) / (1.0 - q0**2)
    sin_i = max(np.sqrt(max(0.0, 1.0 - cos2_i)), 0.3)

    ervenyes_star = (mask_star == 0) & np.isfinite(vel_star) & np.isfinite(r_kpc)
    ervenyes_gas = (mask_gas == 0) & np.isfinite(vel_gas) & np.isfinite(r_kpc)
    ervenyes_flux = np.isfinite(flux) & (flux > 0) & np.isfinite(r_kpc)
    ossz_flux = np.sum(flux[ervenyes_flux])

    max_r = np.nanpercentile(r_kpc[ervenyes_star], 92) if np.any(ervenyes_star) else 10.0
    r_ref_gal = R_REF_RATIO_ELLIPT * max_r if ttype <= -3 else R_REF_RATIO_SPIRAL * max_r
    sugar_hatarok = np.linspace(0.6, max_r, 9)

    r_kozepek, v_star_lista, v_gas_lista, v_kepler_lista, sigma_lista, v_mod_lista = [], [], [], [], [], []

    def gyuru_rotacio(vel_tomb, maszk_tomb, r_min, r_max):
        sel = maszk_tomb & (r_kpc >= r_min) & (r_kpc < r_max)
        if np.sum(sel) < 8:
            return np.nan
        cos_t = np.cos(azimut_rad[sel])
        v_los = vel_tomb[sel]
        med = np.median(v_los)
        mad = np.median(np.abs(v_los - med)) * 1.4826
        if mad > 0:
            tiszta = np.abs(v_los - med) < 3.0 * mad
            cos_t, v_los = cos_t[tiszta], v_los[tiszta]
        if len(v_los) < 6 or np.std(cos_t) < 0.2:
            return np.nan
        meredekseg, _ = np.polyfit(cos_t, v_los, 1)
        return abs(meredekseg) / sin_i

    for idx in range(len(sugar_hatarok) - 1):
        r_belso, r_kulso = sugar_hatarok[idx], sugar_hatarok[idx + 1]
        r_mid = 0.5 * (r_belso + r_kulso)

        v_st = gyuru_rotacio(vel_star, ervenyes_star, r_belso, r_kulso)
        v_gs = gyuru_rotacio(vel_gas, ervenyes_gas, r_belso, r_kulso)

        gyuru_sig = (sigma_mask == 0) & np.isfinite(sigma) & (r_kpc >= r_belso) & (r_kpc < r_kulso)
        sig_med = np.median(sigma[gyuru_sig]) if np.any(gyuru_sig) else np.nan

        if np.isfinite(ossz_csillagtomeg) and ossz_csillagtomeg > 0 and ossz_flux > 0:
            zart_flux = np.sum(flux[ervenyes_flux & (r_kpc <= r_kulso)])
            m_zart = ossz_csillagtomeg * (zart_flux / ossz_flux)
            v_kep = np.sqrt(G_KPC * m_zart / r_kulso)
            v_akt = v_gs if (np.isfinite(v_gs) and (not np.isfinite(v_st) or v_gs < 3.0 * v_st)) else v_st
            v_mod = szamol_modositott_sebesseg(r_kulso, v_kep, m_zart, v_akt, r_ref_gal)
        else:
            v_kep, v_mod = np.nan, np.nan

        r_kozepek.append(r_mid)
        v_star_lista.append(v_st)
        v_gas_lista.append(v_gs)
        v_kepler_lista.append(v_kep)
        sigma_lista.append(sig_med)
        v_mod_lista.append(v_mod)

    return np.array(r_kozepek), np.array(v_star_lista), np.array(v_gas_lista), np.array(v_kepler_lista), np.array(sigma_lista), np.array(v_mod_lista)

osszes_profil_sor = []

def feldolgoz_csoportot(galaxis_lista, kategoria_nev, mappa_nev):
    fits_mappa = os.path.join(mappa_nev, "fits")
    results_mappa = os.path.join(mappa_nev, "results")
    images_mappa = os.path.join(mappa_nev, "images")
    os.makedirs(fits_mappa, exist_ok=True)
    os.makedirs(results_mappa, exist_ok=True)
    os.makedirs(images_mappa, exist_ok=True)

    print(f"\n--- {kategoria_nev} galaxisok feldolgozása ---")
    idx = 0

    for _, sor in galaxis_lista.iterrows():
        galaxis_id = sor['plateifu']
        ttype = sor['ttype']
        plate, ifu = galaxis_id.split('-')

        map_fajl = f"manga-{plate}-{ifu}-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz"
        helyi_fits = os.path.join(fits_mappa, map_fajl)

        if not os.path.exists(helyi_fits):
            if not DOWNLOAD_FITS or idx >= MINTA_DB:
                continue
            try:
                map_url = f"https://data.sdss.org/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/{plate}/{ifu}/{map_fajl}"
                urllib.request.urlretrieve(map_url, helyi_fits)
            except Exception as dl_err:
                print(f"Letöltési hiba ({galaxis_id}): {dl_err}")
                continue

        csillagaszati_nev = simbad_nev_kereses(sor['ra'], sor['dec'], galaxis_id, ttype, sor['iauname'], results_mappa)
        helyi_png = os.path.join(results_mappa, f"{galaxis_id}_T{ttype}_{csillagaszati_nev}.png")

        try:
            optikai_kivagat = letolt_optikai_kep(plate, ifu, mappa_nev)
            if DOWNLOAD_IMAGES or optikai_kivagat is None:
                continue
            print(f"Feldolgoz {helyi_png}")

            with fits.open(helyi_fits) as hdul:
                sebesseg = hdul['STELLAR_VEL'].data
                maszk = hdul['STELLAR_VEL_MASK'].data
                r_kpc, v_star, v_gas, v_kep, v_sig, v_mod = szamol_sebesseg_profilok(hdul, sor['ba'], sor['stellar_mass'], ttype)

            sor_adat = {
                'no': idx + 1,
                'plateifu': galaxis_id,
                'csillagaszati_nev': csillagaszati_nev,
                'kategoria': kategoria_nev,
                'ttype': ttype,
                'tengelyarany_ba': round(sor['ba'], 3),
                'csillagtomeg_mnap': round(sor['stellar_mass'], 1) if np.isfinite(sor['stellar_mass']) else None
            }

            for i in range(8):
                s = i + 1
                sor_adat[f'r_kpc_{s}'] = round(r_kpc[i], 3) if (i < len(r_kpc) and np.isfinite(r_kpc[i])) else None
                sor_adat[f'v_csillag_kms_{s}'] = round(v_star[i], 2) if (i < len(v_star) and np.isfinite(v_star[i])) else None
                sor_adat[f'v_gaz_ha_kms_{s}'] = round(v_gas[i], 2) if (i < len(v_gas) and np.isfinite(v_gas[i])) else None
                sor_adat[f'v_kepler_kms_{s}'] = round(v_kep[i], 2) if (i < len(v_kep) and np.isfinite(v_kep[i])) else None
                sor_adat[f'sigma_kms_{s}'] = round(float(v_sig[i]), 2) if (i < len(v_sig) and np.isfinite(v_sig[i])) else None
                sor_adat[f'v_mod_{s}'] = round(v_mod[i], 2) if (i < len(v_mod) and np.isfinite(v_mod[i])) else None

            kulso_aranyok = []
            for idx_k in range(5, 8):
                if idx_k < len(v_kep) and np.isfinite(v_kep[idx_k]) and v_kep[idx_k] > 0:
                    jeloltek = [x for x in [v_gas[idx_k], v_star[idx_k]] if np.isfinite(x) and x > 10]
                    if jeloltek:
                        kulso_aranyok.append(max(jeloltek) / v_kep[idx_k])

            sor_adat['anomalia_arany_kulso'] = round(float(np.mean(kulso_aranyok)), 3) if kulso_aranyok else None
            osszes_profil_sor.append(sor_adat)

            tiszta_sebesseg = np.ma.masked_where(maszk > 0, sebesseg)
            if tiszta_sebesseg.count() > 0:
                tiszta_sebesseg = tiszta_sebesseg - np.ma.median(tiszta_sebesseg)
            vmax = max(np.nanpercentile(np.abs(tiszta_sebesseg.compressed()), 98) if tiszta_sebesseg.count() > 0 else 200, 50)

            fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 4.8))
            if optikai_kivagat is not None:
                ax0.imshow(optikai_kivagat)
                txt = ["Optical image (SDSS)","Optikai kép (SDSS)"]
                ax0.set_title(txt[LANG])
            else:
                txt = ["Optical image\n(offline mode)","Optikai kép\n(offline mód)"]
                ax0.text(0.5, 0.5, txt[LANG], ha='center', va='center')
            ax0.axis('off')

            kep = ax1.imshow(tiszta_sebesseg, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            txt = ["Line of sight speed (km/s)","Látóirányú sebesség (km/s)"]
            plt.colorbar(kep, ax=ax1, label=txt[LANG])
            ax1.set_title(f"{csillagaszati_nev} ({galaxis_id}, T={ttype})")
            ax1.set_xlabel("X pixel")
            ax1.set_ylabel("Y pixel")

            if len(r_kpc) > 0:
                if np.any(np.isfinite(v_gas)):
                    txt = ["Ionized gas ($H\\alpha$ rotation)","Ionizált gáz ($H\\alpha$ rotation)"]
                    ax2.plot(r_kpc, v_gas, 'o-', color='#2ca02c', linewidth=2.2, label=txt[LANG])
                if np.any(np.isfinite(v_star)):
                    txt = ["Velocity of stars ($V_{star}$)","Csillagok sebessége ($V_{csillag}$)"]
                    ax2.plot(r_kpc, v_star, 's-', color='#d62728', linewidth=1.8, alpha=0.85, label=txt[LANG])
                txt = ["Kepler velocity ($V_{Kepler}$)","Kepler-sebesség ($V_{Kepler}$)"]
                ax2.plot(r_kpc, v_kep, '--', color='#1f77b4', linewidth=2, label=txt[LANG])
                if np.any(np.isfinite(v_sig)):
                    txt = ["Velocity dispersion ($\\sigma$)","Sebességdiszperzió ($\\sigma$)"]
                    ax2.plot(r_kpc, v_sig, '^:', color='#7f7f7f', alpha=0.8, label=txt[LANG])
                if np.any(np.isfinite(v_mod)):
                    txt = ["Modified model ($V_{K+W}$)","Módosított modell ($V_{K+W}$)"]
                    ax2.plot(r_kpc, v_mod, 'd-.', color='#9467bd', linewidth=2, label=txt[LANG])
            txt = ["Radial velocity profiles","Sugárirányú sebességprofilok"]
            ax2.set_title(txt[LANG])
            txt = ["Radius, r (kpc)","Sugár, r (kpc)"]
            ax2.set_xlabel(txt[LANG])
            txt = ["Speed ​​(km/s)","Sebesség (km/s)"]
            ax2.set_ylabel(txt[LANG])
            ax2.grid(True, linestyle='--', alpha=0.5)
            ax2.legend()

            plt.tight_layout()
            fig.savefig(helyi_png, dpi=150)
            plt.close(fig)
            idx += 1

        except Exception as hiba:
            print(f"Hiba történt a(z) {galaxis_id} galaxisnál: {hiba}")

    print(f"Feldolgozott {kategoria_nev} galaxisok száma: {idx} db")

feldolgoz_csoportot(spiral, "Spirálgalaxis", "spiral_adatok")
feldolgoz_csoportot(elliptikus, "Elliptikus galaxis", "elliptikus_adatok")

df_eredmeny = pd.DataFrame(osszes_profil_sor)
df_eredmeny.to_csv("osszesitett_sebesseg_profilok.csv", index=False, encoding='utf-8-sig')
print("\nKész! Az összesített adatok elmentve: osszesitett_sebesseg_profilok.csv")

import statisztika
