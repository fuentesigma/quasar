#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

import glob

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack

C_KMS = 299792.458
CMB_V, CMB_L, CMB_B = 369.82, 264.021, 48.253
TEMPLATES = ["dust", "stars", "m10", "unwise", "unwisescan", "mcs"]
LOG_TEMPLATES = ("stars", "unwise", "unwisescan")


def unit(l_deg, b_deg):
    l, b = np.radians(l_deg), np.radians(b_deg)
    return np.stack([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)], axis=-1)


def lb(v):
    v = np.asarray(v)
    return np.degrees(np.arctan2(v[1], v[0])) % 360, np.degrees(np.arcsin(v[2] / np.linalg.norm(v)))


GC, CMB = unit(0.0, 0.0), unit(CMB_L, CMB_B)


def template_maps():
    maps = {f.split("map_")[1].split("_NSIDE")[0]: np.load(f).astype(float)
            for f in glob.glob("data/templates/*/*.npy")}
    for k in LOG_TEMPLATES:
        maps[k] = np.log10(np.maximum(maps[k], 1.0))
    return np.column_stack([maps[k] for k in TEMPLATES])


def load(g="20.5"):
    t = Table.read(f"data/quaia_G{g}.fits")
    ra, dec = np.asarray(t["ra"], float), np.asarray(t["dec"], float)
    l, b = np.asarray(t["l"], float), np.asarray(t["b"], float)
    pix = hp.ang2pix(64, ra, dec, lonlat=True)
    T_raw = template_maps()[pix]
    z = np.asarray(t["redshift_quaia"], float)
    return dict(ra=ra, dec=dec, l=l, b=b, n=unit(l, b), z=z, y=np.log1p(z),
                zerr=np.asarray(t["redshift_quaia_err"], float),
                G=np.asarray(t["phot_g_mean_mag"], float),
                S=hp.read_map(f"data/selection_function_NSIDE64_G{g}.fits")[pix],
                T_raw=T_raw, T=(T_raw - T_raw.mean(axis=0)) / T_raw.std(axis=0))


def desi_match(d, radius_arcsec=1.0, nside=64):
    t = vstack([Table.read(f"data/desi/QSO_{c}_clustering.dat.fits") for c in ("NGC", "SGC")])
    ra, dec = np.asarray(t["RA"], float), np.asarray(t["DEC"], float)
    q = SkyCoord(ra=d["ra"] * u.deg, dec=d["dec"] * u.deg)
    idx, sep, _ = q.match_to_catalog_sky(SkyCoord(ra=ra * u.deg, dec=dec * u.deg))
    m = sep.arcsec < radius_arcsec
    out = {}
    for key, col in (("z", "Z"), ("ntile", "NTILE"), ("weight", "WEIGHT"), ("weight_comp", "WEIGHT_COMP"),
                     ("weight_zfail", "WEIGHT_ZFAIL")):
        out[key] = np.full(len(q), np.nan)
        out[key][m] = np.asarray(t[col], float)[idx[m]]
    out["north"] = np.zeros(len(q), bool)
    out["north"][m] = np.asarray(t["PHOTSYS"])[idx[m]] == b"N"

    pix = hp.ang2pix(nside, ra, dec, lonlat=True)
    npix = hp.nside2npix(nside)
    occupied = np.bincount(pix, minlength=npix) >= 10
    nb = hp.get_all_neighbours(nside, np.arange(npix))
    interior = occupied & np.all(np.where(nb >= 0, occupied[nb], False), axis=0)
    order = np.argsort(pix, kind="stable")
    per_pix = np.split(np.asarray(t["NTILE"], float)[order], np.cumsum(np.bincount(pix, minlength=npix))[:-1])
    ntile_pix = np.array([np.median(s) if len(s) else 0.0 for s in per_pix])
    qpix = hp.ang2pix(nside, d["ra"], d["dec"], lonlat=True)
    out["interior"], out["ntile_pix"] = interior[qpix], ntile_pix[qpix]
    return out


def ols(X, y):
    X = np.column_stack([np.ones(len(y)), X])
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ X.T @ y
    cov = XtX_inv * (y - X @ coef).var()
    return coef[1:], cov[1:, 1:]


def dipole(n, y, T=None):
    X = n if T is None else np.column_stack([n, T])
    coef, cov = ols(X, y)
    return -coef[:3], cov[:3, :3], coef[3:]


def pixel_galactic(nside=64):
    th, ph = hp.Rotator(coord=["C", "G"])(*hp.pix2ang(nside, np.arange(hp.nside2npix(nside))))
    return np.degrees(ph) % 360, 90 - np.degrees(th)


def describe(v, cov=None):
    l, b = lb(v)
    err = "" if cov is None else f" +- {np.sqrt(np.trace(cov) / 3) * C_KMS:3.0f}"
    to_gc = np.degrees(np.arccos(v @ GC / np.linalg.norm(v)))
    return f"{np.linalg.norm(v) * C_KMS:5.0f}{err} km/s toward (l, b) = ({l:3.0f}, {b:+3.0f}), {to_gc:3.0f} deg from GC"
