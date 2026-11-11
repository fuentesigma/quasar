#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

import sys

import healpy as hp
import numpy as np

from quaia import C_KMS, CMB, CMB_V, TEMPLATES, describe, dipole, lb, load, ols
from results import footprint

NSIDE_ARF = 128
ELL_BINS = [(1, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 14), (14, 20), (20, 30)]
CELL_BINS = 8
CASES = ("none", "linear", "cells")
CORRECTIONS = ("raw", "templates", "cells")


def cells(T_raw, nbins=CELL_BINS):
    idx = np.zeros(len(T_raw), int)
    for name in ("dust", "unwise", "unwisescan"):
        x = T_raw[:, TEMPLATES.index(name)]
        idx = idx * nbins + np.searchsorted(np.quantile(x, np.linspace(0, 1, nbins + 1)[1:-1]), x)
    mcs = T_raw[:, TEMPLATES.index("mcs")]
    return np.unique(idx * 2 + (mcs > np.quantile(mcs, 0.99)), return_inverse=True)[1]


def demean(x, cell):
    n = np.maximum(np.bincount(cell), 1)
    if x.ndim == 1:
        return x - (np.bincount(cell, weights=x) / n)[cell]
    return np.column_stack([demean(x[:, k], cell) for k in range(x.shape[1])])


def dipole_cells(n, y, cell):
    return dipole(demean(n, cell), demean(y, cell))[0]


def dipole_samples(d):
    return {"whole sky": np.ones(len(d["y"]), bool), "|b|>30 S>0.5": footprint(d), "|b|<30": np.abs(d["b"]) < 30,
            "|b|>30 S>0.5 G<20": footprint(d) & (d["G"] < 20.0), "|b|>40 S>0.5": footprint(d, bcut=40.0),
            "|b|>30 S>0.7": footprint(d, smin=0.7)}


def corrected_dipoles(d, y, cell, samples):
    out = {}
    for name, k in samples.items():
        n, yk = d["n"][k], y[k]
        out[f"{name}: raw"] = dipole(n, yk)[0]
        out[f"{name}: templates"] = dipole(n, yk, d["T"][k])[0]
        out[f"{name}: cells"] = dipole_cells(n, yk, cell[k])
    return out


def arf_cl(pix, dz, nbar, mask):
    m = np.bincount(pix, weights=dz, minlength=hp.nside2npix(NSIDE_ARF)) / nbar
    m[~mask] = 0.0
    c = hp.anafast(m, lmax=3 * NSIDE_ARF - 1) / mask.mean()
    return np.array([c[a:b].mean() for a, b in ELL_BINS])


def arf_corrected(d, sb, pix, z, cell, nbar, mask):
    p, zb = pix[sb], z[sb]
    T = (d["T_raw"][sb] - d["T_raw"][sb].mean(axis=0)) / d["T_raw"][sb].std(axis=0)
    coef, _ = ols(T, zb)
    out = {"raw": zb - zb.mean(), "templates": zb - T @ coef, "cells": demean(zb, cell[sb])}
    return {k: arf_cl(p, v - v.mean(), nbar, mask) for k, v in out.items()}


def theory_fields(lmax):
    t = np.load("out/arf_theory.npz")
    out = []
    for key in ("C_1_1", "C_1_2", "C_2_2"):
        cl = np.zeros(lmax + 1)
        cl[1:31] = t[key]
        cl[31:] = t[key][-1] * 30.0 / np.arange(31, lmax + 1)
        out.append(cl)
    return out


def run(n_mocks=200, seed=7):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    d = load()
    N = len(d["y"])
    cell = cells(d["T_raw"])
    v_kin = CMB_V / C_KMS * CMB

    print(f"Systematics cells: {cell.max() + 1}, median {np.median(np.bincount(cell)):.0f} quasars per cell")
    v_lin, cov_lin, c_lin = dipole(d["n"], d["y"], d["T"])
    samples = dipole_samples(d)
    real_dip = corrected_dipoles(d, d["y"], cell, samples)
    print("Real data:")
    for k, v in real_dip.items():
        print(f"  {k:<26} {describe(v)}")
    s_lin = d["T"] @ c_lin
    s_lin -= s_lin.mean()

    mask = hp.ud_grade(hp.read_map("data/selection_function_NSIDE64_G20.5.fits"), NSIDE_ARF) > 0.5
    pix = hp.ang2pix(NSIDE_ARF, d["ra"], d["dec"], lonlat=True)
    keep = mask[pix]
    zmed = np.median(d["z"][keep])
    fields = theory_fields(3 * NSIDE_ARF - 1)
    print(f"ARF theory spectra at l = 1, 2, 10 (x1e6): low-z {fields[0][1] * 1e6:.3f} {fields[0][2] * 1e6:.3f} "
          f"{fields[0][10] * 1e6:.3f}; high-z {fields[2][1] * 1e6:.3f} {fields[2][2] * 1e6:.3f} {fields[2][10] * 1e6:.3f}")
    print(f"Injected: v = {CMB_V:.0f} km/s toward (l, b) = ({lb(CMB)[0]:.0f}, {lb(CMB)[1]:+.0f}); {n_mocks} mocks",
          flush=True)

    bins_real = (keep & (d["z"] < zmed), keep & (d["z"] >= zmed))
    nbars = [np.bincount(pix[sb], minlength=hp.nside2npix(NSIDE_ARF))[mask].mean() for sb in bins_real]
    noise = np.array([np.mean([arf_cl(pix[sb], rng.permutation(d["z"][sb]) - d["z"][sb].mean(), nb, mask)
                               for _ in range(20)], axis=0) for sb, nb in zip(bins_real, nbars)])
    real_arf = np.array([[arf_corrected(d, sb, pix, d["z"], cell, nb, mask)[k] for k in CORRECTIONS]
                         for sb, nb in zip(bins_real, nbars)])

    order_cell = np.argsort(cell, kind="stable")
    dip = {c: [] for c in CASES}
    arf = {c: [] for c in CASES}
    arf_true = []
    for i in range(n_mocks):
        z_glob = d["z"][rng.permutation(N)]
        z_cell = np.empty(N)
        z_cell[order_cell] = d["z"][np.lexsort((rng.random(N), cell))]
        alms = hp.synalm(fields, lmax=len(fields[0]) - 1, new=True)
        fmaps = [hp.alm2map(a, NSIDE_ARF) for a in alms]
        trues = []
        for c in CASES:
            z0 = z_cell if c == "cells" else z_glob
            hi = z0 >= zmed
            dz = np.where(hi, fmaps[1][pix], fmaps[0][pix])
            y = np.log1p(z0 + dz) - d["n"] @ v_kin + (s_lin if c == "linear" else 0.0)
            dip[c].append(list(corrected_dipoles(d, y, cell, samples).values()))
            z = np.expm1(y)
            bins = (keep & ~hi, keep & hi)
            arf[c].append([[arf_corrected(d, sb, pix, z, cell, nb, mask)[k] for k in CORRECTIONS]
                           for sb, nb in zip(bins, nbars)])
            if c == "none":
                trues = [arf_cl(pix[sb], dz[sb] - dz[sb].mean(), nb, mask) for sb, nb in zip(bins, nbars)]
        arf_true.append(trues)
        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1} mocks", flush=True)
            save(dip, arf, arf_true, noise, real_arf, v_kin, real_dip)
    save(dip, arf, arf_true, noise, real_arf, v_kin, real_dip)
    report(dict(np.load("out/redshift_mocks.npz")))


def save(dip, arf, arf_true, noise, real_arf, v_kin, real_dip):
    out = {"v_kin": v_kin, "noise": noise, "real_arf": real_arf, "arf_true": np.array(arf_true),
           "dip_names": np.array(list(real_dip)), "real_dip": np.array(list(real_dip.values()))}
    for c in CASES:
        out[f"arf_{c}"] = np.array(arf[c])
        out[f"dip_{c}"] = np.array(dip[c])
    np.savez("out/redshift_mocks.npz", **out)


def report(o):
    v_kin = o["v_kin"]
    n = len(o["arf_true"])
    print(f"\nDipole, {n} mocks, km/s. Injected {np.linalg.norm(v_kin) * C_KMS:.0f} toward the CMB dipole.")
    print("  bias = mean recovered - injected (+- error of the mean); scatter = sd per component")
    for c in CASES:
        print(f"  mocks '{c}'")
        for j, k in enumerate(o["dip_names"]):
            v = o[f"dip_{c}"][:, j]
            bias = v.mean(axis=0) - v_kin
            err = v.std(axis=0).mean() / np.sqrt(n)
            print(f"    {k:<26} bias {describe(bias)} (+-{err * C_KMS:3.0f}); scatter {v.std(axis=0).mean() * C_KMS:4.0f}")
    print("  real data against the realistic ('cells') mocks: residual = data - (mock mean bias) - injected")
    for j, k in enumerate(o["dip_names"]):
        v = o["dip_cells"][:, j]
        resid = o["real_dip"][j] - (v.mean(axis=0) - v_kin) - v_kin
        chi2 = resid @ np.linalg.inv(np.cov(v.T)) @ resid
        print(f"    {k:<26} data {describe(o['real_dip'][j])}; residual {np.linalg.norm(resid) * C_KMS:5.0f} km/s,"
              f" chi2(3 dof) {chi2:5.1f}")
    print(f"\nARF power x 1e6 (pseudo-C_l / f_sky, noise subtracted), mean over {n} mocks")
    print("  " + " " * 30 + "".join(f"{a}-{b - 1}".rjust(8) for a, b in ELL_BINS))
    for b, name in enumerate(("low-z", "high-z")):
        print(f"  {name}")
        rows = {"injected field": o["arf_true"][:, b].mean(axis=0)}
        for j, k in enumerate(CORRECTIONS):
            rows[f"real data: {k}"] = o["real_arf"][b, j] - o["noise"][b]
        for c in CASES:
            for j, k in enumerate(CORRECTIONS):
                rows[f"{c}: {k}"] = (o[f"arf_{c}"][:, b, j] - o["noise"][b]).mean(axis=0)
        for lab, v in rows.items():
            print(f"    {lab:<28}" + "".join(f"{x * 1e6:8.2f}" for x in v))
        sc = o["arf_none"][:, b, 0].std(axis=0)
        print(f"    {'scatter (none: raw)':<28}" + "".join(f"{x * 1e6:8.2f}" for x in sc))
        for j, k in enumerate(CORRECTIONS[1:], 1):
            kept = (o["arf_none"][:, b, j] - o["noise"][b]).mean(axis=0) / (o["arf_none"][:, b, 0] - o["noise"][b]).mean(axis=0)
            print(f"    {'signal kept, ' + k:<28}" + "".join(f"{x:8.2f}" for x in kept))


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        report(dict(np.load("out/redshift_mocks.npz")))
    else:
        run(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
