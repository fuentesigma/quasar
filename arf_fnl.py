#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

import healpy as hp
import numpy as np
from scipy.special import spherical_jn

from arf_theory import ZMED, conditional
from fnl import ZGRID, desi_nz
from quaia import C_KMS, desi_match, load, ols
from theory import DELTA_C, Cosmology, b_laurent, growth_rate

NSIDE = 128
LMAX = (240, 300)
LMIN_AUTO, LMIN_CROSS = 10, 2
FSKY = 0.61
NKK = 1.3e-7
ELLS_EVAL = np.concatenate([np.arange(2, 61), np.arange(65, 301, 5)])


def kernels(d, D):
    m = np.isfinite(D["z"])
    zt, zo = D["z"][m], d["z"][m]
    n_all = desi_nz(d, D, np.ones(len(d["z"]), bool))
    gs = []
    for lo, hi in ((-np.inf, ZMED), (ZMED, np.inf)):
        inb = (d["z"] >= lo) & (d["z"] < hi)
        zbar, frac = d["z"][inb].mean(), inb.mean()
        w_obs = np.where((zo >= lo) & (zo < hi), zo - zbar, 0.0)
        gs.append(conditional(zt, w_obs, ZGRID, lambda gg: np.where((gg >= lo) & (gg < hi), gg - zbar, 0.0)) / frac)
    return n_all, gs


def theory(cosmo, n, gs, ells, kmax=0.3, nk=3000):
    z = ZGRID
    k = np.logspace(-5, np.log10(kmax), nk)
    pref = 2 / np.pi * k ** 2 * cosmo.pk.P(0.0, k).ravel()
    chi, Dz, f = cosmo.chi(z), cosmo.growth(z), growth_rate(cosmo, z)
    vfac = f * cosmo.H(z) / (1 + z) / C_KMS
    b = b_laurent(z)
    H0 = cosmo.h * 100
    u = 3 * DELTA_C * cosmo.om * (H0 / C_KMS) ** 2 / (k ** 2 * cosmo.T(k)) / cosmo.d0_md
    zk = np.linspace(1e-3, 10.0, 2000)
    chik = cosmo.chi(zk)
    wk = 1.5 * cosmo.om * (H0 / C_KMS) ** 2 * (1 + zk) * (cosmo.chi_star - chik) / (chik * cosmo.chi_star) \
        * C_KMS / cosmo.H(zk) * cosmo.growth(zk)
    pairs = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2)]
    C = {p: np.zeros(len(ells)) for p in pairs}
    dF = {p: np.zeros(len(ells)) for p in pairs}
    dA = {(i, p): np.zeros(len(ells)) for i in range(2) for p in pairs}
    for a, ell in enumerate(ells):
        x = np.outer(k, chi)
        jl, djl = spherical_jn(ell, x), spherical_jn(ell, x, derivative=True)
        Wb, W0, W1 = [], [], []
        for g in gs:
            wb = np.trapezoid((n * g * b * Dz)[None, :] * jl, z, axis=1)
            wv = -np.trapezoid((n * (1 + z) * np.gradient(g, z) * vfac * Dz)[None, :] * djl, z, axis=1) / k
            w1 = np.trapezoid((n * g * (b - 1))[None, :] * jl, z, axis=1) * u
            Wb.append(wb); W0.append(wb + wv); W1.append(w1)
        jk = spherical_jn(ell, np.outer(k, chik))
        lens = np.trapezoid(wk[None, :] * jk, zk, axis=1) * ell * (ell + 1) / k ** 2
        W0.append(lens); W1.append(np.zeros(nk)); Wb.append(np.zeros(nk))
        I = lambda A_, B_: np.trapezoid(pref * A_ * B_, k)
        for p in pairs:
            i, j = p
            C[p][a] = I(W0[i], W0[j])
            dF[p][a] = I(W0[i], W1[j]) + I(W1[i], W0[j])
            for s in range(2):
                dA[(s, p)][a] = (I(Wb[i], W0[j]) if i == s else 0.0) + (I(W0[i], Wb[j]) if j == s else 0.0)
    return C, dF, dA


def fisher(C, dF, dA, N, deltas, ells, lmin_auto=LMIN_AUTO, use=("auto", "cross")):
    F = np.zeros((3, 3))
    B = {name: np.zeros(3) for name in deltas}
    for a, ell in enumerate(ells):
        keys = []
        for i in range(2):
            if "auto" in use and lmin_auto <= ell <= LMAX[i]:
                keys.append((i, i))
            if "cross" in use and LMIN_CROSS <= ell <= LMAX[i]:
                keys.append((i, 2))
        if not keys:
            continue

        def tot(x, y):
            p = (min(x, y), max(x, y))
            return C[p][a] + (N[x] if x == y else 0.0)
        cov = np.array([[(tot(w, y) * tot(x, zz) + tot(w, zz) * tot(x, y)) / ((2 * ell + 1) * FSKY)
                         for (y, zz) in keys] for (w, x) in keys])
        Dm = np.array([[dF[p][a] for p in keys]] + [[dA[(s, p)][a] for p in keys] for s in range(2)])
        ci = np.linalg.inv(cov)
        F += Dm @ ci @ Dm.T
        for name, dC in deltas.items():
            B[name] += Dm @ ci @ np.array([dC[p[0]][a] if p[0] == p[1] else 0.0 for p in keys])
    Finv = np.linalg.inv(F)
    return np.sqrt(Finv[0, 0]), {name: Finv @ v for name, v in B.items()}


def measured(d, n_shuffle=20, seed=4):
    rng = np.random.default_rng(seed)
    mask = hp.ud_grade(hp.read_map("data/selection_function_NSIDE64_G20.5.fits"), NSIDE) > 0.5
    pix = hp.ang2pix(NSIDE, d["ra"], d["dec"], lonlat=True)
    keep = mask[pix]
    zmed = np.median(d["z"][keep])
    lmax = 3 * NSIDE - 1

    def cl(p, v, nbar):
        m = np.bincount(p, weights=v - v.mean(), minlength=hp.nside2npix(NSIDE)) / nbar
        m[~mask] = 0.0
        return hp.anafast(m, lmax=lmax) / mask.mean()

    out = {}
    for h, sb in enumerate((keep & (d["z"] < zmed), keep & (d["z"] >= zmed))):
        p, z = pix[sb], d["z"][sb]
        nbar = np.bincount(p, minlength=hp.nside2npix(NSIDE))[mask].mean()
        T = (d["T_raw"][sb] - d["T_raw"][sb].mean(0)) / d["T_raw"][sb].std(0)
        coef, _ = ols(T, z)
        zhat = T @ coef
        noise = lambda v: np.mean([cl(p, rng.permutation(v), nbar) for _ in range(n_shuffle)], axis=0)
        n_z, n_hat, n_cor = noise(z), noise(zhat), noise(z - zhat)
        out[h] = dict(raw=cl(p, z, nbar) - n_z, templates=cl(p, zhat, nbar) - n_hat,
                      corrected=cl(p, z - zhat, nbar) - n_cor, shot=n_z.mean(),
                      shot_analytic=z.var() / (nbar / hp.nside2pixarea(NSIDE)), zmed=zmed)
    return out


def to_ells(v, ells):
    out = np.zeros(len(ells))
    n = min(len(v), ells[-1] + 1)
    out[:n - 2] = v[2:n]
    return out


def main():
    d = load()
    D = desi_match(d)
    n, gs = kernels(d, D)
    cosmo = Cosmology()
    print(f"theory at {len(ELLS_EVAL)} multipoles from 2 to {ELLS_EVAL[-1]}", flush=True)
    Ce, dFe, dAe = theory(cosmo, n, gs, ELLS_EVAL)
    ells = np.arange(2, max(LMAX) + 1)
    C, dF, dA = {}, {}, {}
    for p in Ce:
        C[p] = np.interp(ells, ELLS_EVAL, Ce[p])
        dF[p] = np.interp(ells, ELLS_EVAL, dFe[p])
    for p in dAe:
        dA[p] = np.interp(ells, ELLS_EVAL, dAe[p])

    M = measured(d)
    N = [M[0]["shot"], M[1]["shot"], NKK]
    print(f"shot noise of the halves x1e6: {N[0] * 1e6:.3f} {N[1] * 1e6:.3f} "
          f"(analytic {M[0]['shot_analytic'] * 1e6:.3f} {M[1]['shot_analytic'] * 1e6:.3f}); split at z = {M[0]['zmed']:.2f}")

    raw = [to_ells(M[h]["raw"], ells) for h in range(2)]
    tmpl = [to_ells(M[h]["templates"], ells) for h in range(2)]
    corr = [to_ells(M[h]["corrected"], ells) for h in range(2)]
    band = (ells >= 6) & (ells <= 13)
    resid = [np.where(band, corr[h] - C[(h, h)], 0.0) for h in range(2)]
    deltas = {"templates": tmpl,
              "difference": [raw[h] - corr[h] for h in range(2)],
              "residual 6-13": resid,
              "templates + residual": [tmpl[h] + resid[h] for h in range(2)]}

    print("band powers x1e6 (high half): l, theory, dC/df_NL x1e6 per unit f_NL, raw, templates, corrected")
    for lo, hi in ((2, 4), (4, 6), (6, 8), (8, 10), (10, 14), (14, 20), (20, 30), (30, 60)):
        s = (ells >= lo) & (ells < hi)
        print(f"  {lo:3d}-{hi - 1:<3d} {C[(1, 1)][s].mean() * 1e6:8.3f} {dF[(1, 1)][s].mean() * 1e6:9.4f} "
              f"{raw[1][s].mean() * 1e6:8.3f} {tmpl[1][s].mean() * 1e6:8.3f} {corr[1][s].mean() * 1e6:8.3f}")

    print("Fisher, Bermejo-Climent setup (auto from l_min, cross from 2):")
    for use in (("auto", "cross"), ("auto",), ("cross",)):
        sig, sh = fisher(C, dF, dA, N, deltas, ells, use=use)
        print(f"  {'+'.join(use):<10} sigma(f_NL) {sig:6.1f}")
        for name in deltas:
            s = sh[name]
            print(f"    {name:<22} Delta f_NL {s[0]:+7.1f} ({s[0] / sig:+.2f} sigma);  Delta A {s[1]:+.3f} {s[2]:+.3f}")

    lmins = np.arange(2, 41)
    sigs = np.zeros(len(lmins))
    curve = {name: np.zeros(len(lmins)) for name in deltas}
    for a, lm in enumerate(lmins):
        sig, sh = fisher(C, dF, dA, N, deltas, ells, lmin_auto=lm)
        sigs[a] = sig
        for name in deltas:
            curve[name][a] = sh[name][0] / sig
    print("against l_min (auto + cross): l_min, sigma, templates, difference, templates + residual")
    for a, lm in enumerate(lmins):
        if lm in (2, 4, 6, 8, 10, 12, 14, 16, 20, 25, 30, 40):
            print(f"  {lm:3d} {sigs[a]:6.1f} {curve['templates'][a]:+6.2f} {curve['difference'][a]:+6.2f} "
                  f"{curve['templates + residual'][a]:+6.2f}")

    np.savez("out/arf_fnl.npz", ells=ells, lmins=lmins, sigma=sigs, shot=np.array(N),
             curve_templates=curve["templates"], curve_difference=curve["difference"],
             curve_residual=curve["residual 6-13"], curve_templates_p_residual=curve["templates + residual"],
             C_00=C[(0, 0)], C_11=C[(1, 1)], dF_00=dF[(0, 0)], dF_11=dF[(1, 1)],
             raw_0=raw[0], raw_1=raw[1], templates_0=tmpl[0], templates_1=tmpl[1],
             corrected_0=corr[0], corrected_1=corr[1])


if __name__ == "__main__":
    main()
