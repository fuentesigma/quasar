#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

import sys

import healpy as hp
import numpy as np
from astropy.table import Table

from quaia import C_KMS, CMB, CMB_V, TEMPLATES, describe, desi_match, dipole, lb, load, ols

BCUT, SMIN = 30.0, 0.5


def footprint(d, bcut=BCUT, smin=SMIN):
    return (np.abs(d["b"]) > bcut) & (d["S"] > smin)


def patch_bootstrap(X, y, patch, nb, rng):
    _, inv = np.unique(patch, return_inverse=True)
    P = inv.max() + 1
    XtX = np.zeros((P, X.shape[1], X.shape[1]))
    Xty = np.zeros((P, X.shape[1]))
    for p in range(P):
        k = inv == p
        XtX[p], Xty[p] = X[k].T @ X[k], X[k].T @ y[k]
    beta = np.linalg.solve(XtX.sum(0), Xty.sum(0))
    draws = []
    for _ in range(nb):
        w = np.bincount(rng.integers(0, P, P), minlength=P)
        draws.append(np.linalg.solve(np.tensordot(w, XtX, 1), np.tensordot(w, Xty, 1)))
    return beta, np.std(draws, axis=0), XtX, Xty, P


def boot_slope(x, y, patch, nb, rng):
    b, e, *_ = patch_bootstrap(np.column_stack([np.ones(len(y)), x]), y, patch, nb, rng)
    return b[1], e[1]


def perm_dipole(d, k, n_perm=200, seed=2, T=None):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(k.sum()), d["n"][k]] + ([] if T is None else [T[k]]))
    A = np.linalg.solve(X.T @ X, X.T)[1:4]
    y = d["y"][k]
    beta = -A @ y
    draws = np.array([-A @ rng.permutation(y) for _ in range(n_perm)])
    return beta, np.cov(draws.T) + np.eye(3) * clustering_var()


def clustering_var():
    o = np.load("out/arf_theory.npz")
    return 3 * float(o["C_0_0"][0]) / (4 * np.pi)


def s3_errors(d, n_perm=200, seed=3):
    rng = np.random.default_rng(seed)
    k = footprint(d)
    n, y, T = d["n"][k], d["y"][k], d["T"][k]
    _, cov, _ = dipole(n, y, T)
    perm = np.array([dipole(n, rng.permutation(y), T)[0] for _ in range(n_perm)])
    print(f"3    dipole error per component, |b|>30, S>0.5, with templates: least squares "
          f"{np.sqrt(np.trace(cov) / 3) * C_KMS:.0f} km/s, {n_perm} permutations {perm.std(axis=0).mean() * C_KMS:.0f} km/s")


def s41_coefficients(d, nb=500, seed=1):
    print("4.1  ln(1+z) = a - beta.n + sum_k c_k T_k, per standard deviation of each map")
    rng = np.random.default_rng(seed)
    for name, k in (("whole sky", np.ones(len(d["y"]), bool)), ("|b|>30, S>0.5", footprint(d))):
        X = np.column_stack([np.ones(k.sum()), d["T"][k], d["n"][k]])
        patch = hp.ang2pix(8, d["ra"][k], d["dec"][k], lonlat=True)
        beta, boot, XtX, _, P = patch_bootstrap(X, d["y"][k], patch, nb, rng)
        res = d["y"][k] - X @ beta
        se = np.sqrt(np.diag(np.linalg.inv(XtX.sum(0))) * (res @ res) / (k.sum() - X.shape[1]))
        print(f"  {name}: N = {k.sum()}, {P} patches; errors from least squares and from the patch bootstrap")
        for j, tn in enumerate(TEMPLATES):
            c = beta[1 + j]
            print(f"    {tn:<11} {c:+.4f}  {se[1 + j]:.4f} ({abs(c) / se[1 + j]:4.1f} sd)  "
                  f"{boot[1 + j]:.4f} ({abs(c) / boot[1 + j]:4.1f} sd)")
        C = np.corrcoef(d["T"][k].T)
        print("    correlation of the maps:\n" + "\n".join("      " + " ".join(f"{v:+.2f}" for v in r) for r in C))
    k = footprint(d)
    dust = d["T_raw"][k, 0]
    lo_cut, hi_cut = np.quantile(dust, [0.1, 0.9])
    diff = d["y"][k][dust > hi_cut].mean() - d["y"][k][dust < lo_cut].mean()
    print(f"  mean ln(1+z), most minus least dusty tenth (|b|>30, S>0.5): {diff:+.4f}; "
          f"our motion between opposite directions: {2 * 370 / C_KMS:.4f}")


def s41_desi(d, D, nb=500, seed=9):
    rng = np.random.default_rng(seed)
    zd = D["z"]
    m = np.isfinite(zd)
    delta = d["y"][m] - np.log1p(zd[m])
    dz = np.abs(d["z"][m] - zd[m]) / (1 + zd[m])
    bad = np.abs(delta) > 0.1
    print(f"4.1  DESI matches within 1 arcsec: {m.sum()}; catastrophic: |Delta ln(1+z)| > 0.1 {bad.mean():.3f}, "
          f"|Delta z|/(1+z) > 0.2 {np.mean(dz > 0.2):.3f}")
    names = TEMPLATES[:-1]
    patch = hp.ang2pix(8, d["ra"][m], d["dec"][m], lonlat=True)
    T_cat = d["T"][m][:, :len(names)]
    T_M = d["T_raw"][m][:, :len(names)]
    T_M = (T_M - T_M.mean(0)) / T_M.std(0)
    print("  standard deviation of each map, whole catalogue / DESI matches: "
          + ", ".join(f"{tn} {r:.2f}" for tn, r in zip(names, d["T_raw"][:, :len(names)].std(0) / d["T_raw"][m][:, :len(names)].std(0))))
    for lab, T in (("maps standardised over the catalogue", T_cat), ("maps standardised over the matches", T_M)):
        for name, s in (("all matches", np.ones(m.sum(), bool)), ("without catastrophic", ~bad)):
            X = np.column_stack([np.ones(s.sum()), T[s]])
            c, e, *_ = patch_bootstrap(X, delta[s], patch[s], nb, rng)
            print(f"  Quaia - DESI against the maps, {name}, {lab}, patch bootstrap:")
            print("    " + ", ".join(f"{tn} {a:+.4f}+-{b:.4f}" for tn, a, b in zip(names, c[1:], e[1:])))
    dust = d["T_raw"][m, 0]
    print("  slope on A_V, same quasars, patch bootstrap:")
    for name, yy in (("Quaia ln(1+z)", d["y"][m]), ("DESI ln(1+z)", np.log1p(zd[m]))):
        b, e = boot_slope(dust, yy, patch, nb, rng)
        print(f"    {name:<14} {b:+.3f} +- {e:.3f}")


def slope_pair(x, yD, yQ, w, patch, n_boot, rng):
    _, inv = np.unique(patch, return_inverse=True)
    S = np.stack([np.bincount(inv, weights=v) for v in (w, w * x, w * x * x, w * yD, w * x * yD, w * yQ, w * x * yQ)])

    def slopes(T):
        n, sx, sxx, sD, sxD, sQ, sxQ = T
        den = n * sxx - sx * sx
        return np.array([(n * sxD - sx * sD) / den, (n * sxQ - sx * sQ) / den])

    b = slopes(S.sum(axis=1))
    draws = np.array([slopes(S[:, rng.integers(0, S.shape[1], S.shape[1])].sum(axis=1)) for _ in range(n_boot)])
    return b, draws.std(axis=0), np.percentile(draws[:, 0] / draws[:, 1], [16, 84])


def s41_completeness(d, D, n_boot=500, seed=5):
    m = np.isfinite(D["z"])
    av, yQ, yD = d["T_raw"][:, 0], d["y"], np.log1p(D["z"])
    print("4.1  DESI completeness: match rate and Quaia slope on A_V inside the DESI footprint, patch bootstrap")
    rng_rate = np.random.default_rng(seed + 100)
    patch = hp.ang2pix(8, d["ra"], d["dec"], lonlat=True)
    for name, foot in (("interior", D["interior"]), ("interior, NTILE >= 3", D["interior"] & (D["ntile_pix"] >= 3))):
        r = boot_slope(av[foot], m[foot].astype(float), patch[foot], n_boot, rng_rate)
        a = boot_slope(av[foot], yQ[foot], patch[foot], n_boot, rng_rate)
        s = boot_slope(av[foot & m], yQ[foot & m], patch[foot & m], n_boot, rng_rate)
        print(f"    {name:<21} rate {m[foot].mean():.3f}, slope {r[0]:+.3f} +- {r[1]:.3f};  "
              f"Quaia slope all {a[0]:+.3f} +- {a[1]:.3f}, matched {s[0]:+.3f} +- {s[1]:.3f}")
    print("  f_sel = s_D / s_Q, patch bootstrap")
    rng = np.random.default_rng(seed)
    ones = np.ones(len(av))
    cases = (("all matches", m, ones), ("WEIGHT", m, D["weight"]),
             ("WEIGHT_COMP x WEIGHT_ZFAIL", m, D["weight_comp"] * D["weight_zfail"]),
             ("NTILE >= 3", m & (D["ntile"] >= 3), ones), ("NTILE >= 4", m & (D["ntile"] >= 4), ones),
             ("interior, NTILE >= 3", m & D["interior"] & (D["ntile_pix"] >= 3), ones),
             ("south imaging", m & ~D["north"], ones), ("north imaging", m & D["north"], ones))
    for name, s, w in cases:
        b, e, (lo, hi) = slope_pair(av[s], yD[s], yQ[s], w[s], patch[s], n_boot, rng)
        print(f"    {name:<27} N {s.sum():6d}  s_D {b[0]:+.3f} +- {e[0]:.3f}  s_Q {b[1]:+.3f} +- {e[1]:.3f}"
              f"  f_sel {b[0] / b[1]:.2f} [{lo:.2f}, {hi:.2f}]")


def colour_cuts(gw, ww, bg):
    return (gw > 2.15) & (ww > 0.4) & (bg > -0.3) & (gw + 1.2 * ww > 3.4)


def s41_selection(d, D, seed=3, nb=500):
    t = Table.read("data/quaia_G20.5.fits")
    BP, G, W1, W2 = (np.asarray(t[c], float) for c in ("phot_bp_mean_mag", "phot_g_mean_mag", "mag_w1_vg", "mag_w2_vg"))
    gw, ww, bg = G - W1, W1 - W2, BP - G
    av = d["T_raw"][:, 0]
    m = np.isfinite(D["z"])
    yD, yQ = np.log1p(np.where(m, D["z"], 1.0)), d["y"]
    patch = hp.ang2pix(8, d["ra"], d["dec"], lonlat=True)
    rng = np.random.default_rng(seed)

    def slope(y, sel, w=None):
        w = np.ones(len(av)) if w is None else w
        b, e, _ = slope_pair(av[sel], y[sel], y[sel], w[sel], patch[sel], nb, rng)
        return b[0], e[0]

    a_min, a_max = 0.02, 0.25
    base = m & (av >= a_min) & (av <= a_max)
    print(f"4.1  Selection tests on the DESI matches with {a_min} <= A_V <= {a_max} (N = {base.sum()})")
    for name, y in (("s_D", yD), ("s_Q", yQ)):
        b, e = slope(y, base)
        print(f"    {name} as observed {b:+.4f} +- {e:.4f}")
    for kG, kBP, kW1, kW2 in ((0.85, 1.07, 0.039, 0.026), (0.80, 0.95, 0.06, 0.04), (0.90, 1.15, 0.03, 0.02)):
        lo, hi = av - a_min, a_max - av
        ok_col = (colour_cuts(gw - (kG - kW1) * lo, ww - (kW1 - kW2) * lo, bg - (kBP - kG) * lo)
                  & colour_cuts(gw + (kG - kW1) * hi, ww + (kW1 - kW2) * hi, bg + (kBP - kG) * hi))
        ok_flux = G + kG * hi < 20.5
        print(f"  A_X/A_V (G, BP, W1, W2) = ({kG}, {kBP}, {kW1}, {kW2}): uniform flux limit removes "
              f"{1 - (base & ok_flux).sum() / base.sum():.3f}, uniform colour cuts {1 - (base & ok_col).sum() / base.sum():.3f}")
        for name, s in (("flux uniform", base & ok_flux), ("colours uniform", base & ok_col), ("both", base & ok_col & ok_flux)):
            (bD, eD), (bQ, eQ) = slope(yD, s), slope(yQ, s)
            print(f"    {name:<16} N {s.sum():6d}  s_D {bD:+.4f} +- {eD:.4f}  s_Q {bQ:+.4f} +- {eQ:.4f}")

    inside = D["interior"]
    zq = np.clip(np.digitize(d["z"], np.arange(0, 4.6, 0.25)), 0, 20)
    gb = np.digitize(G, [18.5, 19, 19.5, 20])
    ab = np.digitize(av, np.quantile(av[inside], [.2, .4, .6, .8]))
    cell = (zq * 5 + gb) * 5 + ab
    rate = np.bincount(cell[inside & m], minlength=cell.max() + 1) / np.maximum(np.bincount(cell[inside], minlength=cell.max() + 1), 1)
    w = np.where(rate[cell] > 0.05, 1 / np.maximum(rate[cell], 1e-3), 0.0)
    sel = inside & m
    for name, weights in (("unweighted", None), ("1 / match rate", w)):
        (bD, eD), (bQ, eQ) = slope(yD, sel, weights), slope(yQ, sel, weights)
        print(f"  DESI interior, {name:<15} s_D {bD:+.4f} +- {eD:.4f}  s_Q {bQ:+.4f} +- {eQ:.4f}  f_sel {bD / bQ:.2f}")


def s41_infrared(d, D, seed=6, nb=500, seed_share=8, nb_share=500):
    t = Table.read("data/quaia_G20.5.fits")
    W2 = np.asarray(t["mag_w2_vg"], float)
    m = np.isfinite(D["z"])
    zD = D["z"]
    yD, yQ = np.log1p(np.where(m, zD, 1.0)), d["y"]
    patch = hp.ang2pix(8, d["ra"], d["dec"], lonlat=True)
    use = [0, 1, 2, 3, 4]

    print("4.1  Infrared depth: fraction of matched quasars with W2 > 15.5 against the true redshift")
    for lo, hi in ((0.8, 1.2), (1.2, 1.6), (1.6, 2.0), (2.0, 2.5), (2.5, 3.5)):
        k = m & (zD >= lo) & (zD < hi)
        print(f"    {lo}-{hi}: {np.mean(W2[k] > 15.5):.3f}")
    p99 = np.full(hp.nside2npix(8), np.nan)
    for p in np.unique(patch):
        k = patch == p
        if k.sum() > 300:
            p99[p] = np.percentile(W2[k], 99)
    ok = np.isfinite(p99[patch])
    print(f"  correlation of the local W2 depth with the unWISE density: "
          f"{np.corrcoef(p99[patch][ok], d['T_raw'][ok, TEMPLATES.index('unwise')])[0, 1]:+.2f}")

    rng = np.random.default_rng(seed)
    print("  joint fit on five maps, 1e-3 per standard deviation (true = DESI, error = Quaia - DESI)")
    print(f"    {'':24}" + "".join(f"{TEMPLATES[i]:>13}" for i in use))
    for name, s in (("all matches", m), ("W2 < 15.5", m & (W2 < 15.5))):
        T = d["T_raw"][s][:, use]
        T = (T - T.mean(0)) / T.std(0)
        X = np.column_stack([np.ones(s.sum()), T])
        for lab, y in (("true", yD), ("Quaia", yQ), ("error", yQ - yD)):
            c, e, *_ = patch_bootstrap(X, y[s], patch[s], nb, rng)
            print(f"    {name:<13} {lab:<10}" + "".join(f"  {1e3 * a:+5.1f}+-{1e3 * b:3.1f}" for a, b in zip(c[1:], e[1:])))

    rng = np.random.default_rng(seed_share)
    av = d["T_raw"][:, 0]
    for name, s in (("all matches", m), ("A_V < 0.3", m & (av < 0.3)), ("north", m & D["north"]), ("south", m & ~D["north"])):
        f, lo, hi = selection_share(d, yQ, yD, s, patch, use, nb_share, rng)
        print(f"  F_sel {name:<12} {f:.2f} [{lo:.2f}, {hi:.2f}]")


def selection_share(d, yQ, yD, s, patch, use, nb, rng):
    T = d["T_raw"][s][:, use]
    T = (T - T.mean(0)) / T.std(0)
    X = np.column_stack([np.ones(s.sum()), T])
    _, inv = np.unique(patch[s], return_inverse=True)
    P = inv.max() + 1
    XX, XQ, XD = np.zeros((P, 6, 6)), np.zeros((P, 6)), np.zeros((P, 6))
    TT, Tm, n = np.zeros((P, 5, 5)), np.zeros((P, 5)), np.zeros(P)
    for p in range(P):
        k = inv == p
        XX[p], XQ[p], XD[p] = X[k].T @ X[k], X[k].T @ yQ[s][k], X[k].T @ yD[s][k]
        TT[p], Tm[p], n[p] = T[k].T @ T[k], T[k].sum(0), k.sum()

    def share(w):
        A = np.tensordot(w, XX, 1)
        bQ = np.linalg.solve(A, np.tensordot(w, XQ, 1))[1:]
        bD = np.linalg.solve(A, np.tensordot(w, XD, 1))[1:]
        N = w @ n
        mu = (w @ Tm) / N
        Sig = np.tensordot(w, TT, 1) / N - np.outer(mu, mu)
        return (bQ @ Sig @ bD) / (bQ @ Sig @ bQ)

    draws = [share(np.bincount(rng.integers(0, P, P), minlength=P).astype(float)) for _ in range(nb)]
    return (share(np.ones(P)), *np.percentile(draws, [16, 84]))


def s41_unmatched(d, D):
    out = (d["z"] < 0.8) | (d["z"] > 3.5)
    inside = D["interior"]
    av = d["T_raw"][:, 0]
    q = np.digitize(av, np.quantile(av[inside], [0.2, 0.4, 0.6, 0.8]))
    print(f"4.1  Quaia redshift outside 0.8-3.5: {out.mean():.3f} of the catalogue; inside the DESI footprint, "
          f"by fifths of A_V: " + ", ".join(f"{out[inside & (q == i)].mean():.3f}" for i in range(5)))


def s41_brightness(d, D, nb=500, seed=10):
    rng = np.random.default_rng(seed)
    zd = D["z"]
    m = np.isfinite(zd)
    delta = d["y"][m] - np.log1p(zd[m])
    dz = np.abs(d["z"][m] - zd[m]) / (1 + zd[m])
    G, dust = d["G"][m], d["T_raw"][m, 0]
    patch = hp.ang2pix(8, d["ra"][m], d["dec"][m], lonlat=True)
    print("4.1  Against G: catastrophic fractions and slope of Quaia - DESI on A_V, patch bootstrap")
    for lo, hi in ((0, 18.5), (18.5, 19.0), (19.0, 19.5), (19.5, 20.0), (20.0, 20.5)):
        s = (G >= lo) & (G < hi)
        b, e = boot_slope(dust[s], delta[s], patch[s], nb, rng)
        print(f"    {lo:4.1f}-{hi:4.1f}: N {s.sum():6d}, catastrophic {np.mean(np.abs(delta[s]) > 0.1):.3f} "
              f"({np.mean(dz[s] > 0.2):.3f}), slope {b:+.3f} +- {e:.3f}")
    print("  uncorrected dipole, whole sample and half with the smaller sigma_z/(1+z), permutations + clustering")
    frac = d["zerr"] / (1 + d["z"])
    for glim in (20.5, 19.5, 19.0, 18.5):
        k = d["G"] < glim
        p = k & (frac < np.median(frac[k]))
        print(f"    G<{glim}: all {describe(*perm_dipole(d, k))}")
        print(f"    {'':6}precise {describe(*perm_dipole(d, p))}")


def s42_arf(d, nside=128, n_shuffle=20, seed=4):
    bins = [(2, 4), (4, 6), (6, 8), (8, 10), (10, 14), (14, 20), (20, 30)]
    rng = np.random.default_rng(seed)
    mask = hp.ud_grade(hp.read_map("data/selection_function_NSIDE64_G20.5.fits"), nside) > 0.5
    pix = hp.ang2pix(nside, d["ra"], d["dec"], lonlat=True)
    keep = mask[pix]
    zmed = np.median(d["z"][keep])

    def cl(p, dz, nbar):
        m = np.bincount(p, weights=dz, minlength=hp.nside2npix(nside)) / nbar
        m[~mask] = 0.0
        c = hp.anafast(m, lmax=3 * nside - 1) / mask.mean()
        return np.array([c[a:b].mean() for a, b in bins])

    print(f"4.2  Mean redshift maps, C_l x 1e6, f_sky {mask.mean():.2f}, halves split at z = {zmed:.2f}")
    print("  " + " " * 24 + "".join(f"{a}-{b - 1}".rjust(8) for a, b in bins))
    for name, sb in (("low z", keep & (d["z"] < zmed)), ("high z", keep & (d["z"] >= zmed))):
        p, z = pix[sb], d["z"][sb]
        nbar = np.bincount(p, minlength=hp.nside2npix(nside))[mask].mean()
        T = (d["T_raw"][sb] - d["T_raw"][sb].mean(axis=0)) / d["T_raw"][sb].std(axis=0)
        coef, _ = ols(T, z)
        zhat = T @ coef
        shuffles = np.array([cl(p, rng.permutation(z) - z.mean(), nbar) for _ in range(n_shuffle)])
        noise = shuffles.mean(axis=0)
        rows = {"data - noise": cl(p, z - z.mean(), nbar) - noise,
                "six maps alone": cl(p, zhat, nbar),
                "corrected - noise": cl(p, z - zhat - (z - zhat).mean(), nbar) - noise,
                "noise scatter": shuffles.std(axis=0)}
        try:
            o = np.load("out/redshift_mocks.npz")
            j = 0 if name == "low z" else 1
            rows["error, isotropic mocks"] = o["arf_none"][:, j, 0, 1:].std(axis=0)
            rows["error corrected, mocks"] = o["arf_none"][:, j, 1, 1:].std(axis=0)
        except (FileNotFoundError, KeyError):
            pass
        print(f"  {name} (mean z {z.mean():.2f})")
        for lab, v in rows.items():
            print(f"    {lab:<22}" + "".join(f"{x * 1e6:8.2f}" for x in v))


def s43_dipole(d):
    print("4.3  Redshift dipole")
    n, y, T = d["n"], d["y"], d["T"]
    every = np.ones(len(y), bool)
    print(f"  uncorrected, whole sky, permutations + clustering: {describe(*perm_dipole(d, every))}")
    print(f"  corrected, whole sky, permutations + clustering:   {describe(*perm_dipole(d, every, T=T))}")
    print(f"  uncorrected, whole sky, least squares:   {describe(*dipole(n, y)[:2])}")
    for name, k in (("|b|>30 S>0.5", footprint(d)), ("|b|>30 S>0.5 G<20", footprint(d) & (d["G"] < 20)),
                    ("|b|>40 S>0.5", footprint(d, 40.0))):
        print(f"  uncorrected, {name}, permutations + clustering: {describe(*perm_dipole(d, k))}")
    print("  uncorrected dipole against the dipole of the map terms alone")
    subsets = {"|b|>30": np.abs(d["b"]) > 30, "|b|<30": np.abs(d["b"]) < 30,
               "|dec|>30": np.abs(d["dec"]) > 30, "|dec|<30": np.abs(d["dec"]) < 30,
               "G<20.0": d["G"] < 20.0, "20.0<G<20.5": d["G"] >= 20.0}
    for name, k in subsets.items():
        _, _, c = dipole(n[k], y[k], T[k])
        raw = dipole(n[k], y[k])[0]
        maps = dipole(n[k], T[k] @ c)[0]
        ang = np.degrees(np.arccos(raw @ maps / np.linalg.norm(raw) / np.linalg.norm(maps)))
        print(f"    {name:<12} uncorrected {describe(raw)}\n    {'':<12} maps alone  {describe(maps)}; {ang:.0f} deg apart")


def s43_residual(d, nside=64, n_shuffle=20, seed=14):
    rng = np.random.default_rng(seed)
    k = footprint(d)
    n, y, T = d["n"][k], d["y"][k], d["T"][k]
    _, _, c = dipole(n, y, T)
    r = y - T @ c
    pix = hp.ang2pix(nside, d["ra"][k], d["dec"][k], lonlat=True)
    cnt = np.bincount(pix, minlength=hp.nside2npix(nside))
    good = cnt > 0
    nbar = cnt[good].mean()

    def cl(v):
        m = np.bincount(pix, weights=v - v.mean(), minlength=len(cnt)) / nbar
        m[~good] = 0.0
        return hp.anafast(m, lmax=10) / good.mean()

    noise = np.mean([cl(rng.permutation(r)) for _ in range(n_shuffle)], axis=0)
    c_l = cl(r) - noise
    v = C_KMS * np.sqrt(3 * np.clip(c_l[2:6], 0, None) / (4 * np.pi))
    print("4.3  Corrected map of ln(1+z), |b|>30 S>0.5: power at l = 2-5 as a velocity per component (km/s): "
          + ", ".join(f"l={l} {x:.0f}" for l, x in zip(range(2, 6), v)))


def s43_limits(seed=0):
    o = np.load("out/redshift_mocks.npz")
    names, vk = list(o["dip_names"]), o["v_kin"]
    rng = np.random.default_rng(seed)
    print("4.3  Corrected dipole against the realistic mocks (km/s)")
    for key in ("|b|>30 S>0.5", "|b|>30 S>0.5 G<20", "|b|>40 S>0.5", "|b|>30 S>0.7"):
        j = names.index(f"{key}: templates")
        mk = o["dip_cells"][:, j]
        cov = np.cov(mk.T) * C_KMS ** 2
        v = (o["real_dip"][j] - (mk.mean(0) - vk)) * C_KMS
        amp = np.linalg.norm(v)
        c, sc = v @ CMB, np.sqrt(CMB @ cov @ CMB)
        noise = rng.multivariate_normal(np.zeros(3), cov, 200000)
        l, b = lb(v)
        print(f"  {key:<20} amplitude {amp:4.0f} toward ({l:3.0f}, {b:+3.0f}), sd per axis {np.sqrt(np.diag(cov)).round(0)}, "
              f"P(noise >= amplitude) {np.mean(np.linalg.norm(noise, axis=1) >= amp):.2f}; "
              f"along the CMB dipole {c:+4.0f} +- {sc:3.0f} ({(CMB_V - c) / sc:.1f} sd below our motion, "
              f"95% upper limit {c + 1.645 * sc:4.0f})")


def main(which):
    d = load()
    if "3" in which:
        s3_errors(d)
        print()
    if "4.1" in which:
        D = desi_match(d)
        s41_coefficients(d)
        print()
        s41_desi(d, D)
        print()
        s41_completeness(d, D)
        print()
        s41_selection(d, D)
        print()
        s41_infrared(d, D)
        print()
        s41_unmatched(d, D)
        print()
        s41_brightness(d, D)
        print()
    if "4.2" in which:
        s42_arf(d)
        print()
    if "4.3" in which:
        s43_dipole(d)
        print()
        s43_residual(d)
        print()
        s43_limits()
        print()


if __name__ == "__main__":
    main(sys.argv[1:] or ["3", "4.1", "4.2", "4.3"])
