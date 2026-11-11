#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

import healpy as hp
import numpy as np

from mocks import cells
from quaia import TEMPLATES, desi_match, load, pixel_galactic, template_maps, unit
from theory import Cosmology, Spectra

ZSPLIT = 1.47
LMAX = (236, 390)
FSKY = (0.61, 0.52)
LMIN_AUTO, LMIN_CROSS = 15, 2
NSIDE = 64
ZGRID = np.linspace(0.01, 5.0, 400)


def cell_index(T_raw, names, nbins=8):
    idx = np.zeros(len(T_raw), int)
    for name in names:
        x = T_raw[:, TEMPLATES.index(name)]
        idx = idx * nbins + np.searchsorted(np.quantile(x, np.linspace(0, 1, nbins + 1)[1:-1]), x)
    return np.unique(idx, return_inverse=True)[1]


def spurious_cls(d, rng):
    S = hp.read_map(f"data/selection_function_NSIDE{NSIDE}_G20.5.fits")
    pix = hp.ang2pix(NSIDE, d["ra"], d["dec"], lonlat=True)
    cnt = np.bincount(pix, minlength=hp.nside2npix(NSIDE)).astype(float)
    good = (S > 0.5) & (cnt > 0)
    T = template_maps()
    T = (T - T[good].mean(0)) / T[good].std(0)
    pl, pb = pixel_galactic()
    X = np.column_stack([np.ones(good.sum()), T[good], unit(pl, pb)[good]])
    sel_cells = cell_index(template_maps(), ("dust", "stars", "unwise", "m10"))
    cell = cells(d["T_raw"])
    hi = (d["z"] >= ZSPLIT).astype(float)
    half = rng.random(len(hi)) < 0.5
    maps = []
    for h in (half, ~half):
        f = np.bincount(cell[h], weights=hi[h], minlength=cell.max() + 1) / np.maximum(
            np.bincount(cell[h], minlength=cell.max() + 1), 1)
        fp = np.bincount(pix, weights=f[cell], minlength=len(cnt)) / np.maximum(cnt, 1)
        maps.append(fp)
    out = {}
    for b, name in enumerate(("low-z", "high-z")):
        dA, dB = [((fp if b else 1 - fp)[good] / (fp if b else 1 - fp)[good].mean() - 1) for fp in maps]
        print(f"  bin migration, {name}: rms of the spurious count map before cleaning "
              f"{np.sqrt(np.mean(dA * dB)):.3f} (cross of the two halves)")
        res = {"linear": [], "selection+linear": []}
        for fp in maps:
            frac = fp if b else 1 - fp
            delta = frac[good] / frac[good].mean() - 1
            lin = delta - X @ np.linalg.lstsq(X, delta, rcond=None)[0]
            sc = sel_cells[good]
            nl = delta - (np.bincount(sc, weights=delta) / np.maximum(np.bincount(sc), 1))[sc]
            nl = nl - X @ np.linalg.lstsq(X, nl, rcond=None)[0]
            for key, v in (("linear", lin), ("selection+linear", nl)):
                m = np.zeros(len(cnt))
                m[good] = v
                res[key].append(hp.map2alm(m, lmax=3 * NSIDE - 1))
        out[name] = {k: hp.alm2cl(a[0], a[1]) / good.mean() for k, a in res.items()}
    return out


def stacked_nz(z, ez):
    n = np.zeros(len(ZGRID))
    for i in range(0, len(z), 20000):
        zi, ei = z[i:i + 20000], ez[i:i + 20000]
        n += (np.exp(-0.5 * ((ZGRID[:, None] - zi[None, :]) / ei[None, :]) ** 2) / ei[None, :]).sum(axis=1)
    return n / np.trapezoid(n, ZGRID)


def desi_nz(d, D, sb, fill_unmatched=False):
    m = np.isfinite(D["z"])
    zq_edges = np.arange(0.0, 5.01, 0.05)
    g_edges = np.array([0, 18.5, 19.0, 19.5, 20.0, 20.5])
    zt_edges = np.append(ZGRID - 0.5 * (ZGRID[1] - ZGRID[0]), ZGRID[-1] + 0.5 * (ZGRID[1] - ZGRID[0]))
    iq = np.clip(np.digitize(d["z"], zq_edges) - 1, 0, len(zq_edges) - 2)
    ig = np.clip(np.digitize(d["G"], g_edges) - 1, 0, len(g_edges) - 2)
    cond = np.zeros((len(zq_edges) - 1, len(g_edges) - 1, len(ZGRID)))
    for a in range(cond.shape[0]):
        for g in range(cond.shape[1]):
            k = m & (iq == a) & (ig == g)
            if k.sum() >= 20:
                cond[a, g] = np.histogram(D["z"][k], zt_edges)[0]
            else:
                cond[a, g] = np.histogram(np.full(1, zq_edges[a] + 0.025), zt_edges)[0]
            cond[a, g] /= cond[a, g].sum()
    if fill_unmatched:
        inside = D["interior"]
        rate = np.array([m[inside & (iq == a)].mean() if (inside & (iq == a)).sum() > 100 else np.nan
                         for a in range(cond.shape[0])])
        plateau = np.nanmedian(rate[(zq_edges[:-1] >= 1.0) & (zq_edges[:-1] < 3.0)])
        missing = np.clip(1 - np.nan_to_num(rate, nan=plateau) / plateau, 0, 1)
        own = np.zeros((cond.shape[0], len(ZGRID)))
        for a in range(cond.shape[0]):
            own[a] = np.histogram(np.full(1, zq_edges[a] + 0.025), zt_edges)[0]
        cond = (1 - missing)[:, None, None] * cond + missing[:, None, None] * own[:, None, :]
    counts = np.zeros(cond.shape[:2])
    np.add.at(counts, (iq[sb], ig[sb]), 1)
    n = np.einsum("ag,agz->z", counts, cond) / (ZGRID[1] - ZGRID[0])
    return n / np.trapezoid(n, ZGRID)


def spectra_vectors(sp, ells):
    C0 = sp.cls(ells, [1.0, 1.0], 0.0)
    Cp = sp.cls(ells, [1.0, 1.0], 1.0)
    Cm = sp.cls(ells, [1.0, 1.0], -1.0)
    dfnl = {k: (Cp[k] - Cm[k]) / 2 for k in C0}
    return C0, dfnl


def fisher_bias(C0, dfnl, nbar, Nkk, deltas, use=("auto", "cross"), lmin_auto=LMIN_AUTO):
    ells = np.arange(2, max(LMAX) + 1)
    F = np.zeros((3, 3))
    B = {name: np.zeros(3) for name in deltas}
    for a, ell in enumerate(ells):
        keys = []
        for i in range(2):
            if "auto" in use and lmin_auto <= ell <= LMAX[i]:
                keys.append(("g", i, i))
            if "cross" in use and LMIN_CROSS <= ell <= LMAX[i]:
                keys.append(("k", i))
        if not keys:
            continue
        def tot(x, y):
            if x == "k" and y == "k":
                return C0[("k", "k")][a] + Nkk
            if x == "k" or y == "k":
                i = y if x == "k" else x
                return C0[("k", i)][a]
            if x != y:
                return C0[("g", min(x, y), max(x, y))][a]
            return C0[("g", x, x)][a] + 1 / nbar[x]
        fields = lambda key: (key[1], key[2]) if key[0] == "g" else ("k", key[1])
        fs = np.sqrt(FSKY[0] * FSKY[1]) if len({k[1] for k in keys}) > 1 else FSKY[keys[0][1]]
        cov = np.zeros((len(keys), len(keys)))
        for p, kp in enumerate(keys):
            for q, kq in enumerate(keys):
                (w, x), (y, z) = fields(kp), fields(kq)
                cov[p, q] = (tot(w, y) * tot(x, z) + tot(w, z) * tot(x, y)) / ((2 * ell + 1) * fs)
        D = np.zeros((3, len(keys)))
        for p, k in enumerate(keys):
            D[0, p] = dfnl[k][a]
            i = k[1]
            D[1 + i, p] = (2 if k[0] == "g" else 1) * C0[k][a]
        ci = np.linalg.inv(cov)
        F += D @ ci @ D.T
        for name, dC in deltas.items():
            B[name] += D @ ci @ np.array([dC.get(k, np.zeros(len(ells)))[a] for k in keys])
    Finv = np.linalg.inv(F)
    return np.sqrt(Finv[0, 0]), {name: Finv @ b for name, b in B.items()}


def main(seed=11):
    rng = np.random.default_rng(seed)
    d = load()
    S = hp.read_map(f"data/selection_function_NSIDE{NSIDE}_G20.5.fits")
    pix = hp.ang2pix(NSIDE, d["ra"], d["dec"], lonlat=True)
    inmask = S[pix] > 0.5
    bins = (inmask & (d["z"] < ZSPLIT), inmask & (d["z"] >= ZSPLIT))
    area = (S > 0.5).sum() * hp.nside2pixarea(NSIDE)
    nbar = [b.sum() / area for b in bins]
    print(f"4.4  f_NL bias, setup of Fabbian et al. N per bin {bins[0].sum()}, {bins[1].sum()}; "
          f"nbar {nbar[0]:.3g}, {nbar[1]:.3g} per sr")

    D = desi_match(d)
    nz = {"stacked": [stacked_nz(d["z"][b], d["zerr"][b]) for b in bins],
          "DESI": [desi_nz(d, D, b) for b in bins],
          "DESI, unmatched correct": [desi_nz(d, D, b, fill_unmatched=True) for b in bins]}
    for name, ns in nz.items():
        desc = []
        for n in ns:
            mu = np.trapezoid(ZGRID * n, ZGRID)
            desc.append(f"mean {mu:.3f} sd {np.sqrt(np.trapezoid((ZGRID - mu) ** 2 * n, ZGRID)):.3f}")
        print(f"  n(z) {name:<26} low-z {desc[0]};  high-z {desc[1]}")

    cosmo = Cosmology()
    ells = np.arange(2, max(LMAX) + 1)
    C = {name: spectra_vectors(Spectra(cosmo, ZGRID, ns), ells) for name, ns in nz.items()}
    C0, dfnl = C["stacked"]
    save_theory(C0, dfnl, nbar, ells)

    sp = spurious_cls(d, rng)
    print("  bin migration: spurious count C_l per bin, mean over l = 15-29 / 30-59 (x 1e-7), and against the "
          "fiducial theory C_l (no shot noise)")
    deltas = {}
    for b, name in enumerate(("low-z", "high-z")):
        for key, cl in sp[name].items():
            v = np.interp(ells, np.arange(len(cl)), cl)
            v[ells > 3 * NSIDE - 1] = 0.0
            g = C0[("g", b, b)]
            print(f"    {name:<7} {key:<17} {v[13:28].mean() * 1e7:6.2f} / {v[28:58].mean() * 1e7:6.2f};  "
                  f"fraction of theory C_l at l = 15-29: {np.mean(v[13:28] / g[13:28]):.3f}")
            deltas.setdefault(f"migration, {key}", {})[("g", b, b)] = v
    for name in ("DESI", "DESI, unmatched correct"):
        deltas[f"n(z) {name}"] = {k: C[name][0][k] - C0[k] for k in C0}

    print("  Fisher: sigma(f_NL) and bias Delta f_NL (in units of sigma); A_i shifts")
    for use in (("auto", "cross"), ("cross",), ("auto",)):
        for Nkk in (1.3e-7, 0.65e-7, 2.6e-7):
            if use != ("auto", "cross") and Nkk != 1.3e-7:
                continue
            sig, shifts = fisher_bias(C0, dfnl, nbar, Nkk, deltas, use)
            print(f"    {'+'.join(use):<10} N_kk {Nkk:.2g}: sigma(f_NL) {sig:5.1f}")
            for name, s in shifts.items():
                print(f"      {name:<34} Delta f_NL {s[0]:+6.1f} ({s[0] / sig:+.2f} sigma);  "
                      f"Delta A {s[1]:+.3f}, {s[2]:+.3f}")
    print("  auto spectra down to l_min = 2, N_kk 1.3e-7")
    for use in (("auto", "cross"), ("auto",)):
        sig, shifts = fisher_bias(C0, dfnl, nbar, 1.3e-7, deltas, use, lmin_auto=2)
        print(f"    {'+'.join(use):<10} sigma(f_NL) {sig:5.1f}")
        for name, s in shifts.items():
            print(f"      {name:<34} Delta f_NL {s[0]:+6.1f} ({s[0] / sig:+.2f} sigma)")
    np.savez("out/fnl.npz", ells=ells, zgrid=ZGRID, **{f"nz_{k}": np.array(v) for k, v in nz.items()},
             **{f"spurious_{b}_{k}": v for b, dd in sp.items() for k, v in dd.items()})


def save_theory(C0, dfnl, nbar, ells):
    keys = list(C0)
    np.savez("out/fnl_theory.npz", ells=ells, nbar=np.array(nbar), keys=np.array([str(k) for k in keys]),
             C0=np.array([C0[k] for k in keys]), dfnl=np.array([dfnl[k] for k in keys]))


def load_theory():
    t = np.load("out/fnl_theory.npz")
    keys = [eval(k) for k in t["keys"]]
    return dict(zip(keys, t["C0"])), dict(zip(keys, t["dfnl"])), list(t["nbar"]), t["ells"]


def migration_models(d, half):
    T_pix = template_maps()
    pix = hp.ang2pix(NSIDE, d["ra"], d["dec"], lonlat=True)
    hi = (d["z"] >= ZSPLIT).astype(float)
    out = {}
    for name, names, nb in (("cells8", ("dust", "unwise", "unwisescan", "mcs"), 8),
                            ("cells6+stars", ("dust", "unwise", "unwisescan", "stars", "mcs"), 6)):
        cp = cell_index(T_pix, names, nb)
        cq = cp[pix]
        f = np.bincount(cq[half], weights=hi[half], minlength=cp.max() + 1) / np.maximum(
            np.bincount(cq[half], minlength=cp.max() + 1), 1)
        out[name] = f[cp]
    Ts = (T_pix - T_pix.mean(0)) / T_pix.std(0)
    cols = [Ts[:, i] for i in range(6)] + [Ts[:, i] * Ts[:, j] for i in range(6) for j in range(i, 6)]
    Xp = np.column_stack([np.ones(len(Ts))] + cols)
    coef = np.linalg.lstsq(Xp[pix[half]], hi[half], rcond=None)[0]
    out["smooth"] = Xp @ coef
    return out


def migration_deltas(d, seed=12, verbose=True):
    rng = np.random.default_rng(seed)
    C0, dfnl, nbar, ells = load_theory()
    S = hp.read_map(f"data/selection_function_NSIDE{NSIDE}_G20.5.fits")
    pix = hp.ang2pix(NSIDE, d["ra"], d["dec"], lonlat=True)
    cnt = np.bincount(pix, minlength=hp.nside2npix(NSIDE)).astype(float)
    good = (S > 0.5) & (cnt > 0)
    T = template_maps()
    T = (T - T[good].mean(0)) / T[good].std(0)
    pl, pb = pixel_galactic()
    X = np.column_stack([np.ones(good.sum()), T[good], unit(pl, pb)[good]])
    deproj = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    lmax = 3 * NSIDE - 1

    def cl(m1, m2=None):
        a = np.zeros(len(cnt)); a[good] = m1
        if m2 is None:
            return hp.anafast(a, lmax=lmax) / good.mean()
        b = np.zeros(len(cnt)); b[good] = m2
        return hp.anafast(a, b, lmax=lmax) / good.mean()

    def as_delta(fp, b):
        frac = fp[good] if b else 1 - fp[good]
        return frac / frac.mean() - 1

    half = rng.random(len(d["z"])) < 0.5
    MA, MB = migration_models(d, half), migration_models(d, ~half)
    names = list(MA)
    if verbose:
        print("4.4  Correction test: signal from one model (half A), correction from another (half B)")
        print("  residual spurious C_l after linear deprojection, l = 2-14 / 15-29 (x 1e-7); noise floor removed")
    deltas = {}
    for b, bname in enumerate(("low-z", "high-z")):
        noise = {m: cl(deproj(as_delta(MA[m], b) - as_delta(MB[m], b))) for m in names}
        for t in names:
            truth = cl(deproj(as_delta(MA[t], b)), deproj(as_delta(MB[t], b)))
            rows = {"no correction": truth}
            for c in names:
                r = cl(deproj(as_delta(MA[t], b) - as_delta(MB[c], b))) - 0.5 * (noise[t] + noise[c])
                rows[f"corrected with {c}"] = r
            for lab, v in rows.items():
                vv = np.interp(ells, np.arange(len(v)), v); vv[ells > lmax] = 0.0
                deltas.setdefault((t, lab), {})[("g", b, b)] = vv
                if verbose:
                    print(f"    {bname:<7} truth {t:<13} {lab:<27} {v[2:15].mean() * 1e7:7.2f} / {v[15:30].mean() * 1e7:6.2f}")
    return dict(deltas=deltas, names=names, C0=C0, dfnl=dfnl, nbar=nbar, ells=ells, S=S, pix=pix, cnt=cnt,
                good=good, cl=cl, deproj=deproj)


def bias_curve(lmins=range(2, 16), seed=12):
    d = load()
    m = migration_deltas(d, seed, verbose=False)
    pairs = {"no correction": [("cells8", "no correction"), ("cells6+stars", "no correction"), ("smooth", "no correction")],
             "corrected": [("cells8", "corrected with cells6+stars"), ("cells6+stars", "corrected with cells8"),
                           ("smooth", "corrected with cells8")]}
    out = {k: np.zeros((len(pairs[k]), len(lmins))) for k in pairs}
    sig = np.zeros(len(lmins))
    for a, lmin in enumerate(lmins):
        s, shifts = fisher_bias(m["C0"], m["dfnl"], m["nbar"], 1.3e-7, m["deltas"], ("auto", "cross"), lmin_auto=lmin)
        sig[a] = s
        for k, lst in pairs.items():
            for i, key in enumerate(lst):
                out[k][i, a] = shifts[key][0] / s
    np.savez("out/fnl_curve.npz", lmin=np.array(list(lmins)), sigma=sig, **{k.replace(" ", "_"): v for k, v in out.items()})
    for a, lmin in enumerate(lmins):
        print(f"  l_min {lmin:2d}: sigma {sig[a]:5.1f}; no correction {out['no correction'][:, a].min():+.2f} to "
              f"{out['no correction'][:, a].max():+.2f}; corrected {out['corrected'][:, a].min():+.2f} to {out['corrected'][:, a].max():+.2f}")


def correction_test(seed=12):
    d = load()
    m = migration_deltas(d, seed)
    deltas, names, C0, dfnl, nbar, ells = (m[k] for k in ("deltas", "names", "C0", "dfnl", "nbar", "ells"))
    S, pix, cnt, good, cl, deproj = (m[k] for k in ("S", "pix", "cnt", "good", "cl", "deproj"))
    print("  Delta f_NL / sigma with auto-spectra from l_min = 2 (and 15), N_kk 1.3e-7")
    for use in (("auto", "cross"), ("auto",)):
        for lmin in (2, 15):
            sig, shifts = fisher_bias(C0, dfnl, nbar, 1.3e-7, deltas, use, lmin_auto=lmin)
            print(f"    {'+'.join(use):<10} l_min {lmin:2d}: sigma(f_NL) {sig:5.1f}")
            for t in names:
                line = "  ".join(f"{lab.replace('corrected with ', 'corr ')}: {shifts[(t, lab)][0] / sig:+.2f}"
                                 for (tt, lab) in deltas if tt == t)
                print(f"      truth {t:<13} {line}")

    print("  Real data: per-bin count power / theory, after linear deprojection; theory amplitude fitted at"
          " l = 30-100; shot noise removed")
    M = migration_models(d, np.ones(len(d["z"]), bool))
    lb = [(2, 6), (6, 10), (10, 15), (15, 30), (30, 60)]
    for b, bname in enumerate(("low-z", "high-z")):
        sb = (d["z"] >= ZSPLIT) if b else (d["z"] < ZSPLIT)
        n = np.bincount(pix[sb], minlength=len(cnt)).astype(float)
        th = C0[("g", b, b)]
        thl = lambda a, c: th[(ells >= a) & (ells < c)].mean()
        for lab, corr in [("raw", np.ones(len(cnt)))] + [(f"corrected {m}", (M[m] if b else 1 - M[m])) for m in names]:
            w = S[good] * corr[good] / corr[good].mean()
            nb_ = n[good].sum() / w.sum()
            delta = n[good] / (nb_ * w) - 1
            c = cl(deproj(delta))
            shot = hp.nside2pixarea(NSIDE) * np.mean(1 / (nb_ * w))
            c = c - shot
            A2 = c[30:101].sum() / th[(ells >= 30) & (ells <= 100)].sum()
            ratios = []
            for a, e in lb:
                modes = ((2 * np.arange(a, e) + 1) * good.mean()).sum()
                err = np.sqrt(2 / modes) * (A2 * thl(a, e) + shot) / (A2 * thl(a, e))
                ratios.append(f"{c[a:e].mean() / (A2 * thl(a, e)):6.2f}+-{err:4.2f}")
            print(f"    {bname:<7} {lab:<22} A^2 {A2:5.2f}   " + "  ".join(f"l{a}-{e - 1}: {r}" for (a, e), r in zip(lb, ratios)))


def heldout(n_rep=3, seed=21):
    d = load()
    pix = hp.ang2pix(NSIDE, d["ra"], d["dec"], lonlat=True)
    good = hp.read_map(f"data/selection_function_NSIDE{NSIDE}_G20.5.fits")[pix] > 0.5
    hi = (d["z"] >= ZSPLIT).astype(float)
    rng = np.random.default_rng(seed)
    gain = {}
    for _ in range(n_rep):
        half = rng.random(len(hi)) < 0.5
        for A in (half, ~half):
            B = ~A & good
            f0 = hi[A & good].mean()
            ll0 = -np.mean(hi[B] * np.log(f0) + (1 - hi[B]) * np.log(1 - f0))
            for k, v in migration_models(d, A).items():
                f = np.clip(v[pix[B]], 1e-3, 1 - 1e-3)
                ll = -np.mean(hi[B] * np.log(f) + (1 - hi[B]) * np.log(1 - f))
                gain.setdefault(k, []).append((ll0 - ll) * B.sum())
    print("4.4  Held out log likelihood gain over a constant fraction")
    for k, v in gain.items():
        print(f"    {k:<14} {np.mean(v):7.1f} +- {np.std(v):5.1f}")


if __name__ == "__main__":
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else ""
    if task == "correction":
        correction_test()
    elif task == "curve":
        bias_curve()
    elif task == "heldout":
        heldout()
    else:
        main()
