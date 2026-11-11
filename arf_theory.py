#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

import numpy as np

from fnl import ZGRID, desi_nz
from quaia import C_KMS, desi_match, load
from theory import Cosmology, b_laurent, weighted_field_cls

ZMED = 1.49


def conditional(zt_pairs, obs_pairs, grid, fallback):
    edges = np.append(grid - 0.5 * (grid[1] - grid[0]), grid[-1] + 0.5 * (grid[1] - grid[0]))
    idx = np.digitize(zt_pairs, edges) - 1
    ok = (idx >= 0) & (idx < len(grid))
    s = np.bincount(idx[ok], weights=obs_pairs[ok], minlength=len(grid))
    n = np.bincount(idx[ok], minlength=len(grid))
    m = np.where(n >= 50, s / np.maximum(n, 1), np.nan)
    inside = (grid > 0.82) & (grid < 3.45) & np.isfinite(m)
    out = fallback(grid).astype(float)
    ker = np.ones(5) / 5
    sm = np.convolve(np.where(inside, m, 0.0), ker, "same") / np.maximum(np.convolve(inside.astype(float), ker, "same"), 1e-9)
    out[inside] = sm[inside]
    return out


def main():
    d = load()
    D = desi_match(d)
    m = np.isfinite(D["z"])
    zt, zo = D["z"][m], d["z"][m]
    n_all = desi_nz(d, D, np.ones(len(d["z"]), bool))
    ybar = d["y"].mean()
    z = ZGRID
    b = b_laurent(z)
    kernels, labels = [], []
    Ey = conditional(zt, np.log1p(zo), z, lambda g: np.log1p(g))
    kernels.append((b * n_all * (Ey - ybar), n_all * (1 + z) * np.gradient(Ey, z)))
    labels.append("y (whole sample)")
    for name, lo, hi in (("low-z", -np.inf, ZMED), ("high-z", ZMED, np.inf)):
        inb = (d["z"] >= lo) & (d["z"] < hi)
        zbar = d["z"][inb].mean()
        frac = inb.mean()
        w_obs = np.where((zo >= lo) & (zo < hi), zo - zbar, 0.0)
        g = conditional(zt, w_obs, z, lambda gg: np.where((gg >= lo) & (gg < hi), gg - zbar, 0.0)) / frac
        kernels.append((b * n_all * g, n_all * (1 + z) * np.gradient(g, z)))
        labels.append(f"ARF {name} (zbar {zbar:.2f})")
    cosmo = Cosmology()
    ells = np.arange(1, 31)
    C = weighted_field_cls(cosmo, z, kernels, ells)
    dens_only = weighted_field_cls(cosmo, z, [(Wd, 0 * Wv) for Wd, Wv in kernels], ells[:3])
    for i, lab in enumerate(labels):
        c = C[(i, i)]
        print(f"{lab:<24} C_l x1e6 at l=1,2,3,5,10,20,30: " +
              " ".join(f"{c[l - 1] * 1e6:.3f}" for l in (1, 2, 3, 5, 10, 20, 30)) +
              f"   (density only at l=1-3: {' '.join(f'{v * 1e6:.3f}' for v in dens_only[(i, i)])})")
    sig_v = C_KMS * np.sqrt(3 * C[(0, 0)][0] / (4 * np.pi))
    print(f"clustering contribution to the redshift dipole (full sky): {sig_v:.1f} km/s per component")
    np.savez("out/arf_theory.npz", ells=ells, labels=np.array(labels),
             **{f"C_{i}_{j}": v for (i, j), v in C.items()})


if __name__ == "__main__":
    main()
