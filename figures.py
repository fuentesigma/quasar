#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

import sys
from pathlib import Path

import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import plotstyle
from quaia import C_KMS, GC, desi_match, dipole, load, ols
from results import perm_dipole

OUT = Path("figs")
COL, FULL = 3.46, 7.09
BLUE, MAGENTA, TEAL = "#1f5fa8", "#b0306a", "#2a9d8f"
TINT, TINT_2 = "#e4ebf5", "#9db4d6"
INK, INK_2, MUTED, BAND = plotstyle.INK, plotstyle.INK_2, plotstyle.MUTED, TINT
V_MOTION = 370.0


def style():
    plotstyle.use()
    mpl.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.6,
                         "lines.markersize": 4.5, "figure.dpi": 150, "pdf.fonttype": 42})


SKY_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "quaia", [BLUE, "#8fafd4", "#f6f6f4", "#d895b3", MAGENTA])
SKY_RANGE = (-0.07, 0.0, 0.025)
SKY_EDGE, SKY_EDGE_LW = "white", 3.0


def galactic_mean_map(d, values, nside=64, fwhm_deg=12.0):
    pix = hp.ang2pix(nside, d["l"], d["b"], lonlat=True)
    npix = hp.nside2npix(nside)
    s = np.bincount(pix, weights=values, minlength=npix)
    n = np.bincount(pix, minlength=npix).astype(float)
    fw = np.radians(fwhm_deg)
    s_sm, n_sm = hp.smoothing(s, fwhm=fw), hp.smoothing(n, fwhm=fw)
    m = s_sm / np.maximum(n_sm, 1e-3 * n_sm.max())
    cover = hp.smoothing((n > 0).astype(float), fwhm=fw)
    return m, cover


def mollweide(ax, m, cover, norm, cmap):
    lon = np.radians(np.linspace(-180, 180, 1441))
    lat = np.radians(np.linspace(-90, 90, 721))
    LON, LAT = np.meshgrid(lon, lat)
    ll, bb = np.degrees(-LON) % 360, np.degrees(LAT)
    val = hp.get_interp_val(m, ll, bb, lonlat=True)
    cov = hp.get_interp_val(cover, ll, bb, lonlat=True)
    val = np.ma.masked_where(cov < 0.5, val)
    im = ax.pcolormesh(LON, LAT, val, cmap=cmap, norm=norm, shading="auto", rasterized=True)
    ax.contour(LON, LAT, cov, levels=[0.5], colors=[SKY_EDGE], linewidths=SKY_EDGE_LW)
    ax.spines["geo"].set(edgecolor=SKY_EDGE, linewidth=SKY_EDGE_LW)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.grid(color="white", lw=0.4, alpha=0.5)
    ax.set_facecolor("white")
    return im


def mark(ax, l, b, marker, **kw):
    x = -np.radians(((l + 180) % 360) - 180)
    ax.plot(x, np.radians(b), marker, **kw)


def fig_sky(d):
    _, _, c = dipole(d["n"], d["y"], d["T"])
    pred = d["T"] @ c
    data, cover = galactic_mean_map(d, d["y"] - d["y"].mean())
    model, _ = galactic_mean_map(d, pred - pred.mean())
    lo, mid, hi = SKY_RANGE
    norm = mpl.colors.TwoSlopeNorm(vmin=lo, vcenter=mid, vmax=hi)
    cmap = SKY_CMAP
    fig = plt.figure(figsize=(FULL, 2.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.03], wspace=0.04)
    for i, m in enumerate((data, model)):
        ax = fig.add_subplot(gs[0, i], projection="mollweide")
        im = mollweide(ax, m, cover, norm, cmap)
        mark(ax, 0, 0, "+", color="white", ms=7, mew=1.2)
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_label(r"$\Delta\langle\ln(1+z)\rangle$", labelpad=2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.set_ticks([-0.06, -0.04, -0.02, 0.0, 0.02])
    fig.savefig(OUT / "sky.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig_dust(d):
    D = desi_match(d)
    m = np.isfinite(D["z"])
    ebv = d["T_raw"][m, 0]
    edges = np.quantile(ebv, np.linspace(0, 1, 11))
    idx = np.clip(np.digitize(ebv, edges) - 1, 0, 9)
    xc = np.array([np.median(ebv[idx == i]) for i in range(10)])
    _, patch = np.unique(hp.ang2pix(8, d["ra"][m], d["dec"][m], lonlat=True), return_inverse=True)
    P = patch.max() + 1
    rng = np.random.default_rng(13)
    boot_w = np.array([np.bincount(rng.integers(0, P, P), minlength=P) for _ in range(500)], float)

    def boot_err(y):
        key = patch * 10 + idx
        s = np.bincount(key, weights=y, minlength=10 * P).reshape(P, 10)
        n = np.bincount(key, minlength=10 * P).reshape(P, 10)
        return ((boot_w @ s) / np.maximum(boot_w @ n, 1)).std(axis=0)
    x0, x1 = xc[0], 0.27
    fig, ax = plt.subplots(figsize=(COL, 2.7))
    ax.axhspan(-V_MOTION / C_KMS * 1e3, V_MOTION / C_KMS * 1e3, color=BAND, lw=0, zorder=0)
    slope = {}
    for y, color, name in ((d["y"][m], BLUE, "Quaia"), (np.log1p(D["z"][m]), INK, "DESI")):
        coef, _ = ols(ebv[:, None], y)
        a0 = y.mean() - coef[0] * ebv.mean()
        ref = a0 + coef[0] * x0
        slope[name] = coef[0]
        mean = np.array([y[idx == i].mean() for i in range(10)]) - ref
        err = boot_err(y)
        xs = np.array([x0, x1])
        ax.plot(xs, coef[0] * (xs - x0) * 1e3, color=color, lw=1.2, zorder=2)
        ax.errorbar(xc, mean * 1e3, err * 1e3, fmt="o", color=color, mec="white", mew=0.6, ms=4.5,
                    elinewidth=0.8, zorder=3)
    yq, yd = slope["Quaia"] * (x1 - x0) * 1e3, slope["DESI"] * (x1 - x0) * 1e3
    xa = x1 + 0.008
    for lo, hi, text in ((yd, 0.0, "selection"), (yq, yd, "redshift\nerrors")):
        ax.annotate("", (xa, lo), (xa, hi), arrowprops=dict(arrowstyle="<->", color=INK_2, lw=0.8,
                                                             shrinkA=0, shrinkB=0))
        ax.text(xa + 0.006, 0.5 * (lo + hi), text, fontsize=6.5, color=INK_2, va="center")
    ax.text(0.205, slope["DESI"] * (0.205 - x0) * 1e3 + 1.6, "DESI", color=INK, fontsize=7, ha="center")
    ax.text(0.205, slope["Quaia"] * (0.205 - x0) * 1e3 - 3.4, "Quaia", color=BLUE, fontsize=7, ha="center")
    ax.text(0.16, 0.0, "shift from our motion", fontsize=6.5, color=INK_2, va="center")
    ax.set_xlim(0.0, 0.33); ax.set_ylim(-17, 4)
    ax.set_xlabel(r"dust extinction $A_V$  [mag]")
    ax.set_ylabel(r"$\langle\ln(1+z)\rangle$, relative  [$10^{-3}$]")
    fig.savefig(OUT / "dust.pdf", bbox_inches="tight")
    plt.close(fig)


def panel_dipole(ax, d):
    rows = [("whole sky", np.ones(len(d["y"]), bool)), (r"$|b|>30^\circ$", np.abs(d["b"]) > 30),
            (r"$|b|<30^\circ$", np.abs(d["b"]) < 30), (r"$|\delta|>30^\circ$", np.abs(d["dec"]) > 30),
            (r"$|\delta|<30^\circ$", np.abs(d["dec"]) < 30), (r"$G<20.0$", d["G"] < 20.0),
            (r"$20.0<G<20.5$", d["G"] >= 20.0)]
    o = np.load("out/redshift_mocks.npz")
    names = list(o["dip_names"])
    j = names.index("|b|>30 S>0.5: templates")
    mocks = o["dip_cells"][:, j]
    corr = o["real_dip"][j] - (mocks.mean(0) - o["v_kin"])
    corr_gc, corr_err = corr @ GC * C_KMS, np.sqrt(GC @ np.cov(mocks.T) @ GC) * C_KMS
    ax.axvspan(-V_MOTION, V_MOTION, color=BAND, lw=0, zorder=0)
    ax.axvline(0, color=plotstyle.AXIS, lw=0.8, zorder=1)
    for i, (name, k) in enumerate(rows):
        n, y, T = d["n"][k], d["y"][k], d["T"][k]
        v_raw, cov = perm_dipole(d, k)
        _, _, c = dipole(n, y, T)
        v_map = dipole(n, T @ c)[0]
        yi = len(rows) - i
        ax.errorbar(v_raw @ GC * C_KMS, yi + 0.13, xerr=np.sqrt(GC @ cov @ GC) * C_KMS, fmt="o", color=BLUE,
                    mec="white", mew=0.6, elinewidth=0.8, zorder=3)
        ax.plot(v_map @ GC * C_KMS, yi - 0.13, "o", mfc="white", mec=MAGENTA, mew=1.4, zorder=3)
    ax.errorbar(corr_gc, -0.4, xerr=corr_err, fmt="D", color=TEAL, mec="white", mew=0.6, elinewidth=0.8, ms=4.5)
    ax.set_yticks(list(range(len(rows), 0, -1)) + [-0.4])
    ax.set_yticklabels([r[0] for r in rows] + [r"corrected, $|b|>30^\circ$"])
    ax.get_yticklabels()[-1].set_color(INK)
    ax.axhline(0.35, color=plotstyle.GRID, lw=0.8)
    ax.set_xlabel("dipole velocity toward the Galactic Centre  [km/s]")
    ax.set_xlim(-900, 6800)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-1.0, len(rows) + 0.6)
    ax.text(-250, 0.35 + 0.5 * (len(rows) + 0.25), "our motion", rotation=90, fontsize=6.5, color=INK_2,
            ha="center", va="center")


def panel_arf(ax, d, nside=128, n_shuffle=20, seed=4):
    from mocks import arf_cl, ELL_BINS
    rng = np.random.default_rng(seed)
    mask = hp.ud_grade(hp.read_map("data/selection_function_NSIDE64_G20.5.fits"), nside) > 0.5
    pix = hp.ang2pix(nside, d["ra"], d["dec"], lonlat=True)
    keep = mask[pix]
    zmed = np.median(d["z"][keep])
    sb = keep & (d["z"] >= zmed)
    p, z = pix[sb], d["z"][sb]
    nbar = np.bincount(p, minlength=hp.nside2npix(nside))[mask].mean()
    T = (d["T_raw"][sb] - d["T_raw"][sb].mean(0)) / d["T_raw"][sb].std(0)
    coef, _ = ols(T, z)
    zhat = T @ coef
    bins = ELL_BINS[1:]
    cl = lambda v: arf_cl(p, v - v.mean(), nbar, mask)[1:]
    sh = np.array([cl(rng.permutation(z)) for _ in range(n_shuffle)])
    noise = sh.mean(0)
    nsd = np.load("out/redshift_mocks.npz")["arf_none"][:, 1, 0, 1:].std(axis=0)
    rows = {"raw": cl(z) - noise, "maps": cl(zhat), "corrected": cl(z - zhat) - noise}
    t = np.load("out/arf_theory.npz")
    th = np.array([t["C_2_2"][a - 1:b - 1].mean() for a, b in bins])
    lc = np.array([np.sqrt(a * (b - 1)) for a, b in bins])
    ax.fill_between(lc, 1e-9, nsd * 1e6, color=BAND, lw=0, zorder=0, alpha=0.8)
    ax.plot(lc, th * 1e6, "--", color=INK_2, lw=1.1, zorder=2)
    for key, color, mfc in (("raw", BLUE, BLUE), ("maps", MAGENTA, "white"), ("corrected", TEAL, TEAL)):
        v = rows[key] * 1e6
        pos = v > 0
        ax.plot(lc, np.where(pos, v, np.nan), "-", color=color, lw=1.0, alpha=0.55, zorder=2)
        ax.plot(lc[pos], v[pos], "o", color=color, mfc=mfc, mec=color if mfc == "white" else "white",
                mew=1.2 if mfc == "white" else 0.6, zorder=3)
        if (~pos).any():
            ax.plot(lc[~pos], nsd[~pos] * 1e6, "v", color=color, ms=4, zorder=3)
    L = mpl.lines.Line2D
    handles = [L([], [], marker="o", color=BLUE, mec="white", lw=1.0, label="measured"),
                       L([], [], marker="o", color=MAGENTA, mfc="white", mew=1.2, lw=1.0, label="six maps alone"),
                       L([], [], marker="o", color=TEAL, mec="white", lw=1.0, label="corrected"),
                       L([], [], marker="v", ls="", color=TEAL, label="upper limit"),
                       L([], [], ls="--", color=INK_2, lw=1.1, label="linear theory"),
            mpl.patches.Patch(color=BAND, label=r"isotropic mocks, $1\sigma$")]
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([2, 3, 5, 10, 20]); ax.set_xticklabels(["2", "3", "5", "10", "20"])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlim(2, 28); ax.set_ylim(0.05, 200)
    ax.set_xlabel(r"multipole $\ell$")
    ax.set_ylabel(r"$C_\ell$ of the mean redshift map  [$10^{-6}$]")
    return handles


def fig_large(d):
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.9), layout="constrained")
    handles = panel_arf(a, d)
    panel_dipole(b, d)
    fig.legend(handles=handles, loc="outside upper center", ncol=6, handlelength=1.8, columnspacing=1.2,
               frameon=False)
    for ax, tag in ((a, "a"), (b, "b")):
        ax.set_title(tag, loc="left", x=-0.07, y=1.0, pad=4, fontsize=10, fontweight="bold")
    fig.savefig(OUT / "large.pdf", bbox_inches="tight")
    plt.close(fig)


FNL_BINS = [(2, 4), (4, 6), (6, 8), (8, 10), (10, 14), (14, 20), (20, 30), (30, 45), (45, 61)]
LMIN_ARF, LMIN_COUNTS = 10, 15


def fig_fnl():
    o = np.load("out/arf_fnl.npz")
    cnt = np.load("out/fnl_curve.npz")
    ells, lmins = o["ells"], o["lmins"]
    sig10 = o["sigma"][list(lmins).index(LMIN_ARF)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.9), layout="constrained")
    L = mpl.lines.Line2D

    band = lambda v: np.array([v[(ells >= lo) & (ells < hi)].mean() for lo, hi in FNL_BINS])
    lc = np.array([np.sqrt(lo * (hi - 1)) for lo, hi in FNL_BINS])
    a.axvspan(2, LMIN_ARF, color=BAND, lw=0, zorder=0)
    a.text(np.sqrt(2 * LMIN_ARF), 1.4e-2, r"$\ell<10$, not used", color=INK_2, fontsize=7, ha="center")
    a.plot(ells, o["C_11"] * 1e6, "--", color=INK_2, lw=1.1, zorder=2)
    a.plot(ells, np.abs(o["dF_11"]) * sig10 * 1e6, "-", color=INK, lw=1.1, zorder=2)
    for key, color, mfc in (("raw_1", BLUE, BLUE), ("templates_1", MAGENTA, "white"), ("corrected_1", TEAL, TEAL)):
        v = band(o[key]) * 1e6
        pos = v > 0
        a.plot(lc, np.where(pos, v, np.nan), "-", color=color, lw=1.0, alpha=0.55, zorder=2)
        a.plot(lc[pos], v[pos], "o", color=color, mfc=mfc, mec=color if mfc == "white" else "white",
               mew=1.2 if mfc == "white" else 0.6, zorder=3)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xticks([2, 3, 5, 10, 20, 50]); a.set_xticklabels(["2", "3", "5", "10", "20", "50"])
    a.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    a.set_xlim(2, 60); a.set_ylim(1e-2, 200)
    a.set_xlabel(r"multipole $\ell$")
    a.set_ylabel(r"$C_\ell$ of the mean redshift map  [$10^{-6}$]")
    a.legend(handles=[L([], [], marker="o", color=BLUE, mec="white", lw=1.0, label="measured"),
                      L([], [], marker="o", color=MAGENTA, mfc="white", mew=1.2, lw=1.0, label="six maps alone"),
                      L([], [], marker="o", color=TEAL, mec="white", lw=1.0, label="corrected"),
                      L([], [], ls="--", color=INK_2, lw=1.1, label=r"theory, $f_{\rm NL}=0$"),
                      L([], [], ls="-", color=INK, lw=1.1, label=r"change for $1\sigma$ in $f_{\rm NL}$")],
             loc="upper right", handlelength=1.8)

    b.axhline(0, color=AXIS_C, lw=0.8, zorder=1)
    b.fill_between(cnt["lmin"], cnt["no_correction"].min(0), cnt["no_correction"].max(0), color=TINT_2,
                   alpha=0.8, lw=0, zorder=2)
    b.plot(lmins, o["curve_templates"], "-", color=MAGENTA, zorder=3)
    b.plot(lmins, o["curve_templates_p_residual"], ":", color=MAGENTA, zorder=3)
    for lm, lab in ((LMIN_ARF, "mean redshift"), (LMIN_COUNTS, "counts")):
        b.axvline(lm, color=INK_2, lw=0.8, ls=(0, (1, 2)), zorder=1)
        b.text(lm - 0.3, 11.5, lab, color=INK_2, fontsize=7, va="top", ha="right", rotation=90)
    b.set_xlim(2, 40); b.set_ylim(-0.5, 12)
    b.set_xlabel(r"smallest multipole $\ell_{\rm min}$ of the auto spectra")
    b.set_ylabel(r"bias $\Delta f_{\rm NL}/\sigma(f_{\rm NL})$")
    b.legend(handles=[L([], [], ls="-", color=MAGENTA, label="mean redshift, six maps"),
                      L([], [], ls=":", color=MAGENTA, label="mean redshift, with residual excess"),
                      mpl.patches.Patch(color=TINT_2, alpha=0.8, label="counts, bin migration")],
             loc="center right", handlelength=1.8)
    for ax, tag in ((a, "a"), (b, "b")):
        ax.set_title(tag, loc="left", x=-0.07, y=1.0, pad=4, fontsize=10, fontweight="bold")
    fig.savefig(OUT / "fnl.pdf", bbox_inches="tight")
    plt.close(fig)


AXIS_C = plotstyle.AXIS


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    style()
    which = sys.argv[1:] or ["sky", "dust", "large", "fnl"]
    if "fnl" in which:
        fig_fnl()
    if which != ["fnl"]:
        d = load()
        if "sky" in which:
            fig_sky(d)
        if "dust" in which:
            fig_dust(d)
        if "large" in which:
            fig_large(d)
