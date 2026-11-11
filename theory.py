#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

import camb
import numpy as np
from scipy.special import spherical_jn

C_KMS = 299792.458
DELTA_C = 1.686
L_EXACT = 40
PLANCK18 = dict(H0=67.66, ombh2=0.02242, omch2=0.11933, ns=0.9665, As=2.105e-9, mnu=0.06)


def b_laurent(z):
    return 0.278 * ((1 + z) ** 2 - 6.565) + 2.393


class Cosmology:
    def __init__(self, params=PLANCK18, zmax=6.0):
        pars = camb.set_params(**params, WantTransfer=True, kmax=20.0, redshifts=[0.0])
        self.res = camb.get_results(pars)
        self.h = params["H0"] / 100
        self.om = (params["ombh2"] + params["omch2"]) / self.h ** 2 + params["mnu"] / 93.14 / self.h ** 2
        self.pk = camb.get_matter_power_interpolator(pars, nonlinear=False, hubble_units=False, k_hunit=False,
                                                     kmax=20.0, zmax=max(zmax, 50.0))
        self.chi_star = self.res.conformal_time(0) - self.res.tau_maxvis
        tr = self.res.get_matter_transfer_data()
        kt = tr.transfer_data[0, :, 0] * self.h
        tt = tr.transfer_data[camb.model.Transfer_tot - 1, :, 0]
        self._lnk_T, self._T = np.log(kt), tt / tt[0]
        kref = 1e-4
        d50 = np.sqrt(self.pk.P(50.0, kref) / self.pk.P(0.0, kref))
        self.d0_md = (1 / 51.0) / d50

    def T(self, k):
        return np.interp(np.log(k), self._lnk_T, self._T)

    def chi(self, z):
        return self.res.comoving_radial_distance(z)

    def H(self, z):
        return self.res.hubble_parameter(z)

    def growth(self, z):
        z = np.asarray(z, float)
        g = np.sqrt(self.pk.P(z.ravel(), 1e-4) / self.pk.P(0.0, 1e-4)).ravel()
        return g.reshape(z.shape)

    def delta_b(self, k, z, b, fnl, p=1.0):
        D_md = self.growth(z) * self.d0_md
        H0 = self.h * 100
        return 3 * fnl * (b - p) * DELTA_C * self.om * (H0 / C_KMS) ** 2 / (k ** 2 * self.T(k) * D_md)


class Spectra:

    def __init__(self, cosmo, z, nz_list, nk=3000, kmin=1e-5, kmax=0.15):
        self.c, self.z = cosmo, z
        self.nz = [n / np.trapezoid(n, z) for n in nz_list]
        self.chi = cosmo.chi(z)
        self.D = cosmo.growth(z)
        self.k = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        self.p0 = cosmo.pk.P(0.0, self.k).ravel()
        H0 = cosmo.h * 100
        self.zk = np.linspace(1e-3, 10.0, 2000)
        chik = cosmo.chi(self.zk)
        self.wk = (1.5 * cosmo.om * (H0 / C_KMS) ** 2 * (1 + self.zk) * chik * (cosmo.chi_star - chik) /
                   cosmo.chi_star * C_KMS / cosmo.H(self.zk))
        self.chik, self.Dk = chik, cosmo.growth(self.zk)

    def _kernels_exact(self, ell, amps, fnl):
        jl = spherical_jn(ell, np.outer(self.k, self.chi))
        out = []
        for n, A in zip(self.nz, amps):
            b = A * b_laurent(self.z)
            btot = b[None, :] + self.c.delta_b(self.k[:, None], self.z[None, :], b[None, :], fnl)
            out.append(np.trapezoid(n[None, :] * btot * self.D[None, :] * jl, self.z, axis=1))
        jk = spherical_jn(ell, np.outer(self.k, self.chik))
        w = self.wk / self.chik ** 2
        lens = np.trapezoid(w[None, :] * self.Dk[None, :] * jk, self.zk, axis=1) * ell * (ell + 1) / self.k ** 2
        return out, lens

    def cls(self, ells, amps, fnl=0.0):
        nb = len(self.nz)
        out = {("g", i, j): np.zeros(len(ells)) for i in range(nb) for j in range(i, nb)}
        out.update({("k", i): np.zeros(len(ells)) for i in range(nb)})
        out[("k", "k")] = np.zeros(len(ells))
        for a, ell in enumerate(ells):
            if ell <= L_EXACT:
                W, Wk = self._kernels_exact(ell, amps, fnl)
                pref = 2 / np.pi * self.k ** 2 * self.p0
                for i in range(nb):
                    for j in range(i, nb):
                        out[("g", i, j)][a] = np.trapezoid(pref * W[i] * W[j], self.k)
                    out[("k", i)][a] = np.trapezoid(pref * W[i] * Wk, self.k)
                out[("k", "k")][a] = np.trapezoid(pref * Wk * Wk, self.k)
            else:
                out_l = self._limber(ell, amps, fnl)
                for key, v in out_l.items():
                    out[key][a] = v
        return out

    def _limber(self, ell, amps, fnl):
        z, chi = self.z, self.chi
        k = (ell + 0.5) / chi
        pk = self.c.pk.P(z, k, grid=False)
        dchidz = C_KMS / self.c.H(z)
        W = []
        for n, A in zip(self.nz, amps):
            b = A * b_laurent(z)
            W.append(n * (b + self.c.delta_b(k, z, b, fnl)) / dchidz)
        wk = np.interp(z, self.zk, self.wk) / dchidz * ell * (ell + 1) / (ell + 0.5) ** 2
        out = {}
        nb = len(W)
        for i in range(nb):
            for j in range(i, nb):
                out[("g", i, j)] = np.trapezoid(W[i] * W[j] * pk / chi ** 2 * dchidz, z)
            out[("k", i)] = np.trapezoid(W[i] * wk * pk / chi ** 2 * dchidz, z)
        zk, chik = self.zk, self.chik
        kk = (ell + 0.5) / chik
        pkk = self.c.pk.P(zk, kk, grid=False)
        wkk = self.wk / (C_KMS / self.c.H(zk)) * ell * (ell + 1) / (ell + 0.5) ** 2
        out[("k", "k")] = np.trapezoid(wkk ** 2 * pkk / chik ** 2 * C_KMS / self.c.H(zk), zk)
        return out


def growth_rate(cosmo, z):
    lnD = np.log(cosmo.growth(z))
    return -(1 + z) * np.gradient(lnD, z)


def weighted_field_cls(cosmo, z, kernels, ells, nk=3000, kmin=1e-5, kmax=0.15):
    k = np.logspace(np.log10(kmin), np.log10(kmax), nk)
    p0 = cosmo.pk.P(0.0, k).ravel()
    chi, D, f = cosmo.chi(z), cosmo.growth(z), growth_rate(cosmo, z)
    vfac = f * cosmo.H(z) / (1 + z) / C_KMS
    out = {(i, j): np.zeros(len(ells)) for i in range(len(kernels)) for j in range(i, len(kernels))}
    for a, ell in enumerate(ells):
        x = np.outer(k, chi)
        jl, djl = spherical_jn(ell, x), spherical_jn(ell, x, derivative=True)
        W = [np.trapezoid(Wd[None, :] * D[None, :] * jl - Wv[None, :] * (vfac * D)[None, :] * djl / k[:, None],
                          z, axis=1) for Wd, Wv in kernels]
        for i in range(len(W)):
            for j in range(i, len(W)):
                out[(i, j)][a] = np.trapezoid(2 / np.pi * k ** 2 * p0 * W[i] * W[j], k)
    return out
