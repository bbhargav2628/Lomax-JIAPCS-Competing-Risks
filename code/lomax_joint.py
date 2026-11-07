"""
Joint Improved Adaptive Progressive Type-II censoring (JIAPCS) with Lomax competing risks.

Population d (d=1,2):   latent cause lifetimes  W_dr ~ Lomax(alpha_dr, lam_d),   S(x) = (1 + lam_d x)^(-alpha_dr)
Observed data are split by population.  For population d the likelihood (up to a constant) is
    l_d = N_d1 ln a_d1 + N_d2 ln a_d2 + N_d ln lam_d - sum_{failures in d} ln(1+lam_d x) - (a_d1+a_d2) D_d(lam_d)
    D_d(lam) = sum_j tw_j ln(1 + lam tt_j)          (tt = every event time of the joint test, tw = number of
                                                      population-d units that failed or were withdrawn then)
so each population is analysed with the same 3-parameter engine.
"""
import os
import sys

# Ensure local venv site-packages is discoverable if needed
for p in [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".venv", "Lib", "site-packages")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "Lib", "site-packages")),
]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from scipy import special as sp

def lomax_q(u, alpha, lam):
    """Lomax quantile function Q(u; alpha, lam)."""
    return ((1.0 - u) ** (-1.0 / alpha) - 1.0) / lam

def make_scheme(kind, k, tot=6):
    """Removal vector R_1..R_{k-1} (length k-1) for progressive schemes."""
    R = np.zeros(k - 1, dtype=int)
    if kind == "early":
        R[0] = tot
    elif kind == "late":
        R[-1] = tot
    elif kind == "mid":
        R[(k - 1) // 2 - 1] = tot // 2
        R[(k - 1) // 2] = tot - tot // 2
    return R

# ---------------------------------------------------------------------------------------------
# Data generation: balanced joint progressive censoring with the improved adaptive rule
# ---------------------------------------------------------------------------------------------
def gen_joint(rng, reps, m1, m2, k, R, T1, T2, th1, th2):
    """th_d = (alpha_d1, alpha_d2, lam_d).  R has length k-1.  Returns list of two data dicts."""
    R = np.asarray(R, int)
    thetas = (th1, th2)
    K = k + 1
    out = []
    for d in range(2):
        out.append(dict(x=np.zeros((reps, k)), mask=np.zeros((reps, k), bool), delta=np.zeros((reps, k), int),
                        tt=np.zeros((reps, K)), tw=np.zeros((reps, K)), N=np.zeros(reps, int),
                        N1=np.zeros(reps, int), N2=np.zeros(reps, int)))
    case = np.zeros(reps, int); kfail = np.zeros(reps, int)
    ms = (m1, m2)
    for r in range(reps):
        alive = []
        for d in range(2):
            a1, a2, lam = thetas[d]
            u = rng.random(ms[d])
            alive.append(np.sort(lomax_q(u, a1 + a2, lam)))
        steps = []                       # (time, pop, cens1, cens2)
        cnt = 0; tend = None; stop = None
        while cnt < k:
            n1 = alive[0][0] if len(alive[0]) else np.inf
            n2 = alive[1][0] if len(alive[1]) else np.inf
            t = min(n1, n2)
            if t >= T2:
                stop = 3; tend = T2; break
            z = 0 if n1 <= n2 else 1
            alive[z] = alive[z][1:]
            cnt += 1
            if cnt == k:
                steps.append((t, z, 0, 0)); tend = t
                stop = 1 if t < T1 else 2
                break
            Ri = R[cnt - 1] if t < T1 else 0
            c = [Ri + 1, Ri + 1]; c[z] = Ri          # balanced rule: failing population loses Ri extra, other Ri+1
            for d in range(2):
                cd = min(c[d], len(alive[d]))
                if cd > 0:
                    keep = np.ones(len(alive[d]), bool)
                    keep[rng.choice(len(alive[d]), cd, replace=False)] = False
                    alive[d] = alive[d][keep]
                c[d] = cd
            steps.append((t, z, c[0], c[1]))
        case[r] = stop; kfail[r] = cnt
        for d in range(2):
            a1, a2, lam = thetas[d]
            j = 0; q = 0
            for (t, z, c1, c2) in steps:
                w = (1 if z == d else 0) + (c1 if d == 0 else c2)
                out[d]["tt"][r, q] = t; out[d]["tw"][r, q] = w; q += 1
                if z == d:
                    out[d]["x"][r, j] = t; out[d]["mask"][r, j] = True
                    out[d]["delta"][r, j] = 1 if rng.random() < a1 / (a1 + a2) else 2
                    j += 1
            out[d]["tt"][r, q] = tend; out[d]["tw"][r, q] = len(alive[d])     # final withdrawal
            out[d]["N"][r] = j
            out[d]["N1"][r] = ((out[d]["delta"][r] == 1) & out[d]["mask"][r]).sum()
            out[d]["N2"][r] = ((out[d]["delta"][r] == 2) & out[d]["mask"][r]).sum()
    for d in range(2):
        out[d]["case"] = case; out[d]["kfail"] = kfail
    return out

def subset(d, idx):
    return {k_: v[idx] for k_, v in d.items()}

# ---------------------------------------------------------------------------------------------
# Likelihood functions (vectorised over replications)
# ---------------------------------------------------------------------------------------------
def D_fun(d, lam):
    lam = np.asarray(lam)
    if lam.ndim == 1:
        return (np.log1p(lam[:, None] * d["tt"]) * d["tw"]).sum(1)
    return (np.log1p(lam[:, :, None] * d["tt"][:, None, :]) * d["tw"][:, None, :]).sum(2)

def Dp_fun(d, lam):
    return (d["tt"] * d["tw"] / (1 + lam[:, None] * d["tt"])).sum(1)

def E_fun(d, lam):
    return (d["tt"] ** 2 * d["tw"] / (1 + lam[:, None] * d["tt"]) ** 2).sum(1)

def sumlog(d, lam):
    lam = np.asarray(lam)
    if lam.ndim == 1:
        return (np.log1p(lam[:, None] * d["x"]) * d["mask"]).sum(1)
    return (np.log1p(lam[:, :, None] * d["x"][:, None, :]) * d["mask"][:, None, :]).sum(2)

def profile_ll(d, lam):
    lam = np.asarray(lam); N = d["N"].astype(float)
    if lam.ndim == 1:
        return N * np.log(lam) - sumlog(d, lam) - N * np.log(D_fun(d, lam))
    return N[:, None] * np.log(lam) - sumlog(d, lam) - N[:, None] * np.log(D_fun(d, lam))

def profile_score(d, lam):
    N = d["N"].astype(float)
    sx = (d["x"] * d["mask"] / (1 + lam[:, None] * d["x"])).sum(1)
    return N / lam - sx - N * Dp_fun(d, lam) / D_fun(d, lam)

def existence_stat(d):
    """Derivative of the profile log-likelihood at lam -> 0+ ; > 0 guarantees a finite MLE of lam."""
    N = d["N"].astype(float)
    S1 = (d["x"] * d["mask"]).sum(1)
    W = (d["tw"] * d["tt"]).sum(1); Q = (d["tw"] * d["tt"] ** 2).sum(1)
    return N * Q / (2 * W) - S1

# ---------------------------------------------------------------------------------------------
# Maximum likelihood (unrestricted and order-restricted alpha_d1 <= alpha_d2)
# ---------------------------------------------------------------------------------------------
def mle(d, ngrid=400, chunk=250):
    reps = len(d["N"])
    lam_hat = np.full(reps, np.nan); ok = np.zeros(reps, bool)
    W = (d["tw"] * d["tt"]).sum(1)
    S = d["N"].clip(1) / np.maximum(W, 1e-300)
    u = np.linspace(-3.0, 3.0, ngrid)
    for s in range(0, reps, chunk):
        sl = slice(s, min(s + chunk, reps)); dd = subset(d, sl)
        grid = S[sl, None] * 10 ** u[None, :]
        lp = profile_ll(dd, grid)
        j = np.nanargmax(lp, axis=1)
        inter = (j > 0) & (j < ngrid - 1)
        lo = np.take_along_axis(grid, np.clip(j - 1, 0, ngrid - 1)[:, None], 1)[:, 0]
        hi = np.take_along_axis(grid, np.clip(j + 1, 0, ngrid - 1)[:, None], 1)[:, 0]
        for _ in range(60):
            mid = np.sqrt(lo * hi); sc = profile_score(dd, mid)
            lo = np.where(sc > 0, mid, lo); hi = np.where(sc > 0, hi, mid)
        lam_hat[sl] = np.sqrt(lo * hi)
        ok[sl] = inter & (d["N1"][sl] >= 1) & (d["N2"][sl] >= 1)
    Dh = D_fun(d, np.where(ok, lam_hat, 1.0))
    a1 = d["N1"] / Dh; a2 = d["N2"] / Dh
    a1r = np.minimum(d["N1"], d["N"] / 2.0) / Dh
    a2r = np.maximum(d["N2"], d["N"] / 2.0) / Dh
    return dict(a1=a1, a2=a2, lam=lam_hat, a1r=a1r, a2r=a2r, ok=ok)

def obs_info(d, est):
    lam = np.where(est["ok"], est["lam"], 1.0)
    a1 = np.maximum(np.where(est["ok"], est["a1"], 1.0), 1e-12)
    a2 = np.maximum(np.where(est["ok"], est["a2"], 1.0), 1e-12)
    Dp = Dp_fun(d, lam); E = E_fun(d, lam); N = d["N"].astype(float)
    sx2 = (d["x"] ** 2 * d["mask"] / (1 + lam[:, None] * d["x"]) ** 2).sum(1)
    I = np.zeros((len(lam), 3, 3))
    I[:, 0, 0] = d["N1"] / a1 ** 2; I[:, 1, 1] = d["N2"] / a2 ** 2
    I[:, 0, 2] = I[:, 2, 0] = Dp; I[:, 1, 2] = I[:, 2, 1] = Dp
    I[:, 2, 2] = N / lam ** 2 - sx2 - (a1 + a2) * E
    return I

def aci(d, est, z=1.959964):
    I = obs_info(d, est); cov = np.full(I.shape, np.nan)
    for r in np.where(est["ok"])[0]:
        try:
            if np.linalg.det(I[r]) > 0 and I[r, 2, 2] > 0 and I[r, 0, 0] > 0 and I[r, 1, 1] > 0:
                cov[r] = np.linalg.inv(I[r])
        except (np.linalg.LinAlgError, ValueError):
            pass
    th = np.stack([est["a1"], est["a2"], est["lam"]], 1)
    diag_cov = np.stack([cov[:, i, i] for i in range(3)], 1)
    diag_cov = np.where(diag_cov > 0, diag_cov, np.nan)
    sd = np.sqrt(diag_cov)
    rel = np.nan_to_num(sd / np.maximum(th, 1e-12), nan=0.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        exp_arg = np.clip(z * rel, 0.0, 300.0)
        llo = th * np.exp(-exp_arg)
        lhi = np.where(z * rel > 300.0, np.inf, th * np.exp(exp_arg))
    return dict(th=th, sd=sd, lo=th - z * sd, hi=th + z * sd, llo=llo, lhi=lhi)

# ---------------------------------------------------------------------------------------------
# Bayesian estimation: collapsed Gibbs sampler (alpha integrated out analytically, MH on log lam)
# ---------------------------------------------------------------------------------------------
def log_marg_lam(d, lam, pr, restricted):
    Dl = D_fun(d, lam); N = d["N"].astype(float)
    s1 = pr["a1"] + d["N1"]; s2 = pr["a2"] + d["N2"]
    r1 = pr["b1"] + Dl; r2 = pr["b2"] + Dl
    lp = ((pr["c"] + N - 1) * np.log(lam) - pr["dl"] * lam - sumlog(d, lam) - s1 * np.log(r1) - s2 * np.log(r2))
    if restricted:
        bet = np.clip(sp.betainc(s1, s2, r1 / (r1 + r2)), 1e-300, 1.0)
        lp = lp + np.log(bet)
    return lp

def mcmc(d, pr, restricted, rng, iters=2000, burn=800, init=None):
    reps = len(d["N"])
    W = (d["tw"] * d["tt"]).sum(1); S = d["N"].clip(1) / np.maximum(W, 1e-300)
    lam = S.copy() if init is None else np.where(np.isfinite(init) & (init > 0), init, S)
    step = np.full(reps, 0.5); cur = log_marg_lam(d, lam, pr, restricted)
    draws = np.empty((reps, iters, 3)); acc_tot = np.zeros(reps)
    for t in range(iters + burn):
        prop = lam * np.exp(np.clip(step * rng.standard_normal(reps), -50.0, 50.0))
        lp = log_marg_lam(d, prop, pr, restricted)
        u_rand = np.clip(rng.random(reps), 1e-300, 1.0)
        acc = np.log(u_rand) < lp - cur + np.log(prop) - np.log(lam)
        lam = np.where(acc, prop, lam); cur = np.where(acc, lp, cur)
        if t < burn:
            step *= np.exp((acc - 0.35) / np.sqrt(t + 10.0))
            step = np.clip(step, 1e-3, 10.0)
        else:
            acc_tot += acc; k = t - burn
            Dl = D_fun(d, lam)
            s1 = pr["a1"] + d["N1"]; s2 = pr["a2"] + d["N2"]
            r1 = pr["b1"] + Dl; r2 = pr["b2"] + Dl
            if not restricted:
                a1 = rng.gamma(s1) / r1; a2 = rng.gamma(s2) / r2
            else:
                c = np.clip(sp.betainc(s1, s2, r1 / (r1 + r2)), 1e-300, 1.0)
                B = sp.betaincinv(s1, s2, np.clip(rng.random(reps) * c, 1e-300, 1.0 - 1e-16))
                Sg = rng.gamma(s1 + s2); a1 = B * Sg / r1; a2 = (1 - B) * Sg / r2
            draws[:, k, 0] = a1; draws[:, k, 1] = a2; draws[:, k, 2] = lam
    return draws, acc_tot / iters

def bayes_est(draws, p=0.5, c=0.5):
    """
    Computes Bayes estimators under SELF, LINEX (p=+-0.5), and GELF (c=+-0.5).
    Uses numerically stable log-sum-exp formulation to prevent exponential overflow.
    """
    se = draws.mean(1)
    
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # LINEX+ : -(1/p) * ln E[exp(-p*theta)]
        val_minus = -p * draws
        m_minus = np.max(val_minus, axis=1, keepdims=True)
        linex_plus = -(m_minus[:, 0, :] + np.log(np.maximum(np.mean(np.exp(val_minus - m_minus), axis=1), 1e-300))) / p

        # LINEX- : (1/p) * ln E[exp(p*theta)] using log-sum-exp to prevent overflow
        val_plus = p * draws
        m_plus = np.max(val_plus, axis=1, keepdims=True)
        linex_minus = (m_plus[:, 0, :] + np.log(np.maximum(np.mean(np.exp(val_plus - m_plus), axis=1), 1e-300))) / p

        # GELF+ and GELF- with safe clipping to prevent divide by zero
        safe_draws = np.maximum(draws, 1e-12)
        ge_plus = np.maximum(np.mean(safe_draws ** (-c), axis=1), 1e-300) ** (-1.0 / c)
        ge_minus = np.maximum(np.mean(safe_draws ** c, axis=1), 1e-300) ** (1.0 / c)

    return {"SE": se, "LINEX+": linex_plus, "LINEX-": linex_minus, "GE+": ge_plus, "GE-": ge_minus}

def hpd(draws, level=0.95):
    reps, M, _ = draws.shape
    srt = np.sort(draws, axis=1); w = int(np.floor(level * M))
    j = (srt[:, w:, :] - srt[:, :M - w, :]).argmin(1)
    lo = np.take_along_axis(srt, j[:, None, :], 1)[:, 0, :]
    hi = np.take_along_axis(srt, (j + w)[:, None, :], 1)[:, 0, :]
    return lo, hi
