import os
import sys

# Ensure execution in the script's directory so all relative paths resolve cleanly
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir:
    os.chdir(_script_dir)

# Ensure local venv site-packages is discoverable if needed
for p in [
    os.path.abspath(os.path.join(_script_dir, "..", "..", ".venv", "Lib", "site-packages")),
    os.path.abspath(os.path.join(_script_dir, "..", ".venv", "Lib", "site-packages")),
]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lomax_joint import *
from real_joint import apply_joint
from realdata import plane
from run_sim_joint import TRUTH, DESIGN, scheme, PQ

out = {}
# ---------- (i) uniqueness check: number of interior local maxima of the profile log-likelihood on a fine grid
rng = np.random.default_rng(1)
th1, th2 = TRUTH["A"]
m1, m2, k, tot = DESIGN["D2"]
q = lambda t, p: lomax_q(p, t[0] + t[1], t[2])
D = gen_joint(rng, 1000, m1, m2, k, scheme("mid", k, tot), q(th1, 0.55), q(th1, 0.90), th1, th2)
d = D[0]
W = (d["tw"] * d["tt"]).sum(1)
S = d["N"] / W
grid = S[:, None] * 10 ** np.linspace(-3, 3, 600)[None, :]
lp = profile_ll(d, grid)
dl = np.diff(lp, axis=1)
nmax = ((dl[:, :-1] > 0) & (dl[:, 1:] <= 0)).sum(1)  # interior local maxima
left = dl[:, 0] <= 0
e = mle(d)
ok = e["ok"]
out["uniq"] = dict(
    reps=1000,
    with_exactly_one_interior_max=int((nmax == 1).sum()),
    with_two_or_more=int((nmax >= 2).sum()),
    with_none=int((nmax == 0).sum()),
    exist_stat_pos=float((existence_stat(d) > 0).mean()),
    mle_exists=float(ok.mean()),
)

# ---------- (ii) vague prior experiment
from lomax_joint import mcmc, bayes_est

nip = dict(a1=1e-3, b1=1e-3, a2=1e-3, b2=1e-3, c=1e-3, dl=1e-3)
m1_, m2_, k_, t_ = DESIGN["D1"]
D1 = gen_joint(rng, 500, m1_, m2_, k_, scheme("mid", k_, t_), q(th1, 0.55), q(th1, 0.90), th1, th2)
dd = D1[0]
dr, acc = mcmc(dd, nip, False, rng, 2000, 800)
est = dr.mean(1)
med = np.median(dr, axis=1)
true = np.array(th1)
out["nip"] = dict(
    post_mean_avg=est.mean(0).tolist(),
    post_median_avg=med.mean(0).tolist(),
    truth=true.tolist(),
    mse_mean=np.mean((est - true) ** 2, 0).tolist(),
    mse_median=np.mean((med - true) ** 2, 0).tolist(),
    linex_minus_inf_frac=float((~np.isfinite(bayes_est(dr)["LINEX-"])).any(1).mean()),
)

# ---------- (iii) real data
t1, c1 = plane(1)
t2, c2 = plane(0)  # Omega1 = plane 7914 (n=24), Omega2 = plane 7913 (n=27)
kR = 18
R = np.zeros(kR - 1, int)
R[7] = 3
R[8] = 3  # 6 withdrawals in the middle: sum(R_i+1) = 23 < 24
T1, T2 = 0.45, 1.8
ex = []
first = None
for seed in range(200):
    Dr = apply_joint(np.random.default_rng(seed), t1, c1, t2, c2, kR, R, T1, T2)
    o = [bool(mle(Dr[0])["ok"][0]), bool(mle(Dr[1])["ok"][0])]
    ex.append(o)
    if first is None and all(o):
        first = seed
ex = np.array(ex)
out["real_design"] = dict(
    k=kR,
    R=R.tolist(),
    T1=T1,
    T2=T2,
    n1=24,
    n2=27,
    P_exist=ex.mean(0).tolist(),
    P_both=float(ex.all(1).mean()),
    seed_used=first,
)

# complete-data check
for g, (t, c) in {"plane7914": (t1, c1), "plane7913": (t2, c2)}.items():
    n = len(t)
    dc = dict(
        x=t[None],
        mask=np.ones((1, n), bool),
        tt=t[None],
        tw=np.ones((1, n)),
        N=np.array([n]),
        N1=np.array([(c == 1).sum()]),
        N2=np.array([(c == 2).sum()]),
    )
    out[f"complete_{g}"] = dict(
        n=n,
        N1=int((c == 1).sum()),
        N2=int((c == 2).sum()),
        CV2=float(t.var() / t.mean() ** 2),
        exist_stat=float(existence_stat(dc)[0]),
        mle_exists=bool(mle(dc)["ok"][0]),
    )

rng = np.random.default_rng(2026)
Dr = apply_joint(np.random.default_rng(first), t1, c1, t2, c2, kR, R, T1, T2)
res = dict(case=int(Dr[0]["case"][0]), k=int(Dr[0]["kfail"][0]))
chains = {}
priors = {
    "WIP": dict(a1=1.0, b1=1.0, a2=1.0, b2=1.0, c=1.0, dl=1.0),
    "Gamma(2,2)": dict(a1=2.0, b1=2.0, a2=2.0, b2=2.0, c=2.0, dl=2.0),
    "NIP": dict(a1=1e-3, b1=1e-3, a2=1e-3, b2=1e-3, c=1e-3, dl=1e-3),
}
for dnum in range(2):
    d = Dr[dnum]
    e = mle(d)
    r = {}
    r["N"], r["N1"], r["N2"] = int(d["N"][0]), int(d["N1"][0]), int(d["N2"][0])
    r["withdrawn_final"] = float(d["tw"][0, np.argmax(d["tt"][0])])
    r["mle"] = [float(e["a1"][0]), float(e["a2"][0]), float(e["lam"][0])]
    r["mle_R"] = [float(e["a1r"][0]), float(e["a2r"][0]), float(e["lam"][0])]
    ci = aci(d, e)
    r["sd"] = ci["sd"][0].tolist()
    r["ACI"] = [ci["lo"][0].tolist(), ci["hi"][0].tolist()]
    r["LogACI"] = [ci["llo"][0].tolist(), ci["lhi"][0].tolist()]
    lam0 = e["lam"][0]
    lp_hat = float(profile_ll(d, np.array([lam0]))[0])
    lp_exp = float(-d["N"][0] * np.log((d["tw"][0] * d["tt"][0]).sum()))
    r["lp_gain_over_exponential"] = lp_hat - lp_exp
    # parametric bootstrap-p
    thh = r["mle"]
    th_other = Dr[1 - dnum]
    eo = mle(th_other)
    tho = [float(eo["a1"][0]), float(eo["a2"][0]), float(eo["lam"][0])]
    pars = (thh, tho) if dnum == 0 else (tho, thh)
    mm = (24, 27)
    Db = gen_joint(np.random.default_rng(100 + dnum), 1500, mm[0], mm[1], kR, R, T1, T2, tuple(pars[0]), tuple(pars[1]))
    eb = mle(Db[dnum])
    vb = eb["ok"]
    bt = np.stack([eb["a1"], eb["a2"], eb["lam"]], 1)[vb]
    r["boot_valid"] = float(vb.mean())
    r["BootP"] = [np.percentile(bt, 2.5, 0).tolist(), np.percentile(bt, 97.5, 0).tolist()]
    r["bayes"] = {}
    for pn, pr in priors.items():
        for restricted in (False, True):
            dr, acc = mcmc(d, pr, restricted, rng, 6000, 2000, init=np.array([lam0]))
            be = bayes_est(dr)
            lo, hi = hpd(dr)
            key = f"{pn}|{'R' if restricted else 'U'}"
            r["bayes"][key] = dict(
                acc=float(acc[0]),
                **{kk: v[0].tolist() for kk, v in be.items()},
                HPD=[lo[0].tolist(), hi[0].tolist()],
                P_a1_lt_a2=float((dr[0, :, 0] < dr[0, :, 1]).mean()),
                ess_lam=None,
            )
            chains[(dnum, key)] = dr[0]
    res[f"pop{dnum+1}"] = r

out["real"] = res
json.dump(out, open(os.path.join(_script_dir, "extras_real.json"), "w"), indent=1, default=float)
np.savez(os.path.join(_script_dir, "chains_real.npz"), **{f"{a}_{b}": v for (a, b), v in chains.items()})

# Also sync to parent directory if already present there (e.g. ready_to_submit root)
try:
    parent_dir = os.path.abspath(os.path.join(_script_dir, ".."))
    if os.path.exists(os.path.join(parent_dir, "extras_real.json")):
        json.dump(out, open(os.path.join(parent_dir, "extras_real.json"), "w"), indent=1, default=float)
    if os.path.exists(os.path.join(parent_dir, "chains_real.npz")):
        np.savez(os.path.join(parent_dir, "chains_real.npz"), **{f"{a}_{b}": v for (a, b), v in chains.items()})
except Exception:
    pass

print(json.dumps({k: out[k] for k in ("uniq", "nip", "real_design", "complete_plane7914", "complete_plane7913")}, indent=1, default=float))
for p in ("pop1", "pop2"):
    r = res[p]
    print(p, {k: r[k] for k in ("N", "N1", "N2", "mle", "mle_R", "lp_gain_over_exponential", "boot_valid")})
    print(" WIP-U", r["bayes"]["WIP|U"]["SE"], r["bayes"]["WIP|U"]["HPD"], "P(a1<a2)", r["bayes"]["WIP|U"]["P_a1_lt_a2"])
    print(" WIP-R", r["bayes"]["WIP|R"]["SE"], r["bayes"]["WIP|R"]["HPD"])
print("case", res["case"], "k", res["k"])
