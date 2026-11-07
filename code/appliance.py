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
from scipy import optimize
from lomax_joint import *

# Lawless appliance data as published in Dey & Kayal (2026), Table 14 (Data I): first 16 failures of n=36 units, no
# intermediate withdrawals, remaining 20 units withdrawn at the 16th failure (time/1000).  Original cause labels:
# 1 = failure mode 9, 2 = all other modes.  For the order-restricted analysis we RELABEL: cause 1 = other modes, cause 2 = mode 9.
x = np.array([11, 35, 49, 170, 329, 381, 708, 958, 1062, 1167, 1594, 1925, 1990, 2223, 2327, 2400.0]) / 1000
c_orig = np.array([2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1])
c = 3 - c_orig
n = 36
m = 16

def mk(x, c, w):
    N = len(x)
    return dict(
        x=x[None],
        mask=np.ones((1, N), bool),
        delta=c[None],
        tt=x[None],
        tw=np.array(w, float)[None],
        N=np.array([N]),
        N1=np.array([(c == 1).sum()]),
        N2=np.array([(c == 2).sum()]),
        case=np.array([1]),
        kfail=np.array([N]),
    )

w = np.ones(m)
w[-1] = 1 + (n - m)
d = mk(x, c, w)

out = dict(n=n, m=m, N1=int(d["N1"][0]), N2=int(d["N2"][0]), exist_stat=float(existence_stat(d)[0]))
e = mle(d)
out["mle_exists"] = bool(e["ok"][0])
th = np.array([e["a1"][0], e["a2"][0], e["lam"][0]])
out["mle"] = th.tolist()
out["mle_R"] = [float(e["a1r"][0]), float(e["a2r"][0]), float(e["lam"][0])]
ci = aci(d, e)
out["sd"] = ci["sd"][0].tolist()
out["ACI"] = [ci["lo"][0].tolist(), ci["hi"][0].tolist()]
out["LogACI"] = [ci["llo"][0].tolist(), ci["lhi"][0].tolist()]

# model comparison with the censored likelihood (total lifetime)
a, lam = th[0] + th[1], th[2]
ll_lom = np.sum(np.log(a * lam) - (a + 1) * np.log1p(lam * x)) - (n - m) * a * np.log1p(lam * x[-1])
rate = m / (x.sum() + (n - m) * x[-1])
ll_exp = m * np.log(rate) - rate * (x.sum() + (n - m) * x[-1])

def nllw(p):
    k_, s_ = np.exp(p)
    z = (x / s_) ** k_
    return -(np.sum(np.log(k_ / s_) + (k_ - 1) * np.log(x / s_)) - z.sum() - (n - m) * z[-1])

rw = optimize.minimize(nllw, [0.0, 0.0], method="Nelder-Mead")
ll_w = -rw.fun
out["loglik"] = dict(Lomax=float(ll_lom), Exponential=float(ll_exp), Weibull=float(ll_w))
out["AIC"] = dict(Lomax=float(4 - 2 * ll_lom), Exponential=float(2 - 2 * ll_exp), Weibull=float(4 - 2 * ll_w))
out["weibull"] = np.exp(rw.x).tolist()  # shape, scale

# bootstrap-p (parametric, Type-II censoring with the same n, m)
rng = np.random.default_rng(7)
B = 3000
u = rng.random((B, n))
X = np.sort(lomax_q(u, a, lam), axis=1)[:, :m]
dl = np.where(rng.random((B, m)) < th[0] / (th[0] + th[1]), 1, 2)
wb = np.ones((B, m))
wb[:, -1] = 1 + n - m
db = dict(
    x=X,
    mask=np.ones((B, m), bool),
    delta=dl,
    tt=X,
    tw=wb,
    N=np.full(B, m),
    N1=(dl == 1).sum(1),
    N2=(dl == 2).sum(1),
)
eb = mle(db)
vb = eb["ok"]
bt = np.stack([eb["a1"], eb["a2"], eb["lam"]], 1)[vb]
out["boot_valid"] = float(vb.mean())
out["BootP"] = [np.percentile(bt, 2.5, 0).tolist(), np.percentile(bt, 97.5, 0).tolist()]

# Bayes
pri = {
    "WIP": dict(a1=1.0, b1=1.0, a2=1.0, b2=1.0, c=1.0, dl=1.0),
    "Gamma(2,2)": dict(a1=2.0, b1=2.0, a2=2.0, b2=2.0, c=2.0, dl=2.0),
    "NIP": dict(a1=1e-3, b1=1e-3, a2=1e-3, b2=1e-3, c=1e-3, dl=1e-3),
}
out["bayes"] = {}
ch = {}
for pn, pr in pri.items():
    for restricted in (False, True):
        dr, acc = mcmc(d, pr, restricted, rng, 8000, 2000, init=np.array([th[2]]))
        be = bayes_est(dr)
        lo, hi = hpd(dr)
        key = f"{pn}|{'R' if restricted else 'U'}"
        out["bayes"][key] = dict(
            acc=float(acc[0]),
            **{k: v[0].tolist() for k, v in be.items()},
            HPD=[lo[0].tolist(), hi[0].tolist()],
            P_a1_lt_a2=float((dr[0, :, 0] < dr[0, :, 1]).mean()),
            median=np.median(dr[0], 0).tolist(),
        )
        ch[key] = dr[0]

# Save outputs
np.savez(os.path.join(_script_dir, "chains_app.npz"), **ch)
json.dump(out, open(os.path.join(_script_dir, "appliance.json"), "w"), indent=1, default=float)

# Also sync to parent directory if already present there (e.g. ready_to_submit root)
try:
    parent_dir = os.path.abspath(os.path.join(_script_dir, ".."))
    if os.path.exists(os.path.join(parent_dir, "chains_app.npz")):
        np.savez(os.path.join(parent_dir, "chains_app.npz"), **ch)
    if os.path.exists(os.path.join(parent_dir, "appliance.json")):
        json.dump(out, open(os.path.join(parent_dir, "appliance.json"), "w"), indent=1, default=float)
except Exception:
    pass

print(json.dumps({k: out[k] for k in ("N1", "N2", "exist_stat", "mle", "mle_R", "sd", "ACI", "LogACI", "BootP", "boot_valid", "loglik", "AIC", "weibull")}, indent=1, default=float))
for k in ("WIP|U", "WIP|R", "Gamma(2,2)|U", "NIP|U"):
    b = out["bayes"][k]
    print(k, [round(v, 3) for v in b["SE"]], [[round(v, 3) for v in h] for h in b["HPD"]], "P", round(b["P_a1_lt_a2"], 3), "acc", round(b["acc"], 2))
