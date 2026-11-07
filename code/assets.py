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
import matplotlib.pyplot as plt
from lomax_joint import *
from realdata import plane

# Ensure output directory exists
tex_dir = os.path.join(_script_dir, "tex")
os.makedirs(tex_dir, exist_ok=True)

pt = pd.read_csv(os.path.join(_script_dir, "sim_point_all.csv"))
it = pd.read_csv(os.path.join(_script_dir, "sim_interval_all.csv"))
me = pd.read_csv(os.path.join(_script_dir, "sim_meta_all.csv"))
ex = json.load(open(os.path.join(_script_dir, "extras_real.json")))
ap = json.load(open(os.path.join(_script_dir, "appliance.json")))

PN = ["alpha11", "alpha12", "lambda1", "alpha21", "alpha22", "lambda2"]
TEX = {
    "alpha11": r"$\alpha_{11}$",
    "alpha12": r"$\alpha_{12}$",
    "lambda1": r"$\lambda_1$",
    "alpha21": r"$\alpha_{21}$",
    "alpha22": r"$\alpha_{22}$",
    "lambda2": r"$\lambda_2$",
}
TRUE = {"alpha11": 0.5, "alpha12": 1.0, "lambda1": 1.0, "alpha21": 0.4, "alpha22": 0.8, "lambda2": 1.5}
A = pt[pt.truth == "A"]

def f(x, d=3):
    return f"{x:.{d}f}"

def fm(x):  # MSE formatter
    return f"{x:.3f}" if x < 10 else f"{x:.1f}"

# ------------- Table: design / cases / existence
rows = []
for dn in ["D1", "D2", "D3"]:
    s = me[(me.truth == "A") & (me.design == dn)]
    rows.append(
        f"{dn} & ({int(s.m1.iloc[0])},{int(s.m2.iloc[0])}) & {int(s.k.iloc[0])} & {s.N_pop1.mean():.1f} / {s.N_pop2.mean():.1f} & "
        f"{s.caseI.mean():.2f} / {s.caseII.mean():.2f} / {s.caseIII.mean():.2f} & {s.P_exist_pop1.mean():.2f} / {s.P_exist_pop2.mean():.2f} \\\\"
    )
open(os.path.join(tex_dir, "tab_design.tex"), "w").write("\n".join(rows))

# ------------- Table: point estimates, design D2 (averaged over 3 patterns x 2 thresholds)
meths = [
    ("MLE", "V", "MLE"),
    ("Bayes[WIP|U|SE]", "K", "WIP-U"),
    ("Bayes[WIP|R|SE]", "K", "WIP-R"),
    ("Bayes[IP|U|SE]", "K", "IP-U"),
    ("Bayes[IP|R|SE]", "K", "IP-R"),
]

def tab_point(dn, fn):
    rows = []
    for p in PN:
        cells = []
        for mm, st, lab in meths:
            s = A[(A.design == dn) & (A.method == mm) & (A.param == p) & (A.set == st)]
            cells.append(
                f"{s.AB.mean():.3f} & {fm(s.MSE.mean())}"
                if lab != "MLE"
                else f"{s.MedBias.mean():+.2f} & {fm(s.MSE.mean())}"
            )
        rows.append(f"{TEX[p]} ({TRUE[p]}) & " + " & ".join(cells) + r" \\")
    open(fn, "w").write("\n".join(rows))

for dn in ("D1", "D2", "D3"):
    tab_point(dn, os.path.join(tex_dir, f"tab_point_{dn}.tex"))

# ------------- Table: loss functions IP-U and WIP-U on D2
rows = []
for p in PN:
    cells = []
    for pr in ("IP", "WIP"):
        for loss in ("SE", "LINEX+", "LINEX-", "GE+", "GE-"):
            s = A[(A.design == "D2") & (A.method == f"Bayes[{pr}|U|{loss}]") & (A.param == p) & (A.set == "K")]
            cells.append(fm(s.MSE.mean()))
    rows.append(f"{TEX[p]} & " + " & ".join(cells) + r" \\")
open(os.path.join(tex_dir, "tab_loss.tex"), "w").write("\n".join(rows))

# ------------- Table: intervals D2
I = it[(it.truth == "A") & (it.design == "D2")]
im = [
    ("ACI", "ACI"),
    ("LogACI", "LogACI"),
    ("HPD[WIP|U]", "HPD WIP-U"),
    ("HPD[WIP|R]", "HPD WIP-R"),
    ("HPD[IP|U]", "HPD IP-U"),
    ("HPD[IP|R]", "HPD IP-R"),
]
rows = []
for p in PN:
    cells = []
    for mm, lab in im:
        s = I[(I.method == mm) & (I.param == p)]
        cells.append(f"{s.MedAW.mean():.2f} & {s.CP.mean():.2f}")
    rows.append(f"{TEX[p]} & " + " & ".join(cells) + r" \\")
open(os.path.join(tex_dir, "tab_int.tex"), "w").write("\n".join(rows))

# ------------- Table: restricted vs unrestricted across truths (D2, averaged)
rows = []
for tr, lab in (
    ("A", r"$\alpha_{d1}<\alpha_{d2}$ (true)"),
    ("B", r"$\alpha_{d1}=\alpha_{d2}$ (boundary)"),
    ("C", r"$\alpha_{d1}>\alpha_{d2}$ (violated)"),
):
    P = pt[(pt.truth == tr) & (pt.design == "D2") & (pt.set == "K")]
    Iq = it[(it.truth == tr) & (it.design == "D2")]
    cells = []
    for pr in ("IP", "WIP"):
        mu = P[(P.method == f"Bayes[{pr}|U|SE]") & P.param.str.startswith("alpha")].MSE.mean()
        mr = P[(P.method == f"Bayes[{pr}|R|SE]") & P.param.str.startswith("alpha")].MSE.mean()
        wu = Iq[(Iq.method == f"HPD[{pr}|U]") & Iq.param.str.startswith("alpha")]
        wr = Iq[(Iq.method == f"HPD[{pr}|R]") & Iq.param.str.startswith("alpha")]
        cells.append(
            f"{mu:.3f} & {mr:.3f} & {wu.MedAW.mean():.2f} & {wr.MedAW.mean():.2f} & {wu.CP.mean():.2f} & {wr.CP.mean():.2f}"
        )
    rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
open(os.path.join(tex_dir, "tab_restr.tex"), "w").write("\n".join(rows))

# ------------- Figure 1: simulation summary
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
dsn = ["D1", "D2", "D3"]
xx = np.arange(3)
e1 = [me[(me.truth == "A") & (me.design == d)].P_exist_pop1.mean() for d in dsn]
e2 = [me[(me.truth == "A") & (me.design == d)].P_exist_pop2.mean() for d in dsn]
ax[0].bar(xx - 0.17, e1, 0.34, label=r"$\Omega_1$")
ax[0].bar(xx + 0.17, e2, 0.34, label=r"$\Omega_2$")
ax[0].set_xticks(xx)
ax[0].set_xticklabels(["D1\n(40,35,25)", "D2\n(60,55,40)", "D3\n(100,90,60)"])
ax[0].set_ylim(0, 1)
ax[0].set_ylabel("P(finite MLE exists)")
ax[0].legend()
ax[0].set_title("(a) Existence of the MLE")

cols = ["#999999", "#6baed6", "#2171b5", "#fd8d3c", "#d94801"]
w = 0.16
for i, (mm, st, lab) in enumerate(meths):
    v = [
        max(A[(A.design == "D2") & (A.method == mm) & (A.param == p) & (A.set == st)].MSE.mean(), 1e-3)
        for p in PN
    ]
    ax[1].bar(np.arange(6) + (i - 2) * w, v, w, label=lab, color=cols[i])
ax[1].set_yscale("log")
ax[1].set_xticks(range(6))
ax[1].set_xticklabels([TEX[p] for p in PN])
ax[1].set_ylabel("MSE (log scale)")
ax[1].legend(ncol=2, fontsize=8)
ax[1].set_title("(b) MSE, design D2")

labs = []
cps = []
for mm, lab in im:
    labs.append(lab)
    cps.append(I[I.method == mm].CP.mean())
ax[2].barh(labs, cps, color="#2171b5")
ax[2].axvline(0.95, color="k", ls="--")
ax[2].set_xlim(0.6, 1.02)
ax[2].set_xlabel("coverage probability")
ax[2].set_title("(c) 95% interval coverage, D2")
plt.tight_layout()

# Save Figure 1 to tex/, parent, and current directories
plt.savefig(os.path.join(tex_dir, "fig_sim.png"), dpi=170)
plt.savefig(os.path.join(_script_dir, "..", "fig1.png"), dpi=170)
plt.savefig(os.path.join(_script_dir, "fig1.png"), dpi=170)
plt.show()
plt.close()

# ------------- Figure 2: aircraft real data
t1, c1 = plane(1)
t2, c2 = plane(0)

# Check for chains_real.npz
chains_real_file = os.path.join(_script_dir, "chains_real.npz")
if not os.path.exists(chains_real_file) and os.path.exists(os.path.join(_script_dir, "..", "chains_real.npz")):
    import shutil
    shutil.copy2(os.path.join(_script_dir, "..", "chains_real.npz"), chains_real_file)

ch = np.load(chains_real_file)
fig, ax = plt.subplots(2, 2, figsize=(10, 7))
from real_joint import apply_joint

R = np.zeros(17, int)
R[7] = 3
R[8] = 3
Dr = apply_joint(np.random.default_rng(ex["real_design"]["seed_used"]), t1, c1, t2, c2, 18, R, 0.45, 1.8)
for dnum, col in ((0, "C0"), (1, "C1")):
    d = Dr[dnum]
    lam = np.exp(np.linspace(-6, 4, 400))
    lp = profile_ll(
        dict(
            N=np.repeat(d["N"], 400),
            x=np.repeat(d["x"], 400, 0),
            mask=np.repeat(d["mask"], 400, 0),
            tt=np.repeat(d["tt"], 400, 0),
            tw=np.repeat(d["tw"], 400, 0),
        ),
        lam,
    )
    lexp = -d["N"][0] * np.log((d["tw"][0] * d["tt"][0]).sum())
    ax[0, 0].plot(lam, lp - lexp, col, label=rf"$\Omega_{dnum+1}$")

ax[0, 0].axhline(0, color="k", ls="--", lw=1)
ax[0, 0].set_xscale("log")
ax[0, 0].set_xlabel(r"$\lambda$")
ax[0, 0].set_ylabel("profile log-lik. minus exponential limit")
ax[0, 0].legend()
ax[0, 0].set_title("(a) Profile likelihood (aircraft, one withdrawal pattern)")

tg = np.linspace(0, 2.2, 200)
for t, col, lab, dn in ((t1, "C0", r"plane 7914 ($\Omega_1$)", 0), (t2, "C1", r"plane 7913 ($\Omega_2$)", 1)):
    ax[0, 1].step(np.r_[0, t], np.r_[1, 1 - np.arange(1, len(t) + 1) / len(t)], where="post", color=col, label=lab + " empirical")
    b = ex["real"][f"pop{dn+1}"]["bayes"]["WIP|U"]["SE"]
    ax[0, 1].plot(tg, (1 + b[2] * tg) ** (-(b[0] + b[1])), col, ls="--")

ax[0, 1].set_xlabel("hours")
ax[0, 1].set_ylabel("survival")
ax[0, 1].legend(fontsize=8)
ax[0, 1].set_title("(b) Complete-data survival vs fitted Lomax (dashed)")

for key, col in (("WIP|U", "C0"), ("WIP|R", "C3")):
    v = ch[f"1_{key}"]
    ax[1, 0].hist(v[:, 0] - v[:, 1], bins=60, density=True, alpha=0.5, color=col, label=key.replace("|U", " unrestricted").replace("|R", " restricted"))

ax[1, 0].set_xlim(-4, 4)
ax[1, 0].axvline(0, color="k", lw=1)
ax[1, 0].set_xlabel(r"$\alpha_{21}-\alpha_{22}$")
ax[1, 0].legend(fontsize=8)
ax[1, 0].set_title(r"(c) Posterior of $\alpha_{21}-\alpha_{22}$ ($\Omega_2$)")

for dnum, col in ((0, "C0"), (1, "C1")):
    ax[1, 1].plot(ch[f"{dnum}_WIP|U"][:3000, 2], color=col, lw=0.5, label=rf"$\lambda_{dnum+1}$")

ax[1, 1].legend()
ax[1, 1].set_xlabel("iteration")
ax[1, 1].set_title("(d) MCMC trace of the scale parameters")
plt.tight_layout()

# Save Figure 2 to tex/, parent, and current directories
plt.savefig(os.path.join(tex_dir, "fig_air.png"), dpi=170)
plt.savefig(os.path.join(_script_dir, "..", "fig2.png"), dpi=170)
plt.savefig(os.path.join(_script_dir, "fig2.png"), dpi=170)
plt.show()
plt.close()

# ------------- Figure 3: appliance data
chains_app_file = os.path.join(_script_dir, "chains_app.npz")
if not os.path.exists(chains_app_file) and os.path.exists(os.path.join(_script_dir, "..", "chains_app.npz")):
    import shutil
    shutil.copy2(os.path.join(_script_dir, "..", "chains_app.npz"), chains_app_file)

cha = np.load(chains_app_file)
xa = np.array([11, 35, 49, 170, 329, 381, 708, 958, 1062, 1167, 1594, 1925, 1990, 2223, 2327, 2400.0]) / 1000
n, m = 36, 16
fig, ax = plt.subplots(2, 2, figsize=(10, 7))
lam = np.exp(np.linspace(-8, 3, 400))
w = np.ones(m)
w[-1] = 1 + n - m
dd = dict(
    N=np.full(400, m),
    x=np.tile(xa, (400, 1)),
    mask=np.ones((400, m), bool),
    tt=np.tile(xa, (400, 1)),
    tw=np.tile(w, (400, 1)),
)
lp = profile_ll(dd, lam)
lexp = -m * np.log((w * xa).sum())
ax[0, 0].plot(lam, lp - lexp)
ax[0, 0].axhline(0, color="k", ls="--", lw=1)
ax[0, 0].axvline(ap["mle"][2], color="r", ls=":")
ax[0, 0].set_xscale("log")
ax[0, 0].set_xlabel(r"$\lambda$")
ax[0, 0].set_ylabel("profile log-lik. minus exponential limit")
ax[0, 0].set_title("(a) Profile likelihood (appliance data)")

atot = ap["mle"][0] + ap["mle"][1]
lm = ap["mle"][2]
tg = np.linspace(0, 2.5, 300)
risk = n - np.arange(m)
S = np.cumprod(1 - 1 / risk)
ax[0, 1].step(np.r_[0, xa], np.r_[1, S], where="post", color="k", label="Kaplan-Meier")
ax[0, 1].plot(tg, (1 + lm * tg) ** (-atot), label="Lomax (MLE)")
rate = m / ((xa).sum() + (n - m) * xa[-1])
ax[0, 1].plot(tg, np.exp(-rate * tg), label="Exponential")
k_, s_ = ap["weibull"]
ax[0, 1].plot(tg, np.exp(-((tg / s_) ** k_)), label="Weibull")
ax[0, 1].legend(fontsize=8)
ax[0, 1].set_xlabel("time (1000 units)")
ax[0, 1].set_title("(b) Fitted survival functions")

for key, col in (("WIP|U", "C0"), ("WIP|R", "C3")):
    v = cha[key]
    ax[1, 0].hist(
        v[:, 0] - v[:, 1],
        bins=70,
        density=True,
        alpha=0.5,
        color=col,
        label=key.replace("|U", " unrestricted").replace("|R", " restricted"),
    )

ax[1, 0].set_xlim(-3, 3)
ax[1, 0].axvline(0, color="k", lw=1)
ax[1, 0].legend(fontsize=8)
ax[1, 0].set_xlabel(r"$\alpha_1-\alpha_2$")
ax[1, 0].set_title(r"(c) Posterior of $\alpha_1-\alpha_2$")

v = cha["WIP|U"]
for j, (nm, col) in enumerate((("alpha1", "C0"), ("alpha2", "C1"), ("lambda", "C2"))):
    ax[1, 1].plot(v[:3000, j], color=col, lw=0.5, label=nm)

ax[1, 1].legend()
ax[1, 1].set_xlabel("iteration")
ax[1, 1].set_title("(d) MCMC traces (WIP, unrestricted)")
plt.tight_layout()

# Save Figure 3 to tex/, parent, and current directories
plt.savefig(os.path.join(tex_dir, "fig_app.png"), dpi=170)
plt.savefig(os.path.join(_script_dir, "..", "fig3.png"), dpi=170)
plt.savefig(os.path.join(_script_dir, "fig3.png"), dpi=170)
plt.show()
plt.close()

# ------------- tables for real data
r = ex["real"]
for nm, p in (("air", r),):
    rows = []
    for pop in ("pop1", "pop2"):
        q = r[pop]
        names = [r"$\alpha_{%s1}$" % pop[-1], r"$\alpha_{%s2}$" % pop[-1], r"$\lambda_%s$" % pop[-1]]
        for j in range(3):
            b = q["bayes"]["WIP|U"]
            bR = q["bayes"]["WIP|R"]
            rows.append(
                f"{names[j]} & {q['mle'][j]:.3f} & {q['mle_R'][j]:.3f} & [{q['BootP'][0][j]:.2f}, {q['BootP'][1][j]:.2f}] & "
                f"{b['SE'][j]:.3f} & {b['LINEX+'][j]:.3f} & {b['GE+'][j]:.3f} & [{b['HPD'][0][j]:.2f}, {b['HPD'][1][j]:.2f}] & "
                f"{bR['SE'][j]:.3f} & [{bR['HPD'][0][j]:.2f}, {bR['HPD'][1][j]:.2f}] \\\\"
            )
    open(os.path.join(tex_dir, "tab_air.tex"), "w").write("\n".join(rows))

rows = []
names = [r"$\alpha_1$", r"$\alpha_2$", r"$\lambda$"]
for j in range(3):
    b = ap["bayes"]["WIP|U"]
    bR = ap["bayes"]["WIP|R"]
    rows.append(
        f"{names[j]} & {ap['mle'][j]:.3f} & {ap['mle_R'][j]:.3f} & [{ap['BootP'][0][j]:.2f}, {ap['BootP'][1][j]:.2f}] & "
        f"{b['SE'][j]:.3f} & {b['LINEX+'][j]:.3f} & {b['GE+'][j]:.3f} & [{b['HPD'][0][j]:.2f}, {b['HPD'][1][j]:.2f}] & "
        f"{bR['SE'][j]:.3f} & [{bR['HPD'][0][j]:.2f}, {bR['HPD'][1][j]:.2f}] \\\\"
    )
open(os.path.join(tex_dir, "tab_app.tex"), "w").write("\n".join(rows))

print("Assets generated successfully:")
print("  - Tables written to tex/ (tab_design, tab_point_*, tab_loss, tab_int, tab_restr, tab_air, tab_app)")
print("  - Figures generated: fig1.png, fig2.png, fig3.png and tex/fig_*.png")
