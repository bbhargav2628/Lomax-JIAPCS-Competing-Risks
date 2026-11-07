import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
SIMRES_DIR = os.path.abspath(os.path.join(_script_dir, "simres"))
os.makedirs(SIMRES_DIR, exist_ok=True)

# Ensure local venv site-packages is discoverable if needed
for p in [
    os.path.abspath(os.path.join(_script_dir, "..", "..", ".venv", "Lib", "site-packages")),
    os.path.abspath(os.path.join(_script_dir, "..", ".venv", "Lib", "site-packages")),
]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import time
import numpy as np
import pandas as pd
from lomax_joint import *

REPS = 1000
ITERS, BURN = 2000, 800
KAP = 4.0

TRUTH = {
    "A": ((0.5, 1.0, 1.0), (0.4, 0.8, 1.5)),    # alpha_d1 < alpha_d2 in both populations
    "B": ((0.75, 0.75, 1.0), (0.6, 0.6, 1.5)),  # boundary alpha_d1 = alpha_d2
    "C": ((1.0, 0.5, 1.0), (0.8, 0.4, 1.5)),    # restriction violated
}
DESIGN = {
    "D1": (40, 35, 25, 6),
    "D2": (60, 55, 40, 10),
    "D3": (100, 90, 60, 20),
}  # (m1,m2,k,total removals)
PQ = {"PA": (0.30, 0.60), "PB": (0.55, 0.90)}

def scheme(pattern, k, tot):
    R = np.zeros(k - 1, int)
    if pattern == "early":
        R[0] = tot
    elif pattern == "late":
        R[-1] = tot
    else:
        R[(k - 1) // 2 - 1] = tot // 2
        R[(k - 1) // 2] = tot - tot // 2
    return R

scen = []
for dn in DESIGN:
    for pat in ["early", "mid", "late"]:
        for tp in ["PA", "PB"]:
            scen.append(("A", dn, pat, tp))
for tr in ["B", "C"]:
    for tp in ["PA", "PB"]:
        scen.append((tr, "D2", "mid", tp))

def priors(th):
    a1, a2, lam = th
    return {
        "WIP": dict(a1=1.0, b1=1.0, a2=1.0, b2=1.0, c=1.0, dl=1.0),
        "IP": dict(a1=KAP * a1, b1=KAP, a2=KAP * a2, b2=KAP, c=KAP * lam, dl=KAP),
    }

def run(si, reps=REPS, iters=ITERS, burn=BURN):
    tr, dn, pat, tp = scen[si]
    th = TRUTH[tr]
    m1, m2, k, tot = DESIGN[dn]
    q = lambda t, p: lomax_q(p, t[0] + t[1], t[2])
    T1, T2 = q(th[0], PQ[tp][0]), q(th[0], PQ[tp][1])  # thresholds on the Omega_1 time scale
    rng = np.random.default_rng([si, 20260611])
    R = scheme(pat, k, tot)
    D = gen_joint(rng, reps, m1, m2, k, R, T1, T2, th[0], th[1])
    case = np.bincount(D[0]["case"], minlength=4)[1:] / reps
    key = dict(truth=tr, design=dn, m1=m1, m2=m2, k=k, pattern=pat, thr=tp)
    meta = dict(
        **key,
        T1=T1,
        T2=T2,
        caseI=case[0],
        caseII=case[1],
        caseIII=case[2],
        N_pop1=D[0]["N"].mean(),
        N_pop2=D[1]["N"].mean(),
    )
    pts, ints = [], []
    for d in range(2):
        dd = D[d]
        thd = np.array(th[d])
        names = [f"alpha{d+1}1", f"alpha{d+1}2", f"lambda{d+1}"]
        est = mle(dd)
        valid = est["ok"]
        nk = (dd["N1"] >= 1) & (dd["N2"] >= 1)
        meta[f"P_exist_pop{d+1}"] = valid.mean()
        ci = aci(dd, est)
        P = {
            "MLE": np.stack([est["a1"], est["a2"], est["lam"]], 1),
            "MLE-R": np.stack([est["a1r"], est["a2r"], est["lam"]], 1),
        }
        I = {"ACI": (ci["lo"], ci["hi"]), "LogACI": (ci["llo"], ci["lhi"])}
        dn_ = subset(dd, np.where(nk)[0])
        init = np.where(est["ok"], est["lam"], np.nan)[nk]
        for pname, pr in priors(thd).items():
            for restricted in (False, True):
                tag = "R" if restricted else "U"
                dr, acc = mcmc(dn_, pr, restricted, rng, iters, burn, init=init)
                be = bayes_est(dr)
                lo, hi = hpd(dr)
                for loss, val in be.items():
                    a = np.full((reps, 3), np.nan)
                    a[nk] = val
                    P[f"Bayes[{pname}|{tag}|{loss}]"] = a
                alo = np.full((reps, 3), np.nan)
                ahi = alo.copy()
                alo[nk] = lo
                ahi[nk] = hi
                I[f"HPD[{pname}|{tag}]"] = (alo, ahi)
        for meth, arr in P.items():
            for j in range(3):
                for sname, sel in (("V", valid), ("K", nk)):
                    if meth.startswith("MLE") and sname == "K":
                        continue
                    e = arr[sel, j]
                    if len(e) > 0:
                        pts.append(
                            dict(
                                **key,
                                method=meth,
                                param=names[j],
                                set=sname,
                                AB=abs(e.mean() - thd[j]),
                                MSE=np.mean((e - thd[j]) ** 2),
                                MedBias=np.median(e) - thd[j],
                            )
                        )
        for meth, (lo, hi) in I.items():
            for j in range(3):
                l, h = lo[valid, j], hi[valid, j]
                g = np.isfinite(l) & np.isfinite(h)
                if g.sum() > 0:
                    ints.append(
                        dict(
                            **key,
                            method=meth,
                            param=names[j],
                            AW=np.mean((h - l)[g]),
                            MedAW=np.median((h - l)[g]),
                            CP=np.mean(((l <= thd[j]) & (thd[j] <= h))[g]),
                        )
                    )
    os.makedirs(SIMRES_DIR, exist_ok=True)
    pd.DataFrame(pts).to_csv(os.path.join(SIMRES_DIR, f"pt_{si}.csv"), index=False)
    pd.DataFrame(ints).to_csv(os.path.join(SIMRES_DIR, f"int_{si}.csv"), index=False)
    pd.DataFrame([meta]).to_csv(os.path.join(SIMRES_DIR, f"meta_{si}.csv"), index=False)
    return key, meta

if __name__ == "__main__":
    if _script_dir:
        os.chdir(_script_dir)
    os.makedirs(SIMRES_DIR, exist_ok=True)

    # Fast test mode
    if "--fast" in sys.argv:
        print("Running quick test scenario (fast mode: 10 reps)...")
        key, meta = run(0, reps=10, iters=200, burn=50)
        print("Fast simulation completed successfully:", key, meta)
        sys.exit(0)

    # Specific scenario execution
    single_si = None
    if "--scenario" in sys.argv:
        idx = sys.argv.index("--scenario")
        if idx + 1 < len(sys.argv):
            single_si = int(sys.argv[idx + 1])

    force_rerun = "--force" in sys.argv

    if single_si is not None:
        print(f"Running single scenario {single_si+1}/{len(scen)}: {scen[single_si]}...")
        key, meta = run(single_si)
        print("Completed scenario:", key, meta)
        sys.exit(0)

    # Budget in seconds (defaults to 3600.0 if not provided)
    budget = 3600.0
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            continue
        try:
            budget = float(arg)
            break
        except ValueError:
            pass

    t0 = time.time()
    print(f"Starting simulation with budget = {budget:.1f}s across {len(scen)} scenarios...")
    for si in range(len(scen)):
        meta_file = os.path.join(SIMRES_DIR, f"meta_{si}.csv")
        if not force_rerun and os.path.exists(meta_file):
            continue
        if time.time() - t0 > budget:
            print(f"Time budget ({budget:.0f}s) reached. Stopping.")
            break
        key, meta = run(si)
        print(
            f"[{si+1}/{len(scen)}] {key} exist=({meta['P_exist_pop1']:.2f},{meta['P_exist_pop2']:.2f}) "
            f"cases=({meta['caseI']:.2f},{meta['caseII']:.2f},{meta['caseIII']:.2f}) t={time.time()-t0:.0f}s",
            flush=True,
        )

    all_done = all(os.path.exists(os.path.join(SIMRES_DIR, f"meta_{i}.csv")) for i in range(len(scen)))
    if all_done:
        print("All 22 scenarios completed! Assembling aggregate simulation files...")
        all_pts = [pd.read_csv(os.path.join(SIMRES_DIR, f"pt_{i}.csv")) for i in range(len(scen))]
        all_ints = [pd.read_csv(os.path.join(SIMRES_DIR, f"int_{i}.csv")) for i in range(len(scen))]
        all_metas = [pd.read_csv(os.path.join(SIMRES_DIR, f"meta_{i}.csv")) for i in range(len(scen))]
        pd.concat(all_pts, ignore_index=True).to_csv(os.path.join(_script_dir, "sim_point_all.csv"), index=False)
        pd.concat(all_ints, ignore_index=True).to_csv(os.path.join(_script_dir, "sim_interval_all.csv"), index=False)
        pd.concat(all_metas, ignore_index=True).to_csv(os.path.join(_script_dir, "sim_meta_all.csv"), index=False)
        print("Aggregate files verified: sim_point_all.csv, sim_interval_all.csv, sim_meta_all.csv")
    else:
        print("done" if all_done else "more to do")
