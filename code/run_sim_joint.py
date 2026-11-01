import numpy as np, pandas as pd, time, sys, os
from lomax_joint import *

REPS = 1000; ITERS, BURN = 2000, 800; KAP = 4.0
os.makedirs("simres", exist_ok=True)
TRUTH = {"A": ((0.5, 1.0, 1.0), (0.4, 0.8, 1.5)),      # alpha_d1 < alpha_d2 in both populations
         "B": ((0.75, 0.75, 1.0), (0.6, 0.6, 1.5)),    # boundary alpha_d1 = alpha_d2
         "C": ((1.0, 0.5, 1.0), (0.8, 0.4, 1.5))}      # restriction violated
DESIGN = {"D1": (40, 35, 25, 6), "D2": (60, 55, 40, 10), "D3": (100, 90, 60, 20)}   # (m1,m2,k,total removals)
PQ = {"PA": (0.30, 0.60), "PB": (0.55, 0.90)}

def scheme(pattern, k, tot):
    R = np.zeros(k - 1, int)
    if pattern == "early": R[0] = tot
    elif pattern == "late": R[-1] = tot
    else: R[(k - 1) // 2 - 1] = tot // 2; R[(k - 1) // 2] = tot - tot // 2
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
    return {"WIP": dict(a1=1., b1=1., a2=1., b2=1., c=1., dl=1.),
            "IP": dict(a1=KAP * a1, b1=KAP, a2=KAP * a2, b2=KAP, c=KAP * lam, dl=KAP)}

def run(si):
    tr, dn, pat, tp = scen[si]
    th = TRUTH[tr]; m1, m2, k, tot = DESIGN[dn]
    q = lambda t, p: lomax_q(p, t[0] + t[1], t[2])
    T1, T2 = q(th[0], PQ[tp][0]), q(th[0], PQ[tp][1])      # thresholds on the Omega_1 time scale
    rng = np.random.default_rng([si, 20260611])
    R = scheme(pat, k, tot)
    D = gen_joint(rng, REPS, m1, m2, k, R, T1, T2, th[0], th[1])
    case = np.bincount(D[0]["case"], minlength=4)[1:] / REPS
    key = dict(truth=tr, design=dn, m1=m1, m2=m2, k=k, pattern=pat, thr=tp)
    meta = dict(**key, T1=T1, T2=T2, caseI=case[0], caseII=case[1], caseIII=case[2],
                N_pop1=D[0]["N"].mean(), N_pop2=D[1]["N"].mean())
    pts, ints = [], []
    for d in range(2):
        dd = D[d]; thd = np.array(th[d]); names = [f"alpha{d+1}1", f"alpha{d+1}2", f"lambda{d+1}"]
        est = mle(dd); valid = est["ok"]; nk = (dd["N1"] >= 1) & (dd["N2"] >= 1)
        meta[f"P_exist_pop{d+1}"] = valid.mean()
        ci = aci(dd, est)
        P = {"MLE": np.stack([est["a1"], est["a2"], est["lam"]], 1), "MLE-R": np.stack([est["a1r"], est["a2r"], est["lam"]], 1)}
        I = {"ACI": (ci["lo"], ci["hi"]), "LogACI": (ci["llo"], ci["lhi"])}
        dn_ = subset(dd, np.where(nk)[0]); init = np.where(est["ok"], est["lam"], np.nan)[nk]
        for pname, pr in priors(thd).items():
            for restricted in (False, True):
                tag = "R" if restricted else "U"
                dr, acc = mcmc(dn_, pr, restricted, rng, ITERS, BURN, init=init)
                be = bayes_est(dr); lo, hi = hpd(dr)
                for loss, val in be.items():
                    a = np.full((REPS, 3), np.nan); a[nk] = val; P[f"Bayes[{pname}|{tag}|{loss}]"] = a
                alo = np.full((REPS, 3), np.nan); ahi = alo.copy(); alo[nk] = lo; ahi[nk] = hi
                I[f"HPD[{pname}|{tag}]"] = (alo, ahi)
        for meth, arr in P.items():
            for j in range(3):
                for sname, sel in (("V", valid), ("K", nk)):
                    if meth.startswith("MLE") and sname == "K": continue
                    e = arr[sel, j]
                    pts.append(dict(**key, method=meth, param=names[j], set=sname, AB=abs(e.mean() - thd[j]),
                                    MSE=np.mean((e - thd[j]) ** 2), MedBias=np.median(e) - thd[j]))
        for meth, (lo, hi) in I.items():
            for j in range(3):
                l, h = lo[valid, j], hi[valid, j]; g = np.isfinite(l) & np.isfinite(h)
                ints.append(dict(**key, method=meth, param=names[j], AW=np.mean((h - l)[g]), MedAW=np.median((h - l)[g]),
                                 CP=np.mean(((l <= thd[j]) & (thd[j] <= h))[g])))
    pd.DataFrame(pts).to_csv(f"simres/pt_{si}.csv", index=False)
    pd.DataFrame(ints).to_csv(f"simres/int_{si}.csv", index=False)
    pd.DataFrame([meta]).to_csv(f"simres/meta_{si}.csv", index=False)
    return key, meta

if __name__ == "__main__":
    budget = float(sys.argv[1]); t0 = time.time()
    for si in range(len(scen)):
        if os.path.exists(f"simres/meta_{si}.csv"): continue
        if time.time() - t0 > budget: break
        key, meta = run(si)
        print(f"[{si+1}/{len(scen)}] {key} exist=({meta['P_exist_pop1']:.2f},{meta['P_exist_pop2']:.2f}) cases=({meta['caseI']:.2f},{meta['caseII']:.2f},{meta['caseIII']:.2f}) t={time.time()-t0:.0f}s", flush=True)
    print("done" if all(os.path.exists(f"simres/meta_{i}.csv") for i in range(len(scen))) else "more to do")
