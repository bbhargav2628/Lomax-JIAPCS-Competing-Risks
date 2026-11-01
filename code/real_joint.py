import numpy as np
from lomax_joint import *
from realdata import plane

def apply_joint(rng, t1, c1, t2, c2, k, R, T1, T2):
    """Apply a JIAPCS plan to two complete samples (sorted times t_d, causes c_d). Returns [data1, data2]."""
    ts = (np.asarray(t1, float), np.asarray(t2, float)); cs = (np.asarray(c1), np.asarray(c2))
    alive = [list(range(len(ts[0]))), list(range(len(ts[1])))]
    steps = []; fails = [[], []]; cnt = 0; tend = None; stop = None
    while cnt < k:
        nx = [ts[d][alive[d][0]] if alive[d] else np.inf for d in range(2)]
        t = min(nx)
        if t >= T2: stop = 3; tend = T2; break
        z = 0 if nx[0] <= nx[1] else 1
        u = alive[z].pop(0); cnt += 1; fails[z].append((t, cs[z][u]))
        if cnt == k:
            steps.append((t, z, 0, 0)); tend = t; stop = 1 if t < T1 else 2; break
        Ri = int(R[cnt - 1]) if t < T1 else 0
        c = [Ri + 1, Ri + 1]; c[z] = Ri
        for d in range(2):
            cd = min(c[d], len(alive[d]))
            if cd > 0:
                rem = set(rng.choice(len(alive[d]), cd, replace=False).tolist())
                alive[d] = [a for i, a in enumerate(alive[d]) if i not in rem]
            c[d] = cd
        steps.append((t, z, c[0], c[1]))
    out = []
    for d in range(2):
        tt = [s[0] for s in steps]; tw = [(1 if s[1] == d else 0) + (s[2] if d == 0 else s[3]) for s in steps]
        tt.append(tend); tw.append(len(alive[d]))
        x = [f[0] for f in fails[d]]; dl = [f[1] for f in fails[d]]
        K = len(tt); Nn = len(x)
        out.append(dict(x=np.array([x + [0.0] * (K - Nn)]), mask=np.array([[True] * Nn + [False] * (K - Nn)]),
                        delta=np.array([dl + [0] * (K - Nn)]), tt=np.array([tt]), tw=np.array([tw], float),
                        N=np.array([Nn]), N1=np.array([sum(1 for v in dl if v == 1)]), N2=np.array([sum(1 for v in dl if v == 2)]),
                        case=np.array([stop]), kfail=np.array([cnt])))
    return out
if __name__ == "__main__":
    t1, c1 = plane(1); t2, c2 = plane(0)       # Omega1 = plane 7914 (n=24), Omega2 = plane 7913 (n=27)
    for k in (12, 14, 16, 18):
        for pat in ("late", "early", "mid"):
            R = np.zeros(k - 1, int)
            nrem = min(23 - (k - 1), 6) if 23 - (k - 1) > 0 else 0
            nrem = 6 if k <= 14 else 3
            if pat == "late": R[-1] = nrem
            elif pat == "early": R[0] = nrem
            else: R[(k - 1) // 2] = nrem
            ok = []
            for seed in range(20):
                D = apply_joint(np.random.default_rng(seed), t1, c1, t2, c2, k, R, 0.5, 1.8)
                ok.append([bool(mle(D[0])["ok"][0]), bool(mle(D[1])["ok"][0])])
            ok = np.array(ok).mean(0)
            print("k",k,pat,"R sum",R.sum(),"MLE exists prob (pop1,pop2) over random withdrawals:",ok, "N",D[0]["N"],D[1]["N"],"case",D[0]["case"])
