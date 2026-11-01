import numpy as np
# Air-conditioning failure data (hours) of Boeing 720 planes 7914 (group 1) and 7913 (group 0),
# as tabulated in Chen & Gui (2026, Table 16) after Proschan (1963) / Mondal & Kundu (2020).
# Table order is (lifetime, cause, plane):  cause 1 = compressor failure, cause 2 = all other causes.
raw = """(0.03,2,1) (0.05,1,1) (0.05,2,1) (0.13,1,1) (0.14,1,1) (0.15,2,1)
(0.22,1,1) (0.22,1,1) (0.23,1,1) (0.30,2,1) (0.36,1,1) (0.39,2,1)
(0.44,1,1) (0.46,1,1) (0.50,2,1) (0.72,1,1) (0.79,2,1) (0.88,2,1)
(0.97,2,1) (1.02,1,1) (1.39,1,1) (1.88,1,1) (1.97,1,1) (2.10,1,1)
(0.01,1,0) (0.04,1,0) (0.11,1,0) (0.16,1,0) (0.18,1,0) (0.18,2,0)
(0.18,2,0) (0.24,1,0) (0.31,1,0) (0.39,1,0) (0.46,2,0) (0.51,2,0)
(0.54,1,0) (0.63,2,0) (0.68,2,0) (0.77,2,0) (0.80,2,0) (0.82,2,0)
(0.97,2,0) (1.06,2,0) (1.11,2,0) (1.41,2,0) (1.42,2,0) (1.63,2,0)
(1.91,2,0) (2.06,1,0) (2.16,2,0)"""
import re
tri = [tuple(float(v) for v in t.split(",")) for t in re.findall(r"\(([^)]*)\)", raw)]
arr = np.array(tri)
def plane(g):
    a = arr[arr[:, 2] == g]; o = np.argsort(a[:, 0], kind="stable")
    return a[o, 0], a[o, 1].astype(int)
if __name__ == "__main__":
    for g in (1, 0):
        t, c = plane(g); print("plane", g, "n=", len(t), "cause1:", (c == 1).sum(), "cause2:", (c == 2).sum(), "mean", t.mean().round(3))
