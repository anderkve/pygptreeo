"""Replace the q qbar colour factor by the g g colour correlator in the MadLoop g g files.
Usage: fix_gg_cc.py <data file> <cc file>"""
import sys
import numpy as np
fin, fcc = sys.argv[1:3]
d = dict(np.load(fin))
ccr = np.load(fcc)["cc_over_tree"]
assert len(ccr) == len(d["tree"]), (len(ccr), len(d["tree"]))
d["cc"] = d["tree"] * ccr
d["me"] = d["finite"] + np.pi ** 2 / d["beta"] * d["cc"]
np.savez(fin, **d)
r = d["me"] / d["tree"]
print(f"{fin}: a/tree in [{r.min():.3g}, {r.max():.3g}], median {np.median(r):.3g}; cc/tree in [{ccr.min():.3f}, {ccr.max():.3f}]")
