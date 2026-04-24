"""Post-process all pygptreeo* npz files for coverage_1sigma[-1]
in [0.60, 0.76]. Emits a markdown report broken down by schedule:
the band only has prescriptive meaning when training and test
distributions are aligned (iid, lhs); under shift / mcmc / de the
test-set coverage *should* drift because the emulator never sees the
test region uniformly.
"""
import glob, json, os, sys
import numpy as np

ROOT = "/home/user/pygptreeo/benchmarks/iterations"
LO, HI = 0.60, 0.76

rows = []
for it_dir in sorted(glob.glob(os.path.join(ROOT, "iteration_*/data"))):
    it = os.path.basename(os.path.dirname(it_dir))
    for f in sorted(glob.glob(os.path.join(it_dir, "*pygptreeo*.npz"))):
        try:
            d = np.load(f, allow_pickle=True)
        except Exception as e:
            continue
        try:
            cfg = json.loads(str(d["config_json"]))
        except Exception:
            cfg = {}
        method = cfg.get("method_name", "")
        problem = cfg.get("problem_name", "")
        seed = cfg.get("seed", -1)
        kind = cfg.get("kind", None)
        if kind in ("assisted", "delayed"):
            schedule = f"{kind}_τ{cfg.get('tau_sigma', '?')}" if kind == "assisted" else "delayed"
        else:
            schedule = cfg.get("schedule", "iid")
        if "coverage_1sigma" not in d.files:
            continue
        arr = d["coverage_1sigma"]
        if len(arr) == 0:
            continue
        cov = float(arr[-1])
        rows.append((it, method, problem, schedule, seed, cov))


def in_band(c):
    return LO <= c <= HI


# Two strata: aligned (iid, lhs) vs shifted (shift, de, mcmc, assisted, delayed).
ALIGNED = {"iid", "lhs"}
def stratum(s):
    return "aligned" if s in ALIGNED else "non-aligned"

aligned = [r for r in rows if stratum(r[3]) == "aligned"]
nonaligned = [r for r in rows if stratum(r[3]) == "non-aligned"]
aligned_in = [r for r in aligned if in_band(r[5])]
nonaligned_in = [r for r in nonaligned if in_band(r[5])]

out = []
out.append("# coverage_1sigma drift across all pygptreeo* runs")
out.append("")
out.append(f"Definition: a run is **in-band** iff `coverage_1sigma[-1] ∈ [{LO}, {HI}]`. The nominal value is 0.6827.")
out.append("")
out.append("The band only has prescriptive meaning when training and test distributions are aligned. Under `shift`, `mcmc`, `de`, and emulator-assisted MCMC, the test set is uniform-iid but the training stream is not, so the emulator's coverage on the test set is *expected* to drift outside this band; that is the substantive content of the iter-11/13/14 chapters, not a methods bug.")
out.append("")
out.append("## Headline")
out.append("")
out.append(f"- Aligned (iid, lhs): **{len(aligned_in)} / {len(aligned)} in-band ({100.0*len(aligned_in)/max(1,len(aligned)):.1f} %)**")
out.append(f"- Non-aligned (shift, de, mcmc, assisted, delayed): {len(nonaligned_in)} / {len(nonaligned)} in-band ({100.0*len(nonaligned_in)/max(1,len(nonaligned)):.1f} %)")
out.append(f"- Combined: {len(aligned_in) + len(nonaligned_in)} / {len(rows)} in-band ({100.0*(len(aligned_in)+len(nonaligned_in))/max(1,len(rows)):.1f} %)")
out.append("")
aligned_out = [r for r in aligned if not in_band(r[5])]
if aligned_out:
    out.append(f"## Aligned-stratum out-of-band cells ({len(aligned_out)})")
    out.append("These would be a regression — investigate.")
    out.append("")
    out.append("| iteration | method | problem | schedule | seed | coverage_1sigma[-1] |")
    out.append("|---|---|---|---|---|---|")
    for r in sorted(aligned_out, key=lambda r: (r[0], r[1], r[2], r[4])):
        out.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]:.3f} |")
else:
    out.append("## Aligned-stratum out-of-band cells: **0** — no aligned-stream regression.")
out.append("")
out.append(f"## Non-aligned out-of-band cells: {len(nonaligned) - len(nonaligned_in)} (expected; not flagged)")

print("\n".join(out))

