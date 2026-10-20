"""pyGPTreeO on 5D tree-level ttH amplitudes: the test functions f1 and f3 of
"Interpolating amplitudes" (Bresó-Pla, Heinrich, Magerya, Olsson, arXiv:2412.09534).

The amplitudes are |M|^2 for q qbar -> t tbar H (f1) and g g -> t tbar H (f3) on
the paper's five phase-space variables (beta^2, f_ttbar, theta_H, theta_t, phi_t),
folded by the amplitude's symmetries onto the unit cube (``--domain full`` for the
full polar-angle domain the paper's networks use), computed with MadGraph
(examples/tth_amplitude/README.md has the recipe). The paper's error is the
relative L1 distance eps = sum |f~ - f| / sum f of the weighted amplitude
f = w |M|^2 over uniform test points, where w is the phase-space measure times
the paper's flux and pseudo-PDF factor rho(beta^2) (``--no-rho`` drops the latter),
and on LO-unweighted points the mean relative error.

The plain tree and the hybrid (GP leaves on the residual of a tree-wide feature
network) are streamed over Sobol points, evaluated at checkpoints. Optionally the
training values carry a relative Gaussian noise (the NoisyEvaluator of the
paper's code uses 3e-3), passed to the tree as the per-point sigma.

Usage: python tth_amplitude_benchmark.py <data_dir> [--channels gg,uu] [--configs tree,hybrid,acq]
           [--nmax 131072] [--noise 0] [--seed 1]
``acq`` is the plain tree choosing which pool points to learn from (the largest
expected error contribution w |M|^2 sigma) instead of taking them in Sobol order.
``hybacq`` is the same with the hybrid.
``leafacq`` (needs ``--driver``) lets the leaves decide: each leaf's expected error
contribution is summed over a Sobol sample of candidate points routed to it, the leaf
with the largest one gets a batch of fresh points, evaluated with the amplitude code.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pygptreeo import GPTree

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results", "tth_amplitude")
COLORS = {"tree": "#2a78d6", "hybrid": "#eb6834", "acq": "#1baf7a", "leafacq": "#8e44ad", "hybacq": "#eda100"}
CHANNEL_NAMES = {"uu": "f1: q qbar -> t tbar H", "gg": "f3: g g -> t tbar H"}


def load(data_dir, ch):
    def arrays(path):
        z = np.load(path)
        d = {k: z[k] for k in z.files}
        ok = np.isfinite(d["me"]) & (d["acc"] < 1e-5 if "acc" in d else True)   # one-loop points below precision
        d = {k: v[ok] for k, v in d.items()}
        if OPTS["ratio"]:        # the one-loop amplitude in units of the Born, the networks' target for f2 and f4
            d["me"] = d["me"] / d["tree"]
        return d
    tr = arrays(os.path.join(data_dir, f"{ch}_train_sobol.npz"))
    unw = "unweighted_rho" if OPTS["rho"] else "unweighted"     # points drawn with probability ~ f
    if OPTS["train"] in ("unweighted", "mixed"):
        tw = arrays(os.path.join(data_dir, f"{ch}_train_{unw}.npz"))
        if OPTS["train"] == "unweighted":
            tr = tw
        else:   # alternate Sobol and f-weighted points
            n = min(len(tr["me"]), len(tw["me"]))
            tr = {k: np.empty((2 * n,) + tr[k].shape[1:]) for k in tr}
            for k in tr:
                tr[k][0::2] = arrays(os.path.join(data_dir, f"{ch}_train_sobol.npz"))[k][:n]; tr[k][1::2] = tw[k][:n]
    te = arrays(os.path.join(data_dir, f"{ch}_test_uniform.npz"))
    path_u = os.path.join(data_dir, f"{ch}_test_{unw}.npz")
    tu = arrays(path_u) if os.path.exists(path_u) else None
    for d in (tr, te, tu):
        if d is not None:
            if OPTS["rho"]:
                d["w"] = d["w"] * rho(ch, d["X"])
            d["X"] = tree_inputs(d["X"])
    return tr, te, tu


NET_KWARGS = {}   # overrides for the hybrid's FeatureNetLearner, from --net
OPTS = dict(kernel="matern32", nbar=100, train="sobol", inputs="cube", features="cube", split="min_lengthscale",
            retrain=25, strategy="gradual", target="log", glob="none", rho=True, scale=1.0, theta=None, ratio=False)
GLUON_TOP_PAIRS = [1, 2, 4, 5]   # columns of invariants(): (0,2), (0,3), (1,2), (1,3)


TARGETS = ["log", "pole1", "pole2", "logs1", "logs2", "lin", "lins1", "lins2", "pow0.5", "pow0.25"]


def log_shat(X_tree):
    """log s-hat (m_t = 1) from the tree's inputs; s-hat = 2 p1.p2, and column 0 of
    invariants() is log(p1.p2)."""
    if OPTS["inputs"] == "cube":
        return np.log((2 * MT + MH) ** 2 / (1 - (0.1 + 0.86 * X_tree[:, 0])))
    return np.log(2.0) + X_tree[:, 5 if OPTS["inputs"] == "both" else 0]


def target_shift(X_tree):
    """log of the factor F(x) multiplying |M|^2 before learning: 1 (``log``, ``lin``),
    the gluon-top invariants to the power p (``pole1``, ``pole2``, flattening the
    t-channel top propagators 1/(p_g . p_t)^p), or s-hat^k (``logs1``, ``logs2``,
    ``lins1``, ``lins2``)."""
    t = OPTS["target"]
    if t in ("log", "lin"):
        return np.zeros(len(X_tree))
    if t.startswith("pole"):
        p = {"pole1": 1.0, "pole2": 2.0}[t]
        inv = X_tree[:, :10] if OPTS["inputs"] in ("invariants", "invbeta") else (X_tree[:, 5:] if OPTS["inputs"] == "both" else invariants(X_tree))
        return p * inv[:, GLUON_TOP_PAIRS].sum(axis=1)
    return float(t[-1]) * log_shat(X_tree)


def target_power():
    """The power p of (|M|^2 F)^p the tree learns; None for the log targets."""
    t = OPTS["target"]
    return 1.0 if t.startswith("lin") else (float(t[3:]) if t.startswith("pow") else None)


def linear_target():
    return target_power() is not None


def to_target(a, X_tree):
    """What the tree learns: log(|M|^2 F), or (|M|^2 F)^p / scale."""
    p = target_power()
    if p is not None:
        return (a * np.exp(target_shift(X_tree))) ** p / OPTS["scale"]
    return np.log(a) + target_shift(X_tree)


def from_target(mu, sd, X_tree):
    """The amplitude and its sigma from the tree's prediction (a power target is
    clipped at a tiny positive value before the root)."""
    p = target_power()
    if p == 1.0:
        f = OPTS["scale"] * np.exp(-target_shift(X_tree))
        return mu * f, sd * f
    if p is not None:
        y = np.maximum(mu * OPTS["scale"], 1e-300)
        a = y ** (1.0 / p) * np.exp(-target_shift(X_tree))
        return a, a * sd * OPTS["scale"] / (p * y)
    a = np.exp(mu - target_shift(X_tree))
    return a, a * sd


def set_scale(a, X_tree):
    """Power targets are learned in units of their median over the first 1000 points."""
    p = target_power()
    OPTS["scale"] = float(np.median(np.abs(a[:1000] * np.exp(target_shift(X_tree[:1000]))) ** p)) if p is not None else 1.0
MT, MH = 1.0, np.sqrt(12.0 / 23.0)
RHO_C = {"uu": (1.0132, 0.9943, 0.3506), "gg": (1.0134, 0.7344, 0.0987)}   # eq. (10) of arXiv:2412.09534
THETA_MAX = 0.5 * np.pi   # theta_H and theta_t range: pi/2 on the folded domain, pi on the full one


def rho(ch, X):
    """The paper's flux times pseudo-PDF factor (1/2s) d rho / d beta^2, which multiplies
    the phase-space measure the driver returns; a function of the unit-cube coordinates."""
    c1, c2, c3 = RHO_C[ch]; b2 = 0.1 + 0.86 * X[:, 0]
    return (1 - c1 * b2) ** 2 / ((1 - c2 * b2) * (1 - c3 * b2))


def momenta(X):
    """Four-momenta (n, 5, 4) of g/q, g/qbar, t, tbar, H from the unit-cube variables,
    as tth_driver.cpp builds them (folded domain, m_t = 1)."""
    X = np.atleast_2d(X)
    b2 = 0.1 + 0.86 * X[:, 0]; frac = X[:, 1]; th = THETA_MAX * X[:, 2]; tt = THETA_MAX * X[:, 3]; ph = np.pi * X[:, 4]
    s = (2 * MT + MH) ** 2 / (1 - b2); rs = np.sqrt(s)
    stt = 4 * MT ** 2 + frac * ((rs - MH) ** 2 - 4 * MT ** 2); rstt = np.sqrt(stt)
    pt = np.sqrt(stt - 4 * MT ** 2) / 2
    t1 = np.column_stack([rstt / 2, pt * np.sin(tt) * np.cos(ph), pt * np.sin(tt) * np.sin(ph), pt * np.cos(tt)])
    t2 = np.column_stack([t1[:, 0], -t1[:, 1], -t1[:, 2], -t1[:, 3]])
    kl = (stt - (rs + MH) ** 2) * (stt - (rs - MH) ** 2); qh2 = kl / (4 * s); qh = np.sqrt(qh2)
    g = np.sqrt(1 + qh2 / stt); gb = qh / rstt
    def boost_rot(p):
        E = g * p[:, 0] + gb * p[:, 3]; pz = gb * p[:, 0] + g * p[:, 3]; px, py = p[:, 1], p[:, 2]
        return np.column_stack([E, -np.cos(th) * px - np.sin(th) * pz, py, np.sin(th) * px - np.cos(th) * pz])
    q1 = np.column_stack([rs / 2, 0 * rs, 0 * rs, rs / 2]); q2 = np.column_stack([rs / 2, 0 * rs, 0 * rs, -rs / 2])
    qH = np.column_stack([np.sqrt(qh2 + MH ** 2), qh * np.sin(th), 0 * qh, qh * np.cos(th)])
    return np.stack([q1, q2, boost_rot(t1), boost_rot(t2), qH], axis=1)


def oracle(driver, X, theta_max=None):
    """Phase-space weight and |M|^2 from the MadGraph driver at unit-cube points X (n, 5)."""
    import subprocess
    theta_max = THETA_MAX if theta_max is None else theta_max
    P = np.column_stack([0.1 + 0.86 * X[:, 0], X[:, 1], theta_max * X[:, 2], theta_max * X[:, 3], np.pi * X[:, 4]])
    inp = "\n".join(" ".join(f"{v:.17g}" for v in row) for row in P) + "\n"
    out = subprocess.run([os.path.abspath(driver)], input=inp, capture_output=True, text=True, check=True,
                         cwd=os.path.dirname(os.path.abspath(driver))).stdout   # the driver reads its card from its own directory
    A = np.array([l.split() for l in out.splitlines() if len(l.split()) == 2], dtype=float)
    return A[:, 0], A[:, 1]


def phase_space_weight(X):
    """The driver's phase-space weight |dPhi / d(frac, theta_H, theta_t, phi_t)| at unit-cube
    points, m_t = 1: the s_tt Jacobian times the two two-body factors, in the driver's
    normalisation (agrees with it to 5e-9)."""
    b2 = 0.1 + 0.86 * X[:, 0]; frac = X[:, 1]; th = THETA_MAX * X[:, 2]; tt = THETA_MAX * X[:, 3]
    s = (2 * MT + MH) ** 2 / (1 - b2); rs = np.sqrt(s)
    jf = (rs - MH) ** 2 - 4 * MT ** 2
    stt = 4 * MT ** 2 + frac * jf
    qh = np.sqrt((stt - (rs + MH) ** 2) * (stt - (rs - MH) ** 2)) / (2 * rs)
    pt = np.sqrt(stt / 4 - MT ** 2)
    return 4.01015e-5 * jf * (qh / rs) * np.sin(th) * (pt / np.sqrt(stt)) * np.sin(tt)


def invariants(X):
    """log(p_i . p_j) for the ten pairs of external momenta (the inputs of the
    networks of arXiv:2412.09534)."""
    P = momenta(X)
    metric = np.array([1.0, -1.0, -1.0, -1.0])
    out = []
    for i in range(5):
        for j in range(i + 1, 5):
            out.append(np.log(np.sum(P[:, i] * P[:, j] * metric, axis=1)))
    return np.column_stack(out)


def beta_tt(X):
    """Velocity of the tops in the t tbar rest frame, the threshold variable of the one-loop amplitudes."""
    X = np.atleast_2d(X)
    s = (2 * MT + MH) ** 2 / (1 - (0.1 + 0.86 * X[:, 0]))
    stt = 4 * MT ** 2 + X[:, 1] * ((np.sqrt(s) - MH) ** 2 - 4 * MT ** 2)
    return np.sqrt(1 - 4 * MT ** 2 / stt)


def tree_inputs(X):
    if OPTS["inputs"] == "invariants":
        return invariants(X)
    if OPTS["inputs"] == "invbeta":
        return np.column_stack([invariants(X), beta_tt(X)])
    if OPTS["inputs"] == "both":
        return np.hstack([X, invariants(X)])
    return X


def make(config, seed):
    from pygptreeo import Default_GPR, AdditiveMaternKernel, AnisotropicRationalQuadratic
    from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, Matern, RBF
    d = {"cube": 5, "invariants": 10, "both": 15, "invbeta": 11}[OPTS["inputs"]]
    kernels = {"matern32": None,
               "matern52": ConstantKernel() * Matern(nu=2.5, length_scale=np.ones(d)),
               "rbf": ConstantKernel() * RBF(length_scale=np.ones(d)),
               "rq": ConstantKernel() * AnisotropicRationalQuadratic(length_scale=np.ones(d)),
               "rbflin": ConstantKernel() * RBF(length_scale=np.ones(d)) + ConstantKernel(0.1) * DotProduct(sigma_0=1.0),
               "additive": AdditiveMaternKernel(d=d, order=2, nu=1.5)}
    gpr = Default_GPR(kernel=kernels[OPTS["kernel"]])
    tree_kw = dict(Nbar=OPTS["nbar"], split_dimension_criteria=OPTS["split"], splitting_strategy=OPTS["strategy"],
                   retrain_every_n_points=OPTS["retrain"])
    if OPTS["theta"] is not None:
        tree_kw["theta"] = OPTS["theta"]
    if OPTS["glob"] == "additive":
        tree_kw["global_mean"] = "additive_gp"
    with contextlib.redirect_stdout(io.StringIO()):
        if config == "hybrid":
            kw = dict(steps_per_update=8, random_state=seed); kw.update(NET_KWARGS)
            if OPTS["features"] == "invariants":
                kw["feature_map"] = invariants if OPTS["inputs"] == "cube" else None
            tree_kw.pop("global_mean", None)
            return GPTree(GPR=gpr, global_mean='net', global_mean_kwargs=kw, **tree_kw)
        return GPTree(GPR=gpr, **tree_kw)


def evaluate(gpt, te, tu):
    with contextlib.redirect_stdout(io.StringIO()):
        mu, sd = gpt.predict(te["X"], mode='loop')
    mu, sd = mu.ravel(), sd.ravel()
    a_true = te["me"]; a_pred = from_target(mu, sd, te["X"])[0]
    born = te["tree"] if OPTS["ratio"] else 1.0
    f_true = te["w"] * a_true * born; f_pred = te["w"] * a_pred * born
    rel = np.abs(a_pred / a_true - 1.0)
    pull = (mu - to_target(a_true, te["X"])) / sd
    out = dict(eps_uniform=float(np.sum(np.abs(f_pred - f_true)) / np.sum(np.abs(f_true))),
               median_rel=float(np.median(rel)), within1pct=float(np.mean(rel < 0.01)),
               cov68=float(np.mean(np.abs(pull) < 1)), median_sigma=float(np.median(sd)))
    if tu is not None:
        with contextlib.redirect_stdout(io.StringIO()):
            mu_u, _ = gpt.predict(tu["X"], mode='loop')
        pred_u = from_target(mu_u.ravel(), mu_u.ravel(), tu["X"])[0]
        if "tree" in tu:    # one-loop data, eq. (30) of arXiv:2412.09534: sum |r~ - r| / sum |r| with r = a / a_LO
            born = 1.0 if OPTS["ratio"] else tu["tree"]
            out["eps_unweighted"] = float(np.sum(np.abs(pred_u - tu["me"]) / born) / np.sum(np.abs(tu["me"] / born)))
            out["mean_rel_unweighted"] = float(np.mean(np.abs(pred_u / tu["me"] - 1.0)))
        else:
            out["eps_unweighted"] = float(np.mean(np.abs(pred_u / tu["me"] - 1.0)))
    return out


def run(ch, config, data_dir, nmax, noise, seed, checkpoints):
    tr, te, tu = load(data_dir, ch)
    X = tr["X"][:nmax]; a = tr["me"][:nmax]
    rng = np.random.RandomState(seed)
    set_scale(a, X)
    y = to_target(a, X)
    sig = np.full(len(a), noise if noise > 0 else 1e-6)
    if linear_target():   # relative noise on the amplitude is multiplicative here
        sig = sig * np.abs(y)
        y = y * (1.0 + rng.normal(0.0, noise, len(a))) if noise > 0 else y
    elif noise > 0:
        y = y + rng.normal(0.0, noise, len(a))
    gpt = make(config, seed)
    rows, done, t0 = [], 0, time.time()
    for N in checkpoints:
        if N > nmax or done >= len(X):
            break
        N = min(N, len(X))      # a stream shorter than the checkpoint (points dropped by the accuracy filter)
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(done, N):
                gpt.update_tree(X[i:i + 1], y[i:i + 1].reshape(1, 1), sig[i])
        done = N
        t_train = time.time() - t0
        m = evaluate(gpt, te, tu); m.update(N=N, train_s=t_train, ms_per_point=1e3 * t_train / N, leaves=len(gpt.root.leaves))
        rows.append(m)
        print(f"{ch} {config:6s} N={N:6d}  eps_uniform {m['eps_uniform']:.2e}  eps_unweighted {m.get('eps_unweighted', float('nan')):.2e}  "
              f"median rel {m['median_rel']:.2e}  within 1 %: {100 * m['within1pct']:.0f} %  cov68 {m['cov68']:.2f}  "
              f"{m['ms_per_point']:.1f} ms/pt  leaves {m['leaves']}", flush=True)
    return rows


def run_acquisition(ch, data_dir, nmax, seed, checkpoints, batch=500, candidates=20000, model="tree"):
    """The tree buys, from the Sobol pool, the points with the largest expected error
    contribution w * |M|^2 * sigma; the phase-space weight w is analytic, the amplitude
    the tree's own prediction."""
    tr, te, tu = load(data_dir, ch)
    X = tr["X"]; a = tr["me"]; w = tr["w"]
    set_scale(a, X)
    y = to_target(a, X)
    sig = 1e-6 * (np.abs(y) if linear_target() else np.ones(len(y)))
    rng = np.random.RandomState(seed)
    gpt = make(model, seed)
    bought = list(range(checkpoints[0]))
    pool = np.arange(checkpoints[0], len(a))
    with contextlib.redirect_stdout(io.StringIO()):
        for i in bought:
            gpt.update_tree(X[i:i + 1], y[i:i + 1].reshape(1, 1), sig[i])
    rows, t0 = [], time.time()
    for N in checkpoints:
        if N > nmax:
            break
        while len(bought) < N and len(pool):
            cand = rng.choice(pool, size=min(candidates, len(pool)), replace=False)
            with contextlib.redirect_stdout(io.StringIO()):
                mu, sd = gpt.predict(X[cand], mode='loop')
            a_pred, sig_a = from_target(mu.ravel(), sd.ravel(), X[cand])
            score = w[cand] * (tr["tree"][cand] if OPTS["ratio"] else np.abs(a_pred)) * sig_a
            take = cand[np.argsort(score)[::-1][:min(batch, N - len(bought))]]
            with contextlib.redirect_stdout(io.StringIO()):
                for i in take:
                    gpt.update_tree(X[i:i + 1], y[i:i + 1].reshape(1, 1), sig[i])
            bought += list(take)
            pool = np.setdiff1d(pool, take, assume_unique=True)
        m = evaluate(gpt, te, tu); t_train = time.time() - t0
        m.update(N=N, train_s=t_train, ms_per_point=1e3 * t_train / N, leaves=len(gpt.root.leaves))
        rows.append(m)
        print(f"{ch} {'acq' if model == 'tree' else 'hybacq':6s} N={N:6d}  eps_uniform {m['eps_uniform']:.2e}  eps_unweighted {m.get('eps_unweighted', float('nan')):.2e}  "
              f"median rel {m['median_rel']:.2e}  within 1 %: {100 * m['within1pct']:.0f} %  cov68 {m['cov68']:.2f}  "
              f"{m['ms_per_point']:.1f} ms/pt  leaves {m['leaves']}", flush=True)
    return rows


def run_leafacq(ch, data_dir, nmax, seed, checkpoints, driver, pool_size=1 << 20, batch=None,
                score_mode="sum", pick="uniform"):
    """The leaves decide where to evaluate next. A Sobol sample of candidate cube points
    is routed to the leaves; a leaf's expected error contribution is the sum (or mean) of
    w |M|^2 sigma over its candidates, from its own GP. The leaf with the largest one gets
    a batch of its candidates (uniform in the leaf, or its highest scores), evaluated with
    the amplitude code; the affected leaves are rescored and re-routed after splits."""
    from scipy.stats import qmc
    tr, te, tu = load(data_dir, ch)
    n0 = checkpoints[0]
    X0, a0 = tr["X"][:n0], tr["me"][:n0]
    set_scale(a0, X0)
    batch = batch or max(1, OPTS["nbar"] // 4)
    rng = np.random.RandomState(seed)

    def learn(Xs, a):
        y = to_target(a, Xs)
        sig = 1e-6 * (np.abs(y) if linear_target() else np.ones(len(y)))
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(len(a)):
                gpt.update_tree(Xs[i:i + 1], y[i:i + 1].reshape(1, 1), sig[i])

    gpt = make("tree", seed)
    learn(X0, a0)
    C = qmc.Sobol(d=5, scramble=True, seed=seed + 100).random(pool_size)
    Xc = tree_inputs(C)
    wc = phase_space_weight(C) * (rho(ch, C) if OPTS["rho"] else 1.0)

    def route(node, idx):
        if node.children is None:
            return {node.name: idx}
        right = Xc[idx, node.split_index] > node.split_position
        out = route(node.left, idx[~right])
        out.update(route(node.right, idx[right]))
        return out

    def rescore(entry):
        idx = entry["idx"]
        if len(idx) == 0:
            entry["scores"] = np.zeros(0); entry["total"] = -1.0
            return
        with contextlib.redirect_stdout(io.StringIO()):
            mu, sd = entry["node"].predict(Xc[idx], return_std=True, use_calibrated_sigma=gpt.use_calibrated_sigma)
        a_pred, sig_a = from_target(mu.ravel(), sd.ravel(), Xc[idx])
        entry["scores"] = wc[idx] * np.abs(a_pred) * sig_a
        entry["total"] = entry["scores"].sum() if score_mode == "sum" else entry["scores"].mean()

    leaves = {}
    for name, idx in route(gpt.root, np.arange(pool_size)).items():
        leaves[name] = dict(node=next(l for l in gpt.root.leaves if l.name == name), idx=idx)
        rescore(leaves[name])
    bought, rows, t0, t_exact = n0, [], time.time(), 0.0
    for N in checkpoints:
        if N > nmax:
            break
        while bought < N:
            name, entry = max(leaves.items(), key=lambda kv: kv[1]["total"])
            idx = entry["idx"]
            k = min(batch, N - bought, len(idx))
            if pick == "top":
                sel = idx[np.argsort(entry["scores"])[::-1][:k]]
            else:
                sel = rng.choice(idx, size=k, replace=False)
            t1 = time.time()
            w_exact, a_exact = oracle(driver, C[sel], THETA_MAX)
            t_exact += time.time() - t1
            if bought == n0:   # the analytic weight against the driver's, once
                w_check = w_exact * (rho(ch, C[sel]) if OPTS["rho"] else 1.0)
                assert np.allclose(w_check, wc[sel], rtol=1e-6), "phase_space_weight disagrees with the driver"
            learn(Xc[sel], a_exact)
            bought += k
            entry["idx"] = np.setdiff1d(idx, sel, assume_unique=True)
            current = {l.name: l for l in gpt.root.leaves}
            changed = set()
            for lname in list(leaves):
                if lname not in current:   # this leaf split: route its candidates to the new leaves
                    old = leaves.pop(lname)
                    for cname, cidx in route(old["node"], old["idx"]).items():
                        leaves[cname] = dict(node=current[cname], idx=cidx)
                        changed.add(cname)
            if name in leaves:
                changed.add(name)
            for cname in changed:
                rescore(leaves[cname])
        m = evaluate(gpt, te, tu); t_train = time.time() - t0 - t_exact
        m.update(N=N, train_s=t_train, ms_per_point=1e3 * t_train / N, leaves=len(gpt.root.leaves))
        rows.append(m)
        print(f"{ch} leafacq N={N:6d}  eps_uniform {m['eps_uniform']:.2e}  eps_unweighted {m.get('eps_unweighted', float('nan')):.2e}  "
              f"median rel {m['median_rel']:.2e}  within 1 %: {100 * m['within1pct']:.0f} %  cov68 {m['cov68']:.2f}  "
              f"{m['ms_per_point']:.1f} ms/pt  leaves {m['leaves']}", flush=True)
    return rows


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chans = sorted({k[0] for k in results})
    fig, axes = plt.subplots(1, len(chans), figsize=(5.2 * len(chans), 3.8), squeeze=False)
    for ax, ch in zip(axes[0], chans):
        for (c, cfg), rows in results.items():
            if c != ch:
                continue
            ax.plot([r["N"] for r in rows], [r["eps_uniform"] for r in rows], "-o", color=COLORS[cfg], lw=2, ms=5,
                    label=f"{cfg}, uniform test")
            if "eps_unweighted" in rows[0]:
                ax.plot([r["N"] for r in rows], [r["eps_unweighted"] for r in rows], "--s", color=COLORS[cfg], lw=1.5, ms=4,
                        label=f"{cfg}, unweighted test")
        ax.axhline(0.01, color="#52514e", lw=1, ls=":")
        ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else 1000, 0.0107, "1 % target", color="#52514e", fontsize=8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("training points"); ax.set_ylabel(r"$\epsilon$ (relative L1 error)")
        ax.set_title(CHANNEL_NAMES[ch], fontsize=10); ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--channels", default="gg,uu")
    ap.add_argument("--configs", default="tree,hybrid")
    ap.add_argument("--nmax", type=int, default=131072)
    ap.add_argument("--noise", type=float, default=0.0, help="relative Gaussian noise added to the training values")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--net", default="", help="hybrid network settings, e.g. hidden=256,depth=4,steps=600,polish_steps=10000")
    ap.add_argument("--kernel", default="matern32", choices=["matern32", "matern52", "rbf", "rq", "rbflin", "additive"],
                    help="leaf kernel: Matern, RBF, anisotropic rational quadratic, RBF plus a linear kernel, or additive Matern")
    ap.add_argument("--nbar", type=int, default=100)
    ap.add_argument("--train", default="sobol", choices=["sobol", "unweighted", "mixed"],
                    help="training points: Sobol, drawn with probability proportional to w |M|^2, or alternating")
    ap.add_argument("--inputs", default="cube", choices=["cube", "invariants", "both", "invbeta"],
                    help="the tree's inputs: cube variables, log invariants, both, or the invariants and the t tbar threshold velocity")
    ap.add_argument("--ratio", action="store_true", help="learn me / tree, the one-loop amplitude in units of the Born (f2, f4)")
    ap.add_argument("--features", default="cube", choices=["cube", "invariants"], help="the hybrid network's inputs")
    ap.add_argument("--split", default="min_lengthscale", help="GPTree split_dimension_criteria")
    ap.add_argument("--retrain", type=int, default=25,
                    help="retrain a leaf's GP every this many new points; a value above Nbar retrains only when the leaf splits")
    ap.add_argument("--strategy", default="gradual", choices=["gradual", "standard"], help="GPTree splitting_strategy")
    ap.add_argument("--target", default="log", choices=TARGETS,
                    help="learn log|M|^2; log|M|^2 with the gluon-top propagator poles 1/(p_g.p_t)^p factored out (pole1, pole2); "
                         "log(|M|^2 s^k) (logs1, logs2); |M|^2 s^k linearly, scaled to its median (lin, lins1, lins2); "
                         "or (|M|^2)^p (pow0.5, pow0.25)")
    ap.add_argument("--global", dest="glob", default="none", choices=["none", "additive"],
                    help="tree-wide global model whose residual the leaves fit (plain tree only)")
    ap.add_argument("--no-rho", dest="rho", action="store_false", help="weights without the flux and pseudo-PDF factor")
    ap.add_argument("--domain", default="folded", choices=["folded", "full"], help="polar angles in [0, pi/2] or [0, pi]")
    ap.add_argument("--theta", type=float, default=None, help="GPTree theta (overlap between sibling leaves as a fraction of the split range)")
    ap.add_argument("--driver", default=None, help="the MadGraph driver, for the leafacq config")
    ap.add_argument("--leaf-batch", type=int, default=None, help="points bought per leafacq step (default Nbar/4)")
    ap.add_argument("--leaf-score", default="sum", choices=["sum", "mean"], help="a leaf's expected contribution: sum or mean over its candidates")
    ap.add_argument("--leaf-pick", default="uniform", choices=["uniform", "top"], help="which of the chosen leaf's candidates to buy")
    ap.add_argument("--leaf-pool", type=int, default=1 << 20, help="number of Sobol candidate points for leafacq")
    args = ap.parse_args()
    globals()["THETA_MAX"] = np.pi if args.domain == "full" else 0.5 * np.pi
    for item in filter(None, args.net.split(",")):
        k, v = item.split("="); NET_KWARGS[k] = None if v == "none" else int(v)
    OPTS.update(rho=args.rho, kernel=args.kernel, nbar=args.nbar, train=args.train, inputs=args.inputs, features=args.features,
                split=args.split, retrain=args.retrain, strategy=args.strategy, target=args.target, glob=args.glob, theta=args.theta,
                ratio=args.ratio)
    os.makedirs(OUT_DIR, exist_ok=True)
    checkpoints = [1000, 2000, 4000, 8000, 16000, 32768, 65536, 131072, 262144, 524288]
    results = {}
    for ch in args.channels.split(","):
        for cfg in args.configs.split(","):
            if cfg in ("acq", "hybacq"):
                results[(ch, cfg)] = run_acquisition(ch, args.data_dir, args.nmax, args.seed, checkpoints,
                                                     model="tree" if cfg == "acq" else "hybrid")
            elif cfg == "leafacq":
                results[(ch, cfg)] = run_leafacq(ch, args.data_dir, args.nmax, args.seed, checkpoints, args.driver, args.leaf_pool,
                                                 args.leaf_batch, args.leaf_score, args.leaf_pick)
            else:
                results[(ch, cfg)] = run(ch, cfg, args.data_dir, args.nmax, args.noise, args.seed, checkpoints)
    tag = args.tag or (f"noise{args.noise:g}" if args.noise > 0 else "exact")
    with open(os.path.join(OUT_DIR, f"results_{tag}.json"), "w") as f:
        json.dump({f"{ch}_{cfg}": rows for (ch, cfg), rows in results.items()}, f, indent=1)
    plot(results, os.path.join(OUT_DIR, f"eps_vs_points_{tag}.png"))
    print("\n| channel | config | N | eps uniform | eps unweighted | median rel | within 1 % | cov68 | ms/pt |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (ch, cfg), rows in results.items():
        for r in rows:
            print(f"| {ch} | {cfg} | {r['N']} | {r['eps_uniform']:.2e} | {r.get('eps_unweighted', float('nan')):.2e} | "
                  f"{r['median_rel']:.2e} | {100 * r['within1pct']:.0f} % | {r['cov68']:.2f} | {r['ms_per_point']:.1f} |")


if __name__ == "__main__":
    main()
