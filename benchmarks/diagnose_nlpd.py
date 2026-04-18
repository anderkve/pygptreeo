"""Diagnose the pygptreeo NLPD spikes.

Replays a rosenbrock_2d run and inspects the distribution of (mean, std,
y_true) on the held-out test set at every checkpoint, so we can see exactly
which test points drive the huge NLPD.
"""
import sys
sys.path.insert(0, "/home/user/pygptreeo")
import contextlib, io
import numpy as np

from benchmarks.adapters import PyGPTreeOAdapter
from benchmarks.problems import PROBLEMS


def main():
    prob = PROBLEMS["rosenbrock_2d"]
    rng = np.random.default_rng(0)
    X_stream, y_stream = prob.sample(2000, rng)
    X_test, y_test = prob.sample(400, rng)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m = PyGPTreeOAdapter(prob.dim, Nbar=200, retrain_step=200)
        cached = {}
        checkpoints = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
        for i in range(max(checkpoints)):
            m.update(X_stream[i:i+1], np.array([[y_stream[i]]]))
            step = i + 1
            if step in checkpoints:
                mean, std = m.predict(X_test)
                cached[step] = (mean.copy(), std.copy())

    for step, (mean, std) in cached.items():
        err = mean.ravel() - y_test
        s = std.ravel()
        with np.errstate(all="ignore"):
            nlpd_i = 0.5*np.log(2*np.pi*s**2) + 0.5*(err/s)**2
        print(f"\nstep {step}:")
        print(f"  mean nlpd = {nlpd_i.mean():.3e}")
        print(f"  max nlpd  = {nlpd_i.max():.3e}")
        print(f"  # pts with std<1e-6: {(s < 1e-6).sum()}")
        print(f"  # pts with std==0:   {(s == 0).sum()}")
        print(f"  std stats: min={s.min():.3e}, median={np.median(s):.3e}, max={s.max():.3e}")
        for j in np.argsort(nlpd_i)[-5:]:
            print(f"    idx {j}: y_true={y_test[j]:.3e}, mean={mean[j,0]:.3e}, "
                  f"std={s[j]:.3e}, err={err[j]:.3e}, nlpd={nlpd_i[j]:.3e}")


if __name__ == "__main__":
    main()
