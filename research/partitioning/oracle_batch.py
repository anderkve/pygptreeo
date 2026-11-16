"""Run oracle.py for several (target, stream) pairs in parallel."""
import itertools
import subprocess
import sys
from multiprocessing.pool import ThreadPool

targets = sys.argv[1].split(',')
streams = sys.argv[2].split(',')
extra = sys.argv[3:]


def job(ts):
    out = subprocess.run([sys.executable, 'oracle.py', '--target', ts[0], '--stream', ts[1]] + extra,
                         capture_output=True, text=True)
    return out.stdout.strip() or out.stderr.strip()[-2000:]


with ThreadPool(4) as pool:
    for line in pool.imap_unordered(job, itertools.product(targets, streams)):
        print(line, flush=True)
