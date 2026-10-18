#!/usr/bin/env python3
"""
Is the code type itself linked to "algorithm wrong | direct correct", or does the pattern
come from the language pair, dataset, model and length mix?

For every type the observed number of direct-only evaluations is compared with the number
expected from the rate of its stratum (dataset x language pair x model x length bucket).
O/E above 1 means the algorithm-based pipeline failed more often than expected for that type.
95% intervals come from a bootstrap that resamples whole pairs (all models of a pair together).

Usage: python reports/rq3_direct_only_analysis/adjusted_type_effect.py
"""
import io, contextlib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
with contextlib.redirect_stdout(io.StringIO()):
    import analyze_direct_only as base
import numpy as np, pandas as pd

df = pd.DataFrame([{k: e[k] for k in ("pair_id", "model", "dataset", "language_pair", "length", "type", "cell")} for e in base.evals])
dc = df[df.cell.isin(["both_correct", "direct_only"])].copy()
dc["y"] = (dc.cell == "direct_only").astype(int)
S = (dc.dataset + "|" + dc.language_pair + "|" + dc.model + "|" + dc.length).astype("category").cat.codes.values
T = dc.type.astype("category"); tnames = list(T.cat.categories); tc = T.cat.codes.values
Y = dc.y.values; pid = dc.pair_id.astype("category").cat.codes.values
nS, nT = S.max() + 1, len(tnames)
order = np.argsort(pid, kind="stable")
cut = np.r_[0, np.flatnonzero(np.diff(pid[order])) + 1, len(order)]
groups = [order[cut[i]:cut[i + 1]] for i in range(len(cut) - 1)]


def oe(idx):
    s, y, t = S[idx], Y[idx], tc[idx]
    cnt = np.bincount(s, minlength=nS); pos = np.bincount(s, weights=y, minlength=nS)
    rate = np.divide(pos, cnt, out=np.zeros(nS), where=cnt > 0)
    return np.bincount(t, weights=y, minlength=nT) / np.bincount(t, weights=rate[s], minlength=nT)


obs = oe(np.arange(len(dc)))
rng = np.random.default_rng(2024)
boots = np.array([oe(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])) for _ in range(2000)])
print(f"Direct-correct evaluations: {len(dc)}; algorithm wrong among them: {dc.y.sum()} ({dc.y.mean():.2%}); unique pairs: {len(groups)}")
print(f"{'type':<26}{'O/E':>7}{'95% CI (pair bootstrap)':>26}   reading")
for i in np.argsort(-obs):
    lo, hi = np.percentile(boots[:, i], [2.5, 97.5])
    print(f"{tnames[i]:<26}{obs[i]:>7.2f}{f'[{lo:.2f}, {hi:.2f}]':>26}   " +
          ("above expected" if lo > 1 else "below expected" if hi < 1 else "not clearly different from 1"))
