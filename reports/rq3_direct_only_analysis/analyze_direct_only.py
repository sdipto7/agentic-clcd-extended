#!/usr/bin/env python3
"""
Type analysis of the "direct correct, algorithm-based wrong" evaluations (RQ3).

Every (model, dataset, pair_id) evaluation is placed in one of four outcomes by
comparing the direct and algorithm-based predictions with the ground truth
(the same rule as scripts/compare_direct_vs_algo_based.py). The 1,977
"direct correct, algorithm wrong" evaluations are then compared with all
31,200 evaluations by code type, code length, language pair, dataset and
model. Types use the same keyword rules as reports/agentic_type_analysis/classify.py.

A random sample of 100 of the 1,977 evaluations (seed 7) is written to
direct_only_random100_for_verification.json.

Usage: python reports/rq3_direct_only_analysis/analyze_direct_only.py
"""
import csv, json, re, sys, random
from pathlib import Path
from collections import Counter, defaultdict
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.core.constants import ERROR, LABEL_TO_VERDICT, NOT_CLONE  # noqa: E402

DATASETS = ["xlcost", "codenet", "gptclonebench"]
MODELS = ["claude-3-haiku", "deepseek-v3.2", "gpt-4o-mini", "llama-3.3-nemotron"]
SEED = 7
SAMPLE_SIZE = 100

# keyword rules: reuse the exact code of the RQ4 classifier
_src = (ROOT / "reports/agentic_type_analysis/classify.py").read_text()
_ns = {"re": re, "Counter": Counter, "defaultdict": defaultdict, "json": json}
exec(_src[_src.index("LOOP_RE"): _src.index("type_cache = {}")], _ns)
RENAME = {
    "Array/List processing": "Array and list handling",
    "Nested/loop-heavy iteration": "Loop-heavy code",
    "String processing": "String handling",
    "Map/Set-based logic": "Map and set handling",
    "Sorting": "Sorting",
    "Arithmetic/numeric computation": "Numeric computation",
    "Conditional/branch-heavy logic": "Condition-heavy code",
    "Short, low-control-flow logic": "Short simple code",
    "Other/mixed logic": "Other or mixed code",
}
TYPES = list(RENAME.values())

pairs = {}
for ds in DATASETS:
    for r in json.load(open(ROOT / "data" / f"{ds}.json")):
        pairs[r["pair_id"]] = r


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["pair_id"]: r for r in csv.DictReader(f)}


evals = []
for ds in DATASETS:
    for m in MODELS:
        d = load_rows(ROOT / "output/direct" / f"results_{m}_{ds}.csv")
        a = load_rows(ROOT / "output/algo_based" / f"results_{m}_{ds}.csv")
        assert set(d) == set(a)
        algs = json.load(open(ROOT / "output/algo_based" / f"algorithms_{m}_{ds}.json"))
        for pid in d:
            assert d[pid]["ground_truth"] == a[pid]["ground_truth"]
            gt = int(d[pid]["ground_truth"])
            gt_label = LABEL_TO_VERDICT.get(gt, NOT_CLONE)
            dp = d[pid]["predicted_label"].strip().upper()
            ap = a[pid]["predicted_label"].strip().upper()
            dc, ac = dp == gt_label and dp != ERROR, ap == gt_label and ap != ERROR
            cell = "both_correct" if dc and ac else "direct_only" if dc else "algo_only" if ac else "both_wrong"
            p = pairs[pid]
            n = _ns["line_count"](p["codeA"])
            evals.append(dict(
                pair_id=pid, model=m, dataset=ds, language_pair=f'{d[pid]["ll1"]}-{d[pid]["ll2"]}',
                gt=gt, gt_label=gt_label, direct_pred=dp, algo_pred=ap, cell=cell,
                type=RENAME[_ns["classify_type"](p["codeA"], p["codeB"])],
                java_lines=n, length=_ns["length_bucket"](n), alg=algs.get(pid, {})))


# ---- data-integrity check: an algorithm text that is identical to the text stored for another
# pair although the underlying code is clearly different (character-level similarity < 0.5) is flagged.
import difflib
_norm = lambda s: " ".join(s.split())
SHARED = set()   # (model, dataset, pair_id)
for ds in DATASETS:
    for m in MODELS:
        _a = json.load(open(ROOT / "output/algo_based" / f"algorithms_{m}_{ds}.json"))
        for key, ck in (("java_algorithm", "codeA"), ("ll2_algorithm", "codeB")):
            _g = defaultdict(list)
            for pid, v in _a.items():
                if v.get(key):
                    _g[v[key]].append(pid)
            for ids in _g.values():
                if len(ids) < 2:
                    continue
                base = _norm(pairs[ids[0]][ck])
                if any(difflib.SequenceMatcher(None, base, _norm(pairs[i][ck]), autojunk=False).quick_ratio() < 0.5
                       for i in ids[1:]):
                    SHARED.update((m, ds, i) for i in ids)
for e in evals:
    e["shared_alg"] = (e["model"], e["dataset"], e["pair_id"]) in SHARED

N = len(evals)
D = [e for e in evals if e["cell"] == "direct_only"]
A = [e for e in evals if e["cell"] == "algo_only"]
print(f"Evaluations: {N}   outcomes: {dict(Counter(e['cell'] for e in evals))}")
assert (N, len(D), len(A)) == (31200, 1977, 3934)


def breakdown(key, order=None, title=""):
    groups = defaultdict(list)
    for e in evals:
        groups[e[key]].append(e)
    print(f"\n=== {title} ===")
    print(f"{'group':<26}{'n':>6}{'D':>6}{'A':>6}{'D/n%':>7}{'A/n%':>7}{'P(algo wrong|direct ok)%':>26}{'share of D%':>13}{'share of all%':>15}")
    for k in order or sorted(groups, key=lambda k: -len(groups[k])):
        g = groups[k]
        d = sum(e["cell"] == "direct_only" for e in g)
        a = sum(e["cell"] == "algo_only" for e in g)
        b = sum(e["cell"] == "both_correct" for e in g)
        print(f"{k:<26}{len(g):>6}{d:>6}{a:>6}{d/len(g)*100:>7.2f}{a/len(g)*100:>7.2f}{d/(d+b)*100:>26.2f}{d/len(D)*100:>13.1f}{len(g)/N*100:>15.1f}")


breakdown("type", TYPES, "TYPE (all 31,200 evaluations)")
breakdown("length", ["short (<=14 lines)", "medium (15-35 lines)", "long (>35 lines)"], "JAVA LENGTH")
breakdown("language_pair", None, "LANGUAGE PAIR")
breakdown("dataset", None, "DATASET")
breakdown("model", None, "MODEL")
print(f"\nDirect-only evaluations: algorithm false negatives {sum(e['gt']==1 for e in D)}, false positives {sum(e['gt']==0 for e in D)}")


# ---- robustness: drop the flagged evaluations and recompute the headline numbers
clean = [e for e in evals if not e["shared_alg"]]
Dc = [e for e in clean if e["cell"] == "direct_only"]
print(f"\nFlagged evaluations: {N - len(clean)} ({(N-len(clean))/N:.1%}); direct-only among them: {len(D) - len(Dc)} of {len(D)}")
print(f"Without flagged: n={len(clean)}, direct-only={len(Dc)}, algo-only={sum(e['cell']=='algo_only' for e in clean)}")
for lp in ["Java-Python", "Java-C#", "Java-C++"]:
    g = [e for e in clean if e["language_pair"] == lp]
    d = sum(e["cell"] == "direct_only" for e in g); a = sum(e["cell"] == "algo_only" for e in g)
    print(f"  {lp:<12} n={len(g):>6} D={d:>4} A={a:>4} D/n={d/len(g)*100:5.2f}% A/n={a/len(g)*100:5.2f}%")
print("  P(algo wrong | direct ok) by type, without flagged:")
for t in TYPES:
    g = [e for e in clean if e["type"] == t]
    d = sum(e["cell"] == "direct_only" for e in g); b = sum(e["cell"] == "both_correct" for e in g)
    print(f"    {t:<26}{d/(d+b)*100:6.2f}%")

# ---- random sample of 100 for manual verification
rng = random.Random(SEED)
sample = rng.sample(D, SAMPLE_SIZE)
out = []
for i, e in enumerate(sample, 1):
    p = pairs[e["pair_id"]]
    out.append({
        "sample_id": f"directonly_{i:03d}", "pair_id": e["pair_id"], "model": e["model"],
        "dataset": e["dataset"], "language_pair": e["language_pair"],
        "ground_truth": e["gt_label"], "direct_prediction": e["direct_pred"],
        "algo_based_prediction": e["algo_pred"],
        "algo_error_kind": "false negative" if e["gt"] == 1 else "false positive",
        "length_bucket": e["length"], "java_lines": e["java_lines"],
        "assigned_type": e["type"], "reviewer_type": None,
        "algorithm_text_shared_with_different_code": e["shared_alg"],
        "java_code": p["codeA"], "other_language_code": p["codeB"],
        "java_algorithm": e["alg"].get("java_algorithm"),
        "other_language_algorithm": e["alg"].get("ll2_algorithm"),
    })
doc = {
    "description": "Random sample of 100 evaluations where the direct pipeline was correct and the algorithm-based pipeline was wrong (1,977 such evaluations in total). Drawn uniformly at random with seed 7. assigned_type comes from the same keyword rules as the RQ4 analysis. reviewer_type is left empty for manual verification. algorithm_text_shared_with_different_code is true when a stored algorithm text is identical to the text stored for another pair whose code is clearly different, so the algorithm may not describe this pair's code.",
    "population_size": len(D), "random_seed": SEED,
    "type_counts_sample": dict(Counter(x["assigned_type"] for x in out)),
    "type_counts_all_1977": dict(Counter(e["type"] for e in D)),
    "samples": out,
}
(json_path := Path(__file__).parent / "direct_only_random100_for_verification.json").write_text(
    json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSample type counts: {doc['type_counts_sample']}")
print(f"Unique pair_ids in sample: {len({x['pair_id'] for x in out})} of {len(out)}")
print(f"Saved {json_path}")
