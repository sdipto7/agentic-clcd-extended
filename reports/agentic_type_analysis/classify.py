#!/usr/bin/env python3
"""
Full-population type analysis of agentic route selection (direct vs algorithm-based).

Loads every (model, dataset, pair_id) agentic evaluation, determines its route
(direct or algorithm-based) using the same logic as scripts/analyze_agentic_strategy.py,
attaches the underlying pair's source code, classifies the pair into one dominant
"type" bucket using transparent regex/keyword rules, and a length bucket, then
cross-tabulates route choice against type across the FULL population (31,200
evaluations), not a sample.

The 300 sampled pairs used for manual verification are stored, with their
assigned types, in random150_samples_for_verification.json.
"""
import json, csv, re, random
from pathlib import Path
from collections import defaultdict, Counter

PROJECT_ROOT = Path("/home/sxa510/research/code/agentic-clcd")
DATASETS = ["xlcost", "codenet", "gptclonebench"]
MODELS = ["claude-3-haiku", "deepseek-v3.2", "gpt-4o-mini", "llama-3.3-nemotron"]

# ---------- 1. load all pair source code, keyed by pair_id ----------
pairs = {}
for ds in DATASETS:
    rows = json.load(open(PROJECT_ROOT / "data" / f"{ds}.json"))
    for r in rows:
        pairs[r["pair_id"]] = r

# ---------- 2. load route choice for every (model, dataset, pair_id) ----------
# Route = "algo" if pair_id appears in that model/dataset's algorithms json
# (agent called record_algorithms), else "direct". Same rule as
# scripts/analyze_agentic_strategy.py.
evaluations = []  # list of dicts: pair_id, model, dataset, route
for ds in DATASETS:
    for model in MODELS:
        algo_path = PROJECT_ROOT / "output" / "agentic" / f"algorithms_{model}_{ds}.json"
        results_path = PROJECT_ROOT / "output" / "agentic" / f"results_{model}_{ds}.csv"
        algo_pair_ids = set(json.load(open(algo_path)).keys())
        with open(results_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pid = row["pair_id"]
                route = "algo" if pid in algo_pair_ids else "direct"
                evaluations.append({"pair_id": pid, "model": model, "dataset": ds, "route": route})

print(f"Total agentic evaluations loaded: {len(evaluations)}")
route_counts = Counter(e["route"] for e in evaluations)
print("Route counts:", dict(route_counts))

# ---------- 3. classify each unique pair_id into one dominant type + length bucket ----------

LOOP_RE = re.compile(r'\b(for\s*\(|while\s*\(|foreach\s*\()', re.IGNORECASE)
COND_RE = re.compile(r'\b(if\s*\(|else\s+if|elif\b)', re.IGNORECASE)
STRING_RE = re.compile(
    r'(substring|charAt|indexOf|\.split\(|strcmp|strlen|\.find\(|\.replace\(|toCharArray|'
    r'\.Substring\(|\.IndexOf\(|\.Split\(|\.strip\(|\.join\(|\.startswith\(|\.endswith\(|'
    r's\[0\]|StringBuilder|std::string|\.compareTo\()',
    re.IGNORECASE,
)
COLLECTION_RE = re.compile(
    r'(HashMap|TreeMap|Map<|unordered_map|std::map|HashSet|TreeSet|Set<|std::set|'
    r'dict\(|Dictionary<|\bdict\b|defaultdict)',
    re.IGNORECASE,
)
SORT_RE = re.compile(r'(\.sort\(|Arrays\.sort|Collections\.sort|sorted\(|qsort|std::sort|Array\.Sort)', re.IGNORECASE)
ARRAYLIST_RE = re.compile(
    r'(ArrayList|std::vector|List<|\[\s*\]|new\s+int\s*\[|new\s+\w+\s*\[|\bvector<|\.append\(|\.push_back\()',
    re.IGNORECASE,
)
MATH_RE = re.compile(r'(Math\.|pow\(|sqrt\(|std::abs|\babs\(|%\s*\d|\bfloor\(|\bceil\()', re.IGNORECASE)

def count_matches(pattern, text):
    return len(pattern.findall(text))


def line_count(src):
    return len([l for l in src.split("\n") if l.strip()])


def length_bucket(n):
    if n <= 14:
        return "short (<=14 lines)"
    if n <= 35:
        return "medium (15-35 lines)"
    return "long (>35 lines)"


def classify_type(java_src, other_src):
    combined = java_src + "\n" + other_src

    loops = count_matches(LOOP_RE, combined)
    conds = count_matches(COND_RE, combined)

    if COLLECTION_RE.search(combined):
        return "Map/Set-based logic"
    if SORT_RE.search(combined):
        return "Sorting"
    string_hits = count_matches(STRING_RE, combined)
    arraylist_hits = count_matches(ARRAYLIST_RE, combined)
    if string_hits >= 2 and string_hits >= arraylist_hits:
        return "String processing"
    if arraylist_hits >= 2:
        return "Array/List processing"
    if loops >= 2:
        return "Nested/loop-heavy iteration"
    if conds >= 3 and loops == 0:
        return "Conditional/branch-heavy logic"
    if MATH_RE.search(combined) and loops <= 1 and conds <= 2:
        return "Arithmetic/numeric computation"
    if line_count(java_src) <= 10 and loops == 0:
        return "Short, low-control-flow logic"
    return "Other/mixed logic"


type_cache = {}
length_cache = {}
for pid, row in pairs.items():
    t = classify_type(row["codeA"], row["codeB"])
    type_cache[pid] = t
    length_cache[pid] = length_bucket(line_count(row["codeA"]))

# ---------- 4. cross-tabulate: type x route, over FULL population ----------
type_route_counts = defaultdict(lambda: Counter())
length_route_counts = defaultdict(lambda: Counter())
for e in evaluations:
    t = type_cache[e["pair_id"]]
    l = length_cache[e["pair_id"]]
    type_route_counts[t][e["route"]] += 1
    length_route_counts[l][e["route"]] += 1
    e["type"] = t
    e["length_bucket"] = l

print("\n=== TYPE x ROUTE (full population, 31,200 evaluations) ===")
print(f"{'Type':<32} {'Direct':>8} {'Algo':>8} {'Total':>8} {'%Direct':>8} {'%Algo':>8}")
for t, counts in sorted(type_route_counts.items(), key=lambda kv: -sum(kv[1].values())):
    d, a = counts.get("direct", 0), counts.get("algo", 0)
    tot = d + a
    print(f"{t:<32} {d:>8} {a:>8} {tot:>8} {d/tot*100:>7.1f}% {a/tot*100:>7.1f}%")

print("\n=== LENGTH BUCKET x ROUTE (full population) ===")
print(f"{'Length':<24} {'Direct':>8} {'Algo':>8} {'Total':>8} {'%Direct':>8} {'%Algo':>8}")
for l, counts in sorted(length_route_counts.items()):
    d, a = counts.get("direct", 0), counts.get("algo", 0)
    tot = d + a
    print(f"{l:<24} {d:>8} {a:>8} {tot:>8} {d/tot*100:>7.1f}% {a/tot*100:>7.1f}%")

# within-route composition (what % of direct-routed evals are each type)
print("\n=== WITHIN DIRECT-ROUTED: type composition ===")
direct_total = route_counts["direct"]
for t, counts in sorted(type_route_counts.items(), key=lambda kv: -kv[1].get("direct", 0)):
    d = counts.get("direct", 0)
    print(f"{t:<32} {d:>8}  ({d/direct_total*100:5.1f}% of all direct-routed)")

print("\n=== WITHIN ALGO-ROUTED: type composition ===")
algo_total = route_counts["algo"]
for t, counts in sorted(type_route_counts.items(), key=lambda kv: -kv[1].get("algo", 0)):
    a = counts.get("algo", 0)
    print(f"{t:<32} {a:>8}  ({a/algo_total*100:5.1f}% of all algo-routed)")

# save full per-evaluation dataset for sampling + audit
out_path = Path(__file__).parent / "all_evaluations.json"
with open(out_path, "w") as f:
    json.dump(evaluations, f)
print(f"\nSaved {len(evaluations)} classified evaluations to {out_path}")
