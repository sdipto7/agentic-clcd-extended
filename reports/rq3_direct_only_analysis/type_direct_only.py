#!/usr/bin/env python3
"""
RQ3: what kinds of code pairs does the algorithm-based pipeline get wrong when direct detection is right?

Step 1  Join the direct and algorithm-based predictions for all 31,200 model-pair evaluations and
        pick the 1,977 "direct correct, algorithm wrong" evaluations (same rule as
        scripts/compare_direct_vs_algo_based.py).
Step 2  Give every pair one code type from the Java fragment and the one-line description that the
        algorithm-extraction step wrote for it. Types that match RQ4 keep the RQ4 name.
Step 3  Report, for every type, how often the algorithm-based pipeline was wrong when direct was
        right, then repeat this after accounting for dataset, language pair, model and length.
Step 4  Check whether the stored second-language algorithm describes the second-language code.
Step 5  Write one row per direct-correct / algorithm-wrong evaluation to
        direct_only_1977_typed.json (code is included for a random sample of 100, seed 7).

Usage: python reports/rq3_direct_only_analysis/type_direct_only.py
"""
import csv, json, re, sys, random, difflib
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.core.constants import ERROR, LABEL_TO_VERDICT, NOT_CLONE  # noqa: E402

DATASETS = ["xlcost", "codenet", "gptclonebench"]
MODELS = ["claude-3-haiku", "deepseek-v3.2", "gpt-4o-mini", "llama-3.3-nemotron"]
SEED_SAMPLE, SAMPLE_SIZE, SEED_BOOT, N_BOOT = 7, 100, 11, 2000

# ------------------------------------------------------------------ typing rules
import re
from collections import Counter

def strip_comments_strings(src):
    src = re.sub(r'//.*', '', src)
    src = re.sub(r'/\*.*?\*/', '', src, flags=re.S)
    return src

def nonempty(src): return len([l for l in src.split("\n") if l.strip()])

def count(rx, s): return len(re.findall(rx, s))

# ---------- feature regexes (applied to Java text mostly) ----------
API_GUI   = r'\b(JFrame|JPanel|JButton|JLabel|JTextField|JProgressBar|JOptionPane|Swing|AWT|awt\.|swing\.|android\.|TextView|Activity|Toast|ActionListener|Graphics|Color\.)'
API_FILE  = r'\b(File|Files|Paths?|FileWriter|FileReader|FileInputStream|FileOutputStream|ZipOutputStream|ZipInputStream|GZIP\w*|FileVisitResult|FileChannel|ObjectOutputStream|ObjectInputStream|DirectoryStream)\b'
API_NET   = r'\b(URL|URLConnection|HttpURLConnection|HttpClient|Socket|ServerSocket|InetAddress|URI|Request|Response|Servlet|HttpServlet)\b'
API_DB    = r'\b(Connection|Statement|PreparedStatement|ResultSet|DriverManager|SQLException|EntityManager)\b'
API_THREAD= r'\b(Thread|Runnable|Semaphore|ReentrantLock|Lock|ExecutorService|Executors|CountDownLatch|synchronized|AtomicInteger|Timer|TimerTask)\b'
API_TIME  = r'\b(Date|Calendar|SimpleDateFormat|LocalDate|LocalDateTime|Instant|Duration|TimeUnit|DateFormat|GregorianCalendar|System\.currentTimeMillis|nanoTime)\b'
API_SYS   = r'\b(System\.getProperty|System\.getenv|Runtime|ProcessBuilder|Logger|Properties|ResourceBundle|Class\.forName|reflect|MessageDigest|Cipher|KeyStore|Base64|UUID|JSON\w*|Gson|XML\w*)\b'
STRING    = r'(substring|charAt|indexOf|lastIndexOf|\.split\(|toCharArray|StringBuilder|StringBuffer|\.replace\(|\.trim\(|toUpperCase|toLowerCase|\.equals\(|\.equalsIgnoreCase|\.contains\(|startsWith|endsWith|\.length\(\)|String\.valueOf|Character\.|Pattern|Matcher)'
BITS      = r'(<<|>>>|>>|(?<!\^)\^(?!\^)|(?<!&)&(?!&)|(?<!\|)\|(?!\|)|bitCount|toBinaryString|numberOfTrailingZeros|highestOneBit)'
NUM       = r'(Math\.\w+|\bgcd\b|\blcm\b|isPrime|prime|sieve|factorial|fibonacci|\bpow\b|\bsqrt\b|\bmod\b|MOD|1000000007|\bdigit|BigInteger|BigDecimal)'
MAPSET    = r'(HashMap|TreeMap|LinkedHashMap|Map<|HashSet|TreeSet|LinkedHashSet|Set<|\.containsKey\(|\.getOrDefault\(|\.entrySet\(|\.keySet\()'
SORT      = r'(Arrays\.sort|Collections\.sort|\.sort\(|\bsort\s*\(|PriorityQueue|bubble|mergeSort|quickSort|insertionSort|selectionSort)'
DS_NODE   = r'(\bNode\b|\.next\b|\.left\b|\.right\b|LinkedList|\bStack<|\bQueue<|Deque<|ArrayDeque|\.push\(|\.pop\(|\.peek\(|\.poll\(|\.offer\(|TreeNode|ListNode)'
GRAPH     = r'(\bbfs\b|\bdfs\b|\bBFS\b|\bDFS\b|visited|\badj\b|adjacency|\bgraph\b|dijkstra|\bdx\b|\bdy\b)'
DP        = r'(\bdp\b|\bdp\[|\bmemo\b|\bmemo\[|\bcache\[|\bways\[)'
ARRAY     = r'(\[\s*\]|new\s+\w+\s*\[|ArrayList|List<|\.add\(|\.get\(|\.size\(\)|Arrays\.)'
MATRIX    = r'(\[\s*\]\s*\[\s*\]|\]\s*\[)'
LOOP      = r'\b(for|while)\s*\('
COND      = r'\b(if|else if|switch)\b'
IO_CP     = r'(Scanner|BufferedReader|InputStreamReader|StreamTokenizer|System\.in|nextInt|readLine)'
DIGITS    = r'(%\s*10\b|/\s*10\b|%\s*\(?\s*1000000007|% *MOD|%\s*\w+\s*==\s*0|%\s*\w+\s*!=\s*0)'
FLOATS    = r'(\bdouble\b|\bfloat\b|Math\.PI|\bprintf\b|String\.format)'
CATCH     = r'\bcatch\s*\('
API_EXTRA = r'\b(ActionEvent|ActionListener|JsonGenerator|JsonParser|Json\w+|MouseEvent|KeyEvent|Component|Container|Icon|Image\w*|Font|Dimension|Rectangle|Point2D|Bundle|Intent|Context|View|Listener|DocumentBuilder|SAXParser|XMLReader|Charset|Locale|NumberFormat|DecimalFormat)\b'
TREE      = r'(\broot\b|inorder|preorder|postorder|\binsert\b|\bleft\b|\bright\b)'
OOP       = r'\b(this\.|get[A-Z]\w*\(|set[A-Z]\w*\(|extends|implements|@Override|interface)\b'

def features(java, other):
    j = strip_comments_strings(java)
    o = strip_comments_strings(other)
    f = {}
    f["lines"] = nonempty(java)
    f["loops"] = count(LOOP, j) or count(LOOP, o)
    f["conds"] = count(COND, j)
    for k, rx in dict(gui=API_GUI, file=API_FILE, net=API_NET, db=API_DB, thread=API_THREAD,
                      time=API_TIME, sys=API_SYS, string=STRING, bits=BITS, num=NUM, mapset=MAPSET,
                      sort=SORT, ds=DS_NODE, graph=GRAPH, dp=DP, array=ARRAY, matrix=MATRIX,
                      cp=IO_CP, oop=OOP).items():
        f[k] = count(rx, j)
    # other-language corroboration for the same families
    f["digits"] = count(DIGITS, j); f["floats"] = count(FLOATS, j); f["catch"] = count(CATCH, j)
    f["api2"] = count(API_EXTRA, j); f["tree"] = count(TREE, j)
    f["o_mapset"] = count(r'(unordered_map|std::map|\bmap<|\bset<|unordered_set|Dictionary|HashSet|\bdict\b|\bdefaultdict|\bset\(|Counter\()', o)
    f["o_sort"] = count(r'(sort\(|sorted\(|\.sort\(|Sort\()', o)
    return f


TYPE_API   = "Library and API usage"
TYPE_DP    = "Dynamic programming"
TYPE_GRAPH = "Graph and grid search"
TYPE_DS    = "Data structure operations"
TYPE_BITS  = "Bit manipulation"
TYPE_SORT  = "Sorting"
TYPE_MAP   = "Map and set handling"
TYPE_STR   = "String handling"
TYPE_NUM   = "Numeric computation"
TYPE_MAT   = "Matrix and 2D array processing"
TYPE_ARR   = "Array and list handling"
TYPE_SHORT = "Short simple code"
TYPE_COND  = "Condition-heavy code"
TYPE_OTHER = "Other"

# ---------- description-based evidence (Java one-line summary written during algorithm extraction) ----------
S_API   = r'(\bfile|director|folder|\bgui\b|swing|button|jtable|\bwindow|\bpanel|text ?field|textview|android|http|\burl\b|socket|json|\bxml\b|database|\bsql\b|thread|semaphore|serializ|\bzip|gzip|calendar|timestamp|\bdates?\b|execution time|clipboard|observable|listener|password|shuffle|command-line|logger|uuid|workbook|excel|runs a process|\bprompts? the user|\bmouse|\bkeyboard|\bclick|\bdialog|\bapplet|\bservlet|\bbrowser)'
S_DP    = r'(dynamic programming|memoiz|knapsack|longest (common|increasing|palindromic)|edit distance|coin change|\blis\b|\blcs\b|staircase)'
S_GRAPH = r'(graph|\bbfs\b|breadth|\bdfs\b|depth-first|shortest path|\bmaze\b|adjacen|vertices|vertex|\bedges?\b|connected component|bipartite|reachab|\bpath\b.*\bgrid|flood)'
S_DS    = r'(linked list|binary tree|\btree\b|\bstack\b|\bqueue\b|\bheap\b|\bnode\b|doubly|traversal|inorder|preorder|postorder|\bbst\b|priority queue)'
S_BITS  = r'(\bbits?\b|bitwise|\bxor\b|binary representation|gray code|power of (two|2)|\bmask)'
S_SORT  = r'(\bsort|ascending|descending|rank)'
S_MAP   = r'(hash ?map|hash ?set|hashmap|hashset|dictionary|frequency (map|table|count)|word frequenc|frequency of each|count of each|occurrences of each|unique words?|treemap|treeset)'
S_STR   = r'(\bstring|character|\bwords?\b|palindrome|substring|parenthes|anagram|vowel|consonant|alphabet|uppercase|lowercase|\bletters?\b|\btext\b|camel|prefix of|\bconcatenat)'
S_MAT   = r'(matrix|\bgrid\b|2d array|two-dimensional|rows and columns|\bboard\b)'
S_ARR   = r'(\barrays?\b|list of|\belements?\b|subarray|prefix sum|sliding window|\bpairs? of (elements|integers)|\blist\b|\bindices\b|\bindex\b)'
S_NUM   = r'(prime|\bgcd\b|\blcm\b|greatest common|least common|factor|divis|digit|\bnumbers?\b|\bsum\b|product|square|cube|probabilit|angle|\barea\b|volume|formula|calculat|comput|percent|modul|\binteger|coordinate|circle|triangle|polygon|series|fibonacci|power|remainder|\beven\b|\bodd\b|equation|\bmean\b|average|ratio|fraction|perfect|divisor|permutation|combination|binomial|catalan|\bsphere|\bcylinder|\bcone\b|distance|perimeter|diagonal|profit|price|cost|\bmax|\bmin)'

def summary_of(alg_text):
    if not alg_text: return ""
    m = re.search(r"Summary:\s*(.*?)(?:\n|$)", alg_text)
    if m: return m.group(1).lower()
    # some outputs have no 'Summary:' line; use the first descriptive sentence
    t = " ".join(alg_text.split()).lower()
    return t[:300]

S_NT = r'(prime|sieve|\bgcd\b|\blcm\b|greatest common|least common|factorial|fibonacci|modular|modulo|divisor|perfect number|catalan|binomial|totient|digit sum|sum of digits|armstrong|\bpower of\b)'

def assign2(java, other, summary=""):
    f = features(java, other)
    s = summary or ""
    sc = lambda rx: 1 if (s and re.search(rx, s)) else 0
    if f["gui"] or f["net"] or f["db"] or f["thread"] or f["time"] or f["file"] or f["sys"] or f["api2"] or (f["catch"] and not f["cp"]) or sc(S_API): return TYPE_API, f
    if f["graph"] >= 4 or sc(S_GRAPH): return TYPE_GRAPH, f
    if f["dp"] >= 2 or sc(S_DP): return TYPE_DP, f
    if f["ds"] >= 4 or f["tree"] >= 5 or sc(S_DS): return TYPE_DS, f
    if f["bits"] >= 4 or sc(S_BITS): return TYPE_BITS, f
    if f["sort"] >= 1 or sc(S_SORT): return TYPE_SORT, f
    if f["mapset"] >= 1 or sc(S_MAP): return TYPE_MAP, f
    if sc(S_NT): return TYPE_NUM, f
    if (f["string"] >= 2 and f["string"] >= f["array"]) or sc(S_STR): return TYPE_STR, f
    if f["matrix"] >= 2 or sc(S_MAT): return TYPE_MAT, f
    if f["array"] >= 3 or sc(S_ARR): return TYPE_ARR, f
    if f["num"] >= 1 or f["digits"] >= 1 or f["floats"] >= 1 or sc(S_NUM): return TYPE_NUM, f
    if f["lines"] <= 10 and f["loops"] == 0: return TYPE_SHORT, f
    if f["conds"] >= 3 and f["loops"] == 0: return TYPE_COND, f
    return TYPE_OTHER, f


# ------------------------------------------------------------------ data
pairs = {}
for ds in DATASETS:
    for r in json.load(open(ROOT / "data" / f"{ds}.json")):
        pairs[r["pair_id"]] = r
algs = {(m, ds): json.load(open(ROOT / "output/algo_based" / f"algorithms_{m}_{ds}.json")) for ds in DATASETS for m in MODELS}


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["pair_id"]: r for r in csv.DictReader(f)}


def length_bucket(n):
    return "short (<=14 lines)" if n <= 14 else "medium (15-35 lines)" if n <= 35 else "long (>35 lines)"


def pair_summary(pid):
    """Java description from the first model (fixed order) that produced one."""
    ds = pid.split("_")[0]
    for m in MODELS:
        a = algs[(m, ds)].get(pid, {}).get("java_algorithm", "") or ""
        if a and not a.startswith("(extraction failed"):
            s = summary_of(a)
            if s:
                return s
    return ""


TYPE, SUMMARY = {}, {}
for pid, p in pairs.items():
    SUMMARY[pid] = pair_summary(pid)
    TYPE[pid] = assign2(p["codeA"], p["codeB"], SUMMARY[pid])[0]

evals = []
for ds in DATASETS:
    for m in MODELS:
        d = load_rows(ROOT / "output/direct" / f"results_{m}_{ds}.csv")
        a = load_rows(ROOT / "output/algo_based" / f"results_{m}_{ds}.csv")
        assert set(d) == set(a)
        for pid in d:
            assert d[pid]["ground_truth"] == a[pid]["ground_truth"]
            gt = int(d[pid]["ground_truth"])
            gl = LABEL_TO_VERDICT.get(gt, NOT_CLONE)
            dp, ap = d[pid]["predicted_label"].strip().upper(), a[pid]["predicted_label"].strip().upper()
            dc, ac = dp == gl and dp != ERROR, ap == gl and ap != ERROR
            n = nonempty(pairs[pid]["codeA"])
            evals.append(dict(
                pair_id=pid, model=m, dataset=ds, lp=f'{d[pid]["ll1"]}-{d[pid]["ll2"]}', gt=gt, gt_label=gl,
                direct_pred=dp, algo_pred=ap,
                cell="both_correct" if dc and ac else "direct_only" if dc else "algo_only" if ac else "both_wrong",
                type=TYPE[pid], java_lines=n, length=length_bucket(n)))
N = len(evals)
D = [e for e in evals if e["cell"] == "direct_only"]
assert (N, len(D), sum(e["cell"] == "algo_only" for e in evals)) == (31200, 1977, 3934)
df = pd.DataFrame(evals)
ORDER = [t for t, _ in Counter(df.type).most_common()]

# ------------------------------------------------------------------ Step 3a: per-type table
print(f"Evaluations {N}; direct correct / algorithm wrong: {len(D)}\n")
print("=== per type ===")
print(f"{'type':<32}{'pairs':>6}{'evals':>7}{'D':>5}{'A':>5}{'algo wrong | direct right':>28}{'D:A':>6}{'p (McNemar)':>13}")
for t in ORDER:
    g = df[df.type == t]
    d, a, b = (g.cell == "direct_only").sum(), (g.cell == "algo_only").sum(), (g.cell == "both_correct").sum()
    p = binomtest(int(d), int(d + a), 0.5).pvalue
    print(f"{t:<32}{sum(v == t for v in TYPE.values()):>6}{len(g):>7}{d:>5}{a:>5}{d/(d+b)*100:>27.2f}%{d/a:>6.2f}{p:>13.2g}")
b_all = (df.cell == "both_correct").sum()
print(f"{'ALL':<32}{len(TYPE):>6}{N:>7}{len(D):>5}{(df.cell=='algo_only').sum():>5}{len(D)/(len(D)+b_all)*100:>27.2f}%")

# ------------------------------------------------------------------ Step 3b: adjusted (O/E, pair bootstrap)
dc = df[df.cell.isin(["both_correct", "direct_only"])].copy()
dc["y"] = (dc.cell == "direct_only").astype(int)
Tc = dc.type.astype("category"); names = list(Tc.cat.categories); tc = Tc.cat.codes.values; Y = dc.y.values
pid_codes = dc.pair_id.astype("category").cat.codes.values
order = np.argsort(pid_codes, kind="stable")
cut = np.r_[0, np.flatnonzero(np.diff(pid_codes[order])) + 1, len(order)]
groups = [order[cut[i]:cut[i + 1]] for i in range(len(cut) - 1)]


def oe_table(strata_cols, label):
    S = dc[strata_cols].astype(str).agg("|".join, axis=1).astype("category").cat.codes.values
    nS, nT = S.max() + 1, len(names)

    def oe(idx):
        s, y, t = S[idx], Y[idx], tc[idx]
        cnt = np.bincount(s, minlength=nS); pos = np.bincount(s, weights=y, minlength=nS)
        rate = np.divide(pos, cnt, out=np.zeros(nS), where=cnt > 0)
        return np.bincount(t, weights=y, minlength=nT) / np.bincount(t, weights=rate[s], minlength=nT)

    obs = oe(np.arange(len(dc))); rng = np.random.default_rng(SEED_BOOT)
    boots = np.array([oe(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])) for _ in range(N_BOOT)])
    print(f"\n=== observed / expected, standardised for {label} (95% CI from a bootstrap over pairs) ===")
    for i in np.argsort(-obs):
        lo, hi = np.percentile(boots[:, i], [2.5, 97.5])
        print(f"{names[i]:<32}{obs[i]:>6.2f}{f'[{lo:.2f}, {hi:.2f}]':>16}   " + ("ABOVE expected" if lo > 1 else "BELOW expected" if hi < 1 else "not clearly different"))


oe_table(["dataset", "lp", "model", "length"], "dataset x language pair x model x length")

# ------------------------------------------------------------------ Step 4: does the stored algorithm describe the code?
def toks(text):
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[_\W]+", " ", text.lower())
    return [t for t in text.split() if (len(t) >= 2 and not t.isdigit()) or (t.isdigit() and len(t) >= 3)]


MISMATCH = {}      # (model, dataset, pair_id, side) -> True if the algorithm matches another pair's code clearly better
for ds in DATASETS:
    ids = [p for p in pairs if p.startswith(ds + "_")]
    idx = {p: i for i, p in enumerate(ids)}
    for side, ck, akey in (("java", "codeA", "java_algorithm"), ("other", "codeB", "ll2_algorithm")):
        vec = TfidfVectorizer(analyzer=toks, sublinear_tf=True)
        X = vec.fit_transform([pairs[p][ck] for p in ids])
        for m in MODELS:
            a = algs[(m, ds)]
            pids = [p for p in a if a[p].get(akey) and not a[p][akey].startswith("(extraction failed")]
            S = (vec.transform([a[p][akey] for p in pids]) @ X.T).toarray()
            own = np.array([S[i, idx[p]] for i, p in enumerate(pids)]); best = S.max(axis=1); bi = S.argmax(axis=1)
            for i, p in enumerate(pids):
                MISMATCH[(m, ds, p, side)] = bool(ids[bi[i]] != p and best[i] > own[i] * 1.5 and best[i] - own[i] > 0.10)

_norm = lambda s: " ".join(s.split())
DUP = set()        # identical algorithm text stored for pairs whose code is clearly different
for ds in DATASETS:
    for m in MODELS:
        for key, ck in (("java_algorithm", "codeA"), ("ll2_algorithm", "codeB")):
            g = defaultdict(list)
            for p, v in algs[(m, ds)].items():
                if v.get(key):
                    g[v[key]].append(p)
            for ids in g.values():
                if len(ids) > 1 and any(difflib.SequenceMatcher(None, _norm(pairs[ids[0]][ck]), _norm(pairs[i][ck]), autojunk=False).quick_ratio() < 0.5 for i in ids[1:]):
                    DUP.update((m, ds, i) for i in ids)
for e in evals:
    e["mismatch_other"] = MISMATCH.get((e["model"], e["dataset"], e["pair_id"], "other"), False)
    e["mismatch_java"] = MISMATCH.get((e["model"], e["dataset"], e["pair_id"], "java"), False)
    e["dup_text"] = (e["model"], e["dataset"], e["pair_id"]) in DUP
print("\n=== stored algorithm vs code (heuristic, second-language side and Java side as control) ===")
for c in ["both_correct", "direct_only", "algo_only", "both_wrong"]:
    g = [e for e in evals if e["cell"] == c]
    print(f"{c:<13} n={len(g):>6}  second-language algorithm flagged {sum(e['mismatch_other'] for e in g)/len(g):6.1%}   Java algorithm flagged {sum(e['mismatch_java'] for e in g)/len(g):5.1%}   identical text under different code {sum(e['dup_text'] for e in g)/len(g):5.1%}")

# ------------------------------------------------------------------ Step 5: output
sample_ids = {(e["model"], e["dataset"], e["pair_id"]) for e in random.Random(SEED_SAMPLE).sample(D, SAMPLE_SIZE)}
rows = []
for e in D:
    key = (e["model"], e["dataset"], e["pair_id"])
    row = {"pair_id": e["pair_id"], "model": e["model"], "dataset": e["dataset"], "language_pair": e["lp"],
           "ground_truth": e["gt_label"], "direct_prediction": e["direct_pred"], "algo_based_prediction": e["algo_pred"],
           "algo_error_kind": "false negative" if e["gt"] == 1 else "false positive",
           "length_bucket": e["length"], "java_lines": e["java_lines"], "assigned_type": e["type"], "reviewer_type": None,
           "java_description": SUMMARY[e["pair_id"]],
           "second_language_algorithm_flagged_as_not_matching_code": e["mismatch_other"],
           "identical_algorithm_text_under_different_code": e["dup_text"],
           "in_random_100_sample": key in sample_ids}
    if key in sample_ids:
        a = algs[(e["model"], e["dataset"])][e["pair_id"]]
        row.update(java_code=pairs[e["pair_id"]]["codeA"], other_language_code=pairs[e["pair_id"]]["codeB"],
                   java_algorithm=a.get("java_algorithm"), other_language_algorithm=a.get("ll2_algorithm"))
    rows.append(row)
doc = {"description": "The 1,977 evaluations where the direct pipeline was right and the algorithm-based pipeline was wrong. assigned_type comes from keyword rules on the Java code and its one-line description (type_direct_only.py). reviewer_type is empty for manual checking. Full code and algorithms are included for the random sample of 100 (in_random_100_sample = true, seed 7). The two algorithm flags are heuristics.",
       "type_counts": dict(Counter(r["assigned_type"] for r in rows)),
       "type_counts_random_100": dict(Counter(r["assigned_type"] for r in rows if r["in_random_100_sample"])),
       "evaluations": rows}
out = Path(__file__).parent / "direct_only_1977_typed.json"
out.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved {len(rows)} rows to {out}")
