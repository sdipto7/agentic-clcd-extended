# NiCad Over Generated Algorithms

This baseline evaluates NiCad on the generated algorithm summaries from the
Claude algorithm pipeline:

```text
output/algo_based/algorithms_claude-3-haiku_xlcost.json
output/algo_based/algorithms_claude-3-haiku_codenet.json
```

NiCad is normally a source-code near-miss clone detector, not a natural-language
algorithm-summary comparator. To make the generated algorithms readable by
NiCad, this adapter converts each Java-side and Python-side algorithm summary
into a small Python function containing the algorithm lines as string
assignments. NiCad is then run at `functions py` granularity.

A pair is predicted as clone only when NiCad reports the two algorithm functions
from the same original pair as a clone pair.

## Setup

Install Open-NiCad separately, or make `nicad` available on PATH. NiCad 7 requires OpenTxl 11+ and Turing+ (`tpc`) before `make` succeeds. The adapter also checks this local path, which has been cloned in this repo:

```text
baselines/nicad_algorithms/official/Open-NiCad/bin/nicad
```

Official Open-NiCad repository:

```text
https://github.com/CordyJ/Open-NiCad
```

Local setup after installing OpenTxl and Turing+:

```bash
cd baselines/nicad_algorithms/official/Open-NiCad
make
cd -
```

## Run

Prepare only:

```bash
python baselines/nicad_algorithms/run_nicad_algorithms.py --dataset xlcost --prepare-only --force
python baselines/nicad_algorithms/run_nicad_algorithms.py --dataset codenet --prepare-only --force
```

Run NiCad and convert predictions:

```bash
bash baselines/nicad_algorithms/run_nicad_algorithms.sh xlcost
bash baselines/nicad_algorithms/run_nicad_algorithms.sh codenet
```

If NiCad is not on PATH:

```bash
python baselines/nicad_algorithms/run_nicad_algorithms.py --dataset xlcost --nicad-root /path/to/Open-NiCad --force
python baselines/nicad_algorithms/run_nicad_algorithms.py --dataset codenet --nicad-root /path/to/Open-NiCad --force
```

Evaluate:

```bash
python evaluate.py --file output/nicad_algorithms/results_nicad_algorithms_xlcost_full.csv
python evaluate.py --file output/nicad_algorithms/results_nicad_algorithms_codenet_full.csv
```
