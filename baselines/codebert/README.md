# CodeBERT Cosine Baseline

This folder runs a CodeBERT cosine-similarity baseline for this project.

Model:

- `microsoft/codebert-base`

Protocol:

1. Load raw calibration data from `raw_data/java_xl.json` or `raw_data/java_cn.json`.
2. Exclude the final Java-Python evaluation pairs that appear in `data/`.
3. Exclude raw Java-Python rows entirely, because in the current repository they are exactly the final evaluation pairs.
4. Use the remaining raw Java-X pairs for calibration only.
5. Split those raw calibration pairs into deterministic stratified train/validation sets.
6. Encode both snippets with CodeBERT and compute cosine similarity.
7. Tune the cosine threshold on the raw calibration train split and report it on the raw validation split.
8. Test once on the full final Java-Python data: 1200 XLCoST pairs or 600 CodeNet pairs.
9. Write `pair_id,ground_truth,predicted_label` under `output/codebert/`.

This does not fine-tune CodeBERT weights. The only learned parameter is the cosine decision threshold, and it is learned from `raw_data`, not from the final `data/` evaluation pairs.

## Run

```bash
bash baselines/codebert/run_codebert.sh xlcost
bash baselines/codebert/run_codebert.sh codenet
```

Outputs:

```text
output/codebert/results_codebert_xlcost_full.csv
output/codebert/results_codebert_codenet_full.csv
baselines/codebert/runs/xlcost/scores_full.csv
baselines/codebert/runs/codenet/scores_full.csv
```

Evaluate:

```bash
python evaluate.py --file output/codebert/results_codebert_xlcost_full.csv
python evaluate.py --file output/codebert/results_codebert_codenet_full.csv
```
