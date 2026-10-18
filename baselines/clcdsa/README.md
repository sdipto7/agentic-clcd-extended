# CLCDSA baseline

This folder contains a project wrapper around the official CLCDSA repository:

<https://github.com/Kawser-nerd/CLCDSA>

The wrapper uses the official Java/ANTLR listeners to extract the nine syntactic
features used by CLCDSA for Java and Python snippets, then evaluates the
paper's on-the-fly cosine-similarity variant on this project's pair datasets.

The original repository also ships old Keras model artifacts, but this project
environment does not currently have TensorFlow/Keras installed, and Python 3.13
is not a friendly target for the historical Keras code. For that reason, the
implemented result should be reported as:

`CLCDSA on-the-fly cosine, reproduced with official CLCDSA feature extractors`

Run with the paper-style fixed threshold of `0.5`:

```bash
bash baselines/clcdsa/run_clcdsa.sh xlcost
bash baselines/clcdsa/run_clcdsa.sh codenet
```

A different threshold can be supplied as the second argument. Full-set threshold
tuning is available in `run_clcdsa.py` only as a diagnostic, because tuning on
the same 1200/600 pairs used for final reporting leaks test labels.

Outputs:

```text
output/clcdsa/results_clcdsa_onthefly_xlcost_full.csv
output/clcdsa/results_clcdsa_onthefly_codenet_full.csv
baselines/clcdsa/runs/xlcost/scores_onthefly_full.csv
baselines/clcdsa/runs/codenet/scores_onthefly_full.csv
```
