# NiCad Direct Source Baseline

Runs Open-NiCad directly on the final Java/Python pair source files from `data/`.

This is a direct-source experiment, unlike `baselines/nicad_algorithms`, which runs NiCad on LLM-generated algorithm summaries. Open-NiCad cross-system mode accepts one language grammar at a time, so this baseline keeps native extensions and runs with a selected NiCad language (`java` by default). This documents the expected limitation: standard NiCad is not a cross-language Java/Python clone detector.

## Run

```bash
bash baselines/nicad_source/run_nicad_source.sh xlcost java
bash baselines/nicad_source/run_nicad_source.sh codenet java
python evaluate.py --file output/nicad_source/results_nicad_source_xlcost_java_full.csv
python evaluate.py --file output/nicad_source/results_nicad_source_codenet_java_full.csv
```

Optional Python-grammar failure/control run:

```bash
bash baselines/nicad_source/run_nicad_source.sh xlcost py
bash baselines/nicad_source/run_nicad_source.sh codenet py
```
