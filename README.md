# Agentic-CLCD: research replication package

Implementation and report snapshots for **“Improving Cross-Language Code Clone Detection with Algorithm-Based Representation and Skill-Guided Agentic Reasoning,”** prepared for ICPC 2027.

The study compares direct source-code comparison, comparison through language-agnostic algorithms, and skill-guided agentic reasoning. The evaluation pairs contain Java code paired with Python, C++, or C#, depending on the dataset. All three pipelines use hosted language models through OpenRouter.

**Artifact availability:** the repository includes implementation, prompts, skills, baseline adapters, reports, raw inputs, processed evaluation datasets, and saved experimental outputs. A fresh Git checkout contains the files needed to recompute detection metrics without model access. Rerunning LLM inference requires an API key and provider access. Local application logs remain excluded from Git. Remaining limitations are listed under [Reproducibility considerations](#reproducibility-considerations).

## Repository structure

| Path | Purpose |
| --- | --- |
| [main.py](main.py) | Runs one pipeline, model, and dataset combination. |
| [prepare_dataset.py](prepare_dataset.py) | Normalizes the supplied XLCoST and CodeNet pair files. |
| [prepare_gptclonebench.py](prepare_gptclonebench.py) | Builds the GPTCloneBench evaluation pairs from an upstream archive. |
| [evaluate.py](evaluate.py), [evaluate_baselines.py](evaluate_baselines.py) | Evaluate saved LLM and baseline predictions, respectively. |
| [src/core/](src/core/) | Model identifiers, paths, runtime settings, and logging. |
| [src/io/](src/io/) | Dataset loading, resumption, prediction writing, and usage logging. |
| [src/inference/](src/inference/) | OpenRouter client, prompts, response parsing, and ReAct agent/tools. |
| [src/workflows/](src/workflows/) | The three detection pipelines. |
| [skills/](skills/) | Markdown instructions loaded by the agent. |
| [scripts/](scripts/) | Pooled performance, complementarity, significance, routing, and token analyses. |
| [baselines/](baselines/) | CLCDSA, CodeBERT, and NiCad adapters and their documentation. |
| [reports/](reports/) | Tracked evaluation summaries and pair-level comparison results. |
| [raw_data/](raw_data/) | Included XLCoST/CodeNet pair files and the GPTCloneBench source archive. |
| [data/](data/) | Included processed evaluation datasets. |
| [output/](output/) | Included predictions, extracted algorithms, and run-level usage CSVs. |
| `logs/` | Local application logs; excluded from Git. |

## Datasets and evaluation pairs

The evaluation subsets are project-specific pair collections, not the complete original benchmarks or their standard train/test splits.

| Dataset and original source | Input expected by this project | Evaluated language pairs | Pairs per language pair | Total pairs |
| --- | --- | --- | --- | --- |
| [XLCoST](https://github.com/reddy-lab-code-research/XLCoST) | `raw_data/java_xl.json` | Java–Python, Java–C++, Java–C# | 600 clones + 600 non-clones | 3,600 |
| [Project CodeNet](https://github.com/IBM/Project_CodeNet) | `raw_data/java_cn.json` | Java–Python, Java–C++, Java–C# | 300 clones + 300 non-clones | 1,800 |
| [GPTCloneBench](https://github.com/srlabUsask/GPTCloneBench) | `raw_data/gptclonebench/cross_language_part_2.zip` | Java–Python, Java–C# | 600 clones + 600 non-clones | 2,400 |

These counts agree with the included processed JSON files and the tracked [LLM evaluation report](reports/llm_evaluation_report_20260916_015933.csv). Each model evaluates the same pairs. Counts pooled across models represent repeated evaluations, not additional distinct pairs.

XLCoST provides parallel code across languages. CodeNet contains programming submissions grouped by problem. The project's `java_xl.json` and `java_cn.json` are already labeled pair collections: **their original selection, negative-pair construction, upstream release, and submission-level provenance are not documented by a supplied preparation script.** Downloading the original benchmarks alone does not reconstruct these two files.

### Pair format and labels

Processed datasets are JSON arrays stored as `data/xlcost.json`, `data/codenet.json`, and `data/gptclonebench.json`. Each record contains:

| Field | Meaning |
| --- | --- |
| `pair_id` | Dataset-prefixed identifier, such as `xlcost_0`. |
| `codeA`, `codeB` | Java source and the second-language source, respectively. |
| `ll1`, `ll2` | Language identifiers: `Java` and `Python`, `C++`, or `C#`. |
| `label` | Integer `1` for clone and `0` for non-clone. |
| `dataset` | `xlcost`, `codenet`, or `gptclonebench`. |
| `type` | Source label, `clone` or `nonclone`. |

Additional metadata, including `problem_id_1` and `problem_id_2`, is preserved. GPTCloneBench rows also record source archive entries and clone identifiers. [DatasetLoader](src/io/dataset_loader.py) validates the required fields and labels. The prompts define clones in terms of matching functionality and outputs; the supplied labels are used as ground truth, without executing the programs to prove equivalence.

### Preparing the inputs

The processed evaluation datasets and their raw inputs are already included. To rebuild the processed files, run the following from the repository root:

```bash
python prepare_dataset.py
python prepare_gptclonebench.py
```

`prepare_dataset.py` reads both XLCoST and CodeNet pair files. It retains rows with Java as `ll1` and Python, C++, or C# as `ll2`, converts `type` to `label`, and assigns sequential IDs. Language blocks are concatenated in that order, preserving source-file order within each block. Invalid rows encountered during normalization are skipped. The script does not create new negative pairs, sample the original benchmarks, deduplicate programs, or perform semantic validation.

For GPTCloneBench, the [upstream archive](https://github.com/srlabUsask/GPTCloneBench/blob/main/cross_language_part_2.zip) is included at `raw_data/gptclonebench/cross_language_part_2.zip`. The preparation script:

- Extracts GPT-4 Java-to-Python and Java-to-C# translations from `cross_java_to_other/Gpt4D_Clone*.java` entries.
- Checks Python snippets with `ast.parse`; C# receives only length and balanced-brace/parenthesis checks, not compilation.
- Uses seed `42`, deduplicates candidate positives by exact Java text, and samples 600 positives per language.
- Constructs negatives by pairing selected Java snippets with translations from other archive entries. Token similarity and Java method-name rules guide candidate exclusion and difficulty selection; 528 negatives per language are targeted as hard, with fallbacks when necessary. These are constructed labels, not a new human-verified negative set.
- Assigns IDs across the Python block followed by the C# block, and writes the combined dataset plus `data/gptclonebench_audit.json`. The audit is a generated output, not a tracked artifact.

Both preparation scripts rebuild their outputs. Pair IDs depend on the exact inputs, their order, and the preparation code; preserve those inputs when matching saved predictions. The additional `clcdsa` dataset key in the CLI is not part of the three-dataset reports, and no corresponding dataset or preparation workflow is supplied here.

## Detection pipelines, models, and skills

| Pipeline | Implementation and behavior |
| --- | --- |
| `direct` | [direct_workflow.py](src/workflows/direct_workflow.py): compares the two source fragments in one model call, before retries. |
| `algo_based` | [algo_based_workflow.py](src/workflows/algo_based_workflow.py): extracts each fragment's algorithm separately, then compares the two algorithms; three model calls before retries. |
| `agentic` | [agentic_workflow.py](src/workflows/agentic_workflow.py): runs a ReAct episode in which the model loads skills, chooses a comparison approach, and records a verdict. The number of model calls varies. |

The direct and algorithm-based prompts are in [src/inference/prompts.py](src/inference/prompts.py). The agent's system prompt and ReAct protocol are in [agent.py](src/inference/agentic/agent.py); its tools are in [tools.py](src/inference/agentic/tools.py). `compare_and_decide` displays content for the agent's own reasoning; it does not run a separate detector or execute the source code.

The four skills are [direct clone detection](skills/clone_detection_direct/SKILL.md), [algorithm extraction](skills/algorithm_extraction/SKILL.md), [algorithm comparison](skills/clone_detection_algo/SKILL.md), and [result recording](skills/result_record/SKILL.md). Skill metadata is discovered from YAML front matter, and bodies are exposed through `load_skill`. Other tools list skills, display comparisons, record algorithms, and write results.

The exact aliases accepted by `main.py` and their configured OpenRouter identifiers are:

| CLI alias | Model identifier |
| --- | --- |
| `deepseek-v3.2` | `deepseek/deepseek-v3.2` |
| `llama-3.3-nemotron` | `nvidia/llama-3.3-nemotron-super-49b-v1.5` |
| `claude-3-haiku` | `anthropic/claude-3-haiku` |
| `gpt-4o-mini` | `openai/gpt-4o-mini-2024-07-18` |

[constants.py](src/core/constants.py) and [llm.py](src/inference/llm.py) define the configuration: temperature `0.0`, request timeout 120 seconds, a 0.5-second pacing delay, and at most 10 agent iterations. Direct and algorithm-based calls receive one retry after an API exception, with a 3-second delay. The client disables its own automatic retries. No model seed or explicit output-token limit is configured. Temperature zero does not guarantee identical responses across providers or reruns.

## Installation

The source uses Python 3.10+ syntax. Run the following in a Bash-compatible shell from the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

[requirements.txt](requirements.txt) includes LangChain, its classic agent package, the OpenAI-compatible client, dotenv, YAML, pandas, and tqdm. Versions are not pinned, and no tested environment lockfile is supplied. Recreating the historical dependency environment therefore requires additional version information.

For inference, set `OPENROUTER_API_KEY` in the environment or in a root `.env` file:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
```

The client uses `https://openrouter.ai/api/v1`. New inference requires network access, provider availability, and a funded account; the commands below can incur charges. Offline evaluation of saved predictions needs no API key or model access. The evaluation scripts use the Python standard library and project configuration.

## Rerunning LLM inference

The supplied datasets are ready for inference. The included predictions are also used for resumption, so completed pairs will be skipped. For a fresh inference run, preserve the supplied outputs separately and use an empty output location at the configured paths. After configuring model access, run one combination:

```bash
python main.py --pipeline direct --model deepseek-v3.2 --dataset xlcost
python main.py --pipeline algo_based --model deepseek-v3.2 --dataset xlcost
python main.py --pipeline agentic --model deepseek-v3.2 --dataset xlcost
```

To rerun the complete three-dataset, four-model, three-pipeline configuration:

```bash
for dataset in xlcost codenet gptclonebench; do
  for model in deepseek-v3.2 llama-3.3-nemotron claude-3-haiku gpt-4o-mini; do
    for pipeline in direct algo_based agentic; do
      python main.py --pipeline "$pipeline" --model "$model" --dataset "$dataset"
    done
  done
done
```

`main.py` processes all records in the selected dataset; it has no language or sample-size option. Use `python main.py --help` to inspect supported choices after installing dependencies.

Runs resume by `pair_id`: existing non-`ERROR` predictions are skipped, and stale `ERROR` rows are removed before retrying. [result_helper.py](src/io/result_helper.py) implements this behavior. To conduct an independent rerun, preserve existing outputs separately and use an empty output location at the configured paths; otherwise old and new predictions can be combined. Do not reuse pair IDs for different source pairs or alter prompts/models midway through a resumed run.

### Baseline experiments

All four adapters restrict evaluation to Java–Python pairs. Their results should be compared with the corresponding language subset of the LLM results.

| Adapter | Implemented variant and prerequisites |
| --- | --- |
| [CLCDSA](baselines/clcdsa/README.md) | Official nine-feature extractors with on-the-fly cosine similarity and threshold `0.5`; not the historical trained Keras model. Requires Java/Javac, the official source tree, and its ANTLR jar. |
| [CodeBERT](baselines/codebert/README.md) | `microsoft/codebert-base` embeddings and a calibrated cosine threshold; no weight fine-tuning. Requires NumPy, scikit-learn, PyTorch, Transformers, and model download access. |
| [NiCad on algorithms](baselines/nicad_algorithms/README.md) | Converts Claude's saved algorithm summaries to Python functions before NiCad comparison. Requires those summaries and a working NiCad/OpenTxl/Turing+ installation. |
| [NiCad on source](baselines/nicad_source/README.md) | Runs with the Java grammar by default. Standard NiCad uses one language grammar, so this is a cross-language applicability control. |

The CLCDSA and Open-NiCad source directories are recorded as Git links, but this checkout has no `.gitmodules` file. If their contents are absent in a fresh checkout, obtain the official repositories at the expected paths before following their setup instructions:

```bash
git clone https://github.com/Kawser-nerd/CLCDSA baselines/clcdsa/official
git -C baselines/clcdsa/official checkout 25f3357068e2c4bcb6b5a7a8eae93cae7cc0c020
git clone https://github.com/CordyJ/Open-NiCad baselines/nicad_algorithms/official/Open-NiCad
git -C baselines/nicad_algorithms/official/Open-NiCad checkout 7a90d11795a7fe585282e25b7fa8d9964f202965
```

The baseline Python dependencies are not all in the root requirements file:

```bash
python -m pip install numpy scikit-learn torch transformers
```

After installing the native prerequisites described in the baseline READMEs and generating the Claude algorithm summaries, the implemented runs are:

```bash
for dataset in xlcost codenet gptclonebench; do
  python baselines/clcdsa/run_clcdsa.py --dataset "$dataset" --threshold 0.5
  python baselines/codebert/run_codebert.py --dataset "$dataset"
  python baselines/nicad_algorithms/run_nicad_algorithms.py --dataset "$dataset" --force
  python baselines/nicad_source/run_nicad_source.py --dataset "$dataset" --language java --force
done
```

CodeBERT defaults are seed `42`, calibration train/validation ratio `0.80`, batch size `16`, and maximum input length `512`. Calibration excludes final evaluation code pairs and all Java–Python rows, using other-language rows from the raw XLCoST/CodeNet files; GPTCloneBench uses its Java–C# rows. This is pair-level exclusion, not a guarantee of program- or problem-level separation. The threshold is selected on the calibration training partition. CLCDSA's optional full-set threshold tuning is diagnostic and should not replace the fixed-threshold reported protocol.

## Reproducing metrics from saved predictions

These commands perform no LLM inference and can be run directly on a fresh checkout using the included `output/` result files. The tracked reports provide reference summaries for comparison. Additional baseline score tables and run logs are available under `baselines/`.

```bash
# All saved LLM result files; baseline directories are excluded automatically.
python evaluate.py

# Alternatively, evaluate a single saved result file.
python evaluate.py --file output/direct/results_deepseek-v3.2_xlcost.csv

# The twelve baseline result files selected in evaluate_baselines.py.
python evaluate_baselines.py
```

`evaluate.py --results_dir output` is the explicit equivalent of its default directory scan. Accuracy is `(TP + TN) / N`; precision, recall, and F1 use `CLONE` as the positive class. `ERROR` and unrecognized predictions count as incorrect: false negatives for true clones and false positives for true non-clones. Grouped metrics are calculated after summing confusion counts, rather than averaging rounded per-run percentages.

The tracked reports cover all three datasets, while several analysis scripts default to only XLCoST and CodeNet. Specify all datasets explicitly:

```bash
# Match the C#-only scope of the tracked pooled-performance snapshot.
python scripts/analyze_pooled_performance.py --datasets xlcost codenet gptclonebench --languages 'C#'

python scripts/compare_direct_vs_algo_based.py --datasets xlcost codenet gptclonebench
python scripts/run_mcnemar_pooled.py --datasets xlcost codenet gptclonebench
python scripts/analyze_agentic_strategy.py --datasets xlcost codenet gptclonebench
python scripts/analyze_token_cost.py --datasets xlcost codenet gptclonebench
```

Omit `--languages` from pooled performance to include all languages. The complementarity, McNemar, and routing scripts also accept a language filter, for example `--languages Python`. Token usage is logged per invocation and cannot be filtered by language. Optional pooled baseline metrics are produced by `python scripts/analyze_pooled_baseline_performance.py --datasets xlcost codenet gptclonebench`.

| Tracked report | Generator and required saved inputs |
| --- | --- |
| [LLM CSV](reports/llm_evaluation_report_20260916_015933.csv), [detailed analysis](reports/llm_detailed_analysis_20260916_015933.txt) | `evaluate.py`; LLM result CSVs. |
| [Baseline CSV](reports/baseline_evaluation_report.csv) | `evaluate_baselines.py`; the selected baseline result CSVs. |
| [Pooled performance](reports/pooled_performance_analysis_20260916_020610.txt) | `analyze_pooled_performance.py`; LLM result CSVs, filtered to C# for this snapshot. |
| [Complementarity](reports/direct_algo_complementarity_20260916_015934.txt), [pair details](reports/direct_correct_algo_wrong_pairs_20260916_015934.csv) | `compare_direct_vs_algo_based.py`; matching direct/algorithm result CSVs. |
| [McNemar comparisons](reports/mcnemar_pooled_all_models.csv) | `run_mcnemar_pooled.py`; matched result CSVs from all three pipelines. |
| [Strategy selection](reports/agentic_strategy_analysis_20260916_015935.txt) | `analyze_agentic_strategy.py`; agentic result CSVs and algorithm JSON files. |
| [Token analysis](reports/token_cost_analysis_20260916_015934.txt) | `analyze_token_cost.py`; result CSVs and run-level token-usage CSVs. |

The analysis generators in the table are under `scripts/` unless otherwise noted. Regeneration uses the current saved outputs and code; timestamps and formatting may differ from the snapshots. Most reports receive new timestamped filenames. `evaluate_baselines.py` and the McNemar script overwrite their fixed report paths, so retain the supplied snapshots separately when comparing them.

## Expected outputs and troubleshooting

| Output | Contents |
| --- | --- |
| `output/<pipeline>/results_<model>_<dataset>.csv` | `pair_id,ll1,ll2,ground_truth,predicted_label`; ground truth is `0`/`1`, prediction is `CLONE`, `NOT_CLONE`, or `ERROR`. |
| `output/<pipeline>/algorithms_<model>_<dataset>.json` | Saved algorithm pairs keyed by `pair_id`, with `java_algorithm` and `ll2_algorithm`; produced by the algorithm pipeline and eligible agentic episodes. |
| `output/<pipeline>/token_usage.csv` | Invocation-level model/dataset, pair count, elapsed seconds, status, successful requests, and prompt/completion/total tokens. |
| `logs/experiment.log` | Application diagnostics and per-pair outcomes. |
| `reports/` | Evaluation CSVs and text analyses. |

Prediction CSVs do not retain the confidence or reasoning requested in prompts. Token logging runs in a `finally` block on normal completion, handled exceptions, and keyboard interruption; it is not guaranteed after forced process termination. On interrupted runs, the logged `pairs` field is the number scheduled for that invocation, not necessarily the number completed. Logs have no console handler; workflow progress normally appears through tqdm.

For missing datasets, check the exact input/output paths above; `prepare_dataset.py` only prepares XLCoST and CodeNet. For missing skills, check the four `skills/*/SKILL.md` files and their YAML metadata. For API or parsing failures, inspect `logs/experiment.log`, confirm the key and configured model access, and rerun to retry `ERROR` rows. Changes to prompts, parsing, or model configuration constitute a different experimental configuration.

## Reproducibility considerations

- **Dataset provenance and versioning:** the exact input pair files, processed datasets, predictions, algorithm JSONs, and usage CSVs are included. Detection metrics can be recomputed from these files. Reconstructing the original XLCoST/CodeNet pair selection from upstream benchmarks still requires the missing selection procedure and source-release information. Record the repository commit used for replication; a separate checksum manifest and archival release identifier are not documented here.
- **Environment and model provenance:** dependency versions, the original execution environment, and hosted provider routing/revisions are not recorded in a complete run manifest. Exact LLM replay is not guaranteed, even with the same model aliases and temperature.
- **Agent behavior:** the system prompt still describes Java–Python, while each task message supplies the actual second language. Missing `write_result` calls can fall back to a verdict parsed from final-answer text. Routing analysis infers algorithm use from saved algorithm records; it does not have a separate explicit route field or full per-pair action trace. These details are part of the current implementation.
- **Estimated costs:** the supplied token report says all model/dataset combinations include estimates, and describes caps based on assumed ratios. Its “raw totals” wording should not be interpreted as complete measured usage. The analysis script aggregates existing logs; a reconstruction script for the historical estimates is not supplied. Exact cost validation requires the original uncapped measurements.
- **Statistical scope:** pooled McNemar tests treat each model–pair evaluation as independent, although models share the same pairs. A recorded p-value of zero reflects floating-point underflow. The reports do not provide repeated-run uncertainty or establish that extracted algorithms are semantically faithful.
- **Audit and paper mapping:** the current local inputs do not include the GPTCloneBench audit JSON that the preparation script would generate. No paper source or mapping from every paper table/figure to an artifact command is supplied; the commands above reproduce the scopes identifiable from the tracked reports.

## Dataset attribution and licensing

Use the original dataset sources and cite their publications using the citation information provided upstream. Dataset licenses are separate from tool licenses and this project's implementation.

- **XLCoST:** the [upstream repository license](https://github.com/reddy-lab-code-research/XLCoST/blob/main/LICENSE) is Apache-2.0. Check the notices accompanying the downloaded corpus and underlying source programs before redistributing a processed pair collection.
- **CodeNet:** IBM's [CodeNet sample documentation](https://github.com/CODAIT/MAX-CodeNet-Code-Complexity/blob/master/samples/README.md) identifies its dataset samples as CDLA-Permissive-2.0. The [CodeNet tooling repository](https://github.com/IBM/Project_CodeNet/blob/main/LICENSE) has a separate Apache-2.0 license. Retain the applicable dataset license and attribution when obtaining or distributing data.
- **GPTCloneBench:** the [upstream license statement](https://github.com/srlabUsask/GPTCloneBench#license) specifies Creative Commons Attribution–NonCommercial–NoDerivatives and discusses the benchmark database and derivatives. Do not assume the locally transformed subsets and constructed negatives can be redistributed; check the applicable terms and obtain clarification or permission from the maintainers where required.

No root project license is supplied in this checkout. Third-party code, model weights, native tools, and datasets retain their respective licenses; this README does not grant additional redistribution rights.
