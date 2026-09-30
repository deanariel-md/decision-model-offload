# results

Aggregate results behind the numbers of the paper and its supplement. They hold no item text, no reply text and no
record-level rows. The per-call outputs and the model responses they are computed from are available from the
corresponding author on reasonable request; given them (under `results/`, as each script reads them) and the rebuilt
records, the commands below rewrite every file here. Some of the commands also write files that are not listed here:
further analyses the paper does not report, and intermediate files that later commands read.

| Files | Contents | Written by |
|---|---|---|
| `reference_performance.json`, `cohort_flow.md` | Outcome references and cohort flow (NHANES) | `scripts/fit_reference.py`, `scripts/build_cohort.py` |
| `eval/analysis.json`, `wide/analysis.json` | Ten-year risk: response to one-field edits, baseline prediction | `scripts/analyze.py eval`, `wide` |
| `eval/exposure.json`, `eval/cost_speed.json` | Masked-field recovery, cost and speed | `scripts/exposure_diagnostic.py`, `cost_speed.py eval` |
| `eval/characteristics.json`, `eval/supplement_descriptives.json` | Characteristics of the record sets, descriptive tables | `scripts/characteristics.py`, `supplement_descriptives.py eval` |
| `wide/arm1_extras.json` | Further descriptive analyses | `scripts/arm1_extras.py` |
| `wide_cheaper/analysis.json` | The smaller LLMs (free-plan models, MedGemma and Gemma) on the wide set | `scripts/run_wide_cheaper.py analyse` |
| `wide_docuse/analysis.json`, `timing_sample.json` | Ten-year risk with Jev in the recommended form | `scripts/docuse/analyze_risk.py`, `run_risk.py --timing` |
| `eicu/analysis.json`, `eicu/exposure.json` | Death in hospital, eICU demo (the main systems, and the free-plan and open-weight models in `secondary_prediction`) | `scripts/analyze_eicu.py eval`, `eicu_exposure.py` |
| `eicu_aki/analysis.json`, `analysis_definitions.json`, `supplement_*.json` | Kidney-injury stage, eICU demo | `scripts/run_categorical.py eicu_aki analyze`, `eicu_aki_supplement.py` |
| `arm2/analysis.json`, `arm2/timing_sample.json` | Patient questions | `scripts/run_categorical.py arm2 analyze`, `arm2 timing` |
| `arm2_ext/analysis*.json`, `arm2_ext/timing_sample.json` | Board-examination questions (all, next-step subgroup, other option count) | `scripts/run_categorical.py arm2_ext analyze`, `arm2_ext timing` |
| `arm3/supplement_names.json` | Colon cancer stage: repeatability on the 40 repeated reports | `scripts/run_categorical.py arm3 analyze`, `arm3_supplement.py` |
| `arm3_confuser/analysis.json`, `misread.json` | Reports with a misleading feature | `scripts/run_categorical.py arm3_confuser analyze`, `arm3_confuser_misread.py` |
| `arm3_docuse/documented/analysis_definitions.json`, `arm3_docuse/timing_sample.json` | Colon cancer stage with Jev in the recommended form | `scripts/docuse/analyze_staging.py`, `run_staging.py --timing` |
| `arm3_nonregional_sites/summary.json` | The same on reports with a non-regional node the notes do not name | `scripts/nonregional/run_nonregional.py --analyze` |
| `arm3_llm_forms/analysis.json` | Colon cancer stage on the pool of 240 reports: tokens and list cost of every system in every form of the question (with `summaries/staging_pool240.json`'s counts of correct answers, the cost per correct answer) | `scripts/llm_forms/analyze.py` |
| `summaries/staging_pool240.json` | Colon cancer stage on the pool of 240 reports: every system in every form of the question, Jev against each LLM, the pairing at full confidence, clinical decisions | `scripts/summaries/staging_pool240.py` |
| `summaries/pairing_*.json` | Jev kept at each confidence threshold and an LLM otherwise: every task, all ten LLMs | `scripts/summaries/pairing_ten.py` |
| `summaries/confidence_table.json` | Answers and correctness by confidence level, every task, form and system | `scripts/summaries/confidence_table.py` |
| `summaries/workflow_groups.json` | The recommended-form pairing by report group, with the pool of 240 and its time shares | `scripts/summaries/workflow_groups.py` |
| `summaries/noninferiority_ten.json` | Jev against each of the ten LLMs, every task: difference in accuracy (points) with its 95% and wider intervals, and the verdict | computed from the stored answers; not written by a script in this repository |
| `summaries/*.json` (the rest) | Summary numbers across tasks | `scripts/summaries/` (below) |

The summary scripts read the files above and each other; run them from the repository root in this order:

```
python scripts/summaries/arm1_baseline_summary.py
python scripts/summaries/arm1_edit_summary.py
python scripts/summaries/arm1_extras_summary.py
python scripts/summaries/arm1_hybrid_summary.py
python scripts/summaries/arm1_secondary_summary.py
python scripts/summaries/arm2_summary.py
python scripts/summaries/arm2_further_summary.py
python scripts/summaries/arm2_ext_summary.py
python scripts/summaries/arm3_summary.py
python scripts/summaries/arm3_confusion.py
python scripts/summaries/confidence_flag.py
python scripts/summaries/sens_spec.py
python scripts/summaries/timing_summary.py
python scripts/summaries/eicu_summary.py
python scripts/summaries/aki_summary.py
python scripts/summaries/hybrid_all.py
python scripts/summaries/chain.py
python scripts/summaries/wide_cheaper_structured.py --root . --cheaper results/wide_cheaper
python scripts/summaries/docuse_summary.py
python scripts/summaries/hybrid_explore.py
python scripts/summaries/docuse_summary.py
python scripts/summaries/risk_classic.py --docuse .
python scripts/summaries/auroc_free_open_ci.py --cheaper results/wide_cheaper
python scripts/summaries/clinical_metrics.py
python scripts/summaries/workflow_checks.py --docuse . --heldout .
python scripts/summaries/workflow_timing.py
python scripts/summaries/reference_missing.py
python scripts/summaries/staging_pool240.py
python scripts/summaries/pairing_ten.py
python scripts/summaries/confidence_table.py
python scripts/summaries/workflow_groups.py --docuse . --heldout . --pool240 data/arm3_pool/pool240.csv
python scripts/summaries/pool_summary.py
python scripts/llm_forms/analyze.py
```

The last six read the pool of 240 reports (`data/arm3_pool/pool240.csv`, written by `scripts/make_pool240.py`) and
the answers of every system in every form (`results/arm3_llm_forms/calls.parquet`, written by
`scripts/llm_forms/collect.py`); `scripts/llm_forms/analyze.py` checks its counts against `pool_summary.json` and
`workflow_groups.json`. `pairing_ten.py` and `confidence_table.py` leave out the kidney-injury task when
`data/eicu_aki/items.csv` is absent.

`cheap_family.py` (then `cheap_family.py --risk`) runs before `scripts/docuse/analyze_risk.py`, which reads its output.

Removed from these files before release: the identifiers of the items Jev answered in each hybrid
(`hybrid_share/jev_items`), the identifiers of individual items and reports where an output listed them, the rows of
the eICU outcome references fitted on the credentialed database, the cost of the free-plan and open-weight models in
`eicu/analysis.json` (not reported), and, in `arm3_llm_forms/analysis.json`, every part other than the systems, forms,
prices and costs (the rest repeats `summaries/staging_pool240.json`).
