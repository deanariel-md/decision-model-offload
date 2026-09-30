# Decision models can offload structured clinical tasks from large language models

Code for a study that compares Jev, a non-generative decision model (TypeSafe, `typesafe/jev-1.13`), with widely used
chatbots on three kinds of clinical question: a probability (ten-year risk of death; death in hospital in intensive
care), a choice (the next step for a patient's message; board-examination questions) and an ordered level (colon cancer
stage; acute kidney injury stage).

The repository holds the code that builds the records and prompts, queries each system and analyses the replies, the
prompt templates (`prompts/`) and the aggregate results the paper reports (`results/`). It holds no data: the code
rebuilds the survey and intensive care records and the board-examination questions from their public sources. The
patient messages, the pathology reports and the model responses are available from the corresponding author on
reasonable request.

## Install

Python 3.12 (tested on Windows 11). PowerShell:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt; pip install -e . --no-deps
python -m pytest -q
```

In bash, activate with `source .venv/bin/activate` and use forward slashes in paths. The tests run on synthetic data
and simulated systems, offline.

## Data

| Source | Where to get it | Built by |
|---|---|---|
| NHANES 1999–2008 and the Linked Mortality Files | National Center for Health Statistics (public) | `scripts/download_nhanes.py`, `scripts/build_cohort.py` |
| NHANES III (reference sensitivity only) | National Center for Health Statistics (public) | `scripts/build_nhanes3.py` |
| eICU Collaborative Research Database demo 2.0.1 | PhysioNet, Open Database License | `scripts/build_eicu.py demo`, `scripts/make_eicu_aki.py` |
| eICU Collaborative Research Database 2.0 (outcome reference only) | PhysioNet, credentialed access | `scripts/build_eicu.py full`, `scripts/fit_eicu_reference.py` |
| MedQA and Medbullets | their original sources | `scripts/build_arm2_ext_items.py --download` |
| Patient messages, pathology reports | the corresponding author, on reasonable request | supplied by the user |

Downloads are checked against `config/nhanes_sha256.txt` and `config/arm2_ext/items_manifest.json`. No record from the
credentialed eICU database is sent to any model; it is used only to fit the outcome reference.

## Running

Every command runs from the repository root. Scripts that call a system need keys in `.env` (names in
`.env.example`); each prints its planned calls and projected cost and asks before sending anything. The systems are
hosted services that their developers can change, so a rerun measures them as they are on the day it runs. Raw replies
are saved under `runs/` before they are parsed.

Commands are for PowerShell; in bash use forward slashes.

**Ten-year risk of death (NHANES).** Build the records and the outcome references, query the systems, analyse:

```powershell
python scripts\download_nhanes.py; python scripts\build_cohort.py; python scripts\build_nhanes3.py
python scripts\fit_reference.py; python scripts\make_states.py eval; python scripts\refit_references.py eval
python scripts\crossfit_reference.py; python scripts\make_states.py wide
python scripts\cost_estimate.py all                        # planned calls and projected cost; nothing is sent
python scripts\run_calls.py eval --batch submit; python scripts\run_calls.py eval --batch collect; python scripts\run_calls.py eval
python scripts\run_calls.py wide --batch submit; python scripts\run_calls.py wide --batch collect; python scripts\run_calls.py wide
python scripts\timing_sample.py; python scripts\route_bridge.py; python scripts\exposure_diagnostic.py
python scripts\hybrid_arm1.py run; python scripts\run_wide_cheaper.py submit    # then collect, sync, pair, build
python scripts\analyze.py eval; python scripts\analyze.py wide
python scripts\characteristics.py; python scripts\cost_speed.py eval; python scripts\supplement_descriptives.py eval
python scripts\arm1_extras.py; python scripts\hybrid_arm1.py analyze; python scripts\run_wide_cheaper.py analyse
python scripts\family_simulation.py --seed0 9000 --reps 200 --R 50 --B 500 --n 12000   # the critical-value factor
```

**Death in hospital (eICU demo).** The outcome reference is fitted on the full database, which needs credentialed
access; only demo records are sent to the systems:

```powershell
python scripts\build_eicu.py demo --src <demo folder>; python scripts\make_eicu_states.py eval --baseline
python scripts\build_eicu.py full --src <eICU-CRD 2.0 folder> --demo-src <demo folder>; python scripts\fit_eicu_reference.py
python scripts\run_eicu.py estimate; python scripts\run_eicu.py eval; python scripts\eicu_exposure.py
python scripts\analyze_eicu.py eval
```

**Next step for a patient's message and board-examination questions.** Put the patient messages in `data\arm2\items.csv`
(`config\arm2.yaml`); the board-examination questions are downloaded and checked against the manifest:

```powershell
python scripts\build_arm2_ext_items.py --download
python scripts\run_categorical.py arm2 plan               # then: batch submit, batch collect, run, timing
python scripts\run_categorical.py arm2 analyze
python scripts\run_categorical.py arm2_ext plan           # likewise; arm2_ext analyze
```

**Colon cancer stage and kidney-injury stage.** Put the reports in `data\arm3\items.csv` and
`data\arm3\confuser_items.csv` (`config\arm3.yaml`, `config\arm3_confuser.yaml`); the kidney-injury records are built
from the eICU demo:

```powershell
python scripts\run_categorical.py arm3 plan               # then: batch submit, batch collect, run, timing, analyze
python scripts\run_categorical.py arm3_confuser plan      # likewise; then arm3_confuser_misread.py
python scripts\arm3_supplement.py
python scripts\make_eicu_aki.py --demo <demo folder>
python scripts\run_categorical.py eicu_aki plan           # likewise for eicu_aki_side
python scripts\eicu_aki_supplement.py; python scripts\eicu_aki_on_arrival.py
```

**Jev given structured input** (colon cancer stage, ten-year risk and death in hospital), with the reworded,
non-regional and held-out report sets (each read from its folder under `data\`). The structure-only version reads the
stored answers of the structured runs, so it runs after them; the intensive care items need the eICU demo built above:

```powershell
python scripts\docuse\run_staging.py --dry-run            # then --full, --repeats, --timing
python scripts\docuse\analyze_staging.py
python scripts\docuse\run_risk.py --dry-run               # then --full, --timing
python scripts\summaries\cheap_family.py; python scripts\summaries\cheap_family.py --risk; python scripts\docuse\analyze_risk.py
python scripts\docuse\run_reworded.py --build; python scripts\docuse\run_reworded.py --run --yes; python scripts\docuse\run_reworded.py --analyze
python scripts\nonregional\run_nonregional.py --dry-run; python scripts\nonregional\run_nonregional.py --run --yes; python scripts\nonregional\run_nonregional.py --analyze
python scripts\heldout\run.py systems --yes; python scripts\heldout\run.py tables; python scripts\heldout\analyze.py
python scripts\docuse\run_structure_only.py --dry-run --set all   # then --full --set all --yes; --analyze --set all
python scripts\gapfill\run.py dry-run; python scripts\gapfill\run.py full --item eicu_structured --yes; python scripts\gapfill\analyze.py --only gapfill_eicu_structured
```

**Every LLM in every form of the staging question**, on a pool of 240 reports (all 120 difficult reports and 120 of
the ordinary ones, drawn at random). Each LLM request is built from Jev's stored request for the same form and report,
so this runs after Jev's runs above; an answer already stored for a character-identical request is reused:

```powershell
python scripts\make_pool240.py
python scripts\arm3_jev_staircase.py build; python scripts\arm3_jev_staircase.py run --yes; python scripts\arm3_jev_staircase.py tables
python scripts\arm3_jev_parts.py build; python scripts\arm3_jev_parts.py run --yes; python scripts\arm3_jev_parts.py tables
python scripts\llm_forms\build_requests.py; python scripts\llm_forms\audit.py
python scripts\llm_forms\run.py --group openrouter --mode dry     # then --mode full; and --group hf --mode full
python scripts\llm_forms\collect.py
```

**Summary numbers**, once the analyses above have run: `results\README.md` gives the command that writes each file
under `results\summaries\`, in the order they run.

`python scripts\dry_run.py` and `python scripts\run_categorical.py <task> dryrun --out <folder>` run the whole chain on
synthetic data with simulated systems, offline.

## Results

`results/` holds the aggregate outputs behind every number in the paper and its supplement, with no item text, reply
text or record-level rows; `results/README.md` lists each file and the script that writes it. `scripts/summaries/`
writes the summary numbers.

## Figures

Figures 1 and 2 and Supplementary Fig. 1 are drawn from the files in `results/` alone:

```powershell
python scripts\figures\figure1.py; python scripts\figures\figure2.py; python scripts\figures\supplementary_figures.py
```

Each writes into `figures\` (`--out` to change it).

## Layout

- `src/jevity/`: the package (records, edits, references, requests, parsers, calls, analyses).
- `scripts/`: one entry point per step; `scripts/figures/` draws the figures.
- `config/`: every setting the code reads; no secrets. `config/docuse_risk.yaml` and `config/gapfill_eicu.yaml` give the
  full source of every named category of the recommended form.
- `prompts/`: the prompt templates, with `{braces}` for the parts that change.
- `results/`: aggregate results.
- `tests/`: tests on synthetic data.

## Licence and citation

The code is released under the MIT License (`LICENSE`). Cite as in `CITATION.cff`. The eICU demo is available under the Open Database License (Johnson et al.,
PhysioNet, doi 10.13026/4mxk-na84); the results derived from it are shared under the same terms.
