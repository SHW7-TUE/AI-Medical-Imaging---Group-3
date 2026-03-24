# Open Project: Prompt Sensitivity in Zero-Shot Medical Image Classification

## Research Question
**How sensitive is zero-shot classification performance of medical vision-language models to clinically related terminology in prompt wording?**



## Hypothesis
This project tests three related hypotheses:

1. Zero-shot performance is **not invariant** to clinically related wording changes in prompts.
2. A medically adapted model (BiomedCLIP) is, on average, **more robust** to wording variation than a general-domain CLIP model.
3. Prompt sensitivity differs by task, with some histopathology concepts showing larger AUC variation than others.



## Study Design (Scientific Overview)

### Core design principle
For each pathology task, the experiment keeps **one negative prompt fixed** and varies only the **positive prompt wording**.

This isolates the effect of positive terminology changes and avoids confounding from changing both sides of the prompt pair.

### Models compared
- **General model:** `ViT-B-16`, pretrained on `laion2b_s34b_b88k` (OpenCLIP)
- **Medical model:** `hf-hub:mgbam/OpenCLIP-BiomedCLIP-Finetuned`

### Dataset
- **Open-MELON-VL-2.5K** (HuggingFace)
- **Test split only** is used for this experiment.

### Tasks
- `arch_junctional_activity`
- `arch_pagetoid_spread`
- `cyto_prominent_nucleoli`
- `cyto_pigment`
- `cyto_melanophages`
- `ctx_inflammation`
- `ctx_fibrosis`
- `ctx_necrosis`

For each task:
- one fixed negative prompt,
- one baseline positive prompt,
- multiple clinically related positive prompt variants.

Baseline prompts are the natural positive counterparts of the fixed negative prompts and are used for delta-AUC comparisons.



## Tasks and Prompt Strategy
The script defines all prompts in `TaskConfig` objects in `open_project_prompt_sensitivity.py`.

Example (`arch_junctional_activity`):
- Fixed negative: **"No junctional component"**
- Baseline positive: **"Junctional component"**
- Other variants:
  - "Junctional activity"
  - "Melanocytic proliferation at the dermoepidermal junction"
  - "Dermoepidermal junction involvement"

This same pattern is applied to all eight tasks.



## Technical Pipeline (What the Script Does)
The script follows an end-to-end analysis pipeline in one standalone file:

`open_project_prompt_sensitivity.py`

### 1) Global setup
- Sets deterministic seeds.
- Defines constants (batch size, bootstrap sample count, output folder).
- Defines task/prompt configurations.

### 2) Model loading
- Loads both OpenCLIP models.
- Loads matching tokenizers and preprocessing transforms.

### 3) Dataset handling
- Loads Open-MELON test split from HuggingFace.
- Uses a generic dataset wrapper for binary metadata fields.
- Filters missing and `"unknown"` labels.
- Safely parses string labels (e.g., `"true"`, `"false"`, `"1"`, `"0"`, `"yes"`, `"no"`).
- Converts images to RGB and applies model-specific preprocessing.

### 4) Image embedding extraction
- Computes normalized image embeddings with each model’s image encoder.
- Embeddings are computed once per task/model and reused across prompt variants.

### 5) Zero-shot classification
For each prompt pair `[negative_prompt, positive_prompt]`:
- encodes text prompts,
- normalizes text embeddings,
- computes image-text similarities,
- applies softmax over the two prompts,
- uses `P(positive_prompt)` as the prediction score.

### 6) Baseline comparison and prompt sensitivity
For each task and model:
- computes ROC-AUC for every positive variant,
- identifies baseline positive prompt AUC,
- computes:
  - `delta_auc = auc_variant - auc_baseline`
  - `abs_delta_auc = |delta_auc|`

Prompt sensitivity is operationalized as changes in AUC when only positive terminology changes.

### 7) Text similarity analysis
- Computes text embeddings for positive prompts.
- Computes cosine similarity between each variant and the baseline positive prompt.
- Supports analysis of whether semantic closeness in text space tracks behavioral stability in zero-shot performance.

### 8) Statistical uncertainty (bootstrap)
- Uses bootstrap resampling on the test set.
- Computes confidence intervals for:
  - ROC-AUC,
  - delta AUC vs baseline.
- Uses **paired bootstrap** for delta AUC: baseline and variant are evaluated on the same resampled indices, making comparisons fair.

### 9) Output generation
- Saves detailed per-prompt CSV results.
- Saves per-task and overall sensitivity summary CSVs.
- Saves per-task ROC plots.
- Saves per-task bar plots of `|delta_auc|`.
- Saves a scatter plot of cosine similarity vs `|delta_auc|`.



## Why These Methodological Choices?

### Why fix the negative prompt?
To isolate the effect of positive wording. If both prompts changed simultaneously, attribution of performance differences would be ambiguous.

### Why use a baseline positive prompt?
The baseline is the direct positive counterpart to the fixed negative prompt, providing a consistent reference for `delta_auc`.

### What does prompt sensitivity mean here?
A model is prompt-sensitive if ROC-AUC changes when clinically related positive wording changes (with the negative prompt fixed).

### Why ROC-AUC?
ROC-AUC is threshold-independent, standard for binary classification, and consistent with prior assignment methodology.

### Why cosine similarity?
It quantifies semantic closeness in text embedding space, allowing us to test whether “closer wording” corresponds to more stable classification performance.

### Why bootstrap confidence intervals?
Bootstrap CIs provide uncertainty estimates with minimal parametric assumptions and help assess whether observed differences may reflect sampling variability.



## Output Files
All outputs are saved under:

`results/open_project_prompt_sensitivity/`

### Main results table
- `prompt_sensitivity_results.csv`

Contains per-task/per-prompt rows including:
- prompts and baseline flags,
- AUCs for both models,
- baseline AUCs,
- delta and absolute delta AUCs,
- bootstrap CIs,
- cosine similarity to baseline,
- sample count and class fraction.

### Sensitivity summaries
- `sensitivity_summary_per_task.csv`
- `sensitivity_summary_overall.csv`

### Figures
- `roc_<task>.png`
- `abs_delta_auc_<task>.png`
- `scatter_similarity_vs_abs_delta_auc.png`



## Interpretation Guide
When reading outputs:

1. **Within each task**, compare AUC across positive variants.
   - Larger spread => higher prompt sensitivity.
2. **Check delta AUC vs baseline**.
   - Near-zero delta => robust to that wording change.
   - Large |delta| => sensitive to wording change.
3. **Compare models**.
   - Lower mean `|delta_auc|` suggests greater robustness.
4. **Use CIs for uncertainty context**.
   - Wide intervals indicate higher uncertainty.
5. **Use similarity-vs-|ΔAUC| scatter**.
   - Helps assess whether semantic closeness predicts behavioral stability.



   ## Repository Overview
- `open_project_prompt_sensitivity.py` — complete standalone experiment pipeline.
- `README.md` — project description, design rationale, and usage guide.
- `assignment_2_VLM_Handin.ipynb` — source notebook used as the original pipeline reference.



## How to Run
This project is designed as a **standalone Python script** (not a notebook).

You can run it by opening and running `open_project_prompt_sensitivity.py` in an IDE/editor.

The script entry point is:

```python
if __name__ == "__main__":
    main()
```

No `launch.json` or special CLI arguments are required.



## Notes and Limitations
- Results are conditioned on the Open-MELON test split and available labels after filtering unknown/missing values.
- Prompt variants are clinically related but not perfectly synonymous; wording can change scope and emphasis.
- Zero-shot behavior depends jointly on image representation quality and language alignment.
- This study evaluates sensitivity to wording changes, not full supervised upper-bound performance.
