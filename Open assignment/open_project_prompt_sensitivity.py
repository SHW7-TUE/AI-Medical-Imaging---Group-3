"""
Open Project: Prompt sensitivity in zero-shot medical image classification.

Study question:
How sensitive is zero-shot classification performance of medical vision-language
models to clinically related terminology in prompt wording?

This script reuses the same core OpenCLIP/BiomedCLIP workflow used in
assignment_2_VLM_Handin.ipynb.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# Keep dependency imports explicit with basic fallback handling.
try:
    import open_clip
    from datasets import load_dataset
    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError as exc:
    raise ImportError(
        "Missing dependency. Please install required packages used in Assignment 2: "
        "datasets, open_clip_torch, scikit-learn, matplotlib, pillow, torch."
    ) from exc


# =========================
# Global configuration
# =========================
SEED = 42
BATCH_SIZE = 32
BOOTSTRAP_SAMPLES = 500
RESULTS_DIR = Path("results/open_project_prompt_sensitivity")


@dataclass(frozen=True)
class TaskConfig:
    """
    Configuration bundle for one binary pathology classification task.

    Attributes:
        field_name: Metadata field name in Open-MELON-VL-2.5K.
        negative_prompt: Fixed negative prompt used for all variants in the task.
        baseline_positive_prompt: Reference positive prompt used for delta-AUC.
        positive_prompts: List of positive prompt variants to evaluate.
    """

    field_name: str
    negative_prompt: str
    baseline_positive_prompt: str
    positive_prompts: List[str]


TASKS: List[TaskConfig] = [
    TaskConfig(
        field_name="arch_junctional_activity",
        negative_prompt="No junctional component",
        baseline_positive_prompt="Junctional component",
        positive_prompts=[
            "Junctional activity",
            "Junctional component",
            "Melanocytic proliferation at the dermoepidermal junction",
            "Dermoepidermal junction involvement",
        ],
    ),
    TaskConfig(
        field_name="arch_pagetoid_spread",
        negative_prompt="No pagetoid spread",
        baseline_positive_prompt="Pagetoid spread",
        positive_prompts=[
            "Pagetoid spread",
            "Pagetoid intraepidermal spread",
            "Suprabasal spread of atypical cells",
            "Upward spread of atypical cells",
        ],
    ),
    TaskConfig(
        field_name="cyto_prominent_nucleoli",
        negative_prompt="Inconspicuous nucleoli",
        baseline_positive_prompt="Conspicuous nucleoli",
        positive_prompts=[
            "Prominent nucleoli",
            "Conspicuous nucleoli",
            "Large nucleoli",
            "Distinct nucleoli",
        ],
    ),
    TaskConfig(
        field_name="cyto_pigment",
        negative_prompt="No pigment",
        baseline_positive_prompt="Melanin pigment",
        positive_prompts=[
            "Melanin pigment",
            "Pigmented cells",
            "Pigment deposition",
            "Extracellular pigment",
        ],
    ),
    TaskConfig(
        field_name="cyto_melanophages",
        negative_prompt="No melanophages",
        baseline_positive_prompt="Melanophages",
        positive_prompts=[
            "Melanophages",
            "Dermal melanophages",
            "Melanin-containing macrophages",
            "Heavily pigmented extravasated melanophages",
        ],
    ),
    TaskConfig(
        field_name="ctx_inflammation",
        negative_prompt="No inflammation",
        baseline_positive_prompt="Inflammation",
        positive_prompts=[
            "Inflammation",
            "Perilesional dermal inflammation",
            "Inflammatory infiltrate",
            "Inflammatory response",
        ],
    ),
    TaskConfig(
        field_name="ctx_fibrosis",
        negative_prompt="No fibrosis",
        baseline_positive_prompt="Fibrosis",
        positive_prompts=[
            "Fibrosis",
            "Stromal fibrosis",
            "Desmoplastic stroma",
            "Fibrotic stroma",
            "Collagen deposition",
        ],
    ),
    TaskConfig(
        field_name="ctx_necrosis",
        negative_prompt="No necrosis",
        baseline_positive_prompt="Necrosis",
        positive_prompts=[
            "Necrosis",
            "Tumor necrosis",
            "Necrotic tissue",
            "Areas of necrosis",
            "Coagulative necrosis",
        ],
    ),
]


def set_seed(seed: int = SEED) -> None:
    """
    Set deterministic random seeds for reproducibility.

    Args:
        seed: Integer seed applied to Python, NumPy, and Torch RNGs.

    Returns:
        None. RNG state is configured in-place.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_models(device: str) -> Dict[str, Dict[str, Any]]:
    """
    Load both vision-language models and corresponding tokenizers/preprocessing.

    Args:
        device: Torch device string (e.g., \"cpu\" or \"cuda\").

    Returns:
        Dictionary keyed by model name (\"general\", \"medical\") with:
        - model: OpenCLIP model object
        - preprocess: image preprocessing transform
        - tokenizer: tokenizer callable for prompt text
    """
    models: Dict[str, Dict[str, Any]] = {}

    try:
        model_general, _, preprocess_val_general = open_clip.create_model_and_transforms(
            "ViT-B-16",
            pretrained="laion2b_s34b_b88k",
            device=device,
        )
        tokenizer_general = open_clip.get_tokenizer("ViT-B-16")

        models["general"] = {
            "model": model_general,
            "preprocess": preprocess_val_general,
            "tokenizer": tokenizer_general,
        }
    except Exception as exc:
        raise RuntimeError("Failed to load general OpenCLIP model.") from exc

    try:
        model_medical, _, preprocess_val_medical = open_clip.create_model_and_transforms(
            "hf-hub:mgbam/OpenCLIP-BiomedCLIP-Finetuned",
            pretrained=None,
            device=device,
        )
        tokenizer_medical = open_clip.get_tokenizer(
            "hf-hub:mgbam/OpenCLIP-BiomedCLIP-Finetuned"
        )

        models["medical"] = {
            "model": model_medical,
            "preprocess": preprocess_val_medical,
            "tokenizer": tokenizer_medical,
        }
    except Exception as exc:
        raise RuntimeError("Failed to load medical BiomedCLIP model.") from exc

    return models


class HFDatasetBinaryField(Dataset):
    """
    Generic HF dataset wrapper for binary metadata fields.

    Filters out samples where field is missing/None/"unknown", then returns:
    (preprocessed_image_tensor, label_string)

    This class centralizes safe label parsing so all downstream evaluation
    uses consistent binary labels.
    """

    def __init__(self, hf_ds, preprocess, field_name: str, pos_label: str, neg_label: str):
        self.ds = hf_ds
        self.preprocess = preprocess
        self.field_name = field_name
        self.pos_label = pos_label
        self.neg_label = neg_label

        self.valid_indices: List[int] = []
        for i in range(len(self.ds)):
            value = self.ds[i].get(field_name, None)
            if value is None:
                continue
            if isinstance(value, str) and value.lower() == "unknown":
                continue
            self.valid_indices.append(i)

        print(
            f"[{self.field_name}] Kept {len(self.valid_indices)} / {len(self.ds)} samples"
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _parse_binary_value(self, value: Any) -> bool:
        """
        Safely parse one raw metadata value into a boolean label.

        Args:
            value: Raw field value from the dataset.

        Returns:
            Boolean class indicator (True for positive, False for negative).

        Raises:
            ValueError: If a string value is not a recognized binary token.

        Notes:
            We parse strings explicitly because bool(\"False\") == True in Python,
            which can silently corrupt labels if not handled carefully.
        """
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes"}:
                return True
            if normalized in {"false", "0", "no"}:
                return False
            raise ValueError(
                f"Unexpected string value for field '{self.field_name}': {value!r}. "
                "Expected one of: true/false, 1/0, yes/no."
            )
        return bool(value)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        ex = self.ds[self.valid_indices[idx]]
        img = ex["image"].convert("RGB")

        value = ex.get(self.field_name, None)
        is_positive = self._parse_binary_value(value)
        label = self.pos_label if is_positive else self.neg_label

        x_img = self.preprocess(img)
        return x_img, label


def collate_fn(batch):
    """
    Stack image tensors and keep labels as a Python list.

    Args:
        batch: List of (image_tensor, label_string) tuples.

    Returns:
        Tuple of:
        - batched tensor of shape [B, C, H, W]
        - list of label strings of length B
    """
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, list(labels)


@torch.no_grad()
def compute_image_embeddings(dataloader: DataLoader, model, device: str) -> Tuple[np.ndarray, List[str]]:
    """
    Encode all images from a dataloader into normalized CLIP image embeddings.

    Args:
        dataloader: DataLoader yielding (image_batch, label_list).
        model: OpenCLIP-compatible model exposing encode_image(...).
        device: Torch device string.

    Returns:
        Tuple:
        - NumPy array of L2-normalized image embeddings [N, D]
        - list of label strings aligned with embedding rows
    """
    img_embs = []
    labels: List[str] = []

    model.eval()

    for imgs, labs in dataloader:
        imgs = imgs.to(device, non_blocking=True)

        feats = model.encode_image(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        img_embs.append(feats.cpu().numpy())
        labels.extend(labs)

    return np.vstack(img_embs), labels


@torch.no_grad()
def zeroshot_clip_scores(model, tokenizer, img_emb_np: np.ndarray, prompts: List[str], device: str) -> np.ndarray:
    """
    Compute two-prompt zero-shot probabilities and return positive-class scores.

    Args:
        model: OpenCLIP-compatible model exposing encode_text(...).
        tokenizer: Matching tokenizer for the model.
        img_emb_np: Image embeddings [N, D].
        prompts: Two prompts [negative_prompt, positive_prompt].
        device: Torch device string.

    Returns:
        NumPy array of positive-class probabilities with length N.
    """
    model.eval()

    tokens = tokenizer(prompts).to(device)
    txt_emb = model.encode_text(tokens)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

    img_emb = torch.from_numpy(img_emb_np).to(device)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

    logits = img_emb @ txt_emb.T
    probs = torch.softmax(logits, dim=1)

    return probs[:, 1].detach().cpu().numpy()


@torch.no_grad()
def compute_text_embeddings(model, tokenizer, prompts: List[str], device: str) -> np.ndarray:
    """
    Compute normalized text embeddings for prompt strings.

    Args:
        model: OpenCLIP-compatible model.
        tokenizer: Matching tokenizer callable.
        prompts: Prompt strings to embed.
        device: Torch device string.

    Returns:
        NumPy array of L2-normalized text embeddings [len(prompts), D].
    """
    model.eval()
    tokens = tokenizer(prompts).to(device)
    txt_emb = model.encode_text(tokens)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    return txt_emb.detach().cpu().numpy()


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Scalar cosine similarity in [-1, 1].
    """
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-12
    return float(np.dot(vec_a, vec_b) / denom)


def make_bootstrap_indices(y_true: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate bootstrap resampling indices with class-validity checks.

    Args:
        y_true: Binary label array.
        n_bootstrap: Number of valid bootstrap samples to generate.
        rng: NumPy random generator.

    Returns:
        Integer index array with shape [n_bootstrap, N].

    Notes:
        ROC-AUC is undefined when a sample contains only one class, so those
        sampled index sets are discarded and redrawn.
    """
    n = len(y_true)
    sampled = []
    while len(sampled) < n_bootstrap:
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        sampled.append(idx)
    return np.stack(sampled, axis=0)


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    bootstrap_indices: np.ndarray,
    ci_alpha: float = 0.95,
) -> Tuple[float, float, np.ndarray]:
    """
    Compute bootstrap confidence interval for ROC-AUC.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted positive-class scores.
        bootstrap_indices: Precomputed bootstrap index matrix [B, N].
        ci_alpha: Confidence level (default 0.95).

    Returns:
        Tuple of (ci_low, ci_high, auc_bootstrap_distribution).
    """
    auc_boot = np.array(
        [roc_auc_score(y_true[idx], y_score[idx]) for idx in bootstrap_indices],
        dtype=np.float64,
    )
    lo_q = (1.0 - ci_alpha) / 2.0
    hi_q = 1.0 - lo_q
    ci_low, ci_high = np.quantile(auc_boot, [lo_q, hi_q])
    return float(ci_low), float(ci_high), auc_boot


def bootstrap_delta_auc_ci(
    auc_variant_boot: np.ndarray,
    auc_baseline_boot: np.ndarray,
    ci_alpha: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute paired bootstrap confidence interval for delta AUC.

    Args:
        auc_variant_boot: Bootstrap AUCs for a prompt variant.
        auc_baseline_boot: Bootstrap AUCs for the baseline prompt using the
            same bootstrap resamples.
        ci_alpha: Confidence level.

    Returns:
        Tuple (delta_ci_low, delta_ci_high).
    """
    delta_boot = auc_variant_boot - auc_baseline_boot
    lo_q = (1.0 - ci_alpha) / 2.0
    hi_q = 1.0 - lo_q
    ci_low, ci_high = np.quantile(delta_boot, [lo_q, hi_q])
    return float(ci_low), float(ci_high)


def task_bootstrap_seed(base_seed: int, task_name: str) -> int:
    """
    Derive a deterministic task-specific bootstrap seed.

    Args:
        base_seed: Global experiment seed.
        task_name: Task identifier (metadata field name).

    Returns:
        Deterministic integer seed stable across runs.
    """
    digest = hashlib.sha256(f"{base_seed}:{task_name}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def evaluate_task(
    task: TaskConfig,
    ds_test,
    models: Dict[str, Dict[str, Any]],
    device: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
    """
    Evaluate one task across all positive prompt variants and both models.

    Args:
        task: Task configuration containing prompts and metadata field.
        ds_test: HuggingFace test split.
        models: Loaded model bundle from load_models(...).
        device: Torch device string.

    Returns:
        Tuple of:
        - list of row dictionaries for CSV export
        - ROC storage dictionary for plotting

    Important design detail:
        The negative prompt is fixed within each task. Only the positive prompt
        changes, so measured AUC differences reflect prompt wording sensitivity.

    """
    print(f"\n=== Evaluating task: {task.field_name} ===")
    print(f"Fixed negative prompt: {task.negative_prompt}")
    print(f"Baseline positive prompt: {task.baseline_positive_prompt}")

    per_model_embeddings: Dict[str, np.ndarray] = {}
    labels_reference: List[str] | None = None
    text_similarities: Dict[str, Dict[str, float]] = {"general": {}, "medical": {}}
    model_auc_boot: Dict[str, Dict[str, np.ndarray]] = {"general": {}, "medical": {}}

    # Build test embeddings once per model for this task.
    # This avoids recomputing image features for each positive prompt.
    for model_name, model_bundle in models.items():
        dataset_test = HFDatasetBinaryField(
            hf_ds=ds_test,
            preprocess=model_bundle["preprocess"],
            field_name=task.field_name,
            pos_label="present",
            neg_label="absent",
        )
        loader_test = DataLoader(
            dataset_test,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

        emb_test, labels_test = compute_image_embeddings(
            dataloader=loader_test,
            model=model_bundle["model"],
            device=device,
        )
        per_model_embeddings[model_name] = emb_test

        if labels_reference is None:
            labels_reference = labels_test

        print(
            f"[{task.field_name}] {model_name:<7} test embeddings shape: {emb_test.shape}"
        )

        # Text-space semantics: compare each positive variant embedding to the
        # baseline positive embedding (same model's text encoder space).
        unique_prompts = [task.baseline_positive_prompt] + [
            p for p in task.positive_prompts if p != task.baseline_positive_prompt
        ]
        text_emb = compute_text_embeddings(
            model=model_bundle["model"],
            tokenizer=model_bundle["tokenizer"],
            prompts=unique_prompts,
            device=device,
        )
        prompt_to_emb = {p: text_emb[i] for i, p in enumerate(unique_prompts)}
        baseline_emb = prompt_to_emb[task.baseline_positive_prompt]
        for prompt in task.positive_prompts:
            text_similarities[model_name][prompt] = cosine_similarity(
                baseline_emb, prompt_to_emb[prompt]
            )

    if labels_reference is None:
        raise RuntimeError(f"No valid labels found for task {task.field_name}")

    y_true = np.array([1 if x == "present" else 0 for x in labels_reference], dtype=np.int32)
    n_samples = int(len(y_true))
    n_positive = int(np.sum(y_true))
    positive_fraction = float(np.mean(y_true)) if n_samples > 0 else float("nan")
    # Diagnostic class-count print for quick sanity checks during runtime.
    print(f"[{task.field_name}] Positive samples: {n_positive} / {n_samples}")

    rows: List[Dict[str, Any]] = []
    roc_store: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {"general": {}, "medical": {}}
    # Paired bootstrap indices are generated once per task and reused for all
    # prompt variants to keep delta-AUC comparisons paired and fair.
    rng = np.random.default_rng(task_bootstrap_seed(SEED, task.field_name))
    bootstrap_indices = make_bootstrap_indices(y_true, BOOTSTRAP_SAMPLES, rng)

    # First pass: compute per-prompt AUC/ROC and AUC bootstrap CI.
    for pos_prompt in task.positive_prompts:
        prompts = [task.negative_prompt, pos_prompt]
        print(f"  -> Positive prompt: {pos_prompt}")

        for model_name, model_bundle in models.items():
            y_score = zeroshot_clip_scores(
                model=model_bundle["model"],
                tokenizer=model_bundle["tokenizer"],
                img_emb_np=per_model_embeddings[model_name],
                prompts=prompts,
                device=device,
            )

            auc = roc_auc_score(y_true, y_score)
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_ci_low, auc_ci_high, auc_boot = bootstrap_auc_ci(
                y_true=y_true,
                y_score=y_score,
                bootstrap_indices=bootstrap_indices,
            )
            model_auc_boot[model_name][pos_prompt] = auc_boot

            roc_store[model_name][pos_prompt] = {
                "fpr": fpr,
                "tpr": tpr,
                "auc": auc,
                "auc_ci_low": auc_ci_low,
                "auc_ci_high": auc_ci_high,
            }
            print(
                f"     {model_name:<7} AUC: {auc:.4f} "
                f"(95% CI {auc_ci_low:.4f}-{auc_ci_high:.4f})"
            )

    # Second pass: add baseline-referenced delta metrics into export rows.
    for pos_prompt in task.positive_prompts:
        is_baseline = pos_prompt == task.baseline_positive_prompt

        row = {
            "task": task.field_name,
            "negative_prompt": task.negative_prompt,
            "baseline_positive_prompt": task.baseline_positive_prompt,
            "positive_prompt": pos_prompt,
            "is_baseline_prompt": bool(is_baseline),
            "n_samples": n_samples,
            "positive_fraction": positive_fraction,
        }

        for model_name in models.keys():
            auc_variant = float(roc_store[model_name][pos_prompt]["auc"])
            auc_variant_boot = model_auc_boot[model_name][pos_prompt]
            baseline_prompt = task.baseline_positive_prompt
            auc_baseline = float(roc_store[model_name][baseline_prompt]["auc"])
            auc_baseline_boot = model_auc_boot[model_name][baseline_prompt]

            if is_baseline:
                # Baseline compared to itself has zero delta by definition.
                delta_auc = 0.0
                abs_delta_auc = 0.0
                delta_ci_low, delta_ci_high = 0.0, 0.0
            else:
                delta_auc = auc_variant - auc_baseline
                abs_delta_auc = abs(delta_auc)
                delta_ci_low, delta_ci_high = bootstrap_delta_auc_ci(
                    auc_variant_boot=auc_variant_boot,
                    auc_baseline_boot=auc_baseline_boot,
                )

            row[f"auc_{model_name}"] = auc_variant
            row[f"baseline_auc_{model_name}"] = auc_baseline
            row[f"delta_auc_{model_name}"] = float(delta_auc)
            row[f"abs_delta_auc_{model_name}"] = float(abs_delta_auc)
            row[f"auc_{model_name}_ci_low"] = float(
                roc_store[model_name][pos_prompt]["auc_ci_low"]
            )
            row[f"auc_{model_name}_ci_high"] = float(
                roc_store[model_name][pos_prompt]["auc_ci_high"]
            )
            row[f"delta_auc_{model_name}_ci_low"] = float(delta_ci_low)
            row[f"delta_auc_{model_name}_ci_high"] = float(delta_ci_high)
            row[f"cosine_similarity_to_baseline_positive_{model_name}"] = float(
                text_similarities[model_name][pos_prompt]
            )

        rows.append(row)

    return rows, roc_store


def plot_task_rocs(task: TaskConfig, roc_store: Dict[str, Dict[str, Dict[str, np.ndarray]]], out_dir: Path) -> None:
    """
    Save ROC plot for one task including both models and all prompt variants.

    Args:
        task: Task configuration used to label the figure.
        roc_store: ROC/AUC data generated by evaluate_task(...).
        out_dir: Destination directory for plot file.

    Returns:
        None. Writes PNG to disk.
    """
    plt.figure(figsize=(10, 8))

    for prompt, roc_obj in roc_store["general"].items():
        plt.plot(
            roc_obj["fpr"],
            roc_obj["tpr"],
            linewidth=1.8,
            linestyle="-",
            label=f"General | {prompt} (AUC={roc_obj['auc']:.3f})",
        )

    for prompt, roc_obj in roc_store["medical"].items():
        plt.plot(
            roc_obj["fpr"],
            roc_obj["tpr"],
            linewidth=1.8,
            linestyle="--",
            label=f"Medical | {prompt} (AUC={roc_obj['auc']:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k:", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"Prompt sensitivity ROC — {task.field_name}\n"
        f"Fixed negative prompt: '{task.negative_prompt}'"
    )
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()

    out_path = out_dir / f"roc_{task.field_name}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved ROC plot: {out_path}")


def save_results_table(rows: List[Dict[str, Any]], out_dir: Path) -> Path:
    """
    Save the full per-experiment results table as CSV.

    Args:
        rows: List of per-prompt result dictionaries.
        out_dir: Output directory.

    Returns:
        Path to the saved CSV file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    expected_cols = [
        "task",
        "negative_prompt",
        "baseline_positive_prompt",
        "positive_prompt",
        "is_baseline_prompt",
        "auc_general",
        "auc_medical",
        "baseline_auc_general",
        "baseline_auc_medical",
        "delta_auc_general",
        "delta_auc_medical",
        "abs_delta_auc_general",
        "abs_delta_auc_medical",
        "auc_general_ci_low",
        "auc_general_ci_high",
        "auc_medical_ci_low",
        "auc_medical_ci_high",
        "delta_auc_general_ci_low",
        "delta_auc_general_ci_high",
        "delta_auc_medical_ci_low",
        "delta_auc_medical_ci_high",
        "cosine_similarity_to_baseline_positive_general",
        "cosine_similarity_to_baseline_positive_medical",
        "n_samples",
        "positive_fraction",
    ]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[expected_cols]

    csv_path = out_dir / "prompt_sensitivity_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results CSV: {csv_path}")
    return csv_path


def plot_similarity_vs_abs_delta(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    """
    Plot cosine similarity to baseline prompt versus absolute delta AUC.

    Args:
        rows: List of per-prompt result rows.
        out_dir: Output directory for saved figure.

    Returns:
        None. Writes PNG to disk.
    """
    df = pd.DataFrame(rows)
    df = df[df["is_baseline_prompt"] == False].copy()  # noqa: E712
    if df.empty:
        return

    plt.figure(figsize=(10, 8))

    # One consistent color per task (same color across both models).
    task_order = [
        "arch_junctional_activity",
        "arch_pagetoid_spread",
        "cyto_prominent_nucleoli",
        "cyto_pigment",
        "cyto_melanophages",
        "ctx_inflammation",
        "ctx_fibrosis",
        "ctx_necrosis",
    ]
    color_map = {task: plt.cm.tab10(i) for i, task in enumerate(task_order)}
    task_label_map = {
        "arch_junctional_activity": "junctional activity",
        "arch_pagetoid_spread": "pagetoid spread",
        "cyto_prominent_nucleoli": "prominent nucleoli",
        "cyto_pigment": "pigment",
        "cyto_melanophages": "melanophages",
        "ctx_inflammation": "inflammation",
        "ctx_fibrosis": "fibrosis",
        "ctx_necrosis": "necrosis",
    }

    marker_map = {"general": "o", "medical": "^"}

    # Plot points task-by-task so color encodes task and marker encodes model.
    for task_name in task_order:
        task_df = df[df["task"] == task_name]
        if task_df.empty:
            continue

        plt.scatter(
            task_df["cosine_similarity_to_baseline_positive_general"],
            task_df["abs_delta_auc_general"],
            color=color_map[task_name],
            marker=marker_map["general"],
            s=70,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
        )
        plt.scatter(
            task_df["cosine_similarity_to_baseline_positive_medical"],
            task_df["abs_delta_auc_medical"],
            color=color_map[task_name],
            marker=marker_map["medical"],
            s=75,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
        )

    plt.xlabel("Cosine similarity to baseline positive prompt", fontsize=11)
    plt.ylabel("|ΔAUC| vs baseline", fontsize=11)
    plt.title("Prompt semantics vs performance sensitivity", fontsize=13)
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.6)

    # Separate legends: one for task colors, one for model marker shapes.
    task_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color_map[task],
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=8,
            label=task_label_map[task],
        )
        for task in task_order
        if task in set(df["task"].tolist())
    ]
    model_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_map["general"],
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=8,
            label="General",
        ),
        Line2D(
            [0],
            [0],
            marker=marker_map["medical"],
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=8,
            label="Medical",
        ),
    ]

    ax = plt.gca()
    legend_tasks = ax.legend(
        handles=task_handles,
        title="Task",
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(legend_tasks)
    ax.legend(
        handles=model_handles,
        title="Model",
        loc="upper left",
        bbox_to_anchor=(0.01, 0.69),
        borderaxespad=0.0,
        fontsize=9,
        title_fontsize=9,
    )

    out_path = out_dir / "scatter_similarity_vs_abs_delta_auc.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved similarity-vs-|delta AUC| scatter: {out_path}")


def summarize_sensitivity(rows: List[Dict[str, Any]], out_dir: Path) -> Tuple[Path, Path]:
    """
    Save sensitivity summary CSVs as point estimates.

    Args:
        rows: List of per-prompt result rows.
        out_dir: Output directory.

    Returns:
        Tuple of paths:
        - per-task mean |delta AUC| CSV
        - overall model-level mean |delta AUC| CSV

    Notes:
        Sensitivity is defined as mean absolute delta AUC across non-baseline
        prompts. No uncertainty intervals are added for these summaries.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    df_non_baseline = df[df["is_baseline_prompt"] == False].copy()  # noqa: E712

    per_task = (
        df_non_baseline.groupby("task", as_index=False)[
            ["abs_delta_auc_general", "abs_delta_auc_medical"]
        ]
        .mean()
        .rename(
            columns={
                "abs_delta_auc_general": "mean_abs_delta_auc_general",
                "abs_delta_auc_medical": "mean_abs_delta_auc_medical",
            }
        )
    )

    per_task_path = out_dir / "sensitivity_summary_per_task.csv"
    per_task.to_csv(per_task_path, index=False)
    print(f"Saved per-task sensitivity CSV: {per_task_path}")

    overall = pd.DataFrame(
        [
            {
                "model": "general",
                "overall_mean_abs_delta_auc": float(
                    df_non_baseline["abs_delta_auc_general"].mean()
                ),
            },
            {
                "model": "medical",
                "overall_mean_abs_delta_auc": float(
                    df_non_baseline["abs_delta_auc_medical"].mean()
                ),
            },
        ]
    )
    overall_path = out_dir / "sensitivity_summary_overall.csv"
    overall.to_csv(overall_path, index=False)
    print(f"Saved overall sensitivity CSV: {overall_path}")

    return per_task_path, overall_path


def plot_task_abs_delta_auc(task: TaskConfig, rows: List[Dict[str, Any]], out_dir: Path) -> None:
    """
    Save per-task bar plot of absolute delta AUC for both models.

    Args:
        task: Task configuration.
        rows: Full result row list; function filters to this task.
        out_dir: Output directory for saved plot.

    Returns:
        None. Writes PNG to disk.
    """
    task_rows = [r for r in rows if r["task"] == task.field_name and not r["is_baseline_prompt"]]
    if not task_rows:
        return

    prompts = [r["positive_prompt"] for r in task_rows]
    y_gen = [r["abs_delta_auc_general"] for r in task_rows]
    y_med = [r["abs_delta_auc_medical"] for r in task_rows]
    x = np.arange(len(prompts))
    width = 0.38

    plt.figure(figsize=(11, 6))
    plt.bar(x - width / 2, y_gen, width=width, label="General")
    plt.bar(x + width / 2, y_med, width=width, label="Medical")
    plt.xticks(x, prompts, rotation=20, ha="right")
    plt.ylabel("|ΔAUC| vs baseline")
    plt.title(f"Prompt sensitivity (abs delta AUC) — {task.field_name}")
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / f"abs_delta_auc_{task.field_name}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved abs delta AUC plot: {out_path}")


def main() -> None:
    """
    Run the full standalone prompt-sensitivity experiment pipeline.

    Steps:
        1) set seed and output directory
        2) load test split and both models
        3) evaluate each task (AUCs, deltas, CIs, similarities)
        4) save plots and CSV summaries

    Returns:
        None. Side effects are saved outputs in RESULTS_DIR.
    """
    # Reproducible global setup.
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load evaluation data.
    print("Loading Open-MELON-VL-2.5K test split from HuggingFace...")
    try:
        ds_test = load_dataset("MartiHan/Open-MELON-VL-2.5K", split="test")
    except Exception as exc:
        raise RuntimeError("Failed to load dataset MartiHan/Open-MELON-VL-2.5K") from exc

    # Load both vision-language models (general and medical).
    print("Loading models...")
    models = load_models(device)

    all_rows: List[Dict[str, Any]] = []
    all_results: Dict[str, Any] = {}

    for task in TASKS:
        # Core evaluation loop: fixed negative prompt + varying positive prompts.
        rows, roc_store = evaluate_task(task=task, ds_test=ds_test, models=models, device=device)
        all_rows.extend(rows)
        all_results[task.field_name] = {
            "negative_prompt": task.negative_prompt,
            "baseline_positive_prompt": task.baseline_positive_prompt,
            "rows": rows,
            "roc": roc_store,
        }
        plot_task_rocs(task=task, roc_store=roc_store, out_dir=RESULTS_DIR)
        plot_task_abs_delta_auc(task=task, rows=rows, out_dir=RESULTS_DIR)

    # Export machine-readable tables and summary figures for analysis/reporting.
    save_results_table(all_rows, RESULTS_DIR)
    summarize_sensitivity(all_rows, RESULTS_DIR)
    plot_similarity_vs_abs_delta(all_rows, RESULTS_DIR)
    print("\nDone. Results are in:", RESULTS_DIR.resolve())


if __name__ == "__main__":
    main()