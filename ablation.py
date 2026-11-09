# =============================================================================
# ablation.py
# =============================================================================
"""Unified benchmark for feature and attention ablation.

Entry points:
  - run_feature_ablation
  - run_attention_deletion_test
  - run_attention_retrain_ablation
  - run_all_ablations
"""
from __future__ import annotations

import argparse
import gc
import random
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_ind, wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dl_training import ACIN, ACINTraining, MLPTraining, BL_MLP
from ml_training import AutoML

warnings.filterwarnings("ignore", message="Using a target size")

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
BENCH_DIR = PROJECT_ROOT / "benchmark"
FEAT_BENCH_DIR = BENCH_DIR / "feature_ablation"
DEL_BENCH_DIR = BENCH_DIR / "attention_deletion"
RETRAIN_BENCH_DIR = BENCH_DIR / "attention_retrain"

FEAT_TASK_CSV = FEAT_BENCH_DIR / "task_results.csv"
DL_INPUTS_CSV = PROJECT_ROOT / "dl_inputs.csv"
BASELINE_ACIN = PROJECT_ROOT / "dl_models" / "ACIN.pth"

N_DESC, N_KRFP = 35, 4860
TRAIN_SIZE, RANDOM_STATE = 0.75, 42
DROPOUT = 0.3
EPOCHS, EARLY_STOP, LR = 1000, 40, 1e-3

FEAT_ABLATION_SEEDS = [42, 43, 44, 45, 46]

FEAT_TASK_DEFS = [
    ("property-only (35)", "MLP"),
    ("property-only (35)", "CatBoost"),
    ("fingerprint-only (4860)", "MLP"),
    ("fingerprint-only (4860)", "CatBoost"),
    ("concatenated (4895)", "MLP"),
    ("concatenated (4895)", "CatBoost"),
    ("concat (4895) + attention", "ACIN"),
]

DELETION_ALPHAS = (0.05, 0.10, 0.20, 0.30, 0.50)
DELETION_N_RANDOM = 100
DELETION_SEED = 123

RETRAIN_ALPHAS = (0.10, 0.20, 0.30)
RETRAIN_N_RANDOM = 3
RETRAIN_N_TRAIN_SEEDS = 3


# ---------------------------------------------------------------------------
# Shared Helpers
# ---------------------------------------------------------------------------
def pick_device(pref: str | None = "auto") -> str:
    if pref is None or pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _reset_rng(seed: int = RANDOM_STATE) -> None:
    _set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _setup_plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 7,
        "font.weight": "normal",
        "axes.titlesize": 8,
        "axes.titleweight": "normal",
        "axes.labelsize": 8,
        "axes.labelweight": "normal",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.color": "black",
        "ytick.color": "black",
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "patch.linewidth": 0.6,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.5,
        "legend.borderaxespad": 0.5,
        "text.color": "black",
        "axes.labelcolor": "black",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    return plt


def build_bundle(device: str) -> dict:
    ds = pd.read_csv(DL_INPUTS_CSV)
    x_all, y_all = ds.iloc[:, :-1], ds.iloc[:, -1].reset_index(drop=True)

    idx_tr, idx_te = train_test_split(
        np.arange(len(y_all)),
        train_size=TRAIN_SIZE, shuffle=True, random_state=RANDOM_STATE,
    )

    feat_scaler = StandardScaler().fit(x_all.iloc[idx_tr, :N_DESC].values)
    krfp_scaler = StandardScaler().fit(x_all.iloc[idx_tr, N_DESC:].values)

    return {
        "feat_scaler": feat_scaler,
        "krfp_scaler": krfp_scaler,
        "all_cols": list(x_all.columns),
        "x_train_krfp_raw": torch.tensor(x_all.iloc[idx_tr, N_DESC:].values.astype(np.float32), device=device),
        "x_test_krfp_raw": torch.tensor(x_all.iloc[idx_te, N_DESC:].values.astype(np.float32), device=device),
        "x_train_feats_np": x_all.iloc[idx_tr, :N_DESC].values.astype(np.float32),
        "x_test_feats_np": x_all.iloc[idx_te, :N_DESC].values.astype(np.float32),
        "y_train": y_all.iloc[idx_tr].reset_index(drop=True),
        "y_test": y_all.iloc[idx_te].reset_index(drop=True),
        "idx_tr": idx_tr,
        "idx_te": idx_te,
    }


def load_baseline_acin(device: str) -> ACIN:
    if not BASELINE_ACIN.is_file():
        raise FileNotFoundError(f"{BASELINE_ACIN} not found.")
    model = ACIN(feat_dim=N_DESC, mol_dim=N_KRFP, drop_rate=DROPOUT).to(device)
    state = torch.load(BASELINE_ACIN, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def per_sample_scores(model: ACIN, krfp_std: torch.Tensor, feats_std: torch.Tensor) -> torch.Tensor:
    return model.attn_projector(model.norm2(feats_std) + model.fb_modulator(model.norm1(krfp_std)))


@torch.no_grad()
def predict_pce(model: ACIN, krfp_std: torch.Tensor, feats_std: torch.Tensor) -> torch.Tensor:
    pce, _ = model(krfp_std, feats_std)
    return pce.squeeze(1)


def raw_to_std(x_raw, mean, scale):
    return (x_raw - mean) / scale


def apply_active_deletion(krfp_raw: torch.Tensor, scores: torch.Tensor, alpha: float, mode: str,
                          generator: torch.Generator | None = None) -> torch.Tensor:
    N, D = krfp_raw.shape
    active = (krfp_raw == 1)
    k_per = torch.minimum(
        (alpha * active.sum(dim=1).float()).round().long().clamp_(min=1),
        active.sum(dim=1),
    )

    if mode == "selective":
        key = scores.clone()
    else:
        key = torch.rand(N, D, device=krfp_raw.device, generator=generator)

    key = key.masked_fill(~active, -1e9)
    sorted_idx = torch.argsort(key, dim=1, descending=True)
    rank = torch.empty_like(sorted_idx)
    rank.scatter_(1, sorted_idx, torch.arange(D, device=krfp_raw.device).unsqueeze(0).expand(N, -1))

    out = krfp_raw.clone()
    out[(rank < k_per.unsqueeze(1)) & active] = 0.0
    return out


# =============================================================================
# PART 1: Feature Ablation
# =============================================================================
def _eval_metrics(y_true, y_pred):
    return r2_score(y_true, y_pred), mean_squared_error(y_true, y_pred), mean_absolute_error(y_true, y_pred)


def _predict_mlp(model, x: pd.DataFrame, device: str) -> np.ndarray:
    with torch.no_grad():
        t = torch.tensor(x.to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
        return model(t.unsqueeze(1)).cpu().numpy().ravel()


def _run_mlp(x_train, y_train, x_test, y_test, device, out_dir) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    _, model_path = MLPTraining(
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        device=device, lr=LR, epoch=EPOCHS, early_stop_threshold=EARLY_STOP,
        dropout=DROPOUT, base_dir=str(out_dir), verbose=False,
    )
    model = BL_MLP(dim=x_train.shape[1], dropout=DROPOUT).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return {
        "train": _eval_metrics(y_train, _predict_mlp(model, x_train, device)),
        "test": _eval_metrics(y_test, _predict_mlp(model, x_test, device)),
    }


def _run_cat(x_train, y_train, x_test, y_test, out_dir, seed) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ml_models").mkdir(exist_ok=True)
    AutoML(
        x_train, x_test, y_train, y_test,
        base_dir=str(out_dir), params_cat=True, n_jobs=-1,
        optzr_init_points=10, optzr_n_iter=50,
        printinfo=False, random_state=seed, queue=None,
    ).train()
    cat = joblib.load(out_dir / "ml_models" / "CatBoostRegressor.pkl")
    return {
        "train": _eval_metrics(y_train, cat.predict(x_train)),
        "test": _eval_metrics(y_test, cat.predict(x_test)),
    }


def _aggregate_metrics(results_list: list) -> dict:
    tr = np.array([r["train"] for r in results_list], dtype=float)
    te = np.array([r["test"] for r in results_list], dtype=float)
    return {
        "tr_mean": tr.mean(axis=0), "tr_std": tr.std(axis=0, ddof=1),
        "te_mean": te.mean(axis=0), "te_std": te.std(axis=0, ddof=1),
    }


# ---------- Checkpointing Utilities ----------
def _make_task_key(seed: int, view: str, model: str) -> tuple:
    return int(seed), str(view), str(model)


def _load_completed_tasks(csv_path: Path) -> set:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return set()
    if df.empty or not {"seed", "view", "model"}.issubset(df.columns):
        return set()
    return set(zip(df["seed"].astype(int), df["view"].astype(str), df["model"].astype(str)))


def _append_task_result(csv_path: Path, seed: int, view: str, model: str, res: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, (r2, mse, mae) in res.items():
        rows.append({"seed": int(seed), "view": str(view), "model": str(model), "split": str(split), "r2": float(r2),
                     "mse": float(mse), "mae": float(mae)})
    df = pd.DataFrame(rows)
    header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    df.to_csv(csv_path, mode="a", header=header, index=False)


def _load_task_results_to_dict(seeds: list, task_defs: list) -> dict:
    if not FEAT_TASK_CSV.exists() or FEAT_TASK_CSV.stat().st_size == 0:
        return {}
    try:
        df = pd.read_csv(FEAT_TASK_CSV)
    except Exception:
        return {}
    if df.empty:
        return {}

    df = df.drop_duplicates(subset=["seed", "view", "model", "split"], keep="last")
    task_def_set = set(task_defs)
    raw = {}
    for _, r in df.iterrows():
        view, model = str(r["view"]), str(r["model"])
        if (view, model) not in task_def_set:
            continue
        key = (int(r["seed"]), view, model)
        raw.setdefault(key, {})[str(r["split"])] = (float(r["r2"]), float(r["mse"]), float(r["mae"]))

    per_seed = {}
    for seed in seeds:
        per_seed[seed] = {}
        for view, model in task_defs:
            entry = raw.get((seed, view, model), {})
            if "train" in entry and "test" in entry:
                per_seed[seed][(view, model)] = {"train": entry["train"], "test": entry["test"]}
    return per_seed


def _print_aggregated_summary(per_seed_results: dict):
    if not per_seed_results:
        return
    seeds = sorted(per_seed_results.keys())
    if not seeds:
        return

    all_keys = []
    for s in seeds:
        for k in per_seed_results[s].keys():
            if k not in all_keys:
                all_keys.append(k)

    print(f"\nAggregated Feature Ablation (mean ± s.d. across up to {len(seeds)} seeds)")
    print("-" * 122)
    print(f"{'View':<28} {'Model':<10} | {'R2_te':>17} {'MSE_te':>17} {'MAE_te':>17}  n")
    print("-" * 122)
    for key in all_keys:
        results = [per_seed_results[s][key] for s in seeds if key in per_seed_results[s]]
        if not results:
            continue
        agg = _aggregate_metrics(results)
        te_m, te_s = agg["te_mean"], agg["te_std"]
        n = len(results)
        print(
            f"{key[0]:<28} {key[1]:<10} | {te_m[0]:>7.4f}±{te_s[0]:<7.4f} {te_m[1]:>7.4f}±{te_s[1]:<7.4f} {te_m[2]:>7.4f}±{te_s[2]:<7.4f}  {n}")
    print("-" * 122)


def _save_feature_ablation_csv(per_seed: dict, seeds: list) -> None:
    FEAT_BENCH_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        if seed not in per_seed:
            continue
        for (view, model_name), m in per_seed[seed].items():
            for split in ("train", "test"):
                r2, mse, mae = m[split]
                rows.append(
                    {"seed": seed, "view": view, "model": model_name, "split": split, "r2": r2, "mse": mse, "mae": mae})
    pd.DataFrame(rows).to_csv(FEAT_BENCH_DIR / "per_seed_metrics.csv", index=False)

    all_keys = []
    for s in seeds:
        if s in per_seed:
            for k in per_seed[s].keys():
                if k not in all_keys:
                    all_keys.append(k)

    agg_rows = []
    for key in all_keys:
        results = [per_seed[s][key] for s in seeds if s in per_seed and key in per_seed[s]]
        if not results:
            continue
        agg = _aggregate_metrics(results)
        row = {"view": key[0], "model": key[1], "n_seeds": len(results)}
        for i, name in enumerate(("r2", "mse", "mae")):
            row[f"{name}_tr_mean"] = float(agg["tr_mean"][i])
            row[f"{name}_tr_std"] = float(agg["tr_std"][i])
            row[f"{name}_te_mean"] = float(agg["te_mean"][i])
            row[f"{name}_te_std"] = float(agg["te_std"][i])
        agg_rows.append(row)
    pd.DataFrame(agg_rows).to_csv(FEAT_BENCH_DIR / "aggregated_metrics.csv", index=False)

    print(f"\n[Feature Ablation] task results -> {FEAT_TASK_CSV}")
    print(f"[Feature Ablation] per-seed     -> {FEAT_BENCH_DIR / 'per_seed_metrics.csv'}")
    print(f"[Feature Ablation] aggregated   -> {FEAT_BENCH_DIR / 'aggregated_metrics.csv'}")


def _build_seed_data(seed: int) -> dict:
    ds = pd.read_csv(DL_INPUTS_CSV)
    y = ds.iloc[:, -1]

    idx_tr, idx_te = train_test_split(np.arange(len(y)), train_size=TRAIN_SIZE, shuffle=True, random_state=seed)
    feat_cols = ds.columns[:N_DESC]
    fp_cols = ds.columns[N_DESC:-1]

    x_prop_all = ds.iloc[:, :N_DESC]
    x_fp_all = ds.iloc[:, N_DESC:-1]
    x_cat_all = ds.iloc[:, :-1]

    prop_scaler = StandardScaler().fit(x_prop_all.iloc[idx_tr])
    x_prop_tr = pd.DataFrame(prop_scaler.transform(x_prop_all.iloc[idx_tr]), columns=feat_cols).reset_index(drop=True)
    x_prop_te = pd.DataFrame(prop_scaler.transform(x_prop_all.iloc[idx_te]), columns=feat_cols).reset_index(drop=True)

    fp_scaler = StandardScaler().fit(x_fp_all.iloc[idx_tr])
    x_fp_tr = pd.DataFrame(fp_scaler.transform(x_fp_all.iloc[idx_tr]), columns=fp_cols).reset_index(drop=True)
    x_fp_te = pd.DataFrame(fp_scaler.transform(x_fp_all.iloc[idx_te]), columns=fp_cols).reset_index(drop=True)

    feat_scaler = StandardScaler().fit(x_cat_all.iloc[idx_tr, :N_DESC])
    krfp_scaler = StandardScaler().fit(x_cat_all.iloc[idx_tr, N_DESC:])
    x_cat_tr = pd.concat([
        pd.DataFrame(feat_scaler.transform(x_cat_all.iloc[idx_tr, :N_DESC]), columns=feat_cols),
        pd.DataFrame(krfp_scaler.transform(x_cat_all.iloc[idx_tr, N_DESC:]), columns=fp_cols),
    ], axis=1).reset_index(drop=True)
    x_cat_te = pd.concat([
        pd.DataFrame(feat_scaler.transform(x_cat_all.iloc[idx_te, :N_DESC]), columns=feat_cols),
        pd.DataFrame(krfp_scaler.transform(x_cat_all.iloc[idx_te, N_DESC:]), columns=fp_cols),
    ], axis=1).reset_index(drop=True)

    return {
        "y_train": y.iloc[idx_tr].reset_index(drop=True),
        "y_test": y.iloc[idx_te].reset_index(drop=True),
        "x_prop_tr": x_prop_tr, "x_prop_te": x_prop_te,
        "x_fp_tr": x_fp_tr, "x_fp_te": x_fp_te,
        "x_cat_tr": x_cat_tr, "x_cat_te": x_cat_te,
        "x_prop_raw_tr": x_prop_all.iloc[idx_tr].reset_index(drop=True),
        "x_prop_raw_te": x_prop_all.iloc[idx_te].reset_index(drop=True),
        "x_fp_raw_tr": x_fp_all.iloc[idx_tr].reset_index(drop=True),
        "x_fp_raw_te": x_fp_all.iloc[idx_te].reset_index(drop=True),
        "x_cat_raw_tr": x_cat_all.iloc[idx_tr].reset_index(drop=True),
        "x_cat_raw_te": x_cat_all.iloc[idx_te].reset_index(drop=True),
    }


def _run_one_feature_task(seed: int, view: str, model: str, data: dict, device: str) -> dict:
    _reset_rng(seed)

    if view == "property-only (35)":
        if model == "MLP":
            return _run_mlp(data["x_prop_tr"], data["y_train"], data["x_prop_te"], data["y_test"], device,
                            FEAT_BENCH_DIR / f"property_only/seed{seed}")
        return _run_cat(data["x_prop_raw_tr"], data["y_train"], data["x_prop_raw_te"], data["y_test"],
                        FEAT_BENCH_DIR / f"property_only_cat/seed{seed}", seed)

    if view == "fingerprint-only (4860)":
        if model == "MLP":
            return _run_mlp(data["x_fp_tr"], data["y_train"], data["x_fp_te"], data["y_test"], device,
                            FEAT_BENCH_DIR / f"fingerprint_only/seed{seed}")
        return _run_cat(data["x_fp_raw_tr"], data["y_train"], data["x_fp_raw_te"], data["y_test"],
                        FEAT_BENCH_DIR / f"fingerprint_only_cat/seed{seed}", seed)

    if view == "concatenated (4895)":
        if model == "MLP":
            return _run_mlp(data["x_cat_tr"], data["y_train"], data["x_cat_te"], data["y_test"], device,
                            FEAT_BENCH_DIR / f"concatenated/seed{seed}")
        return _run_cat(data["x_cat_raw_tr"], data["y_train"], data["x_cat_raw_te"], data["y_test"],
                        FEAT_BENCH_DIR / f"concatenated_cat/seed{seed}", seed)

    if view == "concat (4895) + attention":
        acin_seed_dir = FEAT_BENCH_DIR / f"concatenated_acin/seed{seed}"
        _, _, acin_m = ACINTraining(
            x_train=data["x_cat_tr"], y_train=data["y_train"], x_test=data["x_cat_te"], y_test=data["y_test"],
            device=device, lr=LR, epoch=EPOCHS, early_stop_threshold=EARLY_STOP, dropout=DROPOUT,
            base_dir=str(acin_seed_dir), attn_out_path=str(acin_seed_dir / "attention_weights.csv"), verbose=False,
        )
        return {
            "train": (acin_m["r2_tr"], acin_m["mse_tr"], acin_m["mae_tr"]),
            "test": (acin_m["r2_te"], acin_m["mse_te"], acin_m["mae_te"]),
        }

    raise ValueError(f"Unknown view: {view}")


def run_feature_ablation(device: str = "auto", seeds: list | None = None, save_csv: bool = True) -> None:
    device = pick_device(device)
    if seeds is None:
        seeds = FEAT_ABLATION_SEEDS

    FEAT_BENCH_DIR.mkdir(parents=True, exist_ok=True)
    task_defs = FEAT_TASK_DEFS
    total = len(seeds) * len(task_defs)
    done = _load_completed_tasks(FEAT_TASK_CSV)

    print("\n" + "=" * 80)
    print(f"Starting Feature Ablation (seeds={seeds})")
    print(f"Tasks: {len(task_defs)} views x {len(seeds)} seeds = {total}")
    print(f"Already done: {len(done)} | Pending: {total - len(done)}")
    print(f"Resume file: {FEAT_TASK_CSV}")
    print("=" * 80)

    current_seed = None
    current_data = None
    task_idx = 0

    for seed in seeds:
        seed_pending = [(v, m) for (v, m) in task_defs if _make_task_key(seed, v, m) not in done]
        if not seed_pending:
            print(f"\n--- seed={seed}: all {len(task_defs)} tasks done, skip ---")
            continue

        if seed != current_seed:
            current_data = None
            _cleanup_memory()
            current_seed = seed
            current_data = _build_seed_data(seed)
            print(f"\n--- seed={seed}: data built, {len(seed_pending)} pending ---")

        for view, model in task_defs:
            task_idx += 1
            key = _make_task_key(seed, view, model)
            if key in done:
                print(f"[{task_idx:>2}/{total}] seed={seed} | {view:<28} {model:<10}  SKIP (done)")
                continue

            print(f"[{task_idx:>2}/{total}] seed={seed} | {view:<28} {model:<10} ...", end="", flush=True)
            t0 = time.time()
            try:
                res = _run_one_feature_task(seed, view, model, current_data, device)
            except Exception as exc:
                print(f" FAILED: {exc}")
                continue

            _append_task_result(FEAT_TASK_CSV, seed, view, model, res)
            done.add(key)
            print(f" Done in {time.time() - t0:.0f}s | R2_te={res['test'][0]:.4f}")

    current_data = None
    _cleanup_memory()

    if save_csv:
        per_seed = _load_task_results_to_dict(seeds, task_defs)
        _save_feature_ablation_csv(per_seed, seeds)
        _print_aggregated_summary(per_seed)


# =============================================================================
# PART 2: Attention Ablation (Deletion & Retrain)
# =============================================================================
def run_attention_deletion_test(alphas=DELETION_ALPHAS, n_random=DELETION_N_RANDOM, seed=DELETION_SEED,
                                device=None) -> None:
    device = pick_device(device)
    out_csv = DEL_BENCH_DIR / "deletion_test_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print(f"Starting Attention Deletion Test (alphas={alphas}, R={n_random})")
    print("=" * 80)

    bundle = build_bundle(device)
    fs, ks = bundle["feat_scaler"], bundle["krfp_scaler"]
    krfp_raw, feats_np = bundle["x_test_krfp_raw"], bundle["x_test_feats_np"]

    k_mean = torch.tensor(ks.mean_, device=device, dtype=torch.float32)
    k_scale = torch.tensor(ks.scale_, device=device, dtype=torch.float32)

    krfp_std = raw_to_std(krfp_raw, k_mean, k_scale)
    feats_std = torch.tensor(fs.transform(feats_np), device=device, dtype=torch.float32)

    model = load_baseline_acin(device)
    pce_orig = predict_pce(model, krfp_std, feats_std)
    scores = per_sample_scores(model, krfp_std, feats_std)
    n_active = krfp_raw.sum(dim=1).long().cpu().numpy()

    gen = torch.Generator(device=device).manual_seed(seed)
    rows = []

    print(f"{'alpha':>6} | {'k̄':>4} | {'|Δ|_sel':>8} | {'|Δ|_rnd':>8} | {'ratio':>6} | {'win%':>6} | {'p-value':>10}")
    print("-" * 65)

    for alpha in alphas:
        krfp_sel = apply_active_deletion(krfp_raw, scores, alpha, "selective")
        pce_sel = predict_pce(model, raw_to_std(krfp_sel, k_mean, k_scale), feats_std)
        signed_sel = (pce_sel - pce_orig).cpu().numpy()
        d_sel = np.abs(signed_sel)

        signed_rnd_all = np.empty((n_random, len(pce_orig)), dtype=np.float64)
        for r in range(n_random):
            krfp_rnd = apply_active_deletion(krfp_raw, scores, alpha, "random", gen)
            pce_rnd = predict_pce(model, raw_to_std(krfp_rnd, k_mean, k_scale), feats_std)
            signed_rnd_all[r] = (pce_rnd - pce_orig).cpu().numpy()

        signed_rnd_mean = signed_rnd_all.mean(axis=0)
        d_rnd = np.abs(signed_rnd_all).mean(axis=0)

        win_rate = float(((d_sel - d_rnd) > 0).mean())
        _, p_val = wilcoxon(d_sel, d_rnd, alternative="greater")
        k_bar = int(np.mean(np.clip((alpha * n_active).round(), 1, None)))

        print(f"{alpha:>6.2f} | {k_bar:>4d} | {d_sel.mean():>8.4f} | {d_rnd.mean():>8.4f} | "
              f"{d_sel.mean() / max(d_rnd.mean(), 1e-12):>6.2f} | {win_rate * 100:>5.1f}% | {p_val:>10.2e}")

        for i in range(len(pce_orig)):
            rows.append({
                "sample_idx": i, "alpha": alpha, "n_active": int(n_active[i]), "pce_orig": float(pce_orig[i].cpu()),
                "delta_selective": float(d_sel[i]), "delta_random_mean": float(d_rnd[i]),
                "signed_selective": float(signed_sel[i]), "signed_random_mean": float(signed_rnd_mean[i]),
                "counter_example": bool(d_sel[i] < d_rnd[i]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    _plot_deletion_curve(df, out_csv.with_suffix(".png"))
    _cleanup_memory()


def _plot_deletion_curve(df: pd.DataFrame, out_png: Path):
    try:
        plt = _setup_plot_style()
        g = df.groupby("alpha").agg(sel=("delta_selective", "mean"), rnd=("delta_random_mean", "mean"))
        g["ratio"] = g["sel"] / g["rnd"].clip(lower=1e-12)
        g["sel_marginal"] = g["sel"].diff().fillna(g["sel"])
        g["rnd_marginal"] = g["rnd"].diff().fillna(g["rnd"])
        peak_alpha = float(g["ratio"].idxmax())

        col_sel, col_rnd, col_blue = "#E24A4A", "#888888", "#3274A1"
        fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.3))

        # Panel 1: Deletion curve
        ax = axes[0]
        for label, col, color, marker in [("Selective", "delta_selective", col_sel, "o"),
                                          ("Random (paired)", "delta_random_mean", col_rnd, "s")]:
            m, s = df.groupby("alpha")[col].mean(), df.groupby("alpha")[col].std()
            ax.errorbar(m.index, m.values, yerr=s.values, marker=marker, markersize=4, capsize=2, capthick=0.8,
                        linewidth=1.0, color=color, elinewidth=0.8, markerfacecolor=color, markeredgecolor=color,
                        label=label)

        ax.set_xlabel("Deletion fraction α")
        ax.set_ylabel("Mean |ΔPCE| (%)")
        ax.set_title(f"Deletion curve (peak at α={peak_alpha:.2f})")
        ax.set_ylim(-0.5, 3.5)
        ax.legend(loc="upper left")
        ax.tick_params(which="both", direction="in", top=False, right=False)

        # Panel 2: Signed ΔPCE distribution
        ax = axes[1]
        sub = df[df["alpha"] == peak_alpha]
        ax.hist(sub["signed_random_mean"], bins=40, alpha=0.4, label="Random", color=col_rnd, density=True,
                histtype="stepfilled", edgecolor="black", linewidth=0.5)
        ax.hist(sub["signed_selective"], bins=40, alpha=0.6, label="Selective", color=col_sel, density=True,
                histtype="stepfilled", edgecolor="black", linewidth=0.5)
        ax.axvline(0, ls="--", color="black", linewidth=0.8)
        ax.set_xlabel("Signed ΔPCE (%)")
        ax.set_ylabel("Density")
        ax.set_title(f"Signed ΔPCE (α={peak_alpha:.2f})")
        ax.legend(loc="upper left")
        ax.tick_params(which="both", direction="in", top=False, right=False)

        # Panel 3: Ratio & marginal sensitivity
        ax = axes[2]
        l1, = ax.plot(g.index, g["ratio"], "o-", color=col_blue, markersize=4, linewidth=1.0, label="Ratio")
        ax.set_xlabel("Deletion fraction α")
        ax.set_ylabel("Ratio (sel/rnd)", color=col_blue)
        ax.tick_params(axis='y', labelcolor=col_blue, which="both", direction="in")
        ax.set_title("Ratio vs α")
        ax.tick_params(axis='x', which="both", direction="in")
        ax.set_ylim(0.8, 3.8)

        ax2 = ax.twinx()
        l2, = ax2.plot(g.index, g["sel_marginal"], "s--", color=col_sel, markersize=4, linewidth=0.9, alpha=0.9,
                       label="Sel marginal")
        l3, = ax2.plot(g.index, g["rnd_marginal"], "^--", color=col_rnd, markersize=4, linewidth=0.9, alpha=0.9,
                       label="Rnd marginal")
        ax2.set_ylabel("Marginal |ΔPCE| / α step (%)", color="black", rotation=270, labelpad=15)
        ax2.tick_params(axis='y', colors="black", which="both", direction="in")
        ax2.set_ylim(0.05, 1.1)

        ax.legend(handles=[l1, l2, l3], loc="upper right")
        fig.tight_layout(pad=0.5, w_pad=3.0)
        fig.savefig(out_png)
        plt.close(fig)
        print(f"[Deletion Test] Plot saved -> {out_png}")
    except Exception as e:
        print(f"Plot skipped: {e}")


def _retrain_one(krfp_raw_del, bundle, device, seed, out_dir) -> dict | None:
    _set_seed(seed)
    _cleanup_memory()

    x_tr = pd.DataFrame(np.concatenate([
        bundle["feat_scaler"].transform(bundle["x_train_feats_np"]),
        bundle["krfp_scaler"].transform(krfp_raw_del.cpu().numpy()),
    ], axis=1), columns=bundle["all_cols"])

    x_te = pd.DataFrame(np.concatenate([
        bundle["feat_scaler"].transform(bundle["x_test_feats_np"]),
        bundle["krfp_scaler"].transform(bundle["x_test_krfp_raw"].cpu().numpy()),
    ], axis=1), columns=bundle["all_cols"])

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dl_models").mkdir(exist_ok=True)

    try:
        _, _, metrics = ACINTraining(
            x_train=x_tr, y_train=bundle["y_train"], x_test=x_te, y_test=bundle["y_test"],
            device=device, lr=LR, epoch=EPOCHS, early_stop_threshold=EARLY_STOP, dropout=DROPOUT, batch_size=16,
            base_dir=str(out_dir), attn_out_path=str(out_dir / "dl_models" / "attention_weights.csv"), verbose=False,
        )
        return metrics
    except Exception as exc:
        print(f"    [ERROR] {exc}")
        return None
    finally:
        _cleanup_memory()


def run_attention_retrain_ablation(alphas=RETRAIN_ALPHAS, n_random=RETRAIN_N_RANDOM,
                                   n_train_seeds=RETRAIN_N_TRAIN_SEEDS, device=None) -> None:
    device = pick_device(device)
    csv_path = RETRAIN_BENCH_DIR / "ablation_results.csv"
    RETRAIN_BENCH_DIR.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print(f"Starting Attention Retrain Ablation (alphas={alphas})")
    print("=" * 80)

    bundle = build_bundle(device)
    train_seeds = list(range(42, 42 + n_train_seeds))
    sample_seeds = list(range(1001, 1001 + n_random))

    model = load_baseline_acin(device)
    k_mean = torch.tensor(bundle["krfp_scaler"].mean_, device=device, dtype=torch.float32)
    k_scale = torch.tensor(bundle["krfp_scaler"].scale_, device=device, dtype=torch.float32)

    krfp_std_tr = raw_to_std(bundle["x_train_krfp_raw"], k_mean, k_scale)
    feats_std_tr = torch.tensor(bundle["feat_scaler"].transform(bundle["x_train_feats_np"]), device=device,
                                dtype=torch.float32)
    train_scores = per_sample_scores(model, krfp_std_tr, feats_std_tr)

    del model, krfp_std_tr, feats_std_tr
    _cleanup_memory()

    tasks = [{"alpha": 0.0, "mode": "baseline", "ss": None, "ts": ts} for ts in train_seeds]
    for alpha in alphas:
        for ts in train_seeds:
            tasks.append({"alpha": alpha, "mode": "selective", "ss": None, "ts": ts})
        for ss in sample_seeds:
            for ts in train_seeds:
                tasks.append({"alpha": alpha, "mode": "random", "ss": ss, "ts": ts})

    done_set, results = set(), []
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        for _, r in df_old.iterrows():
            ss_val = r["sample_seed"]
            ss = None if pd.isna(ss_val) else int(ss_val)
            done_set.add((float(r["alpha"]), str(r["mode"]), ss, int(r["train_seed"])))
        results = df_old.to_dict("records")
        print(f"[Retrain] Found {len(done_set)} existing tasks. Resuming...")

    print(f"[Retrain] Total tasks: {len(tasks)} | Pending: {len(tasks) - len(done_set)}")

    for idx, t in enumerate(tasks, 1):
        key = (float(t["alpha"]), str(t["mode"]), t["ss"], int(t["ts"]))
        if key in done_set:
            continue

        print(f"[{idx:>2}/{len(tasks)}] mode={t['mode']:<9} α={t['alpha']:.2f} | ss={str(t['ss']):<4} ts={t['ts']} ...",
              end="", flush=True)

        gen = torch.Generator(device=device).manual_seed(0)
        krfp_del = bundle["x_train_krfp_raw"]
        if t["mode"] != "baseline":
            if t["mode"] == "random":
                gen = torch.Generator(device=device).manual_seed(t["ss"])
            krfp_del = apply_active_deletion(bundle["x_train_krfp_raw"], train_scores, t["alpha"], t["mode"], gen)

        t0 = time.time()
        metrics = _retrain_one(krfp_del, bundle, device, t["ts"],
                               RETRAIN_BENCH_DIR / f"{t['mode']}_a{t['alpha']}_ss{t['ss']}_ts{t['ts']}")

        if metrics:
            print(f" Done in {time.time() - t0:.0f}s | R2={metrics['r2_te']:.4f}")
            results.append(
                {"alpha": t["alpha"], "mode": t["mode"], "sample_seed": t["ss"], "train_seed": t["ts"], **metrics})
            pd.DataFrame(results).to_csv(csv_path, index=False)
        else:
            print(" Failed.")

    df = pd.DataFrame(results)
    _summarize_retrain(df)
    _plot_retrain_bar(csv_path)


def _summarize_retrain(df: pd.DataFrame):
    if df.empty:
        return
    print("\n" + "=" * 80 + "\nMean ± s.d. per condition\n" + "-" * 80)
    for (a, m), g in df.groupby(["alpha", "mode"], sort=False):
        print(
            f"{a:>6.2f} | {m:>10} | n={len(g):>2} | R2={g['r2_te'].mean():.4f}±{g['r2_te'].std():.4f} | MAE={g['mae_te'].mean():.4f}±{g['mae_te'].std():.4f}")

    print("\nSelective vs Random (Welch's t-test; nested sampling averaged)")
    print("-" * 80)
    for alpha in sorted(df["alpha"].unique()):
        if alpha == 0.0:
            continue
        sel = df[(df["alpha"] == alpha) & (df["mode"] == "selective")]
        rnd = df[(df["alpha"] == alpha) & (df["mode"] == "random")]
        if len(sel) < 2 or len(rnd) < 2:
            continue

        rnd_agg = rnd.groupby("train_seed")["r2_te"].mean()
        sel_agg = sel.groupby("train_seed")["r2_te"].mean()
        common = sel_agg.index.intersection(rnd_agg.index)
        if len(common) < 2:
            continue
        _, p_r2 = ttest_ind(sel_agg.loc[common], rnd_agg.loc[common], equal_var=False)
        print(
            f"α={alpha:>4.2f} | n={len(common)}/{len(common)} (train seeds) | ΔR2={sel_agg.loc[common].mean() - rnd_agg.loc[common].mean():+.4f} (p={p_r2:.2e})")
    print("=" * 80 + "\n")


def _plot_retrain_bar(csv_path: Path):
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    plt = _setup_plot_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    alphas = sorted(df[df["alpha"] > 0]["alpha"].unique())
    x = np.arange(len(alphas) + 1)

    base_r2 = df[df["mode"] == "baseline"]["r2_te"].mean()
    base_std = df[df["mode"] == "baseline"]["r2_te"].std()
    sel_means = [df[(df["alpha"] == a) & (df["mode"] == "selective")]["r2_te"].mean() for a in alphas]
    sel_stds = [df[(df["alpha"] == a) & (df["mode"] == "selective")]["r2_te"].std() for a in alphas]
    rnd_means = [df[(df["alpha"] == a) & (df["mode"] == "random")]["r2_te"].mean() for a in alphas]
    rnd_stds = [df[(df["alpha"] == a) & (df["mode"] == "random")]["r2_te"].std() for a in alphas]

    col_base, col_sel, col_rnd = "#7EA6C9", "#E88C8C", "#B0B0B0"
    lw = 0.65
    err_kw = dict(elinewidth=0.6, capthick=0.6, capsize=2.5)

    ax.bar(x[0], base_r2, width=0.3, yerr=base_std, label="Baseline", color=col_base, edgecolor="black", linewidth=lw,
           error_kw=err_kw)
    ax.bar(x[1:] - 0.35 / 2, sel_means, 0.35, yerr=sel_stds, label="Selective", color=col_sel, edgecolor="black",
           linewidth=lw, error_kw=err_kw)
    ax.bar(x[1:] + 0.35 / 2, rnd_means, 0.35, yerr=rnd_stds, label="Random", color=col_rnd, edgecolor="black",
           linewidth=lw, error_kw=err_kw)

    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline\n(No deletion)"] + [f"α={a:.2f}" for a in alphas], fontsize=10)
    ax.set_ylabel("Test $R^2$", fontsize=11)
    ax.tick_params(axis='y', labelsize=9)

    ymax = max(1.0, max(np.concatenate([sel_means, rnd_means])) * 1.1)
    ax.set_ylim(0, ymax)

    ax.legend(loc="upper right", fontsize=10, handlelength=1.5, labelspacing=0.5)
    ax.tick_params(which="both", direction="in", top=False, right=False)

    out_png = csv_path.with_name("retrain_ablation_bar.png")
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[Retrain] Plot saved -> {out_png}")


# =============================================================================
# Unified Runner
# =============================================================================
def run_all_ablations(device: str = "auto") -> None:
    device = pick_device(device)
    run_feature_ablation(device)
    run_attention_deletion_test(device=device)
    run_attention_retrain_ablation(device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", choices=["all", "feature", "deletion", "retrain"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reset-feature", action="store_true", help="Delete task_results.csv and rerun from scratch.")
    args = parser.parse_args()

    dev = pick_device(args.device)

    if args.reset_feature and args.task in ("all", "feature"):
        if FEAT_TASK_CSV.exists():
            FEAT_TASK_CSV.unlink()
            print(f"[Feature Ablation] removed {FEAT_TASK_CSV}")

    if args.task == "all":
        run_all_ablations(dev)
    elif args.task == "feature":
        run_feature_ablation(dev)
    elif args.task == "deletion":
        run_attention_deletion_test(device=dev)
    elif args.task == "retrain":
        run_attention_retrain_ablation(device=dev)