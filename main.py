"""Co-SAM analysis pipeline — entry point.

Available stages:
  mol_calc           : ORCA + Multiwfn + RDKit molecular-property computation.
  ml                 : AutoML training of regressors on ml_inputs.csv.
  eval               : SHAP beeswarm and diagnostics.
  predict            : Co-SAM mixture prediction and SHAP waterfalls.
  acin               : ACIN attention training and substructure gallery.
  feature_ablation   : Feature Ablation (5 seeds).
  attention_deletion : Attention Deletion Test.
  attention_retrain  : Attention Retrain Ablation.
  all_ablation       : Run all ablation tasks sequentially.
  gui                : Launch the PySide6 GUI.
  all                : Run every enabled stage in order.
"""
from __future__ import annotations

import os
import random
import sys
import warnings
from pathlib import Path
from typing import Optional

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from acin_analysis import render_attention_gallery
from dl_training import ACINTraining
from ml_eval import ME
from ml_training import AutoML
from mol_calc import MolCalc

from ablation import (
    run_feature_ablation,
    run_attention_deletion_test,
    run_attention_retrain_ablation,
    run_all_ablations,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ===========================================================================
# Configuration
# ===========================================================================
RUN_STAGE = "gui"

SEED = 42

METHOD = "B3LYP"
BASIS = "def2-TZVP"
NCORE = 5
CPU_USAGE = 0.8
MULTITHREAD = True

MODELS = "lr, cat"
INIT_POINTS = 10
N_ITER = 50

EVAL_MODEL = "CatBoostRegressor"
TOP_N = 20
FITTING = "gam"

DEVICE = "auto"
EPOCHS = 1000
EARLY_STOP = 40
LR = 1e-3
DROPOUT = 0.3
TOP_K = 100
FIG_ROW = 10
RENDER_ONLY = False

# Stage toggles for "all"
LAUNCH_GUI = False
SKIP_MOLCALC = False
SKIP_ML = False
SKIP_EVAL = False
SKIP_PREDICT = False
SKIP_ACIN = False
SKIP_FEAT_ABLATION = False
SKIP_ATTN_ABLATION = False
# ===========================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MOL_DIR = PROJECT_ROOT / "mol_calc"
ML_MODELS_DIR = PROJECT_ROOT / "ml_models"
DL_MODELS_DIR = PROJECT_ROOT / "dl_models"
PLOTS_DIR = PROJECT_ROOT / "plots"
STRUCT_DIR = PLOTS_DIR / "substructures"
BENCH_DIR = PROJECT_ROOT / "benchmark"

ML_INPUTS_CSV = PROJECT_ROOT / "ml_inputs.csv"
DL_INPUTS_CSV = PROJECT_ROOT / "dl_inputs.csv"
ATTN_CSV = PROJECT_ROOT / "acin_weights.csv"

RANDOM_STATE = SEED
TRAIN_SIZE = 0.75
N_KRFP = 4860
N_DESC = 35

AUTOML_BASE = str(PROJECT_ROOT)

MODEL_FLAG = {
    "lr": "params_lr", "ridge": "params_ridge", "svr": "params_svr",
    "knn": "params_knn", "dt": "params_dtr", "rf": "params_rf",
    "et": "params_et", "gbdt": "params_gbdt", "ada": "params_ada",
    "xgb": "params_xgb", "lgbm": "params_lgbm", "cat": "params_cat",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def pick_device(pref: str = "auto") -> str:
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _log(stage: str, msg: str) -> None:
    print(f"[{stage}] {msg}", flush=True)


def _require(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — {hint}")


def _standardize_acin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize descriptors (35D) and KRFP bits (4860D) independently."""
    feats, krfps = df.iloc[:, :N_DESC], df.iloc[:, N_DESC:]
    feats_sc = StandardScaler().fit_transform(feats)
    krfps_sc = StandardScaler().fit_transform(krfps)
    return pd.concat([
        pd.DataFrame(feats_sc, columns=feats.columns, index=df.index),
        pd.DataFrame(krfps_sc, columns=krfps.columns, index=df.index),
    ], axis=1)


def _parse_models(spec: str) -> dict:
    flags = {flag: False for flag in MODEL_FLAG.values()}
    for name in spec.replace(" ", "").split(","):
        if not name:
            continue
        if name not in MODEL_FLAG:
            raise ValueError(f"unknown model '{name}'")
        flags[MODEL_FLAG[name]] = True
    return flags


# ---------------------------------------------------------------------------
# Pipeline Stages
# ---------------------------------------------------------------------------
def run_mol_calc(method: str, basis: str, ncore: int, cpu_usage: float,
                 multithreading: bool) -> None:
    _log("mol_calc", "starting")
    ensure_dirs(MOL_DIR)
    smiles = pd.read_csv(PROJECT_ROOT / "smiles.csv")
    experiment = pd.read_csv(PROJECT_ROOT / "exp_data.csv")
    calc = MolCalc(
        smiles=smiles, method=method, basis_set=basis,
        ncore=ncore, cpu_usage=cpu_usage, multithreading=multithreading,
    )
    calc.mol_calc(output_path=str(MOL_DIR))
    calc.prop_gen(input_path=str(MOL_DIR), output_file=str(MOL_DIR / "mol_prop"))
    ml_df, dl_df = calc.transform(experiment)
    ml_df.to_csv(ML_INPUTS_CSV, index=False)
    dl_df.to_csv(DL_INPUTS_CSV, index=False)
    _log("mol_calc", f"ml={ml_df.shape}, dl={dl_df.shape}")


def run_ml_training(models_spec: str, init_points: int, n_iter: int,
                    base_dir: str, models_dir: Path) -> None:
    _log("ml", "starting")
    _require(ML_INPUTS_CSV, "run RUN_STAGE='mol_calc' first")
    ds = pd.read_csv(ML_INPUTS_CSV)
    x_train, x_test, y_train, y_test = train_test_split(
        ds.iloc[:, :-1], ds.iloc[:, -1],
        train_size=TRAIN_SIZE, shuffle=True, random_state=RANDOM_STATE,
    )
    ensure_dirs(models_dir)
    AutoML(
        x_train, x_test, y_train, y_test,
        base_dir=base_dir, n_jobs=-1,
        optzr_init_points=init_points, optzr_n_iter=n_iter,
        printinfo=True, random_state=RANDOM_STATE, queue=None,
        **_parse_models(models_spec),
    ).train()
    _log("ml", f"done -> {models_dir}")


def _load_me(model: str) -> ME:
    _require(ML_INPUTS_CSV, "run RUN_STAGE='mol_calc' first")
    _require(ML_MODELS_DIR / f"{model}.pkl", "run RUN_STAGE='ml' first")
    ds = pd.read_csv(ML_INPUTS_CSV)
    x_train, x_test, y_train, y_test = train_test_split(
        ds.iloc[:, :-1], ds.iloc[:, -1],
        train_size=TRAIN_SIZE, shuffle=True, random_state=RANDOM_STATE,
    )
    return ME(
        str(ML_MODELS_DIR), ds.iloc[:, :-1], ds.iloc[:, -1],
        x_train, y_train, x_test, y_test,
        random_state=RANDOM_STATE, base_dir=str(PLOTS_DIR),
    )


def run_ml_evaluation(model: str, top_n: int, fitting_model: str) -> ME:
    _log("eval", f"starting ({model})")
    ensure_dirs(PLOTS_DIR)
    plot = _load_me(model)
    plot.score_hist()
    plot.accs()
    plot.beeswarm(model, max_display=top_n)
    plot.deps(model, fitting_model=fitting_model)
    _log("eval", "done")
    return plot


def run_cosam_prediction(model: str, plot: Optional[ME] = None) -> pd.DataFrame:
    _log("predict", f"starting ({model})")
    _require(PROJECT_ROOT / "sams_pred.csv", "provide sams_pred.csv")
    _require(MOL_DIR / "mol_prop.csv", "run RUN_STAGE='mol_calc' first")
    smiles = pd.read_csv(PROJECT_ROOT / "smiles.csv")
    sams_pred = pd.read_csv(PROJECT_ROOT / "sams_pred.csv")
    mol_prop = pd.read_csv(MOL_DIR / "mol_prop.csv")
    ml_sams, _ = MolCalc.transform(
        experiment_condition=sams_pred, smiles=smiles, mol_prop=mol_prop,
    )
    features = ml_sams.copy()
    cat = joblib.load(ML_MODELS_DIR / f"{model}.pkl")
    with_pred = features.copy()
    with_pred["Cat_Predict"] = cat.predict(features)
    wf_dir = PLOTS_DIR / "waterfall" / model
    ensure_dirs(wf_dir)
    with_pred.to_csv(wf_dir / "pred_results.csv", index=False)
    if plot is None:
        plot = _load_me(model)
    plot.waterfall(model, data=features, label_c=None)
    _log("predict", "done")
    return with_pred


def run_acin(device: str, epochs: int, early_stop: int, lr: float,
             dropout: float, show_mols: int, fig_row: int,
             render_only: bool) -> None:
    _log("acin", "render-only" if render_only else "training ACIN")
    ensure_dirs(DL_MODELS_DIR, PLOTS_DIR, STRUCT_DIR)
    if render_only:
        _require(ATTN_CSV, "run RUN_STAGE='acin' first")
        attn_df = pd.read_csv(ATTN_CSV)
    else:
        _require(DL_INPUTS_CSV, "run RUN_STAGE='mol_calc' first")
        ds = pd.read_csv(DL_INPUTS_CSV)
        x = _standardize_acin_features(ds.iloc[:, :-1])
        y = ds.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=TRAIN_SIZE, shuffle=True, random_state=RANDOM_STATE,
        )
        attn_df, model_path, _ = ACINTraining(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            device=device, lr=lr, epoch=epochs,
            early_stop_threshold=early_stop, dropout=dropout,
            base_dir=AUTOML_BASE,
        )
        _log("acin", f"model -> {model_path}")
    out = render_attention_gallery(
        attn_df, plots_dir=PLOTS_DIR, struct_dir=STRUCT_DIR,
        show_mols=show_mols, fig_row=fig_row, log_fn=_log,
    )
    _log("acin", f"gallery -> {out}")


# ---------------------------------------------------------------------------
# GUI & Dispatch
# ---------------------------------------------------------------------------
def _ensure_qt_resources() -> None:
    """Compile res.qrc -> res_rc.py when missing or stale."""
    import shutil
    import subprocess

    base = Path(__file__).resolve().parent
    qrc = base / "res.qrc"
    py = base / "res_rcc.py"
    if not qrc.exists():
        return
    if py.exists() and py.stat().st_mtime >= qrc.stat().st_mtime:
        return

    rcc = shutil.which("pyside6-rcc")

    if not rcc:
        for name in ("pyside6-rcc.exe", "pyside6-rcc"):
            candidate = Path(sys.executable).parent / "Scripts" / name
            if candidate.is_file():
                rcc = str(candidate)
                break

    if not rcc:
        import PySide6
        for name in ("rcc.exe", "rcc"):
            candidate = Path(PySide6.__file__).parent / name
            if candidate.is_file():
                rcc = str(candidate)
                break

    if not rcc:
        raise FileNotFoundError(
            "Cannot find pyside6-rcc. Run 'pyside6-rcc --version' in the same "
            "terminal to verify the tool is installed and on PATH."
        )

    print(f"[build] compiling {qrc.name} -> {py.name}")
    subprocess.run(
        [rcc, str(qrc), "-o", str(py)],
        check=True,
    )

def launch_gui() -> None:
    _ensure_qt_resources()

    import PySide6
    pyside_dir = os.path.dirname(PySide6.__file__)
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH",
                          os.path.join(pyside_dir, "plugins", "platforms"))
    os.environ.setdefault("QT_PLUGIN_PATH", os.path.join(pyside_dir, "plugins"))

    import res_rcc  # noqa: F401  (registers Qt resources)
    from PySide6.QtWidgets import QApplication
    from gui import MainWindow

    plt.close("all")

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


def run_stage_all(device: str) -> None:
    if not SKIP_MOLCALC:
        run_mol_calc(METHOD, BASIS, NCORE, CPU_USAGE, MULTITHREAD)
    if not SKIP_ML:
        run_ml_training(MODELS, INIT_POINTS, N_ITER, AUTOML_BASE, ML_MODELS_DIR)
    if not SKIP_EVAL:
        run_ml_evaluation(EVAL_MODEL, TOP_N, FITTING)
    if not SKIP_PREDICT:
        run_cosam_prediction(EVAL_MODEL)
    if not SKIP_ACIN:
        run_acin(device, EPOCHS, EARLY_STOP, LR, DROPOUT,
                 TOP_K, FIG_ROW, RENDER_ONLY)

    if not SKIP_FEAT_ABLATION:
        run_feature_ablation(device)          # 5 seeds, per-seed + aggregated CSVs
    if not SKIP_ATTN_ABLATION:
        run_attention_deletion_test(device=device)
        run_attention_retrain_ablation(device=device)

    _log("all", "pipeline finished")
    if LAUNCH_GUI:
        launch_gui()


def dispatch(stage: str) -> None:
    device = pick_device(DEVICE)
    stage = stage.strip().lower()

    if stage == "mol_calc":
        run_mol_calc(METHOD, BASIS, NCORE, CPU_USAGE, MULTITHREAD)
    elif stage == "ml":
        run_ml_training(MODELS, INIT_POINTS, N_ITER, AUTOML_BASE, ML_MODELS_DIR)
    elif stage == "eval":
        run_ml_evaluation(EVAL_MODEL, TOP_N, FITTING)
    elif stage == "predict":
        run_cosam_prediction(EVAL_MODEL)
    elif stage == "acin":
        run_acin(device, EPOCHS, EARLY_STOP, LR, DROPOUT,
                 TOP_K, FIG_ROW, RENDER_ONLY)

    elif stage == "feature_ablation":
        run_feature_ablation(device)
    elif stage == "attention_deletion":
        run_attention_deletion_test(device=device)
    elif stage == "attention_retrain":
        run_attention_retrain_ablation(device=device)
    elif stage == "all_ablation":
        run_all_ablations(device)

    elif stage == "gui":
        launch_gui()
    elif stage == "all":
        run_stage_all(device)
    else:
        raise ValueError(f"unknown stage '{stage}'")


if __name__ == "__main__":
    set_seed(SEED)
    pd.set_option("display.max_columns", None)
    pd.set_option("expand_frame_repr", False)
    try:
        dispatch(RUN_STAGE)
    except Exception as exc:
        import traceback
        print(f"\n[FAILED] stage='{RUN_STAGE}': {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)