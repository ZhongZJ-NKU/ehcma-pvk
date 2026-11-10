# An Explainable Agent Guides Hybrid Interface Design

## for Efficient Perovskite Tandem Solar Cells

A closed-loop *predict-explain-design* framework coupling a CatBoost regressor (descriptor-level PCE prediction), an Adaptive Cross-Modal Interaction Network (ACIN, descriptor + Klekota-Roth fingerprint), dual attribution signals (SHAP + ACIN signed attention), and the MSEC reasoning agent (VLM perception -> hierarchy-constrained CoT -> design rules).

## Contents

- [Requirements](#requirements) · [Installation](#installation) · [Quick Start](#quick-start) · [Repository Layout](#repository-layout) · [Data Format](#data-format) · [Pipeline Stages](#pipeline-stages) · [Configuration](#configuration) · [Reproducibility](#reproducibility) · [MSEC Agent](#msec-agent) · [Citation](#citation) · [License](#license)

## Requirements

| Item | Notes |
|---|---|
| OS / Python | Windows x64 (non-GUI stages run on Linux/macOS) / 3.9 |
| RAM / Disk | >= 16 GB / >= 5 GB |
| `numpy == 1.20.3` | Pinned for a shared MKL ABI with `scipy 1.10.1` / `scikit-fingerprints 1.13.1`. Do **not** upgrade to `>= 1.24` without revalidating SHAP / pygam. |
| ORCA + Multiwfn | Optional — required **only** for `mol_calc`; add both to `PATH` |
| CUDA | 11.8 driver >= 522.06 — required **only** for GPU training; CPU fallback automatic |

- `ehcma_main.ui` present in the project root (produced by Qt Designer)
- `res.qrc` alongside it — `res_rcc.py` is compiled automatically by `main._ensure_qt_resources()` when stale

> [!NOTE]
> Neither file is covered by `environment.yml`. Missing `ehcma_main.ui` fails `gui._load_ui`; missing `res.qrc` silently skips compilation.

## Installation

```powershell
conda env create -f environment.yml
conda activate cosam-agent
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.6.0+cu118 True
```

> [!TIP]
> **CPU-only** — replace the three `torch*+cu118` lines in `environment.yml` with `torch==2.6.0`, `torchvision==0.21.0`, `torchaudio==2.6.0`, and drop `--extra-index-url`. `dl_training.pick_device` falls back automatically.

`res.qrc` is compiled on demand by `main._ensure_qt_resources()` (searches `PATH`, `<env>/Scripts/`, `site-packages/PySide6/`). `MOLCALC_MAX_CORES` caps the ORCA worker count when `psutil` over-reports cores; delete it on bare metal.

## Quick Start

```powershell
python main.py                              # RUN_STAGE = "gui" by default
```

or dispatch programmatically:

```python
from main import dispatch
dispatch("acin")
```

## Repository Layout

```text
.
├── main.py              CLI entry + stage dispatcher
├── gui.py               PySide6 application
├── ehcma_main.ui        Qt Designer UI (required)
├── res.qrc              Qt resources -> res_rcc.py (auto-compiled)
├── environment.yml
├── mol_calc.py          ORCA + Multiwfn + RDKit pipeline
├── ml_training.py       AutoML over 12 regressors
├── ml_eval.py           SHAP beeswarm / dependence / waterfall / radar
├── dl_training.py       ACIN + BL_MLP baselines
├── acin_analysis.py     Attention spectrum + substructure gallery
├── ablation.py          Feature / attention ablation benchmarks
├── msec.py              VLM + hierarchy-checked CoT agent
│
├── smiles.csv           IN   SMILES + metadata
├── exp_data.csv         IN   device records
├── sams_pred.csv        IN   mixtures to predict
├── ml_inputs.csv        OUT  35-descriptor matrix
├── dl_inputs.csv        OUT  35 + 4860 bimodal matrix
├── attention_weights.csv OUT 4860 ACIN weights
│
└── mol_calc/  ml_models/  dl_models/  plots/  benchmark
```

## Data Format

| File | Required columns |
|---|---|
| `smiles.csv` | `SMILES`, `Symmetric`, `Chain_length`, `HeteroEndFrag` |
| `exp_data.csv` | `SAM_1`, `SAM_2`, `SAMs_ratio`, `Bandgap`, `Mixture`, `MACl_additive`, `PCE` |
| `sams_pred.csv` | Same as above **minus** `PCE` |

## Pipeline Stages

Set `RUN_STAGE` in `main.py` or call `dispatch("<stage>")`.

| Stage | Input -> Output | Notes |
|---|---|---|
| `mol_calc` | `smiles.csv` + `exp_data.csv` -> `mol_prop.csv`, `ml_inputs.csv`, `dl_inputs.csv` | Requires ORCA + Multiwfn |
| `ml` | `ml_inputs.csv` -> `ml_models/*.pkl` | `MODELS = "lr, cat"` |
| `eval` | + `ml_models/` -> `plots/{beeswarm,deps,score_hist,accs}` | `EVAL_MODEL`, `TOP_N`, `FITTING` |
| `predict` | + `sams_pred.csv` -> `plots/waterfall/<model>/` | Uses the CatBoost pickle |
| `acin` | `dl_inputs.csv` -> `dl_models/ACIN.pth`, `attention_weights.csv`, `plots/substructures/` | `RENDER_ONLY=True` skips training |
| `feature_ablation` | `dl_inputs.csv` -> `benchmark/feature_ablation/` | 5 seeds x 3 views x {MLP, CatBoost} + ACIN |
| `attention_deletion` | `ACIN.pth` -> `benchmark/attention_deletion/` | Selective vs random deletion |
| `attention_retrain` | `dl_inputs.csv` + `ACIN.pth` -> `benchmark/attention_retrain/` | Retrain after deletion |
| `all_ablation` | - | Runs the three ablation stages sequentially |
| `gui` | - | Launches PySide6 app |
| `all` | - | Runs every stage not gated by `SKIP_*` |

```text
mol_calc ──┬──► ml ─────┬──────► eval ───────┐
           │            │                    ├──► MSEC agent
           │            └──────► predict     │
           │                                 │
           └──► acin ────► attention_weights─┘
```

## Configuration

| Constant | Default | Purpose |
|---|---|---|
| `RUN_STAGE` | `"gui"` | Target stage |
| `SEED` | `42` | Global seed (Python / NumPy / PyTorch) |
| `METHOD`, `BASIS` | `B3LYP`, `def2-TZVP` | ORCA level of theory |
| `MODELS` | `"lr, cat"` | Model tags (see `MODEL_FLAG`) |
| `INIT_POINTS`, `N_ITER` | `10`, `50` | Bayesian-optimization budget per model |
| `EVAL_MODEL`, `TOP_N`, `FITTING` | `CatBoostRegressor`, `20`, `gam` | SHAP top-K and trend fit |
| `EPOCHS`, `EARLY_STOP`, `LR`, `DROPOUT` | `1000`, `40`, `1e-3`, `0.3` | ACIN / BL_MLP |
| `TOP_K`, `FIG_ROW`, `RENDER_ONLY` | `100`, `10`, `False` | Substructure gallery |
| `SKIP_*` | `True` / `False` | Per-stage skip switch (`SKIP_<STAGE>`); `True` skips that stage, `False` runs it |
| `LAUNCH_GUI` | `False` | Auto-open GUI on completion |

## Reproducibility

| Layer | Mechanism |
|---|---|
| Global | `set_seed(SEED)` before any model creation |
| AutoML | `BayesianOptimization(random_state=SEED)`; estimators seeded individually |
| ACIN | Attribute declaration order in `ACIN.__init__` fixes initial weights; `dl_worker` re-seeds Python/NumPy/CPU/CUDA and pins cuDNN deterministic |
| Checkpointing | `ACIN_checkpoint.pth` stores model + optimizer + scheduler + RNG states |
| Feature ablation | 5 seeds, resumable via `benchmark/feature_ablation/task_results.csv` |

> [!WARNING]
> Renaming or reordering `ACIN` attributes changes RNG consumption, breaking bit-for-bit reproducibility and `.pth` compatibility.

## MSEC Agent

The Agent page seeds three fields from environment variables at startup. Set them **before** launching `python main.py`:

```powershell
$env:AGENT_API_KEY    = "sk-..."
$env:AGENT_BASE_URL   = "https://<provider>/v1"
$env:AGENT_MODEL_NAME = "<vision-capable model>"
```

| Variable | Purpose |
|---|---|
| `AGENT_API_KEY` | OpenAI-compatible API key |
| `AGENT_BASE_URL` | Endpoint root, e.g. `https://api.openai.com/v1` |
| `AGENT_MODEL_NAME` | Vision-capable model name |

Persist with `conda env config vars set <VAR>=<value>` and re-activate. Missing values leave the field blank (startup warning; fields remain editable). Endpoint settings are captured at process spawn — restart after changes. `max_tokens = 0` disables the token cap; changing it requires a new chat.

## Citation

If you use this code or the agent-guided Co-SAM workflow, please cite the manuscript (currently under review):

> Z. Zhong et al. *An Explainable Agent Guides Hybrid Interface Design for Efficient Perovskite Tandem Solar Cells*. Manuscript under review (2026).

```bibtex
@unpublished{zhong2026ehcma,
  author = {Zhong, Zijian and others},
  title  = {An Explainable Agent Guides Hybrid Interface Design for Efficient
            Perovskite Tandem Solar Cells},
  year   = {2026},
  note   = {Manuscript under review}
}
```

**Key methods**

- CatBoost — Prokhorenkova et al., *NeurIPS* **31** (2018)
- SHAP — Lundberg & Lee, *NeurIPS* **30** (2017)
- Klekota-Roth — Klekota & Roth, *Bioinformatics* **24**, 2518 (2008)
- Bayesian optimization — Snoek et al., *NeurIPS* **25** (2012)

## License

MIT — see `LICENSE`.
