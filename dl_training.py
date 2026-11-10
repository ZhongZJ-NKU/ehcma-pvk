"""ACIN: Adaptive Cross-Modal Interaction Network for Co-SAM PCE regression.

Input:
    x_train, y_train, x_test, y_test -- DataFrame / Series; first 35 columns
                                        are handcrafted physicochemical
                                        descriptors, the remaining 4860 are
                                        Klekota-Roth substructure bits.
    device                            -- ``"cuda"`` or ``"cpu"``.
    lr, epoch, batch_size, dropout    -- optimization schedule.
    early_stop_threshold              -- epochs without val-loss improvement;
                                        ``0`` disables early stopping.
    resume                            -- if True and a checkpoint exists at
                                        ``<base_dir>/dl_models/ACIN_checkpoint.pth``,
                                        restore model/optimizer/scheduler/best
                                        state and continue from the next epoch.
    checkpoint_interval               -- save a full checkpoint every N epochs
                                        in addition to every best epoch; set
                                        ``0`` for best-only saves.
    queue                             -- optional; receives one of:
                                            int             epoch index
                                            DataFrame       [R2, MSE, MAE, Loss]
                                            "_end_"         normal / early stop
                                            "_error_<msg>"  training crashed
    base_dir                          -- root for ``dl_models/``.

Output:
    ACINTraining() -> (acin_weights_df, model_path)
        acin_weights_df : DataFrame[krfp_idx, weights], 4860 rows;
                               ``weights`` is the batch-averaged substructure
                               score ∈ [-1, 1] under the best model.
        model_path           : <base_dir>/dl_models/ACIN.pth.
        Side effect          : writes ``acin_weights.csv`` in the CWD on
                               every best epoch.

    BL_MLP_training() -> (None, model_path)
        model_path           : <base_dir>/dl_models/mlp.pth.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader


ATTN_WEIGHTS_FILE = "acin_weights.csv"
ACIN_CKPT = "ACIN_checkpoint.pth"
MLP_CKPT = "mlp_checkpoint.pth"

_LOG_WIDTH = 118
_LOG_BAR = "─" * _LOG_WIDTH


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TrainDataset(Dataset):
    """Split each row into (35 handcrafted features | 4860 KRFP bits)."""

    def __init__(self, x_train, y_train):
        x_feats_train = x_train[:, :35]
        x_krfp_train = x_train[:, 35:]
        self.x_feats_train = torch.tensor(x_feats_train, dtype=torch.float32)
        self.x_krfp_train = torch.tensor(x_krfp_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)

    def __getitem__(self, idx):
        return (
            self.x_feats_train[idx].unsqueeze(0),
            self.x_krfp_train[idx].unsqueeze(0),
            self.y_train[idx].unsqueeze(0).unsqueeze(1),
        )

    def __len__(self):
        return self.y_train.shape[0]


class TestDataset(Dataset):
    def __init__(self, x_test, y_test):
        x_feats_test = x_test[:, :35]
        x_krfp_test = x_test[:, 35:]
        self.x_feats_test = torch.tensor(x_feats_test, dtype=torch.float32)
        self.x_krfp_test = torch.tensor(x_krfp_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.float32)

    def __getitem__(self, idx):
        return (
            self.x_feats_test[idx].unsqueeze(0),
            self.x_krfp_test[idx].unsqueeze(0),
            self.y_test[idx].unsqueeze(0).unsqueeze(1),
        )

    def __len__(self):
        return self.y_test.shape[0]


class BL_MLP_TrainDataset(Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)

    def __getitem__(self, idx):
        return (
            self.x_train[idx].unsqueeze(0),
            self.y_train[idx].unsqueeze(0).unsqueeze(1),
        )

    def __len__(self):
        return self.y_train.shape[0]


class BL_MLP_TestDataset(Dataset):
    def __init__(self, x_test, y_test):
        self.x_test = torch.tensor(x_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.float32)

    def __getitem__(self, idx):
        return (
            self.x_test[idx].unsqueeze(0),
            self.y_test[idx].unsqueeze(0).unsqueeze(1),
        )

    def __len__(self):
        return self.y_test.shape[0]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ACIN(nn.Module):
    """Adaptive Cross-Modal Interaction Network."""

    def __init__(self, feat_dim, mol_dim, drop_rate, hidden_layer=None):
        super(ACIN, self).__init__()
        if hidden_layer is None:
            hidden_layer = [2048, 1024, 512]
        self.feat_dim = feat_dim
        self.mol_dim = mol_dim
        self.drop_rate = drop_rate

        # Attention Projector: (physchem + substructure feedback) -> 4860 scores.
        self.attn_projector = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, self.mol_dim),
            nn.LayerNorm(self.mol_dim),
            nn.Tanh(),
        )

        self.norm1 = nn.LayerNorm(mol_dim)   # KRFP normalisation
        self.norm2 = nn.LayerNorm(feat_dim)  # physchem normalisation
        self.norm3 = nn.LayerNorm(mol_dim)   # fused normalization

        # Feedback Modulator: substructure context -> physchem space.
        self.fb_modulator = nn.Sequential(
            nn.Linear(mol_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, feat_dim),
            nn.Tanh(),
        )

        # MLP Regressor.
        layers = []
        prev_dim = mol_dim
        for dim in hidden_layer:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(self.drop_rate),
            ])
            prev_dim = dim
        self.mlp_regressor = nn.Sequential(*layers)
        self.pred_output = nn.Linear(prev_dim, 1)

    def forward(self, krfp_bits, physchem_feats):
        """Score and reweight substructures conditioned on physicochemical context.

        Data flow (each step maps 1-to-1 to the reference implementation):

            1. norm_substructures = LayerNorm(krfp_bits)
            2. norm_feats         = LayerNorm(physchem_feats)
            3. feedback           = MLP_feedback(norm_substructures)
            4. enriched_feats     = norm_feats + feedback
            5. scores             = MLP_scorer(enriched_feats)        <- score 4860 bits
            6. weighted           = norm_substructures * scores        <- high keep, low suppress
            7. fused              = LayerNorm(norm_substructures + weighted)
            8. pce                = head(MLP_regressor(fused))

        Args:
            krfp_bits:      [B, 1, 4860] or [B, 4860] -- substructure bits.
            physchem_feats: [B, 1, 35]   or [B, 35]   -- descriptors.

        Returns:
            pce:    [B, 1]  -- predicted PCE.
            scores: [4860]  -- batch-averaged substructure scores.
        """
        krfp_bits = krfp_bits.squeeze().squeeze()
        physchem_feats = physchem_feats.squeeze().squeeze()

        # ---- 1. Normalize both modalities ----
        norm_substructures = self.norm1(krfp_bits)
        norm_feats = self.norm2(physchem_feats)

        # ---- 2. Substructure context refines the descriptor branch ----
        substructure_feedback = self.fb_modulator(norm_substructures)
        enriched_feats = norm_feats + substructure_feedback

        # ---- 3. Score each of the 4860 substructure bits given that context
        #         (score > 0: keep / amplify;  score < 0: suppress) ----
        scores = self.attn_projector(enriched_feats)

        # ---- 4. Higher scores retain more of the substructure bit ----
        weighted_substructures = norm_substructures * scores.squeeze()

        # ---- 5. Residual + regress ----
        fused = self.norm3(norm_substructures + weighted_substructures)
        pce = self.pred_output(self.mlp_regressor(fused))
        return pce, scores.mean(dim=0)


class BL_MLP(nn.Module):
    """Plain MLP baseline over the concatenated feature vector."""

    def __init__(self, dim, hidden_dims=None, dropout=0.3):
        super(BL_MLP, self).__init__()
        if hidden_dims is None:
            hidden_dims = [2048, 1024, 512]
        layers = []
        prev_dim = dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.bl_mlp = nn.Sequential(*layers)

    def forward(self, input_data):
        input_data = input_data.squeeze(1)
        return self.bl_mlp(input_data)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _fmt_duration(seconds):
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _n_params(model):
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _print_header(tag, device, batch_size, lr, dropout, max_epochs, model, verbose=True):
    """One-time configuration summary printed before the first epoch."""
    if not verbose:
        return
    print(_LOG_BAR)
    print(
        f" {tag}  device={device}  batch={batch_size}  "
        f"lr={lr:.2e}  dropout={dropout:.2f}  "
        f"params={_n_params(model)}  max_epochs={max_epochs}"
    )
    print(_LOG_BAR)


def _print_progress(epoch, total, train_loss, test_loss,
                    best_mse_tr, best_mse_te, best_mae_tr, best_mae_te,
                    best_r2_tr, best_r2_te,
                    lr, sec_per_epoch, eta, is_best, verbose=True):
    """Single-line progress row; ``*`` marks a new best epoch."""
    if not verbose:
        return
    marker = "*" if is_best else " "
    print(
        f"{marker}[{epoch:>5}/{total}]  "
        f"loss {train_loss:.4f}/{test_loss:.4f}  "
        f"mse {best_mse_tr:.3f}/{best_mse_te:.3f}  "
        f"mae {best_mae_tr:.3f}/{best_mae_te:.3f}  "
        f"r2 {best_r2_tr:.3f}/{best_r2_te:.3f}  "
        f"lr {lr:.1e}  "
        f"{sec_per_epoch:.1f}s  "
        f"ETA {eta}"
    )


def _print_ending(symbol, tag, epoch, total, best_epoch, best_test, verbose=True):
    """Final banner for ``✓ Finished`` / ``⚠ Early stop`` / ``✗ Failed``."""
    if not verbose:
        return
    print(_LOG_BAR)
    print(
        f" {symbol} {tag}  epoch {epoch}/{total}  "
        f"best_epoch {best_epoch}  best_test {best_test:.6f}"
    )
    print(_LOG_BAR)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(path, epoch, model, optimizer, scheduler,
                     best_test, best_train, best_metrics, early_stop_counter):
    tmp = f"{path}.tmp"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_test": best_test,
        "best_train": best_train,
        "best_metrics": dict(best_metrics),
        "early_stop_counter": early_stop_counter,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }, tmp)
    os.replace(tmp, path)


def _load_checkpoint(path, model, optimizer, scheduler, device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if "torch_rng_state" in ckpt:
        try:
            torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        except Exception:
            pass
    if "numpy_rng_state" in ckpt:
        try:
            np.random.set_state(ckpt["numpy_rng_state"])
        except Exception:
            pass
    return ckpt


def _remove_checkpoint(path):
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_base_dir(base_dir):
    if base_dir is not None:
        return base_dir
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return os.path.dirname(os.path.realpath(sys.executable))
    return os.path.dirname(os.path.realpath(os.path.abspath(__file__)))


def _metrics(y_true, y_pred):
    return (
        r2_score(y_true, y_pred),
        mean_squared_error(y_true, y_pred),
        mean_absolute_error(y_true, y_pred),
    )


def _save_attn_weights(scores, path=ATTN_WEIGHTS_FILE):
    """Persist ``acin_weights.csv``; no-op when ``scores`` is None."""
    if scores is None:
        return
    df = pd.DataFrame(scores)
    df.insert(0, "krfp", [f"krfp_{idx}" for idx in range(4860)])
    df.columns = ["krfp_idx", "weights"]
    df.to_csv(path, index=False)


def _predict_acin(model, x_test_df, x_train_df, device):
    """Evaluate ``model`` on both splits and return (test_pred, train_pred, scores)."""
    with torch.no_grad():
        y_pre_test, scores = model(
            torch.tensor(x_test_df.iloc[:, 35:].to_numpy(),
                         device=device, dtype=torch.float32),
            torch.tensor(x_test_df.iloc[:, :35].to_numpy(),
                         device=device, dtype=torch.float32),
        )
        y_pre_test = y_pre_test.squeeze(1).cpu().numpy()
        if scores is not None:
            scores = scores.cpu().numpy()

        y_pre_train, _ = model(
            torch.tensor(x_train_df.iloc[:, 35:].to_numpy(),
                         device=device, dtype=torch.float32),
            torch.tensor(x_train_df.iloc[:, :35].to_numpy(),
                         device=device, dtype=torch.float32),
        )
        y_pre_train = y_pre_train.squeeze(1).cpu().numpy()
    return y_pre_test, y_pre_train, scores


# ---------------------------------------------------------------------------
# ACIN training
# ---------------------------------------------------------------------------

def ACINTraining(x_train=None, y_train=None, x_test=None, y_test=None,
                 device=None, lr=0.01, epoch=1000, batch_size=32,
                 dropout=0.3, early_stop_threshold=40,
                 queue=None, base_dir=None,
                 resume=False, checkpoint_interval=10,
                 attn_out_path=None, verbose=True):
    """Train the full ACIN model."""
    base_dir = _resolve_base_dir(base_dir)
    models_dir = os.path.join(base_dir, "dl_models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "ACIN.pth")
    ckpt_path = os.path.join(models_dir, ACIN_CKPT)

    if not resume:
        _remove_checkpoint(ckpt_path)

    train_set = TrainDataset(x_train.to_numpy(), y_train.to_numpy())
    train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_set = TestDataset(x_test.to_numpy(), y_test.to_numpy())
    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    if early_stop_threshold == 0:
        early_stop_threshold = epoch

    model = ACIN(feat_dim=35, mol_dim=4860, drop_rate=dropout).to(device)

    criterion = nn.SmoothL1Loss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epoch // 2, eta_min=1e-6)

    start_epoch = 0
    best_test_loss = float("inf")
    best_train_loss = 0.0
    best_r2_test = best_r2_train = 0.0
    best_mse_test = best_mse_train = 0.0
    best_mae_test = best_mae_train = 0.0
    best_metrics = {"r2_tr": 0.0, "mse_tr": 0.0, "mae_tr": 0.0,
                    "r2_te": 0.0, "mse_te": 0.0, "mae_te": 0.0}
    early_stop_counter = 0
    best_epoch = 0

    if resume and os.path.isfile(ckpt_path):
        try:
            ckpt = _load_checkpoint(ckpt_path, model, optimizer, lr_scheduler, device)
            start_epoch = ckpt["epoch"] + 1
            best_test_loss = ckpt["best_test"]
            best_train_loss = ckpt["best_train"]
            best_metrics = dict(ckpt["best_metrics"])
            early_stop_counter = ckpt["early_stop_counter"]
            best_epoch = ckpt["epoch"] + 1
            best_r2_train = best_metrics.get("r2_tr", 0.0)
            best_r2_test = best_metrics.get("r2_te", 0.0)
            best_mse_train = best_metrics.get("mse_tr", 0.0)
            best_mse_test = best_metrics.get("mse_te", 0.0)
            best_mae_train = best_metrics.get("mae_tr", 0.0)
            best_mae_test = best_metrics.get("mae_te", 0.0)
            if verbose:
                print(f"Resumed from checkpoint at epoch {start_epoch}/{epoch}")
            if queue is not None:
                queue.put(f"Resumed ACIN from epoch {start_epoch}/{epoch}")
        except Exception as exc:
            if verbose:
                print(f"Failed to restore checkpoint ({exc}); starting fresh.")
            start_epoch = 0

    scores = None
    t_start = time.perf_counter()
    header_printed = False
    attn_target = attn_out_path if attn_out_path is not None else ATTN_WEIGHTS_FILE

    try:
        for i in range(start_epoch, epoch):
            t_epoch_start = time.perf_counter()

            if queue is not None:
                queue.put(int(i))

            if not header_printed:
                _print_header("ACIN", device, batch_size, lr, dropout,
                              epoch, model, verbose=verbose)
                header_printed = True

            # ---- train ----
            model.train()
            train_loss = 0.0
            for feats, krfps, target in train_dataloader:
                optimizer.zero_grad()
                feats = feats.to(device)
                krfps = krfps.to(device)
                target = target.reshape(-1, 1).to(device)
                output, _ = model(krfp_bits=krfps, physchem_feats=feats)
                batch_loss = criterion(output, target)
                train_loss += batch_loss.item()
                batch_loss.backward()
                optimizer.step()
            train_loss /= len(train_dataloader)

            # ---- eval ----
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for feats, krfps, target in test_dataloader:
                    feats = feats.to(device)
                    krfps = krfps.to(device)
                    target = target.reshape(-1, 1).to(device)
                    output, _ = model(krfp_bits=krfps, physchem_feats=feats)
                    batch_loss = criterion(output, target)
                    test_loss += batch_loss.item()
            test_loss /= len(test_dataloader)

            lr_scheduler.step()
            t_epoch = time.perf_counter() - t_epoch_start

            is_best = test_loss < best_test_loss
            if is_best:
                best_test_loss = test_loss
                best_train_loss = train_loss
                best_epoch = i + 1
                early_stop_counter = 0

                y_pre_test, y_pre_train, scores = _predict_acin(
                    model, x_test, x_train, device,
                )
                (best_r2_test, best_mse_test, best_mae_test) = _metrics(y_test, y_pre_test)
                (best_r2_train, best_mse_train, best_mae_train) = _metrics(y_train, y_pre_train)
                best_metrics = {
                    "r2_tr": best_r2_train, "mse_tr": best_mse_train, "mae_tr": best_mae_train,
                    "r2_te": best_r2_test, "mse_te": best_mse_test, "mae_te": best_mae_test,
                }

                if queue is not None:
                    queue.put(pd.DataFrame(
                        [[best_r2_test, best_mse_test, best_mae_test, best_test_loss]],
                        columns=["R2", "MSE", "MAE", "Loss"],
                    ))

                torch.save(model.state_dict(), model_path)
                _save_attn_weights(scores, path=attn_target)
                _save_checkpoint(ckpt_path, i, model, optimizer, lr_scheduler,
                                 best_test_loss, best_train_loss, best_metrics,
                                 early_stop_counter)
            else:
                early_stop_counter += 1
                if checkpoint_interval > 0 and (i + 1) % checkpoint_interval == 0:
                    _save_checkpoint(ckpt_path, i, model, optimizer, lr_scheduler,
                                     best_test_loss, best_train_loss, best_metrics,
                                     early_stop_counter)

            should_print = is_best or (i + 1) % 10 == 0 or (i + 1) == 1
            if should_print:
                elapsed = time.perf_counter() - t_start
                done = i + 1 - start_epoch
                sec_per_epoch = elapsed / max(1, done)
                eta = sec_per_epoch * (epoch - i - 1)
                _print_progress(
                    i + 1, epoch, train_loss, test_loss,
                    best_mse_train, best_mse_test,
                    best_mae_train, best_mae_test,
                    best_r2_train, best_r2_test,
                    optimizer.param_groups[0]["lr"],
                    sec_per_epoch, _fmt_duration(eta), is_best,
                    verbose=verbose,
                )

            if early_stop_counter >= early_stop_threshold:
                _print_ending("⚠", "Early stop", i + 1, epoch,
                              best_epoch, best_test_loss, verbose=verbose)
                break

            if i + 1 == epoch:
                _print_ending("✓", "Finished", i + 1, epoch,
                              best_epoch, best_test_loss, verbose=verbose)
                break

    except Exception as exc:
        _print_ending("✗", f"Failed: {exc}", 0, epoch, best_epoch, best_test_loss, verbose=verbose)
        if queue is not None:
            queue.put(f"_error_ACIN training failed: {exc}")
        raise
    else:
        if queue is not None:
            queue.put("_end_")
        _remove_checkpoint(ckpt_path)

    if scores is None:
        _, _, scores = _predict_acin(model, x_test, x_train, device)
        _save_attn_weights(scores, path=attn_target)

    attn_df = pd.DataFrame(scores)
    attn_df.insert(0, "krfp", [f"krfp_{idx}" for idx in range(4860)])
    attn_df.columns = ["krfp_idx", "weights"]
    return attn_df, model_path, best_metrics


# ---------------------------------------------------------------------------
# MLP baseline training
# ---------------------------------------------------------------------------

def MLPTraining(x_train=None, y_train=None, x_test=None, y_test=None,
                device=None, lr=0.01, epoch=1000, early_stop_threshold=40,
                dropout=0.3,
                base_dir=None, resume=False, checkpoint_interval=10,
                verbose=True):
    base_dir = _resolve_base_dir(base_dir)
    models_dir = os.path.join(base_dir, "dl_models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "mlp.pth")
    ckpt_path = os.path.join(models_dir, MLP_CKPT)

    if not resume:
        _remove_checkpoint(ckpt_path)

    train_set = BL_MLP_TrainDataset(x_train.to_numpy(), y_train.to_numpy())
    train_dataloader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
    test_set = BL_MLP_TestDataset(x_test.to_numpy(), y_test.to_numpy())
    test_dataloader = DataLoader(dataset=test_set, batch_size=32, shuffle=True)

    dim = x_train.shape[1]
    model = BL_MLP(dim=dim, dropout=dropout).to(device)

    criterion = nn.SmoothL1Loss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epoch // 2, eta_min=1e-6)

    start_epoch = 0
    best_test_loss = float("inf")
    best_train_loss = 0.0
    best_r2_test = best_r2_train = 0.0
    best_mse_test = best_mse_train = 0.0
    best_mae_test = best_mae_train = 0.0
    best_metrics = {"r2_tr": 0.0, "mse_tr": 0.0, "mae_tr": 0.0,
                    "r2_te": 0.0, "mse_te": 0.0, "mae_te": 0.0}
    early_stop_counter = 0
    best_epoch = 0

    if resume and os.path.isfile(ckpt_path):
        try:
            ckpt = _load_checkpoint(ckpt_path, model, optimizer, lr_scheduler, device)
            start_epoch = ckpt["epoch"] + 1
            best_test_loss = ckpt["best_test"]
            best_train_loss = ckpt["best_train"]
            best_metrics = dict(ckpt["best_metrics"])
            early_stop_counter = ckpt["early_stop_counter"]
            best_epoch = ckpt["epoch"] + 1
            best_r2_train = best_metrics.get("r2_tr", 0.0)
            best_r2_test = best_metrics.get("r2_te", 0.0)
            best_mse_train = best_metrics.get("mse_tr", 0.0)
            best_mse_test = best_metrics.get("mse_te", 0.0)
            best_mae_train = best_metrics.get("mae_tr", 0.0)
            best_mae_test = best_metrics.get("mae_te", 0.0)
            if verbose:
                print(f"Resumed from checkpoint at epoch {start_epoch}/{epoch}")
        except Exception as exc:
            if verbose:
                print(f"Failed to restore checkpoint ({exc}); starting fresh.")
            start_epoch = 0

    t_start = time.perf_counter()
    header_printed = False

    try:
        for i in range(start_epoch, epoch):
            t_epoch_start = time.perf_counter()

            if not header_printed:
                _print_header("BL_MLP", device, 32, lr, dropout, epoch, model, verbose=verbose)
                header_printed = True

            model.train()
            train_loss = 0.0
            for xtrain, target in train_dataloader:
                optimizer.zero_grad()
                xtrain = xtrain.to(device)
                target = target.reshape(-1, 1).to(device)
                output = model(xtrain)
                batch_loss = criterion(output, target)
                train_loss += batch_loss.item()
                batch_loss.backward()
                optimizer.step()
            train_loss /= len(train_dataloader)

            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for xtest, target in test_dataloader:
                    xtest = xtest.to(device)
                    target = target.reshape(-1, 1).to(device)
                    output = model(xtest)
                    batch_loss = criterion(output, target)
                    test_loss += batch_loss.item()
            test_loss /= len(test_dataloader)

            lr_scheduler.step()
            t_epoch = time.perf_counter() - t_epoch_start

            is_best = test_loss < best_test_loss
            if is_best:
                best_test_loss = test_loss
                best_train_loss = train_loss
                best_epoch = i + 1
                early_stop_counter = 0

                with torch.no_grad():
                    y_pre_test = model(torch.tensor(
                        x_test.to_numpy(), device=device, dtype=torch.float32,
                    )).squeeze(1).cpu().numpy()
                    y_pre_train = model(torch.tensor(
                        x_train.to_numpy(), device=device, dtype=torch.float32,
                    )).squeeze(1).cpu().numpy()
                (best_r2_test, best_mse_test, best_mae_test) = _metrics(y_test, y_pre_test)
                (best_r2_train, best_mse_train, best_mae_train) = _metrics(y_train, y_pre_train)
                best_metrics = {
                    "r2_tr": best_r2_train, "mse_tr": best_mse_train, "mae_tr": best_mae_train,
                    "r2_te": best_r2_test, "mse_te": best_mse_test, "mae_te": best_mae_test,
                }

                torch.save(model.state_dict(), model_path)
                _save_checkpoint(ckpt_path, i, model, optimizer, lr_scheduler,
                                 best_test_loss, best_train_loss, best_metrics,
                                 early_stop_counter)
            else:
                early_stop_counter += 1
                if checkpoint_interval > 0 and (i + 1) % checkpoint_interval == 0:
                    _save_checkpoint(ckpt_path, i, model, optimizer, lr_scheduler,
                                     best_test_loss, best_train_loss, best_metrics,
                                     early_stop_counter)

            should_print = is_best or (i + 1) % 10 == 0 or (i + 1) == 1
            if should_print:
                elapsed = time.perf_counter() - t_start
                done = i + 1 - start_epoch
                sec_per_epoch = elapsed / max(1, done)
                eta = sec_per_epoch * (epoch - i - 1)
                _print_progress(
                    i + 1, epoch, train_loss, test_loss,
                    best_mse_train, best_mse_test,
                    best_mae_train, best_mae_test,
                    best_r2_train, best_r2_test,
                    optimizer.param_groups[0]["lr"],
                    sec_per_epoch, _fmt_duration(eta), is_best,
                    verbose=verbose,
                )

            if early_stop_counter >= early_stop_threshold or i + 1 == epoch:
                if i + 1 == epoch and early_stop_counter < early_stop_threshold:
                    _print_ending("✓", "Finished", i + 1, epoch,
                                  best_epoch, best_test_loss, verbose=verbose)
                else:
                    _print_ending("⚠", "Early stop", i + 1, epoch,
                                  best_epoch, best_test_loss, verbose=verbose)
                break

    except Exception as exc:
        _print_ending("✗", f"Failed: {exc}", 0, epoch, best_epoch, best_test_loss, verbose=verbose)
        raise
    else:
        _remove_checkpoint(ckpt_path)

    return None, model_path