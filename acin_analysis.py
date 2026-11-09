"""ACIN attention-spectrum bar plot with interactive hover and scroll.

Usage:
    bar = ACINSpectrumBar(csv_path="attn_weights.csv")
    bar.run()

Features:
    - Interactive zoom/pan via scroll and drag.
    - Hover tooltips for specific substructure weights.
    - Static export support.
    - Substructure gallery rendering via RDKit.
"""

import io
import math
import os
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.backend_bases import cursors
from matplotlib.collections import PolyCollection
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from skfp.fingerprints import KlekotaRothFingerprint

# ---------------------------------------------------------------------------
# Global Font Configuration
# ---------------------------------------------------------------------------
FONT_FAMILY = "Arial"
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [FONT_FAMILY, "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = FONT_FAMILY
matplotlib.rcParams["mathtext.it"] = f"{FONT_FAMILY}:italic"
matplotlib.rcParams["mathtext.bf"] = f"{FONT_FAMILY}:bold"


# ---------------------------------------------------------------------------
# Substructure Gallery Helpers
# ---------------------------------------------------------------------------
def _find_font() -> Optional[str]:
    """Select a CJK-compatible bold font across platforms (None => RDKit default)."""
    candidates = (
        Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    )
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _draw_krfp_png(smarts: str, size_hint: int = 500) -> Image.Image:
    """Render a Klekota-Roth SMARTS pattern to a cropped PIL image."""
    mol = Chem.MolFromSmarts(smarts)
    if mol is None:
        raise ValueError(f"Invalid SMARTS: {smarts}")
    if mol.GetNumConformers() == 0:
        rdDepictor.Compute2DCoords(mol)

    conf = mol.GetConformer()
    xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
    ys = [conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]
    width = max(size_hint, int((max(xs) - min(xs)) * 50 * 2))
    height = max(size_hint, int((max(ys) - min(ys)) * 50 * 2))
    size = max(width, height)

    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    opts = drawer.drawOptions()
    opts.fontSize = 20
    font = _find_font()
    if font:
        opts.fontFile = font
    opts.bondLineWidth = 8
    opts.explicitMethyl = True
    opts.showAtomTerminalSymbols = True
    opts.multipleBondOffset = 0.2
    opts.scaleBondWidth = False
    opts.wedgeDashedBonds = False
    opts.splitBonds = False
    opts.fixedBondLength = -1
    opts.baseFontSize = 2.5
    opts.useDefaultAtomPalette()
    opts.padding = 0.1
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()

    img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
    bbox = img.getbbox()
    return img.crop(bbox) if bbox else img


def render_attention_gallery(
    attn_df: pd.DataFrame,
    plots_dir,
    struct_dir,
    show_mols: int = 100,
    fig_row: int = 10,
    log_fn: Optional[Callable[[str, str], None]] = None,
) -> Path:
    """Render top 'show_mols' substructures as a grid image."""
    plots_dir = Path(plots_dir)
    struct_dir = Path(struct_dir)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(struct_dir, exist_ok=True)

    order = np.argsort(-attn_df["weights"].to_numpy())
    weights = attn_df["weights"].to_numpy()
    smarts_list = KlekotaRothFingerprint().get_feature_names_out()

    # Clear previous crops
    for p in struct_dir.iterdir():
        if p.is_file():
            p.unlink()

    fig_col = math.ceil(show_mols / fig_row)
    fig, axes = plt.subplots(fig_row, fig_col, figsize=(fig_col * 2, fig_row * 2.2))
    axes = np.atleast_2d(axes)

    for idx in range(fig_row * fig_col):
        row, col = divmod(idx, fig_col)
        ax = axes[row, col]
        if idx >= show_mols:
            ax.axis("off")
            continue

        krfp_idx = int(order[idx])
        weight = float(weights[krfp_idx])
        try:
            img = _draw_krfp_png(smarts_list[krfp_idx])
        except ValueError as exc:
            if log_fn is not None:
                log_fn("acin", f"Skip {krfp_idx}: {exc}")
            ax.axis("off")
            continue

        img.save(struct_dir / f"{idx}-krfp{krfp_idx}-score{weight:.3f}.png", "PNG", optimize=True)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(f"{weight:.3f}", fontsize=14, labelpad=-15, fontweight="bold", color="black")
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        ax.text(
            x=(x_max - x_min) / 2, y=(y_max - y_min) / 12,
            s=f"KRFP {krfp_idx}",
            ha="center", va="center", fontsize=16,
            fontweight="bold", color="black",
        )

    plt.tight_layout()
    out_png = plots_dir / "Top-100-strs.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight", edgecolor="black")
    plt.close(fig)
    return out_png


def render_top_k_substructures(
    attn_df: pd.DataFrame,
    out_dir,
    top_k: int = 100,
    dpi: int = 100,
    show_index_text: bool = True,
    show_weight_text: bool = True,
    text_format: str = "KRFP ",
    log_fn: Optional[Callable[[str, str], None]] = None,
) -> list:
    """Render top 'top_k' substructures as individual PNGs."""
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for p in out_dir.iterdir():
        if p.is_file():
            p.unlink()

    order = np.argsort(-attn_df["weights"].to_numpy())
    weights = attn_df["weights"].to_numpy()
    smarts_list = KlekotaRothFingerprint().get_feature_names_out()

    written = []
    for rank in range(top_k):
        krfp_idx = int(order[rank])
        weight = float(weights[krfp_idx])
        try:
            img = _draw_krfp_png(smarts_list[krfp_idx])
        except ValueError as exc:
            if log_fn is not None:
                log_fn("acin", f"Skip {krfp_idx}: {exc}")
            continue

        size = max(img.size) if hasattr(img, "size") else 500
        fig, ax = plt.subplots(figsize=(size / dpi, size / dpi))
        ax.set_facecolor("#F2F2F2")
        ax.imshow(img)
        ax.axis("off")

        if show_index_text:
            ax.text(0.5, 0.9, s=f"{text_format}{krfp_idx}", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=28, fontweight="bold", color="black")
        if show_weight_text:
            ax.text(0.5, 0.1, s=f"{weight:.3f}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=24, fontweight="bold", color="black")

        path = out_dir / f"{rank}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        written.append(path)

    return written


# ---------------------------------------------------------------------------
# Main Plotting Class
# ---------------------------------------------------------------------------
class ACINSpectrumBar:
    """Renders an ACIN attention-weight vector as a colored spectrum bar."""

    # --- Data ---
    DIMS = 4860
    CMAP = "RdBu_r"
    VMIN, VMAX = -1.0, 1.0

    # --- Text Content (empty string to hide) ---
    TITLE_TEXT = ""
    XLABEL_TEXT = "Substructure index"
    CBAR_LABEL = "Weight"

    # --- Font Sizes (effective size = base * FONT_SCALE) ---
    TITLE_FONTSIZE = 11
    XLABEL_FONTSIZE = 12
    CBAR_LABEL_FONTSIZE = 12
    CB_TICK_LABELSIZE = 10
    AX_TICK_LABELSIZE = 10
    HOVER_FONTSIZE = 9

    FONT_SCALE = 1.0
    FONT_WEIGHT = "normal"

    # --- Font Weight Overrides (None to use FONT_WEIGHT) ---
    TITLE_FONT_WEIGHT = None
    XLABEL_FONT_WEIGHT = None
    AX_TICK_FONT_WEIGHT = None
    CBAR_LABEL_FONT_WEIGHT = None
    CB_TICK_FONT_WEIGHT = None

    # --- Layout Paddings ---
    TITLE_PAD = 10
    XLABEL_PAD = 4
    AX_TICK_PAD = 2
    CB_TICK_PAD = 2
    CBAR_LABEL_PAD = -10
    CBAR_LABEL_ROTATION = 270
    CBAR_TICKLOW = None
    CBAR_TICKHIGH = None

    # --- Line Widths ---
    AX_SPINE_LW = 0.5
    AX_TICK_WIDTH = 0.8
    CB_OUTLINE_LW = 0.5
    CB_TICK_WIDTH = 0.5

    # --- Spine Toggles & Widths (0.0 to hide) ---
    AX_SPINE_LW_TOP = None
    AX_SPINE_LW_BOTTOM = None
    AX_SPINE_LW_LEFT = None
    AX_SPINE_LW_RIGHT = None
    AX_SPINE_HIDE = False
    CB_OUTLINE_HIDE = False
    SHOW_XTICKLABELS = True
    AX_NBINS = 8

    # --- Colorbar Geometry ---
    CB_FRACTION = 0.02
    CB_PAD = 0.01
    CB_ASPECT = 10

    # --- Figure Geometry ---
    DRAW_MODE = "bar"
    BAR_WIDTH = 1.0
    FIG_WIDTH = 10
    FIG_HEIGHT = 1.7
    FORCE_FIGSIZE = False
    SUBPLOT_ADJUST = None

    # --- Backgrounds ---
    TRANSPARENT_BG = False
    BG_COLOR = None
    AXES_BG_COLOR = None
    CBAR_BG_COLOR = None

    # --- Hover Tooltip ---
    HOVER_THEME = "dark"
    HOVER_LIGHT_BG = "#ffffff"
    HOVER_LIGHT_EDGE = "#c8cdd6"
    HOVER_LIGHT_TEXT = "#1f2430"
    HOVER_DARK_BG = "#232833"
    HOVER_DARK_EDGE = "#4a5261"
    HOVER_DARK_TEXT = "#e8eaf0"
    HOVER_POS_COLOR = "#e05c5c"
    HOVER_NEG_COLOR = "#5c92e0"
    HOVER_ZERO_COLOR = "#8a919f"
    HOVER_ALPHA = 0.95
    HOVER_PAD = 0.55
    HOVER_ROUND = 0.35
    HOVER_EDGE_LW = 0.6
    HOVER_FONT_WEIGHT = "normal"
    HOVER_FONT_FAMILY = None
    HOVER_TEXT_FORMAT = "line"
    HOVER_OFFSET = (14, 14)
    HOVER_FLIP_AT_X = 0.72
    HOVER_FLIP_AT_Y = 0.72
    HOVER_VLINE_COLOR = None
    HOVER_VLINE_LW = 0.7
    HOVER_VLINE_ALPHA = 0.7
    HOVER_VLINE_STYLE = "--"

    # --- Interactivity ---
    INTERACTIVE = True
    USE_BLIT = True
    ZOOM_ENABLED = True
    ZOOM_BASE = 0.8
    MIN_SPAN = 20
    PAN_ENABLED = True
    DBLCLICK_RESET = True

    def __init__(
        self,
        csv_path: Any,
        out_path: Optional[str] = "acin_attention_spectrum_bar.png",
        show_plot: bool = True,
        dims: Optional[int] = None,
        interactive: Optional[bool] = None,
        fig=None,
        ax=None,
        **style,
    ) -> None:
        self.csv_path = csv_path
        self.out_path = out_path
        self.show_plot = show_plot
        self.dims = int(dims) if dims is not None else self.DIMS
        if interactive is not None:
            self.INTERACTIVE = bool(interactive)

        for key, value in style.items():
            if not hasattr(self, key):
                raise AttributeError(f"Unknown style parameter: {key}")
            setattr(self, key, value)

        self._external_fig = fig
        self._external_ax = ax

        self._fig = None
        self._ax = None
        self._cb = None
        self._idx = None
        self._w = None

        self._hover_annot = None
        self._hover_vline = None
        self._hover_last_i = None
        self._bg = None
        self._use_blit = bool(self.USE_BLIT)

        # Callbacks for GUI teardown
        self._cid_move = None
        self._cid_scroll = None
        self._cid_press = None
        self._cid_release = None
        self._cid_dblclick = None
        self._cid_draw = None

        self._dragging = False
        self._drag_start_px = None
        self._drag_start_xlim = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _fs(self, base: float) -> float:
        """Scale font size with a minimum floor of 4pt."""
        return max(4.0, float(base) * float(self.FONT_SCALE))

    def _fs_pad(self, base: float) -> float:
        """Scale padding without a floor (negative values allowed)."""
        return float(base) * float(self.FONT_SCALE)

    def _fw(self, override):
        return override if override is not None else self.FONT_WEIGHT

    def _apply_layout(self, fig) -> None:
        if self.SUBPLOT_ADJUST:
            fig.subplots_adjust(**self.SUBPLOT_ADJUST)

    def _apply_bg(self, fig, ax, cb=None) -> None:
        """Handle background color overrides."""
        if self.BG_COLOR is not None:
            fig.patch.set_facecolor(self.BG_COLOR)
            fig.patch.set_alpha(1.0)
            ax.patch.set_facecolor(self.AXES_BG_COLOR or self.BG_COLOR)
            ax.patch.set_alpha(1.0)
            if cb is not None:
                cb.ax.patch.set_facecolor(self.CBAR_BG_COLOR or self.BG_COLOR)
                cb.ax.patch.set_alpha(1.0)
            return
        if not self.TRANSPARENT_BG:
            return
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        if cb is not None:
            cb.ax.patch.set_alpha(0.0)

    def _hover_colors(self):
        if self.HOVER_THEME == "dark":
            return self.HOVER_DARK_BG, self.HOVER_DARK_EDGE, self.HOVER_DARK_TEXT
        return self.HOVER_LIGHT_BG, self.HOVER_LIGHT_EDGE, self.HOVER_LIGHT_TEXT

    def _weight_color(self, value: float) -> str:
        if value > 1e-6:
            return self.HOVER_POS_COLOR
        if value < -1e-6:
            return self.HOVER_NEG_COLOR
        return self.HOVER_ZERO_COLOR

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------
    def load_weights(self):
        """Load and sort attention weights by numeric substructure index."""
        df = (self.csv_path if isinstance(self.csv_path, pd.DataFrame)
              else pd.read_csv(self.csv_path))

        if {"weights", "krfp_idx"}.issubset(df.columns):
            idx = df["krfp_idx"].astype(str).tolist()
            w = df["weights"].to_numpy(dtype=float)
        elif df.index.name == "krfp_idx" and "weights" in df.columns:
            idx = df.index.astype(str).tolist()
            w = df["weights"].to_numpy(dtype=float)
        else:
            idx = [str(c) for c in df.columns]
            w = df.iloc[0].to_numpy(dtype=float)

        nums = [int(s.split("_")[-1]) for s in idx]
        order = np.argsort(nums)
        return np.asarray(idx)[order], w[order]

    # ------------------------------------------------------------------
    # Drawing Primitives
    # ------------------------------------------------------------------
    def _draw_bar(self, ax, w, cmap_obj, norm):
        x = np.arange(self.dims, dtype=float)
        hw = self.BAR_WIDTH / 2.0
        zeros = np.zeros(self.dims)
        verts = np.stack([
            np.column_stack([x - hw, zeros]),
            np.column_stack([x + hw, zeros]),
            np.column_stack([x + hw, w]),
            np.column_stack([x - hw, w]),
        ], axis=1)
        coll = PolyCollection(
            verts, facecolors=cmap_obj(norm(w)),
            edgecolors="none", linewidths=0, antialiased=False,
        )
        ax.add_collection(coll)
        ax.set_xlim(-0.5, self.dims - 0.5)
        ax.set_ylim(self.VMIN, self.VMAX)

        mappable = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        mappable.set_array([])
        return mappable

    def _draw_imshow(self, ax, w):
        mappable = ax.imshow(
            w.reshape(1, -1), aspect="auto",
            cmap=self.CMAP, vmin=self.VMIN, vmax=self.VMAX,
            extent=[0, self.dims - 1, 0, 1], interpolation="nearest",
        )
        ax.set_xlim(0, self.dims - 1)
        ax.set_ylim(0, 1)
        return mappable

    # ------------------------------------------------------------------
    # Hover Interaction
    # ------------------------------------------------------------------
    def _setup_interaction(self) -> None:
        bg, edge, text_color = self._hover_colors()
        font_family = self.HOVER_FONT_FAMILY or FONT_FAMILY

        self._hover_annot = self._ax.annotate(
            "", xy=(0, 0), xytext=self.HOVER_OFFSET, textcoords="offset points",
            fontsize=self._fs(self.HOVER_FONTSIZE),
            fontfamily=font_family,
            fontweight=self.HOVER_FONT_WEIGHT,
            color=text_color,
            bbox=dict(
                boxstyle=f"round,pad={self.HOVER_PAD},rounding_size={self.HOVER_ROUND}",
                fc=bg, ec=edge, lw=self.HOVER_EDGE_LW, alpha=self.HOVER_ALPHA,
            ),
            visible=False, zorder=20, clip_on=False,
        )

        vline_color = self.HOVER_VLINE_COLOR or self.HOVER_ZERO_COLOR
        self._hover_vline = self._ax.axvline(
            0, color=vline_color, lw=self.HOVER_VLINE_LW,
            alpha=self.HOVER_VLINE_ALPHA, ls=self.HOVER_VLINE_STYLE,
            visible=False, zorder=15,
        )

        self._use_blit = bool(self.USE_BLIT) and self._fig.canvas.supports_blit
        if self._use_blit:
            self._hover_annot.set_animated(True)
            self._hover_vline.set_animated(True)
            self._fig.canvas.draw()
            self._bg = self._fig.canvas.copy_from_bbox(self._fig.bbox)
            self._cid_draw = self._fig.canvas.mpl_connect("draw_event", self._on_draw)

        self._cid_move = self._fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        if self.ZOOM_ENABLED:
            self._cid_scroll = self._fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        if self.PAN_ENABLED:
            self._cid_press = self._fig.canvas.mpl_connect("button_press_event", self._on_press)
            self._cid_release = self._fig.canvas.mpl_connect("button_release_event", self._on_release)
        if self.DBLCLICK_RESET:
            self._cid_dblclick = self._fig.canvas.mpl_connect("button_press_event", self._on_dblclick)

    def _on_draw(self, event) -> None:
        if not self._use_blit:
            return
        self._bg = self._fig.canvas.copy_from_bbox(self._fig.bbox)
        self._hover_last_i = None

    def _blit(self) -> None:
        canvas = self._fig.canvas
        if self._use_blit and self._bg is not None:
            canvas.restore_region(self._bg)
            self._ax.draw_artist(self._hover_vline)
            self._ax.draw_artist(self._hover_annot)
            canvas.blit(self._fig.bbox)
            canvas.flush_events()
        else:
            canvas.draw_idle()

    def _hide_hover(self) -> None:
        if self._hover_annot is None:
            return
        changed = False
        if self._hover_annot.get_visible():
            self._hover_annot.set_visible(False)
            changed = True
        if self._hover_vline is not None and self._hover_vline.get_visible():
            self._hover_vline.set_visible(False)
            changed = True
        if changed:
            self._hover_last_i = None
            self._blit()

    def _compute_hover_offset(self, event):
        xlim = self._ax.get_xlim()
        sx = xlim[1] - xlim[0]
        rel_x = (event.xdata - xlim[0]) / sx if sx > 0 else 0.5

        ylim = self._ax.get_ylim()
        sy = ylim[1] - ylim[0]
        rel_y = (event.ydata - ylim[0]) / sy if sy > 0 else 0.5

        ox, oy = self.HOVER_OFFSET
        if rel_x > self.HOVER_FLIP_AT_X:
            ox = -abs(ox)
        if rel_y > self.HOVER_FLIP_AT_Y:
            oy = -abs(oy)
        return ox, oy

    def _format_hover_text(self, label, value: float) -> str:
        try:
            idx_num = int(str(label).split("_")[-1])
        except Exception:
            idx_num = str(label)
        if self.HOVER_TEXT_FORMAT == "two_line":
            return f"index: {idx_num}\nweight: {value:+.4f}"
        return f"index: {idx_num}   weight: {value:+.4f}"

    # ------------------------------------------------------------------
    # Zoom / Pan / Reset
    # ------------------------------------------------------------------
    def _redraw_view(self) -> None:
        if self._hover_annot is not None:
            self._hover_annot.set_visible(False)
        if self._hover_vline is not None:
            self._hover_vline.set_visible(False)
        self._hover_last_i = None
        self._fig.canvas.draw()

    def _on_scroll(self, event) -> None:
        if event.inaxes is not self._ax or event.xdata is None:
            return
        if event.button == "up":
            scale = self.ZOOM_BASE
        elif event.button == "down":
            scale = 1.0 / self.ZOOM_BASE
        else:
            return

        xlim = self._ax.get_xlim()
        span = xlim[1] - xlim[0]
        new_span = span * scale
        if new_span < self.MIN_SPAN:
            return
        if new_span > (self.dims - 1) * 1.05:
            new_span = (self.dims - 1) * 1.05

        ratio = (event.xdata - xlim[0]) / span if span > 0 else 0.5
        new_left = event.xdata - new_span * ratio
        new_right = new_left + new_span

        lo, hi = -0.5, self.dims - 0.5
        if new_left < lo:
            new_left, new_right = lo, lo + new_span
        if new_right > hi:
            new_right, new_left = hi, hi - new_span

        self._ax.set_xlim(new_left, new_right)
        self._redraw_view()

    def _on_press(self, event) -> None:
        if event.button != 1 or event.inaxes is not self._ax:
            return
        self._dragging = True
        self._drag_start_px = event.x
        self._drag_start_xlim = self._ax.get_xlim()
        try:
            self._fig.canvas.set_cursor(cursors.MOVE)
        except Exception:
            pass
        self._hide_hover()

    def _on_release(self, event) -> None:
        if self._dragging:
            self._dragging = False
            self._drag_start_px = None
            self._drag_start_xlim = None
            try:
                self._fig.canvas.set_cursor(cursors.POINTER)
            except Exception:
                pass

    def _on_dblclick(self, event) -> None:
        if event.inaxes is not self._ax:
            return
        if getattr(event, "dblclick", False):
            self._ax.set_xlim(-0.5, self.dims - 0.5)
            self._ax.set_ylim(self.VMIN, self.VMAX)
            self._redraw_view()

    def _on_mouse_move(self, event) -> None:
        if self._dragging:
            if self._drag_start_px is None:
                return
            bbox = self._ax.get_window_extent()
            if bbox.width <= 0:
                return
            x0, x1 = self._drag_start_xlim
            span = x1 - x0
            dx_data = (event.x - self._drag_start_px) * span / bbox.width
            self._ax.set_xlim(x0 - dx_data, x1 - dx_data)
            self._redraw_view()
            return

        if self._hover_annot is None:
            return
        if event.inaxes is not self._ax or event.xdata is None or event.ydata is None:
            self._hide_hover()
            return

        i = int(round(event.xdata))
        if i < 0 or i >= self.dims:
            self._hide_hover()
            return

        ox, oy = self._compute_hover_offset(event)
        self._hover_annot.xyann = (ox, oy)

        if i == self._hover_last_i:
            return
        self._hover_last_i = i

        label = str(self._idx[i])
        value = float(self._w[i])

        self._hover_annot.xy = (event.xdata, event.ydata)
        self._hover_annot.set_text(self._format_hover_text(label, value))
        self._hover_annot.set_visible(True)

        if self._hover_vline is not None:
            if self.HOVER_VLINE_COLOR is None:
                self._hover_vline.set_color(self._weight_color(value))
            self._hover_vline.set_xdata([i, i])
            self._hover_vline.set_visible(True)

        self._blit()

    # ------------------------------------------------------------------
    # Main Plot Entry Point
    # ------------------------------------------------------------------
    def plot_bar(self, idx, w):
        """Render attention weights and configure interactivity."""
        idx = np.asarray(idx)
        w = np.asarray(w)
        order = np.argsort([int(s.split("_")[-1]) for s in idx])
        idx, w = idx[order], w[order]

        self._idx, self._w = idx, w

        if self._external_fig is not None:
            fig = self._external_fig
            if self.FORCE_FIGSIZE:
                fig.set_size_inches(self.FIG_WIDTH, self.FIG_HEIGHT)
            ax = self._external_ax if self._external_ax is not None else fig.add_subplot(111)
        else:
            fig, ax = plt.subplots(figsize=(self.FIG_WIDTH, self.FIG_HEIGHT))

        self._fig, self._ax = fig, ax

        cmap_obj = plt.get_cmap(self.CMAP)
        norm = mcolors.Normalize(vmin=self.VMIN, vmax=self.VMAX)
        mappable = (self._draw_bar(ax, w, cmap_obj, norm)
                    if self.DRAW_MODE == "bar" else self._draw_imshow(ax, w))
        ax.set_yticks([])

        # Title
        if self.TITLE_TEXT:
            ax.set_title(
                self.TITLE_TEXT,
                fontsize=self._fs(self.TITLE_FONTSIZE),
                fontfamily=FONT_FAMILY,
                fontweight=self._fw(self.TITLE_FONT_WEIGHT),
                loc="left",
                pad=self._fs_pad(self.TITLE_PAD),
            )

        # X-axis Label
        if self.XLABEL_TEXT:
            ax.set_xlabel(
                self.XLABEL_TEXT,
                fontsize=self._fs(self.XLABEL_FONTSIZE),
                fontfamily=FONT_FAMILY,
                fontweight=self._fw(self.XLABEL_FONT_WEIGHT),
                labelpad=self._fs_pad(self.XLABEL_PAD),
            )

        # X-axis Ticks
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=self.AX_NBINS))
        ax.tick_params(
            axis="x",
            labelsize=self._fs(self.AX_TICK_LABELSIZE) if self.SHOW_XTICKLABELS else 0,
            width=self.AX_TICK_WIDTH,
            pad=self._fs_pad(self.AX_TICK_PAD),
            labelbottom=self.SHOW_XTICKLABELS,
        )
        for lbl in ax.get_xticklabels():
            lbl.set_fontfamily(FONT_FAMILY)
            lbl.set_fontweight(self._fw(self.AX_TICK_FONT_WEIGHT))

        # Spines
        if self.AX_SPINE_HIDE:
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            lw = {
                "top": self.AX_SPINE_LW_TOP if self.AX_SPINE_LW_TOP is not None else self.AX_SPINE_LW,
                "bottom": self.AX_SPINE_LW_BOTTOM if self.AX_SPINE_LW_BOTTOM is not None else self.AX_SPINE_LW,
                "left": self.AX_SPINE_LW_LEFT if self.AX_SPINE_LW_LEFT is not None else self.AX_SPINE_LW,
                "right": self.AX_SPINE_LW_RIGHT if self.AX_SPINE_LW_RIGHT is not None else self.AX_SPINE_LW,
            }
            for name, sp in ax.spines.items():
                if name in lw and lw[name] is not None:
                    sp.set_linewidth(float(lw[name]))
                else:
                    sp.set_visible(False)

        # Colorbar
        cb = fig.colorbar(
            mappable, ax=ax,
            fraction=self.CB_FRACTION, pad=self.CB_PAD, aspect=self.CB_ASPECT,
        )
        cb.set_ticks([self.VMIN, self.VMAX])
        cb.set_ticklabels([
            self.CBAR_TICKLOW if self.CBAR_TICKLOW is not None else f"{self.VMIN:g}",
            self.CBAR_TICKHIGH if self.CBAR_TICKHIGH is not None else f"{self.VMAX:g}",
        ])
        self._cb = cb

        if self.CBAR_LABEL:
            cb.set_label(
                self.CBAR_LABEL,
                fontsize=self._fs(self.CBAR_LABEL_FONTSIZE),
                fontfamily=FONT_FAMILY,
                fontweight=self._fw(self.CBAR_LABEL_FONT_WEIGHT),
                rotation=self.CBAR_LABEL_ROTATION,
                labelpad=self.CBAR_LABEL_PAD,
                va="bottom",
            )

        cb.ax.tick_params(
            labelsize=self._fs(self.CB_TICK_LABELSIZE),
            width=self.CB_TICK_WIDTH,
            pad=self._fs_pad(self.CB_TICK_PAD),
        )
        for lbl in cb.ax.get_yticklabels():
            lbl.set_fontfamily(FONT_FAMILY)
            lbl.set_fontweight(self._fw(self.CB_TICK_FONT_WEIGHT))

        if self.CB_OUTLINE_HIDE:
            cb.outline.set_visible(False)
        else:
            cb.outline.set_linewidth(float(self.CB_OUTLINE_LW))

        # Layout and Background
        self._apply_layout(fig)
        self._apply_bg(fig, ax, cb)

        # Save
        if self.out_path:
            fig.savefig(
                self.out_path, bbox_inches="tight", dpi=300,
                transparent=(self.TRANSPARENT_BG and self.BG_COLOR is None),
            )
            print(f"[ok] Saved -> {self.out_path}")

        # Interactive layer
        if self.INTERACTIVE or self.ZOOM_ENABLED or self.PAN_ENABLED:
            self._setup_interaction()

        return fig, ax

    def run(self) -> None:
        """Standalone execution pipeline."""
        if self.csv_path is None:
            raise SystemExit("csv_path must be provided.")
        idx, w = self.load_weights()
        assert len(w) == self.dims, f"Expected {self.dims} weights, got {len(w)}"
        self.plot_bar(idx, w)
        if self.show_plot and self._external_fig is None:
            plt.show()


# ---------------------------------------------------------------------------
# Execution Entry Point
# ---------------------------------------------------------------------------
CSV_PATH = "attn_weights.csv"
OUT_PATH = "acin_attention_spectrum_bar.png"
SHOW_PLOT = True
INTERACTIVE = True
USE_BLIT = True
STYLE_OVERRIDES: dict = {}

if __name__ == "__main__":
    import PySide6
    pyside_dir = os.path.dirname(PySide6.__file__)
    os.environ.setdefault(
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        os.path.join(pyside_dir, "plugins", "platforms"),
    )
    os.environ.setdefault("QT_QPA_PLUGIN_PATH", os.path.join(pyside_dir, "plugins"))
    bar = ACINSpectrumBar(
        csv_path=CSV_PATH,
        out_path=OUT_PATH,
        show_plot=SHOW_PLOT,
        interactive=INTERACTIVE,
        **STYLE_OVERRIDES,
    )
    bar.run()