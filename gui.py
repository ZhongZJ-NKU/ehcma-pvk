"""
Top-level entry point and GUI layer for the **Explainable Hierarchy-Constrained
Multimodal Agent**, a platform for mechanistic analysis of Co-SAM buried
interfaces in perovskite photovoltaics.

Wraps the numerical pipeline (MolCalc, ML/DL training, SHAP analysis, LLM
reasoning) in a PySide6 application.  Heavy jobs run in child processes or
WorkerThreads and communicate back through Signals and Queues.
"""

from __future__ import annotations

import io
import math
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from multiprocessing import Event, Pool, Process, Queue
from typing import Optional, Union
import ctypes
import GPUtil
import joblib
import markdown
import numpy as np
import pandas as pd
import psutil
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image
from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QFile,
    QIODevice,
    QPoint,
    QPointF,
    Property,
    QPropertyAnimation,
    QSize,
    QThread,
    QTimer,
    QUrl,
    QVariantAnimation,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDesktopServices,
    QFont,
    QIcon,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
    QTextCursor,
    QTextDocument,
    QTextFormat,
    QTextOption,
)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from skfp.fingerprints import KlekotaRothFingerprint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ml_eval
from acin_analysis import ACINSpectrumBar
from dl_training import ACINTraining
from ml_training import AutoML
from mol_calc import MolCalc
from msec import MSECAgent

try:
    from BlurWindow.blurWindow import GlobalBlur
except Exception:  # pragma: no cover - platform dependent
    GlobalBlur = None

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# ANSI -> HTML
# ---------------------------------------------------------------------------
_ANSI_FOREGROUND: dict[int, str] = {
    30: "#000000", 31: "#e06c75", 32: "#98c379", 33: "#e5c07b",
    34: "#61afef", 35: "#c678dd", 36: "#56b6c2", 37: "#abb2bf",
    39: "#dcdfe4", 90: "#5c6370", 91: "#f44747", 92: "#98c379",
    93: "#e5c07b", 94: "#61afef", 95: "#c678dd", 96: "#56b6c2",
    97: "#ffffff",
}

_ANSI_RE = re.compile(r"\x1b\[([\d;]*)m")


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def ansi_to_html(text: str, default_color: str = "#dcdfe4") -> str:
    """Translate ANSI SGR foreground codes into inline-styled HTML spans."""
    html_parts: list[str] = []
    current_color = default_color
    last_pos = 0

    for match in _ANSI_RE.finditer(text):
        plain = text[last_pos:match.start()]
        if plain:
            html_parts.append(
                f'<span style="color:{current_color};">{_escape_html(plain)}</span>'
            )

        for code in (int(c) for c in match.group(1).split(";") if c.isdigit()):
            if code == 0:
                current_color = default_color
            elif code in _ANSI_FOREGROUND:
                current_color = _ANSI_FOREGROUND[code]

        last_pos = match.end()

    tail = text[last_pos:]
    if tail:
        html_parts.append(
            f'<span style="color:{current_color};">{_escape_html(tail)}</span>'
        )

    return (
        '<pre style="margin:0;padding:0;'
        "font-family: Consolas, 'Courier New', monospace;"
        'font-size:14px;line-height:0.8;white-space: pre-wrap;">'
        + "".join(html_parts)
        + "</pre>"
    )


# ---------------------------------------------------------------------------
# Markdown table repair
# ---------------------------------------------------------------------------
_LIST_ITEM_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+")

_INLINE_BULLET_RE = re.compile(
    r"(?<=[^\s])\s+-\s+(?="
    r"(?:[A-Z][A-Za-z0-9 _/()\-]{0,60}"
    r"|[\u4e00-\u9fff][\u4e00-\u9fffA-Za-z0-9 ]{0,30})"
    r"\s*:)"
)


def _split_inline_bullets(lines: list[str]) -> list[str]:
    """Promote inline ' - Label:' fragments to standalone '- Label:' lines."""
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Leave tables and headings untouched.
        if stripped.startswith("|") or stripped.startswith("#"):
            out.append(line)
            continue
        line = _INLINE_BULLET_RE.sub("\n- ", line)
        out.extend(line.split("\n"))
    return out


def _ensure_list_blank_lines(lines: list[str]) -> list[str]:
    """Insert blank lines before list items and indent continuation lines."""
    out: list[str] = []
    for line in lines:
        m = _LIST_ITEM_RE.match(line)
        if m:
            indent = m.group(1)
            marker = m.group(2)
            is_unordered = marker in "-*+"

            if 0 < len(indent) < 4:
                line = "    " + line.lstrip()
                indent = "    "

            if out:
                prev = out[-1]
                if prev.strip() != "":
                    pm = _LIST_ITEM_RE.match(prev)
                    same_level_item = (
                        pm is not None
                        and len(pm.group(1)) == len(indent)
                        and (pm.group(2) in "-*+") == is_unordered
                    )
                    if not same_level_item:
                        out.append("")

            out.append(line)
            continue

        if out and line.strip() != "":
            prev = out[-1]
            if _LIST_ITEM_RE.match(prev) and not prev.startswith(("    ", "\t")):
                line = "    " + line

        out.append(line)
    return out


def fix_markdown_tables(md_text: str) -> str:
    """Repair LLM-generated Markdown tables."""
    lines = md_text.splitlines()
    out_lines = []
    table_block = []
    in_table = False

    sep_pattern = re.compile(r'^\|([\s\-:|]+|)+$')

    for line in lines:
        stripped = line.strip()
        is_table_line = stripped.startswith("|")

        if is_table_line:
            if not in_table:
                in_table = True
                table_block = []
            table_block.append(line)
        else:
            if in_table:
                if stripped == "":
                    filtered_block = []
                    for ln in table_block:
                        s = ln.strip()
                        if sep_pattern.fullmatch(s):
                            continue
                        filtered_block.append(ln)

                    full_text = " ".join([ln.strip() for ln in filtered_block])
                    raw_cells = [c.strip() for c in re.split(r"\|", full_text)]
                    raw_cells = [c for c in raw_cells if c != ""]

                    header_line = None
                    for l in filtered_block:
                        if l.strip().startswith("|"):
                            header_line = l
                            break
                    if header_line is None:
                        processed = table_block
                    else:
                        header_cells = [c.strip() for c in re.split(r"\|", header_line) if c.strip()]
                        col_num = len(header_cells)
                        if col_num <= 1:
                            processed = table_block
                        else:
                            output_table = []
                            output_table.append("| " + " | ".join(header_cells) + " |")
                            sep = "| " + " | ".join(["---"] * col_num) + " |"
                            output_table.append(sep)
                            data_cells = raw_cells[col_num:]
                            for i in range(0, len(data_cells), col_num):
                                row_cells = data_cells[i:i+col_num]
                                while len(row_cells) < col_num:
                                    row_cells.append("")
                                row_str = "| " + " | ".join(row_cells) + " |"
                                output_table.append(row_str)
                            processed = output_table
                    out_lines.extend(processed)
                    out_lines.append(line)
                    in_table = False
                    table_block = []
                else:
                    table_block.append(line)
            else:
                out_lines.append(line)
    if in_table and len(table_block) > 0:
        filtered_block = []
        for ln in table_block:
            s = ln.strip()
            if sep_pattern.fullmatch(s):
                continue
            filtered_block.append(ln)

        full_text = " ".join([ln.strip() for ln in filtered_block])
        raw_cells = [c.strip() for c in re.split(r"\|", full_text)]
        raw_cells = [c for c in raw_cells if c != ""]

        header_line = None
        for l in filtered_block:
            if l.strip().startswith("|"):
                header_line = l
                break
        if header_line is None:
            processed = table_block
        else:
            header_cells = [c.strip() for c in re.split(r"\|", header_line) if c.strip()]
            col_num = len(header_cells)
            if col_num <= 1:
                processed = table_block
            else:
                output_table = []
                output_table.append("| " + " | ".join(header_cells) + " |")
                sep = "| " + " | ".join(["---"] * col_num) + " |"
                output_table.append(sep)
                data_cells = raw_cells[col_num:]
                for i in range(0, len(data_cells), col_num):
                    row_cells = data_cells[i:i+col_num]
                    while len(row_cells) < col_num:
                        row_cells.append("")
                    row_str = "| " + " | ".join(row_cells) + " |"
                    output_table.append(row_str)
                processed = output_table
        out_lines.extend(processed)

    out_lines = _split_inline_bullets(out_lines)
    out_lines = _ensure_list_blank_lines(out_lines)

    return "\n".join(out_lines)


def format_markdown_to_styled_html(md_text: str) -> str:
    """Convert Markdown to styled HTML with strict inline styles for QTextDocument."""
    content = fix_markdown_tables(md_text)
    html = markdown.markdown(
        content,
        extensions=["tables", "fenced_code", "sane_lists"],
    )

    html = re.sub(r'<p>', '<p style="color:#dcdfe4; margin-top:4px; margin-bottom:4px;">', html)
    html = re.sub(r'<li>', '<li style="color:#dcdfe4;">', html)
    html = re.sub(r'<td>', '<td style="color:#dcdfe4; border:1px solid #6a7383; padding:6px 10px;word-break:break-all;overflow-wrap:anywhere;">', html)
    html = re.sub(r'<th>',
                  '<th style="color:#ffffff; background-color:#2d3b55; border:1px solid #6a7383; padding:6px 10px;">',
                  html)
    html = re.sub(r'<strong>', '<strong style="color:#ffffff; font-weight:bold;">', html)
    html = re.sub(r'<table>', '<table style="border-collapse:collapse; margin:12px 0; width:100%;table-layout:fixed;">', html)
    html = re.sub(r'<code>', '<code style="background:#2d3b55; padding:2px 6px; border-radius:4px; color:#e8eef8;">',
                  html)

    return (
        "<html><body>"
        "<style>"
        "body { background-color: transparent; color: #dcdfe4; "
        "font-family: Consolas, 'Courier New', monospace; font-size: 13px; line-height: 1.6; }"
        "h1, h2, h3 { color: #e8eef8; border-bottom: 1px solid #3a4a68; padding-bottom: 6px; margin-top: 16px; }"
        "</style>"
        f"<div style='color:#dcdfe4;'>{html}</div>"
        "</body></html>"
    )



# ---------------------------------------------------------------------------
# Shared stylesheet fragments
# ---------------------------------------------------------------------------
DIALOG_BUTTON_QSS = """
QPushButton {
    background-color: rgba(168, 208, 255, 0.45);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-family: 'Consolas';
    font-size: 13px;
}
QPushButton:hover  { background-color: #3b8bca; }
QPushButton:pressed { background-color: #1b6bAA; }
"""


def make_dialog_button(text: str, default: bool = False) -> QPushButton:
    btn = QPushButton(text)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setMinimumWidth(85)
    btn.setMinimumHeight(32)
    btn.setStyleSheet(DIALOG_BUTTON_QSS)
    if default:
        btn.setDefault(True)
    return btn


# ---------------------------------------------------------------------------
# CPK Color & van der Waals Parameters
# ---------------------------------------------------------------------------
# Bondi vdW radii (Å). Marker area scaled proportional to atomic volume
# to retain relative atomic size visually.

CPK: dict[str, tuple[float, float, float]] = {
    "H":  (0.85, 0.90, 1.00),
    "C":  (0.28, 0.32, 0.42),
    "N":  (0.25, 0.65, 1.00),
    "O":  (1.00, 0.30, 0.45),
    "F":  (0.35, 1.00, 0.65),
    "Cl": (0.15, 0.95, 0.55),
    "Br": (0.95, 0.45, 0.25),
    "I":  (0.75, 0.35, 1.00),
    "S":  (1.00, 0.85, 0.20),
    "P":  (1.00, 0.55, 0.15),
}
VDW: dict[str, float] = {
    "H": 0.32, "C": 0.50, "N": 0.48, "O": 0.46, "F": 0.44,
    "Cl": 0.56, "Br": 0.62, "I": 0.70, "S": 0.58, "P": 0.58,
}



# ---------------------------------------------------------------------------
# 3D molecular geometry
# ---------------------------------------------------------------------------
def smiles_to_3d(smiles: str, seed: int = 42) -> tuple[Chem.Mol, np.ndarray]:
    """Embed SMILES into 3D with ETKDGv3 and relax with MMFF94."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=True)

    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        # MMFF94 lacks parameters for some exotic atom types; geometry is
        # still usable as-is.
        pass

    return mol, mol.GetConformer().GetPositions()


def build_molecule(smiles: str, seed: int = 42):
    """Precompute per-atom render parameters for :class:`Molecule3DCanvas`."""
    mol, coords = smiles_to_3d(smiles, seed)
    coords = coords - coords.mean(axis=0)

    # Normalize by max |coord| so marker areas are independent of scale; the
    # 2.0 floor prevents a lone heavy atom from collapsing to zero size.
    scale = max(float(np.abs(coords).max()), 2.0)

    colors, sizes = [], []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        colors.append(CPK.get(sym, (1.0, 0.0, 1.0)))
        r = VDW.get(sym, 0.45)
        sizes.append((r / scale * 55.0) ** 2)

    bonds = np.array(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()],
        dtype=int,
    ).reshape(-1, 2)

    return coords, np.array(colors), np.array(sizes), bonds


def rotate(coords: np.ndarray, theta: float, axis: str = "y") -> np.ndarray:
    """Apply a right-handed rotation about a principal axis."""
    c, s = math.cos(theta), math.sin(theta)
    if axis == "y":
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == "x":
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    else:
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return coords @ R.T


# ---------------------------------------------------------------------------
# Animated 3D molecule canvas
# ---------------------------------------------------------------------------
class Molecule3DCanvas(FigureCanvasQTAgg):
    """Self-animating matplotlib canvas that renders one 3D molecule."""

    def __init__(
        self,
        smiles: str,
        n_frames: int = 180,
        fps: int = 30,
        tilt: float = 0.35,
        background: str = "none",
        title: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        self.figure = Figure(figsize=(5, 5), facecolor=background)
        super().__init__(self.figure)
        self.setParent(parent)

        self._coords, self._colors, self._sizes, self._bonds = build_molecule(smiles)
        self._coords = rotate(self._coords, tilt, "x")
        self._n_frames = n_frames
        self._fps = fps

        self._axes = self.figure.add_subplot(111, projection="3d")
        self._axes.set_facecolor(background)
        self._axes.set_axis_off()

        if title:
            self._axes.set_title(
                title,
                fontsize=4,
                fontweight="normal",
                fontfamily="Consolas",
                pad=12,
                color="#E8EAF0",
                path_effects=[
                    __import__("matplotlib.patheffects", fromlist=["withStroke"])
                    .withStroke(linewidth=1, foreground="#00000055", alpha=0.45)
                ],
            )

        self._scatter = self._axes.scatter(
            self._coords[:, 0], self._coords[:, 1], self._coords[:, 2],
            c=self._colors,
            s=self._sizes,
            edgecolors="white",
            linewidths=0.15,
            depthshade=False,
            alpha=0.95,
        )

        segments = np.stack(
            [self._coords[self._bonds[:, 0]], self._coords[self._bonds[:, 1]]],
            axis=1,
        )
        # Bonds thinner than the smallest atom marker => conventional
        # ball-and-stick look.
        avg_atom_pt = float(np.mean(np.sqrt(self._sizes)))
        self._bond_collection = Line3DCollection(
            segments, colors="#8a93a6",
            linewidths=min(1.0, avg_atom_pt / 5.0),
            alpha=0.65,
        )
        self._axes.add_collection3d(self._bond_collection)

        half = (self._coords.max(axis=0) - self._coords.min(axis=0)) / 2.0
        half = np.maximum(half, 2.0)
        self._axes.set_xlim(-half[0], half[0])
        self._axes.set_ylim(-half[1], half[1])
        self._axes.set_zlim(-half[2], half[2])
        self._axes.set_box_aspect(tuple(half))
        self._axes.view_init(elev=270, azim=180)

        self._animation = FuncAnimation(
            self.figure,
            self._update_frame,
            frames=self._n_frames,
            interval=1000 / self._fps,
            blit=False,
            cache_frame_data=False,
        )
        self.figure.subplots_adjust(left=0, right=1, top=1.05, bottom=-0.05)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    def _update_frame(self, frame: int):
        theta = 2 * math.pi * frame / self._n_frames
        rotated = rotate(self._coords, theta, "x")
        self._scatter._offsets3d = (rotated[:, 0], rotated[:, 1], rotated[:, 2])
        self._bond_collection.set_segments(
            np.stack([rotated[self._bonds[:, 0]], rotated[self._bonds[:, 1]]], axis=1)
        )
        return self._scatter, self._bond_collection

    def stop_animation(self) -> None:
        """Idempotent teardown of the matplotlib animation timer."""
        try:
            anim = getattr(self, "_animation", None)
            if anim is not None:
                anim.event_source.stop()
                self._animation = None
        except Exception:
            pass

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_animation()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Home-page bubbles
# ---------------------------------------------------------------------------
class BubbleParticle:
    """One buoyant particle; recycled rather than reallocated when off-screen."""

    def __init__(self, width: int, height: int) -> None:
        self.is_off_screen = False
        self.reset(width, height, initial=True)

    def reset(
        self,
        width: int,
        height: int,
        initial: bool = False,
        forced_x: Optional[float] = None,
        forced_y: Optional[float] = None,
    ) -> None:
        self.radius = random.uniform(50, 100)
        self.x = forced_x if forced_x is not None else random.uniform(self.radius, max(self.radius, width - self.radius))
        if forced_y is not None:
            self.y = forced_y
        elif initial:
            self.y = random.uniform(0, height)
        else:
            self.y = random.uniform(height, height + 200)

        self.speed = random.uniform(0.1, 0.5)
        self.alpha = random.randint(30, 120)
        self.start_alpha = self.alpha
        self.color = QColor(
            random.randint(150, 255),
            random.randint(150, 255),
            random.randint(200, 255),
        )
        self.wobble = random.uniform(0, 6.28)
        self.wobble_speed = random.uniform(0.05, 0.1)
        self.is_off_screen = False

    def update(self, width: int, height: int) -> None:
        self.y -= self.speed
        self.wobble += self.wobble_speed
        self.x += random.uniform(-0.3, 0.3)
        if self.y + self.radius < 0:
            self.is_off_screen = True


class BubbleBackgroundWidget(QWidget):
    """Animated translucent-bubble background for the home page."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMouseTracking(False)

        self.bubbles: list[BubbleParticle] = [
            BubbleParticle(self.width(), self.height()) for _ in range(30)
        ]
        self.max_bubbles = 50

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_bubbles)
        self._timer.start(16)  # ~60 FPS

    def _idle_bubble(self) -> Optional[BubbleParticle]:
        for b in self.bubbles:
            if b.is_off_screen:
                return b
        return None

    def _update_bubbles(self) -> None:
        w, h = self.width(), self.height()
        for b in self.bubbles:
            b.update(w, h)
        # Recycle at most a few per tick so recycling is visually smooth.
        if random.random() < 0.1:
            idle = self._idle_bubble()
            if idle:
                idle.reset(w, h)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        h_win = self.height()
        for b in self.bubbles:
            if b.is_off_screen:
                continue
            # Fade as the bubble approaches the top so recycling is invisible.
            ratio = max(0.0, b.y / h_win)
            alpha = int(b.start_alpha * ratio * 1.1)

            gradient = QRadialGradient(b.x, b.y, b.radius)
            c = QColor(b.color)
            c.setAlpha(alpha)
            gradient.setColorAt(0.0, QColor(255, 255, 255, alpha // 4))
            gradient.setColorAt(0.4, c)
            gradient.setColorAt(1.0, QColor(c.red(), c.green(), c.blue(), 0))
            painter.setBrush(QBrush(gradient))
            painter.setPen(QPen(QColor(255, 255, 255, 0), 1))
            painter.drawEllipse(QPointF(b.x, b.y), b.radius, b.radius)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        idle = self._idle_bubble()
        if idle is not None:
            pos = event.position()
            idle.reset(self.width(), self.height(), forced_x=pos.x(), forced_y=pos.y())
            idle.speed = random.uniform(1.0, 3.0)
        super().mouseMoveEvent(event)


# ---------------------------------------------------------------------------
# Home-page magnifier
# ---------------------------------------------------------------------------
class MaskCanvas(QWidget):
    """Hover-following circular magnifier over the home-page slogan."""

    MAX_RADIUS = 35

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)

        self._center = QPointF(0, 0)
        self._opacity = 0.0
        self._radius = 0.0

        self._opacity_anim = QPropertyAnimation(self, b"opacity")
        self._opacity_anim.setDuration(200)
        self._opacity_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        self._radius_anim = QPropertyAnimation(self, b"radius")
        self._radius_anim.setDuration(200)
        self._radius_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

    def _get_opacity(self) -> float:
        return self._opacity

    def _set_opacity(self, value: float) -> None:
        self._opacity = value
        self.update()

    opacity = Property(float, _get_opacity, _set_opacity)

    def _get_radius(self) -> float:
        return self._radius

    def _set_radius(self, value: float) -> None:
        self._radius = value
        self._center = self._clamp_center(self._center)
        self.update()

    radius = Property(float, _get_radius, _set_radius)

    def _clamp_center(self, pos: QPointF) -> QPointF:
        # Keep the spotlight fully inside the widget even at the edges.
        r = max(self._radius, 0.0)
        x = min(max(pos.x(), r), max(r, self.width() - r))
        y = min(max(pos.y(), r), max(r, self.height() - r))
        return QPointF(x, y)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)

        p.setFont(QFont("Consolas", 30))
        p.setPen(QColor("#ffffff"))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Materialize the Unseen")

        if self._opacity < 0.001 or self._radius < 0.5:
            return

        p.setOpacity(self._opacity)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#a8d0ff"))
        p.drawEllipse(self._center, self._radius, self._radius)

        clip = QPainterPath()
        clip.addEllipse(self._center, self._radius, self._radius)
        p.setClipPath(clip)
        p.setPen(QColor("#1f3a5f"))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Materialize the Unseen")

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        self._center = self._clamp_center(event.position())
        self.update()
        super().mouseMoveEvent(event)

    def enterEvent(self, event) -> None:  # noqa: N802
        self._opacity_anim.stop()
        self._opacity_anim.setEndValue(1.0)
        self._opacity_anim.start()
        self._radius_anim.stop()
        self._radius_anim.setEndValue(self.MAX_RADIUS)
        self._radius_anim.start()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._opacity_anim.stop()
        self._opacity_anim.setEndValue(0.0)
        self._opacity_anim.start()
        self._radius_anim.stop()
        self._radius_anim.setEndValue(0.0)
        self._radius_anim.start()

    def resizeEvent(self, event) -> None:  # noqa: N802
        if self._opacity < 0.001:
            self._center = QPointF(self.width() / 2, self.height() / 2)
        else:
            self._center = self._clamp_center(self._center)
        super().resizeEvent(event)


# ---------------------------------------------------------------------------
# Chat widgets
# ---------------------------------------------------------------------------
class MessageBubble(QFrame):
    """One chat bubble that renders Markdown incrementally."""

    heightChanged = Signal()

    def __init__(
        self,
        text: str = "",
        align: str = "left",
        kind: str = "ai",
        wrap: bool = True,
        contents_margins: tuple[int, int, int, int] = (12, 8, 12, 8),
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(kind)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*contents_margins)

        self.label = QTextBrowser()
        self.label.setOpenExternalLinks(True)
        self.label.setFrameShape(QFrame.NoFrame)
        if align == "right":
            self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum)
            self.label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum)
            self.label.setMaximumWidth(720)
        else:
            self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
            self.label.setMinimumWidth(720)

        self.label.setStyleSheet(
            "QTextBrowser { background: transparent; border: none; color: #dcdfe4; "
            "font-family: Consolas, 'Courier New', monospace; font-size: 13px;}"
        )
        self.label.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.label.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.label.document().setDocumentMargin(0)
        layout.addWidget(self.label)

        self._bottom_layout = QHBoxLayout()
        self._bottom_layout.setContentsMargins(0, 4, 0, 0)
        self._bottom_layout.setSpacing(6)
        layout.addLayout(self._bottom_layout)

        self._align = align
        self._raw = text
        self._dirty = False

        self._render(text)

        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.setInterval(60)
        self._flush_timer.timeout.connect(self._flush)

    def add_bottom_button(self, text: str, callback) -> None:
        """Attach an action button (e.g. *Save Report*) to the footer."""
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setIcon(QIcon(":/icons/save.svg"))
        btn.setIconSize(QSize(16, 16))
        btn.setStyleSheet(
            "QPushButton { background-color: transparent; color: #8b9bb4; "
            "border: none; border-radius: 6px; padding: 4px 10px; "
            "font-family: 'Consolas'; font-size: 12px; }"
            "QPushButton:hover { background-color: rgba(168, 208, 255, 0.1); "
            "color: #a8d0ff; }"
            "QPushButton:pressed { background-color: rgba(168, 208, 255, 0.2); }"
        )
        btn.clicked.connect(callback)
        self._bottom_layout.addSpacing(30)
        self._bottom_layout.addWidget(btn)
        self._bottom_layout.addStretch(1)

    def setText(self, text: str) -> None:  # noqa: N802
        self._raw = text
        self._dirty = False
        self._flush_timer.stop()
        self._render(text)

    def text(self) -> str:
        return self._raw

    def append(self, chunk: str) -> None:
        if not chunk:
            return
        self._raw += chunk
        self._dirty = True
        if not self._flush_timer.isActive():
            self._flush_timer.start()

    def finalize(self) -> None:
        if self._flush_timer.isActive():
            self._flush_timer.stop()
        self._flush()

    def setBubbleColor(self, bg_color: str, radius: int = 8) -> None:  # noqa: N802
        self.setStyleSheet(
            f"QFrame#{self.objectName()} {{"
            f" background-color: {bg_color};"
            f" border-radius: {radius}px;"
            f" }}"
        )

    def resizeEvent(self, event) -> None:  # noqa: N802
        parent_w = self.parentWidget().width() if self.parentWidget() else 800
        self.setMaximumWidth(int(parent_w * 1.0))
        super().resizeEvent(event)
        self._autosize()

    def _flush(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        self._render(self._raw)
        self.heightChanged.emit()

    def _render(self, md_text: str) -> None:
        if self._align == "right":
            html = (
                f'<div style="white-space:pre-wrap; color:#ffffff;">'
                f"{_escape_html(md_text)}</div>"
            )
        else:
            html = markdown.markdown(
                fix_markdown_tables(md_text),
                extensions=["tables", "fenced_code", "sane_lists"],
            )
            html = (
                    "<style>"
                    "body{font-family: Consolas, 'Courier New', monospace;}"
                    "table{"
                    "border-collapse:collapse;"
                    "margin:6px 0;"
                    "table-layout: fixed;"
                    "width:100%;"
                    "}"
                    "th,td{"
                    "border:1px solid #6a7383;"
                    "padding:4px 8px;"
                    "font-size:12px;"
                    "word-break:break-all;"
                    "overflow-wrap:anywhere;"
                    "}"
                    "th{background:#2d3b55;color:#fff;}"
                    "code{background:#2d3b55;padding:1px 4px;border-radius:3px;}"
                    "</style>" + html
            )
        self.label.setHtml(html)
        self._autosize()

    def _autosize(self) -> None:
        if self._align == "right":
            fm = self.label.fontMetrics()
            lines = self._raw.split("\n") or [""]
            text_w = max(fm.horizontalAdvance(line) for line in lines)
            wrap_w = min(text_w + 4, 480)
            self.label.setFixedWidth(max(wrap_w, 40))
        w = self.label.viewport().width()
        if w <= 0:
            return
        doc = self.label.document()
        doc.setTextWidth(w)
        h = int(doc.size().height()) + 4
        self.label.setMinimumHeight(h)
        self.label.setMaximumHeight(h)


class ThinkBox(QFrame):
    """Collapsible, streaming box that displays the LLM reasoning trace."""

    _SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    heightChanged = Signal()

    def __init__(
        self,
        title: str = " Reasoning process ",
        text: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("think")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 8, 12, 8)
        layout.setSpacing(0)

        self._title = title
        self._raw = text
        self._streaming = False

        self.header = QToolButton()
        self.header.setObjectName("thinkHeader")
        self.header.setCheckable(True)
        self.header.setChecked(False)
        self.header.setIcon(QIcon(":/icons/reasoning_process.svg"))
        self.header.setIconSize(QSize(20, 20))
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setStyleSheet(
            "QToolButton#thinkHeader { border: none; border-radius: 8px; "
            "background-color: rgba(30, 42, 68, 200); color: #FFFFFF; "
            "font-family: 'Consolas'; font-size: 14px; font-weight: bold; "
            "padding-left: 0px; padding-right: 12px; "
            "border-bottom: 1px solid rgba(168, 208, 255, 0.3); text-align: left; }"
            "QToolButton#thinkHeader:hover { color: #3478E8; }"
        )
        self.header.toggled.connect(self._toggle)
        layout.addWidget(self.header)

        self.body = QTextBrowser()
        self.body.setFrameShape(QFrame.NoFrame)
        self.body.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.body.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.body.setStyleSheet(
            "QTextBrowser { font-family: 'Consolas', 'Courier New', monospace; font-size: 12px; "
            "color: #B0B6BF; background: transparent; border: none; }"
        )
        self.body.setObjectName("thinkBody")
        self.body.setVisible(False)
        layout.addWidget(self.body)

        self._spin_idx = 0
        self._spin_timer = QTimer(self)
        self._spin_timer.setInterval(100)
        self._spin_timer.timeout.connect(self._tick_spinner)

        self._dirty = False
        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.setInterval(80)
        self._flush_timer.timeout.connect(self._flush)
        self._update_header_text()

    def text(self) -> str:
        return self._raw

    def append(self, chunk: str) -> None:
        if not chunk:
            return
        self._raw += chunk
        self._dirty = True
        if not self._flush_timer.isActive():
            self._flush_timer.start()

    def expand(self) -> None:
        if not self.header.isChecked():
            self.header.setChecked(True)

    def collapse(self) -> None:
        if self.header.isChecked():
            self.header.setChecked(False)

    def begin_stream(self, auto_expand: bool = True) -> None:
        self._streaming = True
        self._raw = ""
        self.body.setText("")
        self._spin_idx = 0
        if not self._spin_timer.isActive():
            self._spin_timer.start()
        self._update_header_text()
        if auto_expand:
            self.expand()

    def end_stream(self, auto_collapse: bool = True) -> None:
        self._streaming = False
        self._spin_timer.stop()
        self.finalize()
        self.body.adjustSize()
        self.updateGeometry()
        self._update_header_text()
        if auto_collapse:
            QTimer.singleShot(0, self.collapse)

    def finalize(self) -> None:
        if self._flush_timer.isActive():
            self._flush_timer.stop()
        self._flush()
        if self._streaming:
            self._streaming = False
            self._spin_timer.stop()
            self._update_header_text()

    def setText(self, text: str) -> None:  # noqa: N802
        self._raw = text
        self._dirty = False
        self._flush_timer.stop()
        self.body.setPlainText(text)
        self._autosize()
        if self._streaming:
            self._streaming = False
            self._spin_timer.stop()
            self._update_header_text()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._autosize()

    def _flush(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        self.body.setPlainText(self._raw)
        self._autosize()
        self.heightChanged.emit()

    def _update_header_text(self) -> None:
        if self._streaming:
            spin = self._SPINNER[self._spin_idx % len(self._SPINNER)]
            self.header.setText(f"{self._title} {spin}")
        else:
            arrow = "▼" if self.header.isChecked() else "▶︎"
            self.header.setText(f"{self._title} {arrow}")

    def _tick_spinner(self) -> None:
        if not self._streaming:
            self._spin_timer.stop()
            self._update_header_text()
            return
        self._spin_idx = (self._spin_idx + 1) % len(self._SPINNER)
        self._update_header_text()

    def _toggle(self, checked: bool) -> None:
        self.body.setVisible(checked)
        self._update_header_text()

    def _autosize(self) -> None:
        doc = self.body.document()
        doc.setTextWidth(self.body.viewport().width())
        self.body.setFixedHeight(int(doc.size().height()) + 4)


# ---------------------------------------------------------------------------
# Clickable thumbnail
# ---------------------------------------------------------------------------
class ClickablePixmapItem(QGraphicsPixmapItem):
    """'QGraphicsPixmapItem' that opens a full-res dialog on click."""

    def __init__(
        self,
        original: QPixmap,
        thumbnail: QPixmap,
        callback,
        parent: Optional[QGraphicsPixmapItem] = None,
    ) -> None:
        super().__init__(parent)
        self.original = original
        self.thumbnail = thumbnail
        self._callback = callback
        self.setPixmap(self.thumbnail)
        self.setAcceptHoverEvents(True)
        self.setTransformationMode(Qt.SmoothTransformation)
        self.setShapeMode(QGraphicsPixmapItem.ShapeMode.BoundingRectShape)

        self._shadow = QGraphicsDropShadowEffect()
        self._shadow.setBlurRadius(30)
        self._shadow.setColor(QColor(80, 160, 255, 180))
        self._shadow.setOffset(0, 0)
        self._shadow.setEnabled(False)
        self.setGraphicsEffect(self._shadow)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._callback is not None:
            self._callback(self)
        super().mousePressEvent(event)

    def hoverEnterEvent(self, event) -> None:  # noqa: N802
        self._shadow.setEnabled(True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802
        self._shadow.setEnabled(False)
        super().hoverLeaveEvent(event)


# ---------------------------------------------------------------------------
# Molecule queue chips
# ---------------------------------------------------------------------------
class MolChip(QLabel):
    """Small rounded label visualising one SMILES string's status."""

    CHIP_WIDTH = 60
    CHIP_HEIGHT = 26

    def __init__(self, smiles: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.smiles = smiles
        self.setFixedSize(self.CHIP_WIDTH, self.CHIP_HEIGHT)
        self.setToolTip(smiles)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText(smiles if len(smiles) <= 14 else smiles[:13] + "…")
        self.apply_status("pending")

    @staticmethod
    def _rgba(hex_color: str, alpha: float) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    @staticmethod
    def _darken(hex_color: str, amount: int = 50) -> str:
        h = hex_color.lstrip("#")
        r = max(0, int(h[0:2], 16) - amount)
        g = max(0, int(h[2:4], 16) - amount)
        b = max(0, int(h[4:6], 16) - amount)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _style(self, status: str, thread_color: Optional[str] = None) -> str:
        radius = 9
        if status == "pending":
            bg, border, color, weight = (
                "rgba(140,150,170,0.22)",
                "rgba(180,190,210,0.38)",
                "#c8d0dc",
                "normal",
            )
        elif status == "orca":
            c = thread_color or "#61afef"
            cd = self._darken(c, 40)
            bg = (
                f"qlineargradient(x1:0, y1:0, x2:0, y2:1, "
                f"stop:0 {self._rgba(c, 0.95)}, stop:1 {self._rgba(cd, 0.95)})"
            )
            border, color, weight = self._rgba(c, 1.0), "#ffffff", "bold"
        elif status == "prop":
            c = thread_color or "#61afef"
            cd = self._darken(c, 40)
            bg = (
                f"qlineargradient(x1:0, y1:0, x2:0, y2:1, "
                f"stop:0 {self._rgba(c, 0.32)}, stop:1 {self._rgba(cd, 0.28)})"
            )
            border, color, weight = self._rgba(c, 0.50), "#e8eef8", "normal"
        elif status == "error":
            bg = (
                "qlineargradient(x1:0, y1:0, x2:0, y2:1, "
                "stop:0 rgba(224,108,117,0.95), stop:1 rgba(180,70,80,0.95))"
            )
            border, color, weight = "#e06c75", "#ffffff", "bold"
        else:
            bg, border, color, weight = (
                "rgba(140,150,170,0.22)",
                "rgba(180,190,210,0.38)",
                "#c8d0dc",
                "normal",
            )

        return (
            f"QLabel {{ background-color: {bg}; border: 1px solid {border}; "
            f"border-radius: {radius}px; color: {color}; "
            f"font-family: 'Consolas'; font-size: 12px; "
            f"font-weight: {weight}; padding: 1px 6px; }}"
        )

    def apply_status(self, status: str, thread_color: Optional[str] = None) -> None:
        self.setStyleSheet(self._style(status, thread_color))


class MolQueueWidget(QWidget):
    """Horizontal queue of chips showing MolCalc progress per molecule."""

    _PALETTE = [
        "#61afef", "#98c379", "#e5c07b", "#c678dd",
        "#56b6c2", "#d19a66", "#ff9ecb", "#8a93a6",
    ]

    def __init__(self, smiles_list: list[str], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._chips: dict[str, MolChip] = {}
        self._status: dict[str, str] = {}
        self._thread_colors: dict[Union[int, str], str] = {}
        self._smile_thread: dict[str, Union[int, str]] = {}
        self._color_idx = 0
        self._animations: list[QPropertyAnimation] = []

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(6, 3, 6, 3)
        self._layout.setSpacing(6)
        self._layout.addStretch(1)

        for s in smiles_list:
            self._add_chip(s)

    def _add_chip(self, smiles: str) -> None:
        if smiles in self._chips:
            return
        chip = MolChip(smiles, self)
        self._layout.insertWidget(self._layout.count() - 1, chip)
        self._chips[smiles] = chip
        self._status[smiles] = "pending"

    def update_status(
        self, smiles: str, status: str, thread_id: Optional[Union[int, str]] = None
    ) -> None:
        if smiles not in self._chips:
            return

        if status == "orca":
            if thread_id is not None and thread_id not in self._thread_colors:
                self._thread_colors[thread_id] = self._PALETTE[
                    self._color_idx % len(self._PALETTE)
                ]
                self._color_idx += 1
            self._smile_thread[smiles] = thread_id
            color = self._thread_colors.get(thread_id)
            self._chips[smiles].apply_status("orca", color)
            self._status[smiles] = "orca"
            return

        if status == "prop":
            tid = self._smile_thread.get(smiles)
            color = self._thread_colors.get(tid) if tid is not None else None
            self._chips[smiles].apply_status("prop", color)
            self._status[smiles] = "prop"
            return

        if status == "error":
            self._chips[smiles].apply_status("error")
            self._status[smiles] = "error"
            return

        if status == "done":
            self._remove_chip(smiles)

    def _remove_chip(self, smiles: str) -> None:
        chip = self._chips.pop(smiles, None)
        if chip is None:
            return
        self._status.pop(smiles, None)
        self._smile_thread.pop(smiles, None)

        # Freeze the chip at its current screen position, then animate it out.
        local_pos = chip.mapTo(self, QPoint(0, 0))
        w, h = chip.width(), chip.height()

        self._layout.removeWidget(chip)
        chip.setParent(self)
        chip.setGeometry(local_pos.x(), local_pos.y(), w, h)
        chip.raise_()
        chip.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        eff = QGraphicsOpacityEffect(chip)
        chip.setGraphicsEffect(eff)
        fade = QPropertyAnimation(eff, b"opacity", chip)
        fade.setDuration(200)
        fade.setStartValue(1.0)
        fade.setEndValue(0.0)
        fade.setEasingCurve(QEasingCurve.Type.OutCubic)

        rise = QPropertyAnimation(chip, b"pos", chip)
        rise.setDuration(200)
        rise.setStartValue(QPoint(local_pos.x(), local_pos.y()))
        rise.setEndValue(QPoint(local_pos.x(), local_pos.y() - 40))
        rise.setEasingCurve(QEasingCurve.Type.OutCubic)

        def _cleanup() -> None:
            try:
                chip.setParent(None)
                chip.deleteLater()
            except RuntimeError:
                pass
            self._animations = [a for a in self._animations if a is not fade and a is not rise]

        rise.finished.connect(_cleanup)
        self._animations.extend([fade, rise])
        fade.start()
        rise.start()


# ---------------------------------------------------------------------------
# Threading helpers
# ---------------------------------------------------------------------------
class WorkerThread(QThread):
    """Run a callable off the GUI thread and emit its result or error."""

    progress = Signal(int)
    results = Signal(object)
    log = Signal(str)

    def __init__(self, task, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._task = task

    def run(self) -> None:
        try:
            self.results.emit(self._task())
        except Exception as exc:
            traceback.print_exc()
            self.log.emit(str(exc))


class QueueWriter:
    """File-like adapter that funnels child-process stdout into a queue."""

    def __init__(self, message_queue: "Queue") -> None:
        self._q = message_queue

    def write(self, s: str) -> None:
        if s and s.strip():
            self._q.put(s.rstrip("\n"))

    def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    """Application window hosting every functional page."""

    START_BUTTON_QSS = """
    QPushButton {
        background-color: rgba(168, 208, 255, 0.45);
        border-radius: 15px;
        color: #ffffff;
        text-align: center;
    }
    QPushButton:hover   { background-color: #3b8bca; }
    QPushButton:pressed { background-color: #1b6bAA; padding-top:8px;
                          padding-left:2px; padding-bottom:2px; }
    QPushButton:disabled { background-color: rgba(66, 79, 104, 0.5); }
    """

    STOP_BUTTON_QSS = """
    QPushButton {
        background-color: rgba(255, 190, 190, 0.45);
        border-radius: 15px;
        color: #ffffff;
        text-align: center;
    }
    QPushButton:hover   { background-color: #ca3b3b; }
    QPushButton:pressed { background-color: #AA1b1b; padding-left:-2px;
                          padding-bottom:6px; }
    QPushButton:disabled { background-color: rgba(66, 79, 104, 0.5); }
    """

    AGENT_SEND_BUTTON_QSS = """
    QPushButton {
        background-color: rgba(168, 208, 255, 0.45);
        color: #ffffff;
        text-align: center;
        border-radius: 20px;
        padding-left: -3px;
    }
    QPushButton:hover   { background-color: #3b8bca; }
    QPushButton:pressed { background-color: #1b6bAA; padding-top:8px;
                          padding-left:2px; padding-bottom:2px; }
    QPushButton:disabled { background-color: rgba(66, 79, 104, 0.5); }
    """

    AGENT_STOP_BUTTON_QSS = """
    QPushButton {
        background-color: rgba(255, 190, 190, 0.45);
        border-radius: 20px;
        padding-left: 0px;
    }
    QPushButton:hover   { background-color: #ca3b3b; }
    QPushButton:pressed { background-color: #AA1b1b; padding-left:-2px;
                          padding-bottom:6px; }
    QPushButton:disabled { background-color: rgba(66, 79, 104, 0.5); }
    """

    DIALOG_QSS = """
    QDialog { background-color: #1e2a44; }
    QDialog QLabel { color: #e8eef8; font-family: 'Consolas'; font-size: 13px; }

    QTableWidget {
        background-color: #24304d; color: #e8eef8;
        gridline-color: #3a4a68; border: 1px solid #3a4a68;
        border-radius: 8px; outline: none; font-family: 'Consolas';
    }
    QTableWidget::item { padding: 2px 6px; }
    QTableWidget::item:selected { background: transparent; color: #e8eef8; }
    QHeaderView::section {
        background-color: #2d3b55; color: #ffffff;
        border: none; border-right: 1px solid #3a4a68;
        border-bottom: 1px solid #3a4a68;
        padding: 4px 8px; font-weight: bold; font-family: 'Consolas';
    }

    QComboBox {
        font: 12pt "Consolas";
        background-color: rgba(66, 79, 104, 1);
        border: 1px solid transparent;
        border-radius: 10px;
        font-size: 14px;
        color: rgb(255, 255, 255);
        padding-left: 8px; padding-right: 8px;
        min-height: 20px;
    }
    QComboBox:hover { background-color: rgb(74, 88, 115); border: 1px solid #3478e8; }
    QComboBox:focus { border: 1px solid #3478e8; }
    QComboBox:disabled { background-color: rgba(66, 79, 104, 0.35);
                         color: #6b7280;
                         border: 1px solid rgba(120, 130, 150, 0.2); }
    QComboBox::drop-down {
        subcontrol-origin: padding; subcontrol-position: center right;
        width: 28px; border: none;
        border-top-right-radius: 10px; border-bottom-right-radius: 10px;
        background: transparent;
    }
    QComboBox::drop-down:hover   { background-color: rgb(37, 52, 81); }
    QComboBox::drop-down:pressed { background-color: rgb(204, 204, 204); }
    QComboBox::down-arrow { image: url(:/icons/down_arrow.svg); width: 14px; height: 14px; }
    QComboBox::down-arrow:disabled { image: url(:/icons/down_arrow.svg); opacity: 0.4; }
    QComboBox QAbstractItemView {
        background-color: rgba(66, 79, 104, 1);
        color: rgb(255, 255, 255);
        border: 1px solid #3478e8;
        border-radius: 8px;
        selection-background-color: #3478e8;
        selection-color: #ffffff;
        outline: none; padding: 4px; font-family: 'Consolas';
    }
    QComboBox QAbstractItemView::item { min-height: 24px;
                                        padding-left: 8px; padding-right: 8px;
                                        border-radius: 6px; }
    QComboBox QAbstractItemView::item:hover { background-color: rgb(37, 52, 81); }
    QComboBox QAbstractItemView::item:selected {
        background-color: #3478e8; color: #ffffff;
    }

    QSpinBox, QDoubleSpinBox {
        font: 12pt "Consolas";
        background-color: rgba(66, 79, 104, 1);
        border: 0px solid #e2e4e8;
        border-radius: 10px;
        font-size: 14px;
        color: rgb(255, 255, 255);
        padding-right: 26px;
        min-height: 26px;
    }
    QSpinBox:hover, QDoubleSpinBox:hover { border: 1px solid #3478e8; }
    QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid #3478e8; }
    QSpinBox:disabled, QDoubleSpinBox:disabled {
        background-color: rgba(66, 79, 104, 0.35);
        color: #6b7280;
        border: 1px solid rgba(120, 130, 150, 0.2);
    }
    QSpinBox::up-button, QDoubleSpinBox::up-button {
        subcontrol-origin: border; subcontrol-position: top right;
        border-radius: 5px; width: 20px; height: 12px;
        margin: 2px 2px 0px 0px; background: transparent; border: none;
    }
    QSpinBox::down-button, QDoubleSpinBox::down-button {
        subcontrol-origin: border; subcontrol-position: bottom right;
        border-radius: 5px; width: 20px; height: 12px;
        margin: 0px 2px 2px 0px; background: transparent; border: none;
    }
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
        background-color: rgb(37, 52, 81);
    }
    QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
    QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
        background-color: rgb(204, 204, 204);
    }
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
        image: url(:/icons/up_arrow.svg); width: 10px; height: 10px;
    }
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
        image: url(:/icons/down_arrow.svg); width: 10px; height: 10px;
    }

    QCheckBox { color: #ffffff; font-family: 'Consolas'; font-size: 14px; spacing: 8px; }
    QCheckBox::indicator {
        width: 18px; height: 18px;
        border: 1px solid rgb(120, 138, 172);
        border-radius: 5px;
        background-color: rgba(66, 79, 104, 1);
    }
    QCheckBox::indicator:hover   { border: 1px solid #3478e8;
                                   background-color: rgb(74, 88, 115); }
    QCheckBox::indicator:checked { border: 1px solid #3478e8;
                                   image: url(:/icons/check.svg); }
    QCheckBox::indicator:disabled { background-color: rgba(66, 79, 104, 0.35);
                                    border: 1px solid rgba(120, 130, 150, 0.3); }

    QPushButton {
        background-color: rgba(168, 208, 255, 0.45);
        color: #ffffff;
        border: none; border-radius: 8px;
        padding: 6px 18px; min-width: 70px;
        font-family: 'Consolas'; font-size: 13px;
    }
    QPushButton:hover   { background-color: #3b8bca; }
    QPushButton:pressed { background-color: #1b6bAA; }
    """

    MESSAGE_BOX_QSS = """
    QMessageBox { background-color: #1e2a44; }
    QMessageBox QLabel { color: #e8eef8; font-family: 'Consolas';
                         font-size: 13px; min-width: 260px; }
    QMessageBox QPushButton {
        background-color: rgba(168, 208, 255, 0.45);
        color: #ffffff; border: none; border-radius: 8px;
        padding: 6px 18px; min-width: 70px;
        font-family: 'Consolas'; font-size: 13px;
    }
    QMessageBox QPushButton:hover   { background-color: #3b8bca; }
    QMessageBox QPushButton:pressed { background-color: #1b6bAA; }
    """

    # Two-queue protocol between GUI and agent child process.
    MSG_PREFIX_THINK = "think"
    MSG_PREFIX_REPLY = "reply"
    MSG_PREFIX_INIT = "_init"
    MSG_PREFIX_END = "_end"
    MSG_PREFIX_ERROR = "_error_"
    MSG_PREFIX_PROGRESS = "_progr"

    def __init__(self) -> None:
        super().__init__()

        self.mol_calc_dir = os.path.join(self.get_base_dir(), "mol_calc")
        os.makedirs(self.mol_calc_dir, exist_ok=True)
        self.plots_dir = os.path.join(self.get_base_dir(), "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        self.temp_dir = os.path.join(self.get_base_dir(), "temp")
        os.makedirs(self.temp_dir, exist_ok=True)

        self._load_ui()

        # Order matters: each initialiser depends on widgets installed by the
        # previous ones.
        self._init_state_widgets()
        self._init_main_ui()
        self._init_home()
        self._init_mol_calc()
        self._init_data()
        self._init_ml()
        self._init_dl()
        self._init_ranking()
        self._init_evaluation()
        self._init_agent()
        self._init_prediction()

    # ------------------------------------------------------------------
    # UI loading
    # ------------------------------------------------------------------
    def _load_ui(self) -> None:
        """Load 'mygui.ui' and apply window-level cosmetic properties."""
        if sys.platform == "win32":
            my_appid = f"nku.feel.ehcma-pvk.v1"
            try:
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(my_appid)
            except Exception:
                pass

        loader = QUiLoader()
        ui_path = os.path.join(self.get_base_dir(), "mygui.ui")
        ui_file = QFile(ui_path)
        if not ui_file.open(QIODevice.ReadOnly):
            print(f"Cannot open UI file: {ui_path}")
            sys.exit(-1)
        self.central_widget = loader.load(ui_file)
        ui_file.close()

        if self.central_widget is None:
            print(f"ERROR: QUiLoader failed to load {ui_path}.")
            print("Check that the file exists, the XML is valid, and res_rc is compiled.")
            sys.exit(-1)

        self.setWindowTitle("Explainable Hierarchy-Constrained Multimodal Agent")

        icon = QIcon(":/icons/window.png")
        self.setWindowIcon(icon)

        self.setCentralWidget(self.central_widget)

        self.central_widget.adjustSize()
        win_size = self.central_widget.size()

        if win_size.width() > 100 and win_size.height() > 100:
            self.setFixedSize(win_size)
        else:
            self.setFixedSize(1280, 800)

        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMaximizeButtonHint)

        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.central_widget.setAttribute(Qt.WA_TranslucentBackground, True)
        self.central_widget.setStyleSheet("background-color: rgba(40, 80, 160, 60);")

        if GlobalBlur is not None:
            try:
                GlobalBlur(int(self.winId()), Dark=True, Acrylic=False, QWidget=self)
            except Exception:
                pass


    # ------------------------------------------------------------------
    # Status bar / state log
    # ------------------------------------------------------------------
    def _init_state_widgets(self) -> None:
        """Install the four-state indicator, message line, and log page."""
        self.lbl_status_normal = self._find_child_widget(QLabel, "lbl_status_normal")
        self.lbl_status_error = self._find_child_widget(QLabel, "lbl_status_error")
        self.lbl_status_unknown = self._find_child_widget(QLabel, "lbl_status_unknown")
        self.lbl_status_warning = self._find_child_widget(QLabel, "lbl_status_warning")

        self.txt_status_message = self._find_child_widget(QPlainTextEdit, "txt_status_message")
        self.txt_status_message.setReadOnly(True)
        self.txt_status_message.setPlainText("Ready.")

        self.pg_state_log = self._find_child_widget(QWidget, "pg_state_log")
        self.btn_status_log = self._find_child_widget(QPushButton, "btn_status_log")
        self.btn_state_log_close = self._find_child_widget(QPushButton, "btn_state_log_close")
        self.txt_state_log = self._find_child_widget(QTextEdit, "txt_state_log")

        if self.txt_state_log is not None:
            self.txt_state_log.setReadOnly(True)
            self.txt_state_log.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)

        self._previous_page = None
        self._last_logged_state = None
        self._last_logged_message = None

        if self.btn_status_log is not None:
            self.btn_status_log.clicked.connect(self.show_state_log)
        if self.btn_state_log_close is not None:
            self.btn_state_log_close.clicked.connect(self.close_state_log)

        self.set_state("normal")

        self.pbar_main = self._find_child_widget(QProgressBar, "pbar_main")
        self.pbar_main.setValue(0)

    # ------------------------------------------------------------------
    # Main chrome
    # ------------------------------------------------------------------
    def _init_main_ui(self) -> None:
        """Wire the navigation rail, the stacked pages, and the usage meters."""
        self.frm_navigation = self._find_child_widget(QFrame, "frm_navigation")

        self.btn_open_runtime_dir = self._find_child_widget(QPushButton, "btn_open_runtime_dir")
        self.btn_open_runtime_dir.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.get_base_dir()))
        )

        self.btn_clear_cache = self._find_child_widget(QPushButton, "btn_clear_cache")
        self.btn_clear_cache.clicked.connect(self.clear_cache)

        self.stk_main = self._find_child_widget(QStackedWidget, "stk_main")

        # The data-page 'connect' is deferred to '_init_data' because its
        # dependent timer does not exist yet.
        self.pg_data = self._find_child_widget(QWidget, "pg_data")
        self.btn_nav_data = self._find_child_widget(QPushButton, "btn_nav_data")

        self.pg_ml = self._find_child_widget(QWidget, "pg_ml")
        self.btn_nav_ml = self._find_child_widget(QPushButton, "btn_nav_ml")
        self.btn_nav_ml.clicked.connect(lambda: self.stk_main.setCurrentWidget(self.pg_ml))

        self.pg_mol = self._find_child_widget(QWidget, "pg_mol")
        self.btn_nav_mol = self._find_child_widget(QPushButton, "btn_nav_mol")
        self.btn_nav_mol.clicked.connect(lambda: self.stk_main.setCurrentWidget(self.pg_mol))

        self.pg_dl = self._find_child_widget(QWidget, "pg_dl")
        self.btn_nav_dl = self._find_child_widget(QPushButton, "btn_nav_dl")
        self.btn_nav_dl.clicked.connect(lambda: self.stk_main.setCurrentWidget(self.pg_dl))

        self.pg_eval = self._find_child_widget(QWidget, "pg_eval")
        self.btn_nav_eval = self._find_child_widget(QPushButton, "btn_nav_eval")
        self.btn_nav_eval.clicked.connect(lambda: self.stk_main.setCurrentWidget(self.pg_eval))

        self.pg_agent = self._find_child_widget(QWidget, "pg_agent")
        self.btn_nav_agent = self._find_child_widget(QPushButton, "btn_nav_agent")
        self.btn_nav_agent.clicked.connect(lambda: self.stk_main.setCurrentWidget(self.pg_agent))

        self.pg_pred = self._find_child_widget(QWidget, "pg_pred")
        self.btn_nav_pred = self._find_child_widget(QPushButton, "btn_nav_pred")
        self.btn_nav_pred.clicked.connect(lambda: self.stk_main.setCurrentWidget(self.pg_pred))

        self.frm_navigation.setAttribute(Qt.WA_Hover, True)
        self.navigation_collapsed_width = self.frm_navigation.width()
        self.navigation_expanded_width = 171
        self.frm_navigation.setFixedWidth(self.navigation_collapsed_width)

        self.navigation_animation = QVariantAnimation(self)
        self.navigation_animation.setDuration(200)
        self.navigation_animation.setEasingCurve(QEasingCurve.Type.OutQuad)
        self._navigation_buttons = self.frm_navigation.findChildren(QPushButton)
        self.navigation_animation.valueChanged.connect(
            lambda value: self._apply_navigation_width(int(value))
        )
        self._apply_navigation_width(self.navigation_collapsed_width)
        self.frm_navigation.installEventFilter(self)

        # 500 ms is fast enough for a smooth read-out without noticeable cost.
        self.system_usage_timer = QTimer(self)
        self.system_usage_timer.timeout.connect(self.update_system_usage)
        self.system_usage_timer.start(500)

        self.pbar_cpu = self._find_child_widget(QProgressBar, "pbar_cpu")
        self.pbar_cpu.setValue(0)
        self.pbar_gpu = self._find_child_widget(QProgressBar, "pbar_gpu")
        self.pbar_gpu.setValue(0)

    def _apply_navigation_width(self, width: int) -> None:
        """Resize the rail and keep its buttons clickable only when fully open."""
        self.frm_navigation.setFixedWidth(width)
        enabled = width >= self.navigation_expanded_width
        for btn in self._navigation_buttons:
            btn.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Home page
    # ------------------------------------------------------------------
    def _init_home(self) -> None:
        """Install the animated bubble background and the mask magnifier."""
        self.pg_home = self._find_child_widget(QWidget, "pg_home")
        self.lay_home = self._find_child_widget(QVBoxLayout, "lay_home")
        self.bubble_widget = BubbleBackgroundWidget(self.pg_home)
        self.lay_home.addWidget(self.bubble_widget)

        self.frm_logo = self._find_child_widget(QFrame, "frm_logo")
        self.btn_logo = self._find_child_widget(QPushButton, "btn_logo")
        self.btn_logo.clicked.connect(lambda: self.stk_main.setCurrentWidget(self.pg_home))

        # 'MaskCanvas' cannot be instantiated from the .ui without a
        # promoted-widget rule, so the placeholder is replaced in-place.
        placeholder = self._find_child_widget(QWidget, "wgt_mask_canvas")
        self.wgt_mask_canvas = MaskCanvas(placeholder.parentWidget())
        if placeholder.parentWidget().layout() is not None:
            placeholder.parentWidget().layout().replaceWidget(placeholder, self.wgt_mask_canvas)
        else:
            self.wgt_mask_canvas.setGeometry(placeholder.geometry())
            self.wgt_mask_canvas.show()
        placeholder.deleteLater()

    # ------------------------------------------------------------------
    # Molecular calculation
    # ------------------------------------------------------------------
    def _init_mol_calc(self) -> None:
        """Wire the SMILES editor, ORCA/Basis settings, queue, and result table."""
        self.frm_mol_settings = self._find_child_widget(QFrame, "frm_mol_settings")
        self.mol_calc_settings_target_position = self.frm_mol_settings.pos()
        self.mol_calc_settings_hidden_position = QPoint(
            self.mol_calc_settings_target_position.x(),
            self.mol_calc_settings_target_position.y() + 500,
        )
        self.frm_mol_settings.move(self.mol_calc_settings_hidden_position)

        self.mol_calc_settings_shadow = QGraphicsDropShadowEffect(self.frm_mol_settings)
        self.mol_calc_settings_shadow.setBlurRadius(0)
        self.mol_calc_settings_shadow.setColor(QColor(80, 160, 255, 180))
        self.mol_calc_settings_shadow.setOffset(2, 2)
        self.frm_mol_settings.setGraphicsEffect(self.mol_calc_settings_shadow)

        self.mol_calc_settings_target_blur_radius = 10
        self.btn_mol_settings = self._find_child_widget(QPushButton, "btn_mol_settings")
        self.btn_mol_settings.clicked.connect(self.toggle_mol_calc_settings)
        self._is_mol_calc_settings_visible = False

        self.anim_mol_calc_settings_position = QPropertyAnimation(self.frm_mol_settings, b"pos")
        self.anim_mol_calc_settings_position.setDuration(200)
        self.anim_mol_calc_settings_position.setEasingCurve(QEasingCurve.Type.OutBack)

        self.anim_mol_calc_settings_shadow = QPropertyAnimation(
            self.mol_calc_settings_shadow, b"blurRadius"
        )
        self.anim_mol_calc_settings_shadow.setDuration(200)
        self.anim_mol_calc_settings_shadow.setEasingCurve(QEasingCurve.Type.OutBack)

        self.btn_mol_library = self._find_child_widget(QPushButton, "btn_mol_library")
        self.btn_mol_library.clicked.connect(self.open_molecule_library)

        # Validation is debounced to 300 ms so typing does not re-parse every
        # keystroke.
        self.txt_mol_smiles = self._find_child_widget(QPlainTextEdit, "txt_mol_smiles")
        self.txt_mol_smiles.setPlaceholderText("One SMILES per line")
        self.txt_mol_smiles.setLineWrapMode(QPlainTextEdit.NoWrap)

        self.lbl_mol_counts = self._find_child_widget(QLabel, "lbl_mol_counts")
        self.lbl_mol_counts.setText(
            "<span style='color:#98c379;'>Valid - 0</span> "
            "<span style='color:#ffffff;'> | </span> "
            "<span style='color:#e06c75;'>Invalid - 0</span>"
        )

        self.valid_smiles_list: list[str] = []
        self.invalid_smiles_lines: list[tuple[int, str, str]] = []

        self.mol_validate_timer = QTimer(self)
        self.mol_validate_timer.setSingleShot(True)
        self.mol_validate_timer.setInterval(300)
        self.mol_validate_timer.timeout.connect(self.validate_mol_smiles)
        self.txt_mol_smiles.textChanged.connect(self.mol_validate_timer.start)

        self._find_child_widget(QPushButton, "btn_mol_clear").clicked.connect(
            self.txt_mol_smiles.clear
        )
        self._find_child_widget(QPushButton, "btn_mol_import").clicked.connect(
            self.import_mol_smiles_csv
        )
        self._find_child_widget(QPushButton, "btn_mol_export").clicked.connect(
            self.export_mol_smiles_csv
        )

        self.cmb_mol_method = self._find_child_widget(QComboBox, "cmb_mol_method")
        self.cmb_mol_basis = self._find_child_widget(QComboBox, "cmb_mol_basis")
        self.chk_mol_opt = self._find_child_widget(QCheckBox, "chk_mol_opt")
        self.chk_mol_freq = self._find_child_widget(QCheckBox, "chk_mol_freq")
        self.chk_mol_tightscf = self._find_child_widget(QCheckBox, "chk_mol_tightscf")
        self.chk_mol_multithread = self._find_child_widget(QCheckBox, "chk_mol_multithread")
        self.spn_mol_ncore = self._find_child_widget(QSpinBox, "spn_mol_ncore")
        self.dsp_mol_cpu = self._find_child_widget(QDoubleSpinBox, "dsp_mol_cpu")

        self.tbl_mol_results = self._find_child_widget(QTableWidget, "tbl_mol_results")
        self.is_mol_table_dirty = False
        self.is_mol_table_loading = False
        self.tbl_mol_results.setEditTriggers(
            QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed
        )
        self.tbl_mol_results.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tbl_mol_results.customContextMenuRequested.connect(
            self.on_mol_table_context_menu
        )
        self.tbl_mol_results.itemChanged.connect(self.on_mol_table_item_changed)

        self.btn_mol_start = self._find_child_widget(QPushButton, "btn_mol_start")
        self.btn_mol_start.setStyleSheet(self.START_BUTTON_QSS)
        self.btn_mol_start.clicked.connect(self.start_mol_calc)

        self.btn_mol_stop = self._find_child_widget(QPushButton, "btn_mol_stop")
        self.btn_mol_stop.setStyleSheet(self.STOP_BUTTON_QSS)
        self.btn_mol_stop.setEnabled(False)
        self.btn_mol_stop.clicked.connect(self.stop_mol_calc)

        self.mol_calc_process = None
        self.mol_calc_output_dir = self.get_base_dir()
        self.last_mol_calc_row_count = 0
        self.mol_calc_message_queue: Queue = Queue()
        self.mol_queue_widget = None
        self.mol_calc_timer = QTimer(self)
        self.mol_calc_timer.timeout.connect(self.on_mol_calc_tick)
        self.mol_calc_queue_timer = QTimer(self)
        self.mol_calc_queue_timer.timeout.connect(self.on_mol_calc_queue_tick)
        self.frm_mol_queue = self._find_child_widget(QFrame, "frm_mol_queue")

    # ------------------------------------------------------------------
    # Data assembly
    # ------------------------------------------------------------------
    def _init_data(self) -> None:
        """Wire the dataset import, merge button, and live preview table."""
        self.txt_data_exp_path = self._find_child_widget(QPlainTextEdit, "txt_data_exp_path")
        self.txt_data_exp_path.setPlaceholderText(
            "Please select the experiment dataset CSV file..."
        )
        self.txt_data_smiles_path = self._find_child_widget(QPlainTextEdit, "txt_data_smiles_path")
        self.txt_data_smiles_path.setPlaceholderText(
            "Please select the SMILES dataset CSV file..."
        )

        self.cmb_data_method = self._find_child_widget(QComboBox, "cmb_data_method")
        self.cmb_data_basis = self._find_child_widget(QComboBox, "cmb_data_basis")

        self.btn_data_browse_exp = self._find_child_widget(QPushButton, "btn_data_browse_exp")
        self.btn_data_browse_exp.clicked.connect(self.browse_experiment_file)
        self.btn_data_browse_smiles = self._find_child_widget(QPushButton, "btn_data_browse_smiles")
        self.btn_data_browse_smiles.clicked.connect(self.browse_smiles_file)

        self.btn_data_merge = self._find_child_widget(QPushButton, "btn_data_merge")
        self.btn_data_merge.clicked.connect(self.merge_datasets)

        self.tbl_data_preview = self._find_child_widget(QTableWidget, "tbl_data_preview")
        self.lbl_data_counts = self._find_child_widget(QLabel, "lbl_data_counts")
        self.lbl_data_counts.setText(
            "<span style='color:#8b9bb4;'>Select experiment and SMILES datasets...</span>"
        )

        # Throttled to 220 ms so reading large CSVs does not block typing.
        self.data_counts_timer = QTimer(self)
        self.data_counts_timer.setSingleShot(True)
        self.data_counts_timer.setInterval(220)
        self.data_counts_timer.timeout.connect(self.refresh_data_counts)

        self.txt_data_exp_path.textChanged.connect(self.data_counts_timer.start)
        self.txt_data_smiles_path.textChanged.connect(self.data_counts_timer.start)
        self.cmb_data_method.currentTextChanged.connect(lambda _: self.data_counts_timer.start())
        self.cmb_data_basis.currentTextChanged.connect(lambda _: self.data_counts_timer.start())

        # Also refresh when the page is navigated to, so external edits to
        # mol_prop.csv are picked up immediately.
        self.btn_nav_data.clicked.connect(
            lambda: (self.stk_main.setCurrentWidget(self.pg_data),
                     self.data_counts_timer.start())
        )

        QTimer.singleShot(300, self.refresh_data_counts)

    # ------------------------------------------------------------------
    # Machine-learning training
    # ------------------------------------------------------------------
    def _init_ml(self) -> None:
        """Wire the algorithm-selection list, hyperparameters, and logs."""
        self.ml_optimal_model: Optional[str] = None

        self.algorithm_list_widget = self._find_child_widget(QListWidget, "lst_ml_algorithms")
        self.algorithm_list_widget.itemChanged.connect(self.on_algorithm_check_changed)

        self.spn_ml_seed = self._find_child_widget(QSpinBox, "spn_ml_seed")
        self.spn_ml_iterations = self._find_child_widget(QSpinBox, "spn_ml_iterations")
        self.spn_ml_init_points = self._find_child_widget(QSpinBox, "spn_ml_init_points")
        self.dsp_ml_split = self._find_child_widget(QDoubleSpinBox, "dsp_ml_split")

        self.txt_ml_optimal_model = self._find_child_widget(QPlainTextEdit, "txt_ml_optimal_model")
        self.txt_ml_optimal_model.setReadOnly(True)
        self.txt_ml_r2 = self._find_child_widget(QPlainTextEdit, "txt_ml_r2")
        self.txt_ml_r2.setReadOnly(True)
        self.txt_ml_mse = self._find_child_widget(QPlainTextEdit, "txt_ml_mse")
        self.txt_ml_mse.setReadOnly(True)
        self.txt_ml_mae = self._find_child_widget(QPlainTextEdit, "txt_ml_mae")
        self.txt_ml_mae.setReadOnly(True)

        self.cmb_ml_metric = self._find_child_widget(QComboBox, "cmb_ml_metric")

        self.txt_ml_log = self._find_child_widget(QTextEdit, "txt_ml_log")
        self.txt_ml_log.setReadOnly(True)
        self.txt_ml_log.setWordWrapMode(QTextOption.NoWrap)

        self.ml_performance_df = pd.DataFrame(columns=["name", "R2", "MSE", "MAE"])

        # Kept as a dict rather than inferred from widget text, so display
        # names can be localised without touching the worker contract.
        self.algorithm_display_names = {
            "LR": "LinearRegression",
            "Ridge": "RidgeRegression",
            "SVR": "SupportVectorRegression",
            "KNN": "KNeighborsRegressor",
            "DT": "DecisionTreeRegressor",
            "RF": "RandomForestRegressor",
            "ET": "ExtraTreesRegressor",
            "GBDT": "GradientBoostingRegressor",
            "AdaBoost": "AdaBoostRegressor",
            "XGBoost": "XGBRegressor",
            "LGBM": "LGBMRegressor",
            "CatBoost": "CatBoostRegressor",
        }

        self.btn_ml_start = self._find_child_widget(QPushButton, "btn_ml_start")
        self.btn_ml_start.clicked.connect(self.start_ml_training)

        self.ml_process = None
        self.selected_algorithm_count = 0
        self.ml_message_queue: Queue = Queue()
        self.ml_timer = QTimer(self)
        self.ml_timer.timeout.connect(self.on_ml_queue_tick)

        self.btn_ml_stop = self._find_child_widget(QPushButton, "btn_ml_stop")
        self.btn_ml_stop.setEnabled(False)
        self.btn_ml_stop.clicked.connect(self.stop_ml_training)

    # ------------------------------------------------------------------
    # Deep-learning training
    # ------------------------------------------------------------------
    def _init_dl(self) -> None:
        """Wire the ACIN hyperparameters and the live loss monitor."""
        self.spn_dl_epochs = self._find_child_widget(QSpinBox, "spn_dl_epochs")
        self.spn_dl_early_stop = self._find_child_widget(QSpinBox, "spn_dl_early_stop")
        self.cmb_dl_device = self._find_child_widget(QComboBox, "cmb_dl_device")
        self.spn_dl_seed = self._find_child_widget(QSpinBox, "spn_dl_seed")
        self.spn_dl_batch = self._find_child_widget(QSpinBox, "spn_dl_batch")
        self.dsp_dl_learning_rate = self._find_child_widget(
            QDoubleSpinBox, "dsp_dl_learning_rate"
        )
        self.dsp_dl_dropout = self._find_child_widget(QDoubleSpinBox, "dsp_dl_dropout")
        self.dsp_dl_split = self._find_child_widget(QDoubleSpinBox, "dsp_dl_split")

        self.txt_dl_r2 = self._find_child_widget(QPlainTextEdit, "txt_dl_r2")
        self.txt_dl_r2.setReadOnly(True)
        self.txt_dl_mse = self._find_child_widget(QPlainTextEdit, "txt_dl_mse")
        self.txt_dl_mse.setReadOnly(True)
        self.txt_dl_mae = self._find_child_widget(QPlainTextEdit, "txt_dl_mae")
        self.txt_dl_mae.setReadOnly(True)
        self.txt_dl_loss = self._find_child_widget(QPlainTextEdit, "txt_dl_loss")
        self.txt_dl_loss.setReadOnly(True)

        self.txt_dl_log = self._find_child_widget(QTextEdit, "txt_dl_log")
        self.txt_dl_log.setReadOnly(True)
        self.txt_dl_log.setWordWrapMode(QTextOption.NoWrap)

        self.dl_performance_df = pd.DataFrame(columns=["R2", "MSE", "MAE", "Loss"])

        self.btn_dl_start = self._find_child_widget(QPushButton, "btn_dl_start")
        self.btn_dl_start.clicked.connect(self.start_dl_training)

        self.dl_process = None
        self.dl_message_queue: Queue = Queue()
        self.dl_timer = QTimer(self)
        self.dl_timer.timeout.connect(self.on_dl_queue_tick)

        self.btn_dl_stop = self._find_child_widget(QPushButton, "btn_dl_stop")
        self.btn_dl_stop.setEnabled(False)
        self.btn_dl_stop.clicked.connect(self.stop_dl_training)

    # ------------------------------------------------------------------
    # Substructure ranking
    # ------------------------------------------------------------------
    def _init_ranking(self) -> None:
        """Wire the top-K spinbox, the substructure grid, and the spectrum view."""
        self.spn_rank_top_k = self._find_child_widget(QSpinBox, "spn_rank_top_k")

        self.btn_rank_goto = self._find_child_widget(QPushButton, "btn_rank_goto")
        self.btn_rank_goto.clicked.connect(
            lambda: self.stk_main.setCurrentWidget(self.pg_rank)
        )

        self.btn_rank_run = self._find_child_widget(QPushButton, "btn_rank_run")
        self.btn_rank_run.clicked.connect(self.start_ranking)
        self.btn_rank_run.setEnabled(True)
        self.btn_rank_run.setStyleSheet(self.START_BUTTON_QSS)
        self.rank_stop_event = Event()

        self.btn_rank_back = self._find_child_widget(QPushButton, "btn_rank_back")
        self.btn_rank_back.clicked.connect(
            lambda: self.stk_main.setCurrentWidget(self.pg_dl)
        )

        self.pg_rank = self._find_child_widget(QWidget, "pg_rank")

        # Hidden until the first ranking completes so the placeholder text is
        # not overlaid by an empty scene.
        self.gvw_rank_substructures = self._find_child_widget(
            QGraphicsView, "gvw_rank_substructures"
        )
        self.gvw_rank_substructures.setVisible(False)
        self.gvw_rank_substructures.setRenderHint(QPainter.Antialiasing)
        self.ranking_scene = QGraphicsScene()
        self.gvw_rank_substructures.setScene(self.ranking_scene)

        # Replaced lazily by a matplotlib canvas on first render, because the
        # Qt backend requires an active QApplication.
        self.gvw_rank_spectrum = self._find_child_widget(QGraphicsView, "gvw_rank_spectrum")
        self.gvw_rank_spectrum.setRenderHint(QPainter.Antialiasing)
        self.spectrum_scene = QGraphicsScene()
        self.gvw_rank_spectrum.setScene(self.spectrum_scene)

        self.acin_substructure_weights: Optional[pd.DataFrame] = None
        self.spectrum_frame: Optional[QFrame] = None
        self.spectrum_canvas: Optional[FigureCanvasQTAgg] = None
        self.spectrum_plotter: Optional[ACINSpectrumBar] = None
        self.spectrum_fig: Optional[Figure] = None

    # ------------------------------------------------------------------
    # SHAP evaluation
    # ------------------------------------------------------------------
    def _init_evaluation(self) -> None:
        """Wire the SHAP model selector, fitting toggle, and plot views."""
        self.cmb_eval_shap_model = self._find_child_widget(QComboBox, "cmb_eval_shap_model")
        # Editable + read-only line edit lets us display a placeholder while
        # still allowing the popup list to be rebuilt on click.
        self.cmb_eval_shap_model.setEditable(True)
        self.cmb_eval_shap_model.lineEdit().setReadOnly(True)
        self.cmb_eval_shap_model.lineEdit().setPlaceholderText("Please select a model...")
        self.cmb_eval_shap_model.setCurrentIndex(-1)
        self.cmb_eval_shap_model.lineEdit().setStyleSheet("QLineEdit {padding-left: 5px;}")
        self.cmb_eval_shap_model.lineEdit().setFocusPolicy(Qt.NoFocus)
        self.cmb_eval_shap_model.installEventFilter(self)

        self.chk_eval_gam = self._find_child_widget(QCheckBox, "chk_eval_gam")
        self.chk_eval_gam.setChecked(True)
        self.chk_eval_linear = self._find_child_widget(QCheckBox, "chk_eval_linear")
        self.chk_eval_linear.setChecked(False)

        self.btn_group_eval_fitting_type = QButtonGroup(self)
        self.btn_group_eval_fitting_type.setExclusive(True)
        self.btn_group_eval_fitting_type.addButton(self.chk_eval_gam)
        self.btn_group_eval_fitting_type.addButton(self.chk_eval_linear)

        self.spl_eval_plots = self._find_child_widget(QSplitter, "spl_eval_plots")
        self.spl_eval_plots.setSizes([478, 478])

        self.gvw_eval_beeswarm = self._find_child_widget(QGraphicsView, "gvw_eval_beeswarm")
        self.gvw_eval_beeswarm.setRenderHint(QPainter.Antialiasing)
        self.beeswarm_scene = QGraphicsScene()
        self.gvw_eval_beeswarm.setScene(self.beeswarm_scene)

        self.gvw_eval_dependency = self._find_child_widget(QGraphicsView, "gvw_eval_dependency")
        self.gvw_eval_dependency.setRenderHint(QPainter.Antialiasing)
        self.dependency_scene = QGraphicsScene()
        self.gvw_eval_dependency.setScene(self.dependency_scene)

        self.btn_eval_analyse = self._find_child_widget(QPushButton, "btn_eval_analyse")
        self.btn_eval_analyse.clicked.connect(self.start_analysis)
        self.btn_eval_analyse.setEnabled(True)
        self.btn_eval_analyse.setStyleSheet(self.START_BUTTON_QSS)
        self.analysis_stop_event = Event()

    # ------------------------------------------------------------------
    # Agent page
    # ------------------------------------------------------------------
    def _init_agent(self) -> None:
        """Wire the chat transcript, streaming widgets, and LLM configuration."""
        self.has_completed_initial_analysis = False
        self.agent_timer = QTimer(self)
        self.agent_timer.timeout.connect(self.on_agent_queue_tick)

        self._scroll_timer = QTimer(self)
        self._programmatic_scroll = False
        self._auto_scroll = True
        self._scroll_timer.setSingleShot(True)
        self._scroll_timer.setInterval(10)
        self._scroll_timer.timeout.connect(self._do_scroll_to_bottom)

        self.agent_message_queue: Queue = Queue()
        self.agent_command_queue: Queue = Queue()
        self.agent_stop_event = Event()
        self.is_agent_running = False
        self.current_think_box: Optional[ThinkBox] = None
        self.current_reply_bubble: Optional[MessageBubble] = None
        self.agent_process = None
        self.is_agent_reply_finished = False

        self.txt_agent_api_key = self._find_child_widget(QPlainTextEdit, "txt_agent_api_key")
        self.txt_agent_api_key.setPlainText(
            "sk-ws-H.PMDPMML.KJ59.MEUCIQCmXbQGKysMNF_Tw63RINGTT8TJYc-HjfkLnkcJKALvEgIgaPz-sGfwfo7xkR7Plt7GQ1YkdPPqBarIQwo-5HWpxos"
        )
        self.txt_agent_base_url = self._find_child_widget(QPlainTextEdit, "txt_agent_base_url")
        self.txt_agent_base_url.setPlainText("https://maas.qianwenaiapi.com/compatible-mode/v1")
        self.txt_agent_model_name = self._find_child_widget(QPlainTextEdit, "txt_agent_model_name")
        self.txt_agent_model_name.setPlainText("qwen3.8-flash")

        self.dsp_agent_temperature = self._find_child_widget(
            QDoubleSpinBox, "dsp_agent_temperature"
        )
        self.spn_agent_max_tokens = self._find_child_widget(QSpinBox, "spn_agent_max_tokens")
        self.dsp_agent_threshold = self._find_child_widget(QDoubleSpinBox, "dsp_agent_threshold")
        self.spn_agent_descriptor_top_k = self._find_child_widget(
            QSpinBox, "spn_agent_descriptor_top_k"
        )
        self.dsp_agent_split = self._find_child_widget(QDoubleSpinBox, "dsp_agent_split")
        self.spn_agent_seed = self._find_child_widget(QSpinBox, "spn_agent_seed")

        self.cmb_agent_shap_model = self._find_child_widget(QComboBox, "cmb_agent_shap_model")
        self.cmb_agent_shap_model.setEditable(True)
        self.cmb_agent_shap_model.lineEdit().setReadOnly(True)
        self.cmb_agent_shap_model.lineEdit().setPlaceholderText("Please select a model...")
        self.cmb_agent_shap_model.setCurrentIndex(-1)
        self.cmb_agent_shap_model.lineEdit().setStyleSheet("QLineEdit {padding-left: 5px;}")
        self.cmb_agent_shap_model.lineEdit().setFocusPolicy(Qt.NoFocus)
        self.cmb_agent_shap_model.installEventFilter(self)

        self.chk_agent_gam = self._find_child_widget(QCheckBox, "chk_agent_gam")
        self.chk_agent_gam.setChecked(True)
        self.chk_agent_linear = self._find_child_widget(QCheckBox, "chk_agent_linear")
        self.chk_agent_linear.setChecked(False)
        self.btn_group_agent_fitting_type = QButtonGroup(self)
        self.btn_group_agent_fitting_type.setExclusive(True)
        self.btn_group_agent_fitting_type.addButton(self.chk_agent_gam)
        self.btn_group_agent_fitting_type.addButton(self.chk_agent_linear)

        self.spn_agent_substructure_top_k = self._find_child_widget(
            QSpinBox, "spn_agent_substructure_top_k"
        )
        self.chk_agent_thinking = self._find_child_widget(QCheckBox, "chk_agent_thinking")

        self.scl_agent_chat = self._find_child_widget(QScrollArea, "scl_agent_chat")
        self.scl_agent_chat.setWidgetResizable(True)
        self.scl_agent_chat.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scl_agent_chat.setViewportMargins(0, 0, 120, 0)

        self.chat_container = QWidget()
        self.messages_layout = QVBoxLayout(self.chat_container)
        self.messages_layout.setSpacing(14)
        self.messages_layout.addStretch(1)
        self.scl_agent_chat.setWidget(self.chat_container)

        scroll_bar = self.scl_agent_chat.verticalScrollBar()
        scroll_bar.valueChanged.connect(self.on_chat_scrolled)
        scroll_bar.rangeChanged.connect(self.on_chat_scroll_range_changed)

        # Sticky header floats over the transcript while a ThinkBox's own
        # header has scrolled off-screen; geometry is recomputed on scroll.
        self.sticky_header_btn = QToolButton(self.scl_agent_chat.viewport())
        self.sticky_header_btn.setObjectName("stickyHeader")
        self.sticky_header_btn.setIcon(QIcon(":/icons/reasoning_process.svg"))
        self.sticky_header_btn.setIconSize(QSize(20, 20))
        self.sticky_header_btn.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.sticky_header_btn.setStyleSheet(
            "QToolButton#stickyHeader { border: none; border-radius: 8px; "
            "background-color: rgba(30, 42, 68, 200); color: #FFFFFF; "
            "font-family: 'Consolas'; font-size: 14px; font-weight: bold; "
            "padding-left: 0px; padding-right: 12px; "
            "border-bottom: 1px solid rgba(168, 208, 255, 0.3); text-align: left; }"
            "QToolButton#stickyHeader:hover { color: #3478E8; }"
        )
        self.sticky_header_btn.hide()
        self.sticky_header_btn.clicked.connect(self.on_sticky_header_clicked)

        self._sticky_timer = QTimer(self)
        self._sticky_timer.setSingleShot(True)
        self._sticky_timer.setInterval(16)
        self._sticky_timer.timeout.connect(self._update_sticky_header)

        self.txt_agent_input = self._find_child_widget(QTextEdit, "txt_agent_input")
        self.txt_agent_input.setPlaceholderText(
            "Please complete the initial analysis first..."
        )
        self.txt_agent_input.setEnabled(False)
        self.txt_agent_input.installEventFilter(self)

        self.btn_agent_send = self._find_child_widget(QPushButton, "btn_agent_send")
        self.btn_agent_send.clicked.connect(self.send_agent_message)
        self.btn_agent_send.setEnabled(True)
        self.btn_agent_send.setIcon(QIcon(":/icons/send.svg"))
        self.btn_agent_send.setIconSize(QSize(28, 28))
        self.btn_agent_send.setStyleSheet(self.AGENT_SEND_BUTTON_QSS)

        self.btn_agent_new_chat = self._find_child_widget(QPushButton, "btn_agent_new_chat")
        self.btn_agent_new_chat.clicked.connect(self.clear_agent_chat)
        self.btn_agent_new_chat.setEnabled(True)

        self.frm_agent_settings = self._find_child_widget(QFrame, "frm_agent_settings")
        self.agent_settings_target_position = self.frm_agent_settings.pos()
        self.agent_settings_hidden_position = QPoint(
            self.agent_settings_target_position.x(),
            self.agent_settings_target_position.y() + 500,
        )
        self.frm_agent_settings.move(self.agent_settings_hidden_position)

        self.agent_settings_shadow = QGraphicsDropShadowEffect(self.frm_agent_settings)
        self.agent_settings_shadow.setBlurRadius(0)
        self.agent_settings_shadow.setColor(QColor(80, 160, 255, 180))
        self.agent_settings_shadow.setOffset(2, 2)
        self.frm_agent_settings.setGraphicsEffect(self.agent_settings_shadow)

        self.agent_settings_target_blur_radius = 10
        self.btn_agent_settings = self._find_child_widget(QPushButton, "btn_agent_settings")
        self.btn_agent_settings.clicked.connect(self.toggle_agent_settings)
        self._is_agent_settings_visible = False

        self.anim_agent_settings_position = QPropertyAnimation(self.frm_agent_settings, b"pos")
        self.anim_agent_settings_position.setDuration(200)
        self.anim_agent_settings_position.setEasingCurve(QEasingCurve.Type.OutBack)

        self.anim_agent_settings_shadow = QPropertyAnimation(
            self.agent_settings_shadow, b"blurRadius"
        )
        self.anim_agent_settings_shadow.setDuration(200)
        self.anim_agent_settings_shadow.setEasingCurve(QEasingCurve.Type.OutBack)

        self.btn_agent_load_report = self._find_child_widget(
            QPushButton, "btn_agent_load_report"
        )
        self.btn_agent_load_report.clicked.connect(self.load_agent_report)
        self.btn_report_close = self._find_child_widget(QPushButton, "btn_report_close")
        self.btn_report_close.clicked.connect(self.close_report_page)

        self.pg_report = self._find_child_widget(QWidget, "pg_report")
        self.txt_report_content = self._find_child_widget(QTextEdit, "txt_report_content")
        self.txt_report_content.setReadOnly(True)
        self.txt_report_content.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.txt_report_content.setLineWrapMode(QTextEdit.WidgetWidth)

        self.txt_report_content.setStyleSheet("""
            QScrollBar:vertical {
                background: transparent;
                width: 10px;
                margin: 0px;
                border: none;
                padding: 0px;
            }
            QScrollBar::handle:vertical {
                background: #3a4a68;
                min-height: 20px;
                border-radius: 3px;
                margin: 2px 0px 2px 4px; 
            }
            QScrollBar::handle:vertical:hover {
                background: #4a5a78;
                border-radius: 4px;
                margin: 2px 0px 2px 2px; 
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)

        self.btn_save_as = self._find_child_widget(QPushButton, "btn_save_as")
        if self.btn_save_as is not None:
            self.btn_save_as.setCursor(Qt.PointingHandCursor)
            self.btn_save_as.clicked.connect(self.show_save_as_menu)

        self.latest_report_text = ""
        self.current_report_md = ""

    # ------------------------------------------------------------------
    # Prediction page
    # ------------------------------------------------------------------
    def _init_prediction(self) -> None:
        """Wire the waterfall plot, the dual-SAM editor, and the run button."""
        self.gvw_pred_waterfall = self._find_child_widget(QGraphicsView, "gvw_pred_waterfall")
        self.gvw_pred_waterfall.setRenderHints(
            QPainter.Antialiasing | QPainter.SmoothPixmapTransform
        )
        self.waterfall_scene = QGraphicsScene()
        self.gvw_pred_waterfall.setScene(self.waterfall_scene)

        self.txt_pred_sam_1 = self._find_child_widget(QPlainTextEdit, "txt_pred_sam_1")
        self.txt_pred_sam_1.setPlainText(
            "C12N(C3C=CC(SC)=CC=3C=1C=C(C=C2)SC)C1C=CC(P(=O)(O)O)=CC=1"
        )
        self.txt_pred_sam_1.textChanged.connect(self.on_sam_smiles_changed)

        self.txt_pred_sam_2 = self._find_child_widget(QPlainTextEdit, "txt_pred_sam_2")
        self.txt_pred_sam_2.setPlainText(
            "C12N(C3C=CC(SC)=CC=3C=1C=C(C=C2)SC)C1C=CC(P(=O)(O)O)=CC=1"
        )
        self.txt_pred_sam_2.textChanged.connect(self.on_sam_smiles_changed)

        self.dsp_pred_bandgap = self._find_child_widget(QDoubleSpinBox, "dsp_pred_bandgap")
        self.cmb_pred_additive = self._find_child_widget(QComboBox, "cmb_pred_additive")
        self.dsp_pred_ratio = self._find_child_widget(QDoubleSpinBox, "dsp_pred_ratio")

        self.cmb_pred_shap_model = self._find_child_widget(QComboBox, "cmb_pred_shap_model")
        self.cmb_pred_shap_model.setEditable(True)
        self.cmb_pred_shap_model.lineEdit().setReadOnly(True)
        self.cmb_pred_shap_model.lineEdit().setPlaceholderText("Please select a model...")
        self.cmb_pred_shap_model.setCurrentIndex(-1)
        self.cmb_pred_shap_model.lineEdit().setStyleSheet("QLineEdit {padding-left: 5px;}")
        self.cmb_pred_shap_model.lineEdit().setFocusPolicy(Qt.NoFocus)
        self.cmb_pred_shap_model.installEventFilter(self)

        self.btn_pred_run = self._find_child_widget(QPushButton, "btn_pred_run")
        self.btn_pred_run.clicked.connect(self.start_prediction)
        self.btn_pred_run.setEnabled(True)
        self.btn_pred_run.setStyleSheet(self.START_BUTTON_QSS)
        self.prediction_stop_event = Event()

        self.lay_pred_mol_display = self._find_child_widget(QHBoxLayout, "lay_pred_mol_display")
        self.lay_pred_mol_display.setContentsMargins(0, 0, 0, 0)
        self.lay_pred_mol_display.setSpacing(0)
        self.canvas_sam_1: Optional[Molecule3DCanvas] = None
        self.canvas_sam_2: Optional[Molecule3DCanvas] = None

    # ==================================================================
    # Settings panels
    # ==================================================================

    def _toggle_setting_panel(
        self,
        position_animation: QPropertyAnimation,
        shadow_animation: QPropertyAnimation,
        visibility_flag_name: str,
        target_position: QPoint,
        hidden_position: QPoint,
        target_blur_radius: int,
    ) -> None:
        """Animate a settings panel between its hidden and visible poses."""
        if position_animation.state() == QPropertyAnimation.Running:
            return

        show = not getattr(self, visibility_flag_name)

        position_animation.setStartValue(hidden_position if show else target_position)
        position_animation.setEndValue(target_position if show else hidden_position)
        shadow_animation.setStartValue(0 if show else target_blur_radius)
        shadow_animation.setEndValue(target_blur_radius if show else 0)

        position_animation.start()
        shadow_animation.start()
        setattr(self, visibility_flag_name, show)

    def toggle_mol_calc_settings(self) -> None:
        self._toggle_setting_panel(
            self.anim_mol_calc_settings_position,
            self.anim_mol_calc_settings_shadow,
            "_is_mol_calc_settings_visible",
            self.mol_calc_settings_target_position,
            self.mol_calc_settings_hidden_position,
            self.mol_calc_settings_target_blur_radius,
        )

    def toggle_agent_settings(self) -> None:
        self._toggle_setting_panel(
            self.anim_agent_settings_position,
            self.anim_agent_settings_shadow,
            "_is_agent_settings_visible",
            self.agent_settings_target_position,
            self.agent_settings_hidden_position,
            self.agent_settings_target_blur_radius,
        )

    # ==================================================================
    # Chat transcript insertion
    # ==================================================================
    # Every row wraps its widget in a QHBoxLayout; the side the stretch is on
    # determines the alignment.

    def _add_aligned_widget(self, widget: QWidget, align: str) -> None:
        row = QHBoxLayout()
        if align == "right":
            row.addStretch(1)
            row.addWidget(widget)
        else:
            row.addWidget(widget, 1)
            row.addStretch(0)
        self.messages_layout.insertLayout(self.messages_layout.count() - 1, row)

    def add_user_message(self, text: str) -> None:
        """Append a right-aligned user bubble and scroll it into view."""
        bubble = MessageBubble(
            text,
            align="right",
            kind="user",
            contents_margins=(12, 12, 6, 6),
        )
        bubble.setBubbleColor("#4268a8")
        self._add_aligned_widget(bubble, "right")
        self.scroll_to_bottom()

    def add_reply_message(self, text: str = "") -> MessageBubble:
        """Append a left-aligned AI reply bubble and return its handle."""
        bubble = MessageBubble(
            text,
            align="left",
            kind="ai",
            wrap=False,
            contents_margins=(42, 8, 12, 8),
        )
        bubble.heightChanged.connect(self.on_chat_content_height_changed)
        self._add_aligned_widget(bubble, "left")
        self.scroll_to_bottom()
        return bubble

    def add_think_box(self, text: str = "") -> ThinkBox:
        """Append a left-aligned reasoning box and begin streaming."""
        box = ThinkBox(text=text)
        box.begin_stream(auto_expand=True)
        box.heightChanged.connect(self.on_chat_content_height_changed)
        self._add_aligned_widget(box, "left")
        self.scroll_to_bottom()
        return box

    # ==================================================================
    # Chat scrolling
    # ==================================================================
    # Sticky-bottom policy: follow the newest content as long as the user has
    # not scrolled away.

    def on_chat_scroll_range_changed(self, minimum: int, maximum: int) -> None:
        if getattr(self, "_auto_scroll", True):
            self._programmatic_scroll = True
            self.scl_agent_chat.verticalScrollBar().setValue(maximum)
            self._programmatic_scroll = False
        self._sticky_timer.start()

    def on_chat_scrolled(self, value: int) -> None:
        # 50 px tolerance absorbs sub-pixel rounding and new-bubble measurement.
        if getattr(self, "_programmatic_scroll", False):
            return
        scroll_bar = self.scl_agent_chat.verticalScrollBar()
        self._auto_scroll = (scroll_bar.maximum() - value) < 50
        self._sticky_timer.start()

    def scroll_to_bottom(self) -> None:
        """Schedule a coalesced scroll-to-bottom on the next event-loop turn."""
        self._auto_scroll = True
        if not self._scroll_timer.isActive():
            self._scroll_timer.start()

    def _do_scroll_to_bottom(self) -> None:
        if not getattr(self, "_auto_scroll", True):
            return
        scroll_bar = self.scl_agent_chat.verticalScrollBar()
        self._programmatic_scroll = True
        scroll_bar.setValue(scroll_bar.maximum())
        self._programmatic_scroll = False

    def on_chat_content_height_changed(self) -> None:
        """React to a bubble or think-box growing during streaming."""
        self._sticky_timer.start()
        if getattr(self, "_auto_scroll", True):
            if not self._scroll_timer.isActive():
                self._scroll_timer.start()

    # ==================================================================
    # Sticky header
    # ==================================================================

    def _update_sticky_header(self) -> None:
        """Re-evaluate which ThinkBox, if any, should own the sticky header."""
        if not hasattr(self, "sticky_header_btn") or self.sticky_header_btn is None:
            return

        scroll_value = self.scl_agent_chat.verticalScrollBar().value()
        active_box: Optional[ThinkBox] = None

        # At most one expanded ThinkBox can have its header off-screen but its
        # body still partially visible, because boxes do not overlap.
        for i in range(self.messages_layout.count()):
            item = self.messages_layout.itemAt(i)
            if item is None or item.layout() is None:
                continue
            sub_layout = item.layout()
            if sub_layout.count() == 0:
                continue
            widget = sub_layout.itemAt(0).widget()
            if not isinstance(widget, ThinkBox):
                continue
            if not (widget.isVisible() and widget.header.isChecked()):
                continue

            position = widget.mapTo(self.chat_container, QPoint(0, 0))
            top = position.y()
            height = widget.height()
            if top < scroll_value < (top + height):
                active_box = widget
                break

        if active_box is None:
            self.sticky_header_btn.hide()
            return

        self.sticky_header_btn.setText(active_box.header.text())
        self.sticky_header_btn.setIcon(active_box.header.icon())
        self.sticky_header_btn.show()
        self.sticky_header_btn.raise_()

        # Match the source header's x/width; the 18 px y-offset matches the
        # chat container's top margin so the copy appears to emerge from the
        # transcript rather than float above it.
        header_position = active_box.header.mapTo(
            self.scl_agent_chat.viewport(), QPoint(0, 0)
        )
        self.sticky_header_btn.setGeometry(
            header_position.x(),
            17,
            active_box.header.width(),
            active_box.header.height(),
        )
        self.sticky_header_btn._active_box = active_box

    def on_sticky_header_clicked(self) -> None:
        """Collapse the ThinkBox that currently owns the sticky header."""
        box = getattr(self.sticky_header_btn, "_active_box", None)
        if box is None:
            return

        # Capture the pre-collapse y; the height will change during collapse.
        position = box.mapTo(self.chat_container, QPoint(0, 0))
        box.collapse()
        self.scl_agent_chat.verticalScrollBar().setValue(position.y())
        self.sticky_header_btn.hide()

    # ==================================================================
    # Report sub-page
    # ==================================================================

    def show_save_as_menu(self) -> None:
        """Display a dropdown menu to choose the report saving format."""
        if self.btn_save_as is None:
            return

        menu = QMenu(self.btn_save_as)
        menu.setStyleSheet(
            "QMenu { background-color: #1e2a44; color: #e8eef8; border: 1px solid #3a4a68; "
            "border-radius: 4px; font-family: 'Consolas'; font-size: 13px; }"
            "QMenu::item { padding: 6px 24px 6px 24px; }"
            "QMenu::item:selected { background-color: #3b8bca; color: #ffffff; }"
        )

        report_text = self.current_report_md
        if not report_text.strip():
            report_text = self.txt_report_content.toPlainText()

        action_md = menu.addAction("Save as Markdown (.md)")
        action_txt = menu.addAction("Save as Text (.txt)")
        menu.addSeparator()
        action_img = menu.addAction("Save as Image (.png)")

        action_md.triggered.connect(lambda: self.save_agent_report(report_text, forced_format="md"))
        action_txt.triggered.connect(lambda: self.save_agent_report(report_text, forced_format="txt"))
        action_img.triggered.connect(lambda: self.save_report_as_image(report_text))

        menu.exec(self.btn_save_as.mapToGlobal(QPoint(0, self.btn_save_as.height())))

    def save_agent_report(self, text_to_save: Optional[str] = None, forced_format: Optional[str] = None) -> None:
        """Save the last MSEC Analysis Report to a Markdown or plain-text file."""
        if not isinstance(text_to_save, str):
            text_to_save = self.latest_report_text

        target_text = text_to_save if text_to_save.strip() else self.latest_report_text
        if not target_text.strip():
            self.txt_status_message.setPlainText("No report to save.")
            self.set_state("warning")
            return

        file_filter = "Markdown Files (*.md);;Text Files (*.txt);;All Files (*)"
        default_name = "MSEC_Report.md"

        if forced_format == "md":
            file_filter = "Markdown Files (*.md)"
            default_name = "MSEC_Report.md"
        elif forced_format == "txt":
            file_filter = "Text Files (*.txt)"
            default_name = "MSEC_Report.txt"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Report",
            os.path.join(self.get_base_dir(), default_name),
            file_filter,
        )
        if not file_path:
            return

        if forced_format and not file_path.lower().endswith(f".{forced_format}"):
            file_path += f".{forced_format}"

        try:
            with open(file_path, "w", encoding="utf-8") as fh:
                fh.write(target_text)
            self.txt_status_message.setPlainText(f"Report saved to {file_path}")
            self.set_state("normal")
        except Exception as exc:
            self.txt_status_message.setPlainText(f"Failed to save report: {exc}")
            self.set_state("error")

    def load_agent_report(self) -> None:
        """Load a previously saved report and render it in the report page."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Report",
            self.get_base_dir(),
            "Markdown Files (*.md);;Text Files (*.txt);;All Files (*)",
        )
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                content = fh.read()

            self.current_report_md = content
            styled_html = format_markdown_to_styled_html(content)
            self.txt_report_content.setHtml(styled_html)

            self.stk_main.setCurrentWidget(self.pg_report)
            self.txt_status_message.setPlainText(f"Report loaded from {file_path}")
            self.set_state("normal")
        except Exception as exc:
            self.txt_status_message.setPlainText(f"Failed to load report: {exc}")
            self.set_state("error")

    def save_report_as_image(self, text_to_save: Optional[str] = None) -> None:
        """Render the MSEC Analysis Report to a full-length image and save it."""
        if not isinstance(text_to_save, str):
            text_to_save = self.current_report_md
            if not text_to_save.strip():
                text_to_save = self.txt_report_content.toPlainText()

        if not text_to_save.strip():
            self.txt_status_message.setPlainText("No report content to save.")
            self.set_state("warning")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Report as Image",
            os.path.join(self.get_base_dir(), "MSEC_Report.png"),
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)",
        )
        if not file_path:
            return

        if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path += '.png'

        try:
            if text_to_save.strip() == self.txt_report_content.toPlainText().strip():
                doc = self.txt_report_content.document().clone()
            else:
                styled_html = format_markdown_to_styled_html(text_to_save)
                doc = QTextDocument()
                doc.setHtml(styled_html)

            current_width = self.txt_report_content.viewport().width()
            if current_width <= 0:
                current_width = 800
            doc.setTextWidth(current_width)

            padding = 20
            width = int(doc.size().width()) + padding * 2
            height = int(doc.size().height()) + padding * 2

            image = QImage(width, height, QImage.Format_ARGB32)
            image.fill(QColor("#1e2a44"))

            painter = QPainter(image)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.TextAntialiasing)

            painter.translate(padding, padding)
            doc.drawContents(painter)
            painter.end()

            if image.save(file_path):
                self.txt_status_message.setPlainText(f"Report image saved to {file_path}")
                self.set_state("normal")
            else:
                self.txt_status_message.setPlainText("Failed to save report image.")
                self.set_state("error")

        except Exception as exc:
            self.txt_status_message.setPlainText(f"Failed to save image: {exc}")
            self.set_state("error")

    def close_report_page(self) -> None:
        self.stk_main.setCurrentWidget(self.pg_agent)

    # ==================================================================
    # Agent lifecycle
    # ==================================================================
    # The agent lives in a separate process because the LLM client holds an
    # HTTP connection and performs blocking reads.  Out-of-process also
    # contains crashes if the model server disconnects.
    #
    # Two queues:
    #   agent_command_queue    GUI -> child    (init / conv)
    #   agent_message_queue    child -> GUI    (think / reply / error / progress)

    def clear_agent_chat(self) -> None:
        """Stop any running agent, reset IPC, and wipe the transcript."""
        self.btn_agent_send.setEnabled(False)
        self.agent_stop_event.set()

        if self.agent_process is not None:
            try:
                if self.agent_process.is_alive():
                    self.agent_process.join(timeout=1.0)
                    if self.agent_process.is_alive():
                        self.agent_process.terminate()
                        self.agent_process.join(timeout=1.0)
            except Exception:
                pass
            finally:
                self.agent_process = None
                self.txt_status_message.setPlainText("Ready.")
                self.set_state("normal")

        # Replace the queues: the old ones remain valid references for the
        # dying child, but no new messages can be read from them.
        self.agent_message_queue = Queue()
        self.agent_command_queue = Queue()

        if isinstance(self.current_think_box, ThinkBox):
            try:
                self.current_think_box.end_stream(auto_collapse=False)
            except Exception:
                pass
        self.current_think_box = None
        self.current_reply_bubble = None

        # Clear every row except the base layout's trailing stretch.
        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            if item is None:
                break
            if item.layout() is not None:
                sub_layout = item.layout()
                while sub_layout.count():
                    sub_item = sub_layout.takeAt(0)
                    widget = sub_item.widget()
                    if widget is not None:
                        widget.setParent(None)
                        widget.deleteLater()
            elif item.widget() is not None:
                widget = item.widget()
                widget.setParent(None)
                widget.deleteLater()

        self.txt_agent_input.setPlaceholderText(
            "Please complete the initial analysis first..."
        )
        self.txt_agent_input.setEnabled(False)
        self.has_completed_initial_analysis = False

        self._set_task_idle(
            self.btn_agent_send,
            self.send_agent_message,
            idle_text=None,
            idle_qss=self.AGENT_SEND_BUTTON_QSS,
            icon=QIcon(":/icons/send.svg"),
            icon_size=QSize(28, 28),
        )

        self.agent_timer.stop()
        self.scroll_to_bottom()
        self.sticky_header_btn.hide()
        self.pbar_main.setValue(0)
        self.agent_stop_event.clear()

    def stop_agent(self) -> None:
        """Stop the running agent, finalizing any open bubbles. This preserves
        the transcript so the user can inspect the partial result.
        """
        self.agent_stop_event.set()
        self.is_agent_running = False

        if isinstance(self.current_think_box, ThinkBox):
            try:
                self.current_think_box.end_stream(auto_collapse=True)
            except Exception:
                pass
        self.current_think_box = None
        self.current_reply_bubble = None

        if self.agent_process is not None:
            try:
                if self.agent_process.is_alive():
                    self.agent_process.terminate()
                    self.agent_process.join(timeout=1.0)
            except Exception:
                pass
            self.agent_process = None

        self.agent_timer.stop()

        self._set_task_idle(
            self.btn_agent_send,
            self.send_agent_message,
            idle_text=None,
            idle_qss=self.AGENT_SEND_BUTTON_QSS,
            icon=QIcon(":/icons/send"),
            icon_size=QSize(28, 28),
        )

    @staticmethod
    def _drain_queue(message_queue: "Queue") -> None:
        """Consume every item in 'message_queue'; used during teardown."""
        while True:
            try:
                message_queue.get_nowait()
            except queue.Empty:
                break

    def create_agent_process(
        self,
        command_queue: "Queue",
        message_queue: "Queue",
        stop_event: Event,
        api_key: str,
        base_url: str,
        llm_model: str,
        temperature: float,
        max_tokens: int,
        threshold: float,
        descriptor_top_n: int,
        thinking: bool,
        base_dir: str,
    ) -> None:
        """Spawn the MSECAgent child process if it is not already running."""
        if self.agent_process and self.agent_process.is_alive():
            return

        agent_process = Process(
            target=agent_worker,
            args=(
                command_queue,
                message_queue,
                stop_event,
                api_key,
                base_url,
                llm_model,
                temperature,
                max_tokens,
                threshold,
                descriptor_top_n,
                thinking,
                base_dir,
            ),
        )
        agent_process.start()
        self.agent_process = agent_process

    def send_agent_message(self) -> None:
        """Start the initial analysis, or send a follow-up message."""
        # Drain pending messages first so the state acted on is current.
        for _ in range(50):
            if self.agent_message_queue.empty():
                break
            self.on_agent_queue_tick()

        if self.is_agent_running:
            self.txt_status_message.setPlainText(
                "Agent is running. Please wait or click STOP first."
            )
            self.set_state("warning")
            return

        self.is_agent_reply_finished = False
        self.agent_stop_event.clear()
        self.current_think_box = None
        self.current_reply_bubble = None
        self.latest_report_text = ""

        api_key = self.txt_agent_api_key.toPlainText().strip()
        base_url = self.txt_agent_base_url.toPlainText().strip()
        llm_model = self.txt_agent_model_name.toPlainText().strip()
        temperature = self.dsp_agent_temperature.value()
        max_tokens = self.spn_agent_max_tokens.value()
        threshold = self.dsp_agent_threshold.value()
        descriptor_top_n = self.spn_agent_descriptor_top_k.value()
        thinking = self.chk_agent_thinking.isChecked()
        models_path = os.path.join(self.get_base_dir(), "ml_models")
        train_size = self.dsp_agent_split.value()
        random_state = self.spn_agent_seed.value()
        agent_model_sel = self.cmb_agent_shap_model.currentText()
        base_dir = self.temp_dir

        if agent_model_sel.lower() == "optimal" and self.ml_optimal_model is not None:
            agent_model_sel = self.ml_optimal_model
        fitting_model = self.btn_group_agent_fitting_type.checkedButton().text().lower()
        substructure_top_k = self.spn_agent_substructure_top_k.value()

        str_ok = bool(
            api_key.strip()
            and base_url.strip()
            and llm_model.strip()
            and agent_model_sel.strip()
            and fitting_model.strip()
        )
        num_ok = bool(
            temperature is not None
            and max_tokens is not None
            and threshold is not None
            and descriptor_top_n is not None
            and train_size is not None
            and random_state is not None
            and substructure_top_k is not None
        )
        if not (str_ok and num_ok):
            self.txt_status_message.setPlainText("Please complete the setup first!")
            self.set_state("warning")
            return

        if self.acin_substructure_weights is None:
            if self.load_acin_substructure_weights() is None:
                return

        self.txt_status_message.setPlainText("Agent is running...")
        self.set_state("normal")
        self.is_agent_running = True
        self.pbar_main.setValue(0)
        self.agent_timer.stop()
        self.agent_timer.setInterval(300)
        self.agent_timer.start()

        if self.agent_process is None or not self.agent_process.is_alive():
            self.agent_process = None
            self.agent_message_queue = Queue()
            self.agent_command_queue = Queue()

            # Reconnect the scroll slot so a previous session cannot leak a
            # duplicate connection onto the new scrollbar.
            scroll_bar = self.scl_agent_chat.verticalScrollBar()
            try:
                scroll_bar.valueChanged.disconnect(self.on_chat_scrolled)
            except Exception:
                pass
            scroll_bar.valueChanged.connect(self.on_chat_scrolled)

            self.agent_thread_obj = WorkerThread(
                lambda: self.create_agent_process(
                    self.agent_command_queue,
                    self.agent_message_queue,
                    self.agent_stop_event,
                    api_key,
                    base_url,
                    llm_model,
                    temperature,
                    max_tokens,
                    threshold,
                    descriptor_top_n,
                    thinking,
                    base_dir,
                )
            )
            self.agent_thread_obj.finished.connect(self.agent_thread_obj.deleteLater)
            self.agent_thread_obj.start()

        self._set_task_running(
            self.btn_agent_send,
            self.stop_agent,
            running_text=None,
            running_qss=self.AGENT_STOP_BUTTON_QSS,
            icon=QIcon(":/icons/stop.svg"),
            icon_size=QSize(18, 18),

        )

        if not self.has_completed_initial_analysis:
            self.agent_command_queue.put(
                [
                    "init",
                    [
                        models_path,
                        train_size,
                        random_state,
                        agent_model_sel,
                        fitting_model,
                        substructure_top_k,
                        self.acin_substructure_weights,
                        thinking,
                    ],
                ]
            )
        else:
            text = self.txt_agent_input.toPlainText().strip()
            if not text:
                return
            self.agent_command_queue.put(["conv", [text, thinking]])
            self.txt_agent_input.clear()
            self.add_user_message(text)

    def on_agent_queue_tick(self) -> None:
        """Drain up to 'MAX_MSGS' messages from the agent message queue.

        Bounded draining prevents a runaway producer from monopolising the
        event loop; if the queue still has items the slot re-schedules itself.
        """
        MAX_MSGS = 60
        changed = False
        processed = 0
        queue_drained = False

        while processed < MAX_MSGS:
            try:
                message = str(self.agent_message_queue.get_nowait())
            except queue.Empty:
                queue_drained = True
                break
            processed += 1

            lowered = message.lower()
            if lowered.startswith(self.MSG_PREFIX_THINK):
                if self.agent_stop_event.is_set():
                    continue
                if self.current_think_box is None:
                    # Invariant: at most one of current_think_box /
                    # current_reply_bubble is non-None.
                    self.current_reply_bubble = None
                    self.current_think_box = self.add_think_box("")
                self.current_think_box.append(message[len(self.MSG_PREFIX_THINK):])
                changed = True

            elif lowered.startswith(self.MSG_PREFIX_REPLY):
                if self.agent_stop_event.is_set():
                    continue
                if self.current_reply_bubble is None:
                    if isinstance(self.current_think_box, ThinkBox):
                        try:
                            self.current_think_box.end_stream(auto_collapse=True)
                        except Exception:
                            pass
                    self.current_think_box = None
                    self.current_reply_bubble = self.add_reply_message("")
                    self.latest_report_text = ""
                chunk = message[len(self.MSG_PREFIX_REPLY):]
                self.current_reply_bubble.append(chunk)
                self.latest_report_text += chunk
                changed = True

            elif lowered.startswith(self.MSG_PREFIX_INIT):
                self.has_completed_initial_analysis = True
                self.is_agent_reply_finished = False
                self.txt_agent_input.setPlaceholderText(
                    "Input message, press Enter to send, Shift+Enter for line break..."
                )
                self.txt_agent_input.setEnabled(True)

            elif lowered.startswith(self.MSG_PREFIX_END):
                self.agent_stop_event.clear()
                self.is_agent_reply_finished = True

            elif lowered.startswith(self.MSG_PREFIX_ERROR):
                self.txt_status_message.setPlainText(
                    message[len(self.MSG_PREFIX_ERROR):]
                )
                self.set_state("error")

            elif lowered.startswith(self.MSG_PREFIX_PROGRESS):
                try:
                    self.pbar_main.setValue(
                        float(message[len(self.MSG_PREFIX_PROGRESS):]) * 100
                    )
                except ValueError:
                    pass

            else:
                self.txt_status_message.setPlainText(f"Unknown message: {message}")
                self.set_state("unknown")

        if changed and getattr(self, "_auto_scroll", True):
            self.scroll_to_bottom()

        if processed == MAX_MSGS:
            QTimer.singleShot(0, self.on_agent_queue_tick)
            return

        if self.is_agent_reply_finished and queue_drained:
            self.is_agent_reply_finished = False

            if isinstance(self.current_think_box, ThinkBox):
                self.current_think_box.finalize()
            if isinstance(self.current_reply_bubble, MessageBubble):
                self.current_reply_bubble.finalize()
                reply_text = self.current_reply_bubble.text()
                if "# MSEC Analysis Report" in reply_text:
                    self.current_report_md = reply_text
                    self.current_reply_bubble.add_bottom_button(
                        "Save Report",
                        lambda *args, t=reply_text: self.save_agent_report(t),
                    )

            self.current_think_box = None
            self.current_reply_bubble = None

            self._set_task_idle(
                self.btn_agent_send,
                self.send_agent_message,
                idle_text=None,
                idle_qss=self.AGENT_SEND_BUTTON_QSS,
                icon=QIcon(":/icons/send"),
                icon_size=QSize(28, 28),
            )

            self.agent_timer.stop()
            self.is_agent_running = False

    # ==================================================================
    # Status indicator and state log
    # ==================================================================

    def set_state(self, state: str) -> None:
        """Show exactly one of the four status labels and log the transition."""
        state_map = {
            "normal": self.lbl_status_normal,
            "error": self.lbl_status_error,
            "warning": self.lbl_status_warning,
            "unknown": self.lbl_status_unknown,
        }
        for label in state_map.values():
            if label is not None:
                label.setVisible(False)
        target = state_map.get(state)
        if target is not None:
            target.setVisible(True)
        self.append_state_log(state)

    def show_state_log(self) -> None:
        current = self.stk_main.currentWidget()
        if current is not None and current is not self.pg_state_log:
            self._previous_page = current
        self.stk_main.setCurrentWidget(self.pg_state_log)

    def close_state_log(self) -> None:
        target = getattr(self, "_previous_page", None)
        if target is None:
            target = self.pg_home
        self.stk_main.setCurrentWidget(target)

    def append_state_log(self, state: str) -> None:
        """Append one timestamped, colour-coded line to the state log.

        A line is only appended when the (state, message) pair differs from
        the previous append; this keeps the log readable during long streaming
        operations where 'set_state' is invoked many times.
        """
        log_edit = getattr(self, "txt_state_log", None)
        if log_edit is None:
            return

        message = ""
        if getattr(self, "txt_status_message", None) is not None:
            message = self.txt_status_message.toPlainText().strip()
        if not message:
            message = "(no message)"

        if (
            state == getattr(self, "_last_logged_state", None)
            and message == getattr(self, "_last_logged_message", None)
        ):
            return
        self._last_logged_state = state
        self._last_logged_message = message

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        color_map = {
            "normal": "#98c379",
            "error": "#e06c75",
            "warning": "#e5c07b",
            "unknown": "#56b6c2",
        }
        name_map = {
            "normal": "INFO",
            "error": "ERROR",
            "warning": "WARN",
            "unknown": "UNKNOWN",
        }
        color = color_map.get(state, "#dcdfe4")
        label = name_map.get(state, state.upper())

        safe_message = (
            message.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        html = (
            f'<span style="color:#5c6370;">[{timestamp}]</span> '
            f'<span style="color:{color}; font-weight:bold;">[{label}]</span> '
            f'<span style="color:#dcdfe4;">{safe_message}</span>'
        )

        cursor = log_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not log_edit.document().isEmpty():
            cursor.insertBlock()
        cursor.insertHtml(html)
        log_edit.setTextCursor(cursor)
        log_edit.verticalScrollBar().setValue(log_edit.verticalScrollBar().maximum())

    def update_system_usage(self) -> None:
        """Refresh the CPU and (if available) GPU utilisation meters."""
        cpu_usage = psutil.cpu_percent(interval=None)
        self.pbar_cpu.setValue(cpu_usage)

        gpus = GPUtil.getGPUs()
        if gpus:
            total_load = sum(gpu.load for gpu in gpus)
            gpu_usage = total_load / len(gpus) * 100
            self.pbar_gpu.setVisible(True)
            self.pbar_gpu.setValue(gpu_usage)
        else:
            self.pbar_gpu.setValue(0)
            self.pbar_gpu.setVisible(False)

    # ==================================================================
    # Button state machine
    # ==================================================================
    # IDLE -> LOADING -> RUNNING -> IDLE.  '_set_pair_*' apply the three
    # transitions atomically to a (start, stop) pair so UI state can never
    # become inconsistent.

    def _set_button_state(
        self,
        button: QPushButton,
        *,
        text: Optional[str] = None,
        callback=None,
        qss: Optional[str] = None,
        icon: Optional[QIcon] = None,
        icon_size: Optional[QSize] = None,
        enabled: bool = True,
    ) -> None:
        """Reconfigure a push button in one atomic step.

        The previous callback is disconnected before the new one is connected
        because some buttons are reused as both START and STOP; otherwise the
        'clicked' signal would accumulate handlers.
        """
        if callback is not None:
            try:
                button.clicked.disconnect()
            except Exception:
                pass
            button.clicked.connect(callback)
        if text is not None:
            button.setText(text)
        if qss is not None:
            button.setStyleSheet(qss)
        if icon is not None:
            button.setIcon(icon)
            if icon_size is not None:
                button.setIconSize(icon_size)
        button.setEnabled(enabled)

    def _set_task_running(
        self,
        button: QPushButton,
        stop_callback,
        *,
        running_text: Optional[str] = "STOP",
        running_qss: Optional[str] = None,
        icon: Optional[QIcon] = None,
        icon_size: Optional[QSize] = None,
    ) -> None:
        self._set_button_state(
            button,
            text=running_text,
            callback=stop_callback,
            qss=running_qss or self.STOP_BUTTON_QSS,
            icon=icon,
            icon_size=icon_size,
        )

    def _set_task_idle(
        self,
        button: QPushButton,
        start_callback,
        *,
        idle_text: Optional[str] = "START",
        idle_qss: Optional[str] = None,
        icon: Optional[QIcon] = None,
        icon_size: Optional[QSize] = None,
    ) -> None:
        self._set_button_state(
            button,
            text=idle_text,
            callback=start_callback,
            qss=idle_qss or self.START_BUTTON_QSS,
            icon=icon,
            icon_size=icon_size,
        )

    def _set_pair_running(self, start_button: QPushButton, stop_button: QPushButton) -> None:
        start_button.setEnabled(False)
        start_button.setText("Running")
        stop_button.setEnabled(True)
        stop_button.setText("STOP")

    def _set_pair_loading(self, start_button: QPushButton, stop_button: QPushButton) -> None:
        """Both buttons disabled while the child is being spawned."""
        start_button.setEnabled(False)
        start_button.setText("Running")
        stop_button.setEnabled(False)
        stop_button.setText("STOP")

    def _set_pair_idle(self, start_button: QPushButton, stop_button: QPushButton) -> None:
        start_button.setEnabled(True)
        start_button.setText("START")
        stop_button.setEnabled(False)
        stop_button.setText("STOP")

    def check_ml_inputs(self, action: str = "start") -> bool:
        """Verify that 'ml_inputs.csv' exists, prompting the user if not."""
        path = os.path.join(self.get_base_dir(), "ml_inputs.csv")
        if os.path.isfile(path):
            return True

        self._show_info_dialog(
            "Merged dataset required",
            "<div style=\"font-family: 'Consolas'; font-size: 13px; "
            "color: #b8c2d6; line-height: 1.5;\">"
            "<b style='color:#e8eef8;'>ml_inputs.csv</b> was not found in "
            "the runtime folder.<br>"
            f"Please open the <i>Data</i> page, select the datasets and click "
            f"<b style='color:#e8eef8;'>MERGE</b> before you {action}."
            "</div>",
            buttons=("Ok",),
            min_width=480,
        )
        self.txt_status_message.setPlainText(
            "ml_inputs.csv not found — please go to the Data page and click MERGE first."
        )
        self.set_state("warning")
        return False

    # ==================================================================
    # SMILES validation and metadata dialogs
    # ==================================================================

    @staticmethod
    def is_valid_smiles(smiles: Optional[str]) -> tuple[bool, str]:
        """Return '(True, "")' for a valid SMILES, else '(False, reason)'."""
        if smiles is None:
            return False, "SMILES is none."
        cleaned = "".join(smiles.split())
        if not cleaned:
            return False, "SMILES is none."
        mol = Chem.MolFromSmiles(cleaned)
        if mol is None:
            return False, "SMILES is illegal."
        if mol.GetNumAtoms() == 0:
            return False, "SMILES contains no atoms"
        return True, ""

    def validate_mol_smiles(self) -> None:
        """Re-validate the editor, updating counts and highlighting bad lines."""
        valid: list[str] = []
        invalid: list[tuple[int, str, str]] = []
        seen: set[str] = set()

        for lineno, line in enumerate(self.txt_mol_smiles.toPlainText().splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            ok, err = self.is_valid_smiles(stripped)
            if not ok:
                invalid.append((lineno, stripped, err))
            elif stripped not in seen:
                seen.add(stripped)
                valid.append(stripped)

        self.valid_smiles_list = valid
        self.invalid_smiles_lines = invalid

        self.lbl_mol_counts.setText(
            f"<span style='color:#98c379;'>Valid - {len(valid)}</span> "
            f" | "
            f"<span style='color:#e06c75;'>Invalid - {len(invalid)}</span>"
        )

        document = self.txt_mol_smiles.document()
        selections: list[QTextEdit.ExtraSelection] = []
        for lineno, _, _ in invalid:
            block = document.findBlockByNumber(lineno - 1)
            if not block.isValid():
                continue
            sel = QTextEdit.ExtraSelection()
            sel.format.setBackground(QColor(224, 108, 117, 60))
            sel.format.setProperty(QTextFormat.FullWidthSelection, True)
            sel.cursor = QTextCursor(block)
            selections.append(sel)
        try:
            self.txt_mol_smiles.setExtraSelections(selections)
        except AttributeError:
            pass

        if invalid:
            self.txt_status_message.setPlainText(f"Detected {len(invalid)} invalid molecules.")
            self.set_state("warning")
        elif valid:
            self.txt_status_message.setPlainText(f"Identified {len(valid)} valid molecules.")
            self.set_state("normal")

    def import_mol_smiles_csv(self) -> None:
        """Load a SMILES column from a CSV file into the editor."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose SMILES CSV", self.get_base_dir(), "CSV (*.csv)"
        )
        if not path:
            return
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            self.txt_status_message.setPlainText(f"Read failed: {exc}")
            self.set_state("error")
            return

        candidates = ("SMILES", "smiles", "Smiles", "canonical_smiles")
        column = next(
            (c for c in candidates if c in df.columns), df.columns[0]
        )
        self.txt_mol_smiles.setPlainText(
            "\n".join(str(x).strip() for x in df[column].dropna())
        )

    def export_mol_smiles_csv(self) -> None:
        """Append the current valid SMILES to SMILES.csv with user metadata."""
        if not self.valid_smiles_list:
            self.txt_status_message.setPlainText("No valid molecules to export.")
            self.set_state("warning")
            return

        n = len(self.valid_smiles_list)
        dialog = QDialog(self)
        dialog.setWindowTitle("Metadata for Export")
        dialog.resize(820, 480)
        dialog.setStyleSheet(self.DIALOG_QSS)
        layout = QVBoxLayout(dialog)
        table = self._build_metadata_table(self.valid_smiles_list, dialog)
        layout.addWidget(table)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return

        sym_values, chain_values, het_values = self._read_metadata_table(table, n)
        new_rows = pd.DataFrame({
            "SMILES": self.valid_smiles_list,
            "Symmetric": sym_values,
            "Chain_length": chain_values,
            "HeteroEndFrag": het_values,
        })
        csv_path = os.path.join(self.get_base_dir(), "SMILES.csv")
        columns = ["SMILES", "Symmetric", "Chain_length", "HeteroEndFrag"]
        if os.path.isfile(csv_path):
            try:
                old = pd.read_csv(csv_path)
                for col in columns:
                    if col not in old.columns:
                        old[col] = None
                merged = pd.concat([old[columns], new_rows], ignore_index=True)
            except Exception:
                merged = new_rows
        else:
            merged = new_rows
        merged.to_csv(csv_path, index=False)
        self.txt_status_message.setPlainText(f"Appended {n} molecules to {csv_path}")
        self.set_state("normal")

    def _build_metadata_table(self, smiles_list: list[str], parent: Optional[QWidget] = None) -> QTableWidget:
        """Build the editable SMILES/Symmetric/Chain_length/HeteroEndFrag table."""
        n = len(smiles_list)
        table = QTableWidget(n, 4, parent)
        table.setHorizontalHeaderLabels(
            ["SMILES", "Symmetric", "Chain_length", "HeteroEndFrag"]
        )
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in (1, 2, 3):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)

        for row, smiles in enumerate(smiles_list):
            item = QTableWidgetItem(smiles)
            item.setFlags(Qt.ItemIsEnabled)
            table.setItem(row, 0, item)

            sym_combo = QComboBox()
            sym_combo.addItems(["1", "0"])
            sym_combo.setCurrentIndex(0)
            table.setCellWidget(row, 1, sym_combo)

            chain_spin = QSpinBox()
            chain_spin.setRange(1, 20)
            chain_spin.setValue(1)
            table.setCellWidget(row, 2, chain_spin)

            het_combo = QComboBox()
            het_combo.addItems(["0", "1"])
            het_combo.setCurrentIndex(0)
            table.setCellWidget(row, 3, het_combo)
        return table

    def _read_metadata_table(
        self, table: QTableWidget, row_count: int
    ) -> tuple[list[int], list[int], list[int]]:
        sym_values, chain_values, het_values = [], [], []
        for row in range(row_count):
            sym_values.append(int(table.cellWidget(row, 1).currentText()))
            chain_values.append(table.cellWidget(row, 2).value())
            het_values.append(int(table.cellWidget(row, 3).currentText()))
        return sym_values, chain_values, het_values

    def _load_smiles_metadata(self) -> dict[str, dict]:
        """Read SMILES.csv, returning '{smiles: {Symmetric, Chain_length, HeteroEndFrag}}'."""
        base = self.get_base_dir()
        for name in ("SMILES.csv", "smiles.csv", "Smiles.csv"):
            csv_path = os.path.join(base, name)
            if os.path.isfile(csv_path):
                break
        else:
            return {}

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return {}
        if "SMILES" not in df.columns:
            return {}

        metadata: dict[str, dict] = {}
        for _, row in df.iterrows():
            smiles = str(row.get("SMILES", "")).strip()
            if not smiles:
                continue
            try:
                sym = int(row["Symmetric"]) if pd.notna(row.get("Symmetric")) else 0
            except Exception:
                sym = 0
            try:
                chain = int(row["Chain_length"]) if pd.notna(row.get("Chain_length")) else 1
            except Exception:
                chain = 1
            try:
                het = int(row["HeteroEndFrag"]) if pd.notna(row.get("HeteroEndFrag")) else 0
            except Exception:
                het = 0
            metadata[smiles] = {
                "Symmetric": sym,
                "Chain_length": chain,
                "HeteroEndFrag": het,
            }
        return metadata

    def _append_smiles_metadata_csv(self, rows: list[list]) -> None:
        """Append rows of '[smiles, symmetric, chain, hetero]' to SMILES.csv."""
        columns = ["SMILES", "Symmetric", "Chain_length", "HeteroEndFrag"]
        new_df = pd.DataFrame(rows, columns=columns)
        base = self.get_base_dir()
        csv_path = os.path.join(base, "SMILES.csv")
        for name in ("SMILES.csv", "smiles.csv", "Smiles.csv"):
            candidate = os.path.join(base, name)
            if os.path.isfile(candidate):
                csv_path = candidate
                break
        if os.path.isfile(csv_path):
            try:
                old = pd.read_csv(csv_path)
                for col in columns:
                    if col not in old.columns:
                        old[col] = None
                new_df = pd.concat([old[columns], new_df], ignore_index=True)
            except Exception:
                pass
        new_df.to_csv(csv_path, index=False)

    def prompt_missing_smiles_metadata(self, smiles_list: list[str]):
        """Prompt the user to fill in metadata for SMILES not in SMILES.csv."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Missing SMILES Metadata")
        dialog.resize(820, 420)
        dialog.setStyleSheet(self.DIALOG_QSS)
        layout = QVBoxLayout(dialog)

        tip = QLabel(
            "The following SMILES were not found in SMILES.csv. "
            "Please fill in their metadata:"
        )
        tip.setWordWrap(True)
        layout.addWidget(tip)

        table = self._build_metadata_table(smiles_list, dialog)
        layout.addWidget(table)

        append_checkbox = QCheckBox("Append these entries to SMILES.csv")
        append_checkbox.setChecked(True)
        layout.addWidget(append_checkbox)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.Accepted:
            return None, False
        sym_values, chain_values, het_values = self._read_metadata_table(
            table, len(smiles_list)
        )
        filled = {
            smiles: {"Symmetric": sym, "Chain_length": chain, "HeteroEndFrag": het}
            for smiles, sym, chain, het in zip(
                smiles_list, sym_values, chain_values, het_values
            )
        }
        return filled, append_checkbox.isChecked()

    # ==================================================================
    # Molecular calculation
    # ==================================================================

    def start_mol_calc(self) -> None:
        """Validate inputs and dispatch the MolCalc child process."""
        if self.mol_calc_process is not None and self.mol_calc_process.is_alive():
            self.txt_status_message.setPlainText("MolCalc already running.")
            self.set_state("warning")
            return

        if not self.valid_smiles_list:
            self.txt_status_message.setPlainText(
                "No valid SMILES. Please input molecules first."
            )
            self.set_state("warning")
            return

        smiles_df = pd.DataFrame({"SMILES": self.valid_smiles_list})
        self._init_mol_queue(self.valid_smiles_list)

        # Clearing the editor keeps the visible batch identical to the one
        # being computed and blocks a duplicate click from overlapping batches.
        self.txt_mol_smiles.blockSignals(True)
        self.txt_mol_smiles.clear()
        self.txt_mol_smiles.blockSignals(False)
        self.mol_validate_timer.stop()
        self.valid_smiles_list = []
        self.invalid_smiles_lines = []
        self.lbl_mol_counts.setText(
            "<span style='color:#98c379;'>Valid - 0</span>"
            " | "
            "<span style='color:#e06c75;'>Invalid - 0</span>"
        )

        method = self.cmb_mol_method.currentText().strip()
        basis_set = self.cmb_mol_basis.currentText().strip()
        opt = self.chk_mol_opt.isChecked()
        freq = self.chk_mol_freq.isChecked()
        tightscf = self.chk_mol_tightscf.isChecked()
        multithreading = self.chk_mol_multithread.isChecked()
        ncore = max(1, self.spn_mol_ncore.value())
        cpu_usage = self.dsp_mol_cpu.value()
        seed = 0xF00D

        self.mol_calc_output_dir = self.mol_calc_dir
        self.last_mol_calc_row_count = 0
        self.is_mol_table_dirty = False
        self.tbl_mol_results.setRowCount(0)
        self.tbl_mol_results.setColumnCount(0)

        self.txt_status_message.setPlainText("MolCalc is starting...")
        self.set_state("normal")
        self._set_pair_running(self.btn_mol_start, self.btn_mol_stop)
        self.btn_mol_library.setEnabled(False)

        self.mol_calc_thread_obj = WorkerThread(
            lambda: self._create_mol_calc_process(
                smiles_df,
                self.mol_calc_output_dir,
                method,
                basis_set,
                opt,
                freq,
                tightscf,
                ncore,
                cpu_usage,
                multithreading,
                seed,
                self.mol_calc_message_queue,
            )
        )
        self.mol_calc_thread_obj.results.connect(self.on_mol_calc_started)
        self.mol_calc_thread_obj.finished.connect(self.mol_calc_thread_obj.deleteLater)
        self.mol_calc_thread_obj.start()

    @staticmethod
    def _create_mol_calc_process(
        smiles_df,
        output_dir,
        method,
        basis_set,
        opt,
        freq,
        tightscf,
        ncore,
        cpu_usage,
        multithreading,
        seed,
        message_queue,
    ) -> Process:
        process = Process(
            target=mol_calc_worker,
            args=(
                smiles_df, output_dir, method, basis_set, opt, freq,
                tightscf, ncore, cpu_usage, multithreading, seed, message_queue,
            ),
            daemon=True,
        )
        process.start()
        return process

    def on_mol_calc_started(self, process_obj: Process) -> None:
        self.mol_calc_process = process_obj
        self.txt_status_message.setPlainText("MolCalc is running...")
        self.set_state("normal")
        self.mol_calc_timer.start(2000)
        self.mol_calc_queue_timer.start(200)

    def on_mol_calc_tick(self) -> None:
        """Poll 'mol_prop.csv' and append newly completed molecules."""
        csv_path = os.path.join(self.mol_calc_output_dir, "mol_prop.csv")
        if os.path.isfile(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if len(df) > self.last_mol_calc_row_count:
                    self._append_mol_calc_rows(df)
                    self.last_mol_calc_row_count = len(df)
                    self.txt_status_message.setPlainText(
                        f"Completed: {len(df)} molecule(s)"
                    )
            except Exception:
                pass

        if self.mol_calc_process is not None and not self.mol_calc_process.is_alive():
            self.mol_calc_process.join()
            self.mol_calc_process = None
            self.mol_calc_timer.stop()
            if os.path.isfile(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    self._append_mol_calc_rows(df)
                except Exception:
                    pass
            self.on_mol_calc_finished("MolCalc task completed.")

    def on_mol_calc_queue_tick(self) -> None:
        """Drain MolCalc status messages from the child process."""
        latest_log: Optional[str] = None
        latest_progress: Optional[float] = None

        while True:
            try:
                message = self.mol_calc_message_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
            if message is None:
                continue
            text = str(message)
            if text.startswith("_progr"):
                try:
                    latest_progress = float(text[len("_progr"):])
                except ValueError:
                    pass
            else:
                latest_log = text
                if self.mol_queue_widget is not None:
                    parsed = self._parse_mol_status(text)
                    if parsed is not None:
                        smiles, status, thread_id = parsed
                        self.mol_queue_widget.update_status(smiles, status, thread_id)

        if latest_log is not None:
            self.txt_status_message.setPlainText(latest_log)
        if latest_progress is not None:
            self.pbar_main.setValue(int(latest_progress * 100))

    def stop_mol_calc(self) -> None:
        """Terminate the MolCalc child and all of its ORCA descendants.

        ORCA spawns its own subprocesses that do not honour 'terminate' on
        the direct child, so the tree is walked via 'psutil' and each
        descendant killed explicitly.
        """
        if self.mol_calc_process is not None:
            try:
                pid = self.mol_calc_process.pid
                parent = psutil.Process(pid)
                for child in parent.children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                try:
                    parent.kill()
                    parent.wait(timeout=3)
                except psutil.NoSuchProcess:
                    pass
            except psutil.NoSuchProcess:
                pass
            except Exception:
                try:
                    self.mol_calc_process.terminate()
                except Exception:
                    pass
            self.mol_calc_process.join(timeout=2)
            self.mol_calc_process = None

        for exe in ("orca.exe", "orca_2mkl.exe", "multiwfn.exe"):
            try:
                subprocess.run(
                    ["taskkill", "/F", "/IM", exe],
                    capture_output=True,
                    timeout=3,
                )
            except Exception:
                pass

        self.mol_calc_timer.stop()
        self.mol_calc_queue_timer.stop()
        self._clear_mol_queue()
        self.on_mol_calc_finished("MolCalc stopped.")

    def on_mol_calc_finished(self, message: str) -> None:
        self._set_pair_idle(self.btn_mol_start, self.btn_mol_stop)
        self.btn_mol_library.setEnabled(True)
        self.txt_status_message.setPlainText(message)
        self.set_state("normal")
        self.mol_calc_queue_timer.stop()

    @staticmethod
    def _parse_mol_status(raw: str):
        """Parse one MolCalc status message into '(smiles, status, thread_id)'.

        Accepted forms::

            [T<id>] [ORCA] Start|Done|Timeout|Failed <smiles>
            [Prop] Start|Done|Skip <smiles>
            [Skip] <reason>

        Returns 'None' for messages that should not update a chip.
        """
        if not raw:
            return None
        thread_id = None
        text = raw
        if text.startswith("[T"):
            end = text.find("]")
            if end > 0:
                thread_id = text[2:end]
                text = text[end + 1:].lstrip()

        if text.startswith("[ORCA]"):
            rest = text[6:].strip()
            for tag, status in (
                ("Start", "orca"),
                ("Done", None),
                ("Timeout", "error"),
                ("Failed", "error"),
            ):
                if rest.startswith(tag):
                    smiles = rest[len(tag):].strip()
                    if tag == "Failed":
                        smiles = smiles.split(":", 1)[0].strip()
                    if status is None:
                        return None
                    return smiles, status, thread_id

        if text.startswith("[Prop]"):
            rest = text[6:].strip()
            for tag, status in (("Start", "prop"), ("Done", "done"), ("Skip", "done")):
                if rest.startswith(tag):
                    smiles = rest[len(tag):].strip()
                    if tag == "Skip":
                        smiles = smiles.split("—", 1)[0].strip()
                    return smiles, status, None

        if text.startswith("[Skip]"):
            return None
        return None

    def _init_mol_queue(self, smiles_list: list[str]) -> None:
        """Replace 'frm_mol_queue' contents with a fresh chip queue."""
        container = self.frm_mol_queue
        if container is None:
            print("[MolQueue] FAILED: frm_mol_queue not found in .ui")
            return

        old_layout = container.layout()
        if old_layout is None:
            old_layout = QVBoxLayout(container)
            old_layout.setContentsMargins(0, 0, 0, 0)
            old_layout.setSpacing(0)
        else:
            while old_layout.count():
                item = old_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                    widget.deleteLater()

        scroll = QScrollArea(container)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        inner_layout = QHBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.setSpacing(0)

        self.mol_queue_widget = MolQueueWidget(smiles_list, inner)
        inner_layout.addWidget(self.mol_queue_widget)
        scroll.setWidget(inner)
        old_layout.addWidget(scroll)
        container.show()

    def _clear_mol_queue(self) -> None:
        """Stop any in-flight chip animations and empty the queue widget."""
        if self.mol_queue_widget is not None:
            animations = getattr(self.mol_queue_widget, "_animations", None)
            if animations is not None:
                for anim in list(animations):
                    try:
                        anim.stop()
                    except Exception:
                        pass
                animations.clear()

        container = self.frm_mol_queue
        if container is not None:
            layout = container.layout()
            if layout is not None:
                while layout.count():
                    item = layout.takeAt(0)
                    widget = item.widget()
                    if widget is not None:
                        widget.setParent(None)
                        widget.deleteLater()
        self.mol_queue_widget = None

    def on_mol_table_item_changed(self, item: QTableWidgetItem) -> None:
        if getattr(self, "is_mol_table_loading", False):
            return
        if item is None:
            return
        self.is_mol_table_dirty = True
        self.txt_status_message.setPlainText(
            "The table has been modified — right-click → Save to mol_prop.csv."
        )
        self.set_state("warning")

    def on_mol_table_context_menu(self, pos) -> None:
        table = self.tbl_mol_results
        if table is None:
            return

        menu = QMenu(self)
        delete_action = menu.addAction("Delete selected row")
        menu.addSeparator()
        save_action = menu.addAction("Save to mol_prop.csv")
        delete_action.setEnabled(bool(table.selectedIndexes()))
        save_action.setEnabled(self.is_mol_table_dirty)

        action = menu.exec(table.viewport().mapToGlobal(pos))
        if action is delete_action:
            rows = sorted(
                {index.row() for index in table.selectedIndexes()}, reverse=True
            )
            body = (
                "<div style=\"font-family: 'Consolas'; font-size: 13px; "
                "color: #b8c2d6;\">"
                f"Are you sure you want to delete the selected {len(rows)} items?"
                "</div>"
            )
            if self._show_info_dialog(
                "Confirm deletion",
                body,
                icon=QStyle.SP_MessageBoxQuestion,
                buttons=("No", "Yes"),
                default_button="No",
                min_width=400,
            ) != QDialog.Accepted:
                return
            for row in rows:
                table.removeRow(row)
            self.is_mol_table_dirty = True
            self.txt_status_message.setPlainText("Deleted lines, remember to save.")
            self.set_state("warning")
        elif action is save_action:
            self.save_mol_table()

    def save_mol_table(self) -> None:
        table = self.tbl_mol_results
        column_count = table.columnCount()
        if column_count == 0:
            return
        headers = [
            table.horizontalHeaderItem(j).text()
            if table.horizontalHeaderItem(j)
            else f"col{j}"
            for j in range(column_count)
        ]
        data = []
        for i in range(table.rowCount()):
            row = [
                table.item(i, j).text() if table.item(i, j) else ""
                for j in range(column_count)
            ]
            data.append(row)
        df = pd.DataFrame(data, columns=headers)
        csv_path = os.path.join(self.mol_calc_dir, "mol_prop.csv")
        try:
            df.to_csv(csv_path, index=False)
        except Exception as exc:
            self.txt_status_message.setPlainText(f"Save failed: {exc}")
            self.set_state("error")
            return
        self.is_mol_table_dirty = False
        self.last_mol_calc_row_count = len(df)
        self.txt_status_message.setPlainText(f"Saved {len(df)} rows to mol_prop.csv")
        self.set_state("normal")

    def _append_mol_calc_rows(self, df: pd.DataFrame) -> None:
        """Append rows not yet displayed, without rebuilding the whole table.

        Rebuilding on every tick would reset the user's selection and scroll
        position; incremental appends are faster and less disruptive.  No-op
        when the table is dirty because the user is editing it.
        """
        if self.is_mol_table_dirty:
            return
        table = self.tbl_mol_results
        columns = list(df.columns)
        self.is_mol_table_loading = True
        try:
            current_headers = [
                table.horizontalHeaderItem(j).text()
                if table.horizontalHeaderItem(j)
                else ""
                for j in range(table.columnCount())
            ]
            if current_headers != columns:
                table.setColumnCount(len(columns))
                table.setHorizontalHeaderLabels(columns)
                table.setRowCount(0)
                start = 0
                header = table.horizontalHeader()
                header.setSectionResizeMode(QHeaderView.Interactive)
                for j, column in enumerate(columns):
                    if column == "SMILES":
                        table.setColumnWidth(j, 220)
                    elif column in ("METHOD", "BASIS"):
                        table.setColumnWidth(j, 120)
                    else:
                        table.setColumnWidth(j, 120)
            else:
                start = table.rowCount()

            for i in range(start, len(df)):
                row = table.rowCount()
                table.insertRow(row)
                for j, column in enumerate(columns):
                    table.setItem(row, j, QTableWidgetItem(str(df.iloc[i][column])))
        finally:
            self.is_mol_table_loading = False

    def open_molecule_library(self) -> None:
        csv_path = os.path.join(self.mol_calc_dir, "mol_prop.csv")
        if not os.path.isfile(csv_path):
            self.txt_status_message.setPlainText(
                f"mol_prop.csv not found at: {csv_path}"
            )
            self.set_state("warning")
            return
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            self.txt_status_message.setPlainText(f"Failed to read mol_prop.csv: {exc}")
            self.set_state("error")
            return
        self.is_mol_table_dirty = False
        self.tbl_mol_results.setRowCount(0)
        self.tbl_mol_results.setColumnCount(0)
        self._append_mol_calc_rows(df)
        self.last_mol_calc_row_count = len(df)
        self.txt_status_message.setPlainText(
            f"Loaded mol_prop.csv: {len(df)} molecule(s)"
        )
        self.set_state("normal")

    # ==================================================================
    # Dataset assembly
    # ==================================================================

    def browse_experiment_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Experiment Dataset",
            self.get_base_dir(),
            "CSV Files (*.csv);;All Files (*)",
        )
        if path:
            self.txt_data_exp_path.setPlainText(path)

    def browse_smiles_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SMILES Dataset",
            self.get_base_dir(),
            "CSV Files (*.csv);;All Files (*)",
        )
        if path:
            self.txt_data_smiles_path.setPlainText(path)

    def merge_datasets(self) -> None:
        """Validate inputs, check coverage in mol_prop.csv, and run the merge."""
        exp_path = self.txt_data_exp_path.toPlainText().strip()
        smiles_path = self.txt_data_smiles_path.toPlainText().strip()

        if not exp_path or not os.path.isfile(exp_path):
            self.txt_status_message.setPlainText(
                "Please select a valid experiment dataset file."
            )
            self.set_state("warning")
            return
        if not smiles_path or not os.path.isfile(smiles_path):
            self.txt_status_message.setPlainText(
                "Please select a valid SMILES dataset file."
            )
            self.set_state("warning")
            return

        method = self.cmb_data_method.currentText().strip()
        basis = self.cmb_data_basis.currentText().strip()
        if not method or not basis:
            self.txt_status_message.setPlainText("Please select a method and basis set.")
            self.set_state("warning")
            return

        try:
            exp_df = pd.read_csv(exp_path)
            smiles_df = pd.read_csv(smiles_path)
        except Exception as exc:
            self.txt_status_message.setPlainText(f"Failed to read input files: {exc}")
            self.set_state("error")
            return
        exp_df.columns = [str(c).strip() for c in exp_df.columns]
        smiles_df.columns = [str(c).strip() for c in smiles_df.columns]

        required_exp = {
            "SAM_1", "SAM_2", "SAMs_ratio", "Bandgap",
            "Mixture", "MACl_additive", "PCE",
        }
        required_smiles = {"SMILES", "Symmetric", "Chain_length", "HeteroEndFrag"}
        missing_exp = required_exp - set(exp_df.columns)
        missing_smiles = required_smiles - set(smiles_df.columns)
        if missing_exp:
            self.txt_status_message.setPlainText(
                f"Experiment dataset is missing required columns: {sorted(missing_exp)}"
            )
            self.set_state("error")
            return
        if missing_smiles:
            self.txt_status_message.setPlainText(
                f"SMILES dataset is missing required columns: {sorted(missing_smiles)}"
            )
            self.set_state("error")
            return

        mol_prop_path = os.path.join(self.mol_calc_dir, "mol_prop.csv")
        filtered: Optional[pd.DataFrame] = None
        if os.path.isfile(mol_prop_path):
            try:
                mol_prop_df = pd.read_csv(mol_prop_path)
                mol_prop_df.columns = [str(c).strip() for c in mol_prop_df.columns]
                if {"SMILES", "METHOD", "BASIS"}.issubset(mol_prop_df.columns):
                    sub = mol_prop_df[
                        (mol_prop_df["METHOD"].astype(str).str.strip() == method)
                        & (mol_prop_df["BASIS"].astype(str).str.strip() == basis)
                    ].copy()
                    if not sub.empty:
                        sub["SMILES"] = sub["SMILES"].astype(str).str.strip()
                        filtered = sub
            except Exception:
                filtered = None

        available = set(filtered["SMILES"]) if filtered is not None else set()

        required_set: set[str] = set()
        for column in ("SAM_1", "SAM_2"):
            for value in exp_df[column].dropna().astype(str).str.strip():
                if value:
                    required_set.add(value)
        for value in smiles_df["SMILES"].dropna().astype(str).str.strip():
            if value:
                required_set.add(value)

        missing_smiles_list = sorted(required_set - available)

        if missing_smiles_list:
            accepted = self._ask_jump_to_mol_calc(
                n_missing=len(missing_smiles_list),
                missing_list=missing_smiles_list,
                method=method,
                basis=basis,
                no_entries=(filtered is None),
            )
            if accepted != QDialog.Accepted:
                self.txt_status_message.setPlainText("Merge cancelled by user.")
                self.set_state("warning")
                return

            self.stk_main.setCurrentWidget(self.pg_mol)
            self.txt_mol_smiles.blockSignals(True)
            self.txt_mol_smiles.setPlainText("\n".join(missing_smiles_list))
            self.txt_mol_smiles.blockSignals(False)
            self.validate_mol_smiles()
            idx = self.cmb_mol_method.findText(method)
            if idx >= 0:
                self.cmb_mol_method.setCurrentIndex(idx)
            idx = self.cmb_mol_basis.findText(basis)
            if idx >= 0:
                self.cmb_mol_basis.setCurrentIndex(idx)
            if not self._is_mol_calc_settings_visible:
                self.toggle_mol_calc_settings()
            self.txt_status_message.setPlainText(
                f"Jumped to Molecular Calculation page. "
                f"{len(missing_smiles_list)} molecule(s) need to be computed "
                f"under {method}/{basis}."
            )
            self.set_state("normal")
            return

        self.btn_data_merge.setEnabled(False)
        self.btn_data_merge.setText("Merging...")
        self.txt_status_message.setPlainText("Merging datasets, please wait...")
        self.set_state("normal")

        base_dir = self.get_base_dir()
        self.merge_thread_obj = WorkerThread(
            lambda: merge_worker(exp_df, smiles_df, filtered, base_dir)
        )
        self.merge_thread_obj.results.connect(self.on_merge_finished)
        self.merge_thread_obj.log.connect(self.on_merge_failed)
        self.merge_thread_obj.finished.connect(self.merge_thread_obj.deleteLater)
        self.merge_thread_obj.start()

    def on_merge_finished(self, result: Optional[dict]) -> None:
        self.btn_data_merge.setEnabled(True)
        self.btn_data_merge.setText("MERGE")
        if result is None or result.get("error"):
            error_message = (result or {}).get("error") or "Unknown error"
            self.txt_status_message.setPlainText(f"Merge failed: {error_message}")
            self.set_state("error")
            return
        ml_inputs = result["ml_inputs"]
        dl_inputs = result["dl_inputs"]
        self._display_merge_results(ml_inputs)
        self.txt_status_message.setPlainText(
            f"Merge completed. ml_inputs: {ml_inputs.shape[0]}×{ml_inputs.shape[1]}, "
            f"dl_inputs: {dl_inputs.shape[0]}×{dl_inputs.shape[1]}."
        )
        self.set_state("normal")

    def on_merge_failed(self, message: str) -> None:
        self.btn_data_merge.setEnabled(True)
        self.btn_data_merge.setText("MERGE")
        self.txt_status_message.setPlainText(f"Merge failed: {message}")
        self.set_state("error")

    def _ask_jump_to_mol_calc(
        self,
        n_missing: int,
        missing_list: list[str],
        method: str,
        basis: str,
        no_entries: bool,
    ) -> int:
        if no_entries:
            body_html = (
                "<div style=\"font-family: 'Consolas'; font-size: 13px; "
                "color: #b8c2d6; line-height: 1.5;\">"
                "No records were found in <b style='color:#e8eef8;'>mol_prop.csv</b> for "
                f"<b style='color:#e8eef8;'>method={method}</b>, "
                f"<b style='color:#e8eef8;'>basis={basis}</b>.<br>"
                "Please compute the following molecules first:"
                "</div>"
            )
        else:
            body_html = (
                "<div style=\"font-family: 'Consolas'; font-size: 13px; "
                "color: #b8c2d6; line-height: 1.5;\">"
                f"<b style='color:#e8eef8;'>{n_missing}</b> molecule(s) are missing in "
                "<b style='color:#e8eef8;'>mol_prop.csv</b> under "
                f"<b style='color:#e8eef8;'>method={method}</b>, "
                f"<b style='color:#e8eef8;'>basis={basis}</b>.<br>"
                "Please compute them first:"
                "</div>"
            )

        preview = missing_list[:10]
        details_html = (
            "<div style='font-family: Consolas, monospace; font-size: 12px; "
            "color: #a8d0ff; background-color: #1e2a44; padding: 8px 10px; "
            "border-radius: 6px; line-height: 1.55;'>"
        )
        for smiles in preview:
            chunks = [smiles[i:i + 50] for i in range(0, len(smiles), 50)]
            smiles_display = "<br>&nbsp;&nbsp;&nbsp;&nbsp;".join(chunks)
            details_html += f"&nbsp;&nbsp;{smiles_display}<br>"
        if n_missing > len(preview):
            details_html += (
                f"<span style='color:#8b9bb4;'>... and "
                f"{n_missing - len(preview)} more</span>"
            )
        details_html += "</div>"

        ask_html = (
            "<div style=\"font-family: 'Consolas'; font-size: 13px; "
            "color: #b8c2d6;\">"
            "Do you want to jump to the <i>Molecular Calculation</i> page "
            "to compute them now?"
            "</div>"
        )

        return self._show_info_dialog(
            "Missing Molecule Properties",
            body_html,
            details_html=details_html,
            ask_html=ask_html,
            buttons=("No", "Yes"),
            default_button="No",
            min_width=540,
        )

    def _display_merge_results(self, df: pd.DataFrame, max_rows: int = 1000) -> None:
        """Render the merged 'ml_inputs' DataFrame in the preview table."""
        table = self.tbl_data_preview
        if table is None:
            return
        table.setUpdatesEnabled(False)
        try:
            table.clear()
            table.setColumnCount(len(df.columns))
            table.setHorizontalHeaderLabels([str(c) for c in df.columns])
            n = min(len(df), max_rows)
            table.setRowCount(n)
            for i in range(n):
                for j, column in enumerate(df.columns):
                    value = df.iloc[i, j]
                    try:
                        text = (
                            f"{value:.6g}"
                            if isinstance(value, (float, np.floating))
                            else str(value)
                        )
                    except Exception:
                        text = str(value)
                    table.setItem(i, j, QTableWidgetItem(text))
        finally:
            table.setUpdatesEnabled(True)

    def refresh_data_counts(self) -> None:
        """Update 'lbl_data_counts' with live sample statistics."""
        label = getattr(self, "lbl_data_counts", None)
        if label is None:
            return

        exp_path = self.txt_data_exp_path.toPlainText().strip() if self.txt_data_exp_path else ""
        smiles_path = self.txt_data_smiles_path.toPlainText().strip() if self.txt_data_smiles_path else ""
        method = self.cmb_data_method.currentText().strip() if self.cmb_data_method else ""
        basis = self.cmb_data_basis.currentText().strip() if self.cmb_data_basis else ""

        if not exp_path or not smiles_path:
            label.setText(
                "<span style='color:#8b9bb4;'>"
                "Select experiment and SMILES datasets...</span>"
            )
            return

        def show_error(message: str) -> None:
            label.setText(f"<span style='color:#e06c75;'>{message}</span>")

        if not os.path.isfile(exp_path):
            show_error("Experiment dataset not found")
            return
        try:
            exp_df = pd.read_csv(exp_path)
        except Exception as exc:
            show_error(f"Failed to read experiment dataset: {exc}")
            return
        exp_df.columns = [str(c).strip() for c in exp_df.columns]

        if not os.path.isfile(smiles_path):
            show_error("SMILES dataset not found")
            return
        try:
            smiles_df = pd.read_csv(smiles_path)
        except Exception as exc:
            show_error(f"Failed to read SMILES dataset: {exc}")
            return
        smiles_df.columns = [str(c).strip() for c in smiles_df.columns]

        required_exp = {
            "SAM_1", "SAM_2", "SAMs_ratio", "Bandgap",
            "Mixture", "MACl_additive", "PCE",
        }
        required_smiles = {"SMILES", "Symmetric", "Chain_length", "HeteroEndFrag"}
        missing_exp = required_exp - set(exp_df.columns)
        missing_smiles = required_smiles - set(smiles_df.columns)

        if missing_exp or missing_smiles:
            if missing_exp and missing_smiles:
                show_error("Both datasets are missing required columns")
            elif missing_exp:
                show_error("Experiment dataset is missing required columns")
            else:
                show_error("SMILES dataset is missing required columns")
            return

        n_exp = len(exp_df)
        n_smiles = len(smiles_df)

        required_set: set[str] = set()
        for column in ("SAM_1", "SAM_2"):
            for value in exp_df[column].dropna().astype(str).str.strip():
                if value:
                    required_set.add(value)
        for value in smiles_df["SMILES"].dropna().astype(str).str.strip():
            if value:
                required_set.add(value)

        invalid_smiles: list[str] = []
        for smiles in required_set:
            ok, _ = self.is_valid_smiles(smiles)
            if not ok:
                invalid_smiles.append(smiles)
        if invalid_smiles:
            preview = "; ".join(invalid_smiles[:3])
            if len(invalid_smiles) > 3:
                preview += f"; ... (+{len(invalid_smiles) - 3})"
            show_error(f"Invalid SMILES ({len(invalid_smiles)}): {preview}")
            return

        available: set[str] = set()
        if method and basis:
            mol_prop_path = os.path.join(self.mol_calc_dir, "mol_prop.csv")
            if os.path.isfile(mol_prop_path):
                try:
                    mp = pd.read_csv(mol_prop_path)
                    mp.columns = [str(c).strip() for c in mp.columns]
                    if {"SMILES", "METHOD", "BASIS"}.issubset(mp.columns):
                        sub = mp[
                            (mp["METHOD"].astype(str).str.strip() == method)
                            & (mp["BASIS"].astype(str).str.strip() == basis)
                        ]
                        available = set(sub["SMILES"].astype(str).str.strip())
                except Exception:
                    available = set()

        n_ready = len(required_set & available) if available else 0
        n_missing = len(required_set) - n_ready
        miss_color = "#98c379" if n_missing == 0 else "#e06c75"

        parts = [
            f"<span style='color:#a8d0ff;'>Exp</span>&nbsp;{n_exp}",
            f"<span style='color:#a8d0ff;'>SMILES</span>&nbsp;{n_smiles}",
            f"<span style='color:#98c379;'>Ready</span>&nbsp;{n_ready}",
            f"<span style='color:{miss_color};'>Missing</span>&nbsp;{n_missing}",
        ]
        label.setText(" &nbsp;|&nbsp; ".join(parts))

    # ==================================================================
    # Machine-learning training
    # ==================================================================

    def on_algorithm_check_changed(self, item: QListWidgetItem) -> None:
        """Implement the "select-all" master checkbox at row zero."""
        row = self.algorithm_list_widget.row(item)
        if row != 0:
            return
        self.algorithm_list_widget.blockSignals(True)
        if self.algorithm_list_widget.item(0).checkState() == Qt.Checked:
            for i in range(self.algorithm_list_widget.count() - 1):
                self.algorithm_list_widget.item(i + 1).setCheckState(Qt.Checked)
        elif self.algorithm_list_widget.item(0).checkState() == Qt.Unchecked:
            for i in range(self.algorithm_list_widget.count() - 1):
                self.algorithm_list_widget.item(i + 1).setCheckState(Qt.Unchecked)
        self.algorithm_list_widget.blockSignals(False)

    def start_ml_training(self) -> None:
        if self.ml_process and self.ml_process.is_alive():
            return
        if not self.check_ml_inputs("train ML models"):
            return

        self._set_pair_loading(self.btn_ml_start, self.btn_ml_stop)
        self.txt_status_message.setPlainText(
            "ML process has been started and is now executing..."
        )
        self.set_state("normal")
        self.txt_ml_log.clear()
        self.pbar_main.setValue(0)
        self.ml_performance_df = self.ml_performance_df.iloc[0:0]

        self.ml_timer.stop()
        self.ml_timer.start(100)

        selection_flags: list[bool] = []
        total = self.algorithm_list_widget.count()
        self.selected_algorithm_count = total
        for index in range(total):
            item = self.algorithm_list_widget.item(index)
            if index == 0 and item.checkState() == Qt.Checked:
                selection_flags = [True] * (total - 1)
                break
            if index == 0 and item.checkState() == Qt.Unchecked:
                continue
            selection_flags.append(item.checkState() == Qt.Checked)
        self.selected_algorithm_count = sum(selection_flags)

        if all(not flag for flag in selection_flags):
            self._set_pair_idle(self.btn_ml_start, self.btn_ml_stop)
            self.txt_status_message.setPlainText("No algorithms selected!")
            self.set_state("warning")
            self.ml_timer.stop()
            return

        train_size = self.dsp_ml_split.value()
        random_state = self.spn_ml_seed.value()
        base_dir = self.get_base_dir()
        initial_points = self.spn_ml_init_points.value()
        iterations = self.spn_ml_iterations.value()

        self.ml_thread_obj = WorkerThread(
            lambda: self._create_ml_process(
                train_size,
                random_state,
                base_dir,
                self.ml_message_queue,
                selection_flags,
                initial_points,
                iterations,
                random_state,
            )
        )
        self.ml_thread_obj.results.connect(self.on_ml_training_started)
        self.ml_thread_obj.log.connect(self.on_ml_training_failed)
        self.ml_thread_obj.finished.connect(self.ml_thread_obj.deleteLater)
        self.ml_thread_obj.start()

    @staticmethod
    def _create_ml_process(
        train_size,
        random_state,
        base_dir,
        message_queue,
        selection_flags,
        initial_points,
        iterations,
        seed,
    ) -> Process:
        dataset = pd.read_csv("ml_inputs.csv")
        x = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=train_size, shuffle=True, random_state=random_state
        )
        process = Process(
            target=ml_worker,
            args=(
                base_dir, message_queue, x_train, x_test, y_train, y_test,
                selection_flags, initial_points, iterations, seed,
            ),
            daemon=True,
        )
        process.start()
        return process

    def on_ml_training_started(self, process_obj: Process) -> None:
        self.ml_process = process_obj
        self._set_pair_running(self.btn_ml_start, self.btn_ml_stop)
        self.txt_status_message.setPlainText("ML process is running...")
        self.set_state("normal")

    def on_ml_training_failed(self, message: str) -> None:
        self.ml_process = None
        self.selected_algorithm_count = 0
        self.ml_timer.stop()
        self._set_pair_idle(self.btn_ml_start, self.btn_ml_stop)
        self.txt_status_message.setPlainText(f"ML failed to start: {message}")
        self.set_state("error")

    def on_ml_queue_tick(self) -> None:
        """Drain the AutoML message queue and refresh the metrics display."""
        while True:
            try:
                message = self.ml_message_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(message, pd.DataFrame):
                self.ml_performance_df = pd.concat(
                    [self.ml_performance_df, message], ignore_index=True, axis=0
                )
                metric = str(self.cmb_ml_metric.currentText().strip())
                ascending = metric in ("MSE", "MAE")
                self.ml_performance_df.sort_values(
                    by=metric, inplace=True, ascending=ascending, ignore_index=True
                )
                best_name = self.ml_performance_df["name"][0]
                self.txt_ml_optimal_model.setPlainText(
                    self.algorithm_display_names[best_name]
                )
                self.ml_optimal_model = self.algorithm_display_names[best_name]
                self.txt_ml_r2.setPlainText(f"{self.ml_performance_df['R2'][0]:.4f}")
                self.txt_ml_mse.setPlainText(f"{self.ml_performance_df['MSE'][0]:.4f}")
                self.txt_ml_mae.setPlainText(f"{self.ml_performance_df['MAE'][0]:.4f}")
                self.pbar_main.setValue(
                    self.ml_performance_df.shape[0] / self.selected_algorithm_count * 100
                )
                self.txt_status_message.setPlainText("ML process is running...")
                self.set_state("normal")
            else:
                html_log = ansi_to_html(str(message))
                cursor = self.txt_ml_log.textCursor()
                cursor.movePosition(QTextCursor.MoveOperation.End)
                cursor.insertHtml(html_log + "<br>")
                self.txt_ml_log.verticalScrollBar().setValue(
                    self.txt_ml_log.verticalScrollBar().maximum()
                )

        if self.ml_process is not None and not self.ml_process.is_alive():
            self.ml_process.join()
            if self.ml_process.exitcode != 0:
                print(f"Child process exited abnormally: exitcode={self.ml_process.exitcode}")
            self.ml_process = None
            self.selected_algorithm_count = 0
            self.txt_status_message.setPlainText("ML task completed.")
            self.set_state("normal")
            self.ml_timer.stop()
            self._set_pair_idle(self.btn_ml_start, self.btn_ml_stop)

    def stop_ml_training(self) -> None:
        self._set_button_state(
            self.btn_ml_stop,
            text="Stopping",
            callback=None,
            qss=self.STOP_BUTTON_QSS,
            enabled=False,
        )
        if self.ml_process and self.ml_process.is_alive():
            self.ml_process.terminate()
            self.ml_process.join()
            self.ml_process = None
            self.selected_algorithm_count = 0
            self.ml_timer.stop()
            self.txt_status_message.setPlainText("ML task stopped!")
            self.set_state("normal")
        self._set_pair_idle(self.btn_ml_start, self.btn_ml_stop)

    # ==================================================================
    # Deep-learning training
    # ==================================================================

    def start_dl_training(self) -> None:
        if self.dl_process and self.dl_process.is_alive():
            self.txt_status_message.setPlainText("DL process is already running!")
            self.set_state("warning")
            return
        if not self.check_ml_inputs("train DL models"):
            return
        if self.spn_dl_epochs.value() == 0:
            self._set_pair_idle(self.btn_dl_start, self.btn_dl_stop)
            self.txt_status_message.setPlainText("Epochs cannot be 0!")
            self.set_state("warning")
            self.dl_timer.stop()
            return

        self._set_pair_loading(self.btn_dl_start, self.btn_dl_stop)
        self.txt_status_message.setPlainText(
            "DL process has been started and is now executing..."
        )
        self.set_state("normal")
        self.txt_dl_log.clear()
        self.pbar_main.setValue(0)
        self.dl_performance_df = self.dl_performance_df.iloc[0:0]

        self.dl_timer.stop()
        self.dl_timer.start(100)

        train_size = self.dsp_dl_split.value()
        random_state = self.spn_dl_seed.value()
        epochs = self.spn_dl_epochs.value()
        early_stopping = self.spn_dl_early_stop.value()
        device = self.cmb_dl_device.currentText()
        batch_size = self.spn_dl_batch.value()
        learning_rate = self.dsp_dl_learning_rate.value()
        dropout = self.dsp_dl_dropout.value()
        base_dir = self.get_base_dir()

        self.dl_thread_obj = WorkerThread(
            lambda: self._create_dl_process(
                train_size, random_state, epochs, early_stopping, device,
                batch_size, learning_rate, dropout, base_dir,
                self.dl_message_queue,
            )
        )
        self.dl_thread_obj.results.connect(self.on_dl_training_started)
        self.dl_thread_obj.log.connect(self.on_dl_training_failed)
        self.dl_thread_obj.finished.connect(self.dl_thread_obj.deleteLater)
        self.dl_thread_obj.start()

    @staticmethod
    def _create_dl_process(
        train_size,
        random_state,
        epochs,
        early_stopping,
        device,
        batch_size,
        learning_rate,
        dropout,
        base_dir,
        message_queue,
    ) -> Process:
        """Spawn the ACIN training child process and return its handle.

        Dataset loading and splitting are done here rather than in the child
        so a missing or malformed 'dl_inputs.csv' surfaces as an immediate
        error and the split matches every other consumer.
        """
        dataset = pd.read_csv("dl_inputs.csv")
        x = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]

        std_feats = StandardScaler()
        std_krfps = StandardScaler()
        x_feats = x.iloc[:, :35]
        x_krfps = x.iloc[:, 35:]
        x_feats_scaled = std_feats.fit_transform(x_feats)
        x_krfps_scaled = std_krfps.fit_transform(x_krfps)
        x = pd.concat(
            [
                pd.DataFrame(x_feats_scaled, columns=x.columns[:35]),
                pd.DataFrame(x_krfps_scaled, columns=x.columns[35:]),
            ],
            axis=1,
        )

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=train_size, shuffle=True, random_state=random_state
        )
        process = Process(
            target=dl_worker,
            args=(
                base_dir, message_queue, x_train, x_test, y_train, y_test,
                epochs, early_stopping, device, batch_size, learning_rate,
                dropout, random_state,
            ),
            daemon=True,
        )
        process.start()
        return process

    def on_dl_training_started(self, process_obj: Process) -> None:
        self.dl_process = process_obj
        self._set_pair_running(self.btn_dl_start, self.btn_dl_stop)
        self.txt_status_message.setPlainText("DL process is running...")
        self.set_state("normal")

    def on_dl_training_failed(self, message: str) -> None:
        self.dl_process = None
        self.dl_timer.stop()
        self._set_pair_idle(self.btn_dl_start, self.btn_dl_stop)
        self.txt_status_message.setPlainText(f"DL failed to start: {message}")
        self.set_state("error")

    def on_dl_queue_tick(self) -> None:
        """Drain the ACIN message queue and refresh the loss monitor."""
        while True:
            try:
                message = self.dl_message_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(message, pd.DataFrame):
                self.dl_performance_df = pd.concat(
                    [self.dl_performance_df, message], ignore_index=True, axis=0
                )
                self.dl_performance_df.sort_values(
                    by="Loss", inplace=True, ascending=True, ignore_index=True
                )
                self.txt_dl_r2.setPlainText(f"{self.dl_performance_df['R2'][0]:.4f}")
                self.txt_dl_mse.setPlainText(f"{self.dl_performance_df['MSE'][0]:.4f}")
                self.txt_dl_mae.setPlainText(f"{self.dl_performance_df['MAE'][0]:.4f}")
                self.txt_dl_loss.setPlainText(f"{self.dl_performance_df['Loss'][0]:.4f}")
            elif isinstance(message, int):
                self.pbar_main.setValue(message / self.spn_dl_epochs.value() * 100)
                self.txt_status_message.setPlainText("DL process is running...")
                self.set_state("normal")
            elif isinstance(message, str):
                if message.lower().startswith(self.MSG_PREFIX_END):
                    self.pbar_main.setValue(100)
                    self.txt_status_message.setPlainText("DL task completed.")
                    self.set_state("normal")
                else:
                    html_log = ansi_to_html(str(message))
                    cursor = self.txt_dl_log.textCursor()
                    cursor.movePosition(QTextCursor.MoveOperation.End)
                    cursor.insertHtml(html_log + "<br>")
                    self.txt_dl_log.verticalScrollBar().setValue(
                        self.txt_dl_log.verticalScrollBar().maximum()
                    )

        if self.dl_process is not None and not self.dl_process.is_alive():
            self.dl_process.join()
            if self.dl_process.exitcode != 0:
                print(f"Child process exited abnormally: exitcode={self.dl_process.exitcode}")
            self.dl_process = None
            self.txt_status_message.setPlainText("DL task completed.")
            self.set_state("normal")
            self.dl_timer.stop()
            self._set_pair_idle(self.btn_dl_start, self.btn_dl_stop)

    def stop_dl_training(self) -> None:
        self._set_button_state(
            self.btn_dl_stop,
            text="Stopping",
            callback=None,
            qss=self.STOP_BUTTON_QSS,
            enabled=False,
        )
        if self.dl_process and self.dl_process.is_alive():
            self.dl_process.terminate()
            self.dl_process.join()
            self.dl_process = None
            self.dl_timer.stop()
            self.txt_status_message.setPlainText("DL task stopped!")
            self.set_state("normal")
        self._set_pair_idle(self.btn_dl_start, self.btn_dl_stop)

    # ==================================================================
    # Substructure ranking
    # ==================================================================

    def load_acin_substructure_weights(self) -> Optional[pd.DataFrame]:
        """Load 'attention_weights.csv' into a DataFrame, if it exists."""
        try:
            df = pd.read_csv("attention_weights.csv").set_index("krfp_idx")
        except FileNotFoundError:
            self.txt_status_message.setPlainText(
                "Warning: attention_weights.csv not found — please run DL training first."
            )
            self.set_state("warning")
            return None
        except Exception as exc:
            self.txt_status_message.setPlainText(
                f"Warning: failed to read attention_weights.csv — {exc}"
            )
            self.set_state("warning")
            return None
        self.acin_substructure_weights = df
        return df

    def start_ranking(self) -> None:
        self.rank_stop_event.clear()
        self.ranking_scene.clear()
        self.gvw_rank_substructures.setVisible(True)

        if not self.ensure_spectrum_canvas():
            return

        self.txt_status_message.setPlainText("Ranking...")
        self.set_state("normal")
        self._set_task_running(self.btn_rank_run, self.stop_ranking, running_text=None)

        top_k = self.spn_rank_top_k.value()
        substructure_dir = os.path.join(self.plots_dir, "substructures")
        self.rank_thread_obj = WorkerThread(
            lambda: self._create_rank_process(
                top_k, substructure_dir, self.rank_stop_event
            )
        )
        self.rank_thread_obj.results.connect(self.on_ranking_finished)
        self.rank_thread_obj.finished.connect(self.rank_thread_obj.deleteLater)
        self.rank_thread_obj.start()

    @staticmethod
    def _create_rank_process(top_k: int, substructure_dir: str, stop_event: Event):
        return run_pool_with_stop(
            rank_worker,
            args=(top_k, substructure_dir, True, True),
            stop_event=stop_event,
        )

    def stop_ranking(self) -> None:
        self.rank_stop_event.set()
        self._set_button_state(
            self.btn_rank_run,
            text=None,
            callback=None,
            qss=self.STOP_BUTTON_QSS,
            enabled=False,
        )
        self.txt_status_message.setPlainText("Stopping ranking...")
        self.set_state("normal")

    def on_ranking_finished(self, result) -> None:
        if result == -1:
            self.rank_stop_event.clear()
            self.txt_status_message.setPlainText("Ranking stopped!")
            self.set_state("normal")
            self._set_task_idle(self.btn_rank_run, self.start_ranking, idle_text=None)
            return

        n_col = 6
        size = 145
        spacing = 8.5
        files = sorted(
            os.listdir(result),
            key=lambda x: int(os.path.splitext(x)[0]) if x.split(".")[0].isdigit() else 1 << 30,
        )
        for index, filename in enumerate(files):
            row = index // n_col
            col = index % n_col
            original = QPixmap(os.path.join(result, filename))
            if original.isNull():
                print(f"Failed to load: {os.path.join(result, filename)}")
                continue
            x = col * (size + spacing)
            y = row * (size + spacing)
            thumbnail = self.get_thumbnail(original, QSize(size, size))
            item = ClickablePixmapItem(original, thumbnail, callback=self.on_thumbnail_clicked)
            item.setPos(x, y)
            self.ranking_scene.addItem(item)
        self.txt_status_message.setPlainText("Ranking completed.")
        self.set_state("normal")
        self._set_task_idle(self.btn_rank_run, self.start_ranking, idle_text=None)

    def reset_spectrum_content(self) -> None:
        """Tear down the matplotlib canvas and its event connections."""
        if self.spectrum_plotter is not None:
            try:
                plotter = self.spectrum_plotter
                figure = getattr(plotter, "_fig", None)
                canvas = figure.canvas if figure is not None else None
                if canvas is not None:
                    for attr in (
                        "_cid_move", "_cid_scroll", "_cid_press",
                        "_cid_release", "_cid_dblclick", "_cid_draw",
                    ):
                        cid = getattr(plotter, attr, None)
                        if cid is not None:
                            try:
                                canvas.mpl_disconnect(cid)
                            except Exception:
                                pass
                            setattr(plotter, attr, None)
            except Exception:
                pass
            self.spectrum_plotter = None

        if self.spectrum_canvas is not None:
            try:
                self.spectrum_canvas.setParent(None)
                self.spectrum_canvas.deleteLater()
            except Exception:
                pass
            self.spectrum_canvas = None
        self.spectrum_fig = None

    def ensure_spectrum_canvas(self) -> bool:
        """Create the matplotlib spectrum canvas on first use.

        The .ui placeholder 'QGraphicsView' is replaced in-place by a
        'QFrame' hosting 'FigureCanvasQTAgg'; subsequent calls reuse the
        frame and only reset the plot contents.
        """
        if self.load_acin_substructure_weights() is None:
            return False
        if self.spectrum_frame is not None:
            self.reset_spectrum_content()

        self.spectrum_fig = Figure(figsize=(10, 1.7), facecolor="none")
        self.spectrum_canvas = FigureCanvasQTAgg(self.spectrum_fig)
        self.spectrum_canvas.setAttribute(Qt.WA_TranslucentBackground, True)
        self.spectrum_canvas.setStyleSheet("background: transparent;")
        self.spectrum_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.spectrum_canvas.setMinimumHeight(180)

        if self.spectrum_frame is None:
            if self.gvw_rank_spectrum is None:
                self.txt_status_message.setPlainText(
                    "Warning: spectrum view has already been replaced."
                )
                self.set_state("warning")
                return False
            old_view = self.gvw_rank_spectrum
            parent_widget = old_view.parentWidget()

            self.spectrum_frame = QFrame(parent_widget)
            self.spectrum_frame.setObjectName("spectrumFrame")
            self.spectrum_frame.setAttribute(Qt.WA_StyledBackground, True)
            self.spectrum_frame.setStyleSheet(
                "QFrame#spectrumFrame { background-color: #FFFFFF; "
                "border: 1px solid transparent; "
                "border-top-left-radius: 0px; border-top-right-radius: 0px; "
                "border-bottom-right-radius: 18px; border-bottom-left-radius: 18px; }"
            )
            frame_layout = QVBoxLayout(self.spectrum_frame)
            frame_layout.setContentsMargins(0, 0, 0, 0)
            frame_layout.setSpacing(0)
            frame_layout.addWidget(self.spectrum_canvas)

            if parent_widget is not None and parent_widget.layout() is not None:
                parent_widget.layout().replaceWidget(old_view, self.spectrum_frame)
            else:
                self.spectrum_frame.setGeometry(old_view.geometry())
                self.spectrum_frame.show()
            old_view.deleteLater()

            self.spectrum_frame.setSizePolicy(old_view.sizePolicy())
            self.spectrum_frame.setFocusPolicy(Qt.StrongFocus)
            self.gvw_rank_spectrum = None
        else:
            frame_layout = self.spectrum_frame.layout()
            if frame_layout is not None:
                frame_layout.addWidget(self.spectrum_canvas)
            else:
                self.spectrum_canvas.setParent(self.spectrum_frame)
                self.spectrum_canvas.show()

        try:
            self.spectrum_plotter = ACINSpectrumBar(
                csv_path=self.acin_substructure_weights,
                out_path=None,
                show_plot=False,
                fig=self.spectrum_fig,
                USE_BLIT=True,
                FONT_SCALE=0.1,
                FONT_WEIGHT="normal",
                TITLE_FONT_WEIGHT="normal",
                XLABEL_FONT_WEIGHT="normal",
                SUBPLOT_ADJUST={"left": 0.01, "right": 0.98, "top": 0.93, "bottom": 0.14},
                TRANSPARENT_BG=True,
                BG_COLOR=None,
                AXES_BG_COLOR=None,
                CBAR_BG_COLOR=None,
                AX_SPINE_LW=0.3,
                CB_OUTLINE_LW=0.3,
                AX_TICK_WIDTH=0.3,
                CB_TICK_WIDTH=0.3,
                CB_TICK_PAD=4,
                TITLE_TEXT="",
                XLABEL_TEXT="Substructure index",
                XLABEL_PAD=0,
                AX_TICK_PAD=3,
                CBAR_LABEL="Weight",
                CBAR_LABEL_PAD=-3,
                SHOW_XTICKLABELS=True,
                AX_NBINS=6,
                CB_FRACTION=0.025,
                CB_ASPECT=10,
                HOVER_THEME="dark",
                HOVER_FONT_WEIGHT="normal",
                HOVER_TEXT_FORMAT="line",
                HOVER_OFFSET=(5, 5),
                HOVER_FONTSIZE=8,
                HOVER_VLINE_STYLE="--",
                HOVER_VLINE_LW=0.2,
                HOVER_VLINE_ALPHA=0.5,
            )
            idx, weight = self.spectrum_plotter.load_weights()
            self.spectrum_plotter.plot_bar(idx, weight)
        except Exception as exc:
            traceback.print_exc()
            self.txt_status_message.setPlainText(
                f"Warning: failed to plot spectrum — {exc}"
            )
            self.set_state("warning")
            if self.spectrum_canvas is not None:
                try:
                    self.spectrum_canvas.setParent(None)
                    self.spectrum_canvas.deleteLater()
                except Exception:
                    pass
            self.spectrum_canvas = None
            self.spectrum_plotter = None
            self.spectrum_fig = None
            return False

        self.spectrum_canvas.draw()
        return True

    # ==================================================================
    # SHAP evaluation
    # ==================================================================

    def start_analysis(self) -> None:
        self.analysis_stop_event.clear()
        self.beeswarm_scene.clear()
        self.dependency_scene.clear()

        analysis_model = self.cmb_eval_shap_model.currentText()
        if not analysis_model:
            self._set_task_idle(self.btn_eval_analyse, self.start_analysis, idle_text="ANALYSE")
            self.txt_status_message.setPlainText(
                "Please select a SHAP model in the dropdown first!"
            )
            self.set_state("warning")
            return
        if not self.check_ml_inputs("run model analysis"):
            return

        self.txt_status_message.setPlainText("Analysing...")
        self.set_state("normal")
        self._set_task_running(self.btn_eval_analyse, self.stop_analysis)

        models_path = os.path.join(self.get_base_dir(), "ml_models")
        train_size = self.dsp_ml_split.value()
        random_state = self.spn_ml_seed.value()
        if analysis_model.lower() == "optimal":
            analysis_model = self.ml_optimal_model
        fitting_model = self.btn_group_eval_fitting_type.checkedButton().text().lower()
        base_dir = self.plots_dir

        self.analysis_thread_obj = WorkerThread(
            lambda: self._create_analysis_process(
                models_path,
                train_size,
                random_state,
                analysis_model,
                fitting_model,
                base_dir,
                self.analysis_stop_event,
            )
        )
        self.analysis_thread_obj.results.connect(self.on_analysis_finished)
        self.analysis_thread_obj.finished.connect(self.analysis_thread_obj.deleteLater)
        self.analysis_thread_obj.start()

    @staticmethod
    def _create_analysis_process(
        models_path,
        train_size,
        random_state,
        analysis_model,
        fitting_model,
        base_dir,
        stop_event,
    ):
        return run_pool_with_stop(
            analyse_worker,
            args=(models_path, train_size, random_state, analysis_model, fitting_model),
            kwds={"dpi_beeswarm": 500, "dpi_deps": 300, "base_dir": base_dir},
            stop_event=stop_event,
        )

    def stop_analysis(self) -> None:
        self.analysis_stop_event.set()
        self._set_button_state(
            self.btn_eval_analyse,
            text="Stopping",
            callback=None,
            qss=self.STOP_BUTTON_QSS,
            enabled=False,
        )
        self.txt_status_message.setPlainText("Stopping analysis...")
        self.set_state("normal")

    def on_analysis_finished(self, result) -> None:
        if result == -1:
            self.analysis_stop_event.clear()
            self.txt_status_message.setPlainText("Analysis stopped!")
            self.set_state("normal")
            self._set_task_idle(self.btn_eval_analyse, self.start_analysis, idle_text="ANALYSE")
            return

        self.shap_beeswarm_img_path = result["path"]
        self.shap_dependency_img_dir = result["dir"]
        self.shap_descriptor_values_df = result["shap_values"]

        self.txt_status_message.setPlainText("Analysis completed.")
        self.set_state("normal")
        self._set_task_idle(self.btn_eval_analyse, self.start_analysis, idle_text="ANALYSE")

        original = QPixmap(self.shap_beeswarm_img_path)
        thumbnail = self.get_thumbnail(
            original, self.gvw_eval_beeswarm.viewport().size()
        )
        item = ClickablePixmapItem(original, thumbnail, callback=self.on_thumbnail_clicked)
        self.beeswarm_scene.addItem(item)

        dependency_paths = [
            os.path.join(self.shap_dependency_img_dir, name)
            for name in os.listdir(self.shap_dependency_img_dir)
        ]
        n_col = 2
        size = 250
        spacing_col = 20
        spacing_row = -20
        for index, path in enumerate(dependency_paths):
            row = index // n_col
            col = index % n_col
            original = QPixmap(path)
            x = col * (size + spacing_col)
            y = row * (size + spacing_row)
            thumbnail = self.get_thumbnail(original, QSize(size, size))
            item = ClickablePixmapItem(original, thumbnail, callback=self.on_thumbnail_clicked)
            item.setPos(x, y)
            self.dependency_scene.addItem(item)

    # ==================================================================
    # Mixture-property prediction
    # ==================================================================

    def on_sam_smiles_changed(self) -> None:
        """Auto-lock the ratio to 1.0 when both SAMs are identical."""
        smiles_1 = self.txt_pred_sam_1.toPlainText().strip()
        smiles_2 = self.txt_pred_sam_2.toPlainText().strip()
        if smiles_1 == smiles_2:
            self.dsp_pred_ratio.setEnabled(False)
            self.dsp_pred_ratio.setValue(1.0)
        else:
            self.dsp_pred_ratio.setEnabled(True)

    def start_prediction(self) -> None:
        self.prediction_stop_event.clear()
        self.waterfall_scene.clear()
        self.clear_layout(self.lay_pred_mol_display)
        self.canvas_sam_1 = None
        self.canvas_sam_2 = None

        pred_model_sel = self.cmb_pred_shap_model.currentText()
        if not pred_model_sel:
            self._set_task_idle(self.btn_pred_run, self.start_prediction, idle_text="PREDICT")
            self.txt_status_message.setPlainText(
                "Please select a SHAP model in the dropdown first!"
            )
            self.set_state("warning")
            return
        if not self.check_ml_inputs("run prediction"):
            return

        smiles_sam_1 = self.txt_pred_sam_1.toPlainText().strip()
        smiles_sam_2 = self.txt_pred_sam_2.toPlainText().strip()
        ok_1, err_1 = self.is_valid_smiles(smiles_sam_1)
        ok_2, err_2 = self.is_valid_smiles(smiles_sam_2)
        if not ok_1 or not ok_2:
            self._set_task_idle(self.btn_pred_run, self.start_prediction, idle_text="PREDICT")
            messages = []
            if not ok_1:
                messages.append(f"SAM-1: {err_1}")
            if not ok_2:
                messages.append(f"SAM-2: {err_2}")
            self.txt_status_message.setPlainText("Invalid SMILES — " + "; ".join(messages))
            self.set_state("warning")
            return

        metadata = self._load_smiles_metadata()
        missing: list[str] = []
        if smiles_sam_1 and smiles_sam_1 not in metadata:
            missing.append(smiles_sam_1)
        if smiles_sam_2 and smiles_sam_2 != smiles_sam_1 and smiles_sam_2 not in metadata:
            missing.append(smiles_sam_2)
        if missing:
            filled, append_flag = self.prompt_missing_smiles_metadata(missing)
            if filled is None:
                self.txt_status_message.setPlainText(
                    "Prediction cancelled: SMILES metadata not provided."
                )
                self.set_state("warning")
                return
            metadata.update(filled)
            if append_flag:
                rows = [
                    [
                        smiles,
                        metadata[smiles]["Symmetric"],
                        metadata[smiles]["Chain_length"],
                        metadata[smiles]["HeteroEndFrag"],
                    ]
                    for smiles in filled
                ]
                try:
                    self._append_smiles_metadata_csv(rows)
                except Exception as exc:
                    self.txt_status_message.setPlainText(
                        f"Failed to append to SMILES.csv: {exc}"
                    )
                    self.set_state("warning")

        if smiles_sam_1 in metadata:
            m1 = metadata[smiles_sam_1]
            symmetric_1 = m1["Symmetric"]
            hetero_end_frag_1 = m1["HeteroEndFrag"]
            alkyl_chain_length_1 = m1["Chain_length"]
        else:
            symmetric_1, hetero_end_frag_1, alkyl_chain_length_1 = 0, 0, 1

        if smiles_sam_2 in metadata:
            m2 = metadata[smiles_sam_2]
            symmetric_2 = m2["Symmetric"]
            hetero_end_frag_2 = m2["HeteroEndFrag"]
            alkyl_chain_length_2 = m2["Chain_length"]
        else:
            symmetric_2, hetero_end_frag_2, alkyl_chain_length_2 = 0, 0, 1

        mol_prop_csv_path = os.path.join(self.mol_calc_dir, "mol_prop.csv")
        if not os.path.isfile(mol_prop_csv_path):
            self.txt_status_message.setPlainText(
                f"mol_prop.csv not found at: {mol_prop_csv_path}"
            )
            self.set_state("error")
            return
        try:
            mol_prop_df = pd.read_csv(mol_prop_csv_path)
        except Exception as exc:
            self.txt_status_message.setPlainText(f"Failed to read mol_prop.csv: {exc}")
            self.set_state("error")
            return

        property_smiles = set(mol_prop_df["SMILES"].astype(str).str.strip())
        missing_property: list[tuple[str, str]] = []
        if smiles_sam_1 not in property_smiles:
            missing_property.append(("SAM-1", smiles_sam_1))
        if smiles_sam_2 not in property_smiles:
            missing_property.append(("SAM-2", smiles_sam_2))
        if missing_property:
            details_html = (
                "<div style='font-family: Consolas, monospace; font-size: 12px; "
                "color: #a8d0ff; background-color: #1e2a44; padding: 8px; "
                "border-radius: 6px;'>"
            )
            for tag, smiles in missing_property:
                chunks = [smiles[i:i + 50] for i in range(0, len(smiles), 50)]
                smiles_display = "<br>&nbsp;&nbsp;&nbsp;&nbsp;".join(chunks)
                details_html += (
                    f"<b style='color:#e8eef8;'>{tag}:</b>"
                    f"<br>&nbsp;&nbsp;&nbsp;&nbsp;{smiles_display}<br><br>"
                )
            details_html += "</div>"
            self._show_info_dialog(
                "Missing Properties",
                "<div style=\"font-family: 'Consolas'; font-size: 13px; "
                "color: #b8c2d6; line-height: 1.5;\">"
                "The following SMILES are not in <b style='color:#e8eef8;'>mol_prop.csv</b>.<br>"
                "Please run <i>Molecular Calculation</i> for them first:"
                "</div>",
                window_title="SMILES not in mol_prop.csv",
                details_html=details_html,
                buttons=("Ok",),
                min_width=520,
            )
            self.txt_status_message.setPlainText(
                "Prediction aborted: some SMILES have no computed properties."
            )
            self.set_state("warning")
            return

        self.txt_status_message.setPlainText("Prediction is running...")
        self.set_state("normal")
        self._set_task_running(self.btn_pred_run, self.stop_prediction)
        self.txt_pred_sam_1.setEnabled(False)
        self.txt_pred_sam_2.setEnabled(False)

        mixture = 0 if smiles_sam_1 == smiles_sam_2 else 1
        bandgap = self.dsp_pred_bandgap.value()
        additive = (
            1
            if self.cmb_pred_additive.currentText().strip().lower() == "true"
            else 0
        )
        ratio = self.dsp_pred_ratio.value()

        smiles_dir = {"SMILES": [], "Symmetric": [], "Chain_length": [], "HeteroEndFrag": []}
        experiment_condition = {
            "SAM_1": [], "SAM_2": [], "SAMs_ratio": [],
            "Bandgap": [], "Mixture": [], "MACl_additive": [],
        }
        smiles_dir["SMILES"].extend([smiles_sam_1, smiles_sam_2])
        smiles_dir["Symmetric"].extend([symmetric_1, symmetric_2])
        smiles_dir["Chain_length"].extend([alkyl_chain_length_1, alkyl_chain_length_2])
        smiles_dir["HeteroEndFrag"].extend([hetero_end_frag_1, hetero_end_frag_2])
        experiment_condition["SAM_1"].append(smiles_sam_1)
        experiment_condition["SAM_2"].append(smiles_sam_2)
        experiment_condition["SAMs_ratio"].append(ratio)
        experiment_condition["Bandgap"].append(bandgap)
        experiment_condition["Mixture"].append(mixture)
        experiment_condition["MACl_additive"].append(additive)

        smiles_df = pd.DataFrame.from_dict(smiles_dir)
        exp_cond_df = pd.DataFrame.from_dict(experiment_condition)
        data, _ = MolCalc.transform(
            experiment_condition=exp_cond_df,
            smiles=smiles_df,
            mol_prop=mol_prop_df,
        )

        models_path = os.path.join(self.get_base_dir(), "ml_models")
        train_size = self.dsp_ml_split.value()
        random_state = self.spn_ml_seed.value()
        base_dir = self.plots_dir

        self.prediction_thread_obj = WorkerThread(
            lambda: self._create_predict_process(
                models_path,
                train_size,
                random_state,
                pred_model_sel,
                data,
                base_dir,
                self.prediction_stop_event,
            )
        )
        self.prediction_thread_obj.results.connect(self.on_prediction_finished)
        self.prediction_thread_obj.finished.connect(self.prediction_thread_obj.deleteLater)
        self.prediction_thread_obj.finished.connect(
            lambda: (
                self.txt_pred_sam_1.setEnabled(True),
                self.txt_pred_sam_2.setEnabled(True),
            )
        )
        self.prediction_thread_obj.start()

    @staticmethod
    def _create_predict_process(
        models_path,
        train_size,
        random_state,
        pred_model_sel,
        data,
        base_dir,
        stop_event,
    ):
        return run_pool_with_stop(
            predict_worker,
            args=(models_path, train_size, random_state, pred_model_sel, data, base_dir),
            stop_event=stop_event,
        )

    def stop_prediction(self) -> None:
        self.prediction_stop_event.set()
        self._set_button_state(
            self.btn_pred_run,
            text="Stopping",
            callback=None,
            qss=self.STOP_BUTTON_QSS,
            enabled=False,
        )
        self.txt_status_message.setPlainText("Stopping prediction...")
        self.set_state("normal")

    def on_prediction_finished(self, result) -> None:
        try:
            if result == -1:
                self.prediction_stop_event.clear()
                self.txt_status_message.setPlainText("Prediction stopped!")
                self.set_state("normal")
                self._set_task_idle(self.btn_pred_run, self.start_prediction, idle_text="PREDICT")
                return

            original = QPixmap(result["path"])
            thumbnail = self.get_thumbnail(
                original, self.gvw_pred_waterfall.viewport().size()
            )
            item = ClickablePixmapItem(original, thumbnail, callback=self.on_thumbnail_clicked)
            self.waterfall_scene.addItem(item)

            smiles_sam_1 = self.txt_pred_sam_1.toPlainText().strip()
            smiles_sam_2 = self.txt_pred_sam_2.toPlainText().strip()
            if smiles_sam_1 == smiles_sam_2:
                self.canvas_sam_1 = Molecule3DCanvas(
                    smiles_sam_1,
                    n_frames=180,
                    fps=30,
                    title="Single-component SAM",
                )
                self.lay_pred_mol_display.addStretch(1)
                self.lay_pred_mol_display.addWidget(self.canvas_sam_1, 2)
                self.lay_pred_mol_display.addStretch(1)
            else:
                self.canvas_sam_1 = Molecule3DCanvas(
                    smiles_sam_1, n_frames=180, fps=30, title="SAM-1"
                )
                self.canvas_sam_2 = Molecule3DCanvas(
                    smiles_sam_2, n_frames=180, fps=30, title="SAM-2"
                )
                self.lay_pred_mol_display.addWidget(self.canvas_sam_1, 1)
                self.lay_pred_mol_display.addWidget(self.canvas_sam_2, 1)

            self.txt_status_message.setPlainText("Prediction completed.")
            self.set_state("normal")
            self._set_task_idle(self.btn_pred_run, self.start_prediction, idle_text="PREDICT")
        finally:
            self.txt_pred_sam_1.setEnabled(True)
            self.txt_pred_sam_2.setEnabled(True)

    # ==================================================================
    # Thumbnail helpers
    # ==================================================================

    def on_thumbnail_clicked(self, item: ClickablePixmapItem) -> None:
        """Open a zoomable, scrollable dialog showing the full-resolution image."""
        pixmap = item.original
        dialog = QDialog(self)
        dialog.resize(800, 800)
        dialog.setStyleSheet(self.DIALOG_QSS)
        layout = QVBoxLayout(dialog)
        scene = QGraphicsScene()
        pixmap_item = scene.addPixmap(pixmap)
        scene.setSceneRect(pixmap_item.boundingRect())
        view = QGraphicsView(scene)
        view.setStyleSheet("QGraphicsView { background-color: #1e2a44; border: none; }")
        view.setDragMode(QGraphicsView.ScrollHandDrag)

        def wheelEvent(event) -> None:  # noqa: N802
            factor = 1.1 if event.angleDelta().y() >= 0 else 0.9
            view.scale(factor, factor)
            event.accept()

        view.wheelEvent = wheelEvent
        layout.addWidget(view)
        dialog.exec()

    def get_thumbnail(self, original: QPixmap, target_size: QSize, padding: int = 5) -> QPixmap:
        """Scale a pixmap while respecting the HiDPI device-pixel ratio."""
        dpr = self.devicePixelRatioF()
        padding_physical = int(padding * dpr)
        physical = QSize(
            int(target_size.width() * dpr - 2 * padding_physical),
            int(target_size.height() * dpr - 2 * padding_physical),
        )
        thumbnail = original.scaled(physical, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        thumbnail.setDevicePixelRatio(dpr)
        return thumbnail

    def clear_layout(self, layout) -> None:
        """Recursively empty a layout, stopping any molecule-canvas animations."""
        while layout.count() > 0:
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                if isinstance(widget, Molecule3DCanvas):
                    try:
                        widget.stop_animation()
                    except Exception:
                        pass
                widget.setParent(None)
                widget.deleteLater()
            elif item.layout() is not None:
                self.clear_layout(item.layout())

    # ==================================================================
    # Cache management
    # ==================================================================

    def clear_cache(self) -> None:
        """Show a checkbox dialog and delete the selected cache directories."""
        base_dir = self.get_base_dir()
        items = [
            ("ml_models", True, "Trained ML models           (ml_models/)"),
            ("dl_models", True, "Trained DL models           (dl_models/)"),
            ("shap_values", True, "SHAP value cache            (plots/shap_values/)"),
            (
                "plots_other",
                True,
                "Other plot outputs          (plots/beeswarm, dependencies, waterfall, substructures...)",
            ),
            ("temp", True, "Temporary files             (temp/)"),
            (
                "attention_weights",
                False,
                "Attention weights           (attention_weights.csv)",
            ),
        ]

        dialog = QDialog(self)
        dialog.setWindowTitle("Clear Cache")
        dialog.setMinimumWidth(660)
        dialog.setStyleSheet(self.DIALOG_QSS)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(10)
        icon_label = QLabel()
        icon_label.setFixedSize(24, 24)
        icon_label.setPixmap(
            self.style().standardIcon(QStyle.SP_MessageBoxWarning).pixmap(24, 24)
        )
        header.addWidget(icon_label, 0, Qt.AlignVCenter)
        title_label = QLabel("Clear Cache")
        title_label.setStyleSheet(
            "font-family: 'Consolas'; font-size: 15px; font-weight: bold; color: #e8eef8;"
        )
        header.addWidget(title_label, 1, Qt.AlignVCenter)
        layout.addLayout(header)

        body_label = QLabel(
            "<div style=\"font-family: 'Consolas'; font-size: 13px; color: #b8c2d6;\">"
            "Select the caches you want to permanently delete:"
            "</div>"
        )
        body_label.setTextFormat(Qt.RichText)
        body_label.setWordWrap(True)
        layout.addWidget(body_label)

        toggle_row = QHBoxLayout()
        toggle_row.setContentsMargins(0, 0, 0, 0)
        toggle_row.setSpacing(8)
        select_all_button = make_dialog_button("Select All")
        deselect_all_button = make_dialog_button("Deselect All")
        toggle_row.addWidget(select_all_button)
        toggle_row.addWidget(deselect_all_button)
        toggle_row.addStretch(1)
        layout.addLayout(toggle_row)

        checkboxes: dict[str, QCheckBox] = {}
        for key, default, label in items:
            checkbox = QCheckBox(label)
            checkbox.setChecked(default)
            checkbox.setStyleSheet(
                "QCheckBox { color: #dcdfe4; font-family: 'Consolas'; "
                "font-size: 13px; spacing: 8px; }"
            )
            checkboxes[key] = checkbox
            layout.addWidget(checkbox)

        def set_all(state: bool) -> None:
            for checkbox in checkboxes.values():
                checkbox.setChecked(state)

        select_all_button.clicked.connect(lambda: set_all(True))
        deselect_all_button.clicked.connect(lambda: set_all(False))

        layout.addSpacing(4)
        warn_label = QLabel(
            "<div style=\"font-family: 'Consolas'; font-size: 13px; "
            "color: #f0a0a0; font-weight: bold;\">"
            "This action cannot be undone.</div>"
        )
        warn_label.setTextFormat(Qt.RichText)
        layout.addWidget(warn_label)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        cancel_button = make_dialog_button("Cancel")
        clear_button = make_dialog_button("Clear Selected", default=True)
        cancel_button.clicked.connect(dialog.reject)
        clear_button.clicked.connect(dialog.accept)
        button_row.addWidget(cancel_button)
        button_row.addWidget(clear_button)
        layout.addLayout(button_row)

        if dialog.exec() != QDialog.Accepted:
            return

        selected = [key for key, checkbox in checkboxes.items() if checkbox.isChecked()]
        if not selected:
            self.txt_status_message.setPlainText("No cache selected.")
            self.set_state("warning")
            return

        self.txt_status_message.setPlainText("Cache is clearing...")
        self.set_state("normal")

        errors: list[str] = []

        def remove_path(path: str) -> None:
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.isfile(path):
                    os.remove(path)
            except FileNotFoundError:
                pass
            except Exception as exc:
                errors.append(f"{path}: {exc}")

        for key in selected:
            if key == "plots_other":
                plots_root = os.path.join(base_dir, "plots")
                if os.path.isdir(plots_root):
                    for name in os.listdir(plots_root):
                        if name == "shap_values":
                            continue
                        remove_path(os.path.join(plots_root, name))
            elif key == "shap_values":
                remove_path(os.path.join(base_dir, "plots", "shap_values"))
            elif key == "attention_weights":
                remove_path(os.path.join(base_dir, "attention_weights.csv"))
            else:
                remove_path(os.path.join(base_dir, key))

        if errors:
            self.txt_status_message.setPlainText(
                "Cache clear partial failed:\n" + "\n".join(errors)
            )
            self.set_state("error")
        else:
            self.txt_status_message.setPlainText(
                "Cache cleared: " + ", ".join(selected)
            )
            self.set_state("normal")

    # ==================================================================
    # Generic information dialog
    # ==================================================================

    def _show_info_dialog(
        self,
        title: str,
        body_html: str,
        *,
        window_title: Optional[str] = None,
        details_html: Optional[str] = None,
        ask_html: Optional[str] = None,
        icon=QStyle.SP_MessageBoxWarning,
        buttons: tuple[str, ...] = ("Ok",),
        default_button: str = "Ok",
        min_width: int = 520,
    ) -> int:
        """Show a themed information or confirmation dialog.

        Returns 'QDialog.Accepted' when the last button is pressed,
        'QDialog.Rejected' otherwise.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle(window_title or title)
        dialog.setMinimumWidth(min_width)
        dialog.setStyleSheet(self.DIALOG_QSS)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(10)
        icon_label = QLabel()
        icon_label.setFixedSize(24, 24)
        icon_label.setPixmap(self.style().standardIcon(icon).pixmap(24, 24))
        header.addWidget(icon_label, 0, Qt.AlignVCenter)

        title_label = QLabel(title)
        title_label.setStyleSheet(
            "font-family: 'Consolas'; font-size: 15px; font-weight: bold; color: #e8eef8;"
        )
        header.addWidget(title_label, 1, Qt.AlignVCenter)
        layout.addLayout(header)

        body_label = QLabel(body_html)
        body_label.setTextFormat(Qt.RichText)
        body_label.setWordWrap(True)
        layout.addWidget(body_label)

        if details_html:
            details_label = QLabel(details_html)
            details_label.setTextFormat(Qt.RichText)
            details_label.setWordWrap(True)
            layout.addWidget(details_label)

        if ask_html:
            ask_label = QLabel(ask_html)
            ask_label.setTextFormat(Qt.RichText)
            ask_label.setWordWrap(True)
            layout.addWidget(ask_label)

        button_row = QHBoxLayout()
        button_row.addStretch()
        multi = len(buttons) > 1
        for index, text in enumerate(buttons):
            button = make_dialog_button(text, default=(text == default_button))
            if multi and index == 0:
                button.clicked.connect(dialog.reject)
            else:
                button.clicked.connect(dialog.accept)
            button_row.addWidget(button)
        layout.addLayout(button_row)

        return dialog.exec()

    # ==================================================================
    # Qt event filter
    # ==================================================================
    # 'eventFilter' is a Qt protocol method; it must not be renamed.

    def eventFilter(self, obj, event) -> bool:
        """Centralised event handling for widgets that need custom behaviour."""
        if obj is getattr(self, "frm_navigation", None):
            if event.type() == QEvent.Type.HoverEnter:
                self._start_animation(
                    anim=self.navigation_animation,
                    obj=obj,
                    end_width=self.navigation_expanded_width,
                )
                return True
            if event.type() == QEvent.Type.HoverLeave:
                self._start_animation(
                    anim=self.navigation_animation,
                    obj=obj,
                    end_width=self.navigation_collapsed_width,
                )
                return True

        if obj is getattr(self, "txt_agent_input", None) and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                if event.modifiers() & Qt.ShiftModifier:
                    return False
                self.send_agent_message()
                return True

        # Rebuild the model list on click so newly trained models appear
        # without restarting the app.
        model_combos = (
            getattr(self, "cmb_eval_shap_model", None),
            getattr(self, "cmb_agent_shap_model", None),
            getattr(self, "cmb_pred_shap_model", None),
        )
        if obj is not None and obj in model_combos:
            if event.type() == QEvent.Type.MouseButtonPress:
                obj.blockSignals(True)
                obj.clear()
                if getattr(self, "ml_optimal_model", None) is not None:
                    obj.addItem("Optimal")
                models_dir = os.path.join(self.get_base_dir(), "ml_models")
                if os.path.isdir(models_dir):
                    obj.addItems([
                        name[:-4] for name in os.listdir(models_dir) if name.endswith(".pkl")
                    ])
                obj.setCurrentIndex(-1)
                obj.blockSignals(False)
                obj.lineEdit().clear()
        return super().eventFilter(obj, event)

    @staticmethod
    def _start_animation(anim: QVariantAnimation, obj, end_width: int) -> None:
        anim.stop()
        anim.setStartValue(obj.width())
        anim.setEndValue(end_width)
        anim.start()

    def _find_child_widget(self, widget_type, name: str):
        """Locate a named child widget in the loaded .ui, warning on a miss."""
        try:
            widget = self.central_widget.findChild(widget_type, name)
        except AttributeError:
            print(f"FATAL: central_widget is None when searching widget {name}")
            return None
        if widget is None:
            print(f"WARNING: Cannot find child widget: {name}.")
        return widget

    @staticmethod
    def get_base_dir() -> str:
        """Return the runtime base directory, honouring PyInstaller freeze."""
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            return os.path.dirname(os.path.realpath(sys.executable))
        return os.path.dirname(os.path.realpath(os.path.abspath(__file__)))


# ===========================================================================
# Module-level workers
# ===========================================================================


def merge_worker(exp_df, smiles_df, mol_prop_df, base_dir):
    """Merge experiment and SMILES data, writing ml_inputs.csv / dl_inputs.csv."""
    try:
        exp_df = exp_df.copy()
        smiles_df = smiles_df.copy()
        mol_prop_df = None if mol_prop_df is None else mol_prop_df.copy()

        ml_inputs, dl_inputs = MolCalc.transform(
            experiment_condition=exp_df,
            smiles=smiles_df,
            mol_prop=mol_prop_df,
        )
        ml_inputs.to_csv(os.path.join(base_dir, "ml_inputs.csv"), index=False)
        dl_inputs.to_csv(os.path.join(base_dir, "dl_inputs.csv"), index=False)
        return {"ml_inputs": ml_inputs, "dl_inputs": dl_inputs, "error": None}
    except Exception as exc:
        traceback.print_exc()
        return {"error": str(exc)}


def mol_calc_worker(
    smiles_df, output_dir, method, basis_set, opt, freq, tightscf,
    ncore, cpu_usage, multithreading, seed, message_queue,
):
    """Drive a MolCalc run from inside a child process."""
    os.chdir(output_dir)
    calculator = MolCalc(
        smiles=smiles_df,
        method=method,
        basis_set=basis_set,
        opt=opt,
        freq=freq,
        tightscf=tightscf,
        ncore=ncore,
        multithreading=multithreading,
        cpu_usage=cpu_usage,
        random_seed=seed,
        mol_calc_queue=message_queue,
    )
    calculator.mol_calc(output_path=output_dir)


def ml_worker(base_dir, message_queue, x_train, x_test, y_train, y_test,
              selection_flags, init_points, iterations, seed):
    """Drive an AutoML run from inside a child process."""
    sys.stdout = QueueWriter(message_queue)
    models = AutoML(
        x_train, x_test, y_train, y_test, base_dir,
        queue=message_queue,
        params_lr=selection_flags[0], params_ridge=selection_flags[1],
        params_svr=selection_flags[2], params_knn=selection_flags[3],
        params_dtr=selection_flags[4], params_rf=selection_flags[5],
        params_et=selection_flags[6], params_gbdt=selection_flags[7],
        params_ada=selection_flags[8], params_xgb=selection_flags[9],
        params_lgbm=selection_flags[10], params_cat=selection_flags[11],
        n_jobs=-1,
        optzr_init_points=init_points, optzr_n_iter=iterations,
        random_state=seed,
    )
    models.train()
    return


def dl_worker(base_dir, message_queue, x_train, x_test, y_train, y_test,
              epoch, early_stop_threshold, device, batch, lr, dropout, random_state):
    """Drive an ACIN attention-training run from inside a child process.

    Determinism is enforced at four levels (Python, NumPy, PyTorch CPU, PyTorch
    CUDA) and cuDNN is pinned to deterministic algorithms.  This costs some
    throughput but makes the attention weights written to
    'attention_weights.csv' reproducible across runs.
    """
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    sys.stdout = QueueWriter(message_queue)
    if device.strip() == "GPU":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device.strip() == "CPU":
        device = "cpu"
    else:
        device = "cpu"

    ACINTraining(
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
        device=device, lr=lr, epoch=epoch, dropout=dropout,
        batch_size=batch, early_stop_threshold=early_stop_threshold,
        queue=message_queue, base_dir=base_dir,
    )
    return


def run_pool_with_stop(func, args=(), kwds=None, stop_event=None, poll=0.05):
    """Execute 'func' in a single-worker pool, returning '-1' if stopped.

    The pool is terminated rather than drained: in-flight SHAP or ranking jobs
    may hold large arrays, and the user has explicitly requested a stop.
    """
    kwds = kwds or {}
    with Pool(processes=1) as pool:
        result = pool.apply_async(func, args=args, kwds=kwds)
        while not result.ready():
            if stop_event is not None and stop_event.wait(poll):
                pool.terminate()
                pool.join()
                return -1
        return result.get()


def rank_worker(top_k, substructure_dir, show_index_text, show_weight_text,
                dpi=100, text_format="KRFP "):
    """Render the top-K KRFP substructures as RDKit PNGs."""
    from acin_analysis import render_top_k_substructures

    weights = pd.read_csv("attention_weights.csv")
    render_top_k_substructures(
        weights,
        substructure_dir,
        top_k=top_k,
        dpi=dpi,
        show_index_text=show_index_text,
        show_weight_text=show_weight_text,
        text_format=text_format,
    )
    return substructure_dir


def analyse_worker(models_path, train_size, random_state, analysis_model,
                   fitting_model, descriptor_top_n=20, dpi_beeswarm=500,
                   dpi_deps=300, base_dir=None):
    """Generate SHAP beeswarm and dependency plots for one model."""
    dataset = pd.read_csv("ml_inputs.csv")
    x = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=train_size, shuffle=True, random_state=random_state
    )
    plots = ml_eval.ME(
        models_path, x, y, x_train, y_train, x_test, y_test,
        random_state=random_state, base_dir=base_dir,
    )
    beeswarm_path = plots.beeswarm(
        model_sel=analysis_model, max_display=descriptor_top_n, dpi=dpi_beeswarm
    )
    dependencies_dir = plots.deps(
        model_sel=analysis_model, fitting_model=fitting_model, dpi=dpi_deps
    )
    return {
        "path": beeswarm_path,
        "dir": dependencies_dir,
        "shap_values": plots.shap_values,
    }


def agent_worker(command_queue, message_queue, stop_event, api_key, base_url,
                 llm_model, temperature, max_tokens, threshold,
                 descriptor_top_n, thinking, base_dir):
    """Long-running MSECAgent event loop."""
    agent = MSECAgent(
        q=message_queue,
        stop_event=stop_event,
        api_key=api_key,
        base_url=base_url,
        model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        thresh=threshold,
        descriptor_top_n=descriptor_top_n,
        thinking=thinking,
    )

    while True:
        try:
            command, payload = command_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if stop_event.is_set():
            stop_event.clear()

        if command == "init":
            try:
                payload_list = list(payload)
                thinking_now = payload_list[-1]
                (
                    models_path,
                    train_size,
                    random_state,
                    agent_model_sel,
                    fitting_model,
                    substructure_top_k,
                    acin_substructure_weight,
                ) = payload_list[:7]

                agent.thinking = bool(thinking_now)

                temp_substructure_dir = os.path.join(base_dir, "substructures")
                temp_beeswarm_dir = os.path.join(base_dir, "beeswarm")
                temp_dependencies_dir = os.path.join(base_dir, "dependencies")
                os.makedirs(temp_substructure_dir, exist_ok=True)
                os.makedirs(temp_beeswarm_dir, exist_ok=True)
                os.makedirs(temp_dependencies_dir, exist_ok=True)

                pool = Pool(processes=2)
                substructure_result = pool.apply_async(
                    rank_worker,
                    args=(substructure_top_k, temp_substructure_dir, True, False, 50, "krfp_"),
                )
                analysis_result = pool.apply_async(
                    analyse_worker,
                    args=(models_path, train_size, random_state, agent_model_sel, fitting_model),
                    kwds={
                        "descriptor_top_n": descriptor_top_n,
                        "dpi_beeswarm": 50,
                        "dpi_deps": 50,
                        "base_dir": base_dir,
                    },
                )
                while True:
                    if stop_event.is_set():
                        pool.terminate()
                        pool.join()
                        message_queue.put("_End")
                        break
                    if substructure_result.ready() and analysis_result.ready():
                        break
                    time.sleep(0.05)

                substructure_dir = substructure_result.get()
                analysis = analysis_result.get()

                agent.run_pipeline(
                    shap_beeswarm_img_path=analysis["path"],
                    shap_dependency_img_path=analysis["dir"],
                    substructure_img_path=substructure_dir,
                    df_shap_descriptor_vec=analysis["shap_values"],
                    acin_substructure_weight=acin_substructure_weight.T,
                )

                shutil.rmtree(temp_substructure_dir)
                shutil.rmtree(temp_beeswarm_dir)
                shutil.rmtree(temp_dependencies_dir)
            except Exception as exc:
                traceback.print_exc()
                message_queue.put("_Error_" + f"Error: {exc}")
            finally:
                if stop_event.is_set():
                    stop_event.clear()
                message_queue.put("_End")

        elif command == "conv":
            try:
                text, thinking_now = payload
                agent.thinking = bool(thinking_now)
                agent.multi_turn_conv(text)
            except Exception as exc:
                traceback.print_exc()
                message_queue.put("_Error_" + f"Error: {exc}")
            finally:
                message_queue.put("_End")


def predict_worker(models_path, train_size, random_state, prediction_model,
                   data, base_dir):
    """Predict PCE from a mixture-condition DataFrame and produce its waterfall plot."""
    dataset = pd.read_csv("ml_inputs.csv")
    x = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=train_size, shuffle=True, random_state=random_state
    )

    shap_model = joblib.load(os.path.join(models_path, f"{prediction_model}.pkl"))
    predicted_pce = shap_model.predict(data)

    plots = ml_eval.ME(
        models_path, x, y, x_train, y_train, x_test, y_test,
        random_state=random_state, base_dir=base_dir,
    )
    waterfall_path = plots.waterfall(prediction_model, data=data)
    return {"path": waterfall_path, "predicted_pce": predicted_pce}