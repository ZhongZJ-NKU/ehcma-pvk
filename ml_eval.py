"""Model-explainability suite for the Co-SAM pipeline.

'ME' wraps SHAP analysis behind a small stateful facade so the GUI and
background workers can request plots (beeswarm, dependence, waterfall) and
read shared SHAP values without knowing about the underlying explainer.

Input:
    models_path                          -- dir of fitted '*.pkl'.
    X, Y, x_train, y_train, x_test, y_test -- dataset + split.
    random_state, base_dir               -- SHAP seeding + output root.
    cust_colors                          -- optional colour stops.

Output ('ME' instance; plots written under 'base_dir'):
    .beeswarm(model_sel, ...)            -> str  (PNG path)
    .deps(model_sel, fitting_model, ...) -> str  (dir of per-feature PNGs)
    .waterfall(model_sel, data=..., ...) -> str  (PNG path)
    .shap_values                         -> DataFrame | None
"""

import os
import sys
import warnings
from math import pi
from typing import Iterable, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
from pygam import LinearGAM, s
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "serif",
        "font.serif": ["Arial"],
        "figure.figsize": (12, 8),
        "font.weight": "bold",
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 12,
        "xtick.direction": "in",
        "xtick.major.width": 2,
        "ytick.direction": "in",
        "ytick.labelsize": 10,
        "ytick.major.width": 2,
        "lines.linewidth": 1.2,
        "lines.markersize": 8,
        "axes.linewidth": 2,
        "grid.linewidth": 0.6,
        "grid.alpha": 1,
        "figure.dpi": 300.0,
    }
)


# ---------------------------------------------------------------------------
# Display names
# ---------------------------------------------------------------------------
_SHORT_NAME = {
    "LinearRegression":          "LR",
    "RidgeRegression":           "Ridge",
    "SupportVectorRegression":   "SVR",
    "KNeighborsRegressor":       "KNN",
    "DecisionTreeRegressor":     "DT",
    "RandomForestRegressor":     "RF",
    "ExtraTreesRegressor":       "ET",
    "GradientBoostingRegressor": "GBDT",
    "AdaBoostRegressor":         "AdaBoost",
    "XGBRegressor":              "XGBoost",
    "LGBMRegressor":             "LGBM",
    "CatBoostRegressor":         "CatBoost",
}


def _short(name: str) -> str:
    """Map a fitted-model name to its short tag (falls back to 'name')."""
    return _SHORT_NAME.get(name, name)


class ME:
    """Load fitted estimators and produce SHAP / diagnostic plots."""

    DEFAULT_COLORS: Tuple[str, ...] = (
        "#E5CBE1", "#B19CCB", "#917BBD", "#6656A6", "#3C3888",
    )

    def __init__(
        self,
        models_path: str,
        X: pd.DataFrame,
        Y: pd.Series,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        random_state: Optional[int] = None,
        base_dir: Optional[str] = None,
        cust_colors: Optional[List[str]] = None,
    ) -> None:
        self.models_path = models_path
        if base_dir is None:
            if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
                base_dir = os.path.dirname(os.path.realpath(sys.executable))
            else:
                base_dir = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))
            self.plots_path = os.path.join(base_dir, "plots")
        else:
            self.plots_path = base_dir
        os.makedirs(self.plots_path, exist_ok=True)

        self.model_dict = {
            name[:-4]: joblib.load(os.path.join(self.models_path, name))
            for name in sorted(os.listdir(self.models_path))
            if name.endswith(".pkl")
        }

        colors = cust_colors or list(self.DEFAULT_COLORS)
        self.cust_colors = colors
        self.cust_colors_map = LinearSegmentedColormap.from_list(
            "cust_cmp", colors, N=256
        )

        self.X = X
        self.Y = Y
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.random_state = random_state
        self.shap_values: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def corr(self) -> None:
        """Save a heatmap of pairwise feature correlations."""
        x_corr = self.X.corr()
        plt.figure(figsize=(15, 10))
        ax = sns.heatmap(
            x_corr, square=True, annot=True,
            annot_kws={"size": 5, "fontweight": "normal"},
            cmap="RdBu_r", cbar=True,
            cbar_kws={"shrink": 0.65, "pad": 0.02, "aspect": 25,
                      "ticks": [-1, -0.5, 0, 0.5, 1]},
            vmin=-1, vmax=1, center=0, linewidths=0.5, linecolor="White", fmt=".2f",
        )
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=8, width=0.5, length=6, color="gray")
        plt.xticks(fontsize=6, fontweight="bold")
        plt.yticks(fontsize=6, fontweight="bold")
        plt.tick_params(axis="both", width=1, direction="out", pad=1, length=5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path, "features_corr.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    def shap_config(
        self,
        model_sel: str,
        samples: Optional[object] = None,
        kernel_background: int = 100,
        path: Optional[str] = None,
        use_cache: bool = True,
        return_explainer: bool = True,
    ) -> Tuple[np.ndarray, object, Optional[object]]:
        """Return '(shap_values, samples, explainer)' for 'model_sel'."""
        if samples is None:
            samples = self.x_test
        elif isinstance(samples, int):
            samples = shap.sample(self.x_test, samples)

        model = self.model_dict[model_sel]
        shap_cache_dir = os.path.join(self.plots_path, "shap_values")
        os.makedirs(shap_cache_dir, exist_ok=True)
        cache_file = os.path.join(shap_cache_dir, f"{model_sel}_shap_values.csv")

        def build_explainer():
            try:
                return shap.TreeExplainer(model), "tree"
            except Exception:
                pass
            try:
                from sklearn.linear_model import (
                    BayesianRidge, ElasticNet, Lasso, Ridge, SGDRegressor,
                )
                if isinstance(model, (LinearRegression, Ridge, Lasso, ElasticNet,
                                      BayesianRidge, SGDRegressor)):
                    background = shap.utils.sample(
                        self.x_train,
                        min(kernel_background, len(self.x_train)),
                        random_state=self.random_state,
                    )
                    return shap.LinearExplainer(model, background), "linear"
            except Exception:
                pass
            background = shap.utils.sample(
                self.x_train,
                min(kernel_background, len(self.x_train)),
                random_state=self.random_state,
            )
            return shap.KernelExplainer(model.predict, background), "kernel"

        if use_cache and os.path.exists(cache_file):
            shap_values_pd = pd.read_csv(cache_file)
            self.shap_values = shap_values_pd
            explainer = build_explainer()[0] if return_explainer else None
            return shap_values_pd.to_numpy(), samples, explainer

        explainer, kind = build_explainer()
        if kind == "kernel":
            shap_values = explainer.shap_values(samples, nsamples="auto")
        else:
            shap_values = explainer.shap_values(samples)
        if hasattr(shap_values, "values"):
            shap_values = shap_values.values

        shap_values_pd = pd.DataFrame(shap_values, columns=samples.columns)
        shap_values_pd.to_csv(cache_file, index=False)
        self.shap_values = shap_values_pd
        return shap_values, samples, explainer

    def beeswarm(
        self,
        model_sel: str,
        samples: Optional[object] = None,
        kernel_background: int = 100,
        max_display: int = 20,
        dpi: int = 500,
    ) -> str:
        """Save the SHAP beeswarm (with mean-|SHAP| bar overlay) for one model."""
        out_dir = os.path.join(self.plots_path, "beeswarm")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{model_sel}.png")

        shap_values, samples, _ = self.shap_config(
            model_sel=model_sel, samples=samples,
            kernel_background=kernel_background, return_explainer=False,
        )

        shap.summary_plot(
            shap_values, samples, show=False, cmap=self.cust_colors_map,
            max_display=max_display, plot_type="dot",
        )
        ax1 = plt.gca()
        ax1.tick_params(axis="x", labelsize=15)
        ax1.tick_params(axis="y", labelsize=15)
        ax1.set_xlabel("SHAP value (impact on model output)", fontsize=16)
        ax1.spines["top"].set_visible(True)
        ax1.spines["top"].set_linewidth(1)

        fig = plt.gcf()
        cbar = fig.axes[-1]
        cbar.tick_params(labelsize=14, labelrotation=0)
        cbar.set_ylabel("Feature Value", fontsize=16, labelpad=-5)

        ax2 = ax1.twiny()
        shap.summary_plot(
            shap_values, self.X, plot_type="bar", show=False,
            max_display=max_display,
        )
        ax2.xaxis.tick_top()
        ax1.spines["bottom"].set_linewidth(1)
        ax2.spines["bottom"].set_linewidth(1)
        ax1.tick_params(axis="both", width=1)
        ax2.tick_params(axis="both", width=1)
        ax2.set_xlabel(
            "mean(|SHAP value|)(average impact on models output magnitude)",
            labelpad=10, fontsize=16,
        )
        ax2.tick_params(axis="x", labelsize=15)
        for bar in plt.gca().patches:
            bar.set_alpha(0.2)
            bar.set_color("red")

        plt.tight_layout()
        ax1.set_facecolor("#F2F2F2")
        ax2.set_facecolor("#F2F2F2")
        fig = plt.gcf()
        fig.set_size_inches(8, 12)
        plt.savefig(out_file, dpi=dpi, bbox_inches="tight", facecolor="#F2F2F2")
        plt.close("all")
        return out_file

    def deps(
        self,
        model_sel: str,
        fitting_model: Optional[str] = None,
        samples: Optional[object] = None,
        kernel_background: int = 100,
        dpi: int = 300,
    ) -> str:
        """Save per-feature SHAP dependence plots and fitted trend models."""
        deps_dir = os.path.join(self.plots_path, "deps")
        img_out_dir = os.path.join(deps_dir, fitting_model or "", model_sel)
        model_save_dir = os.path.join(deps_dir, "fitting_models",
                                      fitting_model or "", model_sel)
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(model_save_dir, exist_ok=True)

        shap_values, samples, _ = self.shap_config(
            model_sel=model_sel, samples=samples,
            kernel_background=kernel_background, return_explainer=False,
        )

        for idx, feat in enumerate(samples.columns):
            plt.figure(figsize=(10, 8), dpi=300)
            ax = plt.axes([0.1, 0.1, 0.7, 0.75])
            top_hist_ax = plt.axes([0.1, 0.85, 0.7, 0.15])
            right_hist_ax = plt.axes([0.8, 0.1, 0.15, 0.75])

            shap.dependence_plot(
                feat, shap_values, samples, show=False, alpha=0.2, ax=ax,
                interaction_index=None, color=self.cust_colors[-1], dot_size=100,
            )
            ax.set_ylabel(f"SHAP value for {feat}", fontsize=28,
                          fontweight="bold", labelpad=5)
            ax.set_xlabel(f"{feat}", fontsize=28, fontweight="bold")
            ax.xaxis.set_tick_params(labelsize=26)
            ax.yaxis.set_tick_params(labelsize=26)
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
            ax.grid(alpha=0.3, linestyle="--")
            x_line = np.linspace(*ax.get_xlim(), 500)
            ax.plot(x_line, np.zeros_like(x_line), "k--", alpha=0.4, linewidth=1.2)
            ax.set_facecolor("#F2F2F2")
            plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)

            # 'shap_values' is (n_samples, n_features); column 'idx' is the
            # per-feature SHAP vector, row 'idx' would be a single sample.
            feat_shap_values = shap_values[:, idx]
            feat_sample_values = samples.iloc[:, idx].values.reshape(-1, 1)

            if fitting_model == "gam":
                gam = LinearGAM(s(0)).fit(feat_sample_values, feat_shap_values)
                x_gam = gam.generate_X_grid(term=0, n=500)
                y_gam = gam.predict(x_gam)
                ax.plot(x_gam, y_gam, "r-", linewidth=2,
                        label="GAM fitting", alpha=0.8)
                ax.legend(loc="best", fontsize=26)
                intervals = gam.confidence_intervals(x_gam, width=0.95)
                ax.fill_between(x_gam.flatten(), intervals[:, 0], intervals[:, 1],
                                alpha=0.3, color="red")
                joblib.dump(gam, os.path.join(model_save_dir, f"{feat}.pkl"))
            elif fitting_model == "linear":
                lr = LinearRegression().fit(feat_sample_values, feat_shap_values)
                x_lr = np.linspace(feat_sample_values.min(),
                                   feat_sample_values.max(), 500).reshape(-1, 1)
                ax.plot(x_lr, lr.predict(x_lr), "r-", linewidth=2,
                        label="Linear fitting", alpha=0.6)
                ax.legend(loc="best", fontsize=26)
                joblib.dump(lr, os.path.join(model_save_dir, f"{feat}.pkl"))

            top_hist_ax.hist(samples[feat], bins=30, alpha=0.3,
                             color=self.cust_colors[-1],
                             edgecolor=self.cust_colors[0],
                             linewidth=1.5, density=True)
            x_range = np.linspace(samples[feat].min(), samples[feat].max(), 1000)
            top_hist_ax.plot(x_range, gaussian_kde(samples[feat])(x_range),
                             color="purple", linewidth=2, alpha=0.5)
            top_hist_ax.set_xlim(ax.get_xlim())
            top_hist_ax.set_xticklabels([])
            top_hist_ax.set_yticklabels([])
            top_hist_ax.spines["top"].set_visible(False)
            top_hist_ax.spines["right"].set_visible(False)
            top_hist_ax.spines["left"].set_visible(False)
            top_hist_ax.tick_params(axis="x", which="both", bottom=False,
                                    top=False, labelbottom=False)
            top_hist_ax.tick_params(axis="y", which="both", left=False,
                                    right=False, labelleft=True)
            top_hist_ax.set_facecolor("#F2F2F2")

            right_hist_ax.hist(feat_shap_values, bins=50, alpha=0.3,
                               orientation="horizontal",
                               color=self.cust_colors[-1],
                               edgecolor=self.cust_colors[0],
                               linewidth=1.5, density=True)
            y_range = np.linspace(feat_shap_values.min(),
                                  feat_shap_values.max(), 1000)
            right_hist_ax.plot(gaussian_kde(feat_shap_values)(y_range), y_range,
                               color="purple", linewidth=2, alpha=0.5)
            right_hist_ax.set_ylim(ax.get_ylim())
            right_hist_ax.set_xticklabels([])
            right_hist_ax.set_yticklabels([])
            right_hist_ax.spines["top"].set_visible(False)
            right_hist_ax.spines["right"].set_visible(False)
            right_hist_ax.spines["bottom"].set_visible(False)
            right_hist_ax.tick_params(axis="x", which="both", bottom=False,
                                      top=False, labelbottom=False)
            right_hist_ax.tick_params(axis="y", which="both", left=False,
                                      right=False, labelleft=True)
            right_hist_ax.set_facecolor("#F2F2F2")

            plt.savefig(os.path.join(img_out_dir, f"{feat}.png"),
                        dpi=dpi, bbox_inches="tight", facecolor="#F2F2F2")
            plt.close("all")

        return img_out_dir

    def radar(
        self,
        model_sel: str,
        data: pd.DataFrame,
        label_c: Optional[List[Tuple[str, str]]] = None,
        max_display: int = 20,
        fitting_model: str = "linear",
        samples: Optional[object] = None,
        kernel_background: int = 100,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
        """Save a radar chart of fitted per-feature SHAP contributions."""
        if label_c is not None:
            assert data.shape[0] == len(label_c), \
                "Samples of data must match with label_c"

        radar_dir = os.path.join(self.plots_path, "radar")
        os.makedirs(radar_dir, exist_ok=True)

        # 'radar' needs only 'shap_values' for feature ranking; the
        # explainer object itself is unused after this point.
        shap_values, _, _ = self.shap_config(
            model_sel=model_sel, samples=samples,
            kernel_background=kernel_background, return_explainer=False,
        )

        feats_imp = [
            (feat, idx, imp)
            for feat, idx, imp in zip(
                self.x_test.columns.tolist(),
                range(len(self.x_test.columns.tolist())),
                np.abs(shap_values).mean(axis=0),
            )
        ]
        rank_feats = sorted(feats_imp, key=lambda x: x[-1], reverse=True)
        order = [rank_feats[i][1] for i in range(len(rank_feats))]

        # Fitted trend models live under '<plots>/deps/fitting_models/...'.
        fitting_dir = os.path.join(
            self.plots_path, "deps", "fitting_models", fitting_model, model_sel,
        )
        radar_dict = {rank_feats[r][0]: [] for r in range(max_display)}
        for idx in range(data.shape[0]):
            for rank in range(max_display):
                feat = rank_feats[rank][0]
                fitting = joblib.load(os.path.join(fitting_dir, f"{feat}.pkl"))
                fitted, = fitting.predict(data[feat].values[idx].reshape(-1, 1))
                radar_dict[feat].append(fitted)

        radar_df = pd.DataFrame(radar_dict)
        scaler = MinMaxScaler(feature_range=(0, 1))
        radar_scaled = pd.DataFrame(
            scaler.fit_transform(radar_df), columns=radar_df.columns,
        )
        radar_scaled.insert(
            0, "groups",
            [label_c[i][0] for i in range(data.shape[0])]
            if label_c is not None else list(range(data.shape[0])),
        )

        categories = list(radar_scaled)[1:]
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], categories, fontsize=8)
        ax.tick_params(axis="x", pad=25)
        ax.set_rlabel_position(0)
        plt.yticks([-1, 0, 1], ["-1", "0", "1"], color="grey", alpha=1, size=12)
        plt.ylim(-1.5, 1.5)

        for i in range(radar_scaled.shape[0]):
            values = radar_scaled.loc[i].drop("groups").values.flatten().tolist()
            values += values[:1]
            color = label_c[i][1] if label_c is not None else None
            label = label_c[i][0] if label_c is not None else i
            is_last = (i == radar_scaled.shape[0] - 1)
            ax.plot(angles, values,
                    linewidth=1.5 if is_last else 1,
                    linestyle="solid" if is_last else "--",
                    label=label, color=color)
            ax.fill(angles, values, color=color, alpha=0.1)

        plt.legend(loc="upper right", bbox_to_anchor=(0.05, 0.05))
        plt.savefig(os.path.join(radar_dir, f"radar_{model_sel}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close("all")
        return radar_df, radar_scaled, order

    def list_dict(self) -> List[dict]:
        """Return per-model train/test MAE, MSE and R² in long form."""
        metrics = (mean_absolute_error, mean_squared_error, r2_score)
        output = [{"x": [], "y": [], "hue": []} for _ in metrics]
        for bucket, metric in zip(output, metrics):
            for name, model in self.model_dict.items():
                bucket["x"].append(_short(name))
                bucket["y"].append(metric(self.y_train, model.predict(self.x_train)))
                bucket["hue"].append("Training")
            for name, model in self.model_dict.items():
                bucket["x"].append(_short(name))
                bucket["y"].append(metric(self.y_test, model.predict(self.x_test)))
                bucket["hue"].append("Test")
        return output

    def score_hist(self) -> None:
        """Save a grouped bar chart of per-model evaluation metrics."""
        out_dir = os.path.join(self.plots_path, "score_hist")
        os.makedirs(out_dir, exist_ok=True)

        list_dict = self.list_dict()
        fig, axes = plt.subplots(3, 1, figsize=(22, 18), sharex=True)
        eval_indic = ["MAE", "MSE", "R$^2$ score"]
        for i, ax in enumerate(axes.flat):
            data = pd.DataFrame(list_dict[i])
            sns.barplot(x="x", y="y", hue="hue", data=data,
                        palette=["#E5CBE1", "#732E7E"], ax=ax)
            for container in ax.containers:
                ax.bar_label(container, fmt="%.3f", fontsize=16, color="black",
                             padding=5, fontweight="bold", rotation=0)
            ax.grid(alpha=0.2, linestyle="--")
            ax.set_xlabel("")
            ax.set_ylabel("")
            for bar in ax.patches:
                bar.set_alpha(0.75)
                bar.set_linewidth(1.5)
                bar.set_edgecolor("gray")
            if i == 0:
                ax.legend(framealpha=0, fontsize=22, loc="upper right")
            else:
                ax.get_legend().remove()
            ax.tick_params("x", labelsize=21, labelrotation=0, pad=15)
            ax.tick_params("y", labelsize=21, pad=20)
            ylim = ax.get_ylim()
            ax.set_ylim(ylim[0], ylim[1] + 0.1)
            ax.set_ylabel(eval_indic[i], fontsize=21, fontweight="bold")

        plt.savefig(os.path.join(out_dir, "score_hist.png"),
                    dpi=300, bbox_inches="tight")
        plt.close("all")

    def accs(self) -> None:
        """Save one true-vs-predicted scatter plot per model."""
        out_dir = os.path.join(self.plots_path, "accs")
        os.makedirs(out_dir, exist_ok=True)

        for model_name, model in self.model_dict.items():
            short = _short(model_name)
            data = pd.DataFrame({
                "x": list(self.Y.iloc[self.x_train.index])
                     + list(self.Y.iloc[self.x_test.index]),
                "y": list(model.predict(self.x_train))
                     + list(model.predict(self.x_test)),
                "hue": ["Training "] * len(self.x_train)
                       + [" Test"] * len(self.x_test),
            })
            plt.figure(figsize=(6.5, 6), dpi=500)
            ax = sns.scatterplot(
                x="x", y="y", data=data, hue="hue",
                palette=["red", self.cust_colors[-1]],
                s=120, style="hue", markers=["o", "o"],
                alpha=[0.5 if g == "Training " else 0.7 for g in data["hue"]],
            )
            ax.grid(alpha=0.3, linestyle="--")
            legend = plt.legend(
                title=f"    {short}",
                title_fontsize=22, fontsize=18, loc="upper left",
                frameon=True, framealpha=0,
                bbox_to_anchor=(-0.05, 1), edgecolor="gray", facecolor="white",
                labelspacing=0.8, handlelength=2, handletextpad=0.05,
                shadow=False, markerscale=1.2, borderpad=0.5, fancybox=True,
            )
            # 'legendHandles' was renamed in mpl 3.7; support both.
            handles = (getattr(legend, "legend_handles", None)
                       or getattr(legend, "legendHandles", None))
            if handles:
                handles[0].set_alpha(0.6)
                handles[1].set_alpha(0.9)

            ax.set_xlabel("True", fontsize=22)
            ax.set_ylabel("Predict", fontsize=22)
            ax.tick_params(labelsize=14, length=4, width=1.5, pad=6)

            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            lo = min(xlim[0], ylim[0])
            hi = max(xlim[1], ylim[1])
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.6, linewidth=2)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.xaxis.set_major_locator(MultipleLocator(3))
            ax.yaxis.set_major_locator(MultipleLocator(3))

            r2_train = r2_score(self.y_train, model.predict(self.x_train))
            r2_test = r2_score(self.y_test, model.predict(self.x_test))
            ax.text(hi - 5.5, lo + 2, f"R$^{2}$ score = {r2_train: .3f}",
                    color="red", fontsize=18)
            ax.text(hi - 5.5, lo + 1, f"R$^{2}$ score = {r2_test: .3f}",
                    color=self.cust_colors[-1], fontsize=18)

            plt.savefig(os.path.join(out_dir, f"{short}_acc.png"),
                        dpi=300, bbox_inches="tight")
            plt.close("all")

    def waterfall(
        self,
        model_sel: str,
        data: Optional[pd.DataFrame] = None,
        order: Optional[Iterable[int]] = None,
        label_c: Optional[List[Tuple[str, str]]] = None,
        samples: Optional[object] = None,
        kernel_background: int = 100,
    ) -> Optional[str]:
        """Save one SHAP waterfall per row of 'data'."""
        wf_dir = os.path.join(self.plots_path, "waterfall", model_sel)
        os.makedirs(wf_dir, exist_ok=True)

        _, _, explainer = self.shap_config(
            model_sel=model_sel, samples=samples,
            kernel_background=kernel_background, return_explainer=True,
        )

        waterfall_path: Optional[str] = None
        for i in range(data.shape[0]):
            explanation = explainer(round(data.iloc[i:i + 1], 3))
            with plt.rc_context({
                "text.color": "#FFFFFF",
                "axes.labelcolor": "#FFFFFF",
                "axes.edgecolor": "#A8B4C8",
                "xtick.color": "#F0F4FA",
                "ytick.color": "#F0F4FA",
                "figure.facecolor": (35 / 255, 41 / 255, 54 / 255, 0.5),
                "axes.facecolor": (35 / 255, 41 / 255, 54 / 255, 0.5),
                "savefig.facecolor": (35 / 255, 41 / 255, 54 / 255, 0.5),
                "grid.color": "#4A5670",
                "grid.alpha": 0.5,
            }):
                plt.figure(figsize=(16, 10), dpi=300)
                plt.gca().set_axisbelow(True)
                shap.plots.waterfall(explanation[0], max_display=20, show=False)
                if label_c is not None:
                    plt.title(label_c[i][0], fontsize=22, fontweight="bold",
                              color=label_c[i][1])
                plt.tight_layout()

                if label_c is not None:
                    tag = str(label_c[i][0])
                else:
                    tag = f"row{i}"
                waterfall_path = os.path.join(wf_dir, f"{tag}.png")
                plt.savefig(waterfall_path, dpi=300, bbox_inches="tight",
                            transparent=True)
                plt.close("all")
        return waterfall_path