"""Bayesian-optimized comparison of twelve regressors.

Input:
    x_train, x_test, y_train, y_test  -- train/test split.
    base_dir                          -- root for 'ml_models/' pickles.
    queue                             -- optional; one row per fitted model.
    params_<name>                     -- which models to train.
    optzr_init_points, optzr_n_iter   -- BayesianOptimization budget.
    random_state                      -- seeds estimators and BO.

Output (side effects only, 'train()' returns 'None'):
    '<base_dir>/ml_models/<NameMapping>.pkl'   one pickle per model.
    'queue.put(DataFrame[name, R2, MSE, MAE])' one row per fitted model.
    'AutoML.dict_models_eval'                  cumulative metric dict.
"""

import os
import sys
import time
import warnings

import catboost
import joblib
import lightgbm
import numpy as np
import pandas as pd
import xgboost
from bayes_opt import BayesianOptimization
from sklearn.ensemble import (
    AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    explained_variance_score, mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------------------------------------------------------------------------
# Console layout
# ---------------------------------------------------------------------------

_LOG_WIDTH = 80
_LOG_BAR = "─" * _LOG_WIDTH
_NAME_WIDTH = 26  # longest name: "GradientBoostingRegressor" (25) + 1


def _fmt_duration(seconds):
    """Format a duration as ``0.3s`` / ``12.3s`` / ``1m34s`` / ``2h05m``."""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _print_header(n_models, n_jobs, init_points, n_iter, seed):
    """One-time configuration summary printed before the first model."""
    print(_LOG_BAR)
    print(
        f" AutoML  models={n_models}  n_jobs={n_jobs}  "
        f"init_points={init_points}  n_iter={n_iter}  seed={seed}"
    )
    print(_LOG_BAR)


def _print_progress(rank, total, display_name, r2, mse, mae, elapsed):
    """Single-row progress for one fitted model."""
    print(
        f"[{rank:>2}/{total}] {display_name:<{_NAME_WIDTH}}  "
        f"R² {r2:.4f}  MSE {mse:.4f}  MAE {mae:.4f}  {_fmt_duration(elapsed):>6}"
    )


def _print_ending(best_name, r2, mse, mae, total_time):
    """Final banner with the best model over the whole run."""
    print(_LOG_BAR)
    print(
        f" ✓ Finished  best={best_name}  "
        f"R² {r2:.4f}  MSE {mse:.4f}  MAE {mae:.4f}  "
        f"total {_fmt_duration(total_time)}"
    )
    print(_LOG_BAR)


# ---------------------------------------------------------------------------
# Hyper-parameter search spaces
# ---------------------------------------------------------------------------

params_ridge = {"alpha": (1e-5, 20)}

params_svr = {"C": (0.1, 1000), "gamma": (1e-5, 20),
              "epsilon": (0.01, 1), "kernel": (0, 1)}

params_knn = {"n_neighbors": (1, 50), "weights": (0, 1), "metric": (0, 3),
              "p": (1, 3), "leaf_size": (10, 50)}

params_dtr = {"max_depth": (2, 50), "min_samples_split": (2, 50),
              "min_samples_leaf": (2, 50),
              "max_features": (0.1, 1),
              "min_impurity_decrease": (0, 0.5),
              "ccp_alpha": (0, 0.1), "criterion": (0, 3)}

params_rf = {"n_estimators": (50, 500), "max_depth": (10, 30),
             "min_samples_split": (2, 20), "min_samples_leaf": (1, 10),
             "max_features": (0.3, 1), "criterion": (0, 3),
             "bootstrap": (0, 1)}

params_et = dict(params_rf)

params_gbdt = {"n_estimators": (50, 500), "learning_rate": (0.001, 0.3),
               "max_depth": (3, 10),
               "min_samples_split": (2, 20), "min_samples_leaf": (1, 10),
               "subsample": (0.6, 1), "max_features": (0.1, 1),
               "criterion": (0, 1)}

params_ada = {"n_estimators": (50, 500), "learning_rate": (0.01, 2),
              "loss": (0, 2)}

params_xgb = {"n_estimators": (50, 500), "learning_rate": (0.01, 0.3),
              "max_depth": (3, 10), "min_child_weight": (1, 20),
              "subsample": (0.6, 1),
              "colsample_bytree": (0.6, 1.0),
              "colsample_bylevel": (0.6, 1.0),
              "gamma": (0, 10),
              "reg_alpha": (0, 5), "reg_lambda": (0, 5)}

params_lgbm = {"n_estimators": (50, 500), "num_leaves": (20, 500),
               "learning_rate": (0.001, 0.3),
               "min_child_samples": (10, 50),
               "subsample": (0.6, 1),
               "colsample_bytree": (0.6, 1.0),
               "reg_alpha": (0, 5), "reg_lambda": (0, 5)}

params_cat = {"iterations": (500, 2000),
              "learning_rate": (0.005, 0.2),
              "depth": (3, 9),
              "l2_leaf_reg": (0.5, 8),
              "random_strength": (0.5, 6),
              "bagging_temperature": (0.2, 0.8),
              "grow_policy": (0, 2),
              "min_data_in_leaf": (5, 50),
              "rsm": (0.6, 1), "subsample": (0.8, 1)}


# Display name -> artifact filename.  Kept in sync with
# 'MainWindow.algorithm_display_names' in the GUI layer.
NAME_MAP = {
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


# Discrete-choice lookup tables: BO returns a float bucket; this module
# rounds it to an integer key and resolves the categorical value.
_CRITERION = {"0": "squared_error", "1": "friedman_mse",
              "2": "absolute_error", "3": "poisson"}
_CRITERION_GBDT = {"0": "friedman_mse", "1": "squared_error"}
_KERNEL = {"0": "linear", "1": "rbf"}
_WEIGHTS = {"0": "uniform", "1": "distance"}
_METRIC = {"0": "euclidean", "1": "manhattan",
           "2": "minkowski", "3": "minkowski"}
_BOOTSTRAP = {"0": True, "1": False}
_LOSS = {"0": "linear", "1": "square", "2": "exponential"}
_GROW_POLICY = {"0": "SymmetricTree", "1": "Depthwise", "2": "Lossguide"}


def _choice(value, mapping):
    """Resolve a floating-point BO bucket to its categorical value."""
    return mapping[str(int(round(value)))]


# ---------------------------------------------------------------------------
# AutoML
# ---------------------------------------------------------------------------

class AutoML:
    """Fit and persist every enabled regressor on a train/test split."""

    dict_models_eval = {}

    def __init__(
        self, x_train, x_test, y_train, y_test, base_dir=None, queue=None,
        params_lr=False, params_ridge=False, params_svr=False, params_knn=False,
        params_dtr=False, params_rf=False, params_et=False, params_gbdt=False,
        params_ada=False, params_xgb=False, params_lgbm=False, params_cat=False,
        n_jobs=-1, optzr_init_points=10, optzr_n_iter=50,
        printinfo=True, random_state=42,
    ):
        if base_dir is None:
            if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
                base_dir = os.path.dirname(os.path.realpath(sys.executable))
            else:
                base_dir = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))

        self.models_path = os.path.join(base_dir, "ml_models")
        os.makedirs(self.models_path, exist_ok=True)

        # Every 'params_X' flag is stored as a plain bool.
        self.params_lr = bool(params_lr)
        self.params_ridge = bool(params_ridge)
        self.params_svr = bool(params_svr)
        self.params_knn = bool(params_knn)
        self.params_dtr = bool(params_dtr)
        self.params_rf = bool(params_rf)
        self.params_et = bool(params_et)
        self.params_gbdt = bool(params_gbdt)
        self.params_ada = bool(params_ada)
        self.params_xgb = bool(params_xgb)
        self.params_lgbm = bool(params_lgbm)
        self.params_cat = bool(params_cat)

        self.q = queue
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.x_train, self.x_test = x_train, x_test
        self.y_train, self.y_test = y_train, y_test
        self.dict_models_eval = AutoML.dict_models_eval
        self.optzr_init_points = optzr_init_points
        self.optzr_n_iter = optzr_n_iter
        self.printinfo = printinfo
        self.models_instance = {}

    # ------------------------------------------------------------------
    # Training orchestration
    # ------------------------------------------------------------------

    def train(self):
        """Fit every enabled model, save its pickle, and stream its metrics."""
        active_specs = [s for s in self._specs() if s[0]]
        n_models = len(active_specs) + (1 if self.params_lr else 0)

        if self.printinfo:
            _print_header(
                n_models, self.n_jobs, self.optzr_init_points,
                self.optzr_n_iter, self.random_state,
            )

        t_start = time.perf_counter()
        rank = 0
        trained_keys = []

        # LinearRegression has no hyperparameters: fit directly.
        if self.params_lr:
            rank += 1
            t0 = time.perf_counter()
            lr = LinearRegression().fit(self.x_train, self.y_train)
            self.model_eval_save("LR", lr)
            trained_keys.append("LR")
            if self.printinfo:
                self._log_progress(rank, n_models, "LR",
                                   time.perf_counter() - t0)

        # Each spec is '(enabled, key, pbounds, map_raw, cls,
        # cv_kwargs, fit_kwargs)'.
        for _enabled, key, pbounds, map_raw, cls, cv_kwargs, fit_kwargs in active_specs:
            rank += 1
            t0 = time.perf_counter()
            best_raw = self.bayes_optzr(
                self._cv_objective(cls, map_raw, cv_kwargs),
                pbounds,
                model_name=NAME_MAP.get(key, key),
            )
            best = map_raw(best_raw)
            model = cls(**best, **fit_kwargs).fit(self.x_train, self.y_train)
            self.model_eval_save(key, model, best)
            trained_keys.append(key)
            if self.printinfo:
                self._log_progress(rank, n_models, key,
                                   time.perf_counter() - t0)

        if self.printinfo and trained_keys:
            best_key = max(
                trained_keys,
                key=lambda k: float(self.dict_models_eval[k]["R2_score"]),
            )
            m = self.dict_models_eval[best_key]
            _print_ending(
                NAME_MAP.get(best_key, best_key),
                float(m["R2_score"]), float(m["MSE"]), float(m["MAE"]),
                time.perf_counter() - t_start,
            )

    def _log_progress(self, rank, total, key, elapsed):
        """Emit a single progress row for ``key`` using its stored metrics."""
        m = self.dict_models_eval[key]
        _print_progress(
            rank, total, NAME_MAP.get(key, key),
            float(m["R2_score"]), float(m["MSE"]), float(m["MAE"]),
            elapsed,
        )

    def _specs(self):
        """Return '(enabled, key, pbounds, map_raw, cls, cv_kw, fit_kw)'."""
        rs, nj = self.random_state, self.n_jobs
        return (
            (self.params_ridge, "Ridge", params_ridge, self._map_ridge,
             Ridge, {"random_state": rs}, {"random_state": rs}),

            (self.params_svr, "SVR", params_svr, self._map_svr,
             SVR, {"max_iter": 10000}, {}),

            (self.params_knn, "KNN", params_knn, self._map_knn,
             KNeighborsRegressor, {}, {}),

            (self.params_dtr, "DT", params_dtr, self._map_dtr,
             DecisionTreeRegressor, {"random_state": rs}, {"random_state": rs}),

            (self.params_rf, "RF", params_rf, self._map_forest,
             RandomForestRegressor, {"random_state": rs, "n_jobs": nj},
             {"random_state": rs}),

            (self.params_et, "ET", params_et, self._map_forest,
             ExtraTreesRegressor, {"random_state": rs, "n_jobs": nj},
             {"random_state": rs}),

            (self.params_gbdt, "GBDT", params_gbdt, self._map_gbdt,
             GradientBoostingRegressor, {"random_state": rs},
             {"random_state": rs}),

            (self.params_ada, "AdaBoost", params_ada, self._map_ada,
             AdaBoostRegressor,
             {"estimator": DecisionTreeRegressor(max_depth=3),
              "random_state": rs},
             {"random_state": rs}),

            (self.params_xgb, "XGBoost", params_xgb, self._map_xgb,
             xgboost.XGBRegressor, {"random_state": rs}, {"random_state": rs}),

            (self.params_lgbm, "LGBM", params_lgbm, self._map_lgbm,
             lightgbm.LGBMRegressor,
             {"random_state": rs, "verbose": -1, "n_jobs": nj},
             {"random_state": rs, "verbose": -1}),

            (self.params_cat, "CatBoost", params_cat, self._map_cat,
             catboost.CatBoostRegressor,
             {"od_type": "Iter", "od_wait": 100, "verbose": False,
              "random_state": rs},
             {"od_type": "Iter", "od_wait": 100, "verbose": False,
              "random_state": rs}),
        )

    # ------------------------------------------------------------------
    # Reporting / persistence
    # ------------------------------------------------------------------

    def model_eval_save(self, name, model, params_best=None):
        """Compute metrics for 'model', stream them, and pickle the model."""
        y_pred_test = model.predict(self.x_test)
        mse = mean_squared_error(self.y_test, y_pred_test)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred_test)
        r2_test = r2_score(self.y_test, y_pred_test)
        ev = explained_variance_score(self.y_test, y_pred_test)

        self.dict_models_eval[name] = {
            "MSE": f"{mse}", "RMSE": f"{rmse}", "MAE": f"{mae}",
            "R2_score": f"{r2_test}", "EV_score": f"{ev}",
            "Model_with_BestParams": f"{model}",
        }

        if self.q is not None:
            self.q.put(pd.DataFrame(
                [[name, r2_test, mse, mae]],
                columns=["name", "R2", "MSE", "MAE"],
            ))

        # Fall back to 'name' when a future model is added without a
        # NAME_MAP entry; losing a pretty filename is preferable to
        # KeyError-ing after the model was already fitted.
        filename = NAME_MAP.get(name, name)
        joblib.dump(model, os.path.join(self.models_path, f"{filename}.pkl"))

    # ------------------------------------------------------------------
    # Bayesian optimization
    # ------------------------------------------------------------------

    def bayes_optzr(self, f, pbounds, model_name=None):
        """Maximize 'f' over 'pbounds'; return the raw best parameters.

        The iteration table printed by ``BayesianOptimization.maximize`` is
        left untouched: intercepting or reformatting it proved unreliable
        across bayes_opt releases (0.x prints, 1.x goes through a different
        internal path, 2.x routes through ``logging``).  Instead, a header
        and footer bracket each table so it stays visually scoped inside
        the GUI log:

            ┌─ RidgeRegression · Bayesian search (init=5, iter=5)
            | iter | target | alpha |
            ...
            └─ best cv 0.688690
        """
        if self.printinfo and model_name is not None:
            print(
                f" ┌─ {model_name} · Bayesian search "
            )

        optzr = BayesianOptimization(
            f=f, pbounds=pbounds,
            verbose=2 if self.printinfo else -1,
            random_state=self.random_state,
        )
        optzr.maximize(
            init_points=self.optzr_init_points,
            n_iter=self.optzr_n_iter,
        )

        if self.printinfo:
            try:
                best_cv = float(optzr.max["target"])
            except Exception:
                best_cv = float("nan")
            print(f" └─ best cv {best_cv:.6f}")

        return optzr.max["params"]

    def _cv_objective(self, cls, map_raw, cv_kwargs):
        """Build the 5-fold CV objective for one model class."""
        def objective(**raw):
            estimator = cls(**map_raw(raw), **cv_kwargs)
            return cross_val_score(
                estimator, self.x_train, self.y_train,
                cv=5, n_jobs=self.n_jobs,
            ).mean()
        return objective

    # ------------------------------------------------------------------
    # Parameter mappers (BO float dict -> constructor kwargs)
    # ------------------------------------------------------------------

    @staticmethod
    def _map_ridge(p):
        return {"alpha": p["alpha"]}

    @staticmethod
    def _map_svr(p):
        return {
            "C": p["C"], "gamma": p["gamma"], "epsilon": p["epsilon"],
            "kernel": _choice(p["kernel"], _KERNEL),
        }

    @staticmethod
    def _map_knn(p):
        return {
            "n_neighbors": int(round(p["n_neighbors"])),
            "weights": _choice(p["weights"], _WEIGHTS),
            "metric": _choice(p["metric"], _METRIC),
            "p": p["p"],
            "leaf_size": int(round(p["leaf_size"])),
        }

    @staticmethod
    def _map_dtr(p):
        return {
            "max_depth": int(round(p["max_depth"])),
            "min_samples_split": int(round(p["min_samples_split"])),
            "min_samples_leaf": int(round(p["min_samples_leaf"])),
            "max_features": p["max_features"],
            "min_impurity_decrease": p["min_impurity_decrease"],
            "ccp_alpha": p["ccp_alpha"],
            "criterion": _choice(p["criterion"], _CRITERION),
        }

    @staticmethod
    def _map_forest(p):
        return {
            "n_estimators": int(round(p["n_estimators"])),
            "max_depth": int(round(p["max_depth"])),
            "min_samples_split": int(round(p["min_samples_split"])),
            "min_samples_leaf": int(round(p["min_samples_leaf"])),
            "max_features": p["max_features"],
            "criterion": _choice(p["criterion"], _CRITERION),
            "bootstrap": _choice(p["bootstrap"], _BOOTSTRAP),
        }

    @staticmethod
    def _map_gbdt(p):
        return {
            "n_estimators": int(round(p["n_estimators"])),
            "learning_rate": p["learning_rate"],
            "max_depth": int(round(p["max_depth"])),
            "min_samples_split": int(round(p["min_samples_split"])),
            "min_samples_leaf": int(round(p["min_samples_leaf"])),
            "subsample": p["subsample"],
            "max_features": p["max_features"],
            "criterion": _choice(p["criterion"], _CRITERION_GBDT),
        }

    @staticmethod
    def _map_ada(p):
        return {
            "n_estimators": int(round(p["n_estimators"])),
            "learning_rate": p["learning_rate"],
            "loss": _choice(p["loss"], _LOSS),
        }

    @staticmethod
    def _map_xgb(p):
        return {
            "n_estimators": int(round(p["n_estimators"])),
            "learning_rate": p["learning_rate"],
            "max_depth": int(round(p["max_depth"])),
            "min_child_weight": p["min_child_weight"],
            "subsample": p["subsample"],
            "colsample_bytree": p["colsample_bytree"],
            "colsample_bylevel": p["colsample_bylevel"],
            "gamma": p["gamma"],
            "reg_alpha": p["reg_alpha"],
            "reg_lambda": p["reg_lambda"],
        }

    @staticmethod
    def _map_lgbm(p):
        return {
            "n_estimators": int(round(p["n_estimators"])),
            "num_leaves": int(round(p["num_leaves"])),
            "learning_rate": p["learning_rate"],
            "min_child_samples": int(round(p["min_child_samples"])),
            "subsample": p["subsample"],
            "colsample_bytree": p["colsample_bytree"],
            "reg_alpha": p["reg_alpha"],
            "reg_lambda": p["reg_lambda"],
        }

    @staticmethod
    def _map_cat(p):
        return {
            "iterations": int(round(p["iterations"])),
            "learning_rate": p["learning_rate"],
            "depth": int(round(p["depth"])),
            "l2_leaf_reg": p["l2_leaf_reg"],
            "random_strength": p["random_strength"],
            "bagging_temperature": p["bagging_temperature"],
            "grow_policy": _choice(p["grow_policy"], _GROW_POLICY),
            "min_data_in_leaf": p["min_data_in_leaf"],
            "rsm": p["rsm"],
            "subsample": p["subsample"],
        }