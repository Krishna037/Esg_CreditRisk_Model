"""
=============================================================================
UNIFIED CREDIT RISK PIPELINE - MODEL TRAINING
=============================================================================
Unified model training and hyperparameter tuning for all algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from config import (
    LOGISTIC_PARAMS, XGBOOST_PARAMS, CATBOOST_PARAMS,
    RANDOM_FOREST_PARAMS, MLP_PARAMS, RANDOM_STATE
)
from preprocessing import PreprocessedData, get_cv_splitter


class UnifiedModelTrainer:
    """Centralized model training with consistent hyperparameters and CV."""

    def __init__(self, prep_data: PreprocessedData, verbose: bool = True):
        """
        Initialize trainer.
        
        Args:
            prep_data: PreprocessedData object from preprocessing module
            verbose: Whether to print progress
        """
        self.prep_data = prep_data
        self.verbose = verbose
        self.models: Dict[str, Any] = {}
        self.cv_results: Dict[str, dict] = {}

    def train_logistic_regression(self, do_grid_search: bool = True) -> LogisticRegression:
        """Train Logistic Regression with optional grid search."""
        if self.verbose:
            print("\n[Training] Logistic Regression...")

        X_train = self.prep_data.X_train_resampled
        y_train = self.prep_data.y_train_resampled

        if do_grid_search:
            cv = get_cv_splitter(self.prep_data.y_train)
            lr_grid = GridSearchCV(
                estimator=LogisticRegression(**{k: v for k, v in LOGISTIC_PARAMS.items() if k != "C"}),
                param_grid={"C": [0.001, 0.01, 0.1, 1.0, 10.0]},
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
            )
            lr_grid.fit(X_train, y_train)
            model = lr_grid.best_estimator_
            if self.verbose:
                print(f"  Best C: {lr_grid.best_params_['C']}")
        else:
            model = LogisticRegression(**LOGISTIC_PARAMS)
            model.fit(X_train, y_train)

        self.models["Logistic Regression"] = model
        return model

    def train_xgboost(self, do_grid_search: bool = True) -> XGBClassifier:
        """Train XGBoost with optional grid search."""
        if self.verbose:
            print("\n[Training] XGBoost...")

        X_train = self.prep_data.X_train_resampled
        y_train = self.prep_data.y_train_resampled

        # Calculate scale_pos_weight for imbalance
        scale_pw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        params = {**XGBOOST_PARAMS, "scale_pos_weight": scale_pw}

        if do_grid_search:
            cv = get_cv_splitter(self.prep_data.y_train)
            xgb_grid = GridSearchCV(
                estimator=XGBClassifier(**{k: v for k, v in params.items() if k not in ["n_estimators", "learning_rate", "max_depth"]}),
                param_grid={
                    "max_depth": [4, 5, 6],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [300, 500, 700],
                },
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
            )
            xgb_grid.fit(X_train, y_train)
            model = xgb_grid.best_estimator_
            if self.verbose:
                print(f"  Best params: {xgb_grid.best_params_}")
        else:
            model = XGBClassifier(**params)
            model.fit(X_train, y_train)

        self.models["XGBoost"] = model
        return model

    def train_catboost(self) -> CatBoostClassifier:
        """Train CatBoost."""
        if self.verbose:
            print("\n[Training] CatBoost...")

        X_train = self.prep_data.X_train_resampled
        y_train = self.prep_data.y_train_resampled

        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(X_train, y_train, verbose=False)

        self.models["CatBoost"] = model
        return model

    def train_random_forest(self) -> RandomForestClassifier:
        """Train Random Forest."""
        if self.verbose:
            print("\n[Training] Random Forest...")

        X_train = self.prep_data.X_train_resampled
        y_train = self.prep_data.y_train_resampled

        model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
        model.fit(X_train, y_train)

        self.models["Random Forest"] = model
        return model

    def train_mlp(self, do_grid_search: bool = False) -> MLPClassifier:
        """Train MLP neural network with strong regularization and early stopping."""
        if self.verbose:
            print("\n[Training] Neural Network (MLP)...")

        X_train = self.prep_data.X_train_resampled
        y_train = self.prep_data.y_train_resampled

        if do_grid_search:
            cv = get_cv_splitter(self.prep_data.y_train)
            mlp_grid = GridSearchCV(
                estimator=MLPClassifier(**{k: v for k, v in MLP_PARAMS.items() if k != "alpha"}),
                param_grid={"alpha": [0.001, 0.01, 0.05], "hidden_layer_sizes": [(64, 32), (96, 48), (64, 64)]},
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
            )
            mlp_grid.fit(X_train, y_train)
            model = mlp_grid.best_estimator_
            if self.verbose:
                print(f"  Best params: {mlp_grid.best_params_}")
        else:
            model = MLPClassifier(**MLP_PARAMS)
            model.fit(X_train, y_train)

        self.models["Neural Network (MLP)"] = model
        return model

    def train_stacking_ensemble(self, use_best_estimators: bool = True) -> StackingClassifier:
        """Train stacking ensemble."""
        if self.verbose:
            print("\n[Training] Stacking Ensemble...")

        X_train = self.prep_data.X_train_resampled
        y_train = self.prep_data.y_train_resampled
        cv = get_cv_splitter(self.prep_data.y_train)

        # Use trained base models if available
        estimators = []
        if use_best_estimators and len(self.models) > 0:
            for name in ["XGBoost", "CatBoost", "Random Forest", "Neural Network (MLP)"]:
                if name in self.models:
                    estimators.append((name.lower(), self.models[name]))

        # Fallback: train fresh base learners
        if not estimators:
            scale_pw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
            xgb_params = {**XGBOOST_PARAMS, "scale_pos_weight": scale_pw}
            estimators = [
                ("xgb", XGBClassifier(**xgb_params)),
                ("cat", CatBoostClassifier(**CATBOOST_PARAMS)),
                ("rf", RandomForestClassifier(**RANDOM_FOREST_PARAMS)),
                ("mlp", MLPClassifier(**MLP_PARAMS)),
            ]

        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE),
            cv=cv,
            stack_method="predict_proba",
            passthrough=False,
            n_jobs=-1,
        )
        stacking.fit(X_train, y_train)

        self.models["Stacking Ensemble"] = stacking
        return stacking

    def train_soft_voting_ensemble(self) -> VotingClassifier:
        """Train soft voting ensemble (advanced technique) with probabilistic averaging."""
        if self.verbose:
            print("\n[Training] Soft Voting Ensemble...")

        X_train = self.prep_data.X_train_resampled
        y_train = self.prep_data.y_train_resampled

        scale_pw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        xgb_params = {**XGBOOST_PARAMS, "scale_pos_weight": scale_pw}

        voting = VotingClassifier(
            estimators=[
                ("xgb", XGBClassifier(**xgb_params)),
                ("rf", RandomForestClassifier(**RANDOM_FOREST_PARAMS)),
                ("cat", CatBoostClassifier(**CATBOOST_PARAMS)),
                ("mlp", MLPClassifier(**MLP_PARAMS)),
            ],
            voting="soft",
            weights=[2, 2, 2, 1],
            n_jobs=-1,
        )
        voting.fit(X_train, y_train)

        self.models["Soft Voting Ensemble"] = voting
        return voting

    def train_all(self, skip_ensemble: bool = False) -> Dict[str, Any]:
        """Train all models in sequence."""
        if self.verbose:
            print("\n" + "=" * 70)
            print("STAGE: MODEL TRAINING")
            print("=" * 70)

        self.train_logistic_regression(do_grid_search=True)
        self.train_xgboost(do_grid_search=True)
        self.train_catboost()
        self.train_random_forest()
        self.train_mlp(do_grid_search=False)
        self.train_soft_voting_ensemble()

        if not skip_ensemble:
            try:
                self.train_stacking_ensemble(use_best_estimators=True)
            except Exception as e:
                if self.verbose:
                    print(f"  [WARN] Stacking ensemble failed: {e}")

        if self.verbose:
            print(f"\n✓ Trained {len(self.models)} models")

        return self.models

    def get_model(self, name: str) -> Any:
        """Get a trained model by name."""
        return self.models.get(name)

    def get_all_models(self) -> Dict[str, Any]:
        """Get all trained models."""
        return self.models.copy()
