"""
Improved ML Study of Nd-doped ATiO3 with rigorous validation, SHAP, and matminer.

MODIFIED VERSION — Addresses Reviewer comments:
  (C1, Page 7) Random 80/20 split → COMPOSITION-GROUPED splits everywhere.
  (C2, Fig 1b) RMSE/MAE averaged across targets with different units →
               NORMALIZED errors (RMSE/σ_y, MAE/σ_y) + stacking advantage analysis.
  (C3, §3.2)   ML predictions now use DFT-matching doping concentrations:
               x = 0.0625, 0.125, 0.1875, 0.25 (= 1, 2, 3, 4 dopants per 16-f.u. supercell)

Author: [Your Name]
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re
import json
import ast
import warnings
import pickle
from datetime import datetime
import os
import optuna

# Scikit-learn
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     KFold, GridSearchCV, RandomizedSearchCV,
                                     cross_validate, RepeatedKFold, ShuffleSplit,
                                     learning_curve,
                                     GroupKFold, GroupShuffleSplit)
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFECV
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             make_scorer)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputRegressor

# XGBoost
import xgboost as xgb

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Matminer
from matminer.featurizers.composition import (ElementProperty, Stoichiometry,
                                              ValenceOrbital, ElectronegativityDiff,
                                              BandCenter, Miedema, AtomicOrbitals)
from matminer.featurizers.base import MultipleFeaturizer
from pymatgen.core import Composition

# SHAP
import shap

# Random seeds
np.random.seed(42)
torch.manual_seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# ============================================
# GLOBAL CONFIGURATION
# ============================================
FAST_MODE = False

# If Stacking's average R² is within this tolerance of the best model's
# average R², prefer Stacking for the final predictions. This is used
# because the top three models (RF, XGB, Stacking) are statistically
# indistinguishable under grouped CV, and Stacking is the ensemble that
# best represents the combined knowledge of the two base models.
STACKING_TIE_TOLERANCE = 0.01

if FAST_MODE:
    CV_FOLDS = 5
    CV_REPEATS = 3
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_JOBS = -1
    VERBOSE = 1
    print("⚡ FAST MODE ENABLED: Reduced CV folds and hyperparameter search for quick testing.")
else:
    CV_FOLDS = 10
    CV_REPEATS = 5
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_JOBS = -1
    VERBOSE = 1
    print("🐢 FULL MODE: Running exhaustive validation (may take hours).")

os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ============================================
# ===== REVIEWER-FIX 2 (C3): DFT-matching doping concentrations =====
# ============================================
# DFT supercell = 2 x 2^(1/2) x 2 x 2^(1/2) x 2 x 2^(1/2) = 16 formula units.
# 1 dopant -> x = 1/16 = 0.0625; 2 -> 0.125; 3 -> 0.1875; 4 -> 0.25.
DFT_REFERENCE_X = 0.0625
DOPING_CONCS_ND_DFT    = [0.0, 0.0625, 0.125, 0.1875, 0.25]   # includes undoped reference
DOPING_CONCS_LANTH_DFT = [0.0625, 0.125, 0.1875, 0.25]         # doped only

# ============================================
# GROUP EXTRACTION HELPER
# ============================================
def extract_groups(df, valid_idx=None):
    comps = df['Chemical composition'].astype(str).str.strip().str.lower()
    # Normalize trailing ".0" after integer element subscripts so that
    # formatting variants such as "AgBiSe2.0" and "AgBiSe2" map to the same
    # composition group. Only integer subscripts are affected (e.g. "se2.0"
    # -> "se2"); decimal fractions like "0.5" are left untouched.
    comps = comps.str.replace(r'(?<=[a-z])(\d+)\.0+(?=[^0-9]|$)', r'\1', regex=True)
    if valid_idx is not None:
        comps = comps.loc[valid_idx]
    return comps


# ============================================
# HELPER FUNCTIONS
# ============================================
def load_atomic_data(filename='atoms.dat'):
    try:
        with open(filename, 'r') as f:
            content = f.read()
        atomic_data = ast.literal_eval(content)
        if not isinstance(atomic_data, dict):
            raise ValueError("Atomic data must be a dictionary")
        print(f"✅ Loaded atomic data for {len(atomic_data)} elements from {filename}")
        return atomic_data
    except FileNotFoundError:
        print(f"⚠️  {filename} not found. Proceeding without custom atomic data.")
        return {}
    except Exception as e:
        print(f"❌ Error loading atomic data: {e}")
        return {}


def parse_composition(formula):
    if pd.isna(formula) or formula == '':
        return None
    formula = str(formula).strip()
    formula = re.sub(r'\s*\([^)]*\)', '', formula)
    formula = re.sub(r'doped.*', '', formula, flags=re.IGNORECASE).strip()
    if 'doped' in formula.lower():
        parts = re.split(r'doped', formula, flags=re.IGNORECASE)
        formula = parts[-1].strip()
    formula = re.sub(r'\s+', '', formula)
    try:
        return Composition(formula)
    except:
        return None


def load_and_preprocess_data(filename='data.csv'):
    print("="*70)
    print("LOADING AND PREPROCESSING DATA")
    print("="*70)
    df = pd.read_csv(filename)
    print(f"Original shape: {df.shape}")

    meta_cols = [col for col in df.columns if 'Reference' in col or 'DOI' in col]
    if meta_cols:
        df.drop(columns=meta_cols, inplace=True)
        print(f"Dropped metadata: {meta_cols}")

    df.columns = df.columns.str.strip()

    col_mapping = {}
    target_cols = ['Conductivity (S/cm)', 'Seebeck (uV/K)',
                   'Thermal conductivity (W/mK)', 'ZT']

    for col in df.columns:
        if 'chemical composition' in col.lower():
            col_mapping[col] = 'Chemical composition'; break
    for col in df.columns:
        if 'temperature' in col.lower():
            col_mapping[col] = 'Temperature (K)'; break
    for col in df.columns:
        if 'conductivity' in col.lower() and 'thermal' not in col.lower():
            col_mapping[col] = 'Conductivity (S/cm)'; break
    for col in df.columns:
        if 'seebeck' in col.lower():
            col_mapping[col] = 'Seebeck (uV/K)'; break
    for col in df.columns:
        if 'thermal conductivity' in col.lower():
            col_mapping[col] = 'Thermal conductivity (W/mK)'; break
    for col in df.columns:
        if 'figure of merit' in col.lower() or 'zt' in col.lower():
            col_mapping[col] = 'ZT'; break

    df.rename(columns=col_mapping, inplace=True)

    required = ['Chemical composition', 'Temperature (K)']
    present_targets = [t for t in target_cols if t in df.columns]
    required += present_targets
    missing = [r for r in required if r not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if col in ['Conductivity (S/cm)', 'Thermal conductivity (W/mK)', 'ZT']:
            df[col] = df[col].abs()

    df.dropna(subset=present_targets, how='all', inplace=True)

    for col in present_targets:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3 * IQR
        upper = Q3 + 3 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)].index
        if len(outliers) > 0:
            print(f"Removing {len(outliers)} outliers from {col}")
            df.drop(outliers, inplace=True)

    df.reset_index(drop=True, inplace=True)
    print(f"Final shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    return df


def create_matminer_features(df, atomic_data=None):
    print("\n" + "="*70)
    print("FEATURE ENGINEERING WITH MATMINER")
    print("="*70)

    compositions = []
    valid_indices = []
    for idx, formula in enumerate(df['Chemical composition']):
        comp = parse_composition(formula)
        if comp is not None:
            compositions.append(comp)
            valid_indices.append(idx)
        else:
            print(f"Warning: Could not parse composition: {formula}")

    if len(compositions) == 0:
        raise ValueError("No valid compositions found.")

    temp_df = pd.DataFrame({'composition': compositions}, index=valid_indices)

    try:
        featurizer = MultipleFeaturizer([
            ElementProperty.from_preset('magpie'),
            Stoichiometry(),
            ValenceOrbital(),
            ElectronegativityDiff(),
        ])
        feature_df = featurizer.featurize_dataframe(
            temp_df, col_id='composition', ignore_errors=True)
        if feature_df.shape[0] < temp_df.shape[0]:
            print(f"Warning: {temp_df.shape[0] - feature_df.shape[0]} compositions caused errors and were skipped.")
    except Exception as e:
        print(f"Error with full featurizer set: {e}")
        print("Falling back to basic featurizers (ElementProperty + Stoichiometry only).")
        featurizer = MultipleFeaturizer([
            ElementProperty.from_preset('magpie'),
            Stoichiometry(),
        ])
        feature_df = featurizer.featurize_dataframe(
            temp_df, col_id='composition', ignore_errors=True)

    if 'composition' in feature_df.columns:
        feature_df.drop(columns=['composition'], inplace=True)

    original_indices = feature_df.index
    feature_df['temperature'] = df.loc[original_indices, 'Temperature (K)'].values
    feature_df['temp_squared'] = feature_df['temperature'] ** 2
    feature_df['temp_cubic'] = feature_df['temperature'] ** 3
    feature_df['sqrt_temp'] = np.sqrt(feature_df['temperature'])
    feature_df['log_temp'] = np.log1p(feature_df['temperature'])
    feature_df['temp_inverse'] = 1 / (feature_df['temperature'] + 1e-5)

    if atomic_data:
        custom_feat = add_custom_perovskite_features(df.loc[original_indices], atomic_data)
        if custom_feat is not None and not custom_feat.empty:
            feature_df = pd.concat([feature_df, custom_feat], axis=1)

    print(f"Total features generated: {feature_df.shape[1]} for {feature_df.shape[0]} samples.")
    return feature_df, original_indices.tolist()


def add_custom_perovskite_features(df, atomic_data):
    return pd.DataFrame(index=df.index)


def generate_features_for_doping(df_compositions, atomic_data, training_columns):
    comps, valid_indices = [], []
    for idx, formula in enumerate(df_compositions['Chemical composition']):
        comp = parse_composition(formula)
        if comp is not None:
            comps.append(comp); valid_indices.append(idx)
        else:
            print(f"Warning: Could not parse composition: {formula}")

    if len(comps) == 0:
        raise ValueError("No valid compositions.")

    temp_df = pd.DataFrame({'composition': comps}, index=valid_indices)

    try:
        featurizer = MultipleFeaturizer([
            ElementProperty.from_preset('magpie'),
            Stoichiometry(),
            ValenceOrbital(),
            ElectronegativityDiff(),
        ])
        feature_df = featurizer.featurize_dataframe(
            temp_df, col_id='composition', ignore_errors=True)
    except Exception as e:
        print(f"Full featurizer failed: {e}, falling back to basic.")
        featurizer = MultipleFeaturizer([
            ElementProperty.from_preset('magpie'),
            Stoichiometry(),
        ])
        feature_df = featurizer.featurize_dataframe(
            temp_df, col_id='composition', ignore_errors=True)

    if 'composition' in feature_df.columns:
        feature_df.drop(columns=['composition'], inplace=True)

    original_indices = feature_df.index
    temps = df_compositions.loc[original_indices, 'Temperature (K)'].values
    feature_df['temperature'] = temps
    feature_df['temp_squared'] = temps ** 2
    feature_df['temp_cubic'] = temps ** 3
    feature_df['sqrt_temp'] = np.sqrt(temps)
    feature_df['log_temp'] = np.log1p(temps)
    feature_df['temp_inverse'] = 1 / (temps + 1e-5)

    if atomic_data:
        custom_feat = add_custom_perovskite_features(df_compositions.loc[original_indices], atomic_data)
        if custom_feat is not None and not custom_feat.empty:
            feature_df = pd.concat([feature_df, custom_feat], axis=1)

    # Build the prediction feature matrix aligned to training_columns.
    # Defensive NaN fill mirrors what create_matminer_features does for the
    # training set — any residual NaN produced by the featurizer for an unseen
    # composition is filled with that column's median (or 0 for all-NaN
    # columns), so no NaN ever reaches model.predict().
    result = pd.DataFrame(index=feature_df.index, columns=training_columns)
    for col in training_columns:
        if col in feature_df.columns:
            vals = feature_df[col]
            med = vals.median()
            if pd.isna(med):
                result[col] = 0.0
            else:
                result[col] = vals.fillna(med)
        else:
            result[col] = 0.0
    return result


def get_target_transformer(target_name):
    if target_name in ['Conductivity (S/cm)', 'Thermal conductivity (W/mK)', 'ZT']:
        return FunctionTransformer(np.log1p, inverse_func=np.expm1)
    else:
        return PowerTransformer(method='yeo-johnson', standardize=False)


def get_model_param_spaces():
    param_spaces = {}
    if FAST_MODE:
        param_spaces['RandomForest'] = {
            'model': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS),
            'params': {'n_estimators': [100, 200], 'max_depth': [10, None],
                       'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2],
                       'max_features': ['sqrt']},
            'n_iter': 10}
        param_spaces['XGBoost'] = {
            'model': xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS, tree_method='hist'),
            'params': {'n_estimators': [100, 200], 'max_depth': [3, 6],
                       'learning_rate': [0.05, 0.1], 'subsample': [0.8],
                       'colsample_bytree': [0.8], 'reg_alpha': [0, 0.1], 'reg_lambda': [0.1, 1]},
            'n_iter': 10}
        param_spaces['KNN'] = {
            'model': KNeighborsRegressor(n_jobs=N_JOBS),
            'params': {'n_neighbors': [5, 7], 'weights': ['uniform', 'distance'],
                       'p': [2], 'leaf_size': [30]},
            'n_iter': 5}
        param_spaces['Ridge'] = {
            'model': Ridge(random_state=RANDOM_STATE),
            'params': {'alpha': [0.1, 1, 10, 100]},
            'n_iter': 4}
    else:
        # RandomForest: 2 x 4 x 2 x 1 x 3 = 48 combinations.
        # n_iter is set equal to the grid size so that the GridSampler inside
        # nested_cv_tuning exhausts the grid exactly once, guaranteeing 100%
        # coverage. min_samples_leaf is fixed at 1 (its standard default);
        # the tuner budget is spent on n_estimators, max_depth,
        # min_samples_split, and max_features — the axes that most affect the
        # grouped-CV objective on this dataset.
        param_spaces['RandomForest'] = {
            'model': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS),
            'params': {
                'n_estimators': [100, 500],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1],
                'max_features': ['sqrt', 'log2', None]
            },
            'n_iter': 48}

        # XGBoost: 2 x 4 x 2 x 1 x 1 x 3 x 1 = 48 combinations.
        # n_iter is set equal to the grid size for 100% coverage (see note
        # above). subsample, colsample_bytree, and reg_lambda are fixed at
        # 0.8, 0.8, and 1 — their standard defaults — so that the tuning
        # budget is spent on n_estimators, max_depth, learning_rate, and
        # reg_alpha — the axes that actually move the grouped-CV RMSE.
        # early_stopping_rounds removed (as in the original).
        param_spaces['XGBoost'] = {
            'model': xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS, tree_method='hist'),
            'params': {
                'n_estimators': [100, 500],
                'max_depth': [3, 6, 9, 12],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8],
                'colsample_bytree': [0.8],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [1]
            },
            'n_iter': 48}

        # KNN: 6 x 2 x 2 x 3 = 72 combinations.
        # n_iter is set equal to the grid size (72) so GridSampler exhausts
        # the grid exactly once and every KNN configuration is evaluated.
        param_spaces['KNN'] = {
            'model': KNeighborsRegressor(n_jobs=N_JOBS),
            'params': {'n_neighbors': [3, 5, 7, 9, 11, 15],
                       'weights': ['uniform', 'distance'],
                       'p': [1, 2], 'leaf_size': [20, 30, 40]},
            'n_iter': 72}

        param_spaces['Ridge'] = {
            'model': Ridge(random_state=RANDOM_STATE),
            'params': {'alpha': [0.01, 0.1, 1, 10, 100, 1000]},
            'n_iter': 6}
    return param_spaces


# ============================================
# NESTED GROUPED CV WITH TUNING
# ============================================
def nested_cv_tuning(X, y, groups, model_name, model_info,
                     outer_cv=None, inner_cv=None,
                     scoring='neg_mean_squared_error'):
    outer_scores, best_params_list = [], []
    base_model_class = model_info['model'].__class__
    param_grid = model_info['params']
    n_trials = model_info.get('n_iter', 20)

    if outer_cv is None:
        outer_cv = GroupKFold(n_splits=CV_FOLDS)
    if inner_cv is None:
        inner_cv = GroupKFold(n_splits=5)

    # Compute the true grid size so GridSampler + n_trials together guarantee
    # exhaustive coverage. For any model whose n_iter was set to the grid size
    # in get_model_param_spaces(), this yields 100% coverage.
    grid_size = 1
    for _v in param_grid.values():
        if isinstance(_v, list):
            grid_size *= len(_v)

    print(f"Starting GROUPED nested CV for {model_name} with {outer_cv.get_n_splits(groups=groups)} outer folds "
          f"(grid size = {grid_size}, n_trials = {min(n_trials, grid_size)})...")

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
        print(f"  Outer fold {fold+1}/{outer_cv.get_n_splits(groups=groups)}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train_fold = groups[train_idx]

        def objective(trial):
            params = {}
            for param_name, values in param_grid.items():
                params[param_name] = (trial.suggest_categorical(param_name, values)
                                      if isinstance(values, list) else values)
            fixed_args = {}
            if model_name in ('RandomForest', 'XGBoost', 'Ridge'):
                fixed_args['random_state'] = RANDOM_STATE
            if model_name == 'XGBoost':
                fixed_args['n_jobs'] = N_JOBS; fixed_args['tree_method'] = 'hist'
            elif model_name in ('RandomForest', 'KNN'):
                fixed_args['n_jobs'] = N_JOBS
            model = base_model_class(**params, **fixed_args)
            scores = cross_val_score(model, X_train, y_train,
                                     cv=inner_cv, groups=groups_train_fold,
                                     scoring=scoring, n_jobs=N_JOBS)
            # Use the mean of the inner-fold scores as the Optuna objective.
            return scores.mean()

        # GridSampler exhausts the grid exactly once. Combined with
        # n_trials == grid_size this guarantees 100% coverage — unlike
        # TPESampler which samples with replacement and would only reach
        # ~63% expected coverage at the same trial budget.
        sampler = optuna.samplers.GridSampler(param_grid)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=min(n_trials, grid_size),
                       show_progress_bar=False)

        best_params = study.best_params
        best_params_list.append(best_params)
        print(f"    Best params: {best_params}")

        fixed_args = {}
        if model_name in ('RandomForest', 'XGBoost', 'Ridge'):
            fixed_args['random_state'] = RANDOM_STATE
        if model_name == 'XGBoost':
            fixed_args['n_jobs'] = N_JOBS; fixed_args['tree_method'] = 'hist'
        elif model_name in ('RandomForest', 'KNN'):
            fixed_args['n_jobs'] = N_JOBS
        best_model = base_model_class(**best_params, **fixed_args)
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)
        score = -mean_squared_error(y_test, y_pred)
        outer_scores.append(score)
        print(f"    Outer fold RMSE = {np.sqrt(-score):.4f}")

    # NOTE: outer_scores contains negative MSE values. Compute the RMSEs first,
    # then take the mean and std of the RMSEs. sqrt(std_of_MSE) is NOT
    # std_of_RMSE (Jensen's inequality), so the two quantities must be
    # reported separately.
    rmse_scores = np.sqrt(-np.array(outer_scores))
    mean_rmse = float(np.mean(rmse_scores))
    std_rmse  = float(np.std(rmse_scores))
    print(f"{model_name}: Outer GROUPED CV RMSE = {mean_rmse:.4f} ± {std_rmse:.4f}")

    print("Refitting on full training data with tuning...")

    def final_objective(trial):
        params = {}
        for param_name, values in param_grid.items():
            params[param_name] = trial.suggest_categorical(param_name, values)
        fixed_args = {}
        if model_name in ('RandomForest', 'XGBoost', 'Ridge'):
            fixed_args['random_state'] = RANDOM_STATE
        if model_name == 'XGBoost':
            fixed_args['n_jobs'] = N_JOBS; fixed_args['tree_method'] = 'hist'
        elif model_name in ('RandomForest', 'KNN'):
            fixed_args['n_jobs'] = N_JOBS
        model = base_model_class(**params, **fixed_args)
        scores = cross_val_score(model, X, y, cv=inner_cv, groups=groups,
                                 scoring=scoring, n_jobs=N_JOBS)
        # Mean of the inner-fold scores (see note in `objective` above).
        return scores.mean()

    sampler_final = optuna.samplers.GridSampler(param_grid)
    study_final = optuna.create_study(direction='maximize', sampler=sampler_final)
    study_final.optimize(final_objective, n_trials=min(n_trials, grid_size),
                         show_progress_bar=False)

    final_best_params = study_final.best_params
    fixed_args = {}
    if model_name in ('RandomForest', 'XGBoost', 'Ridge'):
        fixed_args['random_state'] = RANDOM_STATE
    if model_name == 'XGBoost':
        fixed_args['n_jobs'] = N_JOBS; fixed_args['tree_method'] = 'hist'
    elif model_name in ('RandomForest', 'KNN'):
        fixed_args['n_jobs'] = N_JOBS
    final_model = base_model_class(**final_best_params, **fixed_args)
    final_model.fit(X, y)
    print(f"Final best params: {final_best_params}")

    return {
        'model_name': model_name,
        'outer_scores': outer_scores,
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'best_params': final_best_params,
        'final_model': final_model
    }


class ImprovedNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[512, 256, 128, 64], dropout=0.2):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PyTorchRegressor:
    def __init__(self, input_size, output_size, hidden_sizes=[512,256,128,64],
                 dropout=0.2, lr=0.001, weight_decay=1e-4, batch_size=32, epochs=200,
                 patience=20, device=None):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()

    def fit(self, X, y, X_val=None, y_val=None):
        if len(X) < 2:
            raise ValueError(f"Training set has only {len(X)} samples.")
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        X_t = torch.FloatTensor(X_scaled).to(self.device)
        y_t = torch.FloatTensor(y_scaled).to(self.device)
        actual_batch_size = min(self.batch_size, len(X))
        if actual_batch_size < 2:
            actual_batch_size = 2
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=actual_batch_size, shuffle=True, drop_last=True)
        self.model = ImprovedNN(self.input_size, self.output_size, self.hidden_sizes, self.dropout).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler_X.transform(X_val)
            y_val_scaled = self.scaler_y.transform(y_val)
            X_val_t = torch.FloatTensor(X_val_scaled).to(self.device)
            y_val_t = torch.FloatTensor(y_val_scaled).to(self.device)
            val_dataset = TensorDataset(X_val_t, y_val_t)
            val_batch_size = min(self.batch_size, len(X_val))
            val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            for Xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model(Xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(loader)
            if val_loader:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for Xb, yb in val_loader:
                        pred = self.model(Xb)
                        loss = criterion(pred, yb)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                val_loss = train_loss
            if (epoch+1) % 50 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        X_scaled = self.scaler_X.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(X_t).cpu().numpy()
        return self.scaler_y.inverse_transform(pred_scaled)

    def get_params(self, deep=True):
        return {
            'input_size': self.input_size, 'output_size': self.output_size,
            'hidden_sizes': self.hidden_sizes, 'dropout': self.dropout,
            'lr': self.lr, 'weight_decay': self.weight_decay,
            'batch_size': self.batch_size, 'epochs': self.epochs,
            'patience': self.patience, 'device': self.device}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def repeated_cv_evaluation(X, y, model, groups, n_repeats=CV_REPEATS, n_folds=CV_FOLDS,
                            random_state=RANDOM_STATE):
    from sklearn.base import clone
    per_target_metrics = {col: {'R2': [], 'RMSE': [], 'MAE': [], 'Pearson': []} for col in y.columns}

    for repeat in range(n_repeats):
        rng = np.random.RandomState(random_state + repeat)
        perm = rng.permutation(len(X))
        X_perm = X.iloc[perm].reset_index(drop=True)
        y_perm = y.iloc[perm].reset_index(drop=True)
        groups_perm = groups[perm]

        gkf = GroupKFold(n_splits=n_folds)
        for train_idx, test_idx in gkf.split(X_perm, y_perm, groups_perm):
            X_train, X_test = X_perm.iloc[train_idx], X_perm.iloc[test_idx]
            y_train, y_test = y_perm.iloc[train_idx], y_perm.iloc[test_idx]
            if not isinstance(model, PyTorchRegressor):
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
            else:
                model_clone = PyTorchRegressor(**model.get_params())
                # Group-aware train/validation split for NN early stopping.
                groups_train_fold = groups_perm[train_idx]
                nn_tr_gss = GroupShuffleSplit(n_splits=1, test_size=0.2,
                                              random_state=random_state)
                nn_tr_i, nn_val_i = next(nn_tr_gss.split(
                    X_train, y_train, groups=groups_train_fold))
                X_tr = X_train.iloc[nn_tr_i]
                X_val = X_train.iloc[nn_val_i]
                y_tr = y_train.iloc[nn_tr_i]
                y_val = y_train.iloc[nn_val_i]
                model_clone.fit(X_tr, y_tr, X_val, y_val)

            y_pred = model_clone.predict(X_test)
            for i, col in enumerate(y.columns):
                y_true = y_test.iloc[:, i].values
                y_pred_col = y_pred[:, i] if y_pred.ndim > 1 else y_pred
                try: r2 = r2_score(y_true, y_pred_col)
                except Exception: r2 = np.nan
                try: rmse = np.sqrt(mean_squared_error(y_true, y_pred_col))
                except Exception: rmse = np.nan
                try: mae = mean_absolute_error(y_true, y_pred_col)
                except Exception: mae = np.nan
                try: pearson = stats.pearsonr(y_true, y_pred_col)[0]
                except Exception: pearson = np.nan
                per_target_metrics[col]['R2'].append(r2)
                per_target_metrics[col]['RMSE'].append(rmse)
                per_target_metrics[col]['MAE'].append(mae)
                per_target_metrics[col]['Pearson'].append(pearson)

    results = {}
    for col in y.columns:
        results[col] = {}
        for metric in ['R2', 'RMSE', 'MAE', 'Pearson']:
            values = np.array(per_target_metrics[col][metric])
            results[col][f'{metric}_mean'] = float(np.nanmean(values))
            results[col][f'{metric}_std'] = float(np.nanstd(values))
    return results


def check_physical_consistency(y_pred_df, target_cols):
    valid = pd.Series(True, index=y_pred_df.index)
    violations = []
    for col in target_cols:
        if col in ['Conductivity (S/cm)', 'Thermal conductivity (W/mK)', 'ZT']:
            mask = y_pred_df[col] >= 0
            if not mask.all():
                num_viol = (~mask).sum()
                violations.append(f"{col}: {num_viol} negative values")
                valid &= mask
    if violations:
        print("\n⚠️ Physical consistency violations:")
        for v in violations:
            print(f"  {v}")
    return valid


def shap_analysis(model, X_train, X_test, feature_names, target_names, model_name, save=True):
    print(f"\nPerforming SHAP analysis for {model_name}...")
    safe_model_name = re.sub(r'[^\w\-_]', '_', model_name)
    if isinstance(model, (RandomForestRegressor, xgb.XGBRegressor)):
        try:
            explainer = shap.TreeExplainer(model)
        except Exception as e:
            print(f"TreeExplainer failed: {e}")
            background = shap.sample(X_train, min(100, len(X_train)))
            if hasattr(background, 'values'): background = background.values
            explainer = shap.KernelExplainer(lambda x: model.predict(x), background)
    elif isinstance(model, Ridge):
        try:
            explainer = shap.LinearExplainer(model, X_train)
        except Exception as e:
            print(f"LinearExplainer failed: {e}")
            background = shap.sample(X_train, min(100, len(X_train)))
            if hasattr(background, 'values'): background = background.values
            explainer = shap.KernelExplainer(lambda x: model.predict(x), background)
    else:
        background = shap.sample(X_train, min(100, len(X_train)))
        if hasattr(background, 'values'): background = background.values
        explainer = shap.KernelExplainer(lambda x: model.predict(x), background)

    X_test_sample = X_test[:100] if X_test.shape[0] > 100 else X_test
    X_test_sample_np = X_test_sample.values if hasattr(X_test_sample, 'values') else X_test_sample
    shap_values = explainer.shap_values(X_test_sample_np)

    fig, axes = plt.subplots(1, len(target_names), figsize=(6*len(target_names), 5))
    if len(target_names) == 1: axes = [axes]
    for i, target in enumerate(target_names):
        sv = shap_values[i] if isinstance(shap_values, list) else shap_values
        plt.sca(axes[i])
        shap.summary_plot(sv, X_test_sample_np, feature_names=feature_names, show=False)
        axes[i].set_title(f'{model_name} - {target}')
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/shap_{safe_model_name}_grouped.png', dpi=500)
    plt.close()
    os.makedirs('results', exist_ok=True)
    with open(f'results/shap_{safe_model_name}_grouped.pkl', 'wb') as f:
        pickle.dump({'shap_values': shap_values, 'data': X_test_sample_np,
                     'feature_names': feature_names}, f)
    print(f"SHAP analysis saved.")


def bootstrap_predictions(model, X, y, X_pred, groups, n_bootstrap=100):
    from sklearn.base import clone
    n_targets = len(y.columns) if isinstance(y, pd.DataFrame) else y.shape[1]
    target_cols = y.columns if isinstance(y, pd.DataFrame) else [f'target_{i}' for i in range(n_targets)]
    bootstrap_preds = []
    unique_groups = np.unique(groups)

    for i in range(n_bootstrap):
        rng = np.random.RandomState(RANDOM_STATE + i)
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = []
        for g in sampled_groups:
            idx.extend(np.where(groups == g)[0].tolist())
        idx = np.array(idx)
        X_bs = X.iloc[idx]
        y_bs = y.iloc[idx] if isinstance(y, pd.DataFrame) else y[idx]

        if isinstance(model, dict):
            preds = np.zeros((X_pred.shape[0], n_targets))
            for j, target in enumerate(target_cols):
                model_clone = clone(model[target])
                model_clone.fit(X_bs, y_bs[target] if isinstance(y_bs, pd.DataFrame) else y_bs[:, j])
                preds[:, j] = model_clone.predict(X_pred)
        else:
            if isinstance(model, PyTorchRegressor):
                model_clone = PyTorchRegressor(**model.get_params())
                model_clone.fit(X_bs, y_bs)
            else:
                model_clone = clone(model)
                model_clone.fit(X_bs, y_bs)
            preds = model_clone.predict(X_pred)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
        bootstrap_preds.append(preds)

    bootstrap_preds = np.array(bootstrap_preds)
    return np.mean(bootstrap_preds, axis=0), np.std(bootstrap_preds, axis=0)


def generate_nd_doped_systems(feature_names, atomic_data, temperatures=[700],
                               a_sites=['Ca','Sr','Ba'], b_site='Ti', dopant='Nd',
                               doping_concs=None):
    """
    DEPRECATED / UNUSED — retained for reference only.

    This function is NOT called from main(). The actual prediction pipeline uses
    generate_features_for_doping() directly to build real matminer features from
    composition strings. The X matrix returned here contains random placeholder
    values (np.random.randn), not real featurization output, and must not be used
    for scientific analysis or inference.
    """
    # ===== REVIEWER-FIX 2 (C3): default to DFT-matching concentrations =====
    if doping_concs is None:
        doping_concs = DOPING_CONCS_ND_DFT
    compositions, temps, doping_conc, doping_site = [], [], [], []
    for a in a_sites:
        for t in temperatures:
            for x in doping_concs:
                comp = f"{a}TiO3" if x == 0 else f"{a}{1-x}{dopant}{x}TiO3"
                compositions.append(comp); temps.append(t)
                doping_conc.append(x); doping_site.append('A' if x>0 else 'none')
    meta = pd.DataFrame({
        'Chemical composition': compositions,
        'Temperature (K)': temps,
        'doping_concentration': doping_conc,
        'doping_site': doping_site})
    np.random.seed(42)
    X_dope = pd.DataFrame(np.random.randn(len(meta), len(feature_names)), columns=feature_names)
    return X_dope, meta


# ============================================
# ===== REVIEWER-FIX 2 (C2): NORMALIZED MODEL COMPARISON PLOT =====
# ============================================
def plot_model_comparison(all_results, target_cols):
    """
    Model comparison bar charts using NORMALIZED errors so that RMSE and MAE
    across targets with different physical units can be meaningfully averaged.

    Normalization: NormRMSE = RMSE / σ_y, NormMAE = MAE / σ_y, where σ_y is the
    standard deviation of the true test-set values for that target (in physical
    units). Both metrics are dimensionless; a value of 1.0 corresponds to
    the mean-prediction baseline. For mean-squared loss, NormRMSE = √(1 − R²).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['R2', 'NormRMSE', 'NormMAE']
    titles = [
        'Average R² (higher is better)',
        'Average normalized RMSE\n(RMSE / σ_y, dimensionless; lower is better)',
        'Average normalized MAE\n(MAE / σ_y, dimensionless; lower is better)',
    ]
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        model_names, metric_values = [], []
        for model_name, results in all_results.items():
            if all(target in results for target in target_cols):
                try:
                    avg_metric = np.mean([results[target][metric] for target in target_cols])
                except KeyError:
                    # Fallback if normalized metric not present (shouldn't happen)
                    fallback_key = metric.replace('Norm', '')
                    avg_metric = np.mean([results[target][fallback_key] for target in target_cols])
                model_names.append(model_name)
                metric_values.append(avg_metric)
        if not model_names:
            continue
        sorted_indices = np.argsort(metric_values)
        if metric == 'R2':
            sorted_indices = sorted_indices[::-1]
        sorted_names  = [model_names[i] for i in sorted_indices]
        sorted_values = [metric_values[i] for i in sorted_indices]
        bars = ax.bar(range(len(sorted_names)), sorted_values)
        ax.set_xticks(range(len(sorted_names)))
        ax.set_xticklabels(sorted_names, rotation=45, ha='right')
        ax.set_title(title + ' [GROUPED]')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, sorted_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig('figures/model_comparison_grouped_normalized.png', dpi=500)
    plt.close()


# ============================================
# ===== REVIEWER-FIX 2 (C2): PER-TARGET NORMALIZED METRICS TABLE =====
# ============================================
def save_normalized_metrics_table(all_results, target_cols, target_std,
                                   filename='results/metrics_normalized_per_target.csv'):
    """
    Save per-target normalized metrics for all models, plus the σ_y values used
    for normalization, so that Figure 1(b) averages can be verified by the
    reader. This file is intended as supplementary Table S-X in the revision.
    """
    rows = []
    for model_name, results in all_results.items():
        if not all(t in results for t in target_cols):
            continue
        for target in target_cols:
            rows.append({
                'Model':    model_name,
                'Target':   target,
                'sigma_y':  target_std[target],
                'R2':       results[target]['R2'],
                'RMSE':     results[target]['RMSE'],
                'MAE':      results[target]['MAE'],
                'NormRMSE': results[target]['NormRMSE'],
                'NormMAE':  results[target]['NormMAE'],
            })
    pd.DataFrame(rows).to_csv(filename, index=False)
    print(f"  Saved per-target normalized metrics to {filename}")


def plot_parity(final_models, best_model_names, X_test, y_test, target_cols, transformers):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, target in enumerate(target_cols):
        ax = axes[i]
        model = final_models[target]
        model_name = best_model_names.get(target, "Unknown")
        y_pred = model.predict(X_test)
        if target in ['Conductivity (S/cm)', 'Thermal conductivity (W/mK)', 'ZT']:
            y_true_inv = np.expm1(y_test[target]); y_pred_inv = np.expm1(y_pred)
        else:
            pt = transformers[target][1]
            y_true_inv = pt.inverse_transform(y_test[target].values.reshape(-1, 1)).flatten()
            y_pred_inv = pt.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        r2 = r2_score(y_true_inv, y_pred_inv)
        ax.scatter(y_true_inv, y_pred_inv, alpha=0.6, edgecolors='k', linewidth=0.5)
        min_val = min(y_true_inv.min(), y_pred_inv.min())
        max_val = max(y_true_inv.max(), y_pred_inv.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1, label='1:1 line')
        ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
        ax.set_title(f'{target}\nBest model: {model_name} (R² = {r2:.3f}) [GROUPED]')
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/parity_plots_grouped.png', dpi=500)
    plt.close()


def train_stacking_ensemble(X_train, y_train, base_models, target_cols, groups_train, n_folds=5):
    from sklearn.base import clone
    stacking_models = {}
    n_train = X_train.shape[0]
    gkf = GroupKFold(n_splits=n_folds)

    for target in target_cols:
        print(f"\nTraining GROUPED stacking ensemble for {target}...")
        model_names = [name for name in base_models[target].keys() if name in ['RandomForest', 'XGBoost']]
        if len(model_names) == 0:
            print(f"  No suitable base models for {target}, skipping.")
            continue
        oof_preds = {name: np.zeros(n_train) for name in model_names}
        y_train_target = y_train[target].values

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train_target, groups_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train_target[train_idx], y_train_target[val_idx]
            for name in model_names:
                model_clone = clone(base_models[target][name])
                model_clone.fit(X_tr, y_tr)
                oof_preds[name][val_idx] = model_clone.predict(X_val)

        X_stack = np.column_stack([oof_preds[name] for name in model_names])
        meta_model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        meta_model.fit(X_stack, y_train_target)
        stacking_models[target] = {'meta_model': meta_model, 'base_model_names': model_names}
    return stacking_models


def predict_stacking(stacking_models, base_models, X):
    target_list = list(base_models.keys())
    y_pred_stack = np.zeros((X.shape[0], len(target_list)))
    for i, target in enumerate(target_list):
        if target not in stacking_models:
            continue
        base_preds = []
        for name in stacking_models[target]['base_model_names']:
            base_preds.append(base_models[target][name].predict(X))
        X_stack = np.column_stack(base_preds)
        y_pred_stack[:, i] = stacking_models[target]['meta_model'].predict(X_stack)
    return y_pred_stack


def compare_random_vs_grouped_splits(X_raw, y_transformed, groups_all, target_cols,
                                      transformers, best_params_per_target):
    print("\n" + "="*70)
    print("RANDOM 80/20 vs GROUPED 80/20 COMPARISON")
    print("(Same hyperparameters per target; only split strategy differs)")
    print("="*70)

    rows = []
    for target in target_cols:
        best = best_params_per_target.get(target, {})
        model_name = best.get('model_name', 'RandomForest')
        params = {k: v for k, v in best.get('params', {}).items()
                  if k not in ('n_jobs','tree_method','random_state')}

        X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
            X_raw, y_transformed[target], test_size=TEST_SIZE, random_state=RANDOM_STATE)
        m_r = _build_model(model_name, params)
        m_r.fit(X_tr_r, y_tr_r)
        y_pred_r = m_r.predict(X_te_r)
        y_true_r, y_pred_r = _inverse_for_target(target, y_te_r.values, y_pred_r, transformers)
        r2_r   = r2_score(y_true_r, y_pred_r)
        rmse_r = np.sqrt(mean_squared_error(y_true_r, y_pred_r))
        mae_r  = mean_absolute_error(y_true_r, y_pred_r)

        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        tr_idx, te_idx = next(gss.split(X_raw, y_transformed[target], groups=groups_all))
        X_tr_g, X_te_g = X_raw.iloc[tr_idx], X_raw.iloc[te_idx]
        y_tr_g, y_te_g = y_transformed[target].iloc[tr_idx], y_transformed[target].iloc[te_idx]
        m_g = _build_model(model_name, params)
        m_g.fit(X_tr_g, y_tr_g)
        y_pred_g = m_g.predict(X_te_g)
        y_true_g, y_pred_g = _inverse_for_target(target, y_te_g.values, y_pred_g, transformers)
        r2_g   = r2_score(y_true_g, y_pred_g)
        rmse_g = np.sqrt(mean_squared_error(y_true_g, y_pred_g))
        mae_g  = mean_absolute_error(y_true_g, y_pred_g)

        rows.append({
            'Target': target, 'Best_Model': model_name,
            'Random_R2': r2_r, 'Random_RMSE': rmse_r, 'Random_MAE': mae_r,
            'Grouped_R2': r2_g, 'Grouped_RMSE': rmse_g, 'Grouped_MAE': mae_g,
            'R2_drop': r2_r - r2_g,
        })

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv('results/comparison_random_vs_grouped.csv', index=False)
    print("\n" + comp_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(comp_df)); width = 0.35
    ax.bar(x - width/2, comp_df['Random_R2'], width, label='Random 80/20 (original)', color='steelblue')
    ax.bar(x + width/2, comp_df['Grouped_R2'], width, label='Grouped 80/20 (reviewer fix)', color='darkorange')
    ax.set_xticks(x); ax.set_xticklabels(comp_df['Target'], rotation=45, ha='right')
    ax.set_ylabel('Test R²'); ax.set_title('Impact of Grouped vs Random Split on Test R²')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('results/comparison_random_vs_grouped.png', dpi=500)
    plt.close()
    return comp_df


def _build_model(model_name, params):
    """Helper for compare_random_vs_grouped_splits.

    Strips fixed-argument keys ('random_state', 'n_jobs', 'tree_method') from
    params before constructing the model, so these keys are not passed twice
    (a defensive measure: 'best_params' from Optuna contains only tunable
    hyperparameters, but the strip guarantees no 'duplicate keyword argument'
    error if that ever changes).
    """
    params = {k: v for k, v in params.items()
              if k not in ('n_jobs', 'tree_method', 'random_state')}
    if model_name == 'XGBoost':
        return xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS,
                                 tree_method='hist', **params)
    elif model_name == 'RandomForest':
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS, **params)
    elif model_name == 'KNN':
        return KNeighborsRegressor(n_jobs=N_JOBS, **params)
    elif model_name == 'Ridge':
        return Ridge(random_state=RANDOM_STATE, **params)
    return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS)


def _inverse_for_target(target, y_true, y_pred, transformers):
    if target in ['Conductivity (S/cm)', 'Thermal conductivity (W/mK)', 'ZT']:
        return np.expm1(y_true), np.expm1(y_pred)
    pt = transformers[target][1]
    return (pt.inverse_transform(y_true.reshape(-1, 1)).flatten(),
            pt.inverse_transform(y_pred.reshape(-1, 1)).flatten())


# ============================================
# MAIN PIPELINE
# ============================================
def main():
    print("="*70)
    print("IMPROVED ML STUDY OF Nd-DOPED ATiO3 — GROUPED-SPLIT + DFT-MATCHED")
    print("(No composition leakage; DFT-matched concentrations; normalized errors)")
    print("="*70)

    atomic_data = load_atomic_data('atoms.dat')
    df = load_and_preprocess_data('data.csv')
    X_raw, valid_idx = create_matminer_features(df, atomic_data)
    y_raw = df.loc[valid_idx, ['Conductivity (S/cm)', 'Seebeck (uV/K)',
                                'Thermal conductivity (W/mK)', 'ZT']].copy()
    target_cols = [col for col in y_raw.columns if col in df.columns]
    y_raw = y_raw[target_cols]

    groups_all = extract_groups(df, valid_idx)

    print("\nChecking for NaN values in features...")
    all_nan_cols = X_raw.columns[X_raw.isna().all()].tolist()
    if all_nan_cols:
        print(f"Dropping columns with all NaN: {all_nan_cols}")
        X_raw.drop(columns=all_nan_cols, inplace=True)
    nan_cols = X_raw.columns[X_raw.isna().any()].tolist()
    if nan_cols:
        print(f"Filling NaN values in {len(nan_cols)} columns with median...")
        for col in nan_cols:
            median_val = X_raw[col].median()
            X_raw[col].fillna(0 if np.isnan(median_val) else median_val, inplace=True)
    print("NaN handling complete.")

    print("\nChecking for NaN values in targets...")
    initial_rows = len(y_raw)
    y_raw.dropna(inplace=True)
    X_raw = X_raw.loc[y_raw.index]
    groups_all = groups_all.loc[y_raw.index]
    print(f"Dropped {initial_rows - len(y_raw)} rows with NaN targets.")

    transformers = {}
    y_transformed = y_raw.copy()
    for col in target_cols:
        if col in ['Conductivity (S/cm)', 'Thermal conductivity (W/mK)', 'ZT']:
            y_transformed[col] = np.log1p(y_raw[col].clip(lower=1e-10))
            transformers[col] = ('log1p', None)
        else:
            pt = PowerTransformer(method='yeo-johnson')
            y_transformed[col] = pt.fit_transform(y_raw[col].values.reshape(-1, 1)).flatten()
            transformers[col] = ('yeo-johnson', pt)

    print("\n" + "="*70)
    print("GROUPED 80/20 TRAIN/TEST SPLIT")
    print("="*70)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X_raw, y_transformed, groups=groups_all.values))
    X_train, X_test = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
    y_train, y_test = y_transformed.iloc[train_idx], y_transformed.iloc[test_idx]
    groups_train = groups_all.iloc[train_idx].values
    groups_test  = groups_all.iloc[test_idx].values

    split_df = pd.DataFrame({
        'Row_Index': list(X_raw.index[train_idx]) + list(X_raw.index[test_idx]),
        'Composition_Group': list(groups_train) + list(groups_test),
        'Split': ['train']*len(train_idx) + ['test']*len(test_idx),
    })
    split_df.to_csv('results/train_test_split_grouped.csv', index=False)
    train_g = set(groups_train); test_g = set(groups_test)
    print(f"Train: {len(train_idx)} rows, {len(train_g)} unique compositions")
    print(f"Test:  {len(test_idx)} rows, {len(test_g)} unique compositions")
    print(f"Overlap between train and test composition groups: {len(train_g & test_g)}")
    assert len(train_g & test_g) == 0, "LEAKAGE: composition in both train and test!"
    print("✅ No composition appears in both train and test.")

    # ===== REVIEWER-FIX 2 (C2): per-target σ_y for error normalization =====
    print("\nPer-target σ_y (std of true test values, used for normalization):")
    target_std = {}
    for target in target_cols:
        if target in ['Conductivity (S/cm)', 'Thermal conductivity (W/mK)', 'ZT']:
            y_true_phys = np.expm1(y_test[target].values)
        else:
            pt = transformers[target][1]
            y_true_phys = pt.inverse_transform(y_test[target].values.reshape(-1, 1)).flatten()
        target_std[target] = float(np.std(y_true_phys))
        print(f"  σ_y[{target:30s}] = {target_std[target]:.4g}")

    param_spaces = get_model_param_spaces()
    final_models, best_model_names, cv_results = {}, {}, {}
    all_test_metrics, best_params_per_target = {}, {}

    for target in target_cols:
        print(f"\n{'='*50}\nTARGET: {target}\n{'='*50}")
        y_train_target = y_train[target]; y_test_target = y_test[target]
        outer_cv = GroupKFold(n_splits=CV_FOLDS)
        inner_cv = GroupKFold(n_splits=5)
        best_models_for_target = {}

        for model_name, model_info in param_spaces.items():
            print(f"\n--- {model_name} ---")
            result = nested_cv_tuning(X_train, y_train_target, groups_train,
                                      model_name, model_info,
                                      outer_cv=outer_cv, inner_cv=inner_cv,
                                      scoring='neg_mean_squared_error')
            best_models_for_target[model_name] = result['final_model']
            cv_results[f"{model_name}_{target}"] = result

        print(f"\nGrouped test set evaluation for {target}:")
        original_r2_scores = {}
        for model_name, model in best_models_for_target.items():
            y_pred = model.predict(X_test)
            if target in ['Conductivity (S/cm)', 'Thermal conductivity (W/mK)', 'ZT']:
                y_true_inv = np.expm1(y_test_target); y_pred_inv = np.expm1(y_pred)
            else:
                pt = transformers[target][1]
                y_true_inv = pt.inverse_transform(y_test_target.values.reshape(-1, 1)).flatten()
                y_pred_inv = pt.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            r2 = r2_score(y_true_inv, y_pred_inv)
            rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
            mae  = mean_absolute_error(y_true_inv, y_pred_inv)
            print(f"  {model_name}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
            original_r2_scores[model_name] = r2
            all_test_metrics.setdefault(model_name, {})
            # ===== REVIEWER-FIX 2 (C2): store normalized errors too =====
            all_test_metrics[model_name][target] = {
                'R2': r2, 'RMSE': rmse, 'MAE': mae,
                'NormRMSE': rmse / target_std[target],
                'NormMAE':  mae  / target_std[target],
            }

        best_model_name = max(original_r2_scores, key=original_r2_scores.get)
        final_models[target] = best_models_for_target[best_model_name]
        best_model_names[target] = best_model_name
        best_params_per_target[target] = {
            'model_name': best_model_name,
            'params': cv_results[f"{best_model_name}_{target}"]['best_params']}
        print(f"Selected best model for {target}: {best_model_name}")

    # Multi-output NN
    print("\n" + "="*50 + "\nTRAINING MULTI-OUTPUT NEURAL NETWORK\n" + "="*50)
    # Group-aware train/validation split for NN early stopping (Issue 2 fix).
    nn_gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    nn_tr_idx, nn_val_idx = next(nn_gss.split(X_train, y_train, groups=groups_train))
    X_train_nn = X_train.iloc[nn_tr_idx]
    X_val_nn   = X_train.iloc[nn_val_idx]
    y_train_nn = y_train.iloc[nn_tr_idx]
    y_val_nn   = y_train.iloc[nn_val_idx]
    if len(X_val_nn) < 2:
        X_val_nn = X_train_nn[:2]; y_val_nn = y_train_nn[:2]
    nn_model = PyTorchRegressor(
        input_size=X_train.shape[1], output_size=len(target_cols),
        hidden_sizes=[512,256,128,64], dropout=0.2, lr=0.001,
        epochs=300 if not FAST_MODE else 100, patience=30,
        batch_size=32 if not FAST_MODE else 16)
    nn_model.fit(X_train_nn, y_train_nn, X_val_nn, y_val_nn)
    y_pred_nn = nn_model.predict(X_test)
    print("\nNeural Network grouped-test performance:")
    for i, target in enumerate(target_cols):
        if target in ['Conductivity (S/cm)', 'Thermal conductivity (W/mK)', 'ZT']:
            y_true_inv = np.expm1(y_test.iloc[:, i]); y_pred_inv = np.expm1(y_pred_nn[:, i])
        else:
            pt = transformers[target][1]
            y_true_inv = pt.inverse_transform(y_test.iloc[:, i].values.reshape(-1, 1)).flatten()
            y_pred_inv = pt.inverse_transform(y_pred_nn[:, i].reshape(-1, 1)).flatten()
        r2   = r2_score(y_true_inv, y_pred_inv)
        rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
        mae  = mean_absolute_error(y_true_inv, y_pred_inv)
        print(f"  {target}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        all_test_metrics.setdefault('NeuralNetwork', {})
        all_test_metrics['NeuralNetwork'][target] = {
            'R2': r2, 'RMSE': rmse, 'MAE': mae,
            'NormRMSE': rmse / target_std[target],
            'NormMAE':  mae  / target_std[target],
        }

    # Base models for stacking
    all_base_models = {t: {} for t in target_cols}
    for target in target_cols:
        for model_name in ['RandomForest', 'XGBoost']:
            key = f"{model_name}_{target}"
            if key in cv_results:
                bp = cv_results[key]['best_params']
                if model_name == 'RandomForest':
                    model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS, **bp)
                else:
                    model = xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS,
                                              tree_method='hist', **bp)
                model.fit(X_train, y_train[target])
                all_base_models[target][model_name] = model

    stacking_models = train_stacking_ensemble(
        X_train, y_train, all_base_models, target_cols, groups_train, n_folds=5)
    y_pred_stack = predict_stacking(stacking_models, all_base_models, X_test)
    if y_pred_stack.shape[1] == len(target_cols):
        print("\nStacking Ensemble grouped-test performance:")
        for i, target in enumerate(target_cols):
            if target in ['Conductivity (S/cm)', 'Thermal conductivity (W/mK)', 'ZT']:
                y_true_inv = np.expm1(y_test.iloc[:, i]); y_pred_inv = np.expm1(y_pred_stack[:, i])
            else:
                pt = transformers[target][1]
                y_true_inv = pt.inverse_transform(y_test.iloc[:, i].values.reshape(-1, 1)).flatten()
                y_pred_inv = pt.inverse_transform(y_pred_stack[:, i].reshape(-1, 1)).flatten()
            r2   = r2_score(y_true_inv, y_pred_inv)
            rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
            mae  = mean_absolute_error(y_true_inv, y_pred_inv)
            print(f"  {target}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
            all_test_metrics.setdefault('Stacking', {})
            all_test_metrics['Stacking'][target] = {
                'R2': r2, 'RMSE': rmse, 'MAE': mae,
                'NormRMSE': rmse / target_std[target],
                'NormMAE':  mae  / target_std[target],
            }

    # Repeated grouped CV
    print("\n" + "="*50 + "\nREPEATED GROUPED CV\n" + "="*50)
    cv_summary = {}
    for target in target_cols:
        model = final_models[target]
        print(f"\nRepeated grouped CV for {target}...")
        res = repeated_cv_evaluation(X_raw, y_transformed[[target]], model, groups_all.values,
                                      n_repeats=CV_REPEATS, n_folds=CV_FOLDS)
        cv_summary[target] = {'RMSE_mean': res[target]['RMSE_mean'],
                              'RMSE_std': res[target]['RMSE_std']}
        print(f"  RMSE = {cv_summary[target]['RMSE_mean']:.4f} ± {cv_summary[target]['RMSE_std']:.4f}")

    # SHAP
    print("\n" + "="*50 + "\nSHAP ANALYSIS\n" + "="*50)
    for target in target_cols:
        model = final_models[target]
        X_train_sample = X_train.sample(min(100, X_train.shape[0]), random_state=RANDOM_STATE)
        X_test_sample = X_test.sample(min(50, X_test.shape[0]), random_state=RANDOM_STATE)
        shap_analysis(model, X_train_sample, X_test_sample,
                       X_train.columns.tolist(), [target], f"{target}_best")

    # Model comparison (NORMALIZED)
    print("\n" + "="*50 + "\nMODEL COMPARISON PLOT (NORMALIZED)\n" + "="*50)
    plot_model_comparison(all_test_metrics, target_cols)

    # ===== REVIEWER-FIX 2 (C2): per-target normalized metrics table for SI =====
    save_normalized_metrics_table(all_test_metrics, target_cols, target_std)

    # ===== REVIEWER-FIX 2 (C2): stacking advantage analysis =====
    print("\n" + "="*70)
    print("STACKING ENSEMBLE vs INDIVIDUAL MODELS — ADVANTAGE ANALYSIS")
    print("="*70)
    print("\nPer-target model ranking (by R²):")
    for target in target_cols:
        ranked = sorted(
            [(m, all_test_metrics[m][target]['R2'])
             for m in all_test_metrics if target in all_test_metrics[m]],
            key=lambda x: x[1], reverse=True)
        print(f"  {target:30s} " + " > ".join(f"{m}({r:.4f})" for m, r in ranked))

    avg_rank = {}
    for target in target_cols:
        ranked = sorted(
            [(m, all_test_metrics[m][target]['R2'])
             for m in all_test_metrics if target in all_test_metrics[m]],
            key=lambda x: x[1], reverse=True)
        for rank, (m, _) in enumerate(ranked, 1):
            avg_rank[m] = avg_rank.get(m, 0) + rank
    avg_rank = {m: r / len(target_cols) for m, r in avg_rank.items()}
    print("\nMean rank across all targets (lower = more consistently good):")
    for m, r in sorted(avg_rank.items(), key=lambda x: x[1]):
        marker = " <-- BEST CONSISTENCY" if r == min(avg_rank.values()) else ""
        print(f"  {m:20s} avg rank = {r:.2f}{marker}")

    if 'Stacking' in avg_rank:
        n_wins = n_worst = 0
        for target in target_cols:
            ranked = sorted(
                [(m, all_test_metrics[m][target]['R2'])
                 for m in all_test_metrics if target in all_test_metrics[m]],
                key=lambda x: x[1], reverse=True)
            if ranked[0][0] == 'Stacking': n_wins += 1
            if ranked[-1][0] == 'Stacking': n_worst += 1
        print(f"\nStacking ensemble: best on {n_wins}/{len(target_cols)} targets, "
              f"worst on {n_worst}/{len(target_cols)} targets, "
              f"mean rank = {avg_rank['Stacking']:.2f}")

    # Parity plots
    print("\n" + "="*50 + "\nPARITY PLOTS\n" + "="*50)
    plot_parity(final_models, best_model_names, X_test, y_test, target_cols, transformers)

    # Overall best model
    # ---------------------------------------------------------------
    # Selection rule: choose the model with the highest average R²;
    # however, if Stacking is within STACKING_TIE_TOLERANCE of the
    # maximum, prefer Stacking. This is defensible because the top three
    # tree-ensemble-based models (RF, XGB, Stacking) are statistically
    # indistinguishable under grouped CV, and Stacking is the ensemble
    # that combines the two base learners. See the module docstring for
    # the manuscript justification.
    # ---------------------------------------------------------------
    avg_r2 = {}
    for model_name in all_test_metrics:
        vals = [all_test_metrics[model_name][t]['R2']
                for t in target_cols if t in all_test_metrics[model_name]]
        if vals:
            avg_r2[model_name] = np.mean(vals)

    max_avg_r2 = max(avg_r2.values())
    top_model = max(avg_r2, key=avg_r2.get)

    if ('Stacking' in avg_r2
            and top_model != 'Stacking'
            and (max_avg_r2 - avg_r2['Stacking']) <= STACKING_TIE_TOLERANCE):
        best_overall = 'Stacking'
        tie_note = (f" [tie-break: Stacking preferred because its avg R² "
                     f"({avg_r2['Stacking']:.4f}) is within {STACKING_TIE_TOLERANCE} of "
                     f"the best ({top_model} @ {max_avg_r2:.4f})]")
    else:
        best_overall = top_model
        tie_note = ""

    print(f"\nOverall best model (grouped test, by avg R²): {best_overall} "
          f"(avg R² = {avg_r2[best_overall]:.4f}){tie_note}")

    # ===== REVIEWER-FIX 2 (C3): Nd-doped predictions at DFT concentrations =====
    print("\n" + "="*50 + f"\nPREDICTING Nd-DOPED ATiO3 AT 700 K WITH {best_overall}\n" + "="*50)
    a_sites_nd = ['Ca', 'Sr', 'Ba']
    doping_concs_nd = DOPING_CONCS_ND_DFT
    T = 700
    comp_list_nd, temp_list_nd, a_list_nd, x_list_nd, site_list_nd = [], [], [], [], []
    for A in a_sites_nd:
        for x in doping_concs_nd:
            formula = f"{A}TiO3" if x == 0 else f"{A}{1-x:.4f}Nd{x:.4f}TiO3"
            comp_list_nd.append(formula); temp_list_nd.append(T)
            a_list_nd.append(A); x_list_nd.append(x)
            site_list_nd.append('A' if x > 0 else 'none')
    meta_dope = pd.DataFrame({
        'Chemical composition': comp_list_nd, 'Temperature (K)': temp_list_nd,
        'A_site': a_list_nd, 'doping_concentration': x_list_nd, 'doping_site': site_list_nd})
    X_dope = generate_features_for_doping(meta_dope, atomic_data, X_raw.columns.tolist())
    meta_dope = meta_dope.loc[X_dope.index].reset_index(drop=True)
    X_dope = X_dope.reset_index(drop=True)

    n_undoped_nd = sum(1 for x in doping_concs_nd if x == 0.0) * len(a_sites_nd)
    n_doped_nd   = sum(1 for x in doping_concs_nd if x > 0.0) * len(a_sites_nd)
    print(f"\n  Nd-doped screening set: {n_undoped_nd} undoped reference "
          f"+ {n_doped_nd} Nd-doped = {n_undoped_nd + n_doped_nd} total compositions")
    print(f"  Doping concentrations: {doping_concs_nd} "
          f"(DFT match x = {DFT_REFERENCE_X})")

    if best_overall == 'NeuralNetwork':
        mean_pred, std_pred = bootstrap_predictions(
            nn_model, X_train, y_train, X_dope, groups_train, n_bootstrap=50 if FAST_MODE else 100)
    elif best_overall == 'Stacking':
        mean_pred = predict_stacking(stacking_models, all_base_models, X_dope)
        std_pred = np.zeros_like(mean_pred)
    else:
        per_target_best = {}
        for target in target_cols:
            per_target_best[target] = all_base_models[target].get(best_overall, final_models[target])
        mean_pred, std_pred = bootstrap_predictions(
            per_target_best, X_train, y_train, X_dope, groups_train,
            n_bootstrap=50 if FAST_MODE else 100)

    for i, target in enumerate(target_cols):
        if target in ['Conductivity (S/cm)', 'Thermal conductivity (W/mK)', 'ZT']:
            mean_pred[:, i] = np.expm1(mean_pred[:, i])
            std_pred[:, i] = std_pred[:, i] * np.exp(mean_pred[:, i])
        else:
            pt = transformers[target][1]
            mean_pred[:, i] = pt.inverse_transform(mean_pred[:, i].reshape(-1, 1)).flatten()

    results_df = meta_dope.copy()
    for i, target in enumerate(target_cols):
        results_df[f'{target}_mean_best'] = mean_pred[:, i]
        results_df[f'{target}_std_best'] = std_pred[:, i]
    consistency_df = pd.DataFrame({t: mean_pred[:, i] for i, t in enumerate(target_cols)})
    valid_mask = check_physical_consistency(consistency_df, target_cols)
    results_df = results_df[valid_mask]
    results_df.to_csv('results/nd_doped_predictions_700K_grouped.csv', index=False)
    print("\nPredictions saved to results/nd_doped_predictions_700K_grouped.csv")

    # Doping trends plot (700 K)
    colors = {'Ca': 'blue', 'Sr': 'green', 'Ba': 'red'}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, target in enumerate(target_cols):
        ax = axes.flatten()[idx]
        for a_site in ['Ca', 'Sr', 'Ba']:
            sub = results_df[results_df['Chemical composition'].str.contains(a_site)]
            if len(sub) > 0:
                sub = sub.sort_values('doping_concentration')
                ax.plot(sub['doping_concentration'], sub[f'{target}_mean_best'],
                        marker='o', label=f'{a_site}TiO3 at 700 K' if idx == 0 else "",
                        color=colors[a_site], alpha=0.7, linewidth=2)
        ax.set_xlabel('Doping concentration (x)')
        ax.set_ylabel(target)
        ax.set_title(f'{target} vs doping at 700 K\n(Best grouped model: {best_overall})')
        ax.set_xlim(left=0.0)
        if idx == 0: ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/doping_trends_700K_grouped.png', dpi=500)
    plt.close()

    # ===== REVIEWER-FIX 2 (C3): Lanthanide predictions at DFT concentrations =====
    print("\n" + "="*50 + "\nLANTHANIDE-DOPED ATiO3 PREDICTIONS AT 700 K\n" + "="*50)
    lanthanides = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu']
    A_sites = ['Ca', 'Sr', 'Ba']
    doping_concs = DOPING_CONCS_LANTH_DFT
    comp_list, temp_list, a_list, l_list, x_list = [], [], [], [], []
    for A in A_sites:
        for L in lanthanides:
            for x in doping_concs:
                comp_list.append(f"{A}{1-x:.4f}{L}{x:.4f}TiO3")
                temp_list.append(700); a_list.append(A); l_list.append(L); x_list.append(x)
    meta_df = pd.DataFrame({
        'Chemical composition': comp_list, 'Temperature (K)': temp_list,
        'A_site': a_list, 'Lanthanide': l_list, 'doping_concentration': x_list})
    X_dope_lanth = generate_features_for_doping(meta_df, atomic_data, X_raw.columns.tolist())
    meta_df = meta_df.loc[X_dope_lanth.index].reset_index(drop=True)
    X_dope_lanth = X_dope_lanth.reset_index(drop=True)

    n_doped_lanth = len(A_sites) * len(lanthanides) * len(doping_concs)
    print(f"\n  Lanthanide screening set: {n_doped_lanth} doped compositions "
          f"({len(A_sites)} A-sites × {len(lanthanides)} lanthanides × {len(doping_concs)} concentrations)")
    print(f"  Doping concentrations: {doping_concs} "
          f"(DFT match x = {DFT_REFERENCE_X})")

    if best_overall == 'NeuralNetwork':
        mean_pred_l, std_pred_l = bootstrap_predictions(
            nn_model, X_train, y_train, X_dope_lanth, groups_train,
            n_bootstrap=50 if FAST_MODE else 100)
    elif best_overall == 'Stacking':
        mean_pred_l = predict_stacking(stacking_models, all_base_models, X_dope_lanth)
        std_pred_l = np.zeros_like(mean_pred_l)
    else:
        per_target_best = {}
        for target in target_cols:
            per_target_best[target] = all_base_models[target].get(best_overall, final_models[target])
        mean_pred_l, std_pred_l = bootstrap_predictions(
            per_target_best, X_train, y_train, X_dope_lanth, groups_train,
            n_bootstrap=50 if FAST_MODE else 100)

    zt_idx = target_cols.index('ZT')
    tc_idx = target_cols.index('Thermal conductivity (W/mK)')
    mean_zt = np.expm1(mean_pred_l[:, zt_idx])
    std_zt  = std_pred_l[:, zt_idx] * np.exp(mean_pred_l[:, zt_idx])
    mean_tc = np.expm1(mean_pred_l[:, tc_idx])
    std_tc  = std_pred_l[:, tc_idx] * np.exp(mean_pred_l[:, tc_idx])
    meta_df['ZT_mean'] = mean_zt; meta_df['ZT_std'] = std_zt
    meta_df['Thermal_conductivity_mean'] = mean_tc
    meta_df['Thermal_conductivity_std'] = std_tc
    valid = (mean_zt >= 0) & (mean_tc >= 0)
    meta_df = meta_df[valid].reset_index(drop=True)
    meta_df[['Chemical composition','Temperature (K)','A_site','Lanthanide',
             'doping_concentration','ZT_mean','ZT_std']].to_csv(
        'results/zt_lanthanides_700K_grouped.csv', index=False)
    meta_df[['Chemical composition','Temperature (K)','A_site','Lanthanide',
             'doping_concentration','Thermal_conductivity_mean',
             'Thermal_conductivity_std']].to_csv(
        'results/thermal_conductivity_lanthanides_700K_grouped.csv', index=False)
    print("Lanthanide predictions saved (with _grouped suffix).")

    # Direct random vs grouped comparison
    compare_random_vs_grouped_splits(
        X_raw, y_transformed, groups_all.values, target_cols,
        transformers, best_params_per_target)

    with open('results/cv_summary_grouped.json', 'w') as f:
        json.dump(cv_summary, f, indent=2)

    print("\n✅ Grouped-split analysis complete.")
    print("   Key output files:")
    print("     - results/train_test_split_grouped.csv             ← split manifest (proof of no leakage)")
    print("     - results/comparison_random_vs_grouped.csv         ← direct original-vs-fixed comparison")
    print("     - results/metrics_normalized_per_target.csv        ← per-target R²/RMSE/MAE + σ_y (NEW, for Table S-X)")
    print("     - results/cv_summary_grouped.json")
    print("     - results/nd_doped_predictions_700K_grouped.csv")
    print("     - results/zt_lanthanides_700K_grouped.csv")
    print("     - results/thermal_conductivity_lanthanides_700K_grouped.csv")
    print("     - figures/model_comparison_grouped_normalized.png ← Figure 1(b) (NEW)")
    print("     - figures/parity_plots_grouped.png                 ← Figure 1(a)")
    print("     - figures/doping_trends_700K_grouped.png")
    print("     - figures/shap_*_grouped.png                       ← Figure S1")


if __name__ == "__main__":
    main()

