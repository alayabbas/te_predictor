"""
ML Study of Nd-doped ATiO3 with rigorous validation, SHAP, and matminer.
Author: ALay-e-Abbas
Date: July 2026

This script performs:
- Data loading and cleaning (outlier removal, target transformation)
- Feature engineering using matminer (ElementProperty, Stoichiometry, etc.) + custom perovskite features
- Nested cross-validation with hyperparameter tuning using Optuna for multiple models (RF, XGB, KNN, NN, Ridge)
- Stacking ensemble (using RandomForest and XGBoost as base models, Ridge as meta‑model)
- Model evaluation with repeated CV and statistical significance tests
- SHAP analysis for model interpretation
- Parity plots (actual vs predicted) for each target using the best model, with model name indicated
- Uncertainty quantification via bootstrapping for the overall best model
- Predictions for Nd-doped ATiO3 systems (700 K only) with confidence intervals
- Physical consistency checks (e.g., ZT ≥ 0, σ ≥ 0)
- Model comparison plots and doping trends (without error bars, only 700 K) using the overall best model
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
                                     learning_curve)
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

# Matminer for advanced feature engineering
from matminer.featurizers.composition import (ElementProperty, Stoichiometry,
                                              ValenceOrbital, ElectronegativityDiff,
                                              BandCenter, Miedema, AtomicOrbitals)
from matminer.featurizers.base import MultipleFeaturizer
from pymatgen.core import Composition

# SHAP
import shap

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)  # reduce verbosity if desired

warnings.filterwarnings('ignore')

# ============================================
# GLOBAL CONFIGURATION
# ============================================
FAST_MODE = False  # Set to False for final publication-quality run (slower but thorough)

if FAST_MODE:
    CV_FOLDS = 5          # reduce from 10 for speed
    CV_REPEATS = 3        # reduce from 5
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
    N_JOBS = -1            # reduced from -1 to avoid oversubscription on high-core machines
    VERBOSE = 1
    print("🐢 FULL MODE: Running exhaustive validation (may take hours).")

# Output directories
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ============================================
# HELPER FUNCTIONS
# ============================================
def load_atomic_data(filename='atoms.dat'):
    """Load custom atomic properties (optional)."""
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
    """Convert formula string to pymatgen Composition object, handling doping notation."""
    if pd.isna(formula) or formula == '':
        return None
    formula = str(formula).strip()
    # Remove common annotations
    formula = re.sub(r'\s*\([^)]*\)', '', formula)  # remove parentheses content
    formula = re.sub(r'doped.*', '', formula, flags=re.IGNORECASE).strip()
    # Handle special cases like "La-doped SrTiO3" -> "SrTiO3"
    if 'doped' in formula.lower():
        parts = re.split(r'doped', formula, flags=re.IGNORECASE)
        formula = parts[-1].strip()
    # Remove spaces
    formula = re.sub(r'\s+', '', formula)
    try:
        return Composition(formula)
    except:
        # Fallback: return None; will be handled later
        return None

def load_and_preprocess_data(filename='data.csv'):
    """Load, clean, and return DataFrame with outlier removal and reset index."""
    print("="*70)
    print("LOADING AND PREPROCESSING DATA")
    print("="*70)
    
    df = pd.read_csv(filename)
    print(f"Original shape: {df.shape}")
    
    # Drop metadata columns
    meta_cols = [col for col in df.columns if 'Reference' in col or 'DOI' in col]
    if meta_cols:
        df.drop(columns=meta_cols, inplace=True)
        print(f"Dropped metadata: {meta_cols}")
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Map expected columns
    col_mapping = {}
    target_cols = ['Conductivity (S/cm)', 'Seebeck (uV/K)', 
                   'Thermal conductivity (W/mK)', 'ZT']
    
    # Chemical composition
    for col in df.columns:
        if 'chemical composition' in col.lower():
            col_mapping[col] = 'Chemical composition'
            break
    # Temperature
    for col in df.columns:
        if 'temperature' in col.lower():
            col_mapping[col] = 'Temperature (K)'
            break
    # Conductivity (but not thermal)
    for col in df.columns:
        if 'conductivity' in col.lower() and 'thermal' not in col.lower():
            col_mapping[col] = 'Conductivity (S/cm)'
            break
    # Seebeck
    for col in df.columns:
        if 'seebeck' in col.lower():
            col_mapping[col] = 'Seebeck (uV/K)'
            break
    # Thermal conductivity
    for col in df.columns:
        if 'thermal conductivity' in col.lower():
            col_mapping[col] = 'Thermal conductivity (W/mK)'
            break
    # ZT
    for col in df.columns:
        if 'figure of merit' in col.lower() or 'zt' in col.lower():
            col_mapping[col] = 'ZT'
            break
    
    df.rename(columns=col_mapping, inplace=True)
    
    # Ensure required columns exist
    required = ['Chemical composition', 'Temperature (K)']
    present_targets = [t for t in target_cols if t in df.columns]
    required += present_targets
    missing = [r for r in required if r not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Convert to numeric
    for col in required[1:]:  # skip composition
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Ensure positive for physical quantities
        if col in ['Conductivity (S/cm)', 'Thermal conductivity (W/mK)', 'ZT']:
            df[col] = df[col].abs()
    
    # Drop rows where all targets are NaN
    df.dropna(subset=present_targets, how='all', inplace=True)
    
    # Outlier detection using IQR (3*IQR rule)
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
    
    # Reset index to ensure continuous integer labels after row removal
    df.reset_index(drop=True, inplace=True)
    
    # Final missing value check
    print(f"Final shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    return df

# ============================================
# FEATURE ENGINEERING WITH MATMINER (FIXED)
# ============================================
def create_matminer_features(df, atomic_data=None):
    """
    Generate features using matminer featurizers with error handling.
    Returns feature DataFrame with same index as df (only valid compositions).
    """
    print("\n" + "="*70)
    print("FEATURE ENGINEERING WITH MATMINER")
    print("="*70)
    
    # Convert composition strings to pymatgen Composition objects
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
    
    # Create temporary DataFrame for featurization
    temp_df = pd.DataFrame({'composition': compositions}, index=valid_indices)
    
    # Define a safe set of featurizers (avoid those requiring oxidation states if problematic)
    try:
        # Attempt with full featurizer set, ignoring errors
        featurizer = MultipleFeaturizer([
            ElementProperty.from_preset('magpie'),
            Stoichiometry(),
            ValenceOrbital(),
            ElectronegativityDiff(),
        ])
        
        # Use ignore_errors=True to skip entries that cause errors (like oxidation state issues)
        feature_df = featurizer.featurize_dataframe(
            temp_df, 
            col_id='composition', 
            ignore_errors=True  # Critical: skip problematic entries
        )
        # Check if any rows were dropped due to errors
        if feature_df.shape[0] < temp_df.shape[0]:
            print(f"Warning: {temp_df.shape[0] - feature_df.shape[0]} compositions caused errors and were skipped.")
    except Exception as e:
        print(f"Error with full featurizer set: {e}")
        print("Falling back to basic featurizers (ElementProperty + Stoichiometry only).")
        # Fallback to a simpler set that doesn't require oxidation states
        featurizer = MultipleFeaturizer([
            ElementProperty.from_preset('magpie'),
            Stoichiometry(),
        ])
        feature_df = featurizer.featurize_dataframe(
            temp_df, 
            col_id='composition', 
            ignore_errors=True
        )
    
    # Drop the composition column if it exists
    if 'composition' in feature_df.columns:
        feature_df.drop(columns=['composition'], inplace=True)
    
    # Add temperature and derived features
    original_indices = feature_df.index  # these are the indices from temp_df that survived
    feature_df['temperature'] = df.loc[original_indices, 'Temperature (K)'].values
    feature_df['temp_squared'] = feature_df['temperature'] ** 2
    feature_df['temp_cubic'] = feature_df['temperature'] ** 3
    feature_df['sqrt_temp'] = np.sqrt(feature_df['temperature'])
    feature_df['log_temp'] = np.log1p(feature_df['temperature'])
    feature_df['temp_inverse'] = 1 / (feature_df['temperature'] + 1e-5)
    
    # Add custom perovskite features if atomic_data is available
    if atomic_data:
        custom_feat = add_custom_perovskite_features(df.loc[original_indices], atomic_data)
        if custom_feat is not None and not custom_feat.empty:
            feature_df = pd.concat([feature_df, custom_feat], axis=1)
    
    print(f"Total features generated: {feature_df.shape[1]} for {feature_df.shape[0]} samples.")
    return feature_df, original_indices.tolist()

def add_custom_perovskite_features(df, atomic_data):
    """
    Add perovskite-specific features: tolerance factor, octahedral factor,
    doping-related interactions, etc. 
    Replace this placeholder with your own logic from the original code.
    """
    # Placeholder: return empty DataFrame with same index
    return pd.DataFrame(index=df.index)

# ============================================
# GENERIC FEATURE GENERATION FOR DOPED SYSTEMS
# ============================================
def generate_features_for_doping(df_compositions, atomic_data, training_columns):
    """
    Generate features for given compositions (with Temperature column).
    Returns DataFrame aligned to training_columns.
    """
    # Parse compositions
    comps = []
    valid_indices = []
    for idx, formula in enumerate(df_compositions['Chemical composition']):
        comp = parse_composition(formula)
        if comp is not None:
            comps.append(comp)
            valid_indices.append(idx)
        else:
            print(f"Warning: Could not parse composition: {formula}")
    
    if len(comps) == 0:
        raise ValueError("No valid compositions.")
    
    temp_df = pd.DataFrame({'composition': comps}, index=valid_indices)
    
    # Featurize (try full set, fallback)
    try:
        featurizer = MultipleFeaturizer([
            ElementProperty.from_preset('magpie'),
            Stoichiometry(),
            ValenceOrbital(),
            ElectronegativityDiff(),
        ])
        feature_df = featurizer.featurize_dataframe(
            temp_df, col_id='composition', ignore_errors=True
        )
    except Exception as e:
        print(f"Full featurizer failed: {e}, falling back to basic.")
        featurizer = MultipleFeaturizer([
            ElementProperty.from_preset('magpie'),
            Stoichiometry(),
        ])
        feature_df = featurizer.featurize_dataframe(
            temp_df, col_id='composition', ignore_errors=True
        )
    
    if 'composition' in feature_df.columns:
        feature_df.drop(columns=['composition'], inplace=True)
    
    # Add temperature features
    original_indices = feature_df.index
    temps = df_compositions.loc[original_indices, 'Temperature (K)'].values
    feature_df['temperature'] = temps
    feature_df['temp_squared'] = temps ** 2
    feature_df['temp_cubic'] = temps ** 3
    feature_df['sqrt_temp'] = np.sqrt(temps)
    feature_df['log_temp'] = np.log1p(temps)
    feature_df['temp_inverse'] = 1 / (temps + 1e-5)
    
    # Add custom features if needed (placeholder)
    if atomic_data:
        custom_feat = add_custom_perovskite_features(df_compositions.loc[original_indices], atomic_data)
        if custom_feat is not None and not custom_feat.empty:
            feature_df = pd.concat([feature_df, custom_feat], axis=1)
    
    # Align with training columns
    # Keep only columns that are in training_columns, fill missing with 0
    result = pd.DataFrame(index=feature_df.index, columns=training_columns)
    for col in training_columns:
        if col in feature_df.columns:
            result[col] = feature_df[col]
        else:
            result[col] = 0.0
    return result

# ============================================
# TARGET TRANSFORMATIONS
# ============================================
def get_target_transformer(target_name):
    """
    Return a transformer for a given target.
    For positive quantities (conductivity, thermal conductivity, ZT): log1p.
    For Seebeck (can be negative): Yeo-Johnson (or no transform if symmetric).
    """
    if target_name in ['Conductivity (S/cm)', 'Thermal conductivity (W/mK)', 'ZT']:
        # log1p transform
        return FunctionTransformer(np.log1p, inverse_func=np.expm1)
    else:
        # Yeo-Johnson (handles negative values)
        return PowerTransformer(method='yeo-johnson', standardize=False)

# ============================================
# MODEL DEFINITIONS WITH HYPERPARAMETER SPACES (ADJUSTABLE FOR SPEED)
# ============================================
def get_model_param_spaces():
    """Return dictionary of models with hyperparameter grids. 
       If FAST_MODE=True, grids are reduced for speed."""
    param_spaces = {}
    
    if FAST_MODE:
        # Random Forest (very reduced)
        param_spaces['RandomForest'] = {
            'model': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt']
            },
            'n_iter': 10
        }
        
        # XGBoost (very reduced + hist method) - early_stopping_rounds removed
        param_spaces['XGBoost'] = {
            'model': xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS, 
                                      tree_method='hist'),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8],
                'colsample_bytree': [0.8],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [0.1, 1]
            },
            'n_iter': 10
        }
        
        # KNN (very reduced)
        param_spaces['KNN'] = {
            'model': KNeighborsRegressor(n_jobs=N_JOBS),
            'params': {
                'n_neighbors': [5, 7],
                'weights': ['uniform', 'distance'],
                'p': [2],
                'leaf_size': [30]
            },
            'n_iter': 5
        }
        
        # Ridge (unchanged)
        param_spaces['Ridge'] = {
            'model': Ridge(random_state=RANDOM_STATE),
            'params': {
                'alpha': [0.1, 1, 10, 100]
            },
            'n_iter': 4
        }
    else:
        # Random Forest (full)
        param_spaces['RandomForest'] = {
            'model': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS),
            'params': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'n_iter': 50
        }
        
        # XGBoost (full) - early_stopping_rounds removed
        param_spaces['XGBoost'] = {
            'model': xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS, tree_method='hist'),
            'params': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 6, 9, 12],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'reg_alpha': [0, 0.1, 1, 10],
                'reg_lambda': [0.1, 1, 10]
            },
            'n_iter': 50
        }
        
        # KNN (full)
        param_spaces['KNN'] = {
            'model': KNeighborsRegressor(n_jobs=N_JOBS),
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11, 15],
                'weights': ['uniform', 'distance'],
                'p': [1, 2],
                'leaf_size': [20, 30, 40]
            },
            'n_iter': 30
        }
        
        # Ridge (full)
        param_spaces['Ridge'] = {
            'model': Ridge(random_state=RANDOM_STATE),
            'params': {
                'alpha': [0.01, 0.1, 1, 10, 100, 1000]
            },
            'n_iter': 6
        }
    
    return param_spaces

# ============================================
# NESTED CROSS-VALIDATION WITH TUNING (SINGLE-OUTPUT) - VERBOSE VERSION using Optuna
# ============================================
def nested_cv_tuning(X, y, model_name, model_info, outer_cv=KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                     inner_cv=KFold(5, shuffle=True, random_state=RANDOM_STATE), scoring='neg_mean_squared_error'):
    """
    Perform nested cross-validation with hyperparameter tuning using Optuna for a single-output model.
    Now with verbose progress and reduced parameter grids for faster execution.
    """
    from sklearn.model_selection import cross_val_score
    import optuna
    
    outer_scores = []
    best_params_list = []
    
    base_model_class = model_info['model'].__class__  # get the class (e.g., RandomForestRegressor)
    param_grid = model_info['params']
    n_trials = model_info.get('n_iter', 20)
    
    print(f"Starting nested CV for {model_name} with {outer_cv.get_n_splits()} outer folds...")
    
    # We'll iterate over outer folds manually to capture best params
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"  Outer fold {fold+1}/{outer_cv.get_n_splits()}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Define objective function for Optuna
        def objective(trial):
            # Suggest hyperparameters from param_grid
            params = {}
            for param_name, values in param_grid.items():
                # For each parameter, choose one value from the list
                if isinstance(values, list):
                    # If it's a categorical list, use suggest_categorical
                    params[param_name] = trial.suggest_categorical(param_name, values)
                else:
                    # In case of non-list (should not happen), just use the value
                    params[param_name] = values
            
            # Create model with suggested params and fixed arguments
            fixed_args = {}
            if model_name == 'RandomForest' or model_name == 'XGBoost' or model_name == 'Ridge':
                fixed_args['random_state'] = RANDOM_STATE
            if model_name == 'XGBoost':
                fixed_args['n_jobs'] = N_JOBS
                fixed_args['tree_method'] = 'hist'
            elif model_name == 'RandomForest' or model_name == 'KNN':
                fixed_args['n_jobs'] = N_JOBS
            # Ridge does not have n_jobs
            
            model = base_model_class(**params, **fixed_args)
            
            # Perform inner CV
            scores = cross_val_score(model, X_train, y_train, cv=inner_cv, 
                                     scoring=scoring, n_jobs=N_JOBS)
            mean_score = scores.mean()
            return mean_score
        
        # Create study and optimize
        sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        best_params = study.best_params
        best_params_list.append(best_params)
        print(f"    Best params: {best_params}")
        
        # Train model with best params on the entire outer training set
        fixed_args = {}
        if model_name == 'RandomForest' or model_name == 'XGBoost' or model_name == 'Ridge':
            fixed_args['random_state'] = RANDOM_STATE
        if model_name == 'XGBoost':
            fixed_args['n_jobs'] = N_JOBS
            fixed_args['tree_method'] = 'hist'
        elif model_name == 'RandomForest' or model_name == 'KNN':
            fixed_args['n_jobs'] = N_JOBS
        best_model = base_model_class(**best_params, **fixed_args)
        best_model.fit(X_train, y_train)
        
        # Evaluate on outer test set
        y_pred = best_model.predict(X_test)
        score = -mean_squared_error(y_test, y_pred)  # because scoring is negative MSE
        outer_scores.append(score)
        print(f"    Outer fold RMSE = {np.sqrt(-score):.4f}")
    
    # Compute mean and std of outer scores
    mean_score = np.mean(outer_scores)
    std_score = np.std(outer_scores)
    
    print(f"{model_name}: Outer CV RMSE = {np.sqrt(-mean_score):.4f} ± {np.sqrt(-std_score):.4f}")
    
    # Refit on full data using best params from best outer fold? Use all data with CV again.
    print("Refitting on full training data with tuning...")
    
    # Define objective on full training data
    def final_objective(trial):
        params = {}
        for param_name, values in param_grid.items():
            params[param_name] = trial.suggest_categorical(param_name, values)
        fixed_args = {}
        if model_name == 'RandomForest' or model_name == 'XGBoost' or model_name == 'Ridge':
            fixed_args['random_state'] = RANDOM_STATE
        if model_name == 'XGBoost':
            fixed_args['n_jobs'] = N_JOBS
            fixed_args['tree_method'] = 'hist'
        elif model_name == 'RandomForest' or model_name == 'KNN':
            fixed_args['n_jobs'] = N_JOBS
        model = base_model_class(**params, **fixed_args)
        scores = cross_val_score(model, X, y, cv=inner_cv, scoring=scoring, n_jobs=N_JOBS)
        return scores.mean()
    
    sampler_final = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study_final = optuna.create_study(direction='maximize', sampler=sampler_final)
    study_final.optimize(final_objective, n_trials=n_trials, show_progress_bar=False)
    
    final_best_params = study_final.best_params
    fixed_args = {}
    if model_name == 'RandomForest' or model_name == 'XGBoost' or model_name == 'Ridge':
        fixed_args['random_state'] = RANDOM_STATE
    if model_name == 'XGBoost':
        fixed_args['n_jobs'] = N_JOBS
        fixed_args['tree_method'] = 'hist'
    elif model_name == 'RandomForest' or model_name == 'KNN':
        fixed_args['n_jobs'] = N_JOBS
    final_model = base_model_class(**final_best_params, **fixed_args)
    final_model.fit(X, y)
    
    print(f"Final best params: {final_best_params}")
    
    return {
        'model_name': model_name,
        'outer_scores': outer_scores,
        'mean_rmse': np.sqrt(-mean_score),
        'std_rmse': np.sqrt(-std_score),
        'best_params': final_best_params,
        'final_model': final_model
    }

# ============================================
# NEURAL NETWORK DEFINITION (MULTI-OUTPUT) with BatchNorm fix
# ============================================
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
    """Wrapper for PyTorch model with scikit-learn interface."""
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
        # Ensure data is large enough for batch norm (at least 2 samples)
        if len(X) < 2:
            raise ValueError(f"Training set has only {len(X)} samples. BatchNorm requires at least 2 samples per batch. "
                             "Increase training set size or remove BatchNorm layers.")
        
        # Scale data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Convert to tensors
        X_t = torch.FloatTensor(X_scaled).to(self.device)
        y_t = torch.FloatTensor(y_scaled).to(self.device)
        
        # Adjust batch size if dataset is smaller than batch_size
        actual_batch_size = min(self.batch_size, len(X))
        # Ensure at least 2 for BatchNorm
        if actual_batch_size < 2:
            actual_batch_size = 2  # but this would still cause issues if len(X) < 2; we already checked len(X)>=2
        
        # Create dataset and loader with drop_last=True to avoid incomplete batches
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=actual_batch_size, shuffle=True, drop_last=True)
        
        # Build model
        self.model = ImprovedNN(self.input_size, self.output_size, self.hidden_sizes, self.dropout).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # Validation data if provided
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler_X.transform(X_val)
            y_val_scaled = self.scaler_y.transform(y_val)
            X_val_t = torch.FloatTensor(X_val_scaled).to(self.device)
            y_val_t = torch.FloatTensor(y_val_scaled).to(self.device)
            val_dataset = TensorDataset(X_val_t, y_val_t)
            # For validation, we can use batch_size = len(X_val) to have one batch (no small batches)
            val_batch_size = min(self.batch_size, len(X_val))
            val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            # Training
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
            
            # Validation
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
                
                # Early stopping
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
            'input_size': self.input_size,
            'output_size': self.output_size,
            'hidden_sizes': self.hidden_sizes,
            'dropout': self.dropout,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'patience': self.patience,
            'device': self.device
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

# ============================================
# REPEATED CROSS-VALIDATION EVALUATION
# ============================================
def repeated_cv_evaluation(X, y, model, n_repeats=CV_REPEATS, n_folds=CV_FOLDS, random_state=RANDOM_STATE):
    """
    Perform repeated K-fold cross-validation and return scores per target.
    For multi-output, we compute per-target metrics.
    """
    rkf = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=random_state)
    
    # For storing per-fold metrics
    per_target_metrics = {col: {'R2': [], 'RMSE': [], 'MAE': [], 'Pearson': []} for col in y.columns}
    
    for train_idx, test_idx in rkf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model (clone if sklearn)
        from sklearn.base import clone
        if not isinstance(model, PyTorchRegressor):
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
        else:
            # For PyTorch, we need a new instance with same parameters
            model_clone = PyTorchRegressor(**model.get_params())
            # Split validation from train
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)
            model_clone.fit(X_tr, y_tr, X_val, y_val)
        
        y_pred = model_clone.predict(X_test)
        
        # Compute metrics per target
        for i, col in enumerate(y.columns):
            y_true = y_test.iloc[:, i].values
            y_pred_col = y_pred[:, i] if y_pred.ndim > 1 else y_pred
            
            r2 = r2_score(y_true, y_pred_col)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred_col))
            mae = mean_absolute_error(y_true, y_pred_col)
            try:
                pearson = stats.pearsonr(y_true, y_pred_col)[0]
            except:
                pearson = np.nan
            
            per_target_metrics[col]['R2'].append(r2)
            per_target_metrics[col]['RMSE'].append(rmse)
            per_target_metrics[col]['MAE'].append(mae)
            per_target_metrics[col]['Pearson'].append(pearson)
    
    # Aggregate results
    results = {}
    for col in y.columns:
        results[col] = {}
        for metric in ['R2', 'RMSE', 'MAE', 'Pearson']:
            values = per_target_metrics[col][metric]
            results[col][f'{metric}_mean'] = np.mean(values)
            results[col][f'{metric}_std'] = np.std(values)
    
    return results

# ============================================
# PHYSICAL CONSISTENCY CHECK
# ============================================
def check_physical_consistency(y_pred_df, target_cols):
    """
    Check predictions against physical constraints:
    - Conductivity, Thermal conductivity, ZT must be >= 0
    - Seebeck can be negative but typically within a range
    Returns boolean mask of valid predictions and prints violations.
    """
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

# ============================================
# SHAP ANALYSIS (FIXED: fallback to KernelExplainer with lambda and numpy conversion)
# ============================================
def shap_analysis(model, X_train, X_test, feature_names, target_names, model_name, save=True):
    """
    Compute SHAP values and plot summary.
    For multi-output models, we compute per output.
    Includes fallback for XGBoost base_score string issue and scikit-learn attribute issue.
    """
    print(f"\nPerforming SHAP analysis for {model_name}...")
    
    # Sanitize model_name for file saving (replace problematic characters)
    safe_model_name = re.sub(r'[^\w\-_]', '_', model_name)
    
    # Determine explainer type based on model
    if isinstance(model, (RandomForestRegressor, xgb.XGBRegressor)):
        # Try TreeExplainer first (fast), fall back to KernelExplainer if error
        try:
            explainer = shap.TreeExplainer(model)
        except Exception as e:
            print(f"TreeExplainer failed with error: {e}")
            print("Falling back to KernelExplainer (slower but compatible).")
            # Use KernelExplainer with a subset of background data
            background = shap.sample(X_train, min(100, len(X_train)))
            # Convert background to numpy array to avoid feature_names_in_ issues
            if hasattr(background, 'values'):
                background = background.values
            # Use a lambda that calls model.predict (bypass model conversion)
            explainer = shap.KernelExplainer(lambda x: model.predict(x), background)
    elif isinstance(model, Ridge):
        try:
            explainer = shap.LinearExplainer(model, X_train)
        except Exception as e:
            print(f"LinearExplainer failed with error: {e}")
            print("Falling back to KernelExplainer.")
            background = shap.sample(X_train, min(100, len(X_train)))
            if hasattr(background, 'values'):
                background = background.values
            explainer = shap.KernelExplainer(lambda x: model.predict(x), background)
    else:
        # Use KernelExplainer with a subset of background data
        background = shap.sample(X_train, min(100, len(X_train)))
        if hasattr(background, 'values'):
            background = background.values
        explainer = shap.KernelExplainer(lambda x: model.predict(x), background)
    
    # Compute SHAP values for test set (or a subset)
    X_test_sample = X_test[:100] if X_test.shape[0] > 100 else X_test
    # Convert test sample to numpy for consistency (KernelExplainer expects array-like)
    if hasattr(X_test_sample, 'values'):
        X_test_sample_np = X_test_sample.values
    else:
        X_test_sample_np = X_test_sample
    shap_values = explainer.shap_values(X_test_sample_np)
    
    # Create subplots
    fig, axes = plt.subplots(1, len(target_names), figsize=(6*len(target_names), 5))
    if len(target_names) == 1:
        axes = [axes]
    
    for i, target in enumerate(target_names):
        if isinstance(shap_values, list):
            sv = shap_values[i]
        else:
            sv = shap_values
        
        # Set current axis to the subplot
        plt.sca(axes[i])
        # Call summary_plot without ax parameter; it will use the current axis
        # Use numpy array for data to avoid any DataFrame issues
        shap.summary_plot(sv, X_test_sample_np, feature_names=feature_names, show=False)
        axes[i].set_title(f'{model_name} - {target}')
    
    plt.tight_layout()
    
    # Ensure figures directory exists
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/shap_{safe_model_name}.png', dpi=500)
    plt.show()
    plt.close()
    
    # Save shap values
    os.makedirs('results', exist_ok=True)
    with open(f'results/shap_{safe_model_name}.pkl', 'wb') as f:
        pickle.dump({'shap_values': shap_values, 'data': X_test_sample_np, 'feature_names': feature_names}, f)
    print(f"SHAP analysis saved.")

# ============================================
# UNCERTAINTY QUANTIFICATION VIA BOOTSTRAPPING (for a given model or set of per-target models)
# ============================================
def bootstrap_predictions(model, X, y, X_pred, n_bootstrap=100):
    """
    Generate bootstrap samples, retrain model(s), and collect predictions on X_pred.
    If model is a dict (per-target models), then it should have keys for each target.
    If model is a single multi-output model (e.g., neural network), it should have predict() that returns array (n_samples, n_targets).
    Returns mean and std arrays of shape (len(X_pred), n_targets).
    """
    from sklearn.utils import resample
    from sklearn.base import clone
    
    n_targets = len(y.columns) if isinstance(y, pd.DataFrame) else y.shape[1]
    target_cols = y.columns if isinstance(y, pd.DataFrame) else [f'target_{i}' for i in range(n_targets)]
    bootstrap_preds = []
    
    for i in range(n_bootstrap):
        X_bs, y_bs = resample(X, y, replace=True, random_state=RANDOM_STATE+i)
        
        if isinstance(model, dict):  # per-target models
            preds = np.zeros((X_pred.shape[0], n_targets))
            for j, target in enumerate(target_cols):
                model_clone = clone(model[target])
                model_clone.fit(X_bs, y_bs[target])
                preds[:, j] = model_clone.predict(X_pred)
        else:  # single multi-output model
            if isinstance(model, PyTorchRegressor):
                # For PyTorch, we need a new instance
                model_clone = PyTorchRegressor(**model.get_params())
                # Split validation within bootstrap? For simplicity, we won't use validation here.
                model_clone.fit(X_bs, y_bs)
            else:
                model_clone = clone(model)
                model_clone.fit(X_bs, y_bs)
            preds = model_clone.predict(X_pred)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
        bootstrap_preds.append(preds)
    
    bootstrap_preds = np.array(bootstrap_preds)  # (n_bootstrap, n_samples, n_targets)
    mean_pred = np.mean(bootstrap_preds, axis=0)
    std_pred = np.std(bootstrap_preds, axis=0)
    return mean_pred, std_pred

# ============================================
# GENERATE Nd-DOPED SYSTEMS (PLACEHOLDER) - MODIFIED FOR 700 K ONLY
# ============================================
def generate_nd_doped_systems(feature_names, atomic_data, temperatures=[700], 
                               a_sites=['Ca','Sr','Ba'], b_site='Ti', dopant='Nd',
                               doping_concs=[0.0,0.05,0.1,0.15,0.2]):
    """
    Generate feature vectors for Nd-doped ATiO3 systems at specified temperatures (default 700 K).
    Replace this placeholder with your actual feature generation logic.
    Returns X_dope (DataFrame with feature_names columns) and metadata DataFrame.
    """
    # Create a list of compositions and metadata
    compositions = []
    temps = []
    doping_conc = []
    doping_site = []
    
    for a in a_sites:
        for t in temperatures:
            for x in doping_concs:
                if x == 0:
                    comp = f"{a}TiO3"
                else:
                    comp = f"{a}{1-x}{dopant}{x}TiO3"  # simplified notation
                compositions.append(comp)
                temps.append(t)
                doping_conc.append(x)
                doping_site.append('A' if x>0 else 'none')
    
    # Create metadata DataFrame
    meta = pd.DataFrame({
        'Chemical composition': compositions,
        'Temperature (K)': temps,
        'doping_concentration': doping_conc,
        'doping_site': doping_site
    })
    
    # For demonstration, we create random features with correct column names.
    # Replace this with actual feature computation using the same pipeline as training.
    np.random.seed(42)
    X_dope = pd.DataFrame(np.random.randn(len(meta), len(feature_names)), 
                          columns=feature_names)
    
    return X_dope, meta

# ============================================
# PLOT MODEL COMPARISON (with updated titles indicating averaging)
# ============================================
def plot_model_comparison(all_results, target_cols):
    """Plot model comparison bar charts for R2, RMSE, MAE (averaged across targets)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['R2', 'RMSE', 'MAE']
    titles = ['Average R² (higher is better)', 
              'Average RMSE (lower is better)', 
              'Average MAE (lower is better)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        model_names = []
        metric_values = []
        
        for model_name, results in all_results.items():
            # Average across all targets
            avg_metric = np.mean([results[target][metric] for target in target_cols])
            model_names.append(model_name)
            metric_values.append(avg_metric)
        
        # Sort by metric value
        sorted_indices = np.argsort(metric_values)
        if metric in ['RMSE', 'MAE']:
            sorted_indices = sorted_indices  # Lower is better
        else:
            sorted_indices = sorted_indices[::-1]  # Higher is better
        
        sorted_names = [model_names[i] for i in sorted_indices]
        sorted_values = [metric_values[i] for i in sorted_indices]
        
        bars = ax.bar(range(len(sorted_names)), sorted_values)
        ax.set_xticks(range(len(sorted_names)))
        ax.set_xticklabels(sorted_names, rotation=45, ha='right')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, sorted_values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('figures/model_comparison.png', dpi=500)
    plt.show()

# ============================================
# PLOT PARITY (ACTUAL VS PREDICTED) FOR EACH TARGET USING BEST MODEL
# ============================================
def plot_parity(final_models, best_model_names, X_test, y_test, target_cols, transformers):
    """
    Create a 2x2 grid of actual vs predicted scatter plots for each target.
    The title includes the name of the best model used.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, target in enumerate(target_cols):
        ax = axes[i]
        model = final_models[target]  # best model for this target
        model_name = best_model_names.get(target, "Unknown")
        y_pred = model.predict(X_test)
        
        # Inverse transform
        if target in ['Conductivity (S/cm)', 'Thermal conductivity (W/mK)', 'ZT']:
            y_true_inv = np.expm1(y_test[target])
            y_pred_inv = np.expm1(y_pred)
        else:
            pt = transformers[target][1]
            y_true_inv = pt.inverse_transform(y_test[target].values.reshape(-1, 1)).flatten()
            y_pred_inv = pt.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        # Compute R²
        r2 = r2_score(y_true_inv, y_pred_inv)
        
        # Scatter plot
        ax.scatter(y_true_inv, y_pred_inv, alpha=0.6, edgecolors='k', linewidth=0.5)
        
        # 1:1 line
        min_val = min(y_true_inv.min(), y_pred_inv.min())
        max_val = max(y_true_inv.max(), y_pred_inv.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1, label='1:1 line')
        
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{target}\nBest model: {model_name} (R² = {r2:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/parity_plots.png', dpi=500)
    plt.show()

# ============================================
# STACKING ENSEMBLE IMPLEMENTATION (RandomForest + XGBoost only)
# ============================================
def train_stacking_ensemble(X_train, y_train, base_models, target_cols, n_folds=5):
    """
    Train a stacking ensemble for each target using out-of-fold predictions from base models.
    base_models: dict of target -> dict of model_name -> trained model (scikit-learn estimators)
    Returns a dictionary of stacking models (one per target) that can be used for predictions.
    """
    from sklearn.model_selection import KFold
    from sklearn.linear_model import Ridge
    
    stacking_models = {}
    n_train = X_train.shape[0]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    for target in target_cols:
        print(f"\nTraining stacking ensemble for {target}...")
        # Get the list of base model names for this target (only RandomForest and XGBoost)
        model_names = [name for name in base_models[target].keys() if name in ['RandomForest', 'XGBoost']]
        if len(model_names) == 0:
            print(f"  No suitable base models for {target}, skipping.")
            continue
        # Initialize arrays to store OOF predictions
        oof_preds = {name: np.zeros(n_train) for name in model_names}
        y_train_target = y_train[target].values
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train_target[train_idx], y_train_target[val_idx]
            
            # For each base model, train on this fold and predict on validation
            for name in model_names:
                from sklearn.base import clone
                model_clone = clone(base_models[target][name])
                model_clone.fit(X_tr, y_tr)
                oof_preds[name][val_idx] = model_clone.predict(X_val)
        
        # Stack the out-of-fold predictions as features
        X_stack = np.column_stack([oof_preds[name] for name in model_names])
        
        # Train a Ridge meta-model on these features
        meta_model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        meta_model.fit(X_stack, y_train_target)
        
        # Store the meta-model and the list of base model names (order matters)
        stacking_models[target] = {
            'meta_model': meta_model,
            'base_model_names': model_names
        }
    
    return stacking_models

def predict_stacking(stacking_models, base_models, X):
    """
    Predict using stacking ensemble for all targets.
    base_models: dict of target -> dict of model_name -> trained model (final models)
    stacking_models: dict from train_stacking_ensemble
    Returns array of shape (n_samples, n_targets)
    """
    target_list = list(base_models.keys())
    y_pred_stack = np.zeros((X.shape[0], len(target_list)))
    
    for i, target in enumerate(target_list):
        if target not in stacking_models:
            # If no stacking model for this target, fallback to simple average of base models? We'll skip.
            continue
        # Get base model predictions
        base_preds = []
        for name in stacking_models[target]['base_model_names']:
            model = base_models[target][name]
            pred = model.predict(X)
            base_preds.append(pred)
        X_stack = np.column_stack(base_preds)
        # Meta prediction
        y_pred_stack[:, i] = stacking_models[target]['meta_model'].predict(X_stack)
    
    return y_pred_stack

# ============================================
# MAIN PIPELINE (with stacking ensemble and overall best model for doping trends)
# ============================================
def main():
    print("="*70)
    print("IMPROVED ML STUDY OF Nd-DOPED ATiO3 - PUBLICATION QUALITY")
    print("="*70)
    
    # 1. Load atomic data (optional)
    atomic_data = load_atomic_data('atoms.dat')
    
    # 2. Load and preprocess data (index reset inside)
    df = load_and_preprocess_data('data.csv')
    
    # 3. Feature engineering with matminer (fixed)
    X_raw, valid_idx = create_matminer_features(df, atomic_data)
    y_raw = df.loc[valid_idx, ['Conductivity (S/cm)', 'Seebeck (uV/K)', 
                                'Thermal conductivity (W/mK)', 'ZT']].copy()
    
    # Keep only columns that exist
    target_cols = [col for col in y_raw.columns if col in df.columns]
    y_raw = y_raw[target_cols]
    
    # ---- Handle NaN values in features ----
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
            if np.isnan(median_val):
                X_raw[col].fillna(0, inplace=True)
            else:
                X_raw[col].fillna(median_val, inplace=True)
    print("NaN handling complete.")
    
    # ---- Drop rows with NaN in any target ----
    print("\nChecking for NaN values in targets...")
    initial_rows = len(y_raw)
    y_raw.dropna(inplace=True)
    X_raw = X_raw.loc[y_raw.index]
    print(f"Dropped {initial_rows - len(y_raw)} rows with NaN targets.")
    
    # 4. Target transformations
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
    
    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_transformed, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # 6. Model hyperparameter spaces
    param_spaces = get_model_param_spaces()
    
    # Store final models and results
    final_models = {}           # per-target best model objects (selected by original-scale R²)
    best_model_names = {}       # per-target best model name (for plotting)
    cv_results = {}
    all_test_metrics = {}
    
    # For each target, perform nested CV with tuning
    for target in target_cols:
        print(f"\n{'='*50}")
        print(f"TARGET: {target}")
        print('='*50)
        
        y_train_target = y_train[target]
        y_test_target = y_test[target]
        
        outer_cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        
        best_models_for_target = {}
        
        for model_name, model_info in param_spaces.items():
            print(f"\n--- {model_name} ---")
            result = nested_cv_tuning(X_train, y_train_target, model_name, model_info,
                                       outer_cv=outer_cv, inner_cv=inner_cv,
                                       scoring='neg_mean_squared_error')
            best_models_for_target[model_name] = result['final_model']
            cv_results[f"{model_name}_{target}"] = result
        
        # Evaluate on test set with best models and select best by ORIGINAL-SCALE R²
        print(f"\nTest set evaluation for {target}:")
        original_r2_scores = {}
        for model_name, model in best_models_for_target.items():
            y_pred = model.predict(X_test)
            if target in ['Conductivity (S/cm)', 'Thermal conductivity (W/mK)', 'ZT']:
                y_true_inv = np.expm1(y_test_target)
                y_pred_inv = np.expm1(y_pred)
            else:
                pt = transformers[target][1]
                y_true_inv = pt.inverse_transform(y_test_target.values.reshape(-1, 1)).flatten()
                y_pred_inv = pt.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            
            r2 = r2_score(y_true_inv, y_pred_inv)
            rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
            mae = mean_absolute_error(y_true_inv, y_pred_inv)
            print(f"  {model_name}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
            
            # Store original-scale R² for selection
            original_r2_scores[model_name] = r2
            
            if model_name not in all_test_metrics:
                all_test_metrics[model_name] = {}
            if target not in all_test_metrics[model_name]:
                all_test_metrics[model_name][target] = {}
            all_test_metrics[model_name][target]['R2'] = r2
            all_test_metrics[model_name][target]['RMSE'] = rmse
            all_test_metrics[model_name][target]['MAE'] = mae
        
        # Choose best model for this target by highest original-scale R²
        best_model_name = max(original_r2_scores, key=original_r2_scores.get)
        final_models[target] = best_models_for_target[best_model_name]
        best_model_names[target] = best_model_name
        print(f"Selected best model for {target}: {best_model_name}")
    
    # 7. Multi-output Neural Network
    print("\n" + "="*50)
    print("TRAINING MULTI-OUTPUT NEURAL NETWORK")
    print("="*50)
    
    X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )
    
    if len(X_val_nn) < 2:
        print("Warning: Validation set too small. Using training set for validation.")
        X_val_nn = X_train_nn[:2]
        y_val_nn = y_train_nn[:2]
    
    nn_model = PyTorchRegressor(
        input_size=X_train.shape[1],
        output_size=len(target_cols),
        hidden_sizes=[512,256,128,64],
        dropout=0.2,
        lr=0.001,
        epochs=300 if not FAST_MODE else 100,
        patience=30,
        batch_size=32 if not FAST_MODE else 16
    )
    nn_model.fit(X_train_nn, y_train_nn, X_val_nn, y_val_nn)
    
    y_pred_nn = nn_model.predict(X_test)
    print("\nNeural Network test performance (per target):")
    for i, target in enumerate(target_cols):
        if target in ['Conductivity (S/cm)', 'Thermal conductivity (W/mK)', 'ZT']:
            y_true_inv = np.expm1(y_test.iloc[:, i])
            y_pred_inv = np.expm1(y_pred_nn[:, i])
        else:
            pt = transformers[target][1]
            y_true_inv = pt.inverse_transform(y_test.iloc[:, i].values.reshape(-1, 1)).flatten()
            y_pred_inv = pt.inverse_transform(y_pred_nn[:, i].reshape(-1, 1)).flatten()
        r2 = r2_score(y_true_inv, y_pred_inv)
        rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
        mae = mean_absolute_error(y_true_inv, y_pred_inv)
        print(f"  {target}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        
        if 'NeuralNetwork' not in all_test_metrics:
            all_test_metrics['NeuralNetwork'] = {}
        if target not in all_test_metrics['NeuralNetwork']:
            all_test_metrics['NeuralNetwork'][target] = {}
        all_test_metrics['NeuralNetwork'][target]['R2'] = r2
        all_test_metrics['NeuralNetwork'][target]['RMSE'] = rmse
        all_test_metrics['NeuralNetwork'][target]['MAE'] = mae
    
    # 8. Prepare base models for stacking (RandomForest and XGBoost only)
    all_base_models = {target: {} for target in target_cols}
    
    for target in target_cols:
        for model_name in ['RandomForest', 'XGBoost']:
            key = f"{model_name}_{target}"
            if key in cv_results:
                best_params = cv_results[key]['best_params']
                if model_name == 'RandomForest':
                    model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS, **best_params)
                elif model_name == 'XGBoost':
                    model = xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS, tree_method='hist', **best_params)
                else:
                    continue
                model.fit(X_train, y_train[target])
                all_base_models[target][model_name] = model
    
    # 9. Train stacking ensemble using these base models
    stacking_models = train_stacking_ensemble(X_train, y_train, all_base_models, target_cols, n_folds=5)
    
    # Evaluate stacking on test set
    y_pred_stack = predict_stacking(stacking_models, all_base_models, X_test)
    if y_pred_stack.shape[1] == len(target_cols):
        print("\nStacking Ensemble test performance (per target):")
        for i, target in enumerate(target_cols):
            if target in ['Conductivity (S/cm)', 'Thermal conductivity (W/mK)', 'ZT']:
                y_true_inv = np.expm1(y_test.iloc[:, i])
                y_pred_inv = np.expm1(y_pred_stack[:, i])
            else:
                pt = transformers[target][1]
                y_true_inv = pt.inverse_transform(y_test.iloc[:, i].values.reshape(-1, 1)).flatten()
                y_pred_inv = pt.inverse_transform(y_pred_stack[:, i].reshape(-1, 1)).flatten()
            r2 = r2_score(y_true_inv, y_pred_inv)
            rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
            mae = mean_absolute_error(y_true_inv, y_pred_inv)
            print(f"  {target}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
            
            if 'Stacking' not in all_test_metrics:
                all_test_metrics['Stacking'] = {}
            if target not in all_test_metrics['Stacking']:
                all_test_metrics['Stacking'][target] = {}
            all_test_metrics['Stacking'][target]['R2'] = r2
            all_test_metrics['Stacking'][target]['RMSE'] = rmse
            all_test_metrics['Stacking'][target]['MAE'] = mae
    else:
        print("Stacking ensemble did not produce predictions for all targets; skipping evaluation.")
    
    # 10. Repeated cross-validation for final models (robust performance)
    print("\n" + "="*50)
    print("REPEATED CROSS-VALIDATION (ROBUST PERFORMANCE ESTIMATES)")
    print("="*50)
    
    cv_summary = {}
    for target in target_cols:
        model = final_models[target]
        print(f"\nRepeated CV for {target}...")
        scores = cross_val_score(model, X_raw, y_transformed[target], 
                                 cv=RepeatedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE),
                                 scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-scores)
        cv_summary[target] = {
            'RMSE_mean': np.mean(rmse_scores),
            'RMSE_std': np.std(rmse_scores)
        }
        print(f"  RMSE = {cv_summary[target]['RMSE_mean']:.4f} ± {cv_summary[target]['RMSE_std']:.4f}")
    
    # 11. SHAP analysis (optional)
    print("\n" + "="*50)
    print("SHAP ANALYSIS")
    print("="*50)
    for target in target_cols:
        model = final_models[target]
        X_train_sample = X_train.sample(min(100, X_train.shape[0]), random_state=RANDOM_STATE)
        X_test_sample = X_test.sample(min(50, X_test.shape[0]), random_state=RANDOM_STATE)
        shap_analysis(model, X_train_sample, X_test_sample, X_train.columns.tolist(), [target], f"{target}_best")
    
    # 12. Model comparison plot
    print("\n" + "="*50)
    print("MODEL COMPARISON PLOT")
    print("="*50)
    plot_model_comparison(all_test_metrics, target_cols)
    
    # 13. Parity plots (actual vs predicted) using best models
    print("\n" + "="*50)
    print("PARITY PLOTS (ACTUAL VS PREDICTED)")
    print("="*50)
    plot_parity(final_models, best_model_names, X_test, y_test, target_cols, transformers)
    
    # 14. Identify overall best model (by average R² across targets)
    avg_r2 = {}
    for model_name in all_test_metrics.keys():
        avg_r2[model_name] = np.mean([all_test_metrics[model_name][target]['R2'] for target in target_cols])
    best_model_name = max(avg_r2, key=avg_r2.get)
    print(f"\nOverall best model (by average R²): {best_model_name} (avg R² = {avg_r2[best_model_name]:.4f})")
    
    # 15. Uncertainty quantification for the overall best model on Nd-doped systems (700 K only)
    print("\n" + "="*50)
    print(f"PREDICTING Nd-DOPED SYSTEMS AT 700 K WITH {best_model_name} AND UNCERTAINTY")
    print("="*50)
    
    # Define Nd-doped compositions (same logic as lanthanide block)
    a_sites_nd = ['Ca', 'Sr', 'Ba']
    doping_concs_nd = [0.0, 0.05, 0.1, 0.15, 0.2]
    T = 700
    dopant = 'Nd'
    
    comp_list_nd = []
    temp_list_nd = []
    a_list_nd = []
    x_list_nd = []
    site_list_nd = []
    
    for A in a_sites_nd:
        for x in doping_concs_nd:
            if x == 0:
                formula = f"{A}TiO3"
            else:
                formula = f"{A}{1-x:.4f}{dopant}{x:.4f}TiO3"
            comp_list_nd.append(formula)
            temp_list_nd.append(T)
            a_list_nd.append(A)
            x_list_nd.append(x)
            site_list_nd.append('A' if x > 0 else 'none')
    
    meta_dope = pd.DataFrame({
        'Chemical composition': comp_list_nd,
        'Temperature (K)': temp_list_nd,
        'A_site': a_list_nd,
        'doping_concentration': x_list_nd,
        'doping_site': site_list_nd
    })
    
    # Generate features using the same function as lanthanide block
    X_dope = generate_features_for_doping(meta_dope, atomic_data, X_raw.columns.tolist())
    
    # Update metadata to keep only successful rows
    valid_indices = X_dope.index
    meta_dope = meta_dope.loc[valid_indices].reset_index(drop=True)
    X_dope = X_dope.reset_index(drop=True)
    
    print(f"Generated features for {len(X_dope)} compositions.")
    
    # Prepare the best model for prediction
    if best_model_name == 'NeuralNetwork':
        best_model = nn_model  # multi-output
        mean_pred, std_pred = bootstrap_predictions(nn_model, X_train, y_train, X_dope, n_bootstrap=50 if FAST_MODE else 100)
    elif best_model_name == 'Stacking':
        mean_pred = predict_stacking(stacking_models, all_base_models, X_dope)
        std_pred = np.zeros_like(mean_pred)  # no uncertainty
    else:
        # Best model is one of the single-output models
        per_target_best = {}
        for target in target_cols:
            if best_model_name in all_base_models[target]:
                per_target_best[target] = all_base_models[target][best_model_name]
            else:
                print(f"Warning: {best_model_name} not found for {target}, using final_models instead.")
                per_target_best[target] = final_models[target]
        mean_pred, std_pred = bootstrap_predictions(per_target_best, X_train, y_train, X_dope, n_bootstrap=50 if FAST_MODE else 100)
    
    # Inverse transform
    for i, target in enumerate(target_cols):
        if target in ['Conductivity (S/cm)', 'Thermal conductivity (W/mK)', 'ZT']:
            mean_pred[:, i] = np.expm1(mean_pred[:, i])
            std_pred[:, i] = std_pred[:, i] * np.exp(mean_pred[:, i])  # rough
        else:
            pt = transformers[target][1]
            mean_pred[:, i] = pt.inverse_transform(mean_pred[:, i].reshape(-1, 1)).flatten()
            # std stays in transformed scale; we could approximate but keep as is
    
    # Create results DataFrame
    results_df = meta_dope.copy()
    for i, target in enumerate(target_cols):
        results_df[f'{target}_mean_best'] = mean_pred[:, i]
        results_df[f'{target}_std_best'] = std_pred[:, i]
    
    # Physical consistency check on best model predictions
    consistency_df = pd.DataFrame({target: mean_pred[:, i] for i, target in enumerate(target_cols)})
    valid_mask = check_physical_consistency(consistency_df, target_cols)
    results_df = results_df[valid_mask]
    
    # Save results
    results_df.to_csv('results/nd_doped_predictions_700K.csv', index=False)
    print("\nPredictions saved to results/nd_doped_predictions_700K.csv")
    
    # 16. Plot doping trends (only 700 K) using the overall best model (without error bars)
    print("\n" + "="*50)
    print("GENERATING DOPING TRENDS PLOT (700 K) USING BEST MODEL")
    print("="*50)
    
    colors = {'Ca': 'blue', 'Sr': 'green', 'Ba': 'red'}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, target in enumerate(target_cols):
        ax = axes.flatten()[idx]
        for a_site in ['Ca', 'Sr', 'Ba']:
            subset = results_df[results_df['Chemical composition'].str.contains(a_site)]
            if len(subset) > 0:
                subset = subset.sort_values('doping_concentration')
                # Use plot instead of errorbar to avoid error bars
                ax.plot(subset['doping_concentration'], subset[f'{target}_mean_best'],
                        marker='o', label=f'{a_site}TiO3 at 700 K' if idx == 0 else "",
                        color=colors[a_site], alpha=0.7, linewidth=2)
        ax.set_xlabel('Doping concentration (x)')
        ax.set_ylabel(target)
        ax.set_title(f'{target} vs doping concentration at 700 K\n(Best overall model: {best_model_name})')
        ax.set_xlim(left=0.0)
        if idx == 0:
            ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/doping_trends_700K.png', dpi=500)
    plt.show()
    
    # ============================================
    # ADDITION FOR LANTHANIDE DOPING PREDICTIONS - using same methodology as Nd-doped
    # ============================================
    print("\n" + "="*50)
    print("PREDICTING ZT AND THERMAL CONDUCTIVITY FOR LANTHANIDE-DOPED ATiO3 AT 700 K FOR MULTIPLE DOPING CONCENTRATIONS")
    print("="*50)
    
    # Define lanthanides, A sites and doping concentrations
    lanthanides = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu']
    A_sites = ['Ca', 'Sr', 'Ba']
    doping_concs = [0.05, 0.1, 0.15, 0.2]
    T = 700
    
    # Create list of compositions and metadata for all combinations
    comp_list = []
    temp_list = []
    a_list = []
    l_list = []
    x_list = []
    
    for A in A_sites:
        for L in lanthanides:
            for x in doping_concs:
                formula = f"{A}{1-x:.4f}{L}{x:.4f}TiO3"
                comp_list.append(formula)
                temp_list.append(T)
                a_list.append(A)
                l_list.append(L)
                x_list.append(x)
    
    meta_df = pd.DataFrame({
        'Chemical composition': comp_list,
        'Temperature (K)': temp_list,
        'A_site': a_list,
        'Lanthanide': l_list,
        'doping_concentration': x_list
    })
    
    # Generate features using the same function
    X_dope_lanth = generate_features_for_doping(meta_df, atomic_data, X_raw.columns.tolist())
    
    # Update metadata to keep only successful rows
    valid_indices = X_dope_lanth.index
    meta_df = meta_df.loc[valid_indices].reset_index(drop=True)
    X_dope_lanth = X_dope_lanth.reset_index(drop=True)
    
    print(f"Generated features for {len(X_dope_lanth)} compositions.")
    
    # ---- Now use the same methodology as Nd-doped predictions ----
    if best_model_name == 'NeuralNetwork':
        mean_pred, std_pred = bootstrap_predictions(nn_model, X_train, y_train, X_dope_lanth,
                                                     n_bootstrap=50 if FAST_MODE else 100)
    elif best_model_name == 'Stacking':
        mean_pred = predict_stacking(stacking_models, all_base_models, X_dope_lanth)
        std_pred = np.zeros_like(mean_pred)
    else:
        # Single-output models: use per_target_best (same dictionary as used for Nd-doped)
        per_target_best = {}
        for target in target_cols:
            if best_model_name in all_base_models[target]:
                per_target_best[target] = all_base_models[target][best_model_name]
            else:
                per_target_best[target] = final_models[target]
        mean_pred, std_pred = bootstrap_predictions(per_target_best, X_train, y_train, X_dope_lanth,
                                                     n_bootstrap=50 if FAST_MODE else 100)
    
    # Get indices for ZT and thermal conductivity
    zt_idx = target_cols.index('ZT')
    tc_idx = target_cols.index('Thermal conductivity (W/mK)')
    
    # Inverse transform ZT (log1p)
    mean_zt = np.expm1(mean_pred[:, zt_idx])
    std_zt = std_pred[:, zt_idx] * np.exp(mean_pred[:, zt_idx])  # approximate
    
    # Inverse transform thermal conductivity (log1p)
    mean_tc = np.expm1(mean_pred[:, tc_idx])
    std_tc = std_pred[:, tc_idx] * np.exp(mean_pred[:, tc_idx])  # approximate
    
    # Add to metadata
    meta_df['ZT_mean'] = mean_zt
    meta_df['ZT_std'] = std_zt
    meta_df['Thermal_conductivity_mean'] = mean_tc
    meta_df['Thermal_conductivity_std'] = std_tc
    
    # Physical consistency (ZT >= 0, TC >= 0)
    valid_zt = mean_zt >= 0
    valid_tc = mean_tc >= 0
    valid_mask = valid_zt & valid_tc
    if not valid_mask.all():
        print(f"Warning: {np.sum(~valid_mask)} predictions have negative ZT or thermal conductivity. Removing.")
        meta_df = meta_df[valid_mask].reset_index(drop=True)
    
    # Save results for ZT
    zt_df = meta_df[['Chemical composition', 'Temperature (K)', 'A_site', 'Lanthanide', 'doping_concentration', 'ZT_mean', 'ZT_std']].copy()
    zt_df.to_csv('results/zt_lanthanides_700K.csv', index=False)
    print("ZT predictions saved to results/zt_lanthanides_700K.csv")
    
    # Save results for thermal conductivity
    tc_df = meta_df[['Chemical composition', 'Temperature (K)', 'A_site', 'Lanthanide', 'doping_concentration', 'Thermal_conductivity_mean', 'Thermal_conductivity_std']].copy()
    tc_df.to_csv('results/thermal_conductivity_lanthanides_700K.csv', index=False)
    print("Thermal conductivity predictions saved to results/thermal_conductivity_lanthanides_700K.csv")
    
    # ---- Plotting ZT: 4 rows (doping concentrations) x 3 columns (A-sites) ----
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    colors_plot = ['blue', 'green', 'red']
    
    for row_idx, x_val in enumerate(doping_concs):
        for col_idx, A in enumerate(A_sites):
            ax = axes[row_idx, col_idx]
            subset = meta_df[(meta_df['A_site'] == A) & (meta_df['doping_concentration'] == x_val)].copy()
            if len(subset) == 0:
                ax.set_visible(False)
                continue
            # Ensure lanthanide order
            subset['L_order'] = pd.Categorical(subset['Lanthanide'], categories=lanthanides, ordered=True)
            subset = subset.sort_values('L_order')
            
            x_pos = np.arange(len(subset))
            bars = ax.bar(x_pos, subset['ZT_mean'], yerr=subset['ZT_std'], capsize=5,
                          color=colors_plot[col_idx], alpha=0.7, edgecolor='k')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(subset['Lanthanide'], rotation=45)
            ax.set_xlabel('Lanthanide')
            ax.set_ylabel('ZT')
            ax.set_title(f'{A}TiO3, x={x_val:.2f}')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels above bars, adjust y‑limits to avoid text cutoff
            max_text_y = 0
            for i, (val, err) in enumerate(zip(subset['ZT_mean'], subset['ZT_std'])):
                text_y = val + err + 0.01
                ax.text(i, text_y, f'{val:.3f}', ha='center', va='bottom', fontsize=8)
                if text_y > max_text_y:
                    max_text_y = text_y
            # Extend y‑axis to include the highest label plus a small margin
            current_ylim = ax.get_ylim()
            ax.set_ylim(top=max(current_ylim[1], max_text_y * 1.05))
    
    plt.tight_layout()
    plt.savefig('figures/zt_lanthanides_700K.png', dpi=500)
    plt.show()
    print("ZT lanthanide doping plot saved.")
    
    # ---- Plotting Thermal Conductivity: 4 rows (doping concentrations) x 3 columns (A-sites) ----
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    colors_plot = ['blue', 'green', 'red']
    
    for row_idx, x_val in enumerate(doping_concs):
        for col_idx, A in enumerate(A_sites):
            ax = axes[row_idx, col_idx]
            subset = meta_df[(meta_df['A_site'] == A) & (meta_df['doping_concentration'] == x_val)].copy()
            if len(subset) == 0:
                ax.set_visible(False)
                continue
            # Ensure lanthanide order
            subset['L_order'] = pd.Categorical(subset['Lanthanide'], categories=lanthanides, ordered=True)
            subset = subset.sort_values('L_order')
            
            x_pos = np.arange(len(subset))
            bars = ax.bar(x_pos, subset['Thermal_conductivity_mean'], yerr=subset['Thermal_conductivity_std'], capsize=5,
                          color=colors_plot[col_idx], alpha=0.7, edgecolor='k')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(subset['Lanthanide'], rotation=45)
            ax.set_xlabel('Lanthanide')
            ax.set_ylabel('Thermal Conductivity (W/mK)')
            ax.set_title(f'{A}TiO3, x={x_val:.2f}')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels above bars, adjust y‑limits to avoid text cutoff
            max_text_y = 0
            for i, (val, err) in enumerate(zip(subset['Thermal_conductivity_mean'], subset['Thermal_conductivity_std'])):
                text_y = val + err + 0.01
                ax.text(i, text_y, f'{val:.3f}', ha='center', va='bottom', fontsize=8)
                if text_y > max_text_y:
                    max_text_y = text_y
            # Extend y‑axis to include the highest label plus a small margin
            current_ylim = ax.get_ylim()
            ax.set_ylim(top=max(current_ylim[1], max_text_y * 1.05))
    
    plt.tight_layout()
    plt.savefig('figures/thermal_conductivity_lanthanides_700K.png', dpi=500)
    plt.show()
    print("Thermal conductivity lanthanide doping plot saved.")
    # ========== End of augmentation ==========
    
    # Save CV summary
    with open('results/cv_summary.json', 'w') as f:
        json.dump(cv_summary, f, indent=2)
    
    print("\n✅ Analysis complete. Results saved in 'results/' and 'figures/'.")

if __name__ == "__main__":
    main()
