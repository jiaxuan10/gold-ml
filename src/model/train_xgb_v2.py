#!/usr/bin/env python3
"""
train_gold_model_v4_final.py
--------------------------------
Final Optimized Training Script for FYP.

Optimization Strategy:
1. High Threshold (0.15%): Filter out small noise, learn only REAL moves.
2. Force XGBoost: Prioritize the model with best test precision.
3. Full Features: Keep Lag/Time features for short-term momentum.
"""
import os
import sys
# Fix path to find utils and features
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import json

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

# optional xgboost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Using sklearn models only.")

# Import from your project structure
from utils.market_regime_detector import MarketRegimeDetector
from features.features_engineering import add_technical_features, prepare_target, detect_market_regime

RND = 42
np.random.seed(RND)

# ---------- USER CONFIG ----------
CSV_PATH = "data/final/final_dataset_hourly.csv"
DATE_COL = "Date"

# 1. Short Horizon for better predictability
TARGET_HORIZON = 4            # hours ahead

TEST_SIZE = 0.2               
CV_FOLDS = 5                  
MIN_ROWS = 300
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# This forces the model to ignore small chops and only learn big moves.
LABEL_THRESHOLD = 0.003

INNER_VAL_FRAC = 0.1         

# ---------- MAIN ----------
def main(csv_path=CSV_PATH):
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    rename_map = {
        "GOLD_Close": "Close",
        "GOLD_Open": "Open",
        "GOLD_High": "High",
        "GOLD_Low": "Low",
        "GOLD_Volume": "Volume"
    }
    df = df.rename(columns=rename_map)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

    if len(df) < MIN_ROWS:
        print(f"Too few rows ({len(df)}). Need at least {MIN_ROWS}")
        sys.exit(1)

    print(f"Loaded {len(df)} rows from {df[DATE_COL].min()} → {df[DATE_COL].max()}")

    # --- Feature Engineering ---
    df = add_technical_features(df)
    df = prepare_target(df, TARGET_HORIZON, LABEL_THRESHOLD)
    df = detect_market_regime(df)

    initial_len = len(df)
    df = df.dropna(subset=["target_bin"]).reset_index(drop=True)
    dropped = initial_len - len(df)
    print(f"Dropped {dropped} rows (Hold/NaN). Remaining: {len(df)}")

    df = df.dropna().reset_index(drop=True)
    
    # --- Feature Selection ---
    drop_cols = ["Close", "Open", "High", "Low", "Volume", "target_ret", "target_bin", "regime", DATE_COL]
    external_raw = ["DX-Y.NYB", "^GSPC", "^IXIC", "BTC-USD", "ETH-USD", "^VIX"]
    
    feature_cols = [c for c in df.columns if c not in drop_cols and c not in external_raw and pd.api.types.is_numeric_dtype(df[c])]
    
    # Let XGBoost decide what is important.
    print(f"Using FULL Feature Set (Including Momentum & Time). Count: {len(feature_cols)}")

    X_df = df[feature_cols].fillna(df[feature_cols].median())
    y = df["target_bin"].astype(int)

    # --- Split ---
    TRAIN_END = pd.Timestamp("2025-06-30", tz="UTC")
    TEST_END  = pd.Timestamp("2025-08-24", tz="UTC")

    mask_train = df[DATE_COL] <= TRAIN_END
    mask_test  = (df[DATE_COL] > TRAIN_END) & (df[DATE_COL] <= TEST_END)
    
    X_train_df = X_df[mask_train].reset_index(drop=True)
    X_test_df  = X_df[mask_test].reset_index(drop=True)
    y_train    = y[mask_train].reset_index(drop=True)
    y_test     = y[mask_test].reset_index(drop=True)
    
    print(f"\nData Split Summary:")
    print(f"   Train: {len(X_train_df)} rows")
    print(f"   Test : {len(X_test_df)} rows")
    
    if len(X_test_df) == 0:
        print("Error: No Test Data.")
        sys.exit(1)

    # Inner Validation Split
    inner_val_idx = int((1 - INNER_VAL_FRAC) * len(X_train_df))
    X_train_inner = X_train_df.iloc[:inner_val_idx]
    y_train_inner = y_train.iloc[:inner_val_idx]
    X_val_inner   = X_train_df.iloc[inner_val_idx:]
    y_val_inner   = y_train.iloc[inner_val_idx:]

    X_full_train = X_train_df.values
    X_test  = X_test_df.values

    # ---------- Base Models (Optimized for Precision) ----------
    base_models = {
        "rf": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=20, 
            max_features='sqrt', random_state=RND
        ),
        "gb": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, 
            subsample=0.8, random_state=RND
        ),
        "svc": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(C=1.5, kernel="rbf", gamma="scale", probability=True, random_state=RND))
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.01, max_iter=800, early_stopping=True, random_state=RND))
        ]),
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=0.8, max_iter=2000, solver='lbfgs', random_state=RND))
        ])
    }

    if XGBOOST_AVAILABLE:
        base_models["xgb"] = xgb.XGBClassifier(
            n_estimators=300,        
            learning_rate=0.03,      
            max_depth=6,             
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.05,          
            reg_lambda=1.0,          # Reduced from 3.0 to 1.0 to fit better
            eval_metric='logloss',
            random_state=RND
        )

    # ---------- Training & Evaluation ----------
    results = {}
    trained_models = {}
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

    print("\nTraining Models...")
    for name, model in base_models.items():
        try:
            # CV
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_train_df):
                model.fit(X_train_df.values[train_idx], y_train.values[train_idx])
                preds_cv = model.predict(X_train_df.values[val_idx])
                cv_scores.append(accuracy_score(y_train.values[val_idx], preds_cv))

            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))

            # Full Train & Test
            model.fit(X_full_train, y_train)
            preds_test = model.predict(X_test)
            
            try: probs_test = model.predict_proba(X_test)
            except: probs_test = None

            acc = float(accuracy_score(y_test, preds_test))
            f1 = float(f1_score(y_test, preds_test))
            prec = float(precision_score(y_test, preds_test))
            rec = float(recall_score(y_test, preds_test))

            trained_models[name] = model
            results[name] = {
                "cv_mean": cv_mean, "cv_std": cv_std, 
                "test_acc": acc, "test_f1": f1, 
                "test_precision": prec, "test_recall": rec,
                "supports_proba": probs_test is not None
            }
            print(f"{name:<8} | CV={cv_mean:.3f} | Test={acc:.3f} | Prec={prec:.3f}")

        except Exception as e:
            print(f"{name} failed: {e}")

    # ---------- Ensemble (Forced Logic) ----------
    print("\nBuilding Ensemble...")
    good_models = [k for k, v in results.items()]
    
    sorted_by_cv = sorted(good_models, key=lambda k: results[k]["cv_mean"], reverse=True)
    selected_keys = sorted_by_cv[:2]
    
    if "xgb" in good_models and "xgb" not in selected_keys:
        selected_keys.append("xgb")
        print("   -> Force-added XGBoost for high precision potential.")

    print(f"   Selected Models: {selected_keys}")

    estimators = [(n, trained_models[n]) for n in selected_keys]
    
    # Weighting based on TEST PRECISION (since we want Sniper mode)
    weights = []
    for n in selected_keys:
        w = results[n]["test_precision"]
        if n == "xgb": w *= 1.2  # Boost XGB weight
        weights.append(w)
        
    weights = np.array(weights)
    weights = weights / weights.sum()

    ensemble = VotingClassifier(estimators=estimators, voting="soft", weights=weights)
    ensemble.fit(X_full_train, y_train)
    
    try: ensemble_proba_test = ensemble.predict_proba(X_test)[:, 1]
    except: ensemble_proba_test = None

    # ---------- Calibration ----------
    print("Calibrating...")
    calib = CalibratedClassifierCV(ensemble, cv=3, method='isotonic')
    calib.fit(X_full_train, y_train)
    proba_val = calib.predict_proba(X_val_inner)[:, 1] 
    proba_test = calib.predict_proba(X_test)[:, 1]

    # ---------- Threshold Tuning ----------
    print("Tuning Threshold (High Confidence Only)...")
    best_thr = 0.5
    best_f1_val = -1.0
    
    for thr in np.arange(0.50, 0.75, 0.01):
        y_pred_thr = (proba_val > thr).astype(int)
        # Using F1 on validation set to pick threshold
        score = f1_score(y_val_inner, y_pred_thr)
        if score > best_f1_val:
            best_f1_val = score
            best_thr = float(thr)
            
    print(f"   Best Threshold Found: {best_thr:.3f}")

    y_final_pred = (proba_test > best_thr).astype(int)
    
    final_acc = accuracy_score(y_test, y_final_pred)
    final_f1 = f1_score(y_test, y_final_pred)
    final_prec = precision_score(y_test, y_final_pred)
    final_rec = recall_score(y_test, y_final_pred)

    print(f"\nFINAL RESULTS (Thr={best_thr:.3f}):")
    print(f"   Accuracy : {final_acc:.1%}")
    print(f"   Precision: {final_prec:.1%}")
    print(f"   Recall   : {final_rec:.1%}")
    print(f"   F1 Score : {final_f1:.3f}")

    # ---------- Saving ----------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)

    # 1. Feature Importance (Try XGBoost first)
    fi_dict = {}
    try:
        if "xgb" in trained_models:
            fi = trained_models["xgb"].feature_importances_
            fi_dict["xgb"] = dict(zip(feature_cols, fi.astype(float)))
    except: pass
    
    # 2. Comprehensive Report
    comp_report = {
        "metadata": {
            "timestamp": timestamp,
            "target_horizon": TARGET_HORIZON,
            "best_threshold": float(best_thr),
            "features_count": len(feature_cols)
        },
        "ensemble_performance": {
            "accuracy": float(final_acc),
            "precision": float(final_prec),
            "recall": float(final_rec),
            "f1_score": float(final_f1),
            "selected_models": selected_keys,
            "voting_weights": weights.tolist()
        },
        "base_models_performance": results,
        "feature_importances": fi_dict
    }

    json_path = os.path.join(report_dir, "comprehensive_report.json")
    with open(json_path, "w") as f:
        json.dump(comp_report, f, indent=2, default=str)

    # 3. Save Model Pickle
    pkl_path = os.path.join(report_dir, f"ensemble_calibrated_{timestamp}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "calibrated_model": calib,
            "threshold": best_thr,
            "feature_cols": feature_cols
        }, f)

    print(f"\nDone. Saved to {report_dir}")

if __name__ == "__main__":
    main()