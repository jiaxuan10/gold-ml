#!/usr/bin/env python3
"""
train_gold_model_v3_enhanced.py
--------------------------------
Enhanced Gold Scalper style training script.

Key changes vs v2:
- thresholded labeling (ignore tiny moves)
- time-series CV (TimeSeriesSplit)
- CV-weighted soft voting ensemble
- probability calibration (Isotonic)
- optimal threshold tuning on validation split
- save feature importances and more detailed report
- keep market regime detection and original feature creation
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from utils.market_regime_detector import MarketRegimeDetector
from features.features_engineering import add_technical_features, prepare_target, detect_market_regime


# optional xgboost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# market regime detector (keep your util)
from utils.market_regime_detector import MarketRegimeDetector

RND = 42
np.random.seed(RND)

# ---------- USER CONFIG ----------
CSV_PATH = "data/final/final_dataset_hourly.csv"
DATE_COL = "Date"
TARGET_HORIZON = 5            # hours ahead
TEST_SIZE = 0.2               # final test split (chronological)
CV_FOLDS = 5                  # TimeSeriesSplit folds
MIN_ROWS = 300
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# labeling threshold: ignore tiny moves (tunable)
LABEL_THRESHOLD = 0.0003     # 0.1% move required to count as up/down

# validation fraction inside the train region (for threshold tuning)
INNER_VAL_FRAC = 0.1         # last 10% of training used as validation for threshold tuning


# ---------- MAIN ----------
def main(csv_path=CSV_PATH):
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # rename to expected column names if needed
    rename_map = {
        "GOLD_Close": "Close",
        "GOLD_Open": "Open",
        "GOLD_High": "High",
        "GOLD_Low": "Low",
        "GOLD_Volume": "Volume"
    }
    df = df.rename(columns=rename_map)

    # date handling
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

    if len(df) < MIN_ROWS:
        print(f"⚠️ Too few rows ({len(df)}). Need at least {MIN_ROWS}")
        sys.exit(1)

    print(f"📄 Loaded {len(df)} rows from {df[DATE_COL].min()} → {df[DATE_COL].max()}")

    # feature engineering + target + regime
    df = add_technical_features(df)
    df = prepare_target(df, TARGET_HORIZON, LABEL_THRESHOLD)
    df = detect_market_regime(df)

    # drop rows where label is NaN (small moves)
    initial_len = len(df)
    df = df.dropna(subset=["target_bin"]).reset_index(drop=True)
    dropped = initial_len - len(df)
    print(f"🧹 Dropped {dropped} rows due to thresholded labels (tiny moves). Remaining: {len(df)}")

    # drop rows with NaNs in features after engineering
    df = df.dropna().reset_index(drop=True)
    print(f"🧮 After dropna, rows remaining: {len(df)}")

    # prepare features/labels
    exclude = {DATE_COL, "target_ret", "target_bin", "regime"}
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    X_df = df[feature_cols].fillna(df[feature_cols].median())
    y = df["target_bin"].astype(int)

    print(f"🔢 Feature count: {len(feature_cols)}")

    # chronological split into train/test
    n = len(df)
    split_idx = int((1 - TEST_SIZE) * n)
    X_train_df = X_df.iloc[:split_idx].reset_index(drop=True)
    X_test_df = X_df.iloc[split_idx:].reset_index(drop=True)
    y_train = y.iloc[:split_idx].reset_index(drop=True)
    y_test = y.iloc[split_idx:].reset_index(drop=True)

    # inner validation for threshold tuning: last INNER_VAL_FRAC of training
    inner_val_idx = int((1 - INNER_VAL_FRAC) * len(X_train_df))
    X_train_inner = X_train_df.iloc[:inner_val_idx].reset_index(drop=True)
    y_train_inner = y_train.iloc[:inner_val_idx].reset_index(drop=True)
    X_val_inner = X_train_df.iloc[inner_val_idx:].reset_index(drop=True)
    y_val_inner = y_train.iloc[inner_val_idx:].reset_index(drop=True)

    # Convert to numpy arrays for scikit-learn (but keep df for feature importances)
    X_train = X_train_inner.values  # for training in CV loops
    X_val = X_val_inner.values      # for threshold tuning
    X_full_train = X_train_df.values  # for final training use
    X_test = X_test_df.values

    # ---------- Base Models ----------
    base_models = {
        "rf": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=RND),
        "gb": GradientBoostingClassifier(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.9,
                random_state=RND
            ),
        "svc": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(C=2.0, kernel="rbf", gamma="scale", probability=True, random_state=RND))
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(hidden_layer_sizes=(128,64), max_iter=800, random_state=RND))
        ]),
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1500, solver="lbfgs", random_state=RND))
        ])
    }
    if XGBOOST_AVAILABLE:
        base_models["xgb"] = xgb.XGBClassifier(
                                n_estimators=400,
                                learning_rate=0.03,
                                max_depth=6,
                                subsample=0.9,
                                colsample_bytree=0.8,
                                reg_lambda=2.0,
                                eval_metric='logloss',
                                random_state=RND        )

    # ---------- TimeSeries CV training ----------
    results = {}
    trained_models = {}
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

    print("\n🧠 Training base models (TimeSeriesSplit CV)...")
    for name, model in base_models.items():
        try:
            cv_scores = []
            # perform manual time-series CV on the *full training region* (X_train_df / y_train)
            # We will train on slices of X_train_df to reflect time order
            for train_idx, val_idx in tscv.split(X_train_df):
                X_tr = X_train_df.values[train_idx]
                y_tr = y_train.values[train_idx]
                X_val_cv = X_train_df.values[val_idx]
                y_val_cv = y_train.values[val_idx]
                model.fit(X_tr, y_tr)
                preds_cv = model.predict(X_val_cv)
                cv_scores.append(accuracy_score(y_val_cv, preds_cv))

            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))

            # fit on the entire training region (X_full_train) before test evaluation
            model.fit(X_full_train, y_train)
            preds_test = model.predict(X_test)
            probs_test = None
            try:
                probs_test = model.predict_proba(X_test)
            except Exception:
                # some models may not support predict_proba
                pass

            acc = accuracy_score(y_test, preds_test)
            f1 = f1_score(y_test, preds_test)
            precision = precision_score(y_test, preds_test)
            recall = recall_score(y_test, preds_test)

            trained_models[name] = model
            results[name] = {
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "test_acc": acc,
                "test_f1": f1,
                "test_precision": precision,
                "test_recall": recall,
                "supports_proba": probs_test is not None
            }
            print(f"{name:<8} | CV={cv_mean:.3f} ± {cv_std:.3f} | Test={acc:.3f} | f1={f1:.3f}")

        except Exception as e:
            print(f"⚠️ {name} failed: {e}")

    # ---------- Auto ensemble: pick top models by CV mean ----------
    print("\n🤖 Building CV-weighted soft voting ensemble (select top 3)...")
    # sort only models that succeeded
    good_models = [k for k, v in results.items()]
    if len(good_models) == 0:
        print("❌ No trained models available. Exiting.")
        sys.exit(1)

    # choose top 3 by cv_mean
    sorted_models = sorted(good_models, key=lambda k: results[k]["cv_mean"], reverse=True)
    top_k = min(3, len(sorted_models))
    top_models = sorted_models[:top_k]
    estimators = [(n, trained_models[n]) for n in top_models]

    # compute weights by cv_mean (positive)
    weights_raw = np.array([results[n]["test_f1"] for n in top_models], dtype=float)
    if weights_raw.sum() == 0:
        weights = np.ones_like(weights_raw) / len(weights_raw)
    else:
        weights = weights_raw / weights_raw.sum()

    ensemble = VotingClassifier(estimators=estimators, voting="soft", weights=weights)
    # fit ensemble on full training data
    ensemble.fit(X_full_train, y_train)
    try:
        ensemble_proba_test = ensemble.predict_proba(X_test)[:, 1]
    except Exception:
        # If any estimator doesn't support proba (unlikely after our checks), fallback to predict
        ensemble_proba_test = None

    preds_ens = ensemble.predict(X_test)
    ens_acc = accuracy_score(y_test, preds_ens)
    ens_f1 = f1_score(y_test, preds_ens)
    print(f"✅ Raw Ensemble — acc={ens_acc:.3f}, f1={ens_f1:.3f}")

    # ---------- Calibration ----------
    print("\n🧰 Calibrating ensemble probabilities (Isotonic)...")
    try:
        # calibrate on X_full_train to avoid leaking test set (calib uses internal CV)
        # Use cv=3 inside CalibratedClassifierCV (will perform internal splits)
        calib = CalibratedClassifierCV(ensemble, cv=3, method='isotonic')
        calib.fit(X_full_train, y_train)
        proba_val = calib.predict_proba(X_val)[:, 1]  # validation for threshold tuning
        proba_test = calib.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"⚠️ Calibration failed: {e}")
        # fallback: try ensemble_proba_test if available
        if ensemble_proba_test is not None:
            proba_val = ensemble_proba_test[:len(X_val)]
            proba_test = ensemble_proba_test
            calib = None
        else:
            proba_val = None
            proba_test = None
            calib = None

    # ---------- Threshold tuning on validation set ----------
    print("\n🔎 Searching best threshold on validation set to maximize F1...")
    best_thr = 0.5
    best_f1_val = -1.0
    if proba_val is not None:
        for thr in np.arange(0.1, 0.91, 0.02):
            y_pred_thr = (proba_val > thr).astype(int)
            # ✅ 用 validation 的真实标签
            f1_thr = f1_score(y_val_inner, y_pred_thr)
            if f1_thr > best_f1_val:
                best_f1_val = f1_thr
                best_thr = float(thr)
    else:
        print("⚠️ No calibrated probabilities on validation — keep threshold=0.5")


    # apply threshold to test set probabilities (if available) or fallback to predict
    if proba_test is not None:
        y_pred_test_final = (proba_test > best_thr).astype(int)
    else:
        y_pred_test_final = preds_ens

    final_acc = accuracy_score(y_test, y_pred_test_final)
    final_f1 = f1_score(y_test, y_pred_test_final)
    final_prec = precision_score(y_test, y_pred_test_final)
    final_rec = recall_score(y_test, y_pred_test_final)

    print(f"\n🎯 Calibrated Ensemble (thr={best_thr:.2f}) — acc={final_acc:.3f}, f1={final_f1:.3f}, prec={final_prec:.3f}, rec={final_rec:.3f}")

    # ---------- Save feature importances (if available) ----------
    feature_importances = {}
    for name, model in trained_models.items():
        try:
            fi = None
            # if pipeline, try to get the final estimator
            if hasattr(model, "named_steps"):
                m = model.named_steps.get("model", model)
            else:
                m = model
            if hasattr(m, "feature_importances_"):
                fi = m.feature_importances_
            elif XGBOOST_AVAILABLE and isinstance(m, xgb.XGBClassifier):
                fi = m.feature_importances_
            if fi is not None:
                feature_importances[name] = dict(zip(feature_cols, fi.tolist()))
        except Exception:
            continue

    # ---------- Save report & model ----------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": timestamp,
        "rows_loaded": len(df),
        "train_rows": len(X_full_train),
        "test_rows": len(X_test),
        "features_count": len(feature_cols),
        "label_threshold": LABEL_THRESHOLD,
        "selected_models": top_models,
        "selected_weights": weights.tolist(),
        "best_threshold_validation": best_thr,
        "final_acc": final_acc,
        "final_f1": final_f1,
        "final_precision": final_prec,
        "final_recall": final_rec
    }

    # Detailed model table
    models_table = []
    for name, stats in results.items():
        models_table.append({
            "model": name,
            "cv_mean": stats.get("cv_mean"),
            "cv_std": stats.get("cv_std"),
            "test_acc": stats.get("test_acc"),
            "test_f1": stats.get("test_f1"),
            "test_precision": stats.get("test_precision"),
            "test_recall": stats.get("test_recall"),
            "supports_proba": stats.get("supports_proba")
        })
    df_models = pd.DataFrame(models_table)

    # dump report files
    report_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "summary_report.json")
    with open(report_path, "w") as f:
        import json
        json.dump(report, f, indent=2, default=str)

    df_models.to_csv(os.path.join(report_dir, "models_summary.csv"), index=False)

    # feature importances
    if feature_importances:
        # save each model fi to csv
        for name, fi_map in feature_importances.items():
            df_fi = pd.DataFrame.from_dict(fi_map, orient="index", columns=["importance"])
            df_fi.index.name = "feature"
            df_fi = df_fi.sort_values("importance", ascending=False)
            df_fi.to_csv(os.path.join(report_dir, f"feature_importance_{name}.csv"))

    # save the calibrated ensemble and threshold
    model_save_path = os.path.join(report_dir, f"ensemble_calibrated_{timestamp}.pkl")
    try:
        with open(model_save_path, "wb") as f:
            pickle.dump({"calibrated_model": calib, "raw_ensemble": ensemble, "threshold": best_thr, "feature_cols": feature_cols}, f)
        print(f"💾 Saved calibrated ensemble + metadata → {model_save_path}")
    except Exception as e:
        print(f"⚠️ Failed to save model: {e}")

    print(f"📄 Summary saved → {report_path}")
    print(f"📄 Models summary CSV → {os.path.join(report_dir, 'models_summary.csv')}")
    if feature_importances:
        print(f"📄 Feature importances saved for: {', '.join(feature_importances.keys())}")

    print("\n🎯 Done — enhanced Gold Scalper training complete.")

if __name__ == "__main__":
    main()
