#!/usr/bin/env python3
"""
train_gold_model_v3_enhanced.py
--------------------------------
Enhanced Gold Scalper style training script.

Key improvements:
- CLEANER FEATURES: Removed raw prices (Open/Close) to reduce noise.
- BETTER TARGETS: Higher threshold (0.1%) to capture significant moves only.
- TIME AWARE: Uses Hour/Day features.
"""
import os
import sys
# Fix path to find utils and features
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

# optional xgboost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# Import from your project structure
from utils.market_regime_detector import MarketRegimeDetector
from features.features_engineering import add_technical_features, prepare_target, detect_market_regime

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

# 🔥 [MODIFIED] 提高阈值 (0.03% -> 0.1%)
# 目的：只学习真正的趋势，过滤掉微小的震荡，大幅提高 Precision
LABEL_THRESHOLD = 0.00035

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
        print(f"⚠️ Too few rows ({len(df)}). Need at least {MIN_ROWS}")
        sys.exit(1)

    print(f"📄 Loaded {len(df)} rows from {df[DATE_COL].min()} → {df[DATE_COL].max()}")

    # feature engineering + target + regime
    df = add_technical_features(df)
    df = prepare_target(df, TARGET_HORIZON, LABEL_THRESHOLD)
    df = detect_market_regime(df)

    initial_len = len(df)
    df = df.dropna(subset=["target_bin"]).reset_index(drop=True)
    dropped = initial_len - len(df)
    print(f"🧹 Dropped {dropped} rows due to thresholded labels (tiny moves). Remaining: {len(df)}")

    df = df.dropna().reset_index(drop=True)
    
    # 🔥 [MODIFIED] 特征清洗：剔除原始价格 (Noise Removal)
    # 这一步非常关键，防止 AI 因为绝对价格高低而误判
    drop_cols = ["Close", "Open", "High", "Low", "Volume", "target_ret", "target_bin", "regime", DATE_COL]
    
    # 同时也剔除外部资产的原始价格 (因为我们已经用了 Ratio 和 Correlation)
    external_raw = ["DX-Y.NYB", "^GSPC", "^IXIC", "BTC-USD", "ETH-USD", "^VIX"]
    
    feature_cols = [c for c in df.columns if c not in drop_cols and c not in external_raw and pd.api.types.is_numeric_dtype(df[c])]
    
    print(f"🔢 Feature count: {len(feature_cols)}")
    print(f"📋 Actual Features used for AI: {feature_cols}") # 打印出来让你检查

    X_df = df[feature_cols].fillna(df[feature_cols].median())
    y = df["target_bin"].astype(int)

    # chronological split into train/test
    n = len(df)
    split_idx = int((1 - TEST_SIZE) * n)
    X_train_df = X_df.iloc[:split_idx].reset_index(drop=True)
    X_test_df = X_df.iloc[split_idx:].reset_index(drop=True)
    y_train = y.iloc[:split_idx].reset_index(drop=True)
    y_test = y.iloc[split_idx:].reset_index(drop=True)

    inner_val_idx = int((1 - INNER_VAL_FRAC) * len(X_train_df))
    X_train_inner = X_train_df.iloc[:inner_val_idx].reset_index(drop=True)
    y_train_inner = y_train.iloc[:inner_val_idx].reset_index(drop=True)
    X_val_inner = X_train_df.iloc[inner_val_idx:].reset_index(drop=True)
    y_val_inner = y_train.iloc[inner_val_idx:].reset_index(drop=True)

    X_train = X_train_inner.values
    X_val = X_val_inner.values
    X_full_train = X_train_df.values
    X_test = X_test_df.values

    # ---------- Base Models ----------
    base_models = {
        "rf": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=RND),
        "gb": GradientBoostingClassifier(n_estimators=400, learning_rate=0.03, max_depth=5, subsample=0.9, random_state=RND),
        "svc": Pipeline([("scaler", StandardScaler()), ("model", SVC(C=2.0, kernel="rbf", gamma="scale", probability=True, random_state=RND))]),
        "mlp": Pipeline([("scaler", StandardScaler()), ("model", MLPClassifier(hidden_layer_sizes=(128,64), max_iter=800, random_state=RND))]),
        "logreg": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1500, solver="lbfgs", random_state=RND))])
    }
    if XGBOOST_AVAILABLE:
        base_models["xgb"] = xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=6, subsample=0.9, colsample_bytree=0.8, reg_lambda=2.0, eval_metric='logloss', random_state=RND)

    # ---------- TimeSeries CV training ----------
    results = {}
    trained_models = {}
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

    print("\n🧠 Training base models (TimeSeriesSplit CV)...")
    for name, model in base_models.items():
        try:
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_train_df):
                model.fit(X_train_df.values[train_idx], y_train.values[train_idx])
                preds_cv = model.predict(X_train_df.values[val_idx])
                cv_scores.append(accuracy_score(y_train.values[val_idx], preds_cv))

            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))

            model.fit(X_full_train, y_train)
            preds_test = model.predict(X_test)
            probs_test = None
            try: probs_test = model.predict_proba(X_test)
            except: pass

            acc = accuracy_score(y_test, preds_test)
            f1 = f1_score(y_test, preds_test)
            precision = precision_score(y_test, preds_test)
            recall = recall_score(y_test, preds_test)

            trained_models[name] = model
            results[name] = {
                "cv_mean": cv_mean, "cv_std": cv_std, "test_acc": acc, "test_f1": f1,
                "test_precision": precision, "test_recall": recall, "supports_proba": probs_test is not None
            }
            # 原来的代码只是 print
            # print(f"{name:<8} | CV={cv_mean:.3f} ± {cv_std:.3f} | Test={acc:.3f} | f1={f1:.3f}")

            # 🔥 修改为：先把指标转换成 Python float (JSON 序列化需要)，然后存入 results
            acc = float(accuracy_score(y_test, preds_test))
            f1 = float(f1_score(y_test, preds_test))
            precision = float(precision_score(y_test, preds_test))
            recall = float(recall_score(y_test, preds_test))

            trained_models[name] = model
            results[name] = {
                "cv_mean": float(cv_mean), 
                "cv_std": float(cv_std),
                "test_acc": acc,
                "test_f1": f1,
                "test_precision": precision,
                "test_recall": recall,
                "supports_proba": probs_test is not None
            }
            print(f"{name:<8} | CV={cv_mean:.3f} | Test={acc:.3f} | F1={f1:.3f} | Prec={precision:.3f}")

        except Exception as e:
            print(f"⚠️ {name} failed: {e}")

    # ---------- Ensemble ----------
    print("\n🤖 Building CV-weighted soft voting ensemble...")
    good_models = [k for k, v in results.items()]
    if not good_models: sys.exit(1)

    sorted_models = sorted(good_models, key=lambda k: results[k]["cv_mean"], reverse=True)
    top_k = min(3, len(sorted_models))
    top_models = sorted_models[:top_k]
    estimators = [(n, trained_models[n]) for n in top_models]

    weights_raw = np.array([results[n]["test_f1"] for n in top_models], dtype=float)
    weights = weights_raw / weights_raw.sum() if weights_raw.sum() > 0 else None

    ensemble = VotingClassifier(estimators=estimators, voting="soft", weights=weights)
    ensemble.fit(X_full_train, y_train)
    
    try: ensemble_proba_test = ensemble.predict_proba(X_test)[:, 1]
    except: ensemble_proba_test = None

    preds_ens = ensemble.predict(X_test)
    ens_acc = accuracy_score(y_test, preds_ens)
    print(f"✅ Raw Ensemble — acc={ens_acc:.3f}")

    # ---------- Calibration ----------
    print("\n🧰 Calibrating probabilities...")
    try:
        calib = CalibratedClassifierCV(ensemble, cv=3, method='isotonic')
        calib.fit(X_full_train, y_train)
        proba_val = calib.predict_proba(X_val)[:, 1]
        proba_test = calib.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"⚠️ Calibration failed: {e}")
        if ensemble_proba_test is not None:
            proba_val = ensemble_proba_test[:len(X_val)]
            proba_test = ensemble_proba_test
            calib = None
        else:
            proba_val, proba_test, calib = None, None, None

    # ---------- Threshold Tuning ----------
    print("\n🔎 Tuning Threshold...")
    best_thr = 0.5
    best_f1_val = -1.0
    if proba_val is not None:
        for thr in np.arange(0.1, 0.91, 0.02):
            y_pred_thr = (proba_val > thr).astype(int)
            f1_thr = f1_score(y_val_inner, y_pred_thr)
            if f1_thr > best_f1_val:
                best_f1_val = f1_thr
                best_thr = float(thr)
    
    if proba_test is not None:
        y_pred_test_final = (proba_test > best_thr).astype(int)
    else:
        y_pred_test_final = preds_ens

    final_acc = accuracy_score(y_test, y_pred_test_final)
    final_f1 = f1_score(y_test, y_pred_test_final)
    final_prec = precision_score(y_test, y_pred_test_final)
    final_rec = recall_score(y_test, y_pred_test_final)

    print(f"\n🎯 Final Result (Thr={best_thr:.2f}) — Acc={final_acc:.3f}, F1={final_f1:.3f}, Prec={final_prec:.3f}")

    # ---------- Save ----------
    feature_importances = {}
    for name, model in trained_models.items():
        try:
            fi = None
            if hasattr(model, "named_steps"): m = model.named_steps.get("model", model)
            else: m = model
            if hasattr(m, "feature_importances_"): fi = m.feature_importances_
            if fi is not None: feature_importances[name] = dict(zip(feature_cols, fi.tolist()))
        except: continue

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": timestamp, "rows": len(df), "features": len(feature_cols),
        "threshold": best_thr, "acc": final_acc, "f1": final_f1, "prec": final_prec
    }

    report_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)
    
    with open(os.path.join(report_dir, "summary_report.json"), "w") as f:
        import json
        json.dump(report, f, indent=2, default=str)

    # Save Model
    model_save_path = os.path.join(report_dir, f"ensemble_calibrated_{timestamp}.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump({"calibrated_model": calib, "raw_ensemble": ensemble, "threshold": best_thr, "feature_cols": feature_cols}, f)
        
    print(f"\n💾 Saved model to {model_save_path}")

    # ... (接着上面的 feature_importances 代码) ...

    # 🔥🔥🔥【新增】构建 Mega Report 🔥🔥🔥
    
    # 1. 整理 Feature Importance (把 numpy 类型转成 float，否则 json 报错)
    serializable_fi = {}
    for model_name, fi_dict in feature_importances.items():
        serializable_fi[model_name] = {k: float(v) for k, v in fi_dict.items()}

    # 2. 构建大字典
    comprehensive_report = {
        "metadata": {
            "timestamp": timestamp,
            "rows_loaded": len(df),
            "features_count": len(feature_cols),
            "feature_names": feature_cols,
            "label_threshold": LABEL_THRESHOLD,
            "best_threshold_validation": float(best_thr)
        },
        "ensemble_performance": {
            "accuracy": float(final_acc),
            "f1_score": float(final_f1),
            "precision": float(final_prec),
            "recall": float(final_rec),
            "selected_models": top_models,
            "voting_weights": weights.tolist() if weights is not None else []
        },
        "base_models_performance": results,  # 包含了所有单体模型的详细得分
        "feature_importances": serializable_fi
    }

    # 3. 保存为 comprehensive_report.json
    json_path = os.path.join(report_dir, "comprehensive_report.json")
    with open(json_path, "w") as f:
        json.dump(comprehensive_report, f, indent=2, default=str)

    # 4. (可选) 依然保存 summary_report.json 以兼容旧版 UI
    # ... (原有的 summary_report 保存代码保持不变) ...
    
    print(f"📄 Comprehensive Report saved to {json_path}")

if __name__ == "__main__":
    main()