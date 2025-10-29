# # train_gold_model.py
# """
# Train ensemble model for gold daily dataset (single-file).
# Input: CSV with Date, Open, High, Low, Close, and other features (your CSV).
# Output: saved ensemble model, model_report.csv, and feature_importances if available.

# Based on ideas from ai-gold-scalper ensemble pipeline (feature generation, base models, ensembles).
# """

# import os
# import sys
# import pickle
# from datetime import datetime
# import numpy as np
# import pandas as pd

# # ML libs
# from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import precision_score, recall_score, f1_score
# from utils.market_regime_detector import MarketRegimeDetector

# # optional
# try:
#     import xgboost as xgb
#     XGBOOST_AVAILABLE = True
# except Exception:
#     XGBOOST_AVAILABLE = False

# RND = 42
# np.random.seed(RND)

# # ---------- USER CONFIG ----------
# CSV_PATH = "data/final/final_dataset_daily.csv"  # <-- 改成你的 CSV 路径（你给的 CSV）
# DATE_COL = "Date"
# TARGET_HORIZON_DAYS = 5   # next-day return classification
# TEST_SIZE = 0.2           # chronological split (tail)
# CV_FOLDS = 5
# MIN_ROWS = 200            # safety check
# OUTPUT_DIR = "models"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ---------- HELPERS: FEATURES ----------
# def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
#     """Add helpful technical features similar to generate_trading_features."""
#     df = df.copy()
#     # Ensure numeric
#     for col in ["Close","Open","High","Low"]:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors="coerce")
#     close = df["Close"]

#     # Moving averages
#     df["sma_5"] = close.rolling(5).mean()
#     df["sma_10"] = close.rolling(10).mean()
#     df["sma_20"] = close.rolling(20).mean()

#     # Volatility
#     df["vol_5"] = close.pct_change().rolling(5).std()
#     df["vol_10"] = close.pct_change().rolling(10).std()

#     # Momentum
#     df["mom_3"] = close.pct_change(3)
#     df["mom_5"] = close.pct_change(5)
#     # Momentum slope (acceleration)
#     df["mom_slope_3"] = df["mom_3"].diff()
#     df["mom_slope_5"] = df["mom_5"].diff()

#     # Bollinger (position)
#     bb_mid = close.rolling(20).mean()
#     bb_std = close.rolling(20).std()
#     df["bb_upper"] = bb_mid + 2 * bb_std
#     df["bb_lower"] = bb_mid - 2 * bb_std
#     df["bb_pos"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

#     # RSI (simple)
#     delta = close.diff()
#     gain = delta.where(delta > 0, 0).rolling(14).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
#     rs = gain / (loss.replace(0, np.nan))
#     df["rsi"] = 100 - (100 / (1 + rs))

#     # MACD (12,26,9)
#     ema12 = close.ewm(span=12, adjust=False).mean()
#     ema26 = close.ewm(span=26, adjust=False).mean()
#     macd = ema12 - ema26
#     signal = macd.ewm(span=9, adjust=False).mean()
#     df["macd"] = macd
#     df["macd_signal"] = macd - signal

#     # Price position relative to MA
#     df["price_sma20"] = close / df["sma_20"]

#     # Lags of close and returns
#     for l in [1,3,5]:
#         df[f"close_lag_{l}"] = close.shift(l)
#         if "Return" in df.columns:
#             df[f"ret_lag_{l}"] = df["Return"].shift(l)
#         else:
#             df[f"ret_lag_{l}"] = close.pct_change().shift(l)

#     # Fill or keep NaNs as will be dropped later
#     # ====== Macro / Cross-Asset Enhancements ======
#     macro_cols = ["DXY", "CPI", "FEDFUNDS", "CRUDE_OIL", "VIX", "US10Y", "M2", "SP500", "NASDAQ", "DJIA"]
#     for col in macro_cols:
#         if col in df.columns:
#             df[f"{col}_change"] = df[col].pct_change()
#             df[f"{col}_lag_3"] = df[col].shift(3)
#             df[f"{col}_lag_5"] = df[col].shift(5)
#     return df

# def prepare_target(df: pd.DataFrame, horizon_days: int = 3):
#     """Create smoothed multi-day target to reduce daily noise."""
#     df = df.copy()
#     # smoothed next-horizon average price / current - 1
#     # use rolling mean of forward prices to reduce single-day noise
#     df["target_ret"] = (df["Close"].shift(-horizon_days).rolling(horizon_days).mean() / df["Close"]) - 1
#     df["target_bin"] = (df["target_ret"] > 0).astype(int)
#     # optional smoothed return for features/analysis
#     df["smooth_ret"] = df["Return"].rolling(horizon_days).mean()
#     return df
# def prepare_regression_target(df: pd.DataFrame, horizon_days: int = 3):
#     """Predict continuous future return instead of binary up/down."""
#     df = df.copy()
#     df["target_reg"] = (df["Close"].shift(-horizon_days) / df["Close"] - 1)
#     # 平滑目标，减少噪音
#     df["target_reg_smooth"] = df["target_reg"].rolling(horizon_days).mean()
#     return df

# def classify_return_strength(df: pd.DataFrame, horizon_days: int = 5):
#     """Classify target into 3 classes: -1=down, 0=neutral, 1=up."""
#     df = df.copy()
#     df["future_ret"] = (df["Close"].shift(-horizon_days) / df["Close"] - 1) * 100
#     threshold = df["future_ret"].std() * 0.5  # 动态阈值：波动率一半
#     df["target_multi"] = np.select(
#         [df["future_ret"] > threshold, df["future_ret"] < -threshold],
#         [1, -1],
#         default=0
#     )
#     return df


# def detect_market_regime(df: pd.DataFrame):
#     detector = MarketRegimeDetector(ma_fast=20, ma_slow=50)
#     regime_df = detector.detect_regime(df)
#     df = pd.merge(df, regime_df[["Date", "regime"]], on="Date", how="left")
#     return df

# # ---------- MAIN ----------
# def main(csv_path=CSV_PATH):
#     if not os.path.exists(csv_path):
#         print(f"CSV not found: {csv_path}")
#         sys.exit(1)

#     df = pd.read_csv(csv_path)
#     # Optional: load regime labels if available
#     regime_path = os.path.join(os.path.dirname(csv_path), "market_regimes.csv")
#     if os.path.exists(regime_path):
#         regime_df = pd.read_csv(regime_path)
#         if "Date" in regime_df.columns and "regime" in regime_df.columns:
#             regime_df["Date"] = pd.to_datetime(regime_df["Date"])
#             df["Date"] = pd.to_datetime(df["Date"])
#             df = pd.merge(df, regime_df[["Date", "regime"]], on="Date", how="left")
#             print("✅ Market regime feature added.")
#         else:
#             print("⚠️ Regime file found but columns missing.")
#     else:
#         print("ℹ️ No market_regimes.csv found — continuing without regime feature.")


#     if DATE_COL not in df.columns:
#         # try to infer
#         df.reset_index(inplace=True)
#         df.rename(columns={"index": DATE_COL}, inplace=True)

#     # Parse date
#     df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
#     df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

#     # Basic check
#     if len(df) < MIN_ROWS:
#         print(f"Not enough rows ({len(df)}) - need >= {MIN_ROWS}")
#         # proceed but warn
#     print(f"Loaded {len(df)} rows from {csv_path} ({df[DATE_COL].min().date()} → {df[DATE_COL].max().date()})")

#     # Add features
#     df = add_technical_features(df)
#     df = prepare_target(df, TARGET_HORIZON_DAYS)
#     df = prepare_regression_target(df, TARGET_HORIZON_DAYS)   # 新增
#     df = classify_return_strength(df, TARGET_HORIZON_DAYS)
#     df = detect_market_regime(df)

#     # Drop rows with NaN in important features (rolling results)
#     df = df.dropna(subset=["sma_20", "vol_10", "rsi", "macd", "target_bin"]).reset_index(drop=True)

#     # Feature list: keep numeric columns except Date and target columns
#     exclude = {DATE_COL, "target_ret", "target_bin", "target_reg", "target_reg_smooth",
#            "future_ret", "target_multi", "regime"}
#     feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
#     X_all = df[feature_cols].copy()
#     y_all = df["target_bin"].copy()
#     X_all = X_all.fillna(X_all.median())

#     print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

#     # Chronological train/test split (not random): use first (1 - TEST_SIZE) for train
#     n = len(df)
#     train_n = int((1 - TEST_SIZE) * n)
#     X_train = X_all.iloc[:train_n].reset_index(drop=True)
#     y_train = y_all.iloc[:train_n].reset_index(drop=True)
#     X_test = X_all.iloc[train_n:].reset_index(drop=True)
#     y_test = y_all.iloc[train_n:].reset_index(drop=True)

#     print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

#     # Standard scaler for models that require scaling
#     scaler = StandardScaler()
#     # We'll create pipelines per-model where needed.

#     # Base model definitions (similar to ai-gold-scalper)
#     base_models = {
#         "logistic": Pipeline([
#             ("scaler", StandardScaler()),
#             ("model", LogisticRegression(max_iter=1500, C=1.0, solver="lbfgs", random_state=RND))
#         ]),
#         "rf": Pipeline([
#             ("model", RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_split=5, random_state=RND))
#         ]),
#         "gb": Pipeline([
#             ("model", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=RND))
#         ]),
#         "svc": Pipeline([
#             ("scaler", StandardScaler()),
#             ("model", SVC(C=2.0, kernel="rbf", gamma="scale", probability=True, random_state=RND))
#         ]),
#         "mlp": Pipeline([
#             ("scaler", StandardScaler()),
#             ("model", MLPClassifier(hidden_layer_sizes=(128,64), alpha=0.001, max_iter=800, random_state=RND))
#         ])
#     }
#     # Regression model to predict target_reg_smooth
#     reg_models = {}
#     if XGBOOST_AVAILABLE:
#         reg_models["xgb_reg"] = xgb.XGBRegressor(
#             n_estimators=200, learning_rate=0.05, max_depth=5,
#             subsample=0.8, colsample_bytree=0.8, random_state=RND
#         )

#     if XGBOOST_AVAILABLE:
#         base_models["xgb"] = Pipeline([
#             ("model", xgb.XGBClassifier(
#                 n_estimators=300,
#                 learning_rate=0.05,
#                 max_depth=5,
#                 subsample=0.8,
#                 colsample_bytree=0.8,
#                 eval_metric='logloss',
#                 random_state=RND
#             ))
#         ])

#     if XGBOOST_AVAILABLE:
#         base_models["xgb"] = Pipeline([("model", xgb.XGBClassifier(n_estimators=150, use_label_encoder=False, eval_metric='logloss', random_state=RND))])

#     # Train base models (fit on train set), evaluate on train (CV) and test
#     results = {}
#     trained_pipelines = {}

#     cv = StratifiedKFold(n_splits=min(CV_FOLDS, max(2, len(y_train)//50)), shuffle=True, random_state=RND)

#     for name, pipe in base_models.items():
#         print(f"\nTraining base model: {name}")
#         try:
#             # cross val score on train
#             cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
#             pipe.fit(X_train, y_train)
#             y_pred_test = pipe.predict(X_test)
#             acc = accuracy_score(y_test, y_pred_test)
#             prec = precision_score(y_test, y_pred_test, zero_division=0)
#             rec = recall_score(y_test, y_pred_test, zero_division=0)
#             f1 = f1_score(y_test, y_pred_test, zero_division=0)
#             trained_pipelines[name] = pipe
#             results[name] = {
#                 "cv_mean": float(np.mean(cv_scores)),
#                 "cv_std": float(np.std(cv_scores)),
#                 "test_accuracy": float(acc),
#                 "test_precision": float(prec),
#                 "test_recall": float(rec),
#                 "test_f1": float(f1)
#             }
#             print(f"  CV acc: {results[name]['cv_mean']:.3f} ± {results[name]['cv_std']:.3f}")
#             print(f"  Test acc: {acc:.3f}, prec: {prec:.3f}, rec: {rec:.3f}, f1: {f1:.3f}")
#         except Exception as e:
#             print(f"  Failed {name}: {e}")
#     # ====== Regression training (continuous return prediction) ======
#     if "xgb_reg" in reg_models:
#         print("\n📈 Training regression model for continuous return prediction")
#         reg_model = reg_models["xgb_reg"]
#         reg_model.fit(X_train, df.loc[:train_n-1, "target_reg_smooth"])
#         y_pred_reg = reg_model.predict(X_test)
#         corr = np.corrcoef(y_pred_reg, df.loc[train_n:, "target_reg_smooth"])[0,1]
#         print(f"📊 Regression correlation with true return: {corr:.3f}")
#         trained_pipelines["xgb_reg"] = reg_model

#     # ======== 市况分模型训练 ========
#     print("\n⚙️ Training separate models for market regimes...")
#     for regime_name, regime_df in df.groupby("regime"):
#         print(f"\n⚙️ Training regime model: {regime_name}")
#         df_r = regime_df.dropna(subset=["target_multi"])
#         if len(df_r) < 200:
#             print(f"  ⚠️ Not enough data for {regime_name} regime, skipping.")
#             continue

#         feature_cols_r = [c for c in df_r.columns if c not in ["Date", "target_multi", "future_ret", "regime"]]
#         X = df_r[feature_cols_r]
#         y = df_r["target_multi"].map({-1: 0, 0: 1, 1: 2})  # 修复 XGBoost 标签问题
#         X = X.fillna(X.median())

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)

#         # 使用 XGBoost + GB 混合示例
#         regime_models = {
#             "rf": RandomForestClassifier(n_estimators=400, max_depth=10, random_state=RND),
#             "svc": SVC(probability=True, random_state=RND),
#             "mlp": MLPClassifier(hidden_layer_sizes=(128,64), max_iter=800, random_state=RND),
#             "gb": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=RND),
#         }
#         if XGBOOST_AVAILABLE:
#             regime_models["xgb"] = xgb.XGBClassifier(
#                 n_estimators=150, learning_rate=0.05, max_depth=5, eval_metric="mlogloss", random_state=RND
#             )

#         for name, model in regime_models.items():
#             print(f"🧠 Training base model: {name}")
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#             acc = accuracy_score(y_test, y_pred)
#             f1 = f1_score(y_test, y_pred, average="macro")
#             # 简易盈利回测：涨买跌空
#             preds = np.where(y_pred == 2, 1, np.where(y_pred == 0, -1, 0))
#             profit = np.mean(preds * df_r.iloc[-len(y_pred):]["future_ret"])
#             print(f"   acc={acc:.3f} f1={f1:.3f} 💰profit={profit:.3f}")


#     # Create Voting ensemble (soft voting) with top models by cv_mean
#     sorted_by_cv = sorted(results.items(), key=lambda x: x[1]["cv_mean"], reverse=True)
#     top_models = [name for name, _ in sorted_by_cv[:5]]  # top 5
#     estimators = [(name, trained_pipelines[name]) for name in top_models if name in trained_pipelines]

#     best_models_summary = {name: results[name] for name in top_models if name in results}
#     print("\nTop models selected for ensembles:", list(best_models_summary.keys()))

#     ensembles = {}
#     # Voting
#     if len(estimators) >= 2:
#         voting = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
#         voting.fit(X_train, y_train)
#         y_vpred = voting.predict(X_test)
#         ensembles["voting"] = {
#             "model": voting,
#             "accuracy": float(accuracy_score(y_test, y_vpred)),
#             "precision": float(precision_score(y_test, y_vpred, average="macro", zero_division=0)),
#             "recall": float(recall_score(y_test, y_vpred, average="macro", zero_division=0)),
#             "f1": float(f1_score(y_test, y_vpred, average="macro", zero_division=0)),
#         }

#     # Stacking ensemble (meta logistic regression)
#     if len(estimators) >= 2:
#         # 更强的 meta 模型 (GB or XGB)
#         meta_learner = GradientBoostingClassifier(
#             n_estimators=200,
#             learning_rate=0.05,
#             max_depth=3,
#             random_state=RND
#         )
#         stacking = StackingClassifier(
#             estimators=estimators,
#             final_estimator=meta_learner,
#             n_jobs=-1,
#             passthrough=True
#         )
#         stacking.fit(X_train, y_train)
#         y_spred = stacking.predict(X_test)
#         ensembles["stacking"] = {
#             "model": stacking,
#             "accuracy": float(accuracy_score(y_test, y_spred)),
#             "precision": float(precision_score(y_test, y_spred, average="macro", zero_division=0)),
#             "recall": float(recall_score(y_test, y_spred, average="macro", zero_division=0)),
#             "f1": float(f1_score(y_test, y_spred,average="macro", zero_division=0)),
#         }
#         print("\n✅ Improved Stacking Ensemble Results:")
#         for k, v in ensembles["stacking"].items():
#             if k != "model":
#                 print(f"  {k}: {v:.3f}")
#     # ---------- Save models ----------
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     for name, model in trained_pipelines.items():
#         out_path = os.path.join(OUTPUT_DIR, f"{name}_model_{timestamp}.pkl")
#         with open(out_path, "wb") as f:
#             pickle.dump(model, f)
#     if "voting" in ensembles:
#         with open(os.path.join(OUTPUT_DIR, f"ensemble_voting_{timestamp}.pkl"), "wb") as f:
#             pickle.dump(ensembles["voting"]["model"], f)
#     if "stacking" in ensembles:
#         with open(os.path.join(OUTPUT_DIR, f"ensemble_stacking_{timestamp}.pkl"), "wb") as f:
#             pickle.dump(ensembles["stacking"]["model"], f)

#     print(f"\n✅ Models saved to: {OUTPUT_DIR}")

#     # ---------- Save report ----------
#     report_df = pd.DataFrame(results).T
#     for ens_name, ens_res in ensembles.items():
#         report_df.loc[f"ensemble_{ens_name}", :] = {
#             "cv_mean": np.nan,
#             "cv_std": np.nan,
#             "test_accuracy": ens_res["accuracy"],
#             "test_precision": ens_res["precision"],
#             "test_recall": ens_res["recall"],
#             "test_f1": ens_res["f1"],
#         }
#     report_path = os.path.join(OUTPUT_DIR, f"model_report_{timestamp}.csv")
#     report_df.to_csv(report_path)
#     print(f"📄 Report saved: {report_path}")

#     # ---------- Feature importances (if available) ----------
#     try:
#         if XGBOOST_AVAILABLE and "xgb" in trained_pipelines:
#             model = trained_pipelines["xgb"]["model"]
#             if hasattr(model, "feature_importances_"):
#                 imp_df = pd.DataFrame({
#                     "feature": feature_cols,
#                     "importance": model.feature_importances_
#                 }).sort_values("importance", ascending=False)
#                 imp_path = os.path.join(OUTPUT_DIR, f"feature_importance_{timestamp}.csv")
#                 imp_df.to_csv(imp_path, index=False)
#                 print(f"📊 Feature importances saved: {imp_path}")
#     except Exception as e:
#         print(f"⚠️ Could not save feature importances: {e}")

#     print("\n🎯 Training complete.")


# if __name__ == "__main__":
#     main()