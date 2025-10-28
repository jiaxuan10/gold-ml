# file: eval_model_predictions.py
import os, glob, pickle
import numpy as np, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler

CSV = "data/final/final_dataset_daily.csv"
df = pd.read_csv(CSV)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)

# prepare features and target same way as training script
# adapt if your training uses different column names/target horizon
# Here we assume target_bin exists; otherwise rebuild using target horizon used in training
if 'target_bin' not in df.columns:
    # adjust horizon if you trained with horizon=3
    H = 3
    df['target_ret'] = df['Close'].shift(-H) / df['Close'] - 1
    df['target_bin'] = (df['target_ret'] > 0).astype(int)
# drop rows with NaN target
df = df.dropna(subset=['target_bin']).reset_index(drop=True)

# create X/y as training did: drop Date & target columns and non-numeric
exclude = {'Date','target_ret','target_bin'}
feat_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
X = df[feat_cols].copy()
y = df['target_bin'].astype(int)

# chronological split (same as your training)
n = len(df)
train_n = int((1 - 0.2) * n)
X_train, X_test = X.iloc[:train_n], X.iloc[train_n:]
y_train, y_test = y.iloc[:train_n], y.iloc[train_n:]

print("Test size:", len(X_test))
print("Class distribution (test):", y_test.value_counts(normalize=True).to_dict())

# Load all saved models in models/ (pick last timestamp for each)
models_dir = "models"
pkl_files = glob.glob(os.path.join(models_dir, "*.pkl"))
print("Found saved models:", len(pkl_files))

# helper: load model by name fragment (xgb, svc, rf, voting, stacking)
def load_model_by_contains(sub):
    for p in sorted(pkl_files, reverse=True):
        fname = os.path.basename(p).lower()
        if sub in fname:
            try:
                with open(p, "rb") as f:
                    obj = pickle.load(f)
                return obj
            except Exception as e:
                print(f"⚠️ Skipped non-pickle file: {fname} ({e})")
                continue
    return None

candidates = ['xgb','svc','rf','mlp','logistic','gb','voting','stacking','ensemble']
loaded = {}
for k in candidates:
    m = load_model_by_contains(k)
    if m is not None:
        loaded[k] = m
print("Loaded models:", list(loaded.keys()))

# Evaluate each model found
results = {}
for name, model in loaded.items():
    try:
        # if pipeline or needs scaler, model.predict works as used in training
        y_pred = model.predict(X_test)
        # if model supports predict_proba:
        proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

        cr = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        roc = roc_auc_score(y_test, proba) if proba is not None else None
        # PR AUC
        if proba is not None:
            precision, recall, _ = precision_recall_curve(y_test, proba)
            pr_auc = auc(recall, precision)
        else:
            pr_auc = None

        results[name] = {
            "accuracy": cr['accuracy'],
            "precision_pos": cr['1']['precision'],
            "recall_pos": cr['1']['recall'],
            "f1_pos": cr['1']['f1-score'],
            "support_pos": cr['1']['support'],
            "confusion": cm.tolist(),
            "roc_auc": roc,
            "pr_auc": pr_auc,
            "pred_pos_frac": float(np.mean(y_pred)),
            "proba_mean": float(np.mean(proba)) if proba is not None else None,
        }
    except Exception as e:
        print("Eval failed for", name, e)

# pretty print
pd.options.display.max_colwidth = 100
res_df = pd.DataFrame(results).T
print(res_df)
res_df.to_csv("models/eval_report.csv", index=True)
print("Saved models/eval_report.csv")
