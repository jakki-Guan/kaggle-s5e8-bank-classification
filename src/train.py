import argparse, os, json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

def load_data(data_dir: str):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    y = train["y"].values
    X = train.drop(columns=["y"])
    return X, y, test

def get_model(model: str, categorical_cols):
    if model == "logreg":
        ct = ColumnTransformer(
            transformers=[("ohe", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
            remainder="passthrough",
        )
        clf = LogisticRegression(max_iter=1000)
        return Pipeline([("prep", ct), ("clf", clf)])
    elif model == "lgbm":
        return lgb.LGBMClassifier(
            n_estimators=2000, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, objective="binary"
        )
    else:
        raise ValueError("Use 'logreg' or 'lgbm'.")

def main(args):
    os.makedirs("outputs", exist_ok=True)
    X, y, test = load_data(args.data_dir)
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    oof = np.zeros(len(X)); preds = np.zeros(len(test))
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_val = y[trn_idx], y[val_idx]
        model = get_model(args.model, categorical_cols)
        if args.model == "lgbm":
            X_tr = X_tr.copy(); X_val = X_val.copy()
            X_tr[categorical_cols] = X_tr[categorical_cols].astype("category")
            X_val[categorical_cols] = X_val[categorical_cols].astype("category")
            test_lgb = test.copy(); test_lgb[categorical_cols] = test_lgb[categorical_cols].astype("category")
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="auc", verbose=False)
            oof[val_idx] = model.predict_proba(X_val)[:, 1]
            preds += model.predict_proba(test_lgb)[:, 1] / args.n_splits
        else:
            model.fit(X_tr, y_tr)
            oof[val_idx] = model.predict_proba(X_val)[:, 1]
            preds += model.predict_proba(test)[:, 1] / args.n_splits
        auc = roc_auc_score(y_val, oof[val_idx]); print(f"Fold {fold} AUC: {auc:.5f}")
    cv_auc = roc_auc_score(y, oof); print(f"CV AUC: {cv_auc:.5f}")
    pd.DataFrame({"id": test.index if "id" in test.columns else range(len(test)), "y": preds}).to_csv("outputs/submission.csv", index=False)
    json.dump({"cv_auc": float(cv_auc)}, open("outputs/metrics.json", "w"))
    print("Saved outputs/submission.csv and outputs/metrics.json")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--model", type=str, default="lgbm", choices=["logreg", "lgbm"])
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--random_state", type=int, default=42)
    main(p.parse_args())
