import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from pathlib import Path

DATA_DIR = Path('data')
train = pd.read_csv(DATA_DIR/'train.csv'); test = pd.read_csv(DATA_DIR/'test.csv')
print("Train shape:", train.shape, " Test shape:", test.shape)
print(train['y'].value_counts(normalize=True))

# LightGBM 5-fold baseline
X = train.drop(columns=['y']); y = train['y'].values
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
oof = np.zeros(len(X))
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for i,(tr,va) in enumerate(skf.split(X,y),1):
    Xtr,Xva = X.iloc[tr].copy(), X.iloc[va].copy()
    ytr,yva = y[tr], y[va]
    Xtr[cat_cols]=Xtr[cat_cols].astype('category'); Xva[cat_cols]=Xva[cat_cols].astype('category')
    m = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
    m.fit(Xtr,ytr, eval_set=[(Xva,yva)], eval_metric='auc', verbose=False)
    oof[va]=m.predict_proba(Xva)[:,1]
    print(f"Fold {i} AUC:", roc_auc_score(yva, oof[va]))
print("CV AUC:", roc_auc_score(y, oof))
