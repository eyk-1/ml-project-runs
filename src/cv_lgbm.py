import pandas as pd,numpy as np,lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
X=pd.read_pickle('artifacts/X.pkl'); tr=pd.read_csv('data/train.csv')
Xt=X.loc[tr.encounter_id]; y=tr.readmit_30d.values
Xt=Xt.drop(columns=['attending_id'])  # high-card
p=dict(objective='binary',learning_rate=0.03,num_leaves=15,min_child_samples=50,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5,cat_smooth=50,min_data_per_group=100,verbose=-1,n_jobs=-1)
oof=np.zeros(len(y));imp=0
for a,b in StratifiedKFold(5,shuffle=True,random_state=0).split(Xt,y):
    m=lgb.train(p,lgb.Dataset(Xt.iloc[a],y[a]),3000,valid_sets=[lgb.Dataset(Xt.iloc[b],y[b])],callbacks=[lgb.early_stopping(100,verbose=False)])
    oof[b]=m.predict(Xt.iloc[b]);print(m.best_iteration,roc_auc_score(y[b],oof[b]),flush=True)
    imp=imp+pd.Series(m.feature_importance('gain'),Xt.columns)
print('OOF AUC',roc_auc_score(y,oof));print(imp.sort_values(ascending=False).head(30))
