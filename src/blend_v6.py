import numpy as np,pandas as pd,lightgbm as lgb
from scipy.stats import rankdata
import cv_v2 as c
X=pd.read_pickle('artifacts/X3.pkl'); tr=pd.read_csv('data/train.csv'); te=pd.read_csv('data/test.csv'); y=tr.readmit_30d.values
Xtr,Xte=X.loc[tr.encounter_id],X.loc[te.encounter_id]
P=dict(c.P,num_leaves=4,min_child_samples=400,feature_fraction=0.2,lambda_l2=60,learning_rate=0.04)
p3=np.zeros(len(te))
for s in range(3):
    A,(B,)=c.add_te(Xtr,y,[Xte],seed=s); A,B=c.prep(A),c.prep(B)
    p3+=lgb.train({**P,'seed':s},lgb.Dataset(A,y),625).predict(B)/3; print('seed',s,flush=True)
np.save('artifacts/test_lgb_x3.npy',p3)
v2=pd.read_csv('submissions/v2_lgbm_reg_te.csv').set_index('encounter_id').loc[te.encounter_id,'readmit_30d'].values
lr=np.load('artifacts/test_logreg_v4.npy'); r=lambda v:rankdata(v)/len(v)
bl=.4*r(v2)+.4*r(p3)+.2*r(lr)
order=rankdata(bl,method='ordinal').astype(int)-1
out=pd.DataFrame({'encounter_id':te.encounter_id,'readmit_30d':np.sort(v2)[order]})
out.to_csv('submissions/v6_blend_lgb_x3_lr.csv',index=False)
ss=pd.read_csv('data/sample_submission.csv')
print(out.shape,set(out.encounter_id)==set(ss.encounter_id),round(out.readmit_30d.mean(),4),'spearman vs v4',round(pd.Series(.8*r(v2)+.2*r(lr)).corr(pd.Series(bl),method='spearman'),4))
