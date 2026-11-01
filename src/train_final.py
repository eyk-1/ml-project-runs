import pandas as pd, numpy as np, lightgbm as lgb, os
X = pd.read_pickle('artifacts/X.pkl').drop(columns=['attending_id'])
tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test.csv')
y = tr.readmit_30d.values
p = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_child_samples=50, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5, cat_smooth=50, min_data_per_group=100, verbose=-1, n_jobs=-1)
pred = np.zeros(len(te))
for s in range(3):
    m = lgb.train({**p, 'seed': s}, lgb.Dataset(X.loc[tr.encounter_id], y), 290)
    pred += m.predict(X.loc[te.encounter_id]) / 3
    print('seed', s, 'done', flush=True)
sub = pd.DataFrame({'encounter_id': te.encounter_id, 'readmit_30d': pred})
os.makedirs('submissions', exist_ok=True)
sub.to_csv('submissions/latest.csv', index=False)
ss = pd.read_csv('data/sample_submission.csv')
print(sub.shape, ss.shape, set(sub.encounter_id) == set(ss.encounter_id))
print(sub.readmit_30d.describe())
