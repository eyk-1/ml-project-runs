import pandas as pd,numpy as np,time,warnings;warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
X=pd.read_pickle('artifacts/X.pkl'); tr=pd.read_csv('data/train.csv')
Xt=X.loc[tr.encounter_id].drop(columns=['attending_id']); y=tr.readmit_30d.values
cat=[c for c in Xt if str(Xt[c].dtype)=='category']; num=[c for c in Xt if c not in cat]
Xt[cat]=Xt[cat].astype(object)
def pre(): return ColumnTransformer([('n',Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]),num),
 ('c',Pipeline([('i',SimpleImputer(strategy='constant',fill_value='missing')),('o',OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=300,sparse_output=False))]),cat)])
M={'GaussianNB':GaussianNB(),'LogReg':LogisticRegression(C=.05,max_iter=300),
 'KNN':KNeighborsClassifier(75,weights='distance',n_jobs=-1),'DecisionTree':DecisionTreeClassifier(max_depth=6,min_samples_leaf=100,random_state=0),
 'RandomForest':RandomForestClassifier(300,min_samples_leaf=20,max_features=.3,n_jobs=-1,random_state=0),
 'AdaBoost':AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=150,learning_rate=.1,random_state=0)}
cv=StratifiedKFold(5,shuffle=True,random_state=0)
for n,m in M.items():
    t=time.time();p=cross_val_predict(Pipeline([('p',pre()),('m',m)]),Xt,y,cv=cv,method='predict_proba')[:,1]
    print(n,round(roc_auc_score(y,p),4),round(time.time()-t),'s',flush=True)
