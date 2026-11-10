import pandas as pd, numpy as np
import os
D=os.environ.get('DATA_DIR','data')+'/'
def rd(n,**k): return pd.read_csv(D+n+'.csv',**k)

# ---------- base table: index encounters + patients + hospitals ----------
ie=rd('index_encounters',parse_dates=['admit_dt','discharge_dt'])
pt=rd('patients'); hp=rd('hospitals')
pt['insurance_type']=pt.insurance_type.str.title()          # MEDICARE -> Medicare
pt['race_ethnicity']=pt.race_ethnicity.replace('?',np.nan)
pt['sex']=pt.sex.replace('U',np.nan)
pt.loc[pt.birth_year<1910,'birth_year']=np.nan
for c in ['admission_type','admission_source','discharge_disposition']:
    ie[c]=ie[c].replace({'Not Available':np.nan,'Not Mapped':np.nan})
df=ie.merge(pt,on='patient_id',how='left').merge(hp,on='hospital_id',how='left')
df['los_days']=(df.discharge_dt-df.admit_dt).dt.total_seconds()/86400
df.loc[(df.los_days<=0)|(df.los_days>120),'los_days']=np.nan
df['age']=df.admit_dt.dt.year-df.birth_year
df['admit_hour']=df.admit_dt.dt.hour; df['admit_dow']=df.admit_dt.dt.dayofweek
df['admit_month']=df.admit_dt.dt.month; df['admit_year']=df.admit_dt.dt.year
df['weight_missing']=df.admit_weight_kg.isna().astype(int)
df['spec_missing']=df.attending_specialty.isna().astype(int)
df['attending_n']=df.groupby('attending_id').encounter_id.transform('count')
df['hosp_n']=df.groupby('hospital_id').encounter_id.transform('count')
df['zip3']=df.zip3.astype('float')

# ---------- diagnoses ----------
dx=rd('diagnoses').drop_duplicates(['encounter_id','seq'])
def fix(c):
    c=c.strip().upper()
    return c if '.' in c else (c[:3]+'.'+c[3:] if len(c)>3 else c)
dx['icd10_code']=dx.icd10_code.map(fix)
lk=rd('icd10_lookup'); dx=dx.merge(lk[['icd10_code','chapter','chronic_flag']],on='icd10_code',how='left')
dx['poa_n']=(dx.present_on_admission=='N').astype(int)
g=dx.groupby('encounter_id')
f_dx=pd.DataFrame({'dx_n':g.size(),'dx_chronic_n':g.chronic_flag.sum(),'dx_poa_n':g.poa_n.sum(),'dx_poa_missing':g.present_on_admission.apply(lambda s:s.isna().sum())})
prim=dx[dx.is_primary==1].drop_duplicates('encounter_id').set_index('encounter_id')
f_dx['dx_primary']=prim.icd10_code; f_dx['dx_primary_chapter']=prim.chapter
f_dx['dx_primary3']=prim.icd10_code.str[:3]
ch=pd.crosstab(dx.encounter_id,dx.chapter).add_prefix('dxch_'); f_dx=f_dx.join(ch)
d3=dx.assign(c3=dx.icd10_code.str[:3])
top=d3.c3.value_counts().head(60).index
f_dx=f_dx.join(pd.crosstab(d3[d3.c3.isin(top)].encounter_id,d3[d3.c3.isin(top)].c3).clip(upper=1).add_prefix('dx3_'))

# ---------- procedures ----------
pr=rd('procedures').drop_duplicates(); pr['code']=pr.code.astype(str)
f_pr=pr.groupby('encounter_id').agg(proc_n=('code','size'),proc_first=('code','first'))

# ---------- labs: unit harmonisation + plausibility ----------
lab=rd('lab_results').drop_duplicates()
lab['collected_dt']=pd.to_datetime(lab.collected_dt,errors='coerce')
conv={('Albumin','g/L'):.1,('Creatinine','umol/L'):1/88.4,('Glucose','mmol/L'):18.016,('Hemoglobin','g/L'):.1}
fac=[conv.get((n,u),1.) for n,u in zip(lab.lab_name,lab.unit)]
lab['v']=lab.value*np.array(fac)
rng={'Albumin':(1,6),'Bnp':(5,40000),'Bun':(2,300),'Creatinine':(.2,20),'Glucose':(30,1000),'Hba1C':(3,18),'Hemoglobin':(3,22),'Inr':(.7,15),'Lactate':(.2,30),'Potassium':(1.5,9),'Sodium':(100,180),'Troponin_I':(0,50),'Wbc':(.5,100)}
lo=lab.lab_name.map(lambda x:rng[x][0]); hi=lab.lab_name.map(lambda x:rng[x][1])
lab.loc[(lab.v<lo)|(lab.v>hi),'v']=np.nan            # unit-slip / typo outliers -> missing
lab=lab.sort_values(['encounter_id','collected_dt'])
g=lab.groupby(['encounter_id','lab_name']).v
f_lab=pd.concat({'last':g.last(),'mean':g.mean(),'min':g.min(),'max':g.max(),'first':g.first(),'cnt':g.count()},axis=1).unstack('lab_name')
f_lab.columns=[f'lab_{b}_{a}' for a,b in f_lab.columns]
for l in rng: f_lab[f'lab_{l}_delta']=f_lab[f'lab_{l}_last']-f_lab[f'lab_{l}_first']
f_lab=f_lab.drop(columns=[c for c in f_lab if c.endswith('_first')])
ab=pd.crosstab(lab.encounter_id,lab.abnormal_flag)
f_lab['lab_abn_H']=ab.get('H'); f_lab['lab_abn_L']=ab.get('L'); f_lab['lab_total']=lab.groupby('encounter_id').size()

# ---------- vitals ----------
v=rd('vitals').drop_duplicates()
v['t']=pd.to_datetime(v.charttime,format='%m/%d/%Y %H:%M',errors='coerce')
v.loc[(v.vital_type=='TEMP')&(v.value>45),'value']=(v.value-32)*5/9      # Fahrenheit -> Celsius
r={'HR':(20,250),'SBP':(50,260),'DBP':(20,160),'RR':(4,60),'SPO2':(50,100),'TEMP':(30,43)}
lo=v.vital_type.map(lambda x:r[x][0]); hi=v.vital_type.map(lambda x:r[x][1])
v.loc[(v.value<lo)|(v.value>hi),'value']=np.nan
v=v.sort_values(['encounter_id','t'])
g=v.groupby(['encounter_id','vital_type']).value
f_v=pd.concat({'last':g.last(),'first':g.first(),'mean':g.mean(),'min':g.min(),'max':g.max(),'std':g.std(),'cnt':g.count()},axis=1).unstack('vital_type')
f_v.columns=[f'vit_{b}_{a}' for a,b in f_v.columns]
for k in r: f_v[f'vit_{k}_delta']=f_v[f'vit_{k}_last']-f_v[f'vit_{k}_first']
f_v=f_v.drop(columns=[c for c in f_v if c.endswith('_first')])

# ---------- medications ----------
m=rd('medications').drop_duplicates()
m['s']=pd.to_datetime(m.start_dt.astype(str),format='%Y%m%d',errors='coerce'); m['e']=pd.to_datetime(m.stop_dt.astype('Int64').astype(str),format='%Y%m%d',errors='coerce')
g=m.groupby('encounter_id')
f_m=pd.DataFrame({'med_n':g.size(),'med_distinct':g.drug_name.nunique(),'med_cont_n':g.continued_at_discharge.sum(),'med_cont_missing':g.continued_at_discharge.apply(lambda s:s.isna().sum()),
  'med_iv_n':g.route.apply(lambda s:(s=='IV').sum()),'med_nostop_n':g.stop_dt.apply(lambda s:s.isna().sum())})
f_m=f_m.join(pd.crosstab(m.encounter_id,m.drug_class).clip(upper=1).add_prefix('med_'))

# ---------- unit transfers ----------
u=rd('unit_transfers',parse_dates=['in_dt','out_dt']).drop_duplicates()
u['h']=(u.out_dt-u.in_dt).dt.total_seconds()/3600; u.loc[(u.h<0),'h']=np.nan
f_u=u.groupby('encounter_id').agg(tr_n=('unit','size'),tr_last_unit=('unit','last'),tr_first_unit=('unit','first'))
f_u=f_u.join(u.pivot_table(index='encounter_id',columns='unit',values='h',aggfunc='sum').add_prefix('tr_hours_'))
f_u['tr_icu']=(u.assign(i=u.unit=='ICU').groupby('encounter_id').i.max()).astype(int)

# ---------- discharge planning ----------
dp=rd('discharge_planning').drop_duplicates('encounter_id').set_index('encounter_id')
dp['home_health_referral']=dp.home_health_referral.map({'Y':1,'N':0}); dp['education_done']=dp.education_done.map({'Yes':1,'No':0})
dp['dp_present']=1; dp['followup_missing']=dp.followup_scheduled_days.isna().astype(int)
dp.loc[dp.followup_scheduled_days>60,'followup_scheduled_days']=np.nan

# ---------- prior utilization (only visits before admit) ----------
pu=rd('prior_utilization',parse_dates=['visit_dt']).drop_duplicates()
pu=pu.merge(df[['patient_id','admit_dt']],on='patient_id'); pu['d']=(pu.admit_dt-pu.visit_dt).dt.days
pu=pu[pu.d>0]
rows={}
for t in ['ED','IP','OP']:
    s=pu[pu.visit_type==t]
    for w in [30,90,180,365]: rows[f'pu_{t}_{w}d']=s[s.d<=w].groupby('patient_id').size()
    rows[f'pu_{t}_all']=s.groupby('patient_id').size(); rows[f'pu_{t}_lastgap']=s.groupby('patient_id').d.min()
rows['pu_all']=pu.groupby('patient_id').size(); rows['pu_nhosp']=pu.groupby('patient_id').hospital_id.nunique()
f_pu=pd.DataFrame(rows)

# ---------- assemble ----------
X=df.set_index('encounter_id')
for f in [f_dx,f_pr,f_lab,f_v,f_m,f_u,dp]: X=X.join(f)
X=X.join(f_pu,on='patient_id')
cnt=[c for c in X if c.startswith(('pu_','dx_n','proc_n','med_n')) and 'lastgap' not in c]
for c in [c for c in X if c.startswith(('dxch_','dx3_','med_')) or c in('proc_n','pu_all')]: X[c]=X[c].fillna(0)
for c in [c for c in X if c.startswith('pu_') and 'lastgap' not in c]: X[c]=X[c].fillna(0)
X['dp_present']=X.dp_present.fillna(0)
X['pu_acute_365']=X.pu_ED_365d+X.pu_IP_365d
X['pu_acute_90']=X.pu_ED_90d+X.pu_IP_90d
X=X.drop(columns=['patient_id','admit_dt','discharge_dt','birth_year','dx_primary'],errors='ignore')
cats=[c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_datetime64_any_dtype(X[c])]
for c in cats: X[c]=X[c].astype('category')
os.makedirs('artifacts',exist_ok=True); X=X.copy(); X.to_pickle('artifacts/X.pkl')
print(X.shape, len(cats),'categoricals'); print(cats)
