"""Write the column inventory for the leak-reduced v4 feature tables.

Run from the repository root after src/features_v4*.py:
    python src/make_final_columns_v4.py
Reads the prior inventory as metadata, removes the two cross-split count columns,
recomputes missingness, and writes experiments/final_columns_v4.csv without
overwriting the inventory used by the earlier report.
"""
import pandas as pd


def main():
    old = pd.read_csv('experiments/final_columns.csv')
    X = pd.read_pickle('artifacts/X4_3.pkl')
    meta = old.set_index('column')
    missing = sorted(set(X.columns) - set(meta.index))
    for col in missing:
        if col.startswith('dx3_'):
            meta.loc[col, ['source_table', 'how_created', 'added_in', 'dtype', 'pct_missing']] = [
                'diagnoses', '1 if the stay has this 3-character diagnosis; vocabulary selected from training encounters',
                'v4', 'numeric', 0.0]
        elif col.startswith('dx3b_'):
            meta.loc[col, ['source_table', 'how_created', 'added_in', 'dtype', 'pct_missing']] = [
                'diagnoses', '1 if the stay has this additional 3-character diagnosis; vocabulary selected from training encounters',
                'v4', 'numeric', 0.0]
        elif col.startswith('m3_drug_'):
            meta.loc[col, ['source_table', 'how_created', 'added_in', 'dtype', 'pct_missing']] = [
                'medications', '1 if the stay has this normalized drug; vocabulary selected from training encounters',
                'v4', 'numeric', 0.0]
        else:
            raise ValueError(f'No inventory metadata for column: {col}')
    rows = []
    for col in X.columns:
        row = meta.loc[col].to_dict()
        row['pct_missing'] = round(float(X[col].isna().mean() * 100), 1)
        if col.startswith('dx3_'):
            row['how_created'] = '1 if the stay has this 3-character diagnosis; vocabulary selected from training encounters'
        elif col.startswith('dx3b_'):
            row['how_created'] = '1 if the stay has this additional 3-character diagnosis; vocabulary selected from training encounters'
        elif col.startswith('m3_drug_'):
            row['how_created'] = '1 if the stay has this normalized drug; vocabulary selected from training encounters'
        if col.startswith('pu_'):
            row['how_created'] = str(row['how_created']).replace(
                'prior visits', 'prior visits before this encounter admission')
        rows.append({'column': col, **row})
    for col, base, what in [
        ('te_hospital_id', 'hospital_id', 'hospital'),
        ('te_attending_id', 'attending_id', 'attending'),
        ('te_dx_primary3', 'dx_primary3', 'primary diagnosis'),
        ('te_proc_first', 'proc_first', 'first procedure code'),
    ]:
        rows.append({
            'column': col,
            'source_table': 'train.csv labels + ' + base,
            'how_created': f'smoothed (m=25 in best run) mean readmission rate of {what}; out-of-fold within each training fold',
            'added_in': 'v4 (model time)', 'dtype': 'numeric', 'pct_missing': 0.0,
        })
    out = pd.DataFrame(rows)
    if out.column.nunique() != len(out):
        raise ValueError('Duplicate feature names in v4 inventory')
    out.to_csv('experiments/final_columns_v4.csv', index=False, lineterminator='\n')
    print(len(X), 'rows,', len(X.columns), 'stored columns + 4 model-time encodings;',
          len(out), 'inventory rows written')


if __name__ == '__main__':
    main()
