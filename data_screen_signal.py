# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import matplotlib.font_manager as fm
from const import ER_PAYMENT_MAP2, LAB_VALUES

warnings.filterwarnings('ignore')

DATA_DIR = 'data/'   # all CSV/XLSX/NPY files live here

df_raw = pd.read_csv(f'{DATA_DIR}revisit_data.csv')
# print(df_raw.head())
# print(df_raw.columns)


# signal = pd.read_csv('vs_2020.csv')
# print(signal.head())

# concatenate signal from data/vs_2020–2023.csv into data/vs.csv
vs_2020 = pd.read_csv(f'{DATA_DIR}vs_2020.csv')
vs_2021 = pd.read_csv(f'{DATA_DIR}vs_2021.csv')
vs_2022 = pd.read_csv(f'{DATA_DIR}vs_2022.csv')
vs_2023 = pd.read_csv(f'{DATA_DIR}vs_2023.csv')

vs = pd.concat([vs_2020, vs_2021, vs_2022, vs_2023])
vs.to_csv(f'{DATA_DIR}vs.csv', index=False)
# %%
df_raw['hn'] = df_raw['hn'].astype(int)
df_raw['register_date'] = (
    pd.to_datetime(df_raw['register_datetime'], errors='coerce')
      .dt.date
)

vs['ps'] = vs['ps'].fillna(0).clip(lower=0, upper=10)

vs = vs[vs['hn'].notna()]
vs['vst_date'] = (
    pd.to_datetime(vs['vstdate'], dayfirst=True, errors='coerce')
      .dt.date
)

# Fill up the missing value of emv with max
vs['e'] = vs['e'].fillna(4)
vs['m'] = vs['m'].fillna(6)
vs['v'] = vs['v'].fillna(5)
vs['emv'] = vs['e'] + vs['m'] + vs['v']

def get_vs_for_df_raw_row(df_raw, vs, iloc_idx):
    """
    Given a df_raw (main dataframe tabular) row
    return the corresponding selected sorted related vs (vital signs) dataframe.
    With same hn and admitted date
    """
    row = df_raw.iloc[iloc_idx]
    
    hn = row['hn']
    date = row['register_date']
    
    if pd.isna(hn) or pd.isna(date):
        return vs.iloc[0:0]  # empty DataFrame
    df_raw
    mask = (
        (vs['hn'] == hn) &
        (vs['vst_date'] == date)
    )
    
    return vs.loc[mask].sort_values(
        by=['vstdate', 'vsttime'], 
        na_position='last'
    )

print(get_vs_for_df_raw_row(df_raw, vs, iloc_idx=7))

# %%
def get_vs_summary_for_df_raw_row(df_raw, vs, iloc_idx):
    """
    Get aggregated vital signs (first,mean, last, std) for a single df_raw row.
    Returns a single-row DataFrame with summary statistics.
    """
    # Get all matching vital sign records
    vs_matches = get_vs_for_df_raw_row(df_raw, vs, iloc_idx)

    # Columns to aggregate
    vital_cols = ['lbpn', 'hbpn', 'pr', 'rr', 'bt', 'o2sat', 'ps', 'e', 'm', 'v', 'emv']

    # Initialize result dictionary
    result = {}

    # Get identifiers from first row
    result['hn'] = vs_matches.iloc[0]['hn']
    result['vstdate'] = vs_matches.iloc[0]['vstdate']
    result['vst_date'] = vs_matches.iloc[0]['vst_date']

    # Compute aggregates for each vital column
    for col in vital_cols:
        values = vs_matches[col]
        result[f'first_{col}'] = values.iloc[0]           # literal first row (NaN if not recorded)
        result[f'mean_{col}']  = values.mean(skipna=True)  # NaN only when all readings are NaN
        result[f'last_{col}']  = values.iloc[-1]           # literal last row (NaN if not recorded)
        # Missing flags (1 = value is NaN; note: imputation uses population median — do globally)
        result[f'first_{col}_missing'] = int(pd.isna(result[f'first_{col}']))
        result[f'mean_{col}_missing']  = int(pd.isna(result[f'mean_{col}']))
        result[f'last_{col}_missing']  = int(pd.isna(result[f'last_{col}']))

    return pd.DataFrame([result])

# Test the function
print("\n--- Test get_vs_summary_for_df_raw_row ---")
summary = get_vs_summary_for_df_raw_row(df_raw, vs, iloc_idx=0)
print(summary.T)  # Transpose for easier reading

# %%
# =============================================================================
# FAST VECTORIZED VERSION: Build supplementary DataFrame for ALL rows at once
# Using merge + groupby instead of row-by-row iteration
# =============================================================================

def build_vs_summary_for_all(df_raw, vs):
    """
    Build a supplementary DataFrame with vital sign summaries for ALL df_raw rows.
    Uses vectorized operations (merge + groupby) - much faster than row-by-row.

    Returns DataFrame with same number of rows as df_raw, aligned by index.
    """
    vital_cols = ['lbpn', 'hbpn', 'pr', 'rr', 'bt', 'o2sat', 'ps', 'e', 'm', 'v', 'emv']

    # Step 1: Add index to df_raw to track original row positions
    df_raw_indexed = df_raw.reset_index().rename(columns={'index': 'df_raw_idx'})

    # Step 2: Merge df_raw with vs on (hn, date)
    merged = df_raw_indexed.merge(
        vs,
        left_on=['hn', 'register_date'],
        right_on=['hn', 'vst_date'],
        how='left'
    )

    # Step 3: Sort by vstdate/vsttime within each group (for correct "last" value)
    merged = merged.sort_values(['df_raw_idx', 'vstdate', 'vsttime'], na_position='last')

    # Step 4: Group by df_raw_idx and compute aggregates
    def agg_vitals(group):
        result = {'hn': group['hn'].iloc[0]}

        for col in vital_cols:
            values = group[col].dropna()
            if len(values) == 0:
                result[f'first_{col}'] = np.nan
                result[f'mean_{col}'] = np.nan
                result[f'last_{col}'] = np.nan
            else:
                result[f'first_{col}'] = values.iloc[0]
                result[f'mean_{col}'] = values.mean()
                result[f'last_{col}'] = values.iloc[-1]

        return pd.Series(result)

    # Apply aggregation
    print("Aggregating vital signs for all rows...")
    vs_summary = merged.groupby('df_raw_idx').apply(agg_vitals)
    vs_summary = vs_summary.reset_index()

    # Step 5: Ensure alignment with original df_raw (fill missing indices)
    all_indices = pd.DataFrame({'df_raw_idx': range(len(df_raw))})
    vs_summary = all_indices.merge(vs_summary, on='df_raw_idx', how='left')

    # Drop the index column (no longer needed)
    vs_summary = vs_summary.drop(columns=['df_raw_idx'])

    print(f"Done! Shape: {vs_summary.shape}")
    return vs_summary


def build_vs_summary_fast(df_raw, vs):
    """
    Build vital sign summary for ALL df_raw rows.

    Per patient-date group computes:
      first_{col} : literal first row value  (NaN if that reading was not recorded)
      mean_{col}  : mean of non-NaN readings  (NaN only when ALL readings are missing)
      last_{col}  : literal last row value    (NaN if that reading was not recorded)

    Missing flags: {first/mean/last}_{col}_missing = 1 when value is NaN.
    Imputation   : all three stats filled with median(mean_{col}) across the population,
                   i.e. the median of per-patient means for that vital sign.
    """
    vital_cols = ['lbpn', 'hbpn', 'pr', 'rr', 'bt', 'o2sat', 'ps', 'e', 'm', 'v', 'emv']

    # Step 1: Tag df_raw rows with a positional index to preserve alignment
    df_raw_indexed = df_raw.reset_index().rename(columns={'index': 'df_raw_idx'})

    # Step 2: Left-merge so every df_raw row is represented; sort for time order
    merged = (
        df_raw_indexed[['df_raw_idx', 'hn', 'register_date']]
        .merge(
            vs[['hn', 'vst_date', 'vstdate', 'vsttime'] + vital_cols],
            left_on=['hn', 'register_date'],
            right_on=['hn', 'vst_date'],
            how='left'
        )
        .sort_values(['df_raw_idx', 'vstdate', 'vsttime'], na_position='last')
    )

    # Step 3: Per-group aggregation in a single apply call
    #   iloc[0] / iloc[-1]  → literal position, NaN-inclusive
    #   mean(skipna=True)   → NaN only when the entire column is NaN for that group
    def agg_group(g):
        out = {'hn': g['hn'].iloc[0]}
        for col in vital_cols:
            vals = g[col]
            out[f'first_{col}'] = vals.iloc[0]
            out[f'mean_{col}']  = vals.mean(skipna=True)
            out[f'last_{col}']  = vals.iloc[-1]
        return pd.Series(out)

    print("Aggregating vital signs for all rows...")
    vs_summary = merged.groupby('df_raw_idx').apply(agg_group).reset_index()

    # Step 4: Ensure every df_raw row is represented (rows with no VS data become all NaN)
    all_indices = pd.DataFrame({'df_raw_idx': range(len(df_raw))})
    vs_summary = (
        all_indices
        .merge(vs_summary, on='df_raw_idx', how='left')
        .drop(columns=['df_raw_idx'])
    )

    # Step 5: Flag missing, then impute with population median of mean_{col}
    # The same pop_median is used for first, mean, and last of the same vital sign.
    for col in vital_cols:
        pop_median = vs_summary[f'mean_{col}'].median()  # .median() skips NaN by default

        for prefix in ['first', 'mean', 'last']:
            feat_col = f'{prefix}_{col}'
            vs_summary[f'{feat_col}_missing'] = vs_summary[feat_col].isna().astype(int)
            vs_summary[feat_col] = vs_summary[feat_col].fillna(pop_median)

    # Step 6: Column order — hn, then per vital: value + flag for first, mean, last
    col_order = ['hn']
    for col in vital_cols:
        for prefix in ['first', 'mean', 'last']:
            col_order += [f'{prefix}_{col}', f'{prefix}_{col}_missing']
    vs_summary = vs_summary[col_order]

    print(f"Done! Shape: {vs_summary.shape}")
    return vs_summary


# %%
# Run the fast version
print("\n--- Building VS summary for ALL rows (fast vectorized) ---")
vs_summary_df = build_vs_summary_fast(df_raw, vs)

print("\nFirst 5 rows:")
print(vs_summary_df.head())

print("\nColumn names:")
print(vs_summary_df.columns.tolist())

print(f"\ndf_raw rows: {len(df_raw)}, vs_summary rows: {len(vs_summary_df)}")

# Save the supplementary dataframe
vs_summary_df.to_csv(f'{DATA_DIR}vs_summary.csv', index=False)
print(f"\nSaved to {DATA_DIR}vs_summary.csv")

# %%
# Verify alignment by checking a few rows
print("\n--- Verification ---")

# 1. HN alignment: every positional row should share the same HN
hn_raw  = df_raw['hn'].values
hn_summ = vs_summary_df['hn'].fillna(-1).astype(int).values
match_mask = hn_raw == hn_summ
print(f"HN alignment: {match_mask.sum()}/{len(match_mask)} rows match")
if not match_mask.all():
    bad = np.where(~match_mask)[0]
    print(f"  First mismatches at rows: {bad[:5].tolist()}")

# 2. Spot-check aggregated values against raw vs data for specific rows
for idx in [0, 7, 100]:
    row = df_raw.iloc[idx]
    hn, date = row['hn'], row['register_date']

    raw_pr = vs[(vs['hn'] == hn) & (vs['vst_date'] == date)]['pr'].dropna()
    expected_mean = raw_pr.mean() if len(raw_pr) > 0 else None  # None = no VS records

    summary_missing = bool(vs_summary_df.iloc[idx]['mean_pr_missing'])
    summary_mean    = vs_summary_df.iloc[idx]['mean_pr']

    if expected_mean is None:
        ok = summary_missing  # missing flag should be 1 when no VS records
    else:
        ok = not summary_missing and round(float(summary_mean), 4) == round(expected_mean, 4)

    print(f"\nRow {idx} — HN {hn} | {date}")
    print(f"  Raw PR readings  : {raw_pr.tolist()}")
    print(f"  Expected mean_pr : {expected_mean}")
    print(f"  Summary mean_pr  : {summary_mean}  (missing flag={int(summary_missing)})")
    print(f"  Check            : {'PASS' if ok else 'FAIL'}")

# 3. Overall missing rate
miss_pct = vs_summary_df['mean_pr_missing'].mean() * 100
print(f"\nRows with no VS data (mean_pr_missing=1): {vs_summary_df['mean_pr_missing'].sum()} ({miss_pct:.1f}%)")


# %%
# Merge df_raw with vs_summary and save
df_with_signal = pd.concat([df_raw.reset_index(drop=True), vs_summary_df.drop(columns=['hn'])], axis=1)

df_with_signal.to_csv(f'{DATA_DIR}revisit_data_signal.csv', index=False)
# df_with_signal.to_excel(f'{DATA_DIR}revisit_data_signal.xlsx', index=False)
print(f"\nSaved {DATA_DIR}revisit_data_signal.csv — shape: {df_with_signal.shape}")

# %%
# Vital sign record count per patient
vs_record_counts = vs.groupby('hn').size()
print("\n--- VS record count per patient ---")
print(f"  mean : {vs_record_counts.mean():.2f}")
print(f"  max  : {vs_record_counts.max()}")
print(f"  min  : {vs_record_counts.min()}")

# %%