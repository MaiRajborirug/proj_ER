# %% ---
# Data Observation — EDA for ED Revisit Dataset
# Reads:  revisit_data.csv (raw), revisit_data_signal.csv (+ VS summary),
#         vs.csv (raw vital sign records), df_process1.csv (processed)
# Writes: plots/ directory
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from const import ER_PAYMENT_MAP2

warnings.filterwarnings('ignore')

DATA_DIR = 'data/'   # all CSV/XLSX/NPY files live here
PLOT_DIR = 'plots'
os.makedirs(PLOT_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', palette='muted')
FIGSIZE_WIDE = (12, 4)
TARGET       = 'is_revisit_visit_72h'
CLS_COLORS   = {0: 'steelblue', 1: 'salmon'}
CLS_LABELS   = {0: 'Non-revisit', 1: 'Revisit 72h'}

# -----------------------------------------------------------------------
df_raw    = pd.read_csv(f'{DATA_DIR}revisit_data.csv')
df_signal = pd.read_csv(f'{DATA_DIR}revisit_data_signal.csv')
df_proc   = pd.read_csv(f'{DATA_DIR}df_process1.csv')

df_raw['register_datetime'] = pd.to_datetime(df_raw['register_datetime'], errors='coerce')
df_raw['year']              = df_raw['register_datetime'].dt.year
df_raw['payment_right']     = df_raw['สิทธิการรักษาที่ติดต่อการเงิน'].map(ER_PAYMENT_MAP2)

print(f"Raw data   : {df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")
print(f"Signal data: {df_signal.shape[0]:,} rows × {df_signal.shape[1]} cols")
print(f"Processed  : {df_proc.shape[0]:,} rows × {df_proc.shape[1]} cols")

# %% -----------------------------------------------------------------------
# 1. CLASS BALANCE  +  revisit_72h_flag vs is_revisit_visit_72h alignment
# -----------------------------------------------------------------------
class_counts = df_raw[TARGET].value_counts().sort_index()
pct          = class_counts / class_counts.sum() * 100

fig = plt.figure(figsize=(14, 5))
gs  = fig.add_gridspec(1, 3, wspace=0.35)

# --- bar (density) ---
ax0 = fig.add_subplot(gs[0])
ax0.bar([CLS_LABELS[0], CLS_LABELS[1]], pct.values,
        color=[CLS_COLORS[0], CLS_COLORS[1]], edgecolor='white')
for i, (v, p) in enumerate(zip(class_counts.values, pct.values)):
    ax0.text(i, p + 0.5, f'{p:.1f}%\n(n={v:,})', ha='center', va='bottom', fontsize=10)
ax0.set_title('Class Balance', fontsize=12)
ax0.set_ylabel('Density (%)')
ax0.set_ylim(0, pct.max() * 1.3)

# --- pie ---
ax1 = fig.add_subplot(gs[1])
ax1.pie(class_counts.values, labels=[CLS_LABELS[0], CLS_LABELS[1]],
        autopct='%1.1f%%', colors=[CLS_COLORS[0], CLS_COLORS[1]],
        startangle=90, wedgeprops=dict(edgecolor='white'))
ax1.set_title('Class Proportion', fontsize=12)

# --- alignment heatmap: revisit_72h_flag × is_revisit_visit_72h ---
ax2 = fig.add_subplot(gs[2])
ct = pd.crosstab(df_raw['revisit_72h_flag'], df_raw[TARGET])
ct.index.name   = 'revisit_72h_flag\n(index visit had revisit)'
ct.columns.name = 'is_revisit_visit_72h (this visit is a revisit)'
sns.heatmap(ct, annot=True, fmt=',d', cmap='Blues', ax=ax2,
            linewidths=0.5, cbar=False,
            annot_kws={'size': 10})
ax2.set_title('Flag Alignment\n(revisit_72h_flag vs is_revisit_visit_72h)', fontsize=10)
ax2.tick_params(axis='x', labelsize=8)
ax2.tick_params(axis='y', labelsize=8, rotation=0)

# annotation
overlap = int(ct.loc[1, 1])
ax2.text(0.5, -0.18, f'Overlap (both=1): {overlap:,} visits\n'
         f'that are revisits AND triggered another revisit',
         transform=ax2.transAxes, fontsize=7, ha='center', style='italic')

plt.suptitle('Class Balance & Label Alignment', fontsize=13, y=1.02)
plt.savefig(f'{PLOT_DIR}/01_class_balance.pdf', bbox_inches='tight')
plt.close()

print(f"\n[1] Class balance saved → {PLOT_DIR}/01_class_balance.pdf")
print(f"    Non-revisit: {class_counts[0]:,}  ({pct[0]:.1f}%)")
print(f"    Revisit 72h: {class_counts[1]:,}  ({pct[1]:.1f}%)")
print(f"    Imbalance ratio: {class_counts[0]/class_counts[1]:.1f}:1")
print(f"\n\n\n    revisit_72h_flag vs is_revisit_visit_72h crosstab:")
print(ct.to_string())

# %% -----------------------------------------------------------------------
# 2. MISSING VALUES — raw data
# -----------------------------------------------------------------------
EXCLUDE_COLS = [
    'วันที่_admit_to_ward', 'วันที่_dc_from_ward', 'ประเภทการจำหน่ายออกจากรพ',
    'icd10_principlediag_of_an', 'diagname_principlediag',
    'next_arrival_dt', 'hours_to_revisit', 'สิทธิการรักษาจากคัดกรอง'   # leakage / output cols
]
df_miss    = df_raw.drop(columns=EXCLUDE_COLS, errors='ignore')
missing_pct = (df_miss.isnull().sum() / len(df_miss) * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]

fig, ax = plt.subplots(figsize=(10, max(4, len(missing_pct) * 0.38)))
colors = ['#d73027' if v >= 50 else '#fc8d59' if v >= 20 else '#fee08b' for v in missing_pct.values]
bars   = ax.barh(missing_pct.index[::-1], missing_pct.values[::-1],
                 color=colors[::-1], edgecolor='white')
ax.set_xlabel('Missing (%)')
ax.set_title('Missing Values in Raw Data (excl. admin/output cols)', fontsize=12)
ax.axvline(50, color='#d73027', linestyle='--', linewidth=0.9, alpha=0.6, label='50%')
ax.axvline(20, color='#fc8d59', linestyle='--', linewidth=0.9, alpha=0.6, label='20%')
for bar, v in zip(bars, missing_pct.values[::-1]):
    ax.text(v + 0.5, bar.get_y() + bar.get_height() / 2,
            f'{v:.1f}%', va='center', fontsize=8)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/02_missing_values.pdf')
plt.close()
print(f"\n[2] Missing values saved → {PLOT_DIR}/02_missing_values.pdf")

# %% -----------------------------------------------------------------------
# 3. LAB MISSING RATE (by class) — with % labels on each bar
# -----------------------------------------------------------------------
lab_cols = ['creatinine', 'sodium', 'potassium', 'chloride', 'bicarb', 'hb', 'plt', 'wbc']
lab_miss = (
    df_raw.groupby(TARGET)[lab_cols]
    .apply(lambda g: g.isnull().mean() * 100)
    .T
    .rename(columns={0: 'Non-revisit', 1: 'Revisit 72h'})
)

fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
x, w = np.arange(len(lab_cols)), 0.35
b0 = ax.bar(x - w/2, lab_miss['Non-revisit'], w,
            label='Non-revisit', color='steelblue', alpha=0.85)
b1 = ax.bar(x + w/2, lab_miss['Revisit 72h'],  w,
            label='Revisit 72h',  color='salmon',    alpha=0.85)
for bars in (b0, b1):
    for bar in bars:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                f'{v:.0f}%', ha='center', va='bottom', fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(lab_cols, rotation=15)
ax.set_ylabel('Missing (%)')
ax.set_title('Lab Value Missing Rate by Class', fontsize=12)
ax.set_ylim(0, 108)
ax.legend()
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/03_lab_missing_by_class.pdf')
plt.close()
print(f"\n[3] Lab missing by class saved → {PLOT_DIR}/03_lab_missing_by_class.pdf")
print(lab_miss.round(1).to_string())

# %% -----------------------------------------------------------------------
# 4. VITAL SIGN MISSING RATE — from raw vs.csv, split by revisit_72h_flag
# -----------------------------------------------------------------------
vs = pd.read_csv(f'{DATA_DIR}vs.csv')
VS_COLS_USE = ['lbpn', 'hbpn', 'pr', 'rr', 'bt', 'o2sat', 'ps', 'e', 'm', 'v']

# Join vs readings with revisit_72h_flag from revisit_data_signal.csv
_vs = vs.dropna(subset=['hn']).copy()
_vs['hn_int'] = _vs['hn'].astype(int)
_vs['vstdate_date'] = pd.to_datetime(_vs['vstdate'], errors='coerce').dt.date

_sig_dates = df_signal[['hn', 'register_datetime', 'revisit_72h_flag']].copy()
_sig_dates['visit_date'] = pd.to_datetime(_sig_dates['register_datetime'], errors='coerce').dt.date

vs_flagged = _vs.merge(
    _sig_dates[['hn', 'visit_date', 'revisit_72h_flag']],
    left_on=['hn_int', 'vstdate_date'], right_on=['hn', 'visit_date'], how='left'
)
vs_matched = vs_flagged[vs_flagged['revisit_72h_flag'].notna()]

vs_miss = (
    vs_matched.groupby('revisit_72h_flag')[VS_COLS_USE]
    .apply(lambda g: g.isnull().mean() * 100)
    .T
    .rename(columns={0.0: 'revisit_72h_flag=0', 1.0: 'revisit_72h_flag=1'})
)

fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
x, w = np.arange(len(VS_COLS_USE)), 0.35
b0 = ax.bar(x - w/2, vs_miss['revisit_72h_flag=0'], w,
            label='revisit_72h_flag=0', color='steelblue', alpha=0.85)
b1 = ax.bar(x + w/2, vs_miss['revisit_72h_flag=1'], w,
            label='revisit_72h_flag=1', color='salmon', alpha=0.85)
for bars in (b0, b1):
    for bar in bars:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                f'{v:.0f}%', ha='center', va='bottom', fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(VS_COLS_USE, rotation=15)
ax.set_ylabel('Missing (%)')
ax.set_title('Vital Sign Missing Rate by revisit_72h_flag\n(raw vs.csv individual readings, matched to ER visits)', fontsize=11)
ax.set_ylim(0, 108)
ax.legend()
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/04_vs_missing.pdf')
plt.close()
print(f"\n[4] VS raw missing rate (by revisit_72h_flag) saved → {PLOT_DIR}/04_vs_missing.pdf")
print(f"    Source: vs.csv ({len(vs):,} rows) → matched {len(vs_matched):,} rows")
print(vs_miss.round(1).to_string())

# %% -----------------------------------------------------------------------
# 5. PROCESSED FEATURE DISTRIBUTIONS — chief complaint & register_time
#    Source: df_process1.csv
#    Layout: 2 rows (non-revisit / revisit) × 3 cols (one per feature)
# -----------------------------------------------------------------------
PROC_FEATURES = {
    'chief_complaint_total_hr':     'Chief Complaint Duration (hours)',
    'chief_complaint_urgent_level': 'Chief Complaint Urgency Level\n(0=no ref, 1=min, 2=hour, 3=day, 4=week, 5=month)',
    'register_time':                'Registration Hour\n(0–23)',
}

fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=False)

for col_i, (col, title) in enumerate(PROC_FEATURES.items()):
    for cls in [0, 1]:
        ax   = axes[cls][col_i]
        data = df_proc.loc[df_proc[TARGET] == cls, col].dropna()

        if col == 'chief_complaint_urgent_level':
            # ordinal integer — bar chart (proportion %)
            vc = data.value_counts(normalize=True).sort_index() * 100
            ax.bar(vc.index.astype(str), vc.values,
                   color=CLS_COLORS[cls], alpha=0.85, edgecolor='white')
            ax.set_xlabel('Level')
        elif col == 'register_time':
            ax.hist(data, bins=24, range=(0, 24),
                    color=CLS_COLORS[cls], alpha=0.85, edgecolor='white', density=True)
            ax.set_xlabel('Hour of day')
        elif col == 'chief_complaint_total_hr':
            # clip display to [0, 200] h; bin only within that range
            data_clip = data[(data >= 0) & (data <= 200)]
            ax.hist(data_clip, bins=40, range=(0, 200),
                    color=CLS_COLORS[cls], alpha=0.85, edgecolor='white', density=True)
            ax.set_xlim(0, 200)
            ax.set_xlabel('Hours (capped at 200)')
        else:
            upper = np.percentile(data[data > 0], 99) if (data > 0).any() else data.max()
            ax.hist(data[data <= upper], bins=40,
                    color=CLS_COLORS[cls], alpha=0.85, edgecolor='white', density=True)
            ax.set_xlabel(col, fontsize=7)

        ax.set_title(f'{title}\n[{CLS_LABELS[cls]}]', fontsize=9)
        ax.set_ylabel('Density', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.text(0.97, 0.95,
                f'n={len(data):,}\nmed={data.median():.1f}',
                transform=ax.transAxes, fontsize=7, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

plt.suptitle('Processed Feature Distributions by Class\n(source: df_process1.csv)',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/05_numerical_distributions.pdf', bbox_inches='tight')
plt.close()
print(f"\n[5] Processed feature distributions saved → {PLOT_DIR}/05_numerical_distributions.pdf")

# %% -----------------------------------------------------------------------
# 6. CATEGORICAL DISTRIBUTIONS
#    Layout: 2 rows (non-revisit / revisit) × 4 cols (one per category)
# -----------------------------------------------------------------------
CAT_FEATURES = {
    'sex':          'Sex',
    'esi':          'ESI Triage Level',
    'payment_right':'Payment Category',
    'icd10_group':  'ICD-10 Group (top 10)',
}
top10_icd = df_raw['icd10_group'].value_counts().head(10).index

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

for col_i, (col, title) in enumerate(CAT_FEATURES.items()):
    for cls in [0, 1]:
        ax   = axes[cls][col_i]
        data = df_raw.loc[df_raw[TARGET] == cls, col]

        n_cls = len(data)
        if col == 'icd10_group':
            data = data[data.isin(top10_icd)]
            vc   = (data.value_counts().reindex(top10_icd).dropna() / n_cls * 100)
            ax.barh(vc.index[::-1], vc.values[::-1],
                    color=CLS_COLORS[cls], alpha=0.85, edgecolor='white')
            ax.set_xlabel('Proportion (%)', fontsize=8)
        else:
            vc = (data.value_counts(normalize=True).sort_index() * 100)
            ax.bar(vc.index.astype(str), vc.values,
                   color=CLS_COLORS[cls], alpha=0.85, edgecolor='white')
            ax.set_xlabel(col, fontsize=8)
            if col == 'payment_right':
                ax.tick_params(axis='x', rotation=15, labelsize=7)

        ax.set_title(f'{title}\n[{CLS_LABELS[cls]}]', fontsize=9)
        ax.set_ylabel('Proportion (%)', fontsize=8)
        ax.tick_params(labelsize=7)

plt.suptitle('Categorical Feature Distributions by Class', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/06_categorical_distributions.pdf', bbox_inches='tight')
plt.close()
print(f"\n[6] Categorical distributions saved → {PLOT_DIR}/06_categorical_distributions.pdf")

# %% -----------------------------------------------------------------------
# 7. REVISIT RATE BY CATEGORY — split by revisit_72h_flag=0/1
# -----------------------------------------------------------------------
FLAG_COLORS = {0: 'steelblue', 1: 'salmon'}
FLAG_LABELS = {0: 'revisit_72h_flag=0', 1: 'revisit_72h_flag=1'}

def plot_revisit_rate_split(ax, col, title, order=None, rot=0):
    if order is not None:
        cats = [c for c in order if c in df_raw[col].values]
    else:
        cats = sorted(df_raw[col].dropna().unique())
    x, w = np.arange(len(cats)), 0.35
    y_max = 0
    for i_flag, flag in enumerate([0, 1]):
        subset = df_raw[df_raw['revisit_72h_flag'] == flag]
        rates  = (subset.groupby(col)[TARGET].mean() * 100).reindex(cats)
        counts = subset.groupby(col)[TARGET].count().reindex(cats)
        offset = (i_flag - 0.5) * w
        bars = ax.bar(x + offset, rates.values, w,
                      color=FLAG_COLORS[flag], alpha=0.85,
                      edgecolor='white', label=FLAG_LABELS[flag])
        for bar, v, n in zip(bars, rates.values, counts.values):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                        f'{v:.1f}%\n(n={int(n):,})', ha='center', va='bottom', fontsize=6)
                y_max = max(y_max, v)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in cats], rotation=rot)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel('Revisit Rate (%)')
    ax.set_ylim(0, y_max * 1.45)
    ax.legend(fontsize=7)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plot_revisit_rate_split(axes[0], 'esi',          'Revisit Rate by ESI',     order=[1,2,3,4,5])
plot_revisit_rate_split(axes[1], 'payment_right', 'Revisit Rate by Payment', rot=12)
plot_revisit_rate_split(axes[2], 'year',          'Revisit Rate by Year')

plt.suptitle('72-Hour Revisit Rate by Category (split by revisit_72h_flag)', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/07_revisit_rate_by_category.pdf', bbox_inches='tight')
plt.close()
print(f"\n[7] Revisit rate by category saved → {PLOT_DIR}/07_revisit_rate_by_category.pdf")

# %% -----------------------------------------------------------------------
# 8. VITAL SIGN DISTRIBUTIONS — first reading, violin plot by class
# -----------------------------------------------------------------------
VS_DISPLAY = ['first_lbpn', 'first_hbpn', 'first_pr',   'first_rr',
              'first_bt',   'first_o2sat', 'first_ps',   'first_emv']
VS_LABELS  = ['SBP (mmHg)', 'DBP (mmHg)', 'Pulse (bpm)', 'RR (br/min)',
              'Temp (°C)',  'O₂ Sat (%)',  'Pain Score',  'GCS Total']

df_signal_merged         = df_signal.copy()
df_signal_merged[TARGET] = df_raw[TARGET].values

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for i, (col, label) in enumerate(zip(VS_DISPLAY, VS_LABELS)):
    ax       = axes[i // 4][i % 4]
    miss_col = col + '_missing'
    data_by_cls = []
    for cls in [0, 1]:
        data = df_signal_merged.loc[
            (df_signal_merged[TARGET] == cls) &
            (df_signal_merged[miss_col] == 0), col
        ].dropna()
        upper = np.percentile(data, 99) if len(data) else data.max()
        data_by_cls.append(data[data <= upper].values)

    parts = ax.violinplot(data_by_cls, positions=[0, 1], showmedians=True, showextrema=False)
    for j, pc in enumerate(parts['bodies']):
        pc.set_facecolor(CLS_COLORS[j])
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)

    # Set y-limits based on actual data range (no forced 0)
    all_vals = np.concatenate(data_by_cls)
    lo, hi   = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
    pad      = (hi - lo) * 0.1
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([CLS_LABELS[0], CLS_LABELS[1]], fontsize=7)
    ax.set_title(label, fontsize=9)
    ax.tick_params(labelsize=7)

plt.suptitle('Vital Sign Distributions (first reading, violin, by class)', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/08_vitals_distribution.pdf', bbox_inches='tight')
plt.close()
print(f"\n[8] Vital sign distributions saved → {PLOT_DIR}/08_vitals_distribution.pdf")

# %% -----------------------------------------------------------------------
# 9. TEMPORAL: monthly visit volume & revisit rate
# -----------------------------------------------------------------------
df_raw['month'] = df_raw['register_datetime'].dt.to_period('M')
monthly = df_raw.groupby('month').agg(
    total   = (TARGET, 'count'),
    revisit = (TARGET, 'sum')
).reset_index()
monthly['rate']     = monthly['revisit'] / monthly['total'] * 100
monthly['month_dt'] = monthly['month'].dt.to_timestamp()

fig, ax1 = plt.subplots(figsize=(14, 4))
ax2 = ax1.twinx()
ax1.bar(monthly['month_dt'], monthly['total'],   width=25, color='steelblue', alpha=0.55, label='Total visits')
ax2.plot(monthly['month_dt'], monthly['rate'],   color='salmon', linewidth=1.8, marker='o', markersize=3, label='Revisit rate (%)')
ax1.set_ylabel('Total Visits', color='steelblue')
ax2.set_ylabel('72h Revisit Rate (%)', color='salmon')
ax1.set_title('Monthly Visit Volume & 72h Revisit Rate (2020–2023)', fontsize=12)
ax1.tick_params(axis='x', rotation=30)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/09_temporal_volume_revisit.pdf', bbox_inches='tight')
plt.close()
print(f"\n[9] Temporal plot saved → {PLOT_DIR}/09_temporal_volume_revisit.pdf")

# %% -----------------------------------------------------------------------
# 11. PRIOR-VISIT COUNT DISTRIBUTIONS — overlaid density by class
#     Source: df_process1.csv
#     Layout: 2 rows (non-revisit / revisit) × 5 cols (one per window)
# -----------------------------------------------------------------------
VC_COLS = [
    'visit_count_prev_3d',
    'visit_count_prev_7d',
    'visit_count_prev_15d',
    'visit_count_prev_30d_check',
    'visit_count_prev_60d',
]
VC_LABELS = ['Prev visits\n(3 days)', 'Prev visits\n(7 days)',
             'Prev visits\n(15 days)', 'Prev visits\n(30 days)',
             'Prev visits\n(60 days)']

# Determine per-column x-axis ceiling (99th pct across both classes, min 3)
x_maxes = [max(3, int(np.percentile(df_proc[c], 99))) for c in VC_COLS]

fig, axes = plt.subplots(2, 5, figsize=(18, 7))

for col_i, (col, label, xmax) in enumerate(zip(VC_COLS, VC_LABELS, x_maxes)):
    for cls in [0, 1]:
        ax   = axes[cls][col_i]
        data = df_proc.loc[df_proc[TARGET] == cls, col].dropna()
        data_clip = data[data <= xmax]

        # Integer bar-chart style histogram (one bar per integer value)
        bins = np.arange(-0.5, xmax + 1.5, 1)
        ax.hist(data_clip, bins=bins, color=CLS_COLORS[cls],
                alpha=0.85, edgecolor='white', density=True)

        zero_pct = (data == 0).mean() * 100
        ax.set_title(f'{label}\n[{CLS_LABELS[cls]}]', fontsize=8)
        ax.set_xlabel('Count', fontsize=7)
        ax.set_ylabel('Density', fontsize=7)
        ax.set_xticks(range(0, xmax + 1))
        ax.tick_params(labelsize=7)
        ax.text(0.97, 0.95,
                f'n={len(data):,}\n0-visits: {zero_pct:.0f}%\nmed={data.median():.0f}',
                transform=ax.transAxes, fontsize=6.5, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

plt.suptitle('Prior Visit Count Distributions by Window & Class\n(source: df_process1.csv)',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/11_visit_count_prev_distributions.pdf', bbox_inches='tight')
plt.close()
print(f"\n[11] Prior visit count distributions saved → {PLOT_DIR}/11_visit_count_prev_distributions.pdf")

# %% -----------------------------------------------------------------------
# SUMMARY PRINT
# -----------------------------------------------------------------------
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total visits          : {len(df_raw):,}")
print(f"Unique patients (HN)  : {df_raw['hn'].nunique():,}")
print(f"Revisit 72h (positive): {class_counts[1]:,} ({pct[1]:.2f}%)")
print(f"Non-revisit (negative): {class_counts[0]:,} ({pct[0]:.2f}%)")
print(f"Imbalance ratio       : {class_counts[0]/class_counts[1]:.1f}:1")
print(f"\nAge  : mean={df_raw['age_ofvisit'].mean():.1f}, median={df_raw['age_ofvisit'].median():.0f}, "
      f"range=[{df_raw['age_ofvisit'].min():.0f}, {df_raw['age_ofvisit'].max():.0f}]")
los_total = df_raw['los_hour'] * 60 + df_raw['los_minute'].fillna(0)
print(f"LOS  : median={los_total.median():.0f} min, 75th pct={np.percentile(los_total.dropna(), 75):.0f} min")
print(f"\nYearly volume:")
print(df_raw.groupby('year')['hn'].count().to_string())
print(f"\nAll plots saved to ./{PLOT_DIR}/")

# %%
