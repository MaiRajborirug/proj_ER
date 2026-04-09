# %% -----------------------------------------------------------------------
# ED Revisit Prediction — Data Processing & Model Training
# Pipeline:
#   1. Feature engineering   → df_process1.csv
#   2. ML preparation        → df_model_ready.csv
#   3. Model training        → results_*.csv
#   4. XAI feature importance
# -----------------------------------------------------------------------

# ── Imports ────────────────────────────────────────────────────────────────
import re
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.calibration    import CalibratedClassifierCV
from sklearn.ensemble       import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection     import permutation_importance
from sklearn.linear_model   import LogisticRegression
from sklearn.metrics        import (accuracy_score, confusion_matrix, f1_score,
                                    recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing  import StandardScaler
from sklearn.svm            import LinearSVC, SVC

from lightgbm  import LGBMClassifier
from xgboost   import XGBClassifier
import optuna
from optuna.samplers import TPESampler

from const import ER_PAYMENT_MAP2, LAB_VALUES

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Global config ──────────────────────────────────────────────────────────
SEED            = 41
N_TOP           = 10   # features to show in XAI
DATA_TYPE       = 1    # 0: prev-visits only | 1: +admission | 2: +labs/VS | 3: +discharge
BAYESIAN_TRIALS = 10   # Optuna trials for quick RF/GB search # --> Exp use 1
THRESH_CANDIDATES = np.linspace(0.05, 0.95, 10).round(4)
DATA_DIR        = 'data/'   # all CSV/XLSX/NPY files live here


# %% -----------------------------------------------------------------------
# SECTION 1 — FEATURE ENGINEERING
# Source : data/revisit_data_signal.csv
# Output : data/df_process1.csv
# -----------------------------------------------------------------------

# ── 1a. Load raw data ──────────────────────────────────────────────────────
df_raw = pd.read_csv(f'{DATA_DIR}revisit_data_signal.csv')

df_raw['payment_right'] = df_raw['สิทธิการรักษาที่ติดต่อการเงิน'].map(ER_PAYMENT_MAP2)

df_raw['los_total_minutes'] = (
    pd.to_numeric(df_raw['los_minute'], errors='coerce')
    + 60 * pd.to_numeric(df_raw['los_hour'], errors='coerce')
)
df_raw['doctor_total_minutes'] = (
    pd.to_numeric(df_raw['doctor_minute'], errors='coerce')
    + 60 * pd.to_numeric(df_raw['doctor_hour'], errors='coerce')
)

df_raw['register_datetime']  = pd.to_datetime(df_raw['register_datetime'],  errors='coerce')
df_raw['discharge_datetime'] = pd.to_datetime(df_raw['discharge_datetime'], errors='coerce')
df_raw['register_time']      = df_raw['register_datetime'].dt.hour

# Drop rows with missing sex
df_raw = df_raw[df_raw['sex'].notna()]


# ── 1b. Register-time group ────────────────────────────────────────────────
def assign_register_group(hour):
    """Map hour-of-day → shift group (2 = night, 4 = day/evening)."""
    if pd.isna(hour):
        return None
    return 2 if 0 <= hour < 8 else 4

df_raw['register_group'] = df_raw['register_time'].apply(assign_register_group).astype(int)


# ── 1c. Doctor wait time imputation ────────────────────────────────────────
# Flag missing, clip negatives, then impute via median LOS ratio
df_raw['doctor_total_minutes_missing'] = df_raw['doctor_total_minutes'].isna().astype(int)
df_raw.loc[df_raw['doctor_total_minutes'] < 0, 'doctor_total_minutes'] = 0

median_ratio = (
    df_raw.loc[df_raw['doctor_total_minutes'].notna(), 'doctor_total_minutes']
    / df_raw.loc[df_raw['los_total_minutes'].notna(),  'los_total_minutes']
).median(skipna=True)

missing_mask = df_raw['doctor_total_minutes'].isna()
df_raw.loc[missing_mask, 'doctor_total_minutes'] = (
    df_raw.loc[missing_mask, 'los_total_minutes'] * median_ratio
)


# ── 1d. ED crowding: patients present at arrival & at doctor contact ───────
# Binary-search approach — O(n log n)
register_sorted  = np.sort(df_raw['register_datetime'].astype('int64').values)
discharge_sorted = np.sort(df_raw['discharge_datetime'].astype('int64').values)

def count_patients_at_time(t):
    """Count patients in ED at timestamp t (register <= t < discharge)."""
    t_ns       = pd.Timestamp(t).value
    arrivals   = np.searchsorted(register_sorted,  t_ns, side='right')
    departures = np.searchsorted(discharge_sorted, t_ns, side='right')
    return arrivals - departures

df_raw['num_patient_arrive']     = df_raw['register_datetime'].apply(count_patients_at_time)
# NOTE: density = count / total_at_that_moment — placeholder, currently mirrors num_patient_arrive
df_raw['density_patient_arrive'] = df_raw['num_patient_arrive']

df_raw['meet_doctor_datetime'] = (
    df_raw['register_datetime']
    + pd.to_timedelta(df_raw['doctor_total_minutes'], unit='m')
)
df_raw['num_patient_meetdoc']     = df_raw['meet_doctor_datetime'].apply(count_patients_at_time)
df_raw['density_patient_meetdoc'] = df_raw['num_patient_meetdoc']


# ── 1e. Lab value missing indicators & median imputation ──────────────────
LAB_COLS = ['sodium', 'potassium', 'chloride', 'bicarb', 'hb', 'plt', 'wbc']

for col in LAB_COLS:
    if col in df_raw.columns:
        df_raw[f'{col}_missing'] = df_raw[col].isna().astype(int)
        median_val = df_raw.loc[df_raw[col].notna(), col].median(skipna=True)
        df_raw.loc[df_raw[col].isna(), col] = median_val
        # Alternative: df_raw.loc[df_raw[col].isna(), col] = LAB_VALUES[col]

# Lab flag: 1 if any lab value was recorded
df_raw['lab'] = (
    df_raw[['creatinine', 'sodium', 'potassium', 'chloride',
            'bicarb', 'hb', 'plt', 'wbc']]
    .any(axis=1).astype(int)
)


# ── 1f. Radiology missing indicators ──────────────────────────────────────
RADIOLOGY_COLS = [
    'ct_chest', 'ecg', 'xray_chest', 'xray_abdomen',
    'xray_head', 'ultrasound_abdomen', 'ultrasound_pelvis',
]
for col in RADIOLOGY_COLS:
    if col in df_raw.columns:
        df_raw[f'{col}_missing'] = df_raw[col].isna().astype(int)


# ── 1g. Chief-complaint duration & urgency parsing ────────────────────────
TIME_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*'
    r'(นาที|ชม\.?|ชั่วโมง|วัน|สัปดาห์|อาทิตย์|เดือน|'
    r'min\.?s?|hr\.?s?|hour\.?s?|day\.?s?|wk\.?s?|week\.?s?|month\.?s?)',
    re.IGNORECASE,
)

# (hours_multiplier, urgency_level, bucket)
UNIT_MAP = {
    # Thai
    'นาที':    (1/60,  1, 'hr'),
    'ชม':      (1,     2, 'hr'),    'ชั่วโมง':  (1,     2, 'hr'),
    'วัน':     (24,    3, 'day'),
    'สัปดาห์': (24*7,  4, 'week'),  'อาทิตย์':  (24*7,  4, 'week'),
    'เดือน':   (24*30, 5, 'month'),
    # English
    'min':     (1/60,  1, 'hr'),    'mins':   (1/60,  1, 'hr'),
    'hr':      (1,     2, 'hr'),    'hrs':    (1,     2, 'hr'),
    'hour':    (1,     2, 'hr'),    'hours':  (1,     2, 'hr'),
    'day':     (24,    3, 'day'),   'days':   (24,    3, 'day'),
    'wk':      (24*7,  4, 'week'),  'wks':    (24*7,  4, 'week'),
    'week':    (24*7,  4, 'week'),  'weeks':  (24*7,  4, 'week'),
    'month':   (24*30, 5, 'month'), 'months': (24*30, 5, 'month'),
}

def parse_chief_complaint(text):
    """Extract duration & urgency features from a free-text chief complaint."""
    total_hr                          = 0.0
    min_level                         = None
    hr_flag = day_flag = week_flag = month_flag = 0

    for num_str, unit in TIME_RE.findall(str(text) if not pd.isna(text) else ''):
        info = UNIT_MAP.get(unit.lower().rstrip('.'))
        if info is None:
            continue
        hrs_mult, level, bucket = info
        total_hr  += float(num_str) * hrs_mult
        min_level  = level if min_level is None else min(min_level, level)
        if   bucket == 'hr':   hr_flag    = 1
        elif bucket == 'day':  day_flag   = 1
        elif bucket == 'week': week_flag  = 1
        else:                  month_flag = 1

    has_time = min_level is not None
    return {
        'chief_complaint_time_missing':  int(not has_time),
        'chief_complaint_total_hr':      round(total_hr, 4),
        'chief_complaint_urgent_level':  min_level if has_time else 0,
        'chief_complaint_hr_flag':       hr_flag,
        'chief_complaint_day_flag':      day_flag,
        'chief_complaint_week_flag':     week_flag,
        'chief_complaint_month_flag':    month_flag,
    }

cc_df  = pd.DataFrame(
    df_raw['chief_complain'].apply(parse_chief_complaint).tolist(),
    index=df_raw.index,
)
df_raw = pd.concat([df_raw, cc_df], axis=1)

print("Chief-complaint features added.")
print(cc_df[['chief_complaint_time_missing', 'chief_complaint_urgent_level']]
      .value_counts().sort_index())
print(f"NaN check: {cc_df.isna().sum().sum()} NaN values")

CC_TEXT_COLS = []


# ── 1h. Chief-complaint text features (three options — activate one) ───────
#
#   1h-A  Rule-based symptom flags  [ACTIVE]  — binary flags per symptom group
#   1h-B  TF-IDF + character n-gram + SVD      — statistical text features
#   1h-C  WangchanBERTa embeddings + PCA       — deep Thai-language features
#
# Only one option should be un-commented at a time.
# CC_TEXT_COLS is consumed by numerical_cols_1 in Section 2.
# ─────────────────────────────────────────────────────────────────────────────

# # ── 1h-A: Rule-based symptom flags (ACTIVE) ──────────────────────────────
# _CLINICAL_GROUPS = {
#     'cc_respiratory': ['ไอ', 'เหนื่อย', 'เสมหะ', 'น้ำมูก', 'เจ็บคอ', 'หายใจ'],
#     'cc_gi':          ['ปวดท้อง', 'ถ่ายเหลว', 'อาเจียน', 'ท้องเสีย', 'คลื่นไส้',
#                        'ปวดแสบท้อง', 'ท้องอืด', 'แน่นท้อง'],
#     'cc_neuro':       ['เวียนศีรษะ', 'ปวดศีรษะ', 'บ้านหมุน', 'อ่อนแรง',
#                        'อ่อนเพลีย', 'วูบ'],
#     'cc_numbness':    ['ชา'],
#     'cc_cardiac':     ['ใจสั่น', 'แน่นหน้าอก', 'เจ็บหน้าอก', 'เจ็บอก'],
#     'cc_urinary':     ['ปัสสาวะ'],
#     'cc_msk':         ['ปวดหลัง', 'ปวดขา', 'ปวดแขน', 'เจ็บเท้า', 'ปวดข้อ',
#                        'ปวดกล้าม', 'ปวดคอ', 'ปวดไหล่'],
#     'cc_derm':        ['ผื่น', 'คัน', 'ตุ่ม', 'แผล'],
#     'cc_eye_ent':     ['ตาแดง', 'คันตา', 'หูอื้อ', 'ปวดหู'],
#     'cc_fever':       ['ไข้'],
#     'cc_trauma':      ['trauma', 'บาดเจ็บ', 'ล้ม', 'อุบัติเหตุ', 'ชน',
#                        'กระแทก', 'หกล้ม'],
# }

# _META_FLAGS = {
#     'cc_atk_pos':      ['atk+', 'atk +', 'atk positive'],
#     'cc_atk_neg':      ['atk neg', 'atk-', 'atk negative'],
#     'cc_covid':        ['[cp]', 'covid', 'โควิด', 'pcr+'],
#     'cc_ems':          ['ems'],
#     'cc_referred':     ['นำส่ง'],
#     'cc_severity_inc': ['มากขึ้น', 'กระสับกระส่าย', 'รุนแรง'],
#     'cc_followup':     ['ตามนัด', 'follow up', 'f/u'],
#     'cc_substance':    ['speeda', 'ยาบ้า', 'สารเสพ', 'เสพติด'],
#     'cc_injection':    ['ฉีด', 'เข็ม', 'วัคซีน'],
#     'cc_device':       ['หลุด', 'สายหลุด', 'ng ออก', 'foley'],
# }

# def extract_symptom_flags(text):
#     """Return binary flag dict for each clinical group and meta category."""
#     t = str(text).lower() if not pd.isna(text) else ''
#     flags = {col: int(any(kw in t for kw in kws))
#              for col, kws in {**_CLINICAL_GROUPS, **_META_FLAGS}.items()}
#     flags['cc_num_symptom_groups'] = sum(flags[c] for c in _CLINICAL_GROUPS)
#     return flags

# cc_text_df = pd.DataFrame(
#     df_raw['chief_complain'].apply(extract_symptom_flags).tolist(),
#     index=df_raw.index,
# )
# df_raw     = pd.concat([df_raw, cc_text_df], axis=1)
# CC_TEXT_COLS = list(cc_text_df.columns)

# flag_cols        = [c for c in CC_TEXT_COLS if c != 'cc_num_symptom_groups']
# zero_flag_mask   = cc_text_df[flag_cols].sum(axis=1) == 0
# print(f"\n1h-A symptom flags added: {len(CC_TEXT_COLS)} columns")
# print(f"Visits with zero symptom flags: {zero_flag_mask.mean():.1%}")
# print(cc_text_df[flag_cols].mean().sort_values().round(4).to_string())


# # ── 1h-B: TF-IDF + character n-gram + SVD (COMMENTED — swap with 1h-A to use)
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.pipeline import Pipeline

# N_SVD_COMPONENTS = 15
# cc_texts = df_raw['chief_complain'].fillna('').tolist()
# tfidf_svd = Pipeline([
#     ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4),
#                               max_features=5000, sublinear_tf=True)),
#     ('svd',   TruncatedSVD(n_components=N_SVD_COMPONENTS, random_state=SEED)),
# ])
# cc_matrix   = tfidf_svd.fit_transform(cc_texts)
# cc_text_df  = pd.DataFrame(
#     cc_matrix,
#     columns=[f'cc_svd_{i}' for i in range(N_SVD_COMPONENTS)],
#     index=df_raw.index,
# )
# df_raw      = pd.concat([df_raw, cc_text_df], axis=1)
# CC_TEXT_COLS = list(cc_text_df.columns)
# print(f"\n1h-B TF-IDF+SVD features added: {len(CC_TEXT_COLS)} columns")


# # ── 1h-C: WangchanBERTa embeddings + PCA (COMMENTED — swap with 1h-A to use)
# # Speed-ups vs. naive row-by-row apply():
# #   1. Embedding cache    — skips BERT entirely on subsequent runs   (~10 min saved)
# #   2. Batched inference  — ~64 rows per forward pass instead of 1   (~50x faster)
# #   3. MPS device         — Apple Silicon GPU acceleration            (~4x faster)
# #   4. torch.inference_mode() — disables autograd tracking           (small gain)

# from transformers.models.camembert.tokenization_camembert import CamembertTokenizer
# from transformers import AutoModel
# from sklearn.decomposition import PCA
# import torch

# BERT_MODEL_NAME  = 'airesearch/wangchanberta-base-att-spm-uncased'
# BERT_BATCH_SIZE  = 64    # lower to 32 if OOM on MPS
# BERT_MAX_LENGTH  = 64    # chief complaints are short; 64 tokens is sufficient
# N_PCA_COMPONENTS = 15
# _BERT_CACHE      = f'{DATA_DIR}bert_embeddings_cache.npy'

# # ── Load cache or run BERT inference ─────────────────────────────────────
# _cache_ok = False
# if os.path.exists(_BERT_CACHE):
#     _embeddings = np.load(_BERT_CACHE)
#     if _embeddings.shape[0] == len(df_raw):
#         print(f"1h-C: loaded cached embeddings from '{_BERT_CACHE}'  {_embeddings.shape}")
#         _cache_ok = True
#     else:
#         print(f"1h-C: cache row count mismatch "
#               f"({_embeddings.shape[0]} cached vs {len(df_raw)} current) — recomputing …")

# if not _cache_ok:
#     # ── Device: Apple Silicon MPS (primary) ──────────────────────────────
#     if torch.backends.mps.is_available():
#         _BERT_DEVICE = torch.device('mps')
#         print("1h-C: using MPS device (Apple Silicon)")
#     # elif torch.cuda.is_available():      # ← uncomment for NVIDIA machines
#     #     _BERT_DEVICE = torch.device('cuda')
#     #     print(f"1h-C: using CUDA — {torch.cuda.get_device_name(0)}")
#     else:
#         _BERT_DEVICE = torch.device('cpu')
#         print("1h-C: MPS not available, using CPU")

#     _tokenizer  = CamembertTokenizer.from_pretrained(BERT_MODEL_NAME)
#     _bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME).to(_BERT_DEVICE)
#     _bert_model.eval()

#     def _embed_batch(batch_texts):
#         enc = _tokenizer(
#             batch_texts,
#             return_tensors='pt',
#             truncation=True,
#             max_length=BERT_MAX_LENGTH,
#             padding=True,
#         )
#         enc = {k: v.to(_BERT_DEVICE) for k, v in enc.items()}
#         with torch.inference_mode():
#             out = _bert_model(**enc)
#         return out.last_hidden_state[:, 0, :].cpu().numpy()   # CLS token

#     _texts = df_raw['chief_complain'].fillna('').tolist()
#     print(f"1h-C: extracting embeddings — {len(_texts):,} rows, "
#           f"batch={BERT_BATCH_SIZE}, device={_BERT_DEVICE} …")

#     _all_emb = []
#     for _i in range(0, len(_texts), BERT_BATCH_SIZE):
#         if _i % (BERT_BATCH_SIZE * 20) == 0:
#             print(f"  {_i:>7,} / {len(_texts):,}", flush=True)
#         _all_emb.append(_embed_batch(_texts[_i : _i + BERT_BATCH_SIZE]))

#     _embeddings = np.vstack(_all_emb)
#     np.save(_BERT_CACHE, _embeddings)
#     print(f"1h-C: embeddings shape = {_embeddings.shape}  — saved to '{_BERT_CACHE}'")

# # PCA fit on TRAIN rows only — avoids leaking val/test structure into components
# _train_years = pd.to_datetime(df_raw['register_datetime'], errors='coerce').dt.year
# _train_mask  = _train_years.isin([2020, 2021]).values
# pca          = PCA(n_components=N_PCA_COMPONENTS, random_state=SEED)
# pca.fit(_embeddings[_train_mask])
# _cc_matrix   = pca.transform(_embeddings)
# print(f"1h-C: PCA cumulative variance = "
#       f"{pca.explained_variance_ratio_.cumsum()[-1]:.3f}")

# cc_text_df   = pd.DataFrame(
#     _cc_matrix,
#     columns=[f'cc_bert_{i}' for i in range(N_PCA_COMPONENTS)],
#     index=df_raw.index,
# )
# df_raw       = pd.concat([df_raw, cc_text_df], axis=1)
# CC_TEXT_COLS = list(cc_text_df.columns)
# print(f"1h-C: {len(CC_TEXT_COLS)} PCA columns added")


# ── 1i. Prior-visit counts (sliding window) ───────────────────────────────
def compute_visit_count_prev_n(df, n_days,
                                arrival_col='register_datetime',
                                patient_col='hn'):
    """Count prior visits by same patient within last n_days (exclusive of current)."""
    cutoff = np.timedelta64(n_days, 'D')
    tmp = pd.DataFrame({
        'hn':       df[patient_col].values,
        'arrival':  pd.to_datetime(df[arrival_col], errors='coerce').values,
        'orig_idx': np.arange(len(df)),
    }).sort_values(['hn', 'arrival'])

    result = np.zeros(len(df), dtype=int)
    for _, grp in tmp.groupby('hn', sort=False):
        times = grp['arrival'].values
        idxs  = grp['orig_idx'].values
        left  = 0
        for i in range(len(times)):
            while (times[i] - times[left]) > cutoff:
                left += 1
            result[idxs[i]] = i - left
    return result

df_raw['visit_count_prev_3d']        = compute_visit_count_prev_n(df_raw,  3)
df_raw['visit_count_prev_7d']        = compute_visit_count_prev_n(df_raw,  7)
df_raw['visit_count_prev_15d']       = compute_visit_count_prev_n(df_raw, 15)
df_raw['visit_count_prev_30d_check'] = compute_visit_count_prev_n(df_raw, 30)
df_raw['visit_count_prev_60d']       = compute_visit_count_prev_n(df_raw, 60)

# Zero-out counts before 2020-03-01 (insufficient history in the dataset)
VISIT_COUNT_COLS = [
    'visit_count_prev_3d', 'visit_count_prev_7d', 'visit_count_prev_15d',
    'visit_count_prev_30d_check', 'visit_count_prev_60d',
]
HISTORY_CUTOFF = pd.Timestamp('2020-03-01')
early_mask = pd.to_datetime(df_raw['register_datetime'], errors='coerce') < HISTORY_CUTOFF
df_raw.loc[early_mask, VISIT_COUNT_COLS] = 0

print("\nPrior-visit count columns added:")
for col in VISIT_COUNT_COLS:
    print(f"  {col}: mean={df_raw[col].mean():.3f}, max={df_raw[col].max()}")
print(f"  Zeroed {early_mask.sum()} rows before {HISTORY_CUTOFF.date()}")


# ── 1j. Save processed data ────────────────────────────────────────────────
df_raw.to_csv(f'{DATA_DIR}df_process1.csv', index=False)
print(f"\nSaved → {DATA_DIR}df_process1.csv")


# %% -----------------------------------------------------------------------
# SECTION 2 — ML DATA PREPARATION
# Source : data/df_process1.csv  (already in memory — no reload needed)
# Output : data/df_model_ready.csv
# -----------------------------------------------------------------------

# delete df_raw valriable if exists and reload df_raw from df_process1.csv
if 'df_raw' in globals():
    del df_raw
df_raw = pd.read_csv(f'{DATA_DIR}df_process1.csv')

TARGET_COL         = 'is_revisit_visit_72h'
ORDINAL_COLS       = ['esi']
CATEG_COLS         = ['sex', 'payment_right']          # available at admission
CATEG_COLS_DISCH   = ['icd10_group']                   # only available post-discharge (DATA_TYPE=3)

# ── Feature column sets (cumulative by DATA_TYPE) ──────────────────────────
# numerical_cols_0 = [
#     'visit_count_prev_3d', 'visit_count_prev_7d',
#     'visit_count_prev_15d', 'visit_count_prev_30d_check',
# ]
numerical_cols_0 = [
    'visit_count_prev_3d', 'visit_count_prev_7d',
    'visit_count_prev_15d', 
]

numerical_cols_1 = [
    'age_ofvisit', 'register_time', 'register_group',
    'num_patient_arrive', 'density_patient_arrive',
    'chief_complaint_time_missing', 'chief_complaint_total_hr',
    'chief_complaint_urgent_level', 
    # 'chief_complaint_hr_flag',
    # 'chief_complaint_day_flag', 
    # 'chief_complaint_week_flag',
    # 'chief_complaint_month_flag',
] + CC_TEXT_COLS  # set by whichever 1h-X option is active

numerical_cols_2 = [
    'sodium', 'potassium', 'chloride', 'bicarb', 'hb', 'plt', 'wbc',
    'doctor_total_minutes', 'doctor_total_minutes_missing',
    'sodium_missing', 'potassium_missing', 'chloride_missing',
    'bicarb_missing', 'hb_missing', 'plt_missing', 'wbc_missing', 'lab',
    'num_patient_meetdoc', 'density_patient_meetdoc',
    'first_lbpn',  'first_lbpn_missing',
    'first_hbpn',  'first_hbpn_missing',
    'first_pr',    'first_pr_missing',
    'first_rr',    'first_rr_missing',
    'first_bt',    'first_bt_missing',
    'first_o2sat', 'first_o2sat_missing',
    'first_ps',    'first_ps_missing',
    'first_e',     'first_e_missing',
    'first_m',     'first_m_missing',
    'first_v',     'first_v_missing',
    'first_emv',   'first_emv_missing',
]

numerical_cols_3 = [
    'los_total_minutes',
    'mean_lbpn',  'mean_lbpn_missing',  'last_lbpn',  'last_lbpn_missing',
    'mean_hbpn',  'mean_hbpn_missing',  'last_hbpn',  'last_hbpn_missing',
    'mean_pr',    'mean_pr_missing',    'last_pr',    'last_pr_missing',
    'mean_rr',    'mean_rr_missing',    'last_rr',    'last_rr_missing',
    'mean_bt',    'mean_bt_missing',    'last_bt',    'last_bt_missing',
    'mean_o2sat', 'mean_o2sat_missing', 'last_o2sat', 'last_o2sat_missing',
    'mean_ps',    'mean_ps_missing',    'last_ps',    'last_ps_missing',
    'mean_e',     'mean_e_missing',     'last_e',     'last_e_missing',
    'mean_m',     'mean_m_missing',     'last_m',     'last_m_missing',
    'mean_v',     'mean_v_missing',     'last_v',     'last_v_missing',
    'mean_emv',   'mean_emv_missing',   'last_emv',   'last_emv_missing',
]

# DATA_TYPE controls which feature groups are included:
#   0 = prior visits only
#   1 = + admission info (age, register time, chief complaint, crowding)
#   2 = + labs & first vital signs                              [default]
#   3 = + discharge-time vitals (mean/last) + icd10_group (post-discharge)
if   DATA_TYPE == 0: numerical_cols = numerical_cols_0
elif DATA_TYPE == 1: numerical_cols = numerical_cols_0 + numerical_cols_1
elif DATA_TYPE == 2: numerical_cols = numerical_cols_0 + numerical_cols_1 + numerical_cols_2
elif DATA_TYPE == 3: numerical_cols = numerical_cols_0 + numerical_cols_1 + numerical_cols_2 + numerical_cols_3
else: raise ValueError(f"DATA_TYPE must be 0–3, got {DATA_TYPE}")

# One-hot encode categorical columns
# icd10_group is post-discharge info — only included at DATA_TYPE=3
active_categ_cols = CATEG_COLS + (CATEG_COLS_DISCH if DATA_TYPE == 3 else [])
df_encoded        = pd.get_dummies(df_raw, columns=active_categ_cols, drop_first=False)
encoded_cat_cols  = [c for c in df_encoded.columns
                     if any(c.startswith(cat + '_') for cat in active_categ_cols)]

feature_cols = (numerical_cols_0 if DATA_TYPE == 0
                else numerical_cols + ORDINAL_COLS + encoded_cat_cols)

# Build model-ready dataframe (keep 'hn' for patient-level splitting)
df_model = df_encoded[['hn', 'register_datetime'] + feature_cols + [TARGET_COL]].copy()

missing_check = df_model[feature_cols].isnull().sum()
if missing_check.any():
    print("Missing values detected in features:")
    print(missing_check[missing_check > 0])

df_model.to_csv(f'{DATA_DIR}df_model_ready.csv', index=False)
print(f"Saved → {DATA_DIR}df_model_ready.csv  {df_model.shape}")


# %% -----------------------------------------------------------------------
# SECTION 3 — MODEL TRAINING
# Source : data/df_model_ready.csv  (already in memory — no reload needed)
# -----------------------------------------------------------------------

# ── 3a. Temporal + HN split ────────────────────────────────────────────────
# Train : 2020–2021  |  Val : 2022  |  Test : 2023
# Splitting by time mirrors real deployment (model trained on past, evaluated
# on future) and avoids data leakage from future visits informing past labels.
# NOTE: the same HN may appear in multiple splits — a patient who visited in
#       2021 and again in 2023 is in both train and test. This is intentional
#       and realistic; we do NOT assert zero HN overlap here.
years = pd.to_datetime(df_model['register_datetime'], errors='coerce').dt.year.values

train_mask = np.isin(years, [2020, 2021])
val_mask   = years == 2022
test_mask  = years == 2023

X = df_model[feature_cols].values
y = df_model[TARGET_COL].values

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

# Patient counts & HN cross-split overlap (informational)
train_hn = set(df_model.loc[train_mask, 'hn'])
val_hn   = set(df_model.loc[val_mask,   'hn'])
test_hn  = set(df_model.loc[test_mask,  'hn'])

print(f"Temporal split summary:")
print(f"  Train (2020–2021) : {train_mask.sum():>6,} visits | {len(train_hn):>5,} unique HNs")
print(f"  Val   (2022)      : {val_mask.sum():>6,} visits | {len(val_hn):>5,} unique HNs")
print(f"  Test  (2023)      : {test_mask.sum():>6,} visits | {len(test_hn):>5,} unique HNs")
print(f"  HN overlap train∩val  : {len(train_hn & val_hn):,}")
print(f"  HN overlap train∩test : {len(train_hn & test_hn):,}")
print(f"  HN overlap val∩test   : {len(val_hn & test_hn):,}")
print(f"  Class 1 rate — Train: {y_train.mean():.3f} | Val: {y_val.mean():.3f} | Test: {y_test.mean():.3f}")

# ── 3b. Feature scaling ────────────────────────────────────────────────────
# Full training set used (no downsampling)
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)


# ── 3c. Models (classical ML + neural network) ────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(random_state=SEED, max_iter=1000),
    'Random Forest':        RandomForestClassifier(random_state=SEED, n_estimators=100),
    'Gradient Boosting':    GradientBoostingClassifier(random_state=SEED, n_estimators=100),
    'LightGBM':             LGBMClassifier(random_state=SEED, n_estimators=100, verbose=-1),
    'MLP':                  MLPClassifier(
                                hidden_layer_sizes  = (64, 32),
                                activation          = 'relu',
                                solver              = 'adam',
                                alpha               = 1e-4,
                                batch_size          = 256,
                                learning_rate       = 'adaptive',
                                max_iter            = 200,
                                early_stopping      = True,
                                validation_fraction = 0.1,
                                n_iter_no_change    = 15,
                                random_state        = SEED,
                                verbose             = False,
                            ),
    # 'XGBoost': XGBClassifier(random_state=SEED, n_estimators=100, eval_metric='logloss'),
    # 'KNN':     KNeighborsClassifier(n_neighbors=5),
}


# ── 3d. Evaluation helper ──────────────────────────────────────────────────
def evaluate_model(y_true, y_pred, y_prob):
    """Return dict of classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'Accuracy':    accuracy_score(y_true, y_pred),
        'Sensitivity': recall_score(y_true, y_pred),          # TP / (TP + FN)
        'Specificity': tn / (tn + fp),                        # TN / (TN + FP)
        'PPV':         tp / (tp + fp) if (tp + fp) > 0 else 0.0,  # precision
        'NPV':         tn / (tn + fn) if (tn + fn) > 0 else 0.0,  # neg. pred. value
        'F1':          f1_score(y_true, y_pred),
        'AUC':         roc_auc_score(y_true, y_prob),
    }


# ── 3e. Train & evaluate all baseline models ──────────────────────────────
trained_models = {}
results_train, results_val = [], []
COL_ORDER = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1', 'AUC']

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model

    y_tr_pred = model.predict(X_train_scaled)
    y_tr_prob = model.predict_proba(X_train_scaled)[:, 1]
    m_train   = evaluate_model(y_train, y_tr_pred, y_tr_prob)
    m_train['Model'] = name
    results_train.append(m_train)

    y_v_pred = model.predict(X_val_scaled)
    y_v_prob = model.predict_proba(X_val_scaled)[:, 1]
    m_val    = evaluate_model(y_val, y_v_pred, y_v_prob)
    m_val['Model'] = name
    results_val.append(m_val)

df_results_train = pd.DataFrame(results_train).set_index('Model')[COL_ORDER]
df_results_val   = pd.DataFrame(results_val).set_index('Model')[COL_ORDER]

print("\n" + "=" * 80)
print("BASELINE — TRAIN")
print("=" * 80)
print(df_results_train.round(4).to_string())

print("\n" + "=" * 80)
print("BASELINE — VALIDATION")
print("=" * 80)
print(df_results_val.round(4).to_string())

df_results_train.to_csv(f'{DATA_DIR}results_train.csv')
df_results_val.to_csv(f'{DATA_DIR}results_test.csv')
print(f"\nSaved → {DATA_DIR}results_train.csv, {DATA_DIR}results_test.csv")


# %% -----------------------------------------------------------------------
# SECTION 4 — HYPERPARAMETER TUNING (Optuna / TPE)
# Strategy: quick Bayesian search on RF & GB, then threshold sweep on all
# -----------------------------------------------------------------------

# ── 4a. Search-space definitions ──────────────────────────────────────────
def make_rf_trial(trial):
    return RandomForestClassifier(
        n_estimators     = trial.suggest_int('n_estimators',   50, 300),
        max_depth        = trial.suggest_int('max_depth',       3,  20),
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20),
        max_features     = trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5]),
        random_state     = SEED,
    )

def make_gb_trial(trial):
    return GradientBoostingClassifier(
        n_estimators  = trial.suggest_int(  'n_estimators',  50, 300),
        max_depth     = trial.suggest_int(  'max_depth',      2,   8),
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        subsample     = trial.suggest_float('subsample',     0.5,  1.0),
        random_state  = SEED,
    )

TRIAL_BUILDERS  = {'Random Forest': make_rf_trial, 'Gradient Boosting': make_gb_trial}
BAYESIAN_MODELS = set(TRIAL_BUILDERS.keys())


# ── 4b. Bayesian search — RF & GB ─────────────────────────────────────────
print("\n" + "=" * 80)
print(f"BAYESIAN HP SEARCH — {', '.join(BAYESIAN_MODELS)}  ({BAYESIAN_TRIALS} trials each)")
print("=" * 80)

for name in BAYESIAN_MODELS:
    def objective(trial, _name=name):
        m    = TRIAL_BUILDERS[_name](trial)
        m.fit(X_train_scaled, y_train)
        prob = m.predict_proba(X_val_scaled)[:, 1]
        return max(
            f1_score(y_val, (prob >= t).astype(int), zero_division=0)
            for t in THRESH_CANDIDATES
        )

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=BAYESIAN_TRIALS, show_progress_bar=False)

    print(f"\n{name}:")
    print(f"  Best val F1 : {study.best_value:.4f}")
    print(f"  Best params : {study.best_params}")

    best_model = TRIAL_BUILDERS[name](study.best_trial)
    best_model.fit(X_train_scaled, y_train)
    trained_models[name] = best_model
    models[name]         = best_model


# ── 4c. Threshold sweep — all models ──────────────────────────────────────
print("\n" + "=" * 80)
print(f"THRESHOLD SWEEP — all models  (candidates: {THRESH_CANDIDATES.tolist()})")
print("=" * 80)

best_thresholds           = {}
results_train_thresh      = []
results_val_thresh        = []
results_test_thresh       = []

for name in models:
    y_tr_prob  = trained_models[name].predict_proba(X_train_scaled)[:, 1]
    y_v_prob   = trained_models[name].predict_proba(X_val_scaled)[:, 1]
    y_te_prob  = trained_models[name].predict_proba(X_test_scaled)[:, 1]

    best_f1, best_thresh = 0.0, 0.5
    for thresh in THRESH_CANDIDATES:
        f1 = f1_score(y_val, (y_v_prob >= thresh).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(thresh)

    best_thresholds[name] = best_thresh
    print(f"\n{name}:  best threshold = {best_thresh:.4f}  (val F1 = {best_f1:.4f})")

    for y_true, y_prob, store in [
        (y_train, y_tr_prob, results_train_thresh),
        (y_val,            y_v_prob,  results_val_thresh),
        (y_test,           y_te_prob, results_test_thresh),
    ]:
        m = evaluate_model(y_true, (y_prob >= best_thresh).astype(int), y_prob)
        m['Model']     = name
        m['Threshold'] = best_thresh
        store.append(m)

THRESH_COL_ORDER  = ['Threshold', 'Accuracy', 'Sensitivity', 'Specificity',
                     'PPV', 'NPV', 'F1', 'AUC']
THRESH_SHORT_COLS = ['Thresh', 'Acc', 'Sen', 'Spec', 'PPV', 'NPV', 'F1', 'AUC']

def make_thresh_df(rows):
    df = pd.DataFrame(rows).set_index('Model')[THRESH_COL_ORDER]
    df.columns = THRESH_SHORT_COLS
    return df

df_thresh_train = make_thresh_df(results_train_thresh)
df_thresh_val   = make_thresh_df(results_val_thresh)
df_thresh_test  = make_thresh_df(results_test_thresh)

for label, df in [('TRAIN', df_thresh_train),
                  ('VAL',   df_thresh_val),
                  ('TEST',  df_thresh_test)]:
    print(f"\n{'=' * 80}\n{label} — optimised threshold\n{'=' * 80}")
    print(df.round(4).to_string())

df_thresh_val.to_csv(f'{DATA_DIR}results_eval_best_threshold.csv')
df_thresh_test.to_csv(f'{DATA_DIR}results_test_best_threshold.csv')
print(f"\nSaved → {DATA_DIR}results_eval_best_threshold.csv, {DATA_DIR}results_test_best_threshold.csv")


# %% -----------------------------------------------------------------------
# SECTION 5 — XAI: FEATURE IMPORTANCE
# -----------------------------------------------------------------------
print("\n" + "=" * 80)
print("XAI — FEATURE IMPORTANCE")
print("=" * 80)

# Logistic Regression: signed coefficients
lr_model      = trained_models['Logistic Regression']
lr_thresh     = best_thresholds['Logistic Regression']
lr_importance = (
    pd.DataFrame({'Feature': feature_cols, 'Coefficient': lr_model.coef_[0]})
    .sort_values('Coefficient', key=abs, ascending=False)
)

print(f"\n--- Logistic Regression (threshold={lr_thresh:.2f}) ---")
print(f"  Top {N_TOP} features increasing revisit risk:")
for _, row in lr_importance[lr_importance['Coefficient'] > 0].head(N_TOP).iterrows():
    print(f"    {row['Feature']:45s} : {row['Coefficient']:+.4f}")

print(f"  Top {N_TOP} features decreasing revisit risk:")
for _, row in lr_importance[lr_importance['Coefficient'] < 0].head(N_TOP).iterrows():
    print(f"    {row['Feature']:45s} : {row['Coefficient']:+.4f}")

# Tree-based models: Gini / gain importance
for model_name in ['Random Forest', 'Gradient Boosting']:
    if model_name not in trained_models:
        continue
    importance = (
        pd.DataFrame({
            'Feature':    feature_cols,
            'Importance': trained_models[model_name].feature_importances_,
        })
        .sort_values('Importance', ascending=False)
    )
    thresh = best_thresholds[model_name]
    print(f"\n--- {model_name} (threshold={thresh:.2f}) — Top {N_TOP} ---")
    for i, (_, row) in enumerate(importance.head(N_TOP).iterrows()):
        print(f"  {i+1:2d}. {row['Feature']:45s} : {row['Importance']:.4f}")


# %% -----------------------------------------------------------------------
# SECTION 6 — MLP PERMUTATION IMPORTANCE (XAI)
# MLP training & results are already included in Sections 3–4 above.
# Permutation importance is computed here because MLP has no built-in
# feature_importances_ or coef_ — it shuffles each feature 15× and
# measures the resulting drop in val F1.  Larger drop = more important.
# -----------------------------------------------------------------------

nn_model = trained_models['MLP']
nn_thresh = best_thresholds['MLP']

print(f"\nComputing permutation importance for MLP on val set (n_repeats=15) ...")
perm = permutation_importance(
    nn_model, X_val_scaled, y_val,
    scoring      = 'f1',
    n_repeats    = 15,
    random_state = SEED,
    n_jobs       = -1,
)

nn_importance = (
    pd.DataFrame({
        'Feature':    feature_cols,
        'Importance': perm.importances_mean,
        'Std':        perm.importances_std,
    })
    .sort_values('Importance', ascending=False)
)

print(f"\n--- MLP Permutation Importance (threshold={nn_thresh:.2f}) — Top {N_TOP} ---")
for i, (_, row) in enumerate(nn_importance.head(N_TOP).iterrows()):
    print(f"  {i+1:2d}. {row['Feature']:45s} : {row['Importance']:+.4f}  ± {row['Std']:.4f}")

print(f"\n  Bottom {N_TOP} (least / negatively important):")
for i, (_, row) in enumerate(nn_importance.tail(N_TOP).iloc[::-1].iterrows()):
    print(f"       {row['Feature']:45s} : {row['Importance']:+.4f}  ± {row['Std']:.4f}")


# %% -----------------------------------------------------------------------
# ARCHIVED — code kept for reference, not executed
# -----------------------------------------------------------------------

# ── Neural Network (PyTorch) ───────────────────────────────────────────────
# import torch, torch.nn as nn, torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
#
# device = (torch.device('mps')  if torch.backends.mps.is_available() else
#           torch.device('cuda') if torch.cuda.is_available()          else
#           torch.device('cpu'))
#
# class NeuralNetwork(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
#             nn.Linear(64, 32),        nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
#             nn.Linear(32, 1),
#         )
#     def forward(self, x): return self.network(x)
#
# # Training loop with early stopping → see git history for full code


# ── Extended Bayesian tuning (50 trials, GB + RF separately) ──────────────
# def gb_objective(trial): ...   # 50-trial GB search (see backup files)
# def rf_objective(trial): ...   # 50-trial RF search


# ── Diagnosis consistency check (same_diag) ───────────────────────────────
# Checks whether revisit visit shares the same ICD-10 as the preceding visit.
# tmp_diag = df_raw[['hn', 'register_datetime', 'diagname_principle']].copy()
# tmp_diag['prev_diag'] = tmp_diag.sort_values(['hn','register_datetime'])
#                                  .groupby('hn')['diagname_principle'].shift(1)
# ...


# ── Prediction inspection (countTF) ───────────────────────────────────────
# def countTF(model, n=100):
#     """Print n sample rows from TP / FP / FN / TN on the val set."""
#     ...

# # %% -----------------------------------------------------------------------
# # ARCHIVED 2 — count word coverages
# # -----------------------------------------------------------------------
# # After 1h-A runs
# flag_cols = [c for c in CC_TEXT_COLS if c != 'cc_num_symptom_groups']

# # What fraction of visits have NO flag at all?
# zero_flag_mask = cc_text_df[flag_cols].sum(axis=1) == 0
# print(f"Visits with zero symptom flags : {zero_flag_mask.mean():.1%}")

# # Distribution of how many groups fire per visit
# cc_text_df['cc_num_symptom_groups'].value_counts().sort_index()

# # Which flags fire most / least
# cc_text_df[flag_cols].mean().sort_values()


# import collections, re

# # All keywords that 1h-A knows about
# known_kws = {kw for kws in {**_CLINICAL_GROUPS, **_META_FLAGS}.values() for kw in kws}

# def residual_tokens(text):
#     t = str(text).lower() if not pd.isna(text) else ''
#     # crude Thai word splitting on spaces / punctuation
#     return [tok.strip() for tok in re.split(r'[\s,/]+', t)
#             if tok.strip() and tok.strip() not in known_kws and len(tok.strip()) > 1]

# residuals = collections.Counter(
#     tok
#     for txt in df_raw.loc[zero_flag_mask, 'chief_complain']
#     for tok in residual_tokens(txt)
# )
# print(residuals.most_common(30))
