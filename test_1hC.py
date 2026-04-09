# %% -----------------------------------------------------------------------
# test_1hC.py  —  Validate 1h-C (WangchanBERTa + PCA) end-to-end
#
# Assumes df_process1.csv already exists (run Section 1 of data_exp.py first).
# PCA is fit on TRAIN rows only to avoid any data leakage.
# -----------------------------------------------------------------------

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModel
from transformers.models.camembert.tokenization_camembert import CamembertTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import is_sentencepiece_available
from sklearn.decomposition import PCA
from sklearn.preprocessing  import StandardScaler
from sklearn.ensemble        import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.neural_network  import MLPClassifier
from sklearn.metrics         import f1_score, roc_auc_score, accuracy_score, \
                                    confusion_matrix, recall_score
import optuna
from optuna.samplers import TPESampler
from lightgbm import LGBMClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Config ─────────────────────────────────────────────────────────────────
SEED             = 41
DATA_TYPE        = 1          # 0=prev-visits | 1=+admission | 2=+labs/VS
DATA_DIR         = 'data/'    # all CSV/XLSX/NPY files live here
N_PCA_COMPONENTS = 15
BERT_MODEL_NAME  = 'airesearch/wangchanberta-base-att-spm-uncased'
BATCH_SIZE       = 64         # BERT inference batch size (lower if OOM)
MAX_LENGTH       = 64         # token truncation length
BAYESIAN_TRIALS  = 20
THRESH_CANDIDATES = np.linspace(0.05, 0.95, 10).round(4)
# ── Device selection ──────────────────────────────────────────────────────
# Apple Silicon (MPS) — primary
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("Using MPS device (Apple Silicon)")
# NVIDIA GPU — uncomment for CUDA-capable machines
# elif torch.cuda.is_available():
#     DEVICE = torch.device('cuda')
#     print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device('cpu')
    print("MPS not available, using CPU")


# ── Load processed data ────────────────────────────────────────────────────
print("Loading df_process1.csv …")
df = pd.read_csv(f'{DATA_DIR}df_process1.csv')
print(f"  Shape: {df.shape}")


# ── Temporal masks (needed for PCA fit-on-train) ───────────────────────────
years       = pd.to_datetime(df['register_datetime'], errors='coerce').dt.year.values
train_mask  = np.isin(years, [2020, 2021])
val_mask    = years == 2022
test_mask   = years == 2023
print(f"  Train rows: {train_mask.sum():,}  |  Val: {val_mask.sum():,}  |  Test: {test_mask.sum():,}")


# ── 1h-C: WangchanBERTa CLS embeddings ────────────────────────────────────
print(f"\nLoading BERT model: {BERT_MODEL_NAME}  (device={DEVICE}) …")
if not is_sentencepiece_available():
    raise RuntimeError("Install sentencepiece: pip install sentencepiece")
if not issubclass(CamembertTokenizer, PreTrainedTokenizer):
    raise RuntimeError(
        f"WangchanBERTa needs transformers 4.x (got {transformers.__version__}). "
        "pip install 'transformers>=4.45,<5'"
    )
tokenizer  = CamembertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
bert_model.eval()

texts = df['chief_complain'].fillna('').tolist()

def embed_batch(batch_texts):
    enc = tokenizer(
        batch_texts,
        return_tensors='pt',
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = bert_model(**enc)
    return out.last_hidden_state[:, 0, :].cpu().numpy()   # CLS token

print(f"Extracting embeddings for {len(texts):,} rows in batches of {BATCH_SIZE} …")
all_embeddings = []
for i in range(0, len(texts), BATCH_SIZE):
    if i % (BATCH_SIZE * 20) == 0:
        print(f"  {i:>7,} / {len(texts):,}", flush=True)
    all_embeddings.append(embed_batch(texts[i : i + BATCH_SIZE]))

embeddings = np.vstack(all_embeddings)
print(f"Embeddings shape: {embeddings.shape}")

# PCA fit on TRAIN only, transform all
print(f"\nFitting PCA (n={N_PCA_COMPONENTS}) on train rows …")
pca         = PCA(n_components=N_PCA_COMPONENTS, random_state=SEED)
pca.fit(embeddings[train_mask])
cc_matrix   = pca.transform(embeddings)

CC_TEXT_COLS = [f'cc_bert_{i}' for i in range(N_PCA_COMPONENTS)]
cc_text_df   = pd.DataFrame(cc_matrix, columns=CC_TEXT_COLS, index=df.index)
df           = pd.concat([df, cc_text_df], axis=1)
print(f"PCA explained variance (cum): "
      f"{pca.explained_variance_ratio_.cumsum()[-1]:.3f}  "
      f"(top-3: {pca.explained_variance_ratio_[:3].round(3).tolist()})")


# ── Section 2: ML data preparation ────────────────────────────────────────
TARGET_COL   = 'is_revisit_visit_72h'
ORDINAL_COLS = ['esi']
CATEG_COLS   = ['sex', 'payment_right', 'icd10_group']

numerical_cols_0 = [
    'visit_count_prev_3d', 'visit_count_prev_7d', 'visit_count_prev_15d',
]
numerical_cols_1 = [
    'age_ofvisit', 'register_time', 'register_group',
    'num_patient_arrive', 'density_patient_arrive',
    'chief_complaint_time_missing', 'chief_complaint_total_hr',
    'chief_complaint_urgent_level',
] + CC_TEXT_COLS
numerical_cols_2 = [
    'sodium', 'potassium', 'chloride', 'bicarb', 'hb', 'plt', 'wbc',
    'doctor_total_minutes', 'doctor_total_minutes_missing',
    'sodium_missing', 'potassium_missing', 'chloride_missing',
    'bicarb_missing', 'hb_missing', 'plt_missing', 'wbc_missing', 'lab',
    'num_patient_meetdoc', 'density_patient_meetdoc',
    'first_lbpn', 'first_lbpn_missing', 'first_hbpn', 'first_hbpn_missing',
    'first_pr',   'first_pr_missing',   'first_rr',   'first_rr_missing',
    'first_bt',   'first_bt_missing',   'first_o2sat','first_o2sat_missing',
    'first_ps',   'first_ps_missing',   'first_e',    'first_e_missing',
    'first_m',    'first_m_missing',    'first_v',    'first_v_missing',
    'first_emv',  'first_emv_missing',
]

if   DATA_TYPE == 0: numerical_cols = numerical_cols_0
elif DATA_TYPE == 1: numerical_cols = numerical_cols_0 + numerical_cols_1
elif DATA_TYPE == 2: numerical_cols = numerical_cols_0 + numerical_cols_1 + numerical_cols_2
else: raise ValueError(f"DATA_TYPE must be 0-2 for this test, got {DATA_TYPE}")

df_encoded       = pd.get_dummies(df, columns=CATEG_COLS, drop_first=False)
encoded_cat_cols = [c for c in df_encoded.columns
                    if any(c.startswith(cat + '_') for cat in CATEG_COLS)]
feature_cols     = numerical_cols + ORDINAL_COLS + encoded_cat_cols
df_model         = df_encoded[['hn', 'register_datetime'] + feature_cols + [TARGET_COL]].copy()

missing = df_model[feature_cols].isnull().sum()
if missing.any():
    print("\nMissing values:")
    print(missing[missing > 0])

print(f"\nFeature matrix: {len(feature_cols)} features × {len(df_model):,} rows")


# ── Section 3a: Temporal split ─────────────────────────────────────────────
X = df_model[feature_cols].values
y = df_model[TARGET_COL].values

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

print(f"Class-1 rate — Train: {y_train.mean():.3f} | Val: {y_val.mean():.3f} | Test: {y_test.mean():.3f}")


# ── Helpers ────────────────────────────────────────────────────────────────
def evaluate_model(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'Accuracy':    accuracy_score(y_true, y_pred),
        'Sensitivity': recall_score(y_true, y_pred),
        'Specificity': tn / (tn + fp),
        'PPV':         tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        'NPV':         tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        'F1':          f1_score(y_true, y_pred),
        'AUC':         roc_auc_score(y_true, y_prob),
    }

COL_ORDER = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1', 'AUC']


# ── Section 3e: Baseline training ─────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(random_state=SEED, max_iter=1000),
    'Random Forest':        RandomForestClassifier(random_state=SEED, n_estimators=100),
    'Gradient Boosting':    GradientBoostingClassifier(random_state=SEED, n_estimators=100),
    'LightGBM':             LGBMClassifier(random_state=SEED, n_estimators=100, verbose=-1),
    'MLP':                  MLPClassifier(
                                hidden_layer_sizes=(64, 32), activation='relu',
                                solver='adam', alpha=1e-4, batch_size=256,
                                learning_rate='adaptive', max_iter=200,
                                early_stopping=True, validation_fraction=0.1,
                                n_iter_no_change=15, random_state=SEED, verbose=False,
                            ),
}

trained_models = {}
results_val    = []

print("\n" + "=" * 70)
print("BASELINE — VALIDATION (default threshold=0.5)")
print("=" * 70)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    y_v_prob = model.predict_proba(X_val_scaled)[:, 1]
    y_v_pred = (y_v_prob >= 0.5).astype(int)
    m = evaluate_model(y_val, y_v_pred, y_v_prob)
    m['Model'] = name
    results_val.append(m)
    print(f"  {name:<25s}  F1={m['F1']:.4f}  AUC={m['AUC']:.4f}")

df_baseline_val = pd.DataFrame(results_val).set_index('Model')[COL_ORDER]


# ── Section 4: Bayesian HP tuning (RF + GB) ────────────────────────────────
def make_rf_trial(trial):
    return RandomForestClassifier(
        n_estimators     = trial.suggest_int('n_estimators',    50, 300),
        max_depth        = trial.suggest_int('max_depth',        3,  20),
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1,  20),
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

TRIAL_BUILDERS = {'Random Forest': make_rf_trial, 'Gradient Boosting': make_gb_trial}

print(f"\n{'=' * 70}")
print(f"BAYESIAN TUNING  ({BAYESIAN_TRIALS} trials each)")
print("=" * 70)

for name, builder in TRIAL_BUILDERS.items():
    def objective(trial, _name=name, _builder=builder):
        m = _builder(trial)
        m.fit(X_train_scaled, y_train)
        prob = m.predict_proba(X_val_scaled)[:, 1]
        return max(f1_score(y_val, (prob >= t).astype(int), zero_division=0)
                   for t in THRESH_CANDIDATES)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=BAYESIAN_TRIALS, show_progress_bar=False)

    print(f"\n  {name}: best val F1={study.best_value:.4f}  params={study.best_params}")
    best = builder(study.best_trial)
    best.fit(X_train_scaled, y_train)
    trained_models[name] = best
    models[name]         = best


# ── Threshold sweep ────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("THRESHOLD SWEEP — VALIDATION")
print("=" * 70)

best_thresholds    = {}
results_val_thresh = []

for name in models:
    prob = trained_models[name].predict_proba(X_val_scaled)[:, 1]
    best_f1, best_thresh = 0.0, 0.5
    for t in THRESH_CANDIDATES:
        f = f1_score(y_val, (prob >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, float(t)
    best_thresholds[name] = best_thresh
    m = evaluate_model(y_val, (prob >= best_thresh).astype(int), prob)
    m['Model'] = name; m['Threshold'] = best_thresh
    results_val_thresh.append(m)

df_tuned_val = pd.DataFrame(results_val_thresh).set_index('Model')
df_tuned_val.columns = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1', 'AUC']

THRESH_COL_ORDER  = ['Threshold'] + COL_ORDER
df_thresh_final   = pd.DataFrame(results_val_thresh).set_index('Model')[THRESH_COL_ORDER]
df_thresh_final.columns = ['Thresh', 'Acc', 'Sen', 'Spec', 'PPV', 'NPV', 'F1', 'AUC']

print(df_thresh_final.round(4).to_string())


# ── Final summary ──────────────────────────────────────────────────────────
best_model_name = df_thresh_final['F1'].idxmax()
best_row        = df_thresh_final.loc[best_model_name]

print(f"\n{'=' * 70}")
print(f"1h-C  BEST MODEL (val, optimised threshold)")
print(f"{'=' * 70}")
print(f"  Model     : {best_model_name}")
print(f"  Threshold : {best_row['Thresh']:.4f}")
print(f"  F1        : {best_row['F1']:.4f}")
print(f"  AUC       : {best_row['AUC']:.4f}")
print(f"  Sensitivity: {best_row['Sen']:.4f}  |  Specificity: {best_row['Spec']:.4f}")
print(f"  PPV        : {best_row['PPV']:.4f}  |  NPV        : {best_row['NPV']:.4f}")
print(f"{'=' * 70}")

df_thresh_final.to_csv(f'{DATA_DIR}results_1hC_val.csv')
print(f"Saved → {DATA_DIR}results_1hC_val.csv")
