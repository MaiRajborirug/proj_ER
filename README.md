# ED Revisit Prediction — Project Notes

**Goal:** Predict unscheduled 72-hour ED revisit (`is_revisit_visit_72h`) from a Thai hospital ED dataset (2020–2023).

---

## Project Structure

```
proj_ER/
├── data/                        # all data files (CSV, XLSX, NPY)
│   ├── revisit_data.csv/xlsx    # raw ED visit records (input)
│   ├── vs_2020–2023.csv/xlsx    # raw vital signs time-series (input)
│   ├── vs.csv/xlsx              # concatenated vitals
│   ├── vs_summary.csv/xlsx      # per-visit VS aggregates
│   ├── revisit_data_signal.csv  # main table + VS summary features
│   ├── df_process1.csv/xlsx     # cleaned + engineered features
│   ├── df_model_ready.csv/xlsx  # model-ready encoded features
│   ├── bert_embeddings_cache.npy  # cached WangchanBERTa CLS embeddings (1h-C)
│   └── results_*.csv            # evaluation results (train/val/test)
├── plots/                       # EDA output plots (PDF)
├── const.py                     # payment mapping dicts, lab reference values
├── data_observe.py              # EDA script → plots/
├── data_screen_signal.py        # vital sign preprocessing pipeline
├── data_exp.py                  # feature engineering + model training
└── test_1hC.py                  # standalone 1h-C (WangchanBERTa) validation
```

---

## Installation

### 1. Install core packages

```bash
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm xgboost optuna
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### 2. 1h-C only — WangchanBERTa (optional)

Only needed if you activate the `1h-C` text feature block in `data_exp.py` or run `test_1hC.py`.

```bash
pip install torch transformers sentencepiece protobuf
```

**Model weights (~500 MB)** are downloaded automatically from HuggingFace on first run:

```
Model : airesearch/wangchanberta-base-att-spm-uncased
Cache : ~/.cache/huggingface/   (default HuggingFace cache location)
```

After the first run, embeddings are also cached locally at `data/bert_embeddings_cache.npy`
(~10 min to generate on Apple Silicon MPS). Subsequent runs load from cache instantly.
Delete the file to force recomputation if the source data changes.

> **Tokenizer note:** always use `CamembertTokenizer` directly — `AutoTokenizer` on this
> checkpoint triggers a Rust tokenizer path that raises
> `TypeError: argument 'vocab': 'str' object cannot be converted to 'PyTuple'`
> on `transformers >= 5.x`.

### 3. Device

| Hardware | Behaviour |
|---|---|
| Apple Silicon (MPS) | Used automatically — no config needed |
| NVIDIA GPU (CUDA) | Uncomment the `cuda` branch in the 1h-C block / `test_1hC.py` |
| CPU fallback | Used automatically if neither MPS nor CUDA is available |

---

## 1. Raw Data

| File | Description |
| ---- | ----------- |
| `data/revisit_data.csv` | Main tabular ED visit records (demographics, labs, chief complaint, payment, timing) |
| `data/vs_2020–2023.csv` | Vital signs time-series per visit (BP, PR, RR, BT, O2sat, pain score, GCS E/M/V) |

---

## 1.5 Data Observation (`data_observe.py`)

EDA covering class balance, missing rates, distributions, and revisit rates by category.
Outputs: `plots/01` – `09`, `11` (PDF)

### Dataset Overview

| Item | Value |
|---|---|
| Total visits | 107,835 |
| Unique patients (HN) | 67,539 |
| Date range | 2020–2023 |
| Yearly volume | 2020: 25,263 · 2021: 18,283 · 2022: 28,936 · 2023: 35,353 |
| Age | mean 49.2 y, median 48 y, range [18, 121] |
| LOS | median 95 min, 75th pct 169 min |

### Plot Index

#### 01 · Class Balance & Label Alignment
- Positive (`is_revisit_visit_72h=1`): **3,238 (3.00%)**
- Negative: **104,597 (97.00%)**
- **Imbalance ratio 32.3 : 1** — threshold tuning and F1/AUC metrics are essential.
- **Flag alignment** (`revisit_72h_flag` × `is_revisit_visit_72h`):

  |  | `is_revisit=0` | `is_revisit=1` |
  |---|---|---|
  | `flag=0` | 102,041 | 2,556 |
  | `flag=1` | 2,556 | 682 |

  → `revisit_72h_flag` marks the **index visit** that was followed by a return; `is_revisit_visit_72h` marks the **return visit** itself. The 682 overlap are visits that are both a revisit and triggered yet another revisit within 72 h.

#### 02 · Missing Values in Raw Data
Admin/ward columns (~99.5%) and leakage columns (`next_arrival_dt`, `hours_to_revisit`, `สิทธิการรักษาจากคัดกรอง`) excluded.

| Column | Missing |
|---|---|
| `bicarb`, `chloride` | ~94.5% |
| `sodium`, `potassium`, `creatinine` | ~72% |
| `hb`, `plt`, `wbc` | ~69% |
| `doctor_hour/minute` | ~30% |
| `chief_complain` | 0.1% |

#### 03 · Lab Missing Rate by Class
Lab missingness is **higher among revisit patients** (e.g. creatinine 78.7% vs 71.4%), suggesting sicker patients have less complete records.

#### 04 · Vital Sign Missing Rate — raw vs.csv
Source: `vs.csv` (834,080 individual reading rows), **before** per-visit aggregation and imputation.

| Vital | Missing in raw rows |
|---|---|
| `ps` | ~93.8% |
| `bt` | ~36.4% |
| `lbpn`, `hbpn` | ~24.9% |
| `rr`, `o2sat`, `pr` | ~21–22% |
| `e`, `m`, `v` | ~12% |

Note: after aggregation (`data_screen_signal.py`), per-visit missingness drops substantially because a patient only needs at least one non-null reading per visit to be non-missing.

#### 05 · Processed Feature Distributions (Chief Complaint & Register Time)
Source: `df_process1.csv`. Layout: 2 rows (non-revisit / revisit) × 3 cols.

- **CC Duration (`chief_complaint_total_hr`):** revisit patients have lower median (1 h vs 24 h).
- **CC Urgency Level:** revisit patients cluster at 0 (no time reference) and level 1–2 (minutes/hours).
- **Registration Hour:** revisit patients arrive slightly earlier in the day (median 13 h vs 16 h).

#### 06 · Categorical Distributions by Class
- **Sex:** Female majority in both classes (~61%).
- **ESI:** Both classes dominated by levels 3 and 4; revisit patients show a slightly higher share at level 3.
- **Payment:** UC and SSO proportionally higher among revisit patients.
- **ICD-10:** Symptoms/Signs and Respiratory top both classes.

#### 07 · Revisit Rate by Category

| Group | Revisit Rate |
|---|---|
| SSO patients | 4.4% |
| UC patients | 4.1% |
| ESI 5 (least urgent) | 3.8% |
| ESI 3 | 3.5% |
| Civil Servant/Gov | 3.4% |
| ESI 2 | 2.9% |
| ESI 4 | 2.3% |
| ESI 1 (most urgent) | 1.3% |
| Private/Self-Pay | 1.9% |

#### 08 · Vital Sign Distributions by Class
Broadly similar between classes. Pain score and GCS show slightly higher mass at extremes for revisit patients.

#### 09 · Temporal Volume & Revisit Rate (2020–2023)
Visit volume dipped in 2021 (likely COVID-related) then recovered. **Revisit rate has risen year-on-year:** 2.0% → 2.8% → 3.1% → 3.7%.

#### 11 · Prior Visit Count Distributions by Window & Class
- The vast majority of visits have 0 prior visits in all windows (~87–96% depending on window).
- Revisit patients have a consistently **heavier tail** (more prior visits) across all windows.
- The 30d and 60d windows show the clearest separation, supporting their inclusion as predictive features.

---

## 2. Signal Preprocessing (`data_screen_signal.py`)

**Purpose:** Merge the multi-row vital sign records into per-visit summary features and attach them to the main table.

### Steps

1. **Concatenate** `vs_2020–2023.csv` → `data/vs.csv`.
2. **Clean vitals**
  - Pain score (`ps`): clamp to [0, 10], fill NaN → 0.
  - GCS components (`e`, `m`, `v`): clean human error (67 -> 6, 5.5->, T->1), fill NaN with max-normal values (E=4, M=6, V=5); compute `emv = e + m + v`.
3. **Align by (HN, visit date):** match each `df_raw` row to its vital sign records using `hn` + `register_date` = `vst_date`.
4. **Vectorized aggregation** (`build_vs_summary_fast`): for each vital column (`lbpn`, `hbpn`, `pr`, `rr`, `bt`, `o2sat`, `ps`, `e`, `m`, `v`, `emv`) compute:
  - `first_`* — literal first-row value (time-ordered).
  - `mean_`* — mean of non-NaN readings.
  - `last_`* — literal last-row value.
  - `*_missing` flag — 1 if value is NaN (i.e., vital was never recorded).
5. **Imputation:** missing values filled with population median of `mean_`* across all patients.
6. **Output:** `data/revisit_data_signal.csv` = `revisit_data.csv` + VS summary columns.

---

## 3. Data Cleaning & Feature Engineering (`data_exp.py`)

Input: `data/revisit_data_signal.csv`.

### Cleaning

- **Payment right** (`payment_right`): map 70+ Thai insurance strings → 4 categories using `ER_PAYMENT_MAP2`: `Private/Self-Pay`, `Universal Coverage (UC)`, `Social Security (SSO)`, `Civil Servant/Gov`.
- **Time features:** derive `los_total_minutes`, `doctor_total_minutes` from hour+minute columns; `register_time` (hour of day); `register_group` (2=night 00–08h, 4=day/eve 08–24h).
- **Missing handling:**
  - `doctor_total_minutes`: impute with `median(doctor / los)` ratio × los; add `_missing` flag.
  - Lab values (`sodium`, `potassium`, `chloride`, `bicarb`, `hb`, `plt`, `wbc`): impute with column median; add `_missing` flag.
  - Radiology columns: add `_missing` flag only.
- **Drop rows** with missing `sex`.

### Engineered Features (Section 1a–1j)

| Section | Feature | Description |
| ------- | ------- | ----------- |
| 1a | Payment mapping | Thai insurance → 4 categories |
| 1b | Time/LOS | `los_total_minutes`, `doctor_total_minutes`, `register_time`, `register_group` |
| 1c | ED crowding | `num_patient_arrive`, `density_patient_arrive` — patients in ED at registration (O(n log n) binary search) |
| 1d | Doctor crowding | `num_patient_meetdoc`, `density_patient_meetdoc` — same count at doctor contact time |
| 1e | Lab imputation | Median imputation + `_missing` flags for 7 lab values; composite `lab` flag |
| 1f | Radiology | `_missing` flags for CT, ECG, X-ray, ultrasound columns |
| 1g | Chief complaint duration | Regex on Thai+English free text → `total_hr`, `urgent_level`, bucket flags, `time_missing` |
| 1h | Chief complaint text | Three switchable options — see below |
| 1i | Prior visit history | `visit_count_prev_{3,7,15,30,60}d` sliding window; zeroed before 2020-03-01 |
| 1j | Save | → `data/df_process1.csv` |

### Section 1h — Chief Complaint Text Features (three options)

Only one option should be active at a time. The active option sets `CC_TEXT_COLS`, which is automatically appended to `numerical_cols_1` in Section 2.

#### 1h-A: Rule-based symptom flags (current default — active)

Binary flags per clinical/meta group. ~17 columns total.

**Required packages:** none beyond standard sklearn/pandas.

**Coverage check** (run after enabling):
```python
zero_flag_mask = cc_text_df[flag_cols].sum(axis=1) == 0
print(f"Visits with zero symptom flags: {zero_flag_mask.mean():.1%}")
```
Target: <10% zero-flag rate.

Clinical groups: `cc_respiratory`, `cc_gi`, `cc_neuro`, `cc_numbness`, `cc_cardiac`, `cc_urinary`, `cc_msk`, `cc_derm`, `cc_eye_ent`, `cc_fever`, `cc_trauma`

Meta flags: `cc_atk_pos`, `cc_atk_neg`, `cc_covid`, `cc_ems`, `cc_referred`, `cc_severity_inc`, `cc_followup`, `cc_substance`, `cc_injection`, `cc_device`

Summary count: `cc_num_symptom_groups`

#### 1h-B: TF-IDF + character n-gram (2–4) + TruncatedSVD → 15 components

**Required packages:** none beyond standard sklearn.

To activate: uncomment the 1h-B block and comment out 1h-A.

#### 1h-C: WangchanBERTa CLS embeddings + PCA → 15 components

**Required packages — must install before first use:**
```bash
pip install sentencepiece protobuf
```

> **Important:** Use `CamembertTokenizer` directly — do **not** use `AutoTokenizer`.
> `AutoTokenizer` on this checkpoint triggers a Rust tokenizer path that fails with
> `TypeError: argument 'vocab': 'str' object cannot be converted to 'PyTuple'`
> on `transformers >= 5.x`.

**Embedding cache:** on first run, embeddings are saved to `data/bert_embeddings_cache.npy` (~10 min on MPS). Subsequent runs load from cache instantly. Delete the file to force recomputation (e.g. if source data changes row count).

**Device:** uses Apple Silicon MPS by default. For NVIDIA machines, uncomment the `cuda` branch in the 1h-C block.

**PCA leakage note:** PCA is fit on train rows only (2020–2021), then transforms all rows.

To activate: uncomment the 1h-C block and comment out 1h-A.

---

### Feature Groups (for ablation — `DATA_TYPE`)

| `DATA_TYPE` | Features included |
| ----------- | ------------------------------------------------------------------ |
| 0           | Prior visit counts only |
| 1           | + Demographics, timing, crowding, chief complaint (1g + 1h) |
| 2           | + Labs, doctor time, vital signs (first only) ← **current default** |
| 3           | + VS mean/last, LOS, **icd10_group** (post-discharge only) |

> `icd10_group` is only available after discharge — it is excluded from DATA_TYPE 0–2 to prevent leakage.

---

## 4. Experiment Design

### Split (Temporal)

Temporal split by year — mirrors real deployment (model trained on past, evaluated on future):

| Split | Years     | Note |
| ----- | --------- | ---- |
| Train | 2020–2021 | Same HN may appear across splits — intentional and realistic |
| Val   | 2022      | Used for threshold tuning and HP search |
| Test  | 2023      | Final held-out evaluation |

### Class Balancing

Full unbalanced training set. Downsampling to `RATIO_NEG2POS` is implemented but commented out.

### Scaling

`StandardScaler` fit on train, applied to val/test.

### Categorical Encoding

One-hot encode `sex`, `payment_right` (all DATA_TYPEs); `icd10_group` added only at DATA_TYPE=3.

---

## 5. Models

| Model               | Library  | Notes                                    |
| ------------------- | -------- | ---------------------------------------- |
| Logistic Regression | sklearn  | `max_iter=1000`                          |
| Random Forest       | sklearn  | Bayesian HP search (Optuna/TPE)          |
| Gradient Boosting   | sklearn  | Bayesian HP search (Optuna/TPE)          |
| LightGBM            | lightgbm | 100 trees                                |
| MLP                 | sklearn  | (64,32) ReLU, Adam, early stopping       |

(XGBoost, KNN, LinearSVC also implemented but currently commented out.)

### Hyperparameter Tuning

RF and Gradient Boosting only — Bayesian search via Optuna TPE, optimising val F1 across `BAYESIAN_TRIALS` trials. Best model replaces the baseline in `trained_models`.

### Threshold Optimisation

All models: grid search over `THRESH_CANDIDATES = linspace(0.05, 0.95, 10)`, optimised on val F1.

---

## 6. Metrics

Evaluated at default threshold (0.5) and best threshold:

| Metric      | Definition                          |
| ----------- | ----------------------------------- |
| Accuracy    | (TP+TN) / N                         |
| Sensitivity | TP / (TP+FN)  — Recall / TPR        |
| Specificity | TN / (TN+FP)  — TNR                 |
| PPV         | TP / (TP+FP)  — Precision           |
| NPV         | TN / (TN+FN)                        |
| F1          | Harmonic mean of Sensitivity & PPV  |
| AUC         | ROC-AUC                             |

Results saved to `data/results_train.csv`, `data/results_test.csv`, `data/results_eval_best_threshold.csv`, `data/results_test_best_threshold.csv`.

---

## 7. XAI — Feature Importance (Section 5–6)

| Model               | Method              | Notes |
| ------------------- | ------------------- | ----- |
| Logistic Regression | Signed coefficients | Scaled features → magnitude = importance |
| Random Forest       | MDI (Gini)          | Built-in `feature_importances_`; fast but biased toward high-cardinality features |
| Gradient Boosting   | MDI (Gain)          | Same as RF |
| MLP                 | Permutation importance | Model-agnostic; 15 repeats on val set, scoring=F1 |

For RF: compare MDI vs permutation importance side-by-side — features high in MDI but low in permutation are likely overfitting or correlation artefacts.

---

## 8. Intermediate Files

| File | Created by | Contents |
| ---- | ---------- | -------- |
| `data/vs.csv` | `data_screen_signal.py` | Concatenated vitals 2020–2023 |
| `data/vs_summary.csv` | `data_screen_signal.py` | Per-visit VS aggregates |
| `data/revisit_data_signal.csv` | `data_screen_signal.py` | Main table + VS summary features |
| `data/df_process1.csv` | `data_exp.py` | Cleaned + engineered features |
| `data/bert_embeddings_cache.npy` | `data_exp.py` (1h-C) | Cached WangchanBERTa CLS embeddings |
| `data/df_model_ready.csv` | `data_exp.py` | Model-ready encoded features |
| `data/results_*.csv` | `data_exp.py` | Evaluation results (train/val/test) |
