# Data Observation Notes

**Script:** `data_observe.py`
**Inputs:** `revisit_data.csv` (raw), `revisit_data_signal.csv` (+ VS summary), `vs.csv` (raw VS readings), `df_process1.csv` (processed)
**Outputs:** `./plots/01` – `09`, `11` (PDF)

---

## Dataset Overview

| Item | Value |
|---|---|
| Total visits | 107,835 |
| Unique patients (HN) | 67,539 |
| Date range | 2020–2023 |
| Yearly volume | 2020: 25,263 · 2021: 18,283 · 2022: 28,936 · 2023: 35,353 |
| Age | mean 49.2 y, median 48 y, range [18, 121] |
| LOS | median 95 min, 75th pct 169 min |

---

## Plot Index

### 01 · Class Balance & Label Alignment
- Positive (`is_revisit_visit_72h=1`): **3,238 (3.00%)**
- Negative: **104,597 (97.00%)**
- **Imbalance ratio 32.3 : 1** — threshold tuning and F1/AUC metrics are essential.
- **Flag alignment** (`revisit_72h_flag` × `is_revisit_visit_72h`):

  |  | `is_revisit=0` | `is_revisit=1` |
  |---|---|---|
  | `flag=0` | 102,041 | 2,556 |
  | `flag=1` | 2,556 | 682 |

  → `revisit_72h_flag` marks the **index visit** that was followed by a return; `is_revisit_visit_72h` marks the **return visit** itself. The 682 overlap are visits that are both a revisit and triggered yet another revisit within 72 h.

### 02 · Missing Values in Raw Data
Admin/ward columns (~99.5%) and leakage columns (`next_arrival_dt`, `hours_to_revisit`, `สิทธิการรักษาจากคัดกรอง`) excluded.

| Column | Missing |
|---|---|
| `bicarb`, `chloride` | ~94.5% |
| `sodium`, `potassium`, `creatinine` | ~72% |
| `hb`, `plt`, `wbc` | ~69% |
| `doctor_hour/minute` | ~30% |
| `chief_complain` | 0.1% |

### 03 · Lab Missing Rate by Class
Lab missingness is **higher among revisit patients** (e.g. creatinine 78.7% vs 71.4%), suggesting sicker patients have less complete records. % labels are shown on each bar.

### 04 · Vital Sign Missing Rate — raw vs.csv
Source: `vs.csv` (834,080 individual reading rows), **before** per-visit aggregation and imputation.

| Vital | Missing in raw rows |
|---|---|
| `ps` | ~93.8% |
| `bt` | ~36.4% |
| `lbpn`, `hbpn` | ~24.9% |
| `rr`, `o2sat`, `pr` | ~21–22% |
| `e`, `m`, `v` | ~12% |

Note: after aggregation (`data_screen_signal.py`), per-visit missingness drops substantially because a patient only needs at least one non-null reading per visit to be non-missing.

### 05 · Processed Feature Distributions (Chief Complaint & Register Time)
Source: `df_process1.csv`. Layout: 2 rows (non-revisit / revisit) × 3 cols.

- **CC Duration (`chief_complaint_total_hr`):** revisit patients have lower median (1 h vs 24 h) — they tend to mention shorter symptom durations.
- **CC Urgency Level:** revisit patients cluster at 0 (no time reference) and level 1–2 (minutes/hours), while non-revisit patients more often mention day-scale durations.
- **Registration Hour:** revisit patients arrive slightly earlier in the day (median 13 h vs 16 h).

### 06 · Categorical Distributions by Class
Layout mirrors plot 05: 2 rows (non-revisit / revisit) × 4 cols (sex, ESI, payment, top-10 ICD-10).

- **Sex:** Female majority in both classes (~61%).
- **ESI:** Both classes dominated by levels 3 and 4; revisit patients show a slightly higher share at level 3.
- **Payment:** Distribution broadly similar; UC and SSO proportionally higher among revisit patients.
- **ICD-10:** Symptoms/Signs and Respiratory top both classes.

### 07 · Revisit Rate by Category

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

### 08 · Vital Sign Distributions by Class (density overlay)
Broadly similar between classes. Pain score and GCS show slightly higher mass at extremes for revisit patients.

### 09 · Temporal Volume & Revisit Rate (2020–2023)
Visit volume dipped in 2021 (likely COVID-related) then recovered. **Revisit rate has risen year-on-year:** 2.0% → 2.8% → 3.1% → 3.7%.

### 11 · Prior Visit Count Distributions by Window & Class
Source: `df_process1.csv`. Layout: 2 rows (non-revisit / revisit) × 5 cols (3d / 7d / 15d / 30d / 60d). Integer bin histogram with density scale; annotation shows % with zero prior visits and median.

- The vast majority of visits have 0 prior visits in all windows (~87–96% depending on window).
- Revisit patients have a consistently **heavier tail** (more prior visits) across all windows, especially the longer ones (30d, 60d).
- The 30d and 60d windows show the clearest separation, supporting their inclusion as predictive features.
