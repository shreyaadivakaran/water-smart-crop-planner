# Requirements — Water-Smart Crop Planner

> **Status:** Draft v1.0 — awaiting approval  
> **Date:** 2026-10-04  
> **Study district:** Tiruvallur, Tamil Nadu, India (13.14 N, 79.91 E)

---

## 1. Purpose

Provide smallholder farmers and agricultural extension officers in water-stressed
districts with a decision-support tool that recommends crops by simultaneously
balancing **profit**, **water use**, and **climate risk** — rather than optimising
yield alone. Outputs are explicitly labelled as indicative estimates.

---

## 2. Stakeholders

| Role | Need |
|------|------|
| Farmer / extension officer | Actionable crop recommendations with plain-language trade-off explanations |
| Agricultural analyst | Pareto front, backtest results, sensitivity analysis |
| Developer / data scientist | Clean pipeline, reproducible models, easy district swap |

---

## 3. Functional Requirements

### 3.1 Configuration

- **FR-CFG-01** A single config file (`config/config.yaml`) stores district name,
  coordinates, candidate crop list, season definitions, data paths, model
  hyperparameters, random seed, and objective weight defaults.
- **FR-CFG-02** Switching to a new district requires only config edits; no code
  changes.

### 3.2 Data Pipeline

- **FR-DAT-01** The system shall provide loaders and cleaners for five raw files:
  `weather.csv`, `yield.csv`, `prices.csv`, `crop_water.csv`, `cost.csv`.
  Column descriptions are sourced from `data/raw/README.txt`.
- **FR-DAT-02** `src/data/fetch_weather.py` shall download daily NASA POWER
  Community "Ag" data for variables: `PRECTOTCORR`, `T2M_MAX`, `T2M_MIN`,
  `RH2M`, `WS2M`, `ALLSKY_SFC_SW_DWN`. It shall convert sentinel value −999 to
  `NaN` and save output to `data/raw/weather.csv`. Fetched data shall be cached
  to `data/raw/weather_cache.json` so subsequent runs do not re-download.
  On network failure the script shall log a warning and fall back to the cache
  (or synthetic data if the cache is absent) rather than aborting.
- **FR-DAT-03** A clearly labelled **synthetic data generator**
  (`src/data/generate_synthetic.py`) shall produce all five raw files so the
  full pipeline runs end-to-end before real data is available. The generator
  shall use the same schemas as real files. Each dataset (weather, yield,
  prices, crop_water, cost) has an independent `use_synthetic` flag in config,
  so real and synthetic sources can be mixed freely. Whenever any dataset is
  synthetic, the Streamlit dashboard shall show a prominent **"⚠ SYNTHETIC DATA"**
  banner identifying which datasets are affected.
- **FR-DAT-04** A data validation step shall check: missing value ratios per
  column (warn if > 5 %), unit plausibility (e.g., rainfall ≥ 0, temperature
  range −5 °C to 60 °C), year coverage (2010–2023). Warnings shall be logged to
  `logs/data_validation.log`.
- **FR-DAT-05** Processed features shall be saved to `data/processed/` so the
  pipeline can be re-run incrementally.

### 3.3 Crop Candidates

- **FR-CRP-01** Initial candidate list: paddy, groundnut, sugarcane, black gram,
  green gram, maize, ragi. Any crop with fewer than `config.crops.min_data_years`
  years of yield data after cleaning shall be dropped. The threshold defaults to
  **8** (configurable). Each dropped crop shall be logged with crop name and
  exact reason (e.g. "only 5 usable years after cleaning").
- **FR-CRP-02** Season labels (kharif / rabi / summer) shall be applied where the
  data supports them; a crop with data for only one season shall be tagged
  accordingly.
- **FR-CRP-03** A config option (`data.merge_kancheepuram: true/false`) shall
  enable merging of Kancheepuram district records into the Tiruvallur series for
  years affected by the 2019 administrative split (years ≤ 2019). When enabled,
  merged rows shall be flagged with a `source` column value of `"merged"` and a
  warning logged.

### 3.4 Prediction Layer

- **FR-MDL-01** A **yield model** shall train Random Forest and XGBoost regressors
  with features: seasonal rainfall, T2M_MAX, T2M_MIN, crop (encoded), year.
- **FR-MDL-02** Training shall use a **time-aware split** (train on years ≤ split
  year, test on later years) to prevent data leakage.
- **FR-MDL-03** Yield model evaluation shall report MAE, RMSE, R² for both models
  and produce actual-vs-predicted plots saved to `outputs/plots/`.
- **FR-MDL-04** A **price model** shall forecast crop prices and output **P10,
  P50, and P90 prediction intervals** using three quantile regressors. The UI
  shall display P50 as the point estimate and P10/P90 as the uncertainty band.
  When price data is synthetic, a visible note shall state: *"Price intervals
  are derived from synthetic data and do not reflect real market conditions."*
- **FR-MDL-05** The best-performing yield model (by RMSE) shall be persisted to
  `models/` and loaded at inference time.

### 3.5 Risk Layer

- **FR-RSK-01** For each crop, compute the **coefficient of variation (CV)** of
  historical yield.
- **FR-RSK-02** For each crop, compute the **probability of yield falling below a
  user-defined threshold** in dry years (rainfall < 30th percentile of historical
  series).
- **FR-RSK-03** Risk scores shall be normalised to [0, 1] for use in optimisation.

### 3.6 Optimisation Layer

- **FR-OPT-01** Three objectives:
  1. **Maximise profit** = predicted yield × predicted price − cost per hectare.
  2. **Minimise water deficit** = crop water requirement − (seasonal rainfall +
     irrigation available).
  3. **Minimise climate risk** (normalised CV + dry-year failure probability).
- **FR-OPT-02** Run **NSGA-II** (via pymoo) over the discrete crop set to produce
  a Pareto front. Fixed random seed for reproducibility.
- **FR-OPT-03** Provide a **weighted-sum ranking** where users supply weights for
  the three objectives (weights sum to 1).
- **FR-OPT-04** Optionally solve a **land-allocation sub-problem**: given fixed
  total acreage and a water budget, allocate area fractions across Pareto-optimal
  crops using a simple LP or greedy heuristic.

### 3.7 Streamlit Dashboard

- **FR-UI-01** Input widgets: season selector, irrigation water available (mm),
  rainfall scenario (normal / 30 % deficit), objective weights (three sliders
  summing to 1).
- **FR-UI-02** Outputs:
  - Pareto scatter plot: profit (x-axis) vs. water use (y-axis), bubble size
    proportional to climate risk, crop name as label.
  - Ranked table of crops with columns: crop, predicted profit (₹/ha), water
    deficit (mm), risk score, weighted rank.
  - Water-saved metric vs. paddy baseline (mm and %).
  - Plain-language explanation card for the top-ranked crop.
- **FR-UI-03** All output panels shall carry a disclaimer: *"Outputs are indicative
  estimates based on historical data and synthetic modelling. Consult local
  agricultural experts before making planting decisions."*
- **FR-UI-04** Dashboard shall run on `localhost` with `streamlit run app/main.py`
  and require no cloud credentials.

### 3.8 Validation

- **FR-VAL-01** **Backtest**: for each year in the test set, compare the system's
  top recommendation against the actual dominant crop in that district; report
  hit rate and profit delta.
- **FR-VAL-02** **Sensitivity analysis**: vary rainfall (±10 %, ±30 %), price
  (±20 %), and objective weights across a grid; report how rankings change.

---

## 4. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-01 | All Python modules shall use type hints and Google-style docstrings |
| NFR-02 | All random operations shall use a single fixed seed from config |
| NFR-03 | Unit tests for risk and optimisation modules using pytest |
| NFR-04 | Pipeline shall complete end-to-end on synthetic data in < 5 minutes on a laptop CPU |
| NFR-05 | No paid APIs, no cloud dependencies for core functionality |
| NFR-06 | NASA POWER fetch is network-dependent; a cached/offline fallback (synthetic) shall always be available |
| NFR-07 | Code shall be compatible with Python 3.11 |
| NFR-08 | All plots shall be saved as static PNG in `outputs/plots/` in addition to interactive display |

---

## 5. Data Sources

| File | Source | Notes |
|------|--------|-------|
| `weather.csv` | NASA POWER Community Ag API | Daily, 2010–2023, auto-fetched |
| `yield.csv` | Dept. of Agriculture, Tamil Nadu / data.gov.in | District-level annual yield (kg/ha) |
| `prices.csv` | Agmarknet / Koyambedu mandi reports | Monthly wholesale price (₹/quintal) |
| `crop_water.csv` | FAO-56 crop coefficients + local evapotranspiration | Seasonal water requirement (mm) |
| `cost.csv` | TNAU cost-of-cultivation bulletins | Input cost per hectare (₹/ha) |

All five files can be replaced by the synthetic generator for development and
testing purposes.

---

## 6. Constraints & Assumptions

- Study period: 2010–2023 (14 seasons per crop).
- Spatial resolution: district-level aggregates; no sub-district or field-level data.
- Irrigation availability is user-supplied; the system does not model groundwater
  depletion.
- Price forecasts are short-horizon and do not model supply shocks.
- Outputs are **not** agronomic advice; legal disclaimer applies.

---

## 7. Out of Scope

- Real-time IoT sensor integration.
- Sub-field precision agriculture.
- Market linkage or procurement logistics.
- Multi-district simultaneous optimisation.
- Mobile / native app.
