# Tasks — Water-Smart Crop Planner

> **Status:** Draft v1.0 — awaiting approval  
> **Date:** 2026-10-04  
> **Companion docs:** requirements.md, design.md

Tasks are grouped into eight milestones that must be executed in order.
Each task lists its acceptance criteria and the files it creates or modifies.

Legend: ⬜ not started · 🔄 in progress · ✅ done

---

## Milestone 0 — Project Scaffold

### T-000 ⬜ Create directory structure and stub files
**Depends on:** nothing  
**Files created:**
```
config/
data/raw/   data/processed/
models/     outputs/plots/   logs/
notebooks/
src/data/   src/models/   src/risk/   src/optimization/   src/utils/
app/components/
tests/
requirements.txt
README.md  (skeleton)
```
**Acceptance criteria:**
- All directories exist.
- Every `src/` sub-package has an `__init__.py`.
- `requirements.txt` lists pinned versions from design.md §7.
- `README.md` contains project title, one-paragraph description, and a
  `## Setup` placeholder.

---

### T-001 ⬜ Write `config/config.yaml`
**Depends on:** T-000  
**Files created:** `config/config.yaml`  
**Acceptance criteria:**
- Contains all keys described in design.md §3.1, including per-dataset
  `use_synthetic` flags for weather, yield, prices, crop_water, cost.
- `min_data_years: 8` (default).
- `merge_kancheepuram: true` (default).
- `weather_cache` path defined.
- Price model `quantiles: [0.10, 0.50, 0.90]` defined.
- Passes a hand-check that `pyyaml.safe_load` reads it without error.

---

### T-002 ⬜ Write `src/utils/config_loader.py` and `src/utils/logger.py`
**Depends on:** T-001  
**Files created:** `src/utils/config_loader.py`, `src/utils/logger.py`  
**Acceptance criteria:**
- `load_config(path)` returns a dict and raises `ValueError` on missing required
  keys.
- `get_logger(name)` returns a Python `logging.Logger` that writes to both
  console and `logs/` file.
- Both modules have type hints and docstrings.

---

## Milestone 1 — Data Layer

### T-010 ⬜ Write `data/raw/README.txt`
**Depends on:** T-000  
**Files created:** `data/raw/README.txt`  
**Acceptance criteria:**
- Describes every column in all five raw CSVs (name, type, unit, notes).
- Includes a section on sentinel values and encoding conventions.

---

### T-011 ⬜ Write `src/data/generate_synthetic.py`
**Depends on:** T-001, T-002  
**Files created:** `data/raw/weather.csv`, `data/raw/yield.csv`,
`data/raw/prices.csv`, `data/raw/crop_water.csv`, `data/raw/cost.csv`  
**Acceptance criteria:**
- Respects per-dataset `use_synthetic` flags; only generates files whose flag
  is `true`.
- Each generated file starts with the synthetic-data disclaimer comment.
- Covers years 2010–2023, all three seasons, all seven candidate crops.
- Uses fixed seed from config; re-running produces identical output.
- Running `python src/data/generate_synthetic.py` completes without error.

---

### T-012 ⬜ Write `src/data/fetch_weather.py`
**Depends on:** T-002  
**Files created:** `src/data/fetch_weather.py`  
**Acceptance criteria:**
- Calls NASA POWER Community Ag API for the district coordinates and date range
  in config.
- Downloads variables: `PRECTOTCORR`, `T2M_MAX`, `T2M_MIN`, `RH2M`, `WS2M`,
  `ALLSKY_SFC_SW_DWN`.
- Converts sentinel −999 to `NaN`.
- Saves output to `data/raw/weather.csv` with the standard schema.
- Caches raw API response to `config.data.weather_cache` (JSON); on a
  subsequent run, loads from cache if the date range is already covered.
- On network failure: logs a `WARNING`, falls back to existing cache if
  available, otherwise falls back to synthetic weather (logs clearly).
  Does **not** raise an unhandled exception.
- Has a `--dry-run` flag that prints the API URL without fetching.
- Type hints and docstrings throughout.

---

### T-013 ⬜ Write `src/data/clean_weather.py`
**Depends on:** T-011 or T-012 (either raw weather source)  
**Files created/modified:** `data/processed/` (intermediate)  
**Acceptance criteria:**
- Computes per-season-per-year aggregates: `seasonal_rainfall_mm`,
  `mean_tmax`, `mean_tmin`, `heat_stress_days`, `dry_spell_days`.
- Handles `NaN` by interpolation or forward-fill; logs filled cell count.
- Returns a clean `pd.DataFrame` with a `(year, season)` index.
- Unit tests in `tests/test_data_pipeline.py` cover at least: correct season
  aggregation, NaN handling, heat-stress day count.

---

### T-014 ⬜ Write `src/data/clean_tabular.py`
**Depends on:** T-011  
**Acceptance criteria:**
- Separate loader + cleaner functions for `yield.csv`, `prices.csv`,
  `crop_water.csv`, `cost.csv`.
- Drops crops with fewer than `config.crops.min_data_years` (default 8) years
  of usable yield data; logs each dropped crop with crop name and exact reason
  (e.g. "only 5 usable years after NaN removal").
- When `config.data.merge_kancheepuram: true`, augments yield data with
  Kancheepuram rows for years ≤ 2019, tags them `source = "merged"`, logs
  warning with year range and row count.
- Applies season labels where data supports them.
- Returns typed DataFrames (dtypes match schema).

---

### T-015 ⬜ Write `src/data/validate.py`
**Depends on:** T-013, T-014  
**Acceptance criteria:**
- Checks missing value ratio per column; logs `WARNING` if > 5 %.
- Checks plausibility: rainfall ≥ 0, temperature in [−5, 60] °C, price > 0,
  yield > 0, cost > 0.
- Checks year coverage 2010–2023; logs `WARNING` for gaps.
- Does **not** raise exceptions for warnings — pipeline continues.
- All checks exercised in `tests/test_data_pipeline.py`.

---

### T-016 ⬜ Merge features and save `data/processed/features.parquet`
**Depends on:** T-013, T-014, T-015  
**Files created:** `data/processed/features.parquet`  
**Acceptance criteria:**
- Joins weather aggregates, yield, prices, crop_water, cost on `(year, crop,
  season)`.
- Final schema has no duplicate columns.
- Saved as Parquet via PyArrow.
- A `python -c "import pandas as pd; df=pd.read_parquet('data/processed/features.parquet'); print(df.shape)"` call prints a non-zero shape.

---

## Milestone 2 — Prediction Layer

### T-020 ⬜ Write `src/models/yield_model.py`
**Depends on:** T-016  
**Files created:** `models/yield_model.pkl`, `outputs/model_metrics.csv`,
`outputs/plots/yield_actual_vs_pred.png`  
**Acceptance criteria:**
- Implements time-aware train/test split using `config.models.train_test_split_year`.
- Trains `RandomForestRegressor` and `XGBRegressor` with fixed seed.
- Reports MAE, RMSE, R² for both models; saves to `outputs/model_metrics.csv`.
- Produces actual-vs-predicted scatter plot saved as PNG.
- Persists the better model (lower RMSE) to `models/yield_model.pkl`.
- Type hints, docstrings, no data leakage (checked by code review criterion:
  test indices must all be > train indices when sorted by year).

---

### T-021 ⬜ Write `src/models/price_model.py`
**Depends on:** T-016  
**Files created:** `models/price_model.pkl`  
**Acceptance criteria:**
- Aggregates monthly prices to seasonal averages per crop.
- Trains three `GradientBoostingRegressor(loss='quantile')` models for
  quantiles **P10 (0.10), P50 (0.50), P90 (0.90)**.
- Returns `(p10, p50, p90)` as named floats at inference time.
- Persists all three quantile models in a single bundle `models/price_model.pkl`.
- When `use_synthetic.prices: true`, stores a `synthetic=True` flag in model
  metadata so the dashboard can surface the synthetic-data note.

---

## Milestone 3 — Risk Layer

### T-030 ⬜ Write `src/risk/risk_scorer.py`
**Depends on:** T-016  
**Files created:** `data/processed/risk_scores.parquet`  
**Acceptance criteria:**
- Computes CV and dry-year failure probability for each (crop, season) pair
  using definitions in design.md §3.4.
- Normalises both metrics and combined score to [0, 1].
- Saves results to `data/processed/risk_scores.parquet`.
- Unit tests in `tests/test_risk_scorer.py` cover:
  - CV of a constant series = 0.
  - CV of a known series matches hand-calculated value.
  - Dry-year probability of 1.0 when all dry years fail.
  - Dry-year probability of 0.0 when no dry years fail.
  - Edge case: fewer than 2 dry years (returns `NaN` or 0 gracefully).

---

## Milestone 4 — Optimisation Layer

### T-040 ⬜ Write `src/optimization/objectives.py`
**Depends on:** T-020, T-021, T-030  
**Acceptance criteria:**
- `compute_objectives(crop, season, scenario, irrigation_mm, config)` returns
  `(profit, water_deficit, risk_score)` as floats.
- Profit uses predicted yield × predicted price − cost (all from loaded models).
- Water deficit clipped at 0 (no negative deficit).
- Unit tests in `tests/test_objectives.py` cover:
  - Known inputs produce expected outputs (hand-calculated).
  - Water deficit is never negative.
  - Profit sign is correct.

---

### T-041 ⬜ Write `src/optimization/nsga2_optimizer.py`
**Depends on:** T-040  
**Acceptance criteria:**
- Wraps pymoo `NSGA2` with the three objectives defined in T-040.
- Uses population size and generations from config; fixed seed.
- Returns a `pd.DataFrame` of Pareto-optimal crops with columns:
  `[crop, season, profit_inr_ha, water_deficit_mm, risk_score]`.
- All returned solutions are non-dominated (tested in
  `tests/test_nsga2_optimizer.py`).
- Seed reproducibility test: two runs with same seed return identical fronts.

---

### T-042 ⬜ Write `src/optimization/weighted_ranker.py`
**Depends on:** T-041  
**Acceptance criteria:**
- Accepts Pareto DataFrame + weight dict `{profit, water, risk}` (must sum to 1).
- Normalises each objective to [0, 1] across the Pareto set.
- Returns ranked DataFrame sorted by ascending weighted score.
- Raises `ValueError` if weights do not sum to 1 (± 0.01 tolerance).

---

### T-043 ⬜ Write `src/optimization/land_allocator.py`
**Depends on:** T-041  
**Acceptance criteria:**
- Solves the LP from design.md §3.5.4 using `scipy.optimize.linprog`.
- Inputs: total acreage (ha), water budget (mm·ha), list of Pareto crops with
  profit and water requirement.
- Returns allocation DataFrame with columns `[crop, area_ha, area_fraction]`.
- Handles infeasible case (water budget too small) with a clear error message.

---

## Milestone 5 — Streamlit Dashboard

### T-050 ⬜ Write `src/utils/plotting.py`
**Depends on:** T-041  
**Acceptance criteria:**
- `pareto_scatter(df, ...)` returns a `plotly.graph_objects.Figure` with:
  profit on x-axis, water deficit on y-axis, bubble size = risk, crop labels.
- `actual_vs_pred_plot(y_true, y_pred, title)` returns a Plotly Figure.
- `sensitivity_heatmap(df)` returns a Plotly Figure.
- Every function also saves a PNG to `outputs/plots/` via `fig.write_image()`.

---

### T-051 ⬜ Write `app/components/sidebar.py`
**Depends on:** T-001  
**Acceptance criteria:**
- Renders: season selector, irrigation water (mm) numeric input, rainfall
  scenario radio, three weight sliders.
- Weight sliders are linked: when one changes, the others adjust to keep sum = 1.
- Returns a validated `params` dict on button click.
- Shows a warning if weights do not sum to 1 before the user clicks Run.

---

### T-052 ⬜ Write `app/components/pareto_chart.py`, `ranked_table.py`, `explanation.py`
**Depends on:** T-050  
**Acceptance criteria:**
- `pareto_chart.py`: renders the Plotly Pareto scatter inside `st.plotly_chart`.
- `ranked_table.py`: renders styled `st.dataframe` with top row highlighted.
  Shows water-saved vs paddy baseline as a metric below the table.
- `explanation.py`: generates a 3-sentence plain-language explanation for the
  top-ranked crop covering profit, water use, and risk.
- All panels include the disclaimer banner text from FR-UI-03.

---

### T-053 ⬜ Write `app/main.py`
**Depends on:** T-051, T-052, T-041, T-042, T-043  
**Acceptance criteria:**
- Runs with `streamlit run app/main.py` with no import errors.
- Loads config, initialises pipeline on first run, caches model loads with
  `@st.cache_resource`.
- On startup, reads `config.data.use_synthetic`; if any dataset flag is `true`,
  displays a sticky **"⚠ SYNTHETIC DATA"** banner at the top of the page
  listing the affected datasets by name. Banner remains visible on all tabs.
- On button click, calls `nsga2_optimizer`, `weighted_ranker`, optionally
  `land_allocator`, then renders all three output components.
- When `price_model.metadata['synthetic']` is `True`, shows the price-interval
  synthetic-data note below the ranked table.
- Shows spinner while computation runs.
- All four tabs (Pareto, Ranked Table, Water Savings, Backtest preview) are
  present and render without errors on synthetic data.

---

## Milestone 6 — Validation

### T-060 ⬜ Write `src/data/backtest.py`
**Depends on:** T-041, T-042  
**Acceptance criteria:**
- Walk-forward loop over test years (years > `train_test_split_year`).
- For each year: trains models on data up to that year, runs ranker, records
  top recommendation.
- Compares to actual dominant crop (by yield) in `yield.csv`.
- Outputs: hit rate (%), mean profit delta (₹/ha), mean water delta (mm) —
  printed to console and saved to `outputs/backtest_results.csv`.

---

### T-061 ⬜ Write `src/data/sensitivity.py`
**Depends on:** T-042  
**Acceptance criteria:**
- Iterates the grid: rainfall ∈ {−30 %, −10 %, 0, +10 %},
  price ∈ {−20 %, 0, +20 %}, weight presets ∈ {profit-heavy, balanced,
  risk-averse}.
- Records rank of each crop for each combination.
- Saves rank-stability heatmap PNG to `outputs/plots/sensitivity_heatmap.png`.
- Saves raw results to `outputs/sensitivity_results.csv`.

---

## Milestone 7 — Tests, Notebook, and Documentation

### T-070 ⬜ Complete unit test suite
**Depends on:** T-030, T-040, T-041, T-013  
**Files modified:** `tests/test_risk_scorer.py`, `tests/test_objectives.py`,
`tests/test_nsga2_optimizer.py`, `tests/test_data_pipeline.py`  
**Acceptance criteria:**
- `pytest tests/ -v --tb=short` exits with code 0.
- At minimum: all acceptance-criteria tests listed in T-013, T-015, T-030,
  T-040, T-041 are present and passing.
- No test imports from `app/` (dashboard is not unit-tested).

---

### T-071 ⬜ Write `notebooks/01_eda.ipynb`
**Depends on:** T-016  
**Acceptance criteria:**
- Sections: data overview, missing value analysis, yield distributions per crop,
  rainfall time series, price trends, correlation heatmap.
- All cells execute top-to-bottom without errors on synthetic data.
- Saved with output cells cleared (clean for version control).

---

### T-072 ⬜ Write final `README.md`
**Depends on:** all milestones  
**Acceptance criteria:**
- Sections: Project Overview, Architecture (embeds Mermaid diagram from
  design.md), Setup Steps (install, generate data, run app), Data Sources,
  Limitations, Disclaimer, Licence.
- Setup steps tested: following them from a clean Python 3.11 environment
  produces a running Streamlit app.
- Mentions that outputs are indicative estimates.

---

## Summary Table

| Milestone | Tasks | Key Deliverable |
|-----------|-------|-----------------|
| 0 — Scaffold | T-000 – T-002 | Directory structure, config, utilities |
| 1 — Data | T-010 – T-016 | Raw/processed data, synthetic generator, validation |
| 2 — Prediction | T-020 – T-021 | Yield & price models, metrics, plots |
| 3 — Risk | T-030 | Risk scores per crop/season |
| 4 — Optimisation | T-040 – T-043 | NSGA-II Pareto front, ranker, land allocator |
| 5 — Dashboard | T-050 – T-053 | Running Streamlit app |
| 6 — Validation | T-060 – T-061 | Backtest results, sensitivity heatmap |
| 7 — Docs & Tests | T-070 – T-072 | Passing test suite, EDA notebook, README |

**Total tasks: 23**

---

## Execution Order (critical path)

```
T-000 → T-001 → T-002
                  ↓
T-010 → T-011 → T-013 → T-015 → T-016
              → T-014 ↗
              → T-012 (independent, network)
                           ↓
              T-020 → T-021
                           ↓
                        T-030
                           ↓
                        T-040 → T-041 → T-042 → T-043
                                    ↓
                        T-050 → T-051 → T-052 → T-053
                                    ↓
                        T-060, T-061 (parallel)
                                    ↓
                        T-070 → T-071 → T-072
```
