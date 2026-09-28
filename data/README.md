# Data

This folder is intentionally close to empty in the repository — the raw and
intermediate CSVs are large (50–100 MB each) and are excluded via
`.gitignore`.

- `raw/` — put `Retail_demand_forecasting_dataset.csv` here (the original
  source file) before running Phase 1.
- `processed/` — every pipeline phase reads its input from here and writes
  its output CSV back here. Run the phases in order (see root `README.md`)
  and this folder fills in automatically:

```
data/processed/
├── Retail_demand_forecasting_CLEANED.csv      (after Phase 1)
├── Retail_demand_outliers_TREATED.csv         (after Phase 4)
├── Retail_demand_YJ_transformed.csv           (after Phase 7)
├── Retail_demand_FE.csv                       (after Phase 8)
└── Retail_demand_FS.csv                       (after Phase 9 — used by Phase 10 / the app)
```
