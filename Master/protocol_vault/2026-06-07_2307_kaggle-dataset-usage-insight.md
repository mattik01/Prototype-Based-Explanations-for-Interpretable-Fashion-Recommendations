---
date: 2026-06-07
time: "23:07"
phase: 3
---
# Kaggle H&M solutions used short recent windows, not the full 2 years

Reviewed top Kaggle competition solutions (`Master/docs/kaggle_sources/`). Key finding: competitors trained and generated candidates on short recent windows (~4–5 weeks; 30-day item "whitelisting" already covers ~90% of holdout-week purchases; ~20% of items carry ~90% of activity), using the full 2-year history mainly for engineered features — recency dominates next-week prediction. This empirically justifies building a short-window H&M variant for our experiments rather than using the full dataset.

[[data]] [[evaluation]] [[phase-3]]
