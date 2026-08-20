# SC.8 fU geometry readout — prototypes × ml-1m

(scrutiny working basis only — no thesis quotes; seed 38210573)

- users 6,034 | items 3,125 | K=76 | d=81 | user branch Embedding

## A. M5' set — angle-to-mean / metadata norm / ID share by |H| band

| band | range | users | angle° mean | angle° p50 | ‖m‖ mean |
|---|---|---:|---:|---:|---:|
| q1 | (|H|<=25) | 1,518 | 79.37 | 77.71 | 0.107 |
| q2 | (25<|H|<=56) | 1,513 | 82.16 | 83.75 | 0.126 |
| q3 | (56<|H|<=121) | 1,495 | 83.05 | 84.39 | 0.149 |
| q4 | (121<|H|<=inf) | 1,508 | 88.02 | 88.95 | 0.178 |

## B. Cloud spread & coverage — two weightings (F-DC05-02)

- user_uniform: top-eig share 0.2171, eff rank 6.11, mean resultant length 0.1142, coverage U→P mean max-sim 1.7425
- interaction_weighted: top-eig share 0.2271, eff rank 6.08, mean resultant length 0.1028, coverage U→P mean max-sim 1.7629
- top-eig-share delta (interaction − uniform): +0.0100
- coverage P→U max-sim: mean 1.9995, min p1 1.9990

## C. B(t) channel (comparative instrument, F-DC05-06)

- Var_i(B) = 1.9161; mean per-user Var_i(personalized) = 0.4498; **B variance share = 0.8099** (2000 sampled users)
- corr(B, log pop) Pearson 0.9740; Spearman(B, pop) 0.9785

## D. Gauge / activation spectrum (3b-i)

- activation eff rank 4.44 (numerical rank 29 of 76)
- pinned dim 76 (of d+1=82 max); exact gauge dim 0
- t gauge mass: mean -0.0000, p50 0.0000, p99 0.0000
