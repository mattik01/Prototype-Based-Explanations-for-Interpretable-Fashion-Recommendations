# SC.8 fU geometry readout — prototypes × hm_1_month

(scrutiny working basis only — no thesis quotes; seed 38210573)

- users 73,418 | items 13,651 | K=76 | d=81 | user branch Embedding

## A. M5' set — angle-to-mean / metadata norm / ID share by |H| band

| band | range | users | angle° mean | angle° p50 | ‖m‖ mean |
|---|---|---:|---:|---:|---:|
| q1 | (|H|<=3) | 18,442 | 88.50 | 86.16 | 0.106 |
| q2 | (3<|H|<=5) | 23,124 | 86.80 | 84.14 | 0.102 |
| q3 | (5<|H|<=8) | 16,658 | 84.89 | 82.27 | 0.105 |
| q4 | (8<|H|<=inf) | 15,194 | 82.82 | 80.25 | 0.118 |
| band[3,3] | [3,3] | 18,442 | 88.50 | 86.16 | 0.106 |
| band[4,5] | [4,5] | 23,124 | 86.80 | 84.14 | 0.102 |
| band[6,12] | [6,12] | 25,527 | 84.37 | 81.74 | 0.107 |
| band[13,inf] | [13,inf] | 6,325 | 82.02 | 79.56 | 0.125 |

## B. Cloud spread & coverage — two weightings (F-DC05-02)

- user_uniform: top-eig share 0.3576, eff rank 4.54, mean resultant length 0.0710, coverage U→P mean max-sim 1.7155
- interaction_weighted: top-eig share 0.3649, eff rank 4.38, mean resultant length 0.0914, coverage U→P mean max-sim 1.7198
- top-eig-share delta (interaction − uniform): +0.0072
- coverage P→U max-sim: mean 1.9997, min p1 1.9982

## C. B(t) channel (comparative instrument, F-DC05-06)

- Var_i(B) = 0.8791; mean per-user Var_i(personalized) = 0.3232; **B variance share = 0.7312** (2000 sampled users)
- corr(B, log pop) Pearson 0.9601; Spearman(B, pop) 0.9421

## D. Gauge / activation spectrum (3b-i)

- activation eff rank 1.82 (numerical rank 29 of 76)
- pinned dim 76 (of d+1=82 max); exact gauge dim 0
- t gauge mass: mean 0.0000, p50 0.0000, p99 0.0000
