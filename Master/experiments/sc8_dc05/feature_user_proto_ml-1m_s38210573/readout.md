# SC.8 fU geometry readout — feature_user_proto × ml-1m

(scrutiny working basis only — no thesis quotes; seed 38210573)

- users 6,034 | items 3,125 | K=76 | d=81 | user branch HistoryFeatureEmbedding (ids)

## A. M5' set — angle-to-mean / metadata norm / ID share by |H| band

| band | range | users | angle° mean | angle° p50 | ‖m‖ mean | ID-share mean |
|---|---|---:|---:|---:|---:|---:|
| q1 | (|H|<=25) | 1,518 | 72.65 | 71.49 | 1.084 | 0.003 |
| q2 | (25<|H|<=56) | 1,513 | 73.97 | 75.05 | 1.022 | 0.005 |
| q3 | (56<|H|<=121) | 1,495 | 74.17 | 74.34 | 0.856 | 0.013 |
| q4 | (121<|H|<=inf) | 1,508 | 78.25 | 79.68 | 0.614 | 0.070 |

## B. Cloud spread & coverage — two weightings (F-DC05-02)

- user_uniform: top-eig share 0.2742, eff rank 4.60, mean resultant length 0.2436, coverage U→P mean max-sim 1.6996
- interaction_weighted: top-eig share 0.2833, eff rank 4.68, mean resultant length 0.2088, coverage U→P mean max-sim 1.6823
- top-eig-share delta (interaction − uniform): +0.0090
- coverage P→U max-sim: mean 1.9987, min p1 1.9975

## C. B(t) channel (comparative instrument, F-DC05-06)

- Var_i(B) = 1.9463; mean per-user Var_i(personalized) = 0.4770; **B variance share = 0.8032** (2000 sampled users)
- corr(B, log pop) Pearson 0.9644; Spearman(B, pop) 0.9667

## D. Gauge / activation spectrum (3b-i)

- activation eff rank 2.63 (numerical rank 11 of 76)
- pinned dim 76 (of d+1=82 max); exact gauge dim 0
- t gauge mass: mean 0.0000, p50 0.0000, p99 0.0000

## E. Profile instruments (M1'/M2'/M4')

- profile softmax entropy: mean 6.888 (uniform ref 6.967)
- top1 cos: mean 0.9614; top1−top2 margin mean 0.0130
- profile pairwise overlap cos: mean 0.3197, p99 1.0000
- proto pairwise latent cos: mean 0.2720, p99 1.0000
- vote-share vs basket-freq Pearson: 0.4437

### Per-value vote share vs basket frequency (top-15 by freq)

| value | basket freq | mean vote share |
|---|---:|---:|
| tags=original | 0.998 | 0.0642 |
| tags=imdb top 250 | 0.996 | 0.0394 |
| genres=Drama | 0.996 | 0.0457 |
| tags=great acting | 0.990 | 0.0152 |
| tags=oscar (best directing) | 0.988 | 0.0212 |
| genres=Comedy | 0.988 | 0.0346 |
| tags=excellent script | 0.987 | 0.0120 |
| tags=great movie | 0.987 | 0.0161 |
| tags=great ending | 0.986 | 0.0087 |
| tags=drama | 0.979 | 0.0111 |
| tags=storytelling | 0.976 | 0.0103 |
| tags=mentor | 0.974 | 0.0134 |
| tags=good | 0.973 | 0.0106 |
| tags=oscar (best picture) | 0.968 | 0.0038 |
| tags=masterpiece | 0.966 | 0.0089 |

## F. Member-user purchase-lift vs intrinsic profiles (F-DC05-03a/-20)

- Spearman(lift, intrinsic) over 76 prototypes: mean 0.427, p25 0.420, p75 0.432 (k=50 members)
- top-5 intrinsic descriptors found in top-10 member-lift: mean 0.00 of 5 (p25 0, p75 0)

| proto | ρ | top-5 intrinsic | top-5 member-lift |
|---|---:|---|---|
| 59 | 0.414 | tags=cancer; tags=cult film; tags=sci fi; tags=religion; tags=cult classic | tags=james bond; tags=tolkien; tags=007; tags=bond; tags=nuclear bomb |
| 41 | 0.420 | tags=destiny; tags=touching; tags=romance; tags=family bonds; tags=children | tags=figure skating; tags=shopping; tags=sappy; tags=ballet; tags=predictable |
| 0 | 0.420 | tags=destiny; tags=touching; tags=romance; tags=family bonds; tags=children | tags=figure skating; tags=shopping; tags=sappy; tags=ballet; tags=predictable |
| 16 | 0.432 | tags=oscar (best writing - screenplay written directly for the screen); tags=british comedy; tags=paranoid; tags=imdb top 250; tags=time loop | tags=world war i; tags=moody; tags=train; tags=screwball; tags=south africa |
| 44 | 0.491 | tags=mental illness; tags=manipulation; tags=isolation; tags=great acting; tags=realistic action | tags=gambling; tags=casino; tags=hard to watch; tags=morality; tags=prohibition |
