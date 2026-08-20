# SC.8 fU geometry readout — feature_user_proto × ml-1m

(scrutiny working basis only — no thesis quotes; seed 38210573)

- users 6,034 | items 3,125 | K=76 | d=81 | user branch HistoryFeatureEmbedding (noid)

## A. M5' set — angle-to-mean / metadata norm / ID share by |H| band

| band | range | users | angle° mean | angle° p50 | ‖m‖ mean |
|---|---|---:|---:|---:|---:|
| q1 | (|H|<=25) | 1,518 | 72.55 | 70.61 | 1.470 |
| q2 | (25<|H|<=56) | 1,513 | 73.72 | 71.01 | 1.404 |
| q3 | (56<|H|<=121) | 1,495 | 71.25 | 69.16 | 1.174 |
| q4 | (121<|H|<=inf) | 1,508 | 70.67 | 71.26 | 0.849 |

## B. Cloud spread & coverage — two weightings (F-DC05-02)

- user_uniform: top-eig share 0.3222, eff rank 3.73, mean resultant length 0.2822, coverage U→P mean max-sim 1.7975
- interaction_weighted: top-eig share 0.3537, eff rank 3.64, mean resultant length 0.2805, coverage U→P mean max-sim 1.7931
- top-eig-share delta (interaction − uniform): +0.0315
- coverage P→U max-sim: mean 1.9992, min p1 1.9968

## C. B(t) channel (comparative instrument, F-DC05-06)

- Var_i(B) = 1.8711; mean per-user Var_i(personalized) = 0.5185; **B variance share = 0.7830** (2000 sampled users)
- corr(B, log pop) Pearson 0.9602; Spearman(B, pop) 0.9596

## D. Gauge / activation spectrum (3b-i)

- activation eff rank 2.46 (numerical rank 10 of 76)
- pinned dim 76 (of d+1=82 max); exact gauge dim 0
- t gauge mass: mean -0.0000, p50 0.0000, p99 0.0000

## E. Profile instruments (M1'/M2'/M4')

- profile softmax entropy: mean 6.858 (uniform ref 6.967)
- top1 cos: mean 0.9855; top1−top2 margin mean 0.0078
- profile pairwise overlap cos: mean 0.2799, p99 1.0000
- proto pairwise latent cos: mean 0.2448, p99 1.0000
- vote-share vs basket-freq Pearson: 0.4571

### Per-value vote share vs basket frequency (top-15 by freq)

| value | basket freq | mean vote share |
|---|---:|---:|
| tags=original | 0.998 | 0.0611 |
| tags=imdb top 250 | 0.996 | 0.0480 |
| genres=Drama | 0.996 | 0.0306 |
| tags=great acting | 0.990 | 0.0158 |
| tags=oscar (best directing) | 0.988 | 0.0229 |
| genres=Comedy | 0.988 | 0.0260 |
| tags=excellent script | 0.987 | 0.0121 |
| tags=great movie | 0.987 | 0.0171 |
| tags=great ending | 0.986 | 0.0087 |
| tags=drama | 0.979 | 0.0095 |
| tags=storytelling | 0.976 | 0.0084 |
| tags=mentor | 0.974 | 0.0146 |
| tags=good | 0.973 | 0.0105 |
| tags=oscar (best picture) | 0.968 | 0.0070 |
| tags=masterpiece | 0.966 | 0.0089 |

## F. Member-user purchase-lift vs intrinsic profiles (F-DC05-03a/-20)

- Spearman(lift, intrinsic) over 76 prototypes: mean 0.324, p25 0.328, p75 0.337 (k=50 members)
- top-5 intrinsic descriptors found in top-10 member-lift: mean 0.04 of 5 (p25 0, p75 0)

| proto | ρ | top-5 intrinsic | top-5 member-lift |
|---|---:|---|---|
| 9 | 0.242 | genres=Sci-Fi; tags=kubrick; tags=medieval; tags=visual; tags=guns | tags=george orwell; tags=intelligent; tags=intelligent sci-fi; tags=zombies; tags=better than the american version |
| 15 | 0.328 | tags=family; tags=series; tags=shopping; tags=historical; tags=romance | tags=tolkien; tags=dogs; tags=dog; tags=simple; tags=peter pan |
| 16 | 0.328 | tags=family; tags=series; tags=shopping; tags=historical; tags=romance | tags=tolkien; tags=dogs; tags=dog; tags=simple; tags=peter pan |
| 37 | 0.341 | tags=super-hero; tags=goth; tags=very funny; tags=confrontational; tags=great acting | tags=skinhead; tags=neo-nazis; tags=poker; tags=confrontational; tags=gypsy accent |
| 53 | 0.375 | tags=interesting; tags=excellent; tags=male nudity; tags=multiple storylines; tags=factual | tags=terminal illness; tags=crazy; tags=sex comedy; tags=stupid; tags=alcoholism |
