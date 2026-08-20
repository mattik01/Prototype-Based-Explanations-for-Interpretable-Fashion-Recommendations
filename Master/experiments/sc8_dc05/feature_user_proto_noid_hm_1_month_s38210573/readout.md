# SC.8 fU geometry readout — feature_user_proto × hm_1_month

(scrutiny working basis only — no thesis quotes; seed 38210573)

- users 73,418 | items 13,651 | K=76 | d=81 | user branch HistoryFeatureEmbedding (noid)

## A. M5' set — angle-to-mean / metadata norm / ID share by |H| band

| band | range | users | angle° mean | angle° p50 | ‖m‖ mean |
|---|---|---:|---:|---:|---:|
| q1 | (|H|<=3) | 18,442 | 81.35 | 81.03 | 0.431 |
| q2 | (3<|H|<=5) | 23,124 | 79.25 | 77.50 | 0.399 |
| q3 | (5<|H|<=8) | 16,658 | 77.04 | 73.90 | 0.364 |
| q4 | (8<|H|<=inf) | 15,194 | 73.58 | 68.86 | 0.323 |
| band[3,3] | [3,3] | 18,442 | 81.35 | 81.03 | 0.431 |
| band[4,5] | [4,5] | 23,124 | 79.25 | 77.50 | 0.399 |
| band[6,12] | [6,12] | 25,527 | 76.42 | 73.05 | 0.355 |
| band[13,inf] | [13,inf] | 6,325 | 71.24 | 65.84 | 0.303 |

## B. Cloud spread & coverage — two weightings (F-DC05-02)

- user_uniform: top-eig share 0.3463, eff rank 3.07, mean resultant length 0.1813, coverage U→P mean max-sim 1.8539
- interaction_weighted: top-eig share 0.3541, eff rank 3.05, mean resultant length 0.2132, coverage U→P mean max-sim 1.8555
- top-eig-share delta (interaction − uniform): +0.0078
- coverage P→U max-sim: mean 1.9999, min p1 1.9997

## C. B(t) channel (comparative instrument, F-DC05-06)

- Var_i(B) = 0.8284; mean per-user Var_i(personalized) = 0.3516; **B variance share = 0.7020** (2000 sampled users)
- corr(B, log pop) Pearson 0.9388; Spearman(B, pop) 0.9169

## D. Gauge / activation spectrum (3b-i)

- activation eff rank 2.20 (numerical rank 10 of 76)
- pinned dim 76 (of d+1=82 max); exact gauge dim 0
- t gauge mass: mean 0.0000, p50 0.0000, p99 0.0000

## E. Profile instruments (M1'/M2'/M4')

- profile softmax entropy: mean 5.916 (uniform ref 6.054)
- top1 cos: mean 0.9907; top1−top2 margin mean 0.0098
- profile pairwise overlap cos: mean 0.3054, p99 1.0000
- proto pairwise latent cos: mean 0.2464, p99 1.0000
- vote-share vs basket-freq Pearson: 0.3100

### Per-value vote share vs basket frequency (top-15 by freq)

| value | basket freq | mean vote share |
|---|---:|---:|
| graphical_appearance_name=Solid | 0.965 | 0.0742 |
| colour_group_name=Black | 0.835 | 0.0188 |
| section_name=Womens Everyday Collection | 0.567 | 0.1431 |
| product_type_name=Trousers | 0.491 | 0.0423 |
| product_type_name=Sweater | 0.428 | 0.0340 |
| section_name=Divided Collection | 0.400 | 0.0730 |
| colour_group_name=White | 0.380 | 0.0153 |
| graphical_appearance_name=Melange | 0.349 | 0.0185 |
| department_name=Knitwear | 0.334 | 0.0872 |
| product_type_name=Top | 0.306 | 0.0222 |
| graphical_appearance_name=Denim | 0.285 | 0.0274 |
| section_name=Womens Tailoring | 0.280 | 0.1472 |
| graphical_appearance_name=All over pattern | 0.273 | 0.0500 |
| colour_group_name=Beige | 0.258 | 0.0270 |
| product_type_name=Dress | 0.248 | 0.1303 |

## F. Member-user purchase-lift vs intrinsic profiles (F-DC05-03a/-20)

- Spearman(lift, intrinsic) over 76 prototypes: mean 0.065, p25 -0.020, p75 0.194 (k=50 members)
- top-5 intrinsic descriptors found in top-10 member-lift: mean 2.18 of 5 (p25 1, p75 3)

| proto | ρ | top-5 intrinsic | top-5 member-lift |
|---|---:|---|---|
| 6 | -0.024 | section_name=Womens Lingerie; department_name=Nightwear; department_name=Expressive Lingerie; department_name=Clean Lingerie; product_type_name=Hair ties | department_name=Expressive Lingerie; department_name=Clean Lingerie; section_name=Womens Lingerie; product_type_name=Unknown; product_type_name=Bra |
| 64 | -0.020 | section_name=Womens Lingerie; department_name=Nightwear; department_name=Expressive Lingerie; department_name=Clean Lingerie; product_type_name=Hair ties | department_name=Expressive Lingerie; department_name=Clean Lingerie; section_name=Womens Lingerie; product_type_name=Unknown; product_type_name=Bra |
| 8 | -0.020 | section_name=Womens Lingerie; department_name=Nightwear; department_name=Expressive Lingerie; department_name=Clean Lingerie; product_type_name=Hair ties | department_name=Expressive Lingerie; department_name=Clean Lingerie; section_name=Womens Lingerie; product_type_name=Unknown; product_type_name=Bra |
| 3 | 0.195 | department_name=Trouser S&T; product_type_name=Dog Wear; graphical_appearance_name=Treatment; department_name=Outwear; department_name=Kids Boy Knitwear | department_name=Scarves; product_type_name=Blouse; department_name=Blouse; product_type_name=Scarf; section_name=Womens Premium |
| 70 | 0.264 | department_name=Outwear; department_name=Kids Boy Knitwear; department_name=Knitwear; product_type_name=Top; department_name=Shoes Other | section_name=Womens Jackets; graphical_appearance_name=Treatment; department_name=Outwear; department_name=Suit jacket; department_name=Knitwear |
