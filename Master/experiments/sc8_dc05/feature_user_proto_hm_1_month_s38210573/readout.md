# SC.8 fU geometry readout — feature_user_proto × hm_1_month

(scrutiny working basis only — no thesis quotes; seed 38210573)

- users 73,418 | items 13,651 | K=76 | d=81 | user branch HistoryFeatureEmbedding (ids)

## A. M5' set — angle-to-mean / metadata norm / ID share by |H| band

| band | range | users | angle° mean | angle° p50 | ‖m‖ mean | ID-share mean |
|---|---|---:|---:|---:|---:|---:|
| q1 | (|H|<=3) | 18,442 | 83.32 | 81.23 | 0.484 | 0.009 |
| q2 | (3<|H|<=5) | 23,124 | 82.04 | 79.53 | 0.446 | 0.013 |
| q3 | (5<|H|<=8) | 16,658 | 79.73 | 76.81 | 0.406 | 0.022 |
| q4 | (8<|H|<=inf) | 15,194 | 76.32 | 72.49 | 0.356 | 0.039 |
| band[3,3] | [3,3] | 18,442 | 83.32 | 81.23 | 0.484 | 0.009 |
| band[4,5] | [4,5] | 23,124 | 82.04 | 79.53 | 0.446 | 0.013 |
| band[6,12] | [6,12] | 25,527 | 79.12 | 75.98 | 0.394 | 0.025 |
| band[13,inf] | [13,inf] | 6,325 | 74.02 | 69.87 | 0.332 | 0.050 |

## B. Cloud spread & coverage — two weightings (F-DC05-02)

- user_uniform: top-eig share 0.3677, eff rank 3.12, mean resultant length 0.1536, coverage U→P mean max-sim 1.7679
- interaction_weighted: top-eig share 0.3672, eff rank 3.15, mean resultant length 0.1906, coverage U→P mean max-sim 1.7693
- top-eig-share delta (interaction − uniform): -0.0005
- coverage P→U max-sim: mean 1.9996, min p1 1.9988

## C. B(t) channel (comparative instrument, F-DC05-06)

- Var_i(B) = 0.9842; mean per-user Var_i(personalized) = 0.3493; **B variance share = 0.7381** (2000 sampled users)
- corr(B, log pop) Pearson 0.9320; Spearman(B, pop) 0.8980

## D. Gauge / activation spectrum (3b-i)

- activation eff rank 1.79 (numerical rank 10 of 76)
- pinned dim 76 (of d+1=82 max); exact gauge dim 0
- t gauge mass: mean 0.0000, p50 0.0000, p99 0.0000

## E. Profile instruments (M1'/M2'/M4')

- profile softmax entropy: mean 5.912 (uniform ref 6.054)
- top1 cos: mean 0.9925; top1−top2 margin mean 0.0068
- profile pairwise overlap cos: mean 0.4987, p99 1.0000
- proto pairwise latent cos: mean 0.4520, p99 1.0000
- vote-share vs basket-freq Pearson: 0.3347

### Per-value vote share vs basket frequency (top-15 by freq)

| value | basket freq | mean vote share |
|---|---:|---:|
| graphical_appearance_name=Solid | 0.965 | 0.0771 |
| colour_group_name=Black | 0.835 | 0.0250 |
| section_name=Womens Everyday Collection | 0.567 | 0.1386 |
| product_type_name=Trousers | 0.491 | 0.0335 |
| product_type_name=Sweater | 0.428 | 0.0189 |
| section_name=Divided Collection | 0.400 | 0.1010 |
| colour_group_name=White | 0.380 | 0.0309 |
| graphical_appearance_name=Melange | 0.349 | 0.0221 |
| department_name=Knitwear | 0.334 | 0.1080 |
| product_type_name=Top | 0.306 | 0.0097 |
| graphical_appearance_name=Denim | 0.285 | 0.0371 |
| section_name=Womens Tailoring | 0.280 | 0.1445 |
| graphical_appearance_name=All over pattern | 0.273 | 0.0576 |
| colour_group_name=Beige | 0.258 | 0.0309 |
| product_type_name=Dress | 0.248 | 0.1187 |

## F. Member-user purchase-lift vs intrinsic profiles (F-DC05-03a/-20)

- Spearman(lift, intrinsic) over 76 prototypes: mean 0.061, p25 -0.001, p75 0.163 (k=50 members)
- top-5 intrinsic descriptors found in top-10 member-lift: mean 2.37 of 5 (p25 3, p75 3)

| proto | ρ | top-5 intrinsic | top-5 member-lift |
|---|---:|---|---|
| 46 | -0.002 | section_name=Ladies H&M Sport; department_name=Ladies Sport Bras; department_name=Ladies Sport Bottoms; product_type_name=Sweater; section_name=Womens Nightwear, Socks & Tigh | department_name=Ladies Sport Bras; section_name=Ladies H&M Sport; department_name=Ladies Sport Bottoms; graphical_appearance_name=Mesh; product_type_name=Leggings/Tights |
| 24 | -0.001 | section_name=Ladies H&M Sport; department_name=Ladies Sport Bras; department_name=Ladies Sport Bottoms; product_type_name=Sweater; section_name=Womens Nightwear, Socks & Tigh | department_name=Ladies Sport Bras; section_name=Ladies H&M Sport; department_name=Ladies Sport Bottoms; graphical_appearance_name=Mesh; colour_group_name=Green |
| 57 | -0.001 | section_name=Ladies H&M Sport; department_name=Ladies Sport Bras; department_name=Ladies Sport Bottoms; product_type_name=Sweater; section_name=Womens Nightwear, Socks & Tigh | department_name=Ladies Sport Bras; section_name=Ladies H&M Sport; department_name=Ladies Sport Bottoms; graphical_appearance_name=Mesh; colour_group_name=Green |
| 51 | 0.172 | department_name=Trouser; colour_group_name=Beige; department_name=Divided+ inactive from s.1; graphical_appearance_name=Mixed solid/pattern; colour_group_name=Off White | department_name=Suit; department_name=Skirt; section_name=Womens Tailoring; graphical_appearance_name=Other structure; colour_group_name=Dark Turquoise |
| 66 | 0.296 | section_name=Divided Projects; department_name=Suit jacket; section_name=Womens Trend; department_name=Knit & Woven; department_name=Jewellery Extended | department_name=Boots; department_name=OL Extended Sizes; department_name=Tops Knitwear; product_type_name=Boots; department_name=Trousers |
