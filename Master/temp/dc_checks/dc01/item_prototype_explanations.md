# feature_item_proto — Intrinsic Item-Prototype Explanations

> Generated from the **hm_1_month dev checkpoint** (seed 38210573, TEST HR@10 = 0.6626). Model: feature_item_proto — item factor = Σ of feature-value embeddings, scored by a prototype layer.

> 11 item-prototypes · feature vocabulary = 426 values over 5 fields (department_name, product_type_name, section_name, colour_group_name, graphical_appearance_name).

> **Every number below is read directly from the trained parameters / forward pass — nothing is post-hoc.**


## A. Prototype dictionary — what each item-prototype *means*

For each item-prototype $p_k$, the feature-values most aligned with it, $\cos(e_f, p_k)$ (the model-intrinsic attribute profile, read from parameters alone):

| # | top feature-values (cosine) |
|---|---|
| 0 | All over pattern (+0.73) · Underwear Jersey (+0.72) · Men Underwear (+0.61) · Underwear bottom (+0.54) · Dark Purple (+0.37) · Green (+0.34) |
| 1 | White (+0.56) · Tops Woven (+0.53) · Solid (+0.53) · Divided Collection (+0.52) · Shirt (+0.44) · Off White (+0.29) |
| 2 | Solid (+0.76) · Womens Big accessories (+0.64) · Bag (+0.63) · Bags (+0.58) · Young Boy Socks (+0.38) · Tote bag (+0.37) |
| 3 | Black (+0.94) · Contemporary Smart (+0.45) · Dress-up Boys (+0.38) · Jacket Street (+0.36) · Bucket hat (+0.36) · Shoes (+0.36) |
| 4 | Solid (+0.72) · White (+0.70) · Mixed solid/pattern (+0.35) · Denim Trousers (+0.34) · Melange (+0.31) · Small Accessories (+0.30) |
| 5 | Solid (+0.91) · Young Boy Socks (+0.53) · Mixed solid/pattern (+0.53) · Divided Accessories (+0.45) · Knit & Woven (+0.45) · Hat/beanie (+0.45) |
| 6 | Solid (+0.66) · Womens Small accessories (+0.59) · Jewellery (+0.59) · Gold (+0.57) · Earring (+0.57) · Necklace (+0.50) |
| 7 | Swimwear (+0.71) · White (+0.70) · Womens Swimwear, beachwear (+0.62) · Swimwear bottom (+0.56) · Bikini top (+0.53) · Other structure (+0.41) |
| 8 | Dress (+0.69) · Solid (+0.62) · Divided Collection (+0.48) · Projects Dresses (+0.33) · Melange (+0.30) · Young Boy Socks (+0.28) |
| 9 | Solid (+0.66) · Womens Small accessories (+0.59) · Jewellery (+0.59) · Gold (+0.57) · Earring (+0.57) · Necklace (+0.50) |
| 10 | Solid (+0.76) · Dark Blue (+0.63) · Socks (+0.43) · Men Sport Acc (+0.38) · Braces (+0.31) · Men Accessories (+0.30) |

## B. Per-item attribution — exact additive decomposition

For a handful of prototypes, the item that activates them most strongly, and the EXACT decomposition of that activation over the item's feature rows: $t^*_k = 1\text{ (shift)} + \sum_f c_{f,k}$.


**Prototype #3** — most-activating item: `899155001` (item 10297), $t^*_{3} = 2.000 = 1 + 1.000$

| feature row | contribution $c_{f,k}$ |
|---|---|
| department_name = Accessories | +0.0112 |
| product_type_name = Cap/peaked | +0.0003 |
| section_name = Men Accessories | +0.0571 |
| colour_group_name = Black | +0.8882 |
| graphical_appearance_name = Treatment | +0.0402 |
| <item-ID row> | +0.0029 |
| **Σ (= t\*−1)** | **+0.9999** |

feature-explained +0.9971 vs ID-row +0.0029 → features carry **100%** of the (non-shift) activation.


**Prototype #5** — most-activating item: `798883002` (item 11608), $t^*_{5} = 2.000 = 1 + 1.000$

| feature row | contribution $c_{f,k}$ |
|---|---|
| department_name = Small Accessories | +0.0409 |
| product_type_name = Hair/alice band | +0.0269 |
| section_name = Divided Accessories | +0.1456 |
| colour_group_name = Yellow | +0.0204 |
| graphical_appearance_name = Solid | +0.7670 |
| <item-ID row> | -0.0007 |
| **Σ (= t\*−1)** | **+1.0000** |

feature-explained +1.0007 vs ID-row -0.0007 → features carry **100%** of the (non-shift) activation.


**Prototype #2** — most-activating item: `891771001` (item 10074), $t^*_{2} = 2.000 = 1 + 1.000$

| feature row | contribution $c_{f,k}$ |
|---|---|
| department_name = Bags | +0.1231 |
| product_type_name = Bag | +0.1215 |
| section_name = Womens Big accessories | +0.2131 |
| colour_group_name = Dark Orange | +0.0290 |
| graphical_appearance_name = Solid | +0.5142 |
| <item-ID row> | -0.0008 |
| **Σ (= t\*−1)** | **+1.0000** |

feature-explained +1.0008 vs ID-row -0.0008 → features carry **100%** of the (non-shift) activation.


**Prototype #10** — most-activating item: `858306002` (item 12433), $t^*_{10} = 2.000 = 1 + 1.000$

| feature row | contribution $c_{f,k}$ |
|---|---|
| department_name = Baby Toys/Acc | +0.0031 |
| product_type_name = Accessories set | +0.0013 |
| section_name = Baby Essentials & Complements | +0.0657 |
| colour_group_name = Dark Blue | +0.3622 |
| graphical_appearance_name = Solid | +0.5673 |
| <item-ID row> | +0.0004 |
| **Σ (= t\*−1)** | **+1.0000** |

feature-explained +0.9996 vs ID-row +0.0004 → features carry **100%** of the (non-shift) activation.


## C. A full recommendation explanation (real test pair)

**Why was item `829618004` (item 782) recommended to user 30238?**

- The item's features: department_name=Expressive Lingerie · product_type_name=Bra · section_name=Womens Lingerie · colour_group_name=Light Blue · graphical_appearance_name=Solid.
- Total score **3.491** = item-prototype half (û·t\*) **0.179** + user-prototype half (u\*·t̂) **3.313**.
- The recommendation is led by **item-prototype #1**, whose profile is:
  *White* (+0.56), *Tops Woven* (+0.53), *Solid* (+0.53), *Divided Collection* (+0.52), *Shirt* (+0.44).
- The user's affinity to that prototype is û_1 = +0.040; the item activates it at t\*_1 = 1.415 → contribution **+0.056** to the score.
- **Why does the item activate prototype #1?** t\*_1 = 1 + 0.415, decomposed over its features:

| feature | share of activation |
|---|---|
| department_name=Expressive Lingerie | +0.0253 |
| product_type_name=Bra | +0.0002 |
| section_name=Womens Lingerie | +0.0162 |
| colour_group_name=Light Blue | +0.0630 |
| graphical_appearance_name=Solid | +0.3117 |
| <item-ID row> | -0.0015 |

## Honesty notes (per dc01 §3.4)

- Per-feature shares are **jointly normalized through ‖q_i‖** — exact additive shares of the cosine part, *not* counterfactual effects (removing a feature rescales the others).
- The **+1 shift** is a feature-independent baseline, shown explicitly, never absorbed.
- The **item-prototype (û·t\*) half is feature-grounded**; the user-prototype (u\*·t̂) half is CF-grounded (interpreted post-hoc, as in stock ProtoMF) — they are distinguished above.
- Single-seed dev checkpoint; profiles show the known open risks (M2 'Solid' frequent-value recurrence; M4 near-duplicate prototypes) — see dc01 §3.8.
