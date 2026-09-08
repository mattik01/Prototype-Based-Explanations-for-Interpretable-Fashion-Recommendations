# Angle E — what each menu form does to the prototype layer and the explanation contract (2026-09-07)

Menu fixed (MENU.md M1–M16). Host recap: q_u = Σ_k c_{u,k} r_k (rows r_k = word rows e_f, optionally e_ID(u)),
activation a_{u,l} = 1 + cos(q_u, p_l) = 1 + Σ_k c_{u,k} ⟨r_k, p_l⟩ / (‖q_u‖‖p_l‖), score = Σ_l t_{i,l} a_{u,l}.
Two contract facts follow directly and are used throughout:
- **Naming** is intrinsic because p_l lives in the row space and every row that can move a_{u,l} has a word label;
  `cos(e_f, p_l)` (`utilities/explanations/feature_readout.py`, `pipeline.py` "intrinsic_user") is a *complete* name
  only while the row families composing q_u are all nameable. A name is "faithful" when the rows it is computed over
  carry the activation mass; the measurable check is the **name-coverage share** Σ_{word rows}|contrib| / Σ_{all rows}|contrib| per prototype.
- **Attribution** is exact for any linear composition: shares of a_{u,l}−1 are c_{u,k}⟨r_k,p_l⟩ divided by the same
  per-user scalar ‖q_u‖‖p_l‖, so any new row family joins the existing shares and they still sum to one. What the shared
  scalar does NOT give is *fairness* across families: free rows (e_ID, item-ID history rows) choose their own norm, so an
  exact share can still be an optimizer-chosen share (the norm handover already measured for e_ID, sc05 draft l.290–296).
Measured pathology to judge against: activation-cloud effective rank hm fU 1.79 / noid 2.20 / host 1.82, ml-1m 2.63 /
2.46 / 4.44; profile entropy ≈ uniform, p99 pairwise overlap 1.0, ~5–6 distinct profiles of 76; coverage regularizer
effectively off in the winning config (sim_batch 0.0018; at 5.7 a probe gave 56/58 distinct names) — `sc05_chapter_draft.md` l.281–289.

## 1. Per-form table

Columns: naming intact? · attribution exact? · common direction (R removes / L leaves / F feeds) · plausibly moves the
prototype-layer loss (fU vs lightfm_hist −0.06…−0.09)? · sentence licensed.

| # | Form | Naming intact? | Attribution exact? | Common dir. | Moves proto-layer loss? | Sentence licensed |
|---|---|---|---|---|---|---|
| M1 | words + item-ID history rows y_j | **Partial.** `cos(e_f,p_l)` still computes but the y_j rows live in the same space with no label; a prototype whose mass comes from y_j rows wears a word name it did not earn. Needs name-coverage share disclosed per prototype. | Yes; unit = purchase → {words, "this item itself"}; shares comparable within user, but y_j norms are free (second handover, A-F4) | L → F (words keep μ; popular-item rows can learn a shared direction under Gao's mechanism) | No (composition only; lifts lightfm_hist, gap stays) | "Prototype *Casual knitwear*: 25 % from your cardigans' attributes, 15 % from those exact articles." |
| M2 | leave-one-out pooling in training | Yes (rows unchanged) | Yes (unchanged) | L (drops one item's words, μ remains; at \|H\|=1 q_u → e_ID only) | **Plausibly yes** — removes the 20 % self-term at \|H\|=5 and the train/test mismatch that hits *both* layers; cheapest run on the menu | none new |
| M3 | target-conditioned attention over purchases | Rows unchanged so p_l names hold; but the **user's profile becomes per-candidate** — "your prototype bars" is no longer a user object; the explanation object changes kind | Yes given the weights (coefficients of an exact sum); the weights themselves have no nameable reason | L (convex re-weighting of vectors that all contain μ keeps μ) | Plausibly yes on ml-1m (NAIS>FISM precedent), little to weight on hm | "For this item we weighted your purchase of X 3× more than Y" — disclosed as weight, not as why (see F5) |
| M4 | prototype score + additive item-item term | Yes (prototype path untouched) | Yes: two exact sums in score units, disclosed side by side; shares across the two objects are score shares, not cosine shares | L for the prototype path; risk that the bypass makes the prototype path decorative (gradient pressure moves to the dot term) | Moves the **score** (imports the dot piece that wins); does not repair the prototype layer — must be read out as "how much of the score bypasses the prototypes" | "Because you bought X and Y (+0.31 direct) and because you match *Basics* and *Denim* (+0.52)." |
| M5 | conjunction tokens (black∧dress), shared with fI | **Yes, richer**: p_l also nameable by `cos(e_{a∧b},p_l)`; conjunction names read as phrases | Yes (more words) | L/F: the frequent conjunctions carry their own constant; expected constant share of the mean unchanged, row count up | No (composition) | "Because you bought black dresses and dark denim trousers." |
| M6 | attribute-level association-rule tokens | Rule rows are labelled, but the label is a population statement ("A∧B → C-type", conf. c); a prototype named by rules reads as "shoppers who …" — legible only with the two-part wording (B-F1) | Yes; confidence disclosed as the weight | **F**: high-support rules fire on the catalog-constant antecedents, so rule mass is maximally aligned with μ | No | "You bought A and B; N % of shoppers who did also bought C-type items." |
| M7 | co-purchase-subgraph re-weighting of own purchases | Yes | Yes, per purchase with a disclosed multiplier | **F**: population co-purchase degree upweights popular items, the ones closest to μ | No | "We counted X more: it goes with 3 other things you bought." |
| M8 | (purchase, target) complement term beside the prototypes | Yes (prototype path untouched) | Yes, per (purchase, target) pair; second object | L | Moves the score, not the prototype layer (same status as M4) | "Because you bought X, which is usually bought with items like this." |
| M9 | population-mean centring (fU′) ± count shrink | **Yes, and sharper**: rows are still e_f, so `cos(e_f,p_l)` holds, but its meaning becomes *signed* — positive = "over-indexes on f", negative = "under-indexes"; names must be rendered with sign | Yes; word credit becomes deviation credit; the crowd vector is the disclosed removed unit (no residual crowd term in the cosine) | **R** — the only form that removes the input-inherited direction (Mu & Viswanath; SIF step 2) | **Yes, mechanistically**: the cosine layer receives a cloud without the shared direction, which is exactly what eff. rank 1.8–2.6 and duplicate profiles indicate. Caveat: the *training-created* cone (Gao) can re-form on the rows; keep the coverage regularizer as the second lever | "Compared with the average shopper you buy far more *black* and *skirts*; that is what matches this item." |
| M10 | Dirichlet shrink toward the crowd (noid only) | Yes | Yes, crowd share (1−λ_u) is a named term | **F by construction** — mixes μ back in, strongest for thin users; the anti-M9 | No; expected to deepen collapse in `_noid` | "With only 5 purchases we lean half on what shoppers like you buy." |
| M11 | per-user-per-field concentration gate | Yes | Yes per word; weight cites the field's spread — and the weight's reason IS nameable ("4 of 5 black"), unlike M3 | L (gate reweights fields, μ persists); on hm it systematically upweights the field every user is concentrated in, which is the section/department field | No | "You are very consistent about *colour* (4 of 5 black), so colour drives this match." |
| M12 | top-k sparsified profile (Balog coverage×significance) | Yes | Yes over kept words; 0/1 gate is a data-dependent function | **R if significance-vs-population drives selection; F if coverage does** (coverage keeps the commonest words). At \|H\|=5, k ≥ #distinct words → no-op (D3) | No on hm; marginal on ml-1m | "Your profile is *ladieswear, black, trousers* — covering 4 of your 5 purchases." |
| M13 | saturating per-(user,word) counts (EFM) | Yes | Yes | L on hm (linear at n ≤ 5); mild F on ml-1m (flattens repeated words → profiles closer to presence sets, i.e. closer to each other) | No | "Repeated purchases count more, with diminishing returns." |
| M14 | two-vocabulary concat, block-structured prototypes | **Split**: word block of p_l intrinsically named; the ID block can only be named post-hoc by nearest items — the ProtoMF default the lineage rejects. Block shares are exact (joint ‖q_u‖), so the name comes with a disclosed "x % of this prototype is unnamed" | Yes per block | Word block L; ID block starts μ-free (mean of 5 free rows) but can grow a shared direction under training | No (composition) | "Because you bought *this exact item* (beyond its attributes) and items like it." |
| M15 | EFM-style direct attribute-match term beside the prototypes | Yes (prototype path untouched) | Yes; two objects | L; same decorative-prototype risk as M4 | Moves the score (a word-restricted lightfm_hist piece), not the prototype layer | "You might be interested in *black jersey*, which this product is." |
| M16 | disclosed ID-share cap ‖e_ID‖ ≤ α‖q_words‖ | Yes | Yes (already) | **F** — e_ID is today the only μ-free component of q_u; capping it raises μ's share of every q_u. Neutral only if paired with M9 | Alone: likely negative on collapse; with M9: neutral/positive | "At most 30 % of your profile is your personal ID row; the rest is these words." |

## 2. Findings

1. **Only M9 removes the common direction; M6, M7, M10, M16 feed it; the rest leave it.** The user-side collapse is the
   cosine layer's reading of an input cloud that shares one direction (the item-side 85 %-constant composition is the same
   object, brief given). Mu & Viswanath's remedy is literally "eliminate the common mean vector and a few top dominating
   directions" [verified abstract]; SIF's second step removes "the top singular direction" [verified via offconvex]; Gao et al.
   describe the *training-side* cause — embeddings "degenerate and be distributed into a narrow cone" under likelihood training
   [verified abstract]. fU has both causes: μ comes in through the mean (M9 addresses it) and can re-form on the rows through
   training (only the coverage regularizer addresses that; sc05 probe 56/58 names at sim_batch 5.7). Consequence for the
   loss: M9 is the single composition-side form with a mechanism for the prototype-layer gap; M4/M8/M15 move the total
   score by importing the winning dot-product piece but leave the prototype layer as it is and put a bypass beside it.
2. **M16 is a trap without M9.** The free ID row is currently the escape route from μ (the norm handover is the optimizer
   using it). Capping it pushes users back onto the shared direction. Sequence: centre first, cap second.
3. **M10 and M9 are contradictory; M12's sign depends on the selection criterion.** Shrinking toward the crowd is
   adding μ back; coverage-based selection keeps the μ-words, significance-based selection keeps the deviations.
4. **Item-ID history rows (M1, M14) break the completeness of intrinsic naming, not its computability.** `cos(e_f,p_l)`
   still yields a name, but the prototype now has an unnamed channel in the same space; the failure mode is a word-named
   prototype whose activation is carried by ID rows (the item-side fI lesson, now structural and per purchase). Zhang &
   Chen's distinction between explanations that "directly come from an explainable model" and post-hoc ones [verified
   abstract] is exactly what M14's ID block falls on the wrong side of. Minimum disclosure if either is run: per-prototype
   name-coverage share and per-user ID-family share; pair every run with its `_noid` twin (A-F5).
5. **M3 vs M11 under the attention-faithfulness debate.** Jain & Wallace: attention weights "largely do not" provide
   meaningful explanations; different distributions can yield the same prediction [verified abstract]. Wiegreffe & Pinter:
   the verdict depends on the definition, and propose a uniform-weights baseline, seed-variance calibration, frozen-weight
   diagnostics and adversarial attention [verified abstract]. Transfer: (i) in fU the weights multiply an *exact linear* sum,
   so share attribution is exact by construction — the faithfulness question is only "is the weight's reason nameable";
   (ii) W&P's uniform baseline IS today's fU, so M3 carries its own falsification test — if it does not beat the mean, its
   weights are not even useful, let alone explanatory; (iii) an adversarial-weights check (alternative a_{ij} with the same
   score) is cheap in a linear model and should ship with any M3 run; (iv) M11's gate is a deterministic function of the
   user's own counts, so its reason is nameable ("4 of 5 black") — it is not attention in the J&W sense and is the safer of
   the two. M3's real contract cost is elsewhere: the prototype profile stops being a per-user object.
6. **Sentences.** Tintarev & Masthoff list seven aims — transparency, scrutability, trust, effectiveness, persuasiveness,
   efficiency, satisfaction — and stress they can conflict [verified: portal abstract + Tintarev RecSys'07 Table 1]. The
   "because you bought X" family (M1, M4, M8, M14) scores on persuasiveness/efficiency; transparency holds only where the
   named purchase is the actual score term (M4, M8: yes; M1, M14: only via the y_j share, which is not the purchase's
   *identity* but a learned row). Herlocker 2000 rates "similarity to other movies rated" mid-table (mean 4.97, rank 5/21;
   Table 1 as read by angle A, not reopened here). "Strongly co-occur" (M6, M7) is honest only in the two-part wording of
   B-F1: presence is the user's, strength is the population's. M9's sentence is the one form whose sentence and whose
   geometry fix are the same object (deviation from the crowd).
7. **Protected attributes (ProtoMF §5.3).** The paper finds 36 male- vs 9 female-leaning user prototypes of 93 on ml-1m and
   7 vs 6 of 43 on lfm2b-1mon, using the users above the 95th similarity percentile as representatives, Fisher exact tests at
   α=1 % with Bonferroni [verified, local PDF §5.3]. For fU: (a) word-named prototypes make the proxy *visible* — on hm the
   section/department vocabulary contains gender-typed labels, so a gendered prototype is auditable by its name, an
   improvement in scrutability over the host; (b) **more prone / less auditable**: M1, M14 (unnamed ID channel can encode
   anything), M11 (the gate upweights the field users are most concentrated in — on hm that is the gender-typed
   section/department), M12 with coverage selection (keeps the gender-proxy word because it covers every purchase), M6/M7
   (population co-purchase structure imports population-level gendered correlations as weights); (c) **neutral in
   encoding, better in auditing**: M9 makes deviations along a gendered field explicit rather than hiding them in the shared
   μ — bias becomes a signed, readable coordinate; M16 moves whatever the ID row carried into named words; (d) M2, M3, M4,
   M5, M8, M13, M15 do not change the rows and inherit the host's exposure. The §5.3 representative-user audit transfers
   unchanged to every form and should be a standing read-out for user prototypes regardless of the form chosen.
8. **Ordering for the prototype-layer question.** Mechanism-backed: M9 (removes μ) and M2 (fixes the mismatch) — both cheap,
   both keep the contract intact. Score-backed but contract-widening: M4/M8/M15 (two explanation objects, prototype path
   possibly decorative — read out the bypass share). Contract-costly with no prototype-layer mechanism: M1, M14 (naming
   completeness), M3 (per-candidate profile). Feeders to avoid or pair: M6, M7, M10, M16-alone.

## 3. Citations

Verified this session (abstract or text opened):
- Gao, He, Tan, Qin, Wang, Liu. *Representation Degeneration Problem in Training Natural Language Generation Models*, ICLR 2019, arXiv:1907.12009 — abstract.
- Mu, Bhat, Viswanath. *All-but-the-Top: Simple and Effective Postprocessing for Word Representations*, ICLR 2018, arXiv:1702.01417 — abstract.
- Arora, Liang, Ma. *A Simple but Tough-to-Beat Baseline for Sentence Embeddings*, ICLR 2017 — method via the authors' offconvex.org post (a/(a+p_w) weighting, top-singular-direction removal); OpenReview blocked by a verification wall.
- Jain, Wallace. *Attention is not Explanation*, NAACL 2019, arXiv:1902.10186 — abstract.
- Wiegreffe, Pinter. *Attention is not not Explanation*, EMNLP 2019, arXiv:1908.04626 — abstract (four tests).
- Zhang, Chen. *Explainable Recommendation: A Survey and New Perspectives*, FnTIR 14(1) 2020, arXiv:1804.11192 — abstract (post-hoc vs explainable-model distinction; the "also bought" wording is not in the abstract and is not attributed to it above).
- Tintarev, Masthoff. *A Survey of Explanations in Recommender Systems*, ICDE Workshops 2007 — Aberdeen portal abstract ("seven possible advantages"); the seven aims read from Table 1 of Tintarev, *Explanations of Recommendations*, RecSys 2007 doctoral symposium (PDF, macs.hw.ac.uk).
- Melchiorre, Rekabsaz, Ganhör, Schedl. *ProtoMF*, RecSys 2022 — local PDF `Master/literature/papers/A_protomf/A1_melchiorre_2022_protomf.pdf`, §5.3 "Showcasing Societal Biases" (numbers quoted above).
- Repo measurements: `Master/docs/scrutiny/sc05_chapter_draft.md` l.281–296 (effective ranks, entropy, p99 overlap, sim_batch probe, norm handover); `utilities/explanations/feature_readout.py`, `utilities/explanations/pipeline.py` (intrinsic naming route).

Partially verified / relied on sibling angles:
- Herlocker, Konstan, Riedl. *Explaining Collaborative Filtering Recommendations*, CSCW 2000 — Semantic Scholar record (title/venue/year) opened; the GroupLens PDF failed (expired certificate) and the ACM page returned 403; Table 1 numbers taken from angle A's verified read `[not reopened here]`.
- Koren & Bell (item-based methods "more amenable to explaining"), NAIS > FISM, Balog 2019 selection utilities — as verified in angles A/B/D; not reopened.

`[unverified]` (memory only): the claim that hm `section`/`department` value labels include gender-typed names (e.g. Ladies/Men sections) is from the raw H&M `articles.csv` as remembered, not re-checked against `feature_ids.py` this session; the 85 % catalog-constant item-side figure is a brief given (source doc not located by grep).
