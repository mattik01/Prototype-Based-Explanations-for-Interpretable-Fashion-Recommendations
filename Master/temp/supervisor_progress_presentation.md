# Supervisor Progress Update — Presentation Draft

> Working draft (slide-by-slide + speaker notes + figures checklist). Lives in `Master/temp/`
> per the repo convention for temporary artifacts — move/rename when finalized.
> Built from the protocol vault, design docs, `feature_aware_reading_list.md`,
> `replication_report.md`, the regularizer experiment (`reg_sensitivity/ANALYSIS.md`),
> and the dc01 implementation.
>
> **Narrative arc (decided):** data/eval foundations → lay of the land (reveals the two
> axes) → the two facets as our response → interpretability work done → design process
> → **double-tie recap → Candidate 1 (the climax; the talk ends here).**
>
> **Length target:** ~14 core slides (~20 min) + a backup bundle for Q&A.

---

## SLIDE 0 — Where we are (context, keep brief)

**On slide**
- Replication of ProtoMF: **complete and validated** (the baseline we extend is trustworthy).
- 10/10 in-scope experiments reproduced; **now building the feature-aware extension.**

**Speaker notes**
- 5 models × {ml-1m, amazon2014}. LFM-2b dropped (pulled from its official source over licensing).
- ml-1m reproduces the paper *closely* and preserves model ranking (`user_item_proto` best);
  amazon2014 in the right ballpark.
- One sentence only — the supervisor likely knows this is done; it's the foundation, not the news.

---

## SLIDE 1 — H&M dataset: reduced to ~1M, and why that's clean

**On slide**
- Full H&M: 31.8M transactions → **V1 `hm_1_month`: last month + paper-standard 5-core ≈ 0.6M interactions** (raw window ~1.1M).
- Why it's methodologically OK — **three independent justifications:**
  1. Only the **time window** deviates from ProtoMF's protocol (5-core is paper-faithful).
  2. Kaggle winners *independently* used ~4–5 week windows — **recency dominates**.
  3. The 1-month set is an **"alive catalog" by construction**: 98.9% of last-week purchases are on items active the prior week.
- **Two-version strategy:** V1 = all *quantitative* claims; V2 `hm_3_month` = later *qualitative* explanation showcase ("showing, not proving").

**Speaker notes**
- The slide that answers "is cutting to 1M OK?" — lead with the three justifications; they're mutually reinforcing.
- V1/V2 split of labor: V1 is cheap (3–4 h/model on LEO5) so it carries the numbers; V2's metrics are explicitly *not* headline claims.
- If asked about sparsity: V1 stays the **sparsest** dataset by density (0.062% vs ml-1m 3.0%) — the shrink was designed *not* to destroy that.

---

## SLIDE 2 — Off the Kaggle leaderboard, onto a paper-comparable metric

**On slide**
- ProtoMF = **single-stage CF scorer**; the leaderboard = **two-stage retrieve→rerank** on repeat-purchase heuristics → MAP@12 is **apples-to-oranges**.
- Realigned to: (1) **beat ProtoMF's CF-only baseline** with feature-aware prototypes; (2) **explanation quality**.
- Primary metric = **HR@10 / NDCG@10, leave-one-out** — *the paper's own protocol* → directly comparable. MAP@12 kept only as a lightweight positioning number (+ popularity / repeat-purchase reference baselines).
- Honest scoping: dedup + leave-one-out makes **repeat-purchase structurally invisible** → we measure **discovery** recommendation.

**Speaker notes**
- This is a *realignment*, not a retreat: we're not competing against a different system class.
- Citable methodology point (have it ready): Chen et al. (2020) §3.5.2 — AUC/RMSE penalize all rank
  positions equally, so they're poor for top-N → justifies reporting position-aware metrics and *not* RMSE/AUC.
- The "discovery recommendation only" framing is an honesty point reviewers like — ~12% of window transactions are re-buys (Kaggle's strongest signal), and our protocol deliberately removes them for comparability with the paper.

---

## SLIDE 3 — Lay of the land I: how features enter a CF model (which families fit)

**On slide**
- The literature (and the thesis) splits into **two axes**: **Q1 backbone** (how side-features enter MF) and **Q2 explanation** (how the model explains itself).
- On the backbone, the distinction is **attribute-aware CF** (features fold *into* the latent factors) vs **hybrid/content** (a separate content model bolted on).
- **Suited as our core mechanism** — the "bridge" families: **SVDFeature, LightFM** → features→factors *additively* (`q_i = Σ e_f`); feature embeddings live in the **same space as prototypes**, so similarities decompose exactly per feature. (CMF = ancestor.)
- **Less suited as the core mechanism** (kept only as **baselines**): **FM / NFM / DeepFM / GMF** — feature-centric or reconstruction-based, expressive but **opaque**; they don't ground prototypes. **RLFM rejected** — shifts *where embeddings sit, not what prototypes mean*.

**Speaker notes**
- The headline: *a hybrid is only worth it if there is a bridge between the added feature module and the grounding of the prototypes.* Additive families provide that bridge; FM/GMF families don't.
- Three concrete ways attributes can enter (from the survey taxonomy): **ADD / INTERACT / REGULARIZE.**
  H&M's asymmetry (rich item features, thin customer features) **argues against** the symmetric INTERACT route → steers the design asymmetric/item-first.
- FM/NFM/DeepFM/NCF+ stay in as the **feature-aware accuracy baselines** (NCF+ was the strongest in the survey) — the control for "does interpretable grounding cost accuracy?"

---

## SLIDE 4 — Lay of the land II: explanation vehicles, and the open wedge

**On slide**
- Prototypes are **one explanation vehicle among several**: concept-bottleneck models, attention over feature interactions (AFM — with the "Attention is not Explanation" caveat), disentanglement (MacridVAE — with the critical-review caveat that *disentanglement ≠ interpretability*).
- **The wedge we fill:** closest prior art **ProtoCF** builds prototypes from *interaction few-shot, not item metadata*; and **all 31 ProtoMF forward-citations stay CF-only**.
- → **Grounding prototypes in item side-features is open** even in ProtoMF's own citation neighbourhood.

**Speaker notes**
- "We surveyed the menu and chose prototypes deliberately — not because it's the only option." Naming the alternatives + their caveats shows the choice is informed.
- The gap claim is corroborated, not asserted: a forward-citation sweep of all 31 ProtoMF citers (UIPC-MF, debiasing, HMF…) found none grounding prototypes in features.
- Bias-auditing is the trustworthy-AI hook: feature-grounded prototypes make auditing *more direct* — a prototype keyed on a problematic attribute is readable from its profile, no probe needed.

---

## SLIDE 5 — Framing & contribution: the two facets

**On slide**
- **Positioning chain:** CF (transparent, weak) → MF (strong, opaque) → **ProtoMF (strong + re-adds the interpretability MF threw away).**
- **ProtoMF = the inverse of a Concept Bottleneck Model:** it gets the **concept→prediction** half transparent *for free* (score is additive over prototypes) but leaves the **meaning of the concepts** open (prototypes named post-hoc). CBMs do the opposite. → *the field points here; we go here.*
- **Two facets, deliberately connected:**
  - **Facet A — Interpretability:** move prototype *grounding* from **post-hoc → intrinsic**.
  - **Facet B — Predictive + flexibility:** leverage features (sparsity / cold-start).
- **→ Research question:** *Can grounding ProtoMF's prototypes in item features make prototype meaning **intrinsic** and improve recommendation under **sparsity** without sacrificing accuracy — and can **interpretability then be tuned as an explicit objective**, not just an accuracy by-product?*
- **→ Contribution (one line):** a feature-aware ProtoMF in which **item features are first-class citizens of the prototype space** (intrinsic, feature-grounded meaning + sparsity robustness), *plus* treating **interpretability-bearing hyperparameters as a multi-objective selection problem** — evaluated on H&M against CF-only and feature-aware baselines on **both** predictive (HR@10/NDCG@10) and explanation-quality axes.

**Speaker notes**
- The two facets *are* the contribution; state the RQ and contribution line out loud as the anchor the rest of the talk delivers against.
- Why ProtoMF over plain CF at all: memory-based CF is *already* interpretable, so interpretability alone isn't the reason — MF buys scaling + sparse-data performance but throws interpretability away; ProtoMF keeps MF's tier and re-adds it. The flavour differs: **abstract concepts** ("you like minimalist basics") vs CF's **concrete item lists**. *The abstraction is the point.*
- Cold-start is the sharpest Facet-B motivation: feature-aware prototypes can place a brand-new item from metadata alone — **CF literally cannot.** Fashion's high item turnover makes this a real problem.
- They meet in one model — Candidate 1 (the closing section) is the first instantiation that delivers both.

---

## SLIDE 6 — Interpretability work I: extensions to the explanation module

**On slide** *(show artifacts — see figures checklist)*
- Grew the upstream single file into a full **explanation pipeline** (`utilities/explanations/`): 5 registered explainers — naming, t-SNE (now with named prototypes), top-k items, per-recommendation **weight decomposition**, feature small-multiples.
- **Lift-based prototype naming** (distinctive values beat merely common ones).
- **New intrinsic per-feature read-out** (`feature_readout.py`) — the genuinely novel piece (used by Candidate 1).
- Contrast: upstream = **post-hoc** nearest-items naming; new = a step toward **intrinsic** grounding.

**Speaker notes**
- Show: a named-prototype t-SNE + the `weight_viz` decomposition of one user-item score (per-prototype contributions).
- Caveat to state: the on-disk plots are from a short smoke run — illustrative of *what the pipeline produces*; will regenerate from a properly-trained checkpoint for the final figures.
- Note the finer axis: base ProtoMF's *reasoning* is already intrinsic (the score *is* prototype similarities); only the prototype **naming** is post-hoc. We're careful not to imply base ProtoMF is a black box.

---

## SLIDE 7 — Interpretability work II: the regularizer experiment

**On slide** *(show the inverted-U table)*
- ProtoMF has **two prototype regularizers**: `reg_batch` (every entity covered by a prototype) and `reg_proto` (no dead prototypes).
- Swept their strength on ml-1m + H&M. Two curves diverge:
  - **Accuracy** falls **monotonically** with regularization (reg-off most accurate).
  - **Interpretability** (top-k category purity) is an **inverted-U** — ml-1m genre purity **0.70 (off) → 0.87 (×1) → collapse at ×100**.
- Geometric reading: stronger reg reshapes a diffuse embedding cloud into **prototype-centred clusters**.

**Speaker notes**
- The point: accuracy and interpretability are tuned by the *same knobs* but want *different settings* — they're genuinely distinct objectives.
- Honest limitation to state: the sweep scaled all four weights by a single multiplier → it **can't disentangle** the two regularizers or the user-vs-item side (and is `sim_proto`-dominated). A per-weight grid is future work.
- Figure note: the clean "diffuse → clustered" contrast wants **off vs ×1**; ×1 isn't on disk yet (only the degenerate ×100), so regenerate from LEO5 checkpoints.

---

## SLIDE 8 — Interpretability culmination: tuning becomes multi-objective

**On slide**
- The two regularizers are **pure interpretability penalties** — yet currently tuned **against accuracy** (`hit_ratio@10`). **Conceptual mismatch.**
- Fix: tune them by an **interpretability metric within an accuracy tolerance ε** → model selection becomes explicitly **multi-objective**.
- Framed as **searching the Rashomon set** for its most interpretable member (Breiman / Rudin) — once we *define what prototype interpretability means in our case*, the same logic extends to other knobs: **n_prototypes, prototype sparsity-style constraints**.
- Interpretability constraints double as **expert-knowledge infusion**.

**Speaker notes**
- This is the conceptual payoff of the interpretability facet: interpretability is a *design objective*, not a free by-product of accuracy tuning.
- **Honesty flag (our own protocols self-corrected on this):** present Rashomon as a *conceptual lens / one-paragraph justification for affordable interpretability* — **not** a measured quantity (the set can't be measured; leaderboard flatness is partly a metric artifact). Empirical hook: Dacrema et al. 2019 "Are We Really Making Much Progress?" + our own ProtoMF ≈ MF replication.
- Expert-knowledge precedent (have it ready): interpretable sleep-stage scoring (Niknazar & Mednick) — an expert-constrained model became a *trusted tool* **and** performed better; constraining to known-good vocabulary guides learning. Same pattern as feature-grounded prototypes.

---

## SLIDE 9 — From objective to design: requirements + a disciplined process

**On slide**
- **Requirement spec (R1–R8, strong defaults not hard rules):** R1 tied (not a bolt-on) · R2 intrinsic (not post-hoc) · R4 feature-grounded prototypes · R5 sparsity robustness · **R7 user-perceivable vocabulary** · **R8 conceptual parsimony** ("one clean idea at ProtoMF's granularity").
- **Design Candidate Protocol:** one literature-grounded candidate per cycle · built-in **adversarial checks** (math sanity check + devil's advocate) · **no LLM self-ranking** (ranking is mine) · a scratchpad of spillover ideas for future candidates.
- Candidate 1 is the **first product** of this process.

**Speaker notes**
- R7 tension is worth a sentence: the most *user-visible* features (colour/pattern ~1.3× signal) are the *weakest* predictors; internal taxonomy (department ~5.2×) is strong. Resolution: taxonomy/price/CF can **drive the score**, but the prototype **read-out** is constrained to user-perceivable vocabulary — and we measure what that costs.
- The process is deliberately bias-resistant (isolated adversarial subagents, no model self-ranking) — emphasize rigor; supervisors value it.
- Candidate sequencing plan: item-only → unified feature set → separate user/item sets (mirrors ProtoMF's own progression). Stage 1 is the must-have; later stages are stretch — *"a master's thesis, not a dissertation."*

---

## SLIDE 10 — Recap: how double-tie ProtoMF (`user_item_proto`) works

> *Building block for Candidate 1 — assume ProtoMF is understood; this is the one piece C1 modifies.*

**On slide** *(schematic — adapt the diagram in `06a_double_tie_reference.md`)*
- Two prototype spaces: **user prototypes** and **item prototypes**.
- Each entity has **one embedding feeding two branches**:
  - **proto branch** → shifted-cosine similarity to its own prototypes (a *similarity profile*),
  - **projection branch** → a linear map into the *other* side's prototype space.
- **Score = U-score + I-score** = `u*·t̂ + û·t*` — a **sum**, so every contribution traces back to a user prototype or an item prototype (clean decomposition).
- **The "tie":** the item's two branches read the **same item embedding** (one vector per item); same for the user side.

**Speaker notes**
- Walk it in one breath: user's prototype profile × item's response to user-prototypes, **plus** user's response to item-prototypes × item's prototype profile.
- Why a sum and not a neural merge: a sum keeps the two sub-scores **fully separable** — that's the explainability.
- Honest nuance (sets up later caveats): the projection branches are **opaque** linear combos, and the shared embedding is a **compromise** between two pressures — but the prototype-similarity *profiles* stay interpretable cosines. This is the `prototypes_double_tie` explainability tradeoff.
- **The one thing to plant:** *each item is represented by a single learned item-embedding vector.* That is exactly what Candidate 1 changes.

---

## SLIDE 11 — Candidate 1: `feature_item_proto` (the climax)

> *Keep the entire double-tie architecture. Change ONE thing.*

**On slide**
- **Before:** the item's vector is a **free per-item ID embedding** (one learned vector per item).
- **After:** the item's vector is the **sum of its feature-value embeddings** — `q_i = Σ_{f∈f_i} e_f` (Trousers + Dark + Menswear + price-band + …). *(LightFM / SVDFeature template.)*
- The **same `q_i` still feeds both item branches** → the tie, prototype layer, regularizers, loss, and user side are **untouched**.
- **Payoff — exact, intrinsic decomposition:** because feature embeddings and prototypes share one space,
  each item-prototype similarity splits *exactly* per feature:
  `t*_k = 1 + Σ_f (e_f·p_k)/(‖q_i‖‖p_k‖)`
  → *"this item matches prototype k because of Trousers (+0.21), Dark (+0.14)…"*
- And every prototype gets an **intrinsic attribute profile** (`cos(e_f, p_k)` over the vocabulary) — computed from parameters, **no nearest-item lookup**.
- → **Prototype grounding moves post-hoc → intrinsic.** Facet A delivered by a Facet B mechanism — *the two facets meet in one model.*
- **Status:** implemented; **14/14 local correctness checks pass**; at F=0 it is **bit-identical to `user_item_proto`** (a strict generalization of the baseline). **Not yet trained on the cluster** — empirical validation is the immediate next step.

**Speaker notes**
- Land the one-liner: *swap the item's free ID vector for the sum of its feature embeddings; everything downstream is identical; now every item-prototype similarity decomposes into per-feature shares.*
- The decomposition is **exact algebra** (verified to ~1e-16), not a saliency estimate — that's the interpretability win over post-hoc methods.
- Be explicit it's early (your steer): *"Architecture and its correctness invariants are proven locally. What we don't yet have: a single accuracy number or a trained-model explanation figure. That's the next step on LEO5 — and if it lands, Candidate 1 is the centerpiece of the next update."*
- If asked about the main risk: R4↔R6 are coupled through whether we keep an ID feature — the setting that maximizes feature-explained activations also constrains capacity. We chart that with an ablation; we don't decide it a priori. (Backup slide.)

---

## SLIDE 12 — Close (one line; we end on C1)

**On slide**
- **Immediate next step:** train Candidate 1 on LEO5 → measure vs CF-only + LightFM-MF baselines; check whether prototype **profiles come out sharp**; cold-item / tail split for sparsity (R5).

**Speaker notes**
- Keep it to one sentence — the talk's weight is on C1 itself, not a roadmap.

---

# BACKUP SLIDES (for Q&A — not in the main flow)

- **Compute envelope.** LEO5 10-day job wall, no fragmentation; V1 runs 3–4 h/model (why V1 carries the numbers; `hm_3_month`/full cost 3–5+ days).
- **Replication numbers.** Table from `replication_report.md` (ml-1m within a few points of the paper; amazon noisier; single-seed vs paper's 3-seed; best H&M V1 = `user_item_proto` HR@10 0.6606 / NDCG@10 0.4174).
- **Honesty caveats.** (1) **Cosine-readout caveat (Steck, WWW 2024)** — cosine is near-arbitrary for MF embeddings, and ProtoMF *is* MF, so it lands harder; threatens the *semantic* reading of similarity (not the exact computation). Mitigation = a **cosine-reliability probe** (rescaling-perturbation test, metric-swap, seed-stability). (2) **Single-seed** runs. (3) **R4↔R6 coupling** via the ID-feature fork. (4) **Partial image set** (S3 deferred).
- **Feature-analysis signal ranking.** `product_code` 35.9× (identity-like → explanation machinery, not a model feature); department 5.2×, section 3.3×, product_type 2.8×; price decile 2.36×; colour/pattern ~1.3×. Decision: feed **~4 quasi-orthogonal feature groups**, not all 9 columns.
- **Why a SUM and the single L_rec** in UI-ProtoMF (forces the two sub-models to cooperate; keeps decomposition clean).
- **t-SNE/DR caveat.** t-SNE distorts global geometry (Rudin GC#7) → no *visual distance* claims off the 2D plot; all quantitative explanation claims computed in the original embedding space (cosine/lift); PaCMAP as the cleaner DR.
- **Explanation-quality eval ladder.** qualitative case studies → prototype coherence/purity → fidelity (activation↔score) → user study (likely out of scope). Profile **entropy/sharpness** is a committed primary metric for C1.
- **`s^user` contribution-equivalence concern** (from the paper re-read): "strong user × weak weight" vs "weak user × strong weight" score equally — may not be semantically equivalent for explanations; flagged to check in practice.
- **Feature-injection fork: route A (into representation/score) vs route B (prior/regularizer on the latent space).** C1 is route A; route B (attribute-aware priors on the prototype space) is flagged in the survey as **unexplored** — a citable angle for a future candidate.

---

# FIGURES TO INCLUDE — checklist with paths

> Verified on disk 2026-06-17. ⚠️ = needs regeneration before the final deck.

**Ready / solid**
- **Replication table** → numbers in `Master/docs/replication_report.md`.
- **Regularizer inverted-U** (accuracy-down + purity-up tables, the 0.70→0.87 figure) → `Master/experiments/reg_sensitivity/ANALYSIS.md` (+ `Master/temp/regsweep_smoke/sweep_results_ml-1m_s38210573.csv`).
- **Double-tie schematic + worked decomposition** → adapt the alignment diagram and `u*·t̂ + û·t*` trace in `Master/understanding/codebase/06a_double_tie_reference.md`.

**Exist on disk, but only from a 2-epoch smoke (reg_off endpoint) — fine to illustrate the *pipeline*, ⚠️ regenerate from a trained checkpoint for real results**
- Named-prototype t-SNE → `Master/temp/regsweep_smoke/user_item_proto_ml-1m_s38210573_reg_off/explanations/lift/tsne_item_prototypes.png` (and `tsne_user_prototypes.png`).
- Per-recommendation weight decomposition → same dir, `weight_viz_user42_item_side.png` + `weight_viz_user42_user_side.png`.
- Feature small-multiples → same dir, `feature_small_multiples_genres_item.png`.
- Prototype naming tables → same dir, `prototype_names_item.csv`, `top_k_items_per_item_prototype.csv`.

**⚠️ Needs generating**
- **Clustering contrast t-SNE: off vs ×1** (the interpretability peak) — only `reg_off` and `reg_x100` (degenerate) are on disk. Regenerate ×1 (and intermediate points) from the real LEO5 checkpoints via `Master/temp/assemble_proto_tsne_grid.py` → `tsne_grid_<dataset>.png`.
- **Candidate-1 per-feature decomposition figure** ("matches prototype k because of Trousers +0.21, Dark +0.14") — read-out exists (`utilities/explanations/feature_readout.py`) but isn't wired into the pipeline yet; build from a toy/trained model. This is the single most compelling C1 visual once available.

---

# Notes for Matteo

- **What I'd cut to stay "small":** compute envelope, full kaggle-source breakdown, ProtoMF-vs-PPM detail — all backup, not main flow.
- **Research question / contribution:** now stated explicitly as the anchor of Slide 5 (RQ + one-line contribution).
- **Held back (ongoing C1 work):** the C1 per-feature decomposition figure and the C1 evaluation plan (profile entropy, LightFM-MF attribution baseline, cold-item split) are deliberately **not** promoted — C1 stays the in-progress climax (Slide 11). Promote these once the C1 work settles.
- **Ordering choice:** the literature section is one hinge (Slides 3–4) that reveals the two axes, which then map onto the two facets. Alternative: split it — backbone-families into Facet B, explanation-vehicles into Facet A. I recommend the single hinge; it reads more coherently.
