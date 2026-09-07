# Deep research: is summing attribute-word embeddings the best composition? (dc01)

> Four-agent literature sweep, 2026-08-24, synthesized. Question (Matteo): dc01 composes an
> item as the unweighted sum of one embedding per attribute field plus a free ID row
> (`t_i = Σ_f e(x_if) + e_ID(i)`, shifted cosine to prototypes). Is the sum the right
> composition; what are the alternatives? Angles swept: (A) recsys feature-interaction
> lineage (FM→CTR), (B) side-information MF & cold-start, (C) set-pooling & additive-
> composition theory, (D) interpretability-preserving composition (GAM/SENN/attention debate).
> All claims below were verified by the agents against retrieved abstracts/full text unless
> marked [unverified].

## 0. Headline verdict

1. **The sum is the canonical choice, not a naive one.** The linear feature-based-MF family
   (FM 2010, SVDFeature 2012, LightFM 2015) composes entities exactly this way; LightFM —
   built at Lyst, a *fashion* company — defends the unweighted sum on cold-start, statistical,
   and interpretability grounds, and its Table 1 is the only direct published ablation of
   dc01's exact design: attribute-sum **+ free ID row beats attributes-only in warm AND cold**
   (ML AUC 0.763 vs 0.744 warm; 0.716 vs 0.707 item-cold). No paper cleanly ablates
   sum vs mean vs max vs attention for categorical side features in an MF scorer.
2. **The literature's near-unanimous critique targets the *uniform weighting*, not the
   additivity** (AFM: uniform weights let noisy features pollute the sum; FwFM: field-pair
   scalars match FFM at 4% of the parameters; FiBiNET: SENET reweighting of field embeddings;
   DIN: unweighted sum pooling as a bottleneck). The cheap fixes preserve exact attribution.
3. **Theory locates dc01's sum precisely**: it is the linear corner of the Deep Sets family /
   k=1 Janossy pooling — first-order effects only, no attribute conjunctions expressible
   before pooling. And exact additive semantics is *provably correct only under uniform
   component frequencies* (Gittens et al. 2017); under Zipfian attribute frequencies
   (H&M reality) frequent-value embeddings are predicted to drift toward a common direction
   and dominate sums — precisely the SIF / all-but-the-top pathology.
4. **Everything input-dependent or deep forfeits the explanation contract.** The
   attention-is-not-explanation verdict (Jain–Wallace; Serrano–Smith; Wiegreffe–Pinter):
   input-dependent pooling weights are at best noisy importance correlates needing per-model
   validation, never attributions by construction. SENN marks the negotiated middle: dynamic
   weights are admissible only if bought back with faithfulness/stability regularizers.
5. **Gains beyond tuned low-order structure are small and fragile** (Zhu et al. 2021 open
   CTR benchmark, 7k+ experiments: deep-interaction gains largely vanish under fair tuning;
   Rendle et al. 2020: tuned dot product beats learned MLP similarity). dc01 is not
   obviously leaving accuracy on the table by staying additive — especially since the
   prototype layer supplies nonlinearity downstream of the composition.

## 1. Documented pathologies of the current unweighted sum (checkable on dc01)

| # | Pathology | Source | dc01 relevance |
|---|---|---|---|
| P1 | Common-direction / anisotropy: shared frequent components create a large mean + few dominant directions that inflate cosines everywhere | Mu & Viswanath 2018 (All-but-the-top); Arora et al. 2017 (SIF) | Popular attribute values ("Black", big garment groups) are this setting's stop-words; plausibly feeds the observed eff.-rank-1.1–1.9 geometry collapse |
| P2 | Cone collapse of embedding tables under likelihood-style training; frequency causal | Gao et al. ICLR 2019 (representation degeneration) | Literature anchor that the collapse is a known *optimization* pathology, with in-training (cosine-reg) and post-hoc (centering) fixes |
| P3 | Norm–popularity coupling: frequent items/values grow large norms and dominate compositions | NISER (Gupta et al. 2019) | Shifted cosine normalizes the *composed* vector, not the summands — within-sum dominance by high-norm popular words remains possible |
| P4 | Uniform weighting admits noise from useless features | AFM (Xiao et al. 2017), DIN (Zhou et al. 2018) | ID row silently weighted 1.0 = one word; per-field balance never learned |
| P5 | Additivity provably exact only under uniform frequencies; Zipf ⇒ distortion | Gittens et al. ACL 2017; Seonwoo et al. 2019 | The theory statement of P1 for the word-composition reading |
| P6 | Invasive additive fusion of side info into item embeddings can hurt vs. no side info; add/concat/gating differences second-order | NOVA-BERT AAAI 2021 (controlled Table 3); DIF-SR SIGIR 2022 | Caveat: shown in attention architectures, not MF; still the closest controlled pooling comparison found |
| P7 | Cosine on regularized-MF embeddings has rescaling degrees of freedom → similarity values can be arbitrary | Steck et al. WWW 2024 | Read per-word shares comparatively, not absolutely; the exact claim is "additive attribution of the pre-normalization score, up to an item-global scale" |
| P8 | Deep embedding smoothing over-generalizes in sparse high-rank regimes (item identity washed out) | Wide&Deep 2016 | The warm-side mirror of the ID-share question |

## 2. The upgrade ladder, ordered by what happens to the share read-out

**Tier A — keeps exact additive attribution, near-zero cost (literature-sanctioned):**
- **A1. Learned constant per-field / per-value weights** (`t_i = Σ_f w_f·e(x_if) + w_ID·e_ID`).
  FwFM/FiBiNET precedent; weights fold into terms (input-independent ⇒ still exact, per the
  GAM/NAM criterion); makes the silent ID-weight-1.0 choice explicit and adds a
  "which fields matter" read-out. ~10 params; the bag layout in `FeatureEmbedding` already
  supports per-token weights.
- **A2. SIF-style deterministic frequency weights** `w_v = a/(a+p(v))` — the
  "popularity-corrected additive composition" with a generative-model justification;
  constants, so shares stay exact. Directly attacks P1/P3/P5.
- **A3. Centering / common-component removal** (subtract mean word vector; optionally
  project off top PCs) — any *fixed* linear map commutes with the sum
  (`P(Σe) = ΣPe`), so attribution survives exactly. Attacks P1/P2. Doubles as a
  geometry-collapse instrument regardless of whether it becomes a variant.
- **A4. Cold-path training alignment: ID-embedding dropout** (DropoutNet 2017; Heater's
  randomized training) — randomly zero e_ID during training so train matches the cold
  drop-the-ID protocol. Zero architectural change, zero effect on the read-out; directly
  targets the observed fI warm→cold gap (cold-vs-all: noid 0.394 vs fI 0.218).

**Tier B — attribution stays exact but becomes pair/block-granular:**
- **B1. FM-style pairwise terms** (selected pairs, **constant** coefficients). GA²M/EBM is
  direct precedent that a named pair term ("colour × garment") counts as a first-class
  faithful attribution; NFM's +7.3% over FM is the evidence conjunctions carry signal.
  Three cautions from the literature: (i) **identifiability** — Lengerich et al. AISTATS
  2020: main-vs-interaction shares can be shuffled without changing predictions; the split
  needs a functional-ANOVA/purification convention or explicit disclosure; (ii) keep pairs
  **sparse/selected** (GA²M's FAST) or the explanation object explodes; (iii) keep pair
  coefficients **input-independent** — AFM is the cautionary contrast (attention over pairs
  ⇒ back into the attention-debate camp).
- **B2. Field-blocked structure / one-hot-role TPR** — concatenating per-field blocks is
  the degenerate tensor-product representation (Smolensky): still a literal sum over
  attributes under a dot-product score, removes cross-field interference, adds slot
  identity ("which field contributed"), at the price of hard per-field capacity allocation.

**Tier C — forfeits attribution-by-construction (avoid as *explanation*; usable as ceiling):**
- Attention/gated pooling over the words (FDSA, FiBiNET-SENET, AutoInt): weights are not
  attributions (attention debate); admissible only with SENN-style regularizers + per-model
  validation — re-imports the problem dc01 exists to avoid.
- MLP / DeepSets-full composition: universal in theory (Zaheer 2017; latent-dim fine print
  Wagstaff 2019 — harmless here, few attributes, d large), zero decomposition. Worth keeping
  only as a "cost of additivity" ceiling comparison, not a candidate.

## 3. Points that feed the thesis text directly

- The Janossy-pooling vocabulary pins the design: dc01's sum = k=1 pooling; the exact-
  decomposability requirement *mathematically* selects k=1 (or additive pair terms at k=2
  with pair-level attribution) — a clean way to state the trade in §5/§6.
- LightFM's own interpretability sentence ("the representation for denim jacket is simply
  a sum of the representation of denim and the representation of jacket") is a quotable
  precedent for dc01's composition reading; and LightFM is already the S0 CBF baseline,
  so dc01-vs-lightfm is literally "prototype layer added to the canonical composition".
- ProtoMF's own faithfulness argument (RecSys 2022 abstract: prediction "linearly broken
  down into the contributions of distinct definable prototypes") is the same additivity
  argument one level up — dc01 extends it downward one level, exactly because the cosine
  numerator is linear in the composition.
- Cosine caveat for the share meter (P7 + joint-norm point): shares are exact for the
  numerator, jointly scaled by ‖t_i‖‖p_k‖, and not counterfactual. Already disclosed in the
  implementation plan; Steck 2024 is the citation that makes it a literature-grounded caveat.
- CBM's intervention test (edit one word's embedding → only its share moves) is the
  operational audit for whatever composition is chosen.

## 4. Candidate cheap experiments (working evidence tier, not thesis-grade)

1. **Measure first (no retraining):** per-word embedding norms vs value frequency (P3);
   mean word vector norm / top-PC variance share of the word table (P1); ID-vs-words share
   distributions already exist. Login-node class.
2. **A3 as instrument:** re-render prototype profiles and shares with centred word table
   (post-hoc, fixed map) — does the geometry-collapse picture change?
3. **A1/A2 as ablation arms:** learned field weights and SIF weights are one-line variants
   of the existing weighted-bag path; hyperopt-light.
4. **A4 ID-dropout arm:** one knob, targets the cold gap specifically.
5. **B1 only if conjunctions are judged a real gap** (the wI fork independently flagged the
   conjunction limitation) — and then with FAST-style pair selection + the Lengerich
   disclosure.

## 5. Key citations (verification status as reported by agents)

Verified against retrieved abstracts/full text: Wagstaff et al. ICML 2019; Lee et al. ICML
2019 (Set Transformer); Murphy et al. ICLR 2019 (Janossy); Ilse et al. ICML 2018; Arora et
al. TACL 2016 (RAND-WALK); Gittens et al. ACL 2017; Seonwoo et al. 2019; Mu & Viswanath
ICLR 2018; Arora et al. ICLR 2017 (SIF); Gao et al. ICLR 2019; Steck et al. WWW 2024;
Nickel et al. AAAI 2016 (HolE); Lou et al. KDD 2013 (GA²M); Caruana et al. KDD 2015;
Nori et al. 2019 (EBM); Agarwal et al. NeurIPS 2021 (NAM); Chang et al. ICLR 2022
(NODE-GAM); Radenović et al. NeurIPS 2022 (NBM); Lengerich et al. AISTATS 2020; SENN
NeurIPS 2018; CBM ICML 2020; Jain & Wallace NAACL 2019; Wiegreffe & Pinter EMNLP 2019;
Serrano & Smith ACL 2019; ProtoMF RecSys 2022; EFM SIGIR 2014; AFM IJCAI 2017; NFM SIGIR
2017; Wide&Deep DLRS 2016; DeepFM 2017; PNN ICDM 2016/TOIS 2018; xDeepFM KDD 2018; AutoInt
CIKM 2019; FiBiNET RecSys 2019; FwFM WWW 2018; FEFM 2020; DCN-V2 WWW 2021; DIN KDD 2018;
NISER 2019; Zhu et al. CIKM 2021; Rendle et al. RecSys 2020; LightFM 2015 (full text);
Gantner et al. ICDM 2010; CB2CF RecSys 2019; DropoutNet NeurIPS 2017; Heater SIGIR 2020;
CLCRec MM 2021; FDSA IJCAI 2019; NOVA-BERT AAAI 2021 (full text); DIF-SR SIGIR 2022.
[unverified, memory-only, flagged by agents]: Deep Sets theorem statement from its own
abstract (cross-verified via Wagstaff); FM/FFM equations; Plate HRR 1995; Mikolov analogies;
Mitchell & Lapata; Bien et al. hierarchical lasso; several mechanism internals noted inline
in the agent reports.

LightFM + NOVA-BERT PDFs downloaded to the session scratchpad (`downloads/`).
