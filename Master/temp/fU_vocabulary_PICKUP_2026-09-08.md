# fU user-vocabulary research — pickup note for 2026-09-08

**Status 2026-09-07 evening:** the seven-agent deep research on "how should a user's interaction history be represented in
fU-ProtoMF beyond mean-of-attribute-words" is COMPLETE. Nothing has been decided, nothing committed, nothing run on the
cluster. This note is the single entry point; the full synthesis and the raw agent notes are linked at the bottom.

Session history: research session ran 17:31–22:39, killed accidentally twice (rate limit at 18:12, killed session at 22:39);
angle G was re-run in a fresh session and the memo finalised at ~22:45. Everything below is on disk.

---

## 1. The question and the fixed scope

Matteo's question (verbatim): *"right now it averages interactions for user embeddings, which brings it back to the
attribute vocabulary which is fine, but truly lets think about what appropriate and interpretable vocabulary would be
based on interactions for a user. I am not sure how much we should impact the way the model uses interactions, but maybe
we can find way better solutions."* Motivating example: *"we recommend this, because these two interactions in your
history strongly co-occur."*

Scope decisions taken in-session:
- Input = ONLY which items the user interacted with in the train window. Event context (date, season, channel, price,
  order structure) OUT. Demographics OUT, not even as fallback.
- Item-identity-level and attribute-level history MAY be combined inside the user representation.
- Every form is priced on two axes: (1) is exact additive attribution to nameable units preserved? (2) how much does it
  alter what the host does with interactions (none / re-weighting / new additive terms / new object)?

Today's fU (dc05, `feature_user_proto`): q_u = Σ_f (n_{u,f}/|H_u|) e_f + e_ID(u); mean over purchases of item-attribute
word rows (hm V=426, ml-1m genre+tag bags), plus a free ID row (`_noid` drops it); no timestamps, cap, recency, IDF or
centring. Vocabulary/codes shared with fI, embedding table fU's own. Then U-ProtoMF's prototype layer.

---

## 2. Headline findings (decision-grade)

1. **"Strongly co-occur in your history" is never faithful for one hm user.** With 5 purchases every pair is present
   once; "strongly" can only be a population weight. The faithful form is two-part: presence is the user's, strength is
   the population's. Any co-occurrence sentence must say both halves.
2. **Cross-item word pairs add no information.** A pair term with weight n_a·n_b is a deterministic quadratic of the
   user's word-count vector. Via the NFM identity it is ≈ ½ c⊙c with c the catalog-constant vector, so it *amplifies*
   the common-direction collapse unless centred first. The only genuinely new pair signal in a history is within-item
   conjunction (black∧dress = an item-side vocabulary extension, fI's E3 rival, found cold in today's item probe) and
   item identity.
3. **With a free ID row, almost nothing changes the function class.** Every form that is a fixed function of the user's
   own train row is absorbable by e_ID(u) (the D2 lesson generalised; realised on the item side today, D2 0.07→0.49).
   Only target-conditioned attention escapes. So in the ids arm the menu changes inductive bias and attribution, not
   capacity; the honest test bed for composition forms is `_noid`.
4. **The measured loss sits at the prototype layer, and exactly one composition form has a mechanism there:
   population-mean centring (fU′).** It removes the shared input direction the cosine layer cannot ignore (Mu &
   Viswanath; SIF step 2). Dirichlet shrinkage, rule tokens, co-purchase re-weighting and an ID-cap alone *feed* that
   direction; the rest leave it. Forms that move the score more (FISM item-item term, EFM direct-match term,
   complement term) do so by adding a winning dot-product path *beside* the prototypes, making the prototype bars
   decorative: contract-widening, not repair.
5. **DECISION-GRADE CODE FINDING: during training the scored positive's own words are inside q_u.** The history buffer
   is the full train file and the user extractor sees only the user index (verified: `feature_ids.py:496-510`,
   `feature_extractors.py:459-470`, `rec_sys.py:103`, `protomf_dataset.py __getitem__`). At hm median |H|=5 that is
   20 % of the metadata mass; at test the target is never in the history — a FISM `\{i}` train/test mismatch.
   `lightfm_hist` shares it (same module), so fU-vs-control is unbiased, but every absolute hm number of both is
   inflated by an unknown amount, and any item-ID history row would turn it into memorisation of the positive.
   **Leave-one-out pooling in training is hygiene owed before any composition claim.**
6. **What the transparency literature actually does:** every scrutable profile = frequency-weighted mean + deviation or
   shrinkage + sparsification to few units, grounded with *item examples*. Human evidence (Balog 2019, 122 users): the
   single most-endorsed unit is the example item; single-tag statements beat pair statements (57 % vs 34 % agreement);
   Tagsplanations found plain genre-level words beat specific ones. Cheapest interpretability gains are
   explanation-layer only (item examples under words; deviation-from-crowd display).
7. **Bottom-up check (surveys' own taxonomies) confirms rather than overturns.** Recsys surveys are order-first;
   the unordered-set cell is filled by rule mining, session-KNN and topic models, not by a different pooling operator.
   Cross-domain (EHR, IR, NLP): attribution survives by keeping the sum and linear rows; structure beyond the mean
   comes from *grouping* (visit, session, topic), not a new operator. Deep Sets: every order-free form is ρ(Σφ).

---

## 3. Pathologies of the mean-of-words user (each tied to a dc05 measurement)

| # | Pathology | Evidence | Acts on it |
|---|---|---|---|
| P1 | Shared input direction: every q_u contains the population mean → near-rank-1 cloud at the cosine layer | eff. rank hm fU 1.79 / noid 2.20; ml-1m 2.63 / 2.46 (host 4.44); profile entropy ≈ uniform; p99 overlap 1.0 | centring removes; shrinkage/rules/ID-cap-alone feed |
| P2 | Norm handover: ‖mean of words‖ shrinks with |H| → ID share grows mechanically | hm ID-energy 0.009→0.050 thin→heavy; ml-1m 0.003→0.070 | ID cap; centring partly; else disclose |
| P3 | Count bottleneck: item identity and within-item conjunction lost (Janossy k=1, linear φ) | structural | item-ID rows; conjunction vocabulary |
| P4 | Evidence mass inexpressible: 1-purchase user activates prototypes as decisively as 100-purchase user | structural (cosine kills per-user scalars) | none in score; explanation layer only |
| P5 | Training self-inclusion of the positive (20 % at |H|=5) | verified in code this session | leave-one-out pooling |
| P6 | Dataset meaning flip: hm = 5 same-day purchases × 5 fields; ml-1m = 56 movies × 3–72 tokens; on hm any cross-item pair term is a within-basket term by accident | 94.3 % same-day; bag-size channel 0.67 | disclosure obligation for every form |
| P7 | Free rows choose their own norm: exact shares can be optimizer-chosen shares | ID norm handover; would recur for item-ID rows | norm read-outs mandatory |

---

## 4. The menu of forms and the critic's ranking

Ranking (exact attribution first; host alteration none → re-weighting → new terms → new object; ID-redundancy tiebreak):
1 **M2** leave-one-out pooling · 2 **M9** population-mean centring (no count-shrink half) · 3 M16 ID-share cap (only
beside M9) · 4 M5 conjunction vocabulary · 5 M1 item-ID history rows · 6 M12 top-k sparsified profile · 7 M13
saturating counts · 8 M11 per-field concentration gate · 9 M7 co-purchase re-weighting · 10 M10 Dirichlet shrink ·
11 M15 EFM direct-match term · 12 M6 rule tokens · 13 M4 FISM item-item term · 14 M14 two-vocabulary concat ·
15 M3 target-conditioned attention · 16 M8 complement term.

Rejected outright: FM/NFM pair terms, AFM attention over word pairs, explicit pair-row table, Janossy k=2 MLP, IDF/BM25/SIF
weights (redundant), PPMI weights, NL/LLM profile bottleneck, post-hoc Amazon-style lookups.

Explanation-layer only, ship regardless: X1 item examples under each word/prototype; X2 deviation-from-crowd display
with standard errors.

Families the top-down menu missed (angle G): F3 neighbour-user additive term (exact, thin-history-tolerant, but a second
path beside the prototypes); F4 LightGCN-style propagation inside q_u (reject: crowd words in "your history"); **F5
mixed-membership user over word-topics** (the only nameable *replacement* for the prototype layer; a different host,
not an fU′ knob — future work / part two); F6 multi-interest sub-profiles (ml-1m only, breaks additivity); F10 RETAIN
exactness constraint (a requirement on any attention variant, keeps per-(purchase, word) attribution exact).

Answer to the two framing questions:
- Richer vocabulary inside the contract? Yes, three: signed deviations from the crowd (M9), within-item conjunction
  phrases (M5), the purchased items themselves (M1). Only M9 acts where the accuracy is lost.
- How much to alter the host's use of interactions? As little as a shift (M9) plus hygiene (M2), because the ID row
  absorbs anything bigger, the loss is not in the composition, and every larger form flips meaning between hm and ml-1m.

---

## 5. Shortlist for a possible fU′ cycle (NO decision taken; Matteo decides tomorrow)

1. **M2 first, as hygiene, not a candidate.** Leave-one-out pooling in training for fU *and* lightfm_hist (the user
   extractor must receive the positive's word codes in the training branch). Re-run the two winning hm configs. Every
   later read-out is contaminated until done.
2. **fU′ = population-mean centring (M9), noid arm primary, ids arm as pre-registered D2 read-out** — the mirror of the
   fI′ cycle already queued; the one form with a prototype-layer mechanism and an interpretability gain
   (deviation-from-crowd unit). Expect the ids arm to re-house μ in e_ID as fI′-ids did. Cheap: module flag mirroring dc06.
3. **M1 item-ID history rows only after 1 and 2**, with `_noid` twin, norm read-out ‖Σy‖/‖Σc‖ and per-prototype
   name-coverage share pre-registered. Honest expectation: lightfm_hist-style lift, prototype-layer gap untouched.

Not shortlisted: M5 until the user-side conjunction instrument shows non-zero encoding; M4/M15 (bypass the prototypes);
M3 (profile stops being a per-user object); F5 (different host; record as future work).

## 6. Cheap instruments (login-node class, working evidence)

1. User-side H1/D2 twin of the item probe (‖mean q_meta‖/mean‖q_meta‖ and same for e_ID) on fU hm + ml-1m checkpoints;
   reuse `id_row_probe.py` Section B.
2. Self-inclusion cost: one M2 retrain of the winning fU-noid and lightfm_hist configs on hm (~1 h each), read the delta.
   Only item here that trains.
3. Does within-history pair signal exist at all: lift of item pairs / cross-field value pairs vs chance, same-day vs
   cross-day, hm users |H| ≥ 5.
4. User-side conjunction encoding: regress e_ID(u) on within-item conjunction indicators controlling for unary counts.
5. Name-coverage share per user prototype on today's checkpoints (baseline for any M1/M14 run).

## 7. Points that feed the thesis text

Hidden-effects/limitations: P3, P4, P5 (disclose, control shares it), P6, and the two-part faithfulness rule for any
co-occurrence sentence. "Why not attention / pairs": cross-item pairs information-free given n_u; attention weights exact
multipliers with unnameable reasons (Jain & Wallace). Scrutability: fU linear-editable like Balog 2019, ID row is the
non-scrutable residual; M9 makes the edit unit "deviation from the crowd", the unit human studies endorse. Fairness:
word-named user prototypes make the ProtoMF §5.3 gender proxy auditable by name on hm. Future work: mixed-membership
topic host (F5); fUfI merge owns item pairs, complement terms, shared-vs-separate tables.

---

## 8. Open threads to pick up tomorrow

- [ ] Matteo reads the memo and decides: M2 hygiene yes/no; fU′ centring cycle yes/no (would be a `/design-candidate`
      cycle mirroring fI′); which instruments to run on the login node.
- [ ] **Still untouched from the killed session:** put the fU arms on the 100-trial config to match fI (Matteo's request
      at 17:41: "just like fI I need for fU to have everything in 100 trial config for better comparison"). Needs a
      run plan and one `/leo5-submit` per run.
- [ ] Protocol entry not yet written (suggested topic: fU vocabulary research verdict — mean-of-words pathologies,
      centring as the one prototype-layer lever, train-time target self-inclusion, topic-model host as future work).
- [ ] Nothing committed: memo, pickup note and the research folder are untracked under `Master/temp/`. Also from the
      parallel fI session: naming fix 1410c6f and probe instrument 06e4292 are on feat/scrutiny, not pushed.
- [ ] Verify the P5 self-inclusion finding once more by reading the code yourself before acting on it (it was
      verified by an agent and spot-checked in-session, but it drives a retrain).

## 9. Files

- Synthesis memo (full): `Master/temp/user_vocabulary_deep_research_2026-09-07.md` (240 lines, §0–§7)
- Raw agent notes: `Master/temp/user_vocabulary_research_2026-09-07/` — `BRIEF.md`, `MENU.md`, `angle_A.md` (items as
  vocabulary), `angle_B.md` (pairs/co-occurrence), `angle_C.md` (profile as estimator), `angle_D.md` (scrutable user
  models), `angle_E.md` (prototype layer + contract per form), `angle_F.md` (adversarial critic + ranking),
  `angle_G.md` (bottom-up survey taxonomies, cross-domain)
- Related: `Master/docs/design_candidates/dc05_history_composed_user_factors.md`, `Master/docs/scrutiny/sc05_chapter_draft.md`,
  fI′ plan `Master/temp/composition_upgrades_plan_2026-08-24.md`
- Citations: verified vs `[unverified]` marked per angle file; `paper-search` MCP was down all day, retrieval via
  WebSearch/WebFetch. Tier: working evidence / design input, NOT thesis text.
