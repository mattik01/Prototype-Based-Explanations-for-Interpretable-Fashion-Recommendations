# Angle F — adversarial critic; ranking along "how much do we alter how the host uses the interaction signal" (2026-09-07)

Menu fixed (MENU.md). Code facts checked this session: `feature_extraction/feature_ids.py:445-561`
(`build_user_history_weights` → `(hist_value_ids, hist_weights, n_features)`, full train file, dedup on
(user,item), w̄ = n/|H|, padded buffers), `feature_extractors.py:383-471` (`HistoryFeatureEmbedding.forward`:
static buffer lookup by user index, ID row at `n_features + u`), `feature_extractor_factories.py:167-220` (fU) and
`:320-350` (`lightfm_hist`: the SAME `HistoryFeatureEmbedding`), `feature_ids.py:586-670` (`inject_feature_ids`:
the two ft_types share one builder call), `rec_sys/protomf_dataset.py:202-231` and `rec_sys/rec_sys.py:89-110`.
Sizes: hm_1_month 73,418 users / 13,651 items (`s0_foundation.md` l.55-56); ml-1m 6,040 / 3,706.
Today's probe (`instruments/id_row_probe/results_memo.md`): the D2 0.07→0.49 reading is on the **item-side**
fI′-ids arm (ml-1m); the user side has never been probed — I use it as the general lesson, not as a fU measurement.

## 0. Standing objections that cut across the menu

O1. **The D2 lesson generalises: every form that is a fixed function of the user's own train row is
absorbable by e_ID(u).** Of M1–M16 only M3 (target-conditioned) escapes this in the ids arm, because e_ID is
target-independent. So in the ids arm most forms change *inductive bias and attribution*, not the function class;
the honest test bed for M5–M7, M9–M13 is `_noid`, and any ids-arm claim needs the D2/ID-share read-out
pre-registered *with a magnitude floor* (memo lesson 5).
O2. **The measured loss is at the prototype layer** (lightfm_hist +0.06–0.09 with the same q_u). Composition
forms (M1, M5–M7, M9–M14) can at best move the noid arm and the explanation; only M4, M9 (mechanism: removes the
shared direction the cosine cannot ignore) and M15 touch the layer where the loss sits — and M4/M15 do so by
adding the winning dot-product path *beside* the prototypes, which makes the prototype bars decorative.
O3. **Two datasets, one form, two meanings** is the default, not the exception: hm history = 5 same-day
purchases × 5 fixed fields; ml-1m = 56 movies × 3–72 tokens. Below, "silently changes meaning" is flagged per form.
O4. **Nesting.** info(U-ProtoMF) ⊂ info(fU) ⊆ info(fI) holds for every form whose q_u is a function of (own row,
item attributes, shared learned tables). M6/M7 inject *population co-occurrence statistics as fixed inputs*
(cross-user, non-learned) — a channel fI does not have; the record's nesting sentence must be amended if either is
run. M1/M14 add a history item-ID table: still a function of the interaction matrix, but fI-comparability now
means "same word table + a second table fI lacks on the user side".

## 1. Per-form audit

| # | 1 capacity vs ID row (noid-only?) | 2 nesting / control | 3 thin-hm ↔ ml-1m | 4 units seen, 5-purchase user | 5 cost |
|---|---|---|---|---|---|
| M1 | Adds n_items×d rows. NOT a covert user ID on either dataset (rank(H) ≤ items < users: 13,651<73,418; 3,706<6,040). With e_ID present the ID row can absorb per-user fit; y_j only earns its keep as *sharing* across baskets (weight decay favours it). Meaningful in both arms; decisive in noid. | Nesting kept (function of matrix). Control: needs `lightfm_hist(+y)` — free, same module. fI comparability: the per-purchase composition c_j+y_j becomes the exact mirror of fI's item c_i+e_ID(i) — cleaner, not worse. | hm: tail items' y_j (1–10 buyers) stay ~untrained → falls back to words (good). ml-1m: 56 well-trained y_j per user → NSVD with a decorative word table; words demoted at popular items (A-F4). Meaning flips between datasets. | 25 word instances + 5 "this article itself" = 30 (≈15–20 distinct words + 5). | **Builder-only**: append (n_features+item_id, 1/|H|) codes, pass n_features' = V+n_items; module agnostic; renderer needs item labels. Acute item-6 leak (see §2) → must ship with M2. |
| M2 | No capacity change. Removes the positive's 1/|H| self-share from training q_u. | Applies identically to fU and lightfm_hist (shared module) → comparability kept if applied to both; fI unaffected (item side has no history). | hm |H|=5 → trains on 4; with k-core the train history is ≥ core−2, so no user empties (hm_1_month median 5 ⇒ its core ≤ 7; verify the actual value). ml-1m: 1/56 self-share, near-benign. | unchanged | **Not a knob.** `RecSys.forward` passes only `u_idxs` to the user extractor; the positive's word codes must reach `HistoryFeatureEmbedding` (training-only branch: q − (1/|H|)c_i, rescaled |H|/(|H|−1)). Small forward/RecSys change, no new ft_type. |
| M3 | Context-dependent → the ONE form e_ID cannot absorb; real capacity in both arms. But the "user profile" object dissolves: prototype bars become per (user, candidate). | q_{u,i} still f(own row, attrs, target). Control: NAIS itself (attention over plain dot) — a new control row. fI comparability lost (fI has no target-side conditioning). | hm: softmax over 5 items has nothing to discriminate; c_j∈R^d vs t_i∈R^L need a projection (new params). ml-1m: NAIS's home turf; β-smoothing issue is real. | 5 disclosed weights + 25 words, *per recommended item* (profile no longer static). | **New ft_type** (PrototypeEmbedding forward needs item index). |
| M4 | New score term Σ_j⟨t'_i, y_j⟩: two new tables. Absorption moves up a level: whatever the dot path can fit, it will, and it fits better (O2) → prototype path decorative. | Nesting kept. Control = FISM+MF integrated — not lightfm_hist. | hm: 5 direct terms, legible. Cold candidates lack t'_i → term=0 → warm and cold candidates scored on different scales (mixed rankings biased against cold). ml-1m fine. | 5 direct-item terms + prototype object = two objects. | **New ft_type** (score composition in RecSys). |
| M5 | Vocabulary rows shared across users — same status as words, i.e. absorbable per user like words are. Item-side probe (memo A): ID row is NOT linearly predictable from up to 5000 pair columns (R²≈0 all arms) → the item side did not *need* conjunctions; user side unmeasured. | Nesting kept exactly; fI inherits the same table → a two-model vocabulary cycle. lightfm_hist inherits via builder. | hm: +10 conjunctions per purchase (C(5,2)); support pruning keeps rows warm. ml-1m: C(72,2)=2,556 per movie → must prune hard; "conjunction" = arbitrary tag pair, not a product type. Meaning flips. | 25 words + up to 50 conjunctions = 75 → illegible without collapsing. | Item-builder change (`build_feature_ids/_bags` emit pair codes); user builder inherits; config afterwards. |
| M6 | Fired rules are a fixed AND-function of the history → fully absorbable; **noid-only**. If consequents reuse word rows it is a pair-conditioned re-weighting of existing words. | **Breaks O4**: rule confidences are cross-user statistics fed as input. Control must get the same tokens (builder-shared, ok) but fI has no analogue. | hm: 25 words → many attribute-level antecedent pairs fire → consequent mass floods q_u with crowd words. ml-1m: rules over 1,061 tags; mining cost and count explode. | fired rules: easily 10–50. | Builder-only (mining preprocessing + appended codes/weights). |
| M7 | Per-purchase weight from population co-purchase with the rest of the basket; context-dependent but a fixed function of own row → absorbable; **noid-only**. | Breaks O4 (population edges as input). Control inherits via builder. | hm: 94% same-day → measures basket coherence, explained as taste. ml-1m: co-rating counts scale with popularity → silent popularity up-weighting. Meaning flips. | 5 multipliers + 25 words = 30. | Builder-only (weight column per purchase instead of 1 in `long_df`). |
| M9 | Shift. ids arm: function-class redundant (e_ID' = e_ID − μ); expect the fI′-ids D2 result (0.07→0.49) to recur on the user side. noid: genuine shift. The "count-dependent shrink toward zero deviation" is a per-user scalar on (m_u−μ): killed by the cosine in noid, absorbed in ids → **a no-op in both arms; drop it from M9**. | Nesting kept (p̄ is an attribute statistic). Control inherits via module. fI′ is the mirror → symmetric cycle. | hm: deviations at 0.2 granularity, SE ≈0.1–0.2; long tail of tiny negative deviations for ~400 unheld words → must be collapsed into one crowd unit. ml-1m: fine. | 25 signed word deviations + 1 crowd unit (+ ID row) = 27. | Small module change mirroring dc06 (`center` flag + registered μ; builder supplies p̄). A dense negative-weight builder trick works but costs 73k×426 buffers. |
| M10 | ids: invisible. noid: direction moves with |H| → thin users pulled to the crowd → *more* collapse at the prototype layer, which already sits at ~5–6 effective clusters. **noid-only and counter-productive at hm depth.** | Nesting kept. | hm: κ≈5 ⇒ half crowd; ml-1m: λ≈0.9, ≈plain mean. Meaning flips. | 25 words + crowd share = 26. | Builder-only (dense crowd codes) or module. |
| M11 | Fixed function of own row → absorbable; **noid-only**. Word credit now cites a field statistic. | Nesting kept. | hm: entropy from 5 draws biased low → all gates ≈1 → ≈plain mean. ml-1m: "field" = 1,061-value tag bag → normalised entropy meaningless. Degenerate on hm, ill-defined on ml-1m. | 25 words + 5 field gates. | Builder-only. |
| M12 | Data-dependent 0/1 gate → not absorbed in noid; absorbed in ids. | Nesting kept (info shrinks). Balog's significance needs ratings; on implicit data the criterion collapses to top-k-by-count. | hm: ≤20 distinct words → k≥15 is a no-op. ml-1m: real compaction (≈300 distinct tokens → k). | ≤ k. | Builder-only. |
| M13 | Sub-linear per-(user,word) count → absorbable; **noid-only**. | Nesting kept. | hm: n≤5 → ≈linear → no-op. ml-1m: compresses genre counts up to 56 → real. | unchanged. | Builder-only. |
| M14 | Strict sub-case of M1 (block-diagonal prototypes = orthogonality constraint on M1). Same absorption story; less capacity; the norm couples the blocks so "which block lit it" is exact only in the numerator. | As M1, plus the block constraint makes prototypes half-nameable (word block) and half post-hoc (ID block). | As M1; the concat only pays off when an MLP follows (YouTube); under a linear cosine it is M1 with a constraint. | 30. | Builder (offsets) + module (block masks) → effectively new ft_type. |
| M15 | Score += α Σ_{c∈words(i)} x_{u,c}: a parameter-free CBF overlap term (or per-word weights). Not absorbable by e_ID (it is item-attribute-side) → real in both arms — and it is the item-6 content leak made into a feature. | Nesting kept. Control: lightfm_hist + same term. fI comparability: fI could take the mirrored term. | hm: repeat-type buying → term strong, cold-safe. ml-1m: big bags → overlap scales with bag size (popularity proxy 0.67) → silent popularity term. | + |words(i) ∩ profile| ≤ 5 units, the most legible sentence in the menu. | Factory/RecSys change: fU host's item branch is a plain `embedding`; needs an attribute lookup on the item side → new ft_type or a RecSys score-addend knob. |
| M16 | Fixed cap restricts the class (learned α is absorbed). ids-arm only by definition. Does NOT touch D2: a shared direction can be housed at any norm. Fights the norm handover (‖q_words‖ shrinks with |H| → cap shrinks → ID share pushed down for long histories). | Nesting kept; control inherits via module. | hm: cap bites where ‖q_words‖ is largest (short histories) — i.e. where the handover is weakest; ml-1m: the opposite. Effect direction differs. | unchanged (+1 disclosed share number). | Small module change (projection/reparam of e_ID). Config-only after that. |
| X1/X2 | No score change; X2's SE display is honest at |H|=5 (binomial SE 0.1–0.2 says most deviations are noise). | — | — | X1: +1 example per word; X2: 25 words annotated. | Renderer only. |

## 2. Item 6 — is the scored positive inside q_u during training? (verified)

TRUE. `build_user_history_weights` consumes the whole `listening_history_train.csv` (dedup on (user,item)) and
`HistoryFeatureEmbedding.forward` looks up static per-user buffers by `o_idxs` only; `ProtoRecDataset.__getitem__`
returns `(user_idx, [pos, negs], labels)` and `RecSys.forward` computes `u_embed = self.user_feature_extractor(u_idxs)`
once per row, never seeing the item indices. Hence for every training row the positive's own words carry weight
1/|H_u| inside q_u (20 % of the metadata mass at hm median 5; 1.8 % on ml-1m). `lightfm_hist` shares it exactly
(same module, same injection seam) — so the fU-vs-control comparison is not biased by it. At test the target is by
construction absent from the train history → the pair (q_u, target) is drawn from a different distribution in
train and test: a **real train/test mismatch** of the FISM `\{i}` kind, not a bug in the code's own terms (SVD++
also keeps the rated item in N(u); FISM/NAIS exclude it). With words only it is a *content* leak: the optimiser is
rewarded for aligning t_i's prototypes with i's own words — the same content signal that also generalises, so I
grade it moderate, not fatal; with item-ID rows (M1/M14) it becomes memorisation of the positive and is fatal. It is
also the reason M15's overlap term will look better in training than it deserves. Verdict: M2 is hygiene the
current fU and control both owe; cheap; run it on both arms before any composition claim. Not a vocabulary answer.

## 3. Ranking (exact attribution first; host alteration none → re-weighting → new terms → new object; ID-redundancy tiebreak)

All M-forms keep exact additive attribution (M3 only as coefficients of opaque origin; M14 only in the numerator).
1 **M2** (none; training hygiene) · 2 **M9 without the shrink** (shift) · 3 **M16** (constraint) · 4 **M5** (host none,
vocabulary) · 5 **M1** (new rows in q_u, builder-only) · 6 M12 · 7 M13 · 8 M11 · 9 M7 · 10 M10 · 11 M15 · 12 M6 ·
13 M4 · 14 M14 · 15 M3 · 16 M8. X1/X2 outside the ranking: ship regardless.
Why M5 is not above M16: both are "none" for the host, but M5 has today's negative item-side evidence (U4 cold) and
the 75-unit legibility failure; M16 is the only form that acts on a *measured* fU defect (norm handover) in the arm
that actually ships (ids). Why M15 sits at 11 despite the best sentence: it adds the winning path beside the
prototypes — the contract survives, the prototype story does not. Re-opened rejections: none; I agree with all eight.

## 4. Adversarial verdicts, top five

**M2.** The strongest thing about it is that it is not a design idea — it is the one change nobody can argue with,
and it exposes whether the 20 % self-share inflated hm numbers for fU *and* lightfm_hist alike. Weakness: it can
only lower the dev numbers; if fU drops more than the control, the composition looks worse, not better. It changes
no explanation and answers none of Matteo's question. Do it first anyway, on both arms, because every other form's
read-out is contaminated until it is done.

**M9 (centring, no shrink).** The only composition form with a causal story at the prototype layer, and the only one
whose re-attribution ("the crowd baseline" as a nameable unit) is an interpretability gain rather than a cost. The
adversary's case: today's item-side probe realised exactly the predicted failure — the ids arm re-housed the removed
direction in the ID block (D2 0.07→0.49, ID share on the tail 0.52). Expect the same on fU′-ids; so the cycle is a
noid-arm cycle with the ids arm as a pre-registered D2 read-out, and the "+shrink" half of the menu entry is dead
weight in both arms. It also duplicates the fI′ cycle already in flight — cheap to mirror, but not new evidence.

**M16.** Honest about what fU actually shows (norm handover, head/tail ID share 0.84/0.19 on the item side). But
a norm cap is blind to the thing that hurts (a shared direction in the ID block) and its bite goes the wrong way on
hm (strongest where histories are short). Verdict: a disclosure device dressed as a form; run only as an α-control
beside M9, never alone.

**M5.** Contract-perfect and host-free, and the only "pair" that carries new information (B-F2). The adversary
points at the memo: 5,000 pair columns explain nothing of the item-side ID row. If the model wanted conjunctions it
had a free row to put them in and did not. Add the shopper cost (up to 75 units) and the ml-1m explosion, and M5 is a
vocabulary paper, not a fU cycle — unless a user-side probe first shows non-zero conjunction encoding in e_ID(u).

**M1.** The one form that literally answers "use the interactions, not just their words", it is builder-only, and it
is provably not a covert user ID on either dataset. The adversary: in the ids arm the free row will out-compete y_j
for the fit; on ml-1m y_j will out-compete the words and fU becomes NSVD with a word-shaped explanation; and without
M2 it memorises the positive. So: noid twin mandatory, M2 mandatory, norm read-out (‖Σy‖ vs ‖Σc‖) mandatory, and the
honest expectation is a lightfm_hist lift with the prototype-layer gap untouched (O2).
