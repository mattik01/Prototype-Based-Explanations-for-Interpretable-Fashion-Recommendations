# Scrutiny Findings Ledger

> Maintained by the Scrutiny Protocol (`Master/docs/scrutiny_protocol.md`).
> One entry per non-trivial finding; trivial fixes get a one-line entry.
> **This ledger owns the WHY** (finding, severity, disposition, resolving
> commit); `Master/docs/modifications_log.md` owns the file-centric WHAT,
> cross-referenced by finding ID. Rationale is written once, here.
>
> **Schema per entry:**
> - ID: `F-S0-NN` (foundation) or `F-DC0N-NN` (candidate-scoped), sequential
> - applies-to: `shared` | `dc01`..`dc04` — `shared` findings are re-checked
>   at every later candidate's SC.1a
> - severity: `trivial` | `minor` | `major` | `blocker`
> - status: `open` → `fixed(<commit>)` | `wontfix` | `superseded(<by-ID>)` | `stale`
> - body: the finding, then `**Proposal:**` (the bundled fix), then
>   `**Disposition:**` (gate decision + date)
>
> Entry template:
>
> ```
> ### F-DC0N-NN [severity] [applies-to] [status]
> <one-sentence finding>
> <evidence / where>
> **Proposal:** <the bundled fix, implementation-ready>
> **Disposition:** <gate decision, date, resolving commit if fixed>
> ```

## Foundation (F-S0-…)

### F-S0-01 [trivial] [shared] [fixed]
`hm_feature_analysis.md` Part E claimed the splitter dedups `(customer, article, date)`; `data/hm/hm_splitter.py` dedups `(customer_id, article_id)` keep-first (ALL repeat purchases removed, not just same-day). Doc corrected to match code (S0.1, 2026-07-11). Commit: see S0.1 artifact commit.

### F-S0-02 [minor] [shared] [wontfix]
Eval/train negative sampling uses the global numpy RNG inside DataLoader workers, which fork without numpy reseeding: with `num_workers>0` both workers run identical RNG streams and the parent state never advances, so val negatives are bit-identical across epochs; with `num_workers=0` (Windows) negatives are re-drawn every epoch — two different eval regimes by platform.
Evidence: `rec_sys/protomf_dataset.py` `_neg_sample_uniform`/`_neg_sample_popular` (np.random, no `worker_init_fn`); `utilities/consts.py` `NUM_WORKERS` platform switch.
**Proposal:** document-only (this entry + S0.1 dossier §3) and **wontfix**: all scrutiny runs are LEO5-only (`num_workers=2`, consistent regime); cross-worker stream duplication is masked per-user by the exclusion vector and has no plausible metric impact; the epoch-fixed val negatives are actually beneficial for early-stopping stability. Alternative (not recommended): `worker_init_fn` with fixed per-worker numpy seed — would shift every sampled metric and buy nothing scientific.
**Disposition:** gate 2026-07-11 — accepted as proposed: document-only, wontfix.

### F-S0-03 [minor] [shared] [fixed(543e2d5)]
`Evaluator.get_results()` divides accumulated metric sums by `n_users`, silently assuming every user contributes exactly one eval row. Holds today (verified: ml-1m 6,034; amazon2014 6,950; hm_3_month 256,701 val=test=n_users; hm_1_month pending — F-S0-05), but any future split/eval change (cold-start eval!) breaks it silently.
Evidence: `utilities/eval.py:87`; `trainer.val()` / `tester.test()` construct `Evaluator(dataset.n_users)`. *(Edit 2026-07-11: hm_1_month verified after F-S0-05 resolution — val = test = 73,418 = n_users, one row per user. Invariant now confirmed on all four in-scope datasets.)*
**Proposal:** S0-build adds a cheap assert (total evaluated rows == n_users) in `Trainer.val()` and `Tester.test()` (no behavior change while the invariant holds; test in `dc_checks/s0/`). Constraint recorded for S0.3: the cold-start evaluator must define its own divisor, never inherit this one blindly.
**Disposition:** gate 2026-07-11 — accepted: assert lands in S0-build (stays open until resolving commit); S0.3 divisor constraint recorded. **Resolved S0-build (543e2d5):** implemented inside `Evaluator.get_results` (divide by counted rows + assert vs the declared expectation) rather than in `Trainer.val()`/`Tester.test()` — the assert thereby covers both call paths plus any future caller; expectation arg optional (None = caller owns its divisor, the S0.3 constraint honored). Bit-identity while rows == n_users pinned by `dc_checks/s0/t01` (11 checks green). Built under the user's away-authorization; component gate pending batch review.

### F-S0-04 [minor] [shared] [wontfix]
Standard test sets already contain train-unseen items scored on random-init (never-trained) ID embeddings: amazon2014 101 items → 490/6,950 test rows (**7.1%**, +140 val rows); hm_3_month 23/256,701 rows; ml-1m 0. Paper-faithful accident of k-core + temporal LOO (k-core counts include val/test occurrences).
Evidence: S0.1 empirical check (2026-07-11, item-set diff train vs val/test). *(Edit 2026-07-11: hm_1_month measured after F-S0-05 resolution — 3 train-unseen items, 9/73,418 test rows ≈ 0.012%. Negligible on the primary dataset; amazon remains the only materially affected set.)*
**Proposal:** **wontfix** in the headline eval (paper comparability). Consequences routed instead: (a) S0.3 defines cold-item sets *deliberately* rather than relying on this accident; (b) thesis discloses the amazon 7.1% slice when interpreting replication numbers.
**Disposition:** gate 2026-07-11 — accepted as proposed: wontfix, routed to S0.3 + thesis disclosure.

### F-S0-05 [minor] [shared] [fixed]
`data/hm_1_month/` — the V1 primary scrutiny dataset — does not exist on the laptop (splits gitignored, live on LEO5/GPU machine); LEO5 unreachable today (no VPN). Blocks local S0.5 (signature-collision recompute needs V1 `item_features.csv`) and the F-S0-03 invariant check for V1.
**Proposal:** next VPN session, `scp` the 6 split files + `item_features.csv` from LEO5 to `data/hm_1_month/`. Prefer copy over local rebuild: `hm_splitter.py` uses non-stable `sort_values` (quicksort), so a rebuild is not guaranteed byte-identical to the canonical artifact.
**Disposition:** resolved 2026-07-11 same day — VPN restored (Cisco Secure Client replaced by OpenConnect NM profile after reconnect-loop instability), all 6 files scp'd from `leo5:/scratch/c7031336/protomf_data/hm_1_month/` (byte sizes identical to source; counts match the canonical 623,230/73,418/13,651). No resolving commit — the artifacts are gitignored data files. Sanity re-checks recorded under F-S0-03/F-S0-04.

### F-S0-06 [minor] [shared] [fixed(a46d4e8)]
NaN-policy asymmetry in `feature_extraction/feature_ids.py`: `build_attr_multi_hot` raises on NaN cells, but `build_feature_ids` calls `astype(str)` first and would silently encode NaN as a legitimate `'nan'` vocab value (a phantom feature). Currently inert — V1 `item_features.csv` has 0 NaN cells across all 9 columns (measured 2026-07-11, S0.5), consistent with the 4.0 no-missingness finding.
Evidence: `feature_ids.py:67` (`vals = df[field].astype(str)`, no isna guard) vs `feature_ids.py:126-129` (explicit raise).
**Proposal:** add the symmetric NaN guard to `build_feature_ids` in S0-build — no behavior change on current data; covered by a new `dc_checks/s0/` test feeding a NaN row and expecting the raise.
**Disposition:** gate 2026-07-11 (S0.5) — accepted; lands in S0-build (stays open until resolving commit). **Resolved S0-build (a46d4e8):** symmetric guard added; `dc_checks/s0/t02` pins the raise, builder symmetry (incl. the pandas literal-`'nan'`-parses-as-NaN case), and V1 cleanliness (0 NaN, V=426 under the canonical 5). dc01 t01/t06 regressions green. Built under away-authorization; component gate pending batch review.

### F-S0-07 [major] [dc01] [fixed(865de3b)]
The S0.3 §5 convention "dc01: ID column dropped at cold inference" has no implementation anywhere: `cold_eval.build_model` rebuilds a `feature_item_proto` config natively, so a dc01 run with `use_id_feature=1` would score cold items *including their untrained per-item ID embedding row* — the same silent-depressant class as the bias gap, one level down. Dormant today (no dc01 cold run exists; no dc01 row in the S0.7 reference fleet).
Evidence: S0.7 black-box audit (`s0_7_audit_cold_machinery.md` §5, divergence #1); `grep cold feature_extraction/*.py` empty.
**Proposal:** two-part. (a) Now (S0.7): `cold_eval.load_config` refuses native cold eval of `ft_type='feature_item_proto'` configs with `use_id_feature` enabled — loud error naming the missing drop, symmetric to the `use_bias` guard; test in `dc_checks/s0/`. (b) The actual drop (zero/skip the cold items' ID rows at cold scoring, per dc01 design doc §3.4/§3.7.6) is implemented and tested at dc01's SC.3, where that design doc is in scope.
**Disposition:** S0.7 gate 2026-07-11 — approved as proposed. Guard landed (t06 pins refusal incl. the absent-key=default-True case; `_noid` passes). The generic drop primitive already covers the direct FeatureEmbedding case (see F-S0-13); the dc01 nested case stays REFUSED until dc01 SC.3 verifies it. Guard commit: 30a570c. **Resolved dc01 SC.3 (865de3b, 2026-07-12):** part (b) closed by *verification*, not new machinery — the SC.1a code audit found `_find_feature_embedding` already recurses to dc01's nested branch, and the R1 tie (ONE shared `FeatureEmbedding` instance feeding both item halves) means the single drop covers both halves of the UI-score. Pinned by `dc_checks/dc01/t11` (29 checks: cold repr == pure feature composition in both halves, warm rows bit-identical, drop == manual zeroing state-equality, `_noid` no-op, full-runner e2e on a trained dc01 toy checkpoint incl. report entry + kNN-seam skip + restore/re-apply). The load-time refusal is lifted and REPLACED by `config_expects_id_drop` + a post-drop abort in `run_cold_eval` — configs whose item branch is tags+ids (dc01/lightfm, absent key = factory-default True) abort loudly if the drop cannot apply, so future structure drift cannot silently mis-score. s0/t06 updated to the new semantics. dc01 `_ids` cold runs are hereby unblocked for SC.6.

### F-S0-08 [minor] [shared] [fixed(30a570c)]
The popularity reference row is degenerate on cold-vs-cold: all 100 candidates have popularity 0 → sigmoid maps all to 0.5 → `bn.argpartition` resolves the 100-way tie by introselect internals, which deterministically exclude index 0 — the row reads HR@10 = 0.00 instead of the advertised 0.10 chance floor (verified empirically, both first-party and by the black-box audit). Relatedly, raw-count scores saturate `torch.sigmoid` at 1.0 for pop ≥ ~17, collapsing warm order on cold-vs-all.
Evidence: `utilities/cold_eval.py:187-201` (`PopularityScorer`), `utilities/eval.py:18`; empirical all-tie probe 2026-07-11.
**Proposal:** `PopularityScorer` returns the dense popularity *rank* scaled to [0,1] plus small seeded uniform tie-break noise (deterministic under the run seed; equal-pop ties break randomly; no sigmoid saturation). Report the analytic chance line for all K on cold-vs-cold. New `dc_checks/s0/` check: all-tie input reads ≈ K/100 under the new scorer.
**Disposition:** S0.7 gate 2026-07-11 — approved; implemented as proposed (noise amplitude = half a rank gap, provably never reorders distinct popularity levels; t06 pins range, order-preservation, chance behavior on all-tie pools, seed determinism). Chance line now reported for all K. Status: **fixed** (30a570c).

### F-S0-09 [minor] [shared] [fixed(30a570c)]
Variant warm val/test negative sampling excludes only *variant* consumption, so a user's canonically-consumed cold items (their actual removed purchases) can be drawn as warm-eval negatives (~1.2% of rows) — spec-silent, uniform across models, but directionally biased: it penalizes exactly the models that score removed purchases high (cold-generalizing ones), including in early stopping.
Evidence: S0.7 audit B3; `ProtoRecDataset.load_data` CSR construction (variant files only).
**Proposal:** on marked variant dirs (`cold_items.csv` present), `ProtoRecDataset` val/test additionally folds the user's `cold_test.csv` rows into the exclusion CSR; canonical behavior untouched; t05 extended. Alternative (not recommended): document-accept.
**Disposition:** S0.7 gate 2026-07-11 — approved; implemented (exclusion-CSR fold-in, positives untouched; t06 sweeps full variant val+test — 0 violations — and pins canonical no-op). Modifications log cross-ref added. Status: **fixed** (30a570c).

### F-S0-10 [minor] [shared] [wontfix]
The S0.3 §6 "dev-profile epochs" retrain convention is operator-memory only: `run_combo.py` defaults to `--profile production`, whose `n_epochs=100` always overrides the saved config's epochs (the `RETRAIN_CONFIG_KEYS` inclusion of `n_epochs` is dead in the CLI path). Forgetting `--profile dev` silently deviates from charter C4.
Evidence: S0.7 audit divergence #5; `run_combo.py:470-474`.
**Proposal:** when `--retrain-config` is given and `--profile` is not explicitly passed, default to `dev` with a loud print. Profile already lands in the C8 manifest.
**Disposition:** S0.7 gate 2026-07-11 — **wontfix, amended per user reasoning:** profile selection is a queue-time decision for EVERY run type (all scrutiny hyperopts equally require `--profile dev` at submission; nothing is hardwired per-model anywhere), so a retrain-only default would be the fleet's sole special case. Existing safeguards suffice: resolved profile loudly printed, non-production profiles auto-tagged in W&B, C8 manifests record it, SC.8 audits it. The S0.7 run plan carries explicit `--profile dev` on every submission line. Status: **wontfix**.

### F-S0-11 [minor] [shared] [fixed(30a570c)]
S0.3 §3 mandates the variant-warm-vs-canonical-warm sanity comparison ("removal must not distort the warm regime"), but the runner only prints the variant warm numbers — the comparison is left to the operator.
Evidence: S0.7 audit divergence #4; `utilities/cold_eval.py:246-250`.
**Proposal:** optional `--canonical-results-dir` argument; when given, `report.json`/`report.md` include the canonical run's test metrics and the warm delta row.
**Disposition:** S0.7 gate 2026-07-11 — approved; implemented as proposed. Status: **fixed** (30a570c).

### F-S0-12 [minor] [shared] [wontfix]
fp32 `torch.sigmoid` saturates at 1.0 for logits ≥ ~16.6, collapsing rank order among saturated items in every eval pass (Tester and cold runner alike) — pre-existing host behavior, uniform across all rows and all historical numbers.
Evidence: S0.7 audit B2, verified empirically.
**Proposal:** document-only **wontfix**: sigmoid is rank-monotone below saturation; removing it (or ranking raw logits) would shift every historical/replication number for zero scientific gain. The one materially distorted row (popularity reference) is fixed structurally by F-S0-08's rank-based scorer.
**Disposition:** S0.7 gate 2026-07-11 — approved: **wontfix**, documented here and in the learnings ledger.

### F-S0-13 [minor] [shared] [fixed(30a570c)]
The S0.4 §2 spec commits `lightfm_tags_ids` to "cold inference via dc01's ID-column-drop convention," but no drop existed anywhere (same root cause as F-S0-07) — and unlike dc01, `lightfm_tags_ids` IS in the S0.7 reference fleet, so this could not wait for dc01's SC.3. Discovered first-party at gate execution; both black-box audits missed it because it sat in the seam between their scopes (audit A had only the S0.3 spec, audit B did not receive `cold_eval.py`).
Evidence: S0.4 §2 spec text; `grep`-verified absence pre-fix.
**Proposal (applied):** generic `drop_cold_id_rows` primitive in `cold_eval.py` — zeros cold items' per-item ID rows in a `FeatureEmbedding` item branch (q_cold = Σ_f e_f, warm rows untouched), applied automatically by the runner after checkpoint load, re-applied after the kNN weight restore, recorded in the report. Verified for the direct (lightfm) branch by t06 (repr == pure feature composition after drop; warm untouched; no-op on tags-only/CF branches). The dc01 nested case remains guarded by F-S0-07 until SC.3 verifies it.
**Disposition:** S0.7 gate execution 2026-07-11 — implemented within the ratified F-S0-07/fleet scope; flagged to the user in the gate follow-up. Status: **fixed** (30a570c). Audit-scope-seam watch-out added to the learnings ledger.

## dc01 (F-DC01-…)

> **⟳ Host redirection 2026-07-12 (user decision at the SC.3 gate):** dc01 re-hosted
> UI-ProtoMF → I-ProtoMF (**fI-ProtoMF**). F-DC01-01..06 substance is composition-level
> and transfers unchanged (formal transfer table ratified at the re-host amendment
> gate); F-DC01-07 is parked (below). Dossier redirection note = source of truth.

### F-DC01-01 [minor] [dc01] [fixed]
Design doc §3.5/§4 field set (F=6 incl. price-band decile, "vocab ≈436") is superseded by charter C5's canonical 5-field set (V=426, price_band deferred per S0.5); the implemented config (`confs/hyper_params.py:225-231`) already uses the canonical 5 — the doc is the outlier.
Evidence: dc01 design doc §3.5 vs S0.5 gate decision + code check (SC.1a, 2026-07-11).
**Proposal:** dated amendment to §3.5 (+§4 vocab/param numbers): adopt the canonical 5, note price_band's deferral with the recorded performance-upgrade caveat, note config compliance. Bundle includes the trivial §5 S3-row edit (stale "partial image set" caveat — set verified complete 2026-07-11). *(Extended at SC.1b gate review, user directive:)* also extend §3.7 deviation 4 with its noid-arm consequence — dropping LightFM's feature-composed biases removes even the class-level popularity channel, so dc01's noid arm is strictly more constrained than the published LightFM(tags) precedent; SC.8 sets expectations for the ablation gap accordingly. *(Second extension, same gate:)* amend §3.6 with the ablation's interpretation rule — the ids-vs-noid gap decomposes into (1) popularity blindness (within-class, likely dominant), (2) genuine taste-structure loss (the honest grounding cost), (3) minus a sharing/regularization offset; raw gap never quoted as "the cost of interpretability"; D4 popularity strata separate (1) from (2). Plus the classification note: noid = collaborative filtering at attribute granularity, not CBF (LightFM §2.3; S0.4 rev. 1 LSI-LR distinction). *(Third extension, P6:)* two disclosure lines — ties resolve by deterministic ranking-code convention (disclosed, not mechanized); the twin ceiling belongs to the ablation arm, never blurred into the headline config.
**Disposition:** SC.1a gate 2026-07-11 — base accepted as proposed. SC.1b gate walkthrough 2026-07-12 — **all six accumulated extensions approved by user** (plain-language restatement reviewed item by item: tuner sentence, cold-spec pointer, popularity consequence, gap-interpretation rule, CF-not-CBF classification, twin disclosures). Amendment lands at SC.2 (status stays open until the resolving edit is applied).

### F-DC01-02 [minor] [dc01] [fixed]
The design doc has no disposition for the Steck cosine caveat (vault 2026-06-16_2102, post-dates the doc): the read-out's semantic reading (shares c_{f,k}, profiles cos(e_f, p_k)) is undefended in writing.
Evidence: SC.1a §2f analysis — the exact rescaling-invariance argument does not transfer (ProtoMF trains through the same shifted cosine it reads out), but Rashomon multiplicity, label confounding (5b-6), and the unprotected dot-product projection half survive as risks.
**Proposal:** dated amendment adding the §2f analysis + a committed SC.8 profile-validity spot-check on the real checkpoint (legible values vs the items carrying them); metric-swap/seed-stability probes deferred to full-profile stage (C4 single-seed budget).
**Disposition:** SC.1a gate 2026-07-11 — accepted as proposed; amendment lands at SC.2 (status stays open until the resolving edit is applied).

### F-DC01-03 [minor] [dc01] [fixed]
The ID-keyed item-bias cold channel (`dc01_feature_composed_bias_gap.md`, 2026-07-07) is absent from the design doc — the concept record understates dc01's ID-keyed side channels at cold inference.
Evidence: design doc §3.4/§3.7 name only the ID embedding row; the bias scalar is dispositioned only in S0 artifacts (fleet `use_bias=0` per C3; runner refusal; F-S0-07 guard).
**Proposal:** dated amendment to §3.4/§3.7 folding in the channel + its S0 disposition; feature-composed bias stays an unstacked scratchpad seed; retire the temp note with a pointer.
**Disposition:** SC.1a gate 2026-07-11 — accepted as proposed; amendment lands at SC.2 (status stays open until the resolving edit is applied).

### F-DC01-04 [minor] [dc01] [fixed]
Train/cold-inference operating-point mismatch: a `use_id_feature=True` model always trains with the ID row present, but is cold-scored on the feature-only composition (post-drop) — a point never visited in training. LightFM's own tags+ids cold result (B5 Table 1, tags+ids ≥ tags) is the precedent that this works regardless; K3's DropoutNet-style ID dropout is the unstacked mitigation if empirics disagree.
Evidence: SC.1a §2c/§2e (interaction of modes M2×M3 with the S0.3 cold protocol).
**Proposal:** dated amendment to §3.4 cold notes naming the mismatch + precedent + mitigation pointer; SC.8 interprets `_ids` cold rows with this caveat. No mechanism change.
**Disposition:** SC.1a gate 2026-07-11 — accepted as proposed; amendment lands at SC.2 (status stays open until the resolving edit is applied).

### F-DC01-05 [minor] [dc01] [fixed]
The design doc's parenthetical "prototype distinctness (which `sim_proto` rewards)" (§5 identifiability note / §3.8 M4) is imprecise: no term rewards distinctness *directly* (Eq. 4 alone permits zero-cost collapse), and the real separation effect is an **indirect, data-spread-mediated secondary effect of the coverage pair (chiefly `reg_batch`** — facility-location-style: collapsed duplicates waste coverage; empirically demonstrated by the reg-sweep's prototype-centred clusters and rising purity, vault 2026-06-15_1238). dc01-relevant residue: the effect is only as strong as the composed item cloud's directional spread, which value dominance (M2, "Solid" 58%) could shrink — a named, measurable weakening mechanism for M1/M4 (which remain open risks, NOT expected outcomes). `max_norm` does not address the directional component of value dominance.
Evidence: host Eqs. 4–5/9; `2026-06-15_1238_protomf-two-regularizers-geometric-effect.md`; `Master/experiments/reg_sensitivity/ANALYSIS.md`. Rebutted sub-claims: prototype separation absent (see above — indirect but real and host-demonstrated); Adagrad mitigation dropped (optimizer choice `{adam, adagrad}` in the inherited search space, `confs/hyper_params.py:21`). *(Entry revised at SC.1b gate review 2026-07-11 upon user challenge — the original "no repulsion / M1-M4 expected outcome" framing overclaimed.)*
**Proposal:** dated amendments — fix the parenthetical to the indirect-coverage mechanism (with vault/sweep citations), not delete the distinctness claim; sharpen the 4c M2 row (max_norm caps norms, not direction); add the spread-mediation sentence to M1/M4 (open risks with named weakening mechanism); amend the §3.8 Rashomon sketch to name its actual (indirect, host-demonstrated) levers + the K-adoption contingency (SC.1b P7). Committed SC.8 metrics: t* activation spread, prototype pairwise latent cosine, profile pairwise overlap, per-value share vs value frequency.
**Disposition:** SC.1b gate walkthrough 2026-07-12 — **approved by user, itemized (1.1–1.4), with recorded extensions:**
(1.1) parenthetical fix approved as revised after the user's challenge (indirect coverage mechanism, vault/sweep citations).
(1.2) approved with precision sharpened at the gate: max_norm never acts on profiles (angle-only); it acts three steps upstream by equalizing vote weights (ingredient lengths) in the composed sum — protects against loudness-dominance, NOT against omnipresence (equal-length e_Solid still votes in 58% of sums; the cloud tilts anyway). **User directive: the omnipresence residue is recorded as a definite risk and MUST be tested** — covered by the committed SC.8 geometry set (per-value share vs frequency, cloud spread/tilt, prototype angles, profile overlap). Amendment carries the "vote weights / loudness vs omnipresence" wording.
(1.3) approved in substance with 1.2's directive (the spread-mediation sentence IS the risk-plus-instruments record: composition shares ingredients → cloud may contract → coverage force weakens → profiles may blur; each link measurable).
(1.4) approved with user extensions: (a) sharpness and distinctness are liked but **not exhaustive** — the interpretability-metric set is itself a research question (the literature holds notions of what counts as interpretable, incl. that the *features used* be understandable rather than opaque/constructed ones); (b) the knob inventory is broader than the regularizer weights — n_prototypes and others count; (c) the regularizers should be **analyzed further** as candidate indirect knobs for sharpness/distinctness among other measures; (d) **timing scoping:** Rashomon tuning enters the thesis late (the "Interpretability quantified" part); if the knobs then prove insufficient, introducing new interpretability-targeted regularizers is a future lineage candidate ("merged prime prime") — a concern for that point, explicitly not dc01's.
Amendment lands at SC.2.

### F-DC01-06 [minor] [dc01] [fixed]
Per-feature shares are non-identifiable across strongly-associated fields: where values of different fields co-occur (near-)deterministically (product-kind clique: department/product_type/section), only the SUM of their rows is pinned by the training objective — the per-field split of that sum is gauge-like and seed-dependent, so the flagship rendered attribution performs precision the objective does not define (SC.1b P3; distinct from 5b-6 label confounding).
Evidence: q_i = Σ e_f linearity + 4.0 FD/NMI analysis (department→garment_group ≈1.0, section→index_group ≈1.0, etc.); the doc's §4 "per-value = per-field" line answers within-field roll-up, not cross-field correlation.
**Proposal:** dated amendment to §3.4 rendering notes — grouped shares over strongly-associated field cliques as the identified quantity, per-field split shown as informative-but-not-identified detail; thesis wording rule; cross-seed share-stability check recorded as deferred to full-profile stage (C4 single-seed budget). *(Extended at gate walkthrough, user phrasing "naming or breaking down a score":)* the same grouping discipline applies to profile/naming read-outs — for lockstep values each arrow is (trained sum ± frozen init noise)/2, so within-clique profile ORDERING is noise; naming treats clique-mates as one concept (clique-level aggregation), cross-clique ordering stays meaningful. *(Second and third extensions, user challenges — thresholds are production logic; final thesis framing fixed at this gate:)* the amendment presents the issue as **characterize → exhibit → remedy → delimit**, threshold-free on the research side.
**(1) Problem, exhibited:** exact characterization — for a value pair, the split's determinacy is the loss curvature along the redistribution direction (e_a→e_b, sum fixed); lockstep pairs receive identical gradients every step, so the SUM is trained and the DIFFERENCE is frozen initialization noise (a gauge freedom); first-order proxy = the pair's decoupling mass in the catalog (computed field-level for presentation, value-level where the mechanism lives; scripts committed). Plus a **recorded toy exhibit** (slots into SC.2's toy re-check): synthetic lockstep pair, multiple inits — sum stable, printed split wanders.
**(2) Remedies, both presented, distinct and complementary:** (A) read-out quotient — grouped shares/naming over merged concepts; group sums are exact, trained summands of the computed score, so intrinsicness (R2) is preserved (coarser reading, no reconstruction); (B) determinacy annotation — raw splits printed with their measured determinacy, continuum never binarized.
**(3) General resolution stated as a math problem, delegated to production:** *find a transformation of the value set into a reduced space where all splits are identifiable up to tolerance, with the map fixed, data-derived, and human-performable* (each merged group must admit one coherent name — R7 preserved through the map; "Jeans/Trousers-Denim" legal, "Jeans/Red" forbidden). Build-time token merging noted as the strongest variant (gauge never exists) — an option, not adopted in dc01 scrutiny. Tolerance choice explicitly production's, per the "production decides, research characterizes" principle.
**(4)** Seed-to-seed split stability deferred to full-profile stage (direct measurement); section's clique membership is an empirical output, not an assumption (4.0 placed it in its own context group — the "dept+ptype+section clique" in the SC.1b P3 text was the adversary's presumption).
Plain-language source of truth for the SC.2 amendment: vault entry `2026-07-11_2310_dc01-share-identifiability-problem-and-thesis-framing.md` (verbatim-capture rule) — the amendment decodes from THERE, not from this procedural summary.
**Disposition:** SC.1b gate walkthrough 2026-07-11 — **approved by user in the final form above** (characterize→exhibit→remedy→delimit, threshold-free research side, delegated production math problem, human-performability condition). Amendment lands at SC.2; toy exhibit joins SC.2's re-check round; counting script committed when code is touched. *(Script landed dc01 SC.3, commit 9f11577: `dc_checks/dc01/decoupling_mass.py` — value-level + item-weighted field-level decoupling mass, selftest included. V1 canonical-5 headline: 8,745 observed cross-field value pairs, only **2 perfect-lockstep** (3 items); weakest field pair department×section at **0.705** item-weighted mean normalized decoupling — splits on V1 are far better determined than the adversary's worst case, and section×department is indeed the closest association, confirming the "empirical output, not assumption" stance. Near-lockstep heavy pairs exist exactly where expected: Swimwear↔Womens Swimwear 0.029, Jersey Basic↔Womens Everyday Basics 0.032.)*

### F-DC01-07 [minor] [dc01] [fixed]
The read-out list misses the exact linear-half attribution — t̂ = W^t(Σ_f e_f) = Σ_f W^t e_f gives contribution(f) = u*ᵀ(W^t e_f), additive, norm-free, and exactly equal to the removal effect on the u*·t̂ half — while the adopted counterfactual re-score (5b-5 amendment) is an off-manifold evaluation (model trained under 0% missingness never sees removed-field compositions) with heuristic meaning only (SC.1b P4).
Evidence: linearity of the projection branch; §3.4 read-out list as amended 2026-06-11.
**Proposal:** dated amendment to §3.4 — add the linear-half per-feature decomposition as read-out 4 (also softens the S0.2 Block-U opacity note for dc01: the projection branch becomes feature-attributable, though not prototype-shaped); demote the counterfactual re-score to optional diagnostic with the off-manifold caveat stated. *(Extended at gate walkthrough:)* the amendment also records the host-specificity insight (user question): the linear-half read-out exists ONLY because the double-tie has a projection branch — I-ProtoMF would be all-cosine attribution (grounded prototypes, entangled shares), U-ProtoMF all-linear (clean shares, no grounded item prototypes), UI-ProtoMF carries both explanation surfaces at once — a quiet additional argument for the double-tie host, thesis material. Both new mathematical claims (exactness; counterfactual property of the linear shares) go through SC.2's toy re-check. F-DC01-06's grouping/determinacy discipline applies to this read-out unchanged (linearity does not rescue identifiability).
**Disposition:** SC.1b gate walkthrough 2026-07-11 — **approved by user: (a)+(b) both** (add read-out 4; demote re-score to diagnostic), finalized after the fantasy-land detour established the read-out's role in the four-component inventory and the lineage's stage 3. Amendment lands at SC.2. *(Scope note 2026-07-12, host redirection: the linear-half read-out requires a projection branch — in fI-ProtoMF it has no object and is **parked to the lineage's fU/fUfI stages**; the entry's own host-specificity insight ("I-ProtoMF would be all-cosine attribution") anticipated this. C7a/C7b remain verified math. Status stays fixed — the amendment was faithfully applied to the concept record; the redirection re-scopes, it does not un-fix.)*

## dc02 (F-DC02-…)

_(none yet)_

## dc03 (F-DC03-…)

_(none yet)_

## dc04 (F-DC04-…)

_(none yet)_
