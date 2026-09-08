# dc07 — stage-1 run plan (host level), pre-registered

**Written:** 2026-09-08, build session. **Status (21:30):** committed + pushed (`799d4d5`, `0b42e01`),
pulled on LEO5; dc_checks/dc07 t01–t07 green on the cluster python too; **login-node smoke PASSED**
(§1a); sc8_fu cosine regression **clean** (§1b). Ready for the four `/leo5-submit`s in §2 — after
Matteo reads §1c. Dev tier only (two-tier numbers policy).

### 1a. Smoke result (login-node A30, 2026-09-08 21:09–21:18)
`user_proto_sm × hm_1_month`, 2 trials × 3 epochs, `--gpu-per-trial 0.5`: **exit 0**, both trials
trained and reported (build print `- membership: softmax`, `- temperature: 1.585 / 0.189`),
best-checkpoint extraction + results persistence OK, test HR@10 0.257 (toy budget, meaningless).
Results: `/scratch/c7031336/smoke_runs/dc07/user_proto_sm_hm_1_month_s38210573/` (+ `smoke_…log`).
RAM peak 19.2 GB for 2 trials (same band as the dc05 smokes; 40G at conc 5 holds per the hm anchor).
Log noise: `ConnectionResetError: Connection lost` from the W&B offline service client at trial
shutdown (asyncio drain) — after the trial result was reported, non-fatal, exit code 0.
Both instruments ran on the checkpoint: `dc07_membership/readout.md`, `sc8_fu_geometry/readout.md`.

### 1b. sc8_fu regression on a cosine checkpoint
Frozen `user_proto_hm_1_month_s38210573` copied to `smoke_runs/dc07/regress_user_proto_hm/`; the
generalised script vs the stored pre-change `readout.json`: **188 shared keys, 0 differences**
(the stored run had 4 extra explicit `--hist-bands`; those keys are CLI-dependent, not code).

### 1c. ⚠ Observation from the smoke that Matteo must read before submitting
On the smoke checkpoint the memberships are **exactly uniform**: H/log K = 1.000 for all 73,418
users, mean ‖q‖ = 0.0006 (p95 0.002), top-prototype share = 1/K, winning τ = 1.585 flagged near
τ_max. I.e. within 3 epochs at wd = 0.0047 the free user table shrank to ~0, the softmax went flat,
and the model became the per-item scalar Σ_l t_{i,l}/K — the §9 popularity degeneracy, reached not
via τ but via **weight decay on ‖q‖**. Under cosine the norm was a gauge and weight decay on the
user table was harmless; under softmax ‖q‖ IS the confidence, so the host's `wd` search range
(shared optimizer knob) now directly buys the popularity solution, which hyperopt on HR@10 rewards
on hm. The stage-1 void-comparison clause (§6) catches this after the fact; the question is whether
to spend 4 runs finding out. Options (Matteo's call, not the build session's): (i) run stage 1 as
designed and let the read-outs speak (cleanest single-variable story, risk = 4 void runs);
(ii) exclude the prototype-side object table from weight decay in the `_sm/_sg` arms (one extra
variable, but arguably part of "the similarity change" since wd only became load-bearing through
it); (iii) tighten the wd range for the arms. A 3-epoch smoke with bpr + popular negatives is not
a hyperopt outcome — treat this as a warning, not a verdict.
Requirements: `Master/temp/dc07_membership_similarity_requirements_2026-09-08.md` (§6–§8 govern).

## 0. What lands on the cluster

Commit on `feat/scrutiny` (Matteo triggers via `/commit`), then on LEO5:
```
ssh leo5 "cd ~/UIFProtoMF/ProtoMF && git pull"
```
Files: `feature_extraction/feature_extractors.py`, `feature_extraction/feature_extractor_factories.py`,
`confs/hyper_params.py`, `start.py`, `Master/scripts/run_combo.py`, guards in
`utilities/explanations/{attr_readout,breakdown,history_readout,accessor/base}.py`,
`Master/scripts/{id_row_probe,sc8_fu_geometry_readout}.py`, new `Master/scripts/dc07_membership_readout.py`,
`Master/temp/dc_checks/dc07/`, docs (`modifications_log.md`, `candidate_index.md`), this plan + the
prior-art memo.

## 1. Smoke (login-node A30, smoke-one-before-batch rule) — BEFORE any submission

Quarantined results dir; Ray tmpdir on node-local `/tmp` (AF_UNIX 107-byte path cap — dc05 lesson);
W&B offline; explanations skipped (renderer refuses membership modes by design).
```
ssh leo5
cd ~/UIFProtoMF/ProtoMF
module load python/3.11.6-pytorch-2.5.1-cuda-conda-2026.02
source /scratch/c7031336/venvs/protomf/bin/activate
export PROTOMF_DATA_PATH=/scratch/c7031336/protomf_data
export PROTOMF_RESULTS_PATH=/scratch/c7031336/smoke_runs/dc07
export RAY_TMPDIR=/tmp/c7031336_ray_dc07 WANDB_MODE=offline
mkdir -p $PROTOMF_RESULTS_PATH $RAY_TMPDIR
# (1) the dc07 checks on the cluster python (torch 2.5.1 — the golden tolerance covers cross-BLAS)
for t in Master/temp/dc_checks/dc07/t0*.py; do python $t | tail -1; done
# (2) one instance end-to-end: user_proto_sm × hm_1_month, 2 trials × 3 epochs
python Master/scripts/run_combo.py -m user_proto_sm -d hm_1_month -s 38210573 \
    --profile smoke --num-samples 2 --n-epochs 3 --gpu-per-trial 0.5 --num-workers 1 \
    --wandb-mode offline --skip-explanations --wandb-tag dc07-smoke
# (3) instruments on the smoke checkpoint
R=$PROTOMF_RESULTS_PATH/user_proto_sm_hm_1_month_s38210573
python Master/scripts/dc07_membership_readout.py --results-dir $R
python Master/scripts/sc8_fu_geometry_readout.py --results-dir $R
# (4) regression of the generalised sc8_fu on a COSINE checkpoint (numbers must be unchanged vs the
#     pre-2026-09-08 script except the new `similarity` block): run it on the frozen user_proto row
python Master/scripts/sc8_fu_geometry_readout.py --results-dir /scratch/c7031336/protomf_results/user_proto_hm_1_month_s38210573
```
Pass = exit 0 on (2), the build print shows `- membership: softmax` and `- temperature: <τ>`,
both read-outs write `readout.md`, and (4) reproduces the earlier `sc8_fu_geometry/readout.json`
values (diff the JSON ignoring `similarity`). Known toy-budget edge (dc01/dc05 smokes): a nonzero
exit at best-checkpoint extraction when no toy trial improves is path-clean, not a failure.
Peak-VRAM sanity: read `nvidia-smi` during (2); the host's anchor is ~415 MB/trial (dc05 smokes),
softmax adds nothing material.

## 2. Submissions — 4 runs, one `/leo5-submit` each (hard gate)

Fleet shape (`.claude/commands/leo5-submit.md` FLEET MODE; requirements §7.3): `--profile dev`
(30 trials / 60 epochs / patience 7 / grace 4), concurrency 5 (`--gpu-per-trial 0.2`),
`--num-workers 1`, `--mem 40G`, `--gpu-type any` (A30 untyped), seed 38210573, tag `dc07-stage1`,
`--skip-explanations`. Wall-time anchors: user_proto hm 3h35m, item_proto hm 2h54m (dev);
ml-1m dev ≈ 2–3 h → `--time 08:00:00` for all four.

| # | model | dataset | command (after `cd ~/UIFProtoMF/ProtoMF`) |
|---|---|---|---|
| 1 | `user_proto_sm` | `hm_1_month` | `bash slurm/run_combo.slurm -m user_proto_sm -d hm_1_month -s 38210573 --profile dev --gpu-per-trial 0.2 --num-workers 1 --mem 40G --gpu-type any --time 08:00:00 --skip-explanations --wandb-tag dc07-stage1` |
| 2 | `item_proto_sm` | `hm_1_month` | same with `-m item_proto_sm` |
| 3 | `user_proto_sm` | `ml-1m` | same with `-d ml-1m` |
| 4 | `item_proto_sm` | `ml-1m` | same with `-m item_proto_sm -d ml-1m` |

Results: `/scratch/c7031336/protomf_results/<model>_<dataset>_s38210573/` (dev budget → bare name).
Sigmoid twins (`user_proto_sg`, `item_proto_sg`) and the composed `_sm` arms are registered but
NOT queued in stage 1.

## 3. Temperature range (pinned by `dc_checks/dc07/t03`, stock init, N=3000)

`temperature = tune.loguniform(0.1, 2.0)`, fixed per trial (D2). Measured init-time regimes
(smallest τ with mean H(m) > 0.9·log K = uniform / popularity-channel regime; τ at which the
one-hot regime ends = H > 0.1·log K):

| shape | d | mean ‖q‖ | hard-regime exit τ | uniform-regime entry τ |
|---|---|---:|---:|---:|
| host (free Embedding) | 10 / 30 / 60 / 100 | 3.1 / 5.4 / 7.7 / 10.0 | 0.28–0.32 / 0.45–0.56 / 0.71–0.79 / 0.89–1.0 | 3.2–4.5 / 6.3–7.1 / 7.9–11.2 / 11.2–14.1 |
| fU (mean of ~20 words + ID) | 10 … 100 | 3.2 … 10.5 | 0.25 … 1.0 | 3.5 … 14.1 |
| fI (5 fields + ID) | 10 … 100 | 7.6 … 24.7 | 0.63 … 2.5 | 7.9 … 35.5 |

Reading: ‖q‖ ≈ √d at init; uniform at τ ≈ 1.1–1.5·‖q‖, one-hot below τ ≈ ‖q‖/10. τ_max = 2.0 <
3.16 (smallest uniform entry, host d=10 K=100); τ_min = 0.1 = one decade below the largest host
hard-exit (1.0, d=100). **Deviation from the memo:** the memo's τ_min = 0.05 sat in the saturated
regime for every shape and is raised to 0.1 (Matteo to confirm or override). For stage 2 the fI
shape (2.5× the norm) may want the range shifted up — recorded, not acted on. τ and ‖q‖ are
jointly redundant in training; the range fixes the init sharpness the search can start from.

## 4. Comparators (dev tier unless marked) — copied from requirements §6

| dataset | cosine host (frozen) | acf | mf | lightfm_tags / _ids | UI host |
|---|---|---|---|---|---|
| hm_1_month | user_proto 0.5725/0.3322; item_proto 0.4938/0.2631 | 0.5764/0.3307 | 0.3652/0.2233 | 0.4582/0.2709 ; 0.5087/0.3063 | 0.6587/0.4164 |
| ml-1m (Wave D dev) | user_proto 0.5668/0.3230; item_proto 0.5711/0.3187 | 0.6006/0.3364 (prod) | 0.5022/0.2793 | 0.5565/0.3197 ; 0.5612/0.3261 | 0.6246/0.3573 (prod) |

Noise yardstick (hm, one draw): |f0 − item_proto| = 0.0026 HR@10 / 0.0011 N@10; no ml-1m yardstick
exists — quote the hm value as scale only. Sanity anchor (D5): acf ml-1m 0.6006 is the
membership-family reference the softmax host should land near or above.

## 5. Read-outs per arm (pre-registered; login node)

- test HR@10 / NDCG@10 (`test_metrics.json`).
- `Master/scripts/sc8_geometry_readout.py` (item_proto_sm arms) / `sc8_fu_geometry_readout.py`
  (user_proto_sm arms): effective rank of the centred activation spectrum, prototype pairwise
  latent cosine > 0.999 (clone blocks), and — membership runs — section C = top-prototype score
  share + mean membership vector instead of B(t).
- `Master/scripts/dc07_membership_readout.py`: H(m) distribution, fraction in the uniform regime
  (H > 0.9 log K) and hard regime (H < 0.1 log K), Spearman(H, history length | popularity),
  Spearman(H, ‖q‖), top-prototype score share over top-10, winning τ with edge flag (10 % of the
  range in log-space).

## 6. Stage-1 gate (decision rule, verbatim from requirements §6)

- A. Accuracy within the yardstick of the cosine host AND geometry still collapsed → the layer
  is the bottleneck independent of the similarity. Record; stop dc07 as an experiment family;
  the result goes to the knobs chapter as a controlled negative.
- B. Accuracy below the host by > yardstick → the +1 channel was load-bearing; record how
  much; D6 decides on a bias arm; stop unless the bias arm recovers it.
- C. Accuracy ≥ host with visibly healthier geometry (effective rank up, clone blocks gone,
  membership entropy varying with evidence) → stage 2.

Void-comparison clause (§9): if the winner sits at τ_max with near-uniform memberships (uniform
fraction high, top-prototype share ≈ 1), the arm is a popularity model and the comparison is void
— report as such, do not grade A/B/C.

## 7. After the runs

Post-run memo `Master/temp/dc07_stage1_results_<date>.md` (→ `Master/docs/scrutiny/sc07_…` once
the cycle is registered) applying §6; protocol entries at build landed / stage-1 results / gate
decision; D6 bias arm decided by Matteo on a B verdict.
