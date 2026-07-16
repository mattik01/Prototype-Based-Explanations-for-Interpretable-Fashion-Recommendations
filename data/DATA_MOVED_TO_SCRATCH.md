# Local datasets deleted — everything lives on LEO5 scratch (2026-07-16)

User decision 2026-07-16: no datasets are kept on the laptop — all runs
happen on the cluster. What was here and where it went:

| Was here | Now lives at |
|---|---|
| All split datasets (hm_1_month, hm_1_month_cold, hm_3_month, hm_full, ml-1m, ml-1m_cold, amazon2014) | `leo5:/scratch/c7031336/protomf_data/<dataset>/` (byte/checksum-verified before local deletion) |
| Raw H&M Kaggle files + 105,100 images | `leo5:/scratch/c7031336/hm_raw_kaggle/` (fresh 30.8 GB zip + extracted; also re-downloadable from Kaggle) |
| ml-1m raw sources (`movies.dat`, `ratings.dat`, tag-genome) | in the local backup tarball below; also publicly re-downloadable (GroupLens) |
| `lfm2b-interactions.zip` (20.8 GB, publicly withdrawn dataset) | **deleted entirely** (user decision 2026-07-16, mid-transfer: not worth the 7h VPN copy) — supervisor holds a copy, likely also the legacy GPU desktop; never used in this project (`lfm2b-1mon` unavailable per CLAUDE.md) |

**Local backup (kept deliberately):** `frozen_splits_backup_2026-07-16.tar.gz`
(191 MB, 46 files) in this directory — the frozen, byte-irreproducible
split artifacts (hm splitter uses a non-stable sort; see F-S0-05) plus the
ml-1m raw sources. This exists because **scratch is not backed up**; the
tarball is the only off-cluster copy of the artifacts every thesis number
depends on. Do not delete it without creating an equivalent backup.

The splitter/build scripts in the per-dataset folders are git-tracked and
remain. To restore any dataset locally: `scp -r
leo5:/scratch/c7031336/protomf_data/<dataset> data/` (or untar the backup).
