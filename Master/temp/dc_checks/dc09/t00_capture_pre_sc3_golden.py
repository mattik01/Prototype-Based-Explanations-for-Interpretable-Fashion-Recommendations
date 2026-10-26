# dc09 SC.3 — capture the PRE-change golden (run ONCE, on the unmodified code at 27dc914).
#
# Why a new golden: dc01's t03_golden.pt was captured under torch 1.9.1 (2026-06-17) and no
# longer matches bitwise under this machine's torch 2.5.1 (init drift ~1e-6), so it cannot
# guard the SC.3 edits here. This file records every surface the dc09 build touches
# (PrototypeEmbedding, FeatureEmbedding fixed/bags, HistoryFeatureEmbedding padded/bag with
# leave-one-out, and the fI / fU / UI factory branches through one training step) so t03 can
# demand bit-identity of the unchanged single-table path after the change.
# Valid for the environment it was captured in (recorded in meta); refuses to overwrite.
import sys

import torch

from _fixture import GOLDEN, all_recipes

if __name__ == '__main__':
    import os
    if os.path.exists(GOLDEN) and '--force' not in sys.argv:
        print(f'{GOLDEN} exists — refusing to overwrite (it must come from the unmodified code).')
        sys.exit(1)
    golden = all_recipes()
    golden['meta'] = {'torch': torch.__version__, 'captured_at_commit': '27dc914'}
    torch.save(golden, GOLDEN)
    n = sum(1 for _ in golden)
    print(f'captured {n - 1} recipes under torch {torch.__version__} -> {GOLDEN}')
