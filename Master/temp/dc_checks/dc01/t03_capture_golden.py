# dc01 P3 golden capture — RUN ON UNMODIFIED CODE, BEFORE editing PrototypeEmbedding.
# Captures the bit-exact state_dict + forward output of the stock PrototypeEmbedding
# so that t03_prototype_injection.py can prove the new optional `embedding_ext=None`
# path reproduces the original behaviour exactly (RNG draw order preserved).
#
# Run once (current code): python Master/temp/dc_checks/dc01/t03_capture_golden.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import torch

from utilities.utils import reproducible
from feature_extraction.feature_extractors import PrototypeEmbedding

SEED = 42
N, D, K = 20, 8, 4
INPUT = torch.tensor([[0, 3, 5], [2, 1, 4]], dtype=torch.long)

reproducible(SEED)
m = PrototypeEmbedding(N, D, K, use_weight_matrix=False, cosine_type='shifted',
                       reg_proto_type='max', reg_batch_type='max')
m.init_parameters()
with torch.no_grad():
    out = m(INPUT)

golden = {
    'state_dict': {k: v.clone() for k, v in m.state_dict().items()},
    'output': out.clone(),
    'meta': dict(seed=SEED, n=N, d=D, k=K, input=INPUT.clone()),
}
path = os.path.join(os.path.dirname(__file__), 't03_golden.pt')
torch.save(golden, path)
print('Saved golden ->', path)
print('state_dict keys:', list(golden['state_dict'].keys()))
print('output shape  :', tuple(out.shape))
print('output[0,0,:3]:', out[0, 0, :3].tolist())
