"""Cold-start evaluation runner (Scrutiny S0.3 §§3–6) — evaluates ONE trained run on the cold
variant and writes the cold results block every model row of the comparison shares.

Given a run directory (run_combo layout: ``config.json`` + ``best_model.pth``, the model having
been RETRAINED on the cold variant per the S0.3 §6 single-config convention), this runner
produces, into ``<results_dir>/cold_eval/``:

- **warm test** on the variant (sanity: cold removal must not distort the warm regime),
- **cold-vs-cold** (PRIMARY, LightFM/B5-faithful: negatives from the cold pool; chance HR@K=K/100),
- **cold-vs-all** (secondary diagnostic: negatives from the full catalog — deliberately
  calibration-confounded),
- per-popularity-decile cold slices + per-item cluster-bootstrap 95% CIs,
- optionally the **attr-kNN fallback** re-evaluation for CF rows (S0.3 §5 / dc02 §3.6 #7: cold
  item's base embedding := mean of its top-n attribute-cosine warm neighbors' trained
  embeddings, patched at eval time),
- optionally the model-agnostic **tie-block diagnostic** (signature-class statistics of the
  full-catalog top-10 over sampled users — the honest place for dc02's signature-tie ceiling),
- a **popularity reference** row (variant-train popularity scores; cold items share popularity 0).

Guard rails: refuses ``use_bias != 0`` configs (a cold item's per-ID bias is an untrained zero —
a silent depressant; see ``dc01_feature_composed_bias_gap.md``) unless a cold-bias policy is
declared; tags+ids FeatureEmbedding configs (dc01 ``feature_item_proto`` / S0.4 ``lightfm``)
MUST receive the ID-column drop — the runner verifies the drop actually applied and aborts
otherwise (F-S0-07(b), verified at dc01 SC.3); every metric divides by counted rows with a
declared expectation (F-S0-03); negative draws are seeded and identical across model rows
(fresh dataset per evaluation, same seed, sequential iteration with num_workers=0).

Usage:
    python -m utilities.cold_eval --results-dir Master/experiments/results/mf_hm_1_month_cold_s38210573
    (defaults: variant data/hm_1_month_cold, canonical inferred by stripping '_cold')
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data

from feature_extraction.feature_extractor_factories import FeatureExtractorFactory
from feature_extraction.feature_extractors import (AnchorBasedCollaborativeFiltering,
                                                   ConcatenateFeatureExtractors, Embedding,
                                                   FeatureEmbedding, HistoryFeatureEmbedding,
                                                   PrototypeEmbedding)
from feature_extraction.feature_ids import build_attr_multi_hot, inject_feature_ids
from rec_sys.protomf_dataset import ColdTestDataset, ProtoRecDataset
from rec_sys.rec_sys import RecSys
from utilities.consts import DATA_PATH, K_VALUES, NEG_VAL, SINGLE_SEED
from utilities.eval import Evaluator
from utilities.utils import reproducible

# S0.5 canonical field set (charter C5) — used for kNN attribute space + signature classes.
CANONICAL_FIELDS = ['department_name', 'product_type_name', 'section_name',
                    'colour_group_name', 'graphical_appearance_name']


def load_config(results_dir: str) -> dict:
    with open(os.path.join(results_dir, 'config.json')) as f:
        conf = json.load(f)
    use_bias = conf.get('rec_sys_param', {}).get('use_bias', 1)
    if use_bias and float(use_bias) > 0:
        raise RuntimeError(
            f'cold eval REFUSED: config has use_bias={use_bias} but no declared cold-bias policy '
            f'(a cold item\'s per-ID bias is an untrained zero — a silent depressant; see '
            f'dc01_feature_composed_bias_gap.md and S0.3 §5). Fleet convention is use_bias=0.')
    # F-S0-07(b) resolved at dc01 SC.3: the ID-column drop covers dc01's nested branch (the one
    # shared FeatureEmbedding instance feeds both item halves — dc_checks/dc01/t11), so the former
    # load-time refusal of feature_item_proto+use_id_feature is lifted. Enforcement moved to the
    # runner: after drop_cold_id_rows, configs that expect the drop abort if it did not apply
    # (config_expects_id_drop below) — structure drift fails loudly, never silently mis-scores.
    return conf


def config_expects_id_drop(conf: dict) -> bool:
    """True when the config's item branch is a tags+ids FeatureEmbedding composition — dc01
    ``feature_item_proto`` or S0.4 ``lightfm`` with ``use_id_feature`` enabled (absent key =
    enabled: the factory defaults to True). Such a run MUST receive the ID-column drop at cold
    inference (F-S0-07/F-S0-13): scoring cold items on their untrained per-item ID rows is the
    silent-depressant failure class this runner exists to prevent."""
    ft_param = conf.get('ft_ext_param', {})
    if ft_param.get('ft_type') not in ('feature_item_proto', 'lightfm'):
        return False
    return bool(ft_param.get('item_ft_ext_param', {}).get('use_id_feature', True))


def build_model(conf: dict, variant_path: str, device: str = 'cpu'):
    """Rebuild the trained model against the variant dir (Tester._build_model without the loader —
    the variant's ID/feature files are byte-identical to canonical, so tensors match)."""
    n_users = len(pd.read_csv(os.path.join(variant_path, 'user_ids.csv')))
    n_items = len(pd.read_csv(os.path.join(variant_path, 'item_ids.csv')))
    ft_ext_param = inject_feature_ids(conf['ft_ext_param'], variant_path)
    u_fe, i_fe = FeatureExtractorFactory.create_models(ft_ext_param, n_users, n_items)
    rec = RecSys(n_users, n_items, conf['rec_sys_param'], u_fe, i_fe,
                 conf['loss_func_name'], conf.get('loss_func_aggr', 'mean'))
    rec.init_parameters()
    return rec, n_users, n_items


def load_checkpoint(model: nn.Module, results_dir: str, device: str = 'cpu'):
    ckpt = os.path.join(results_dir, 'best_model.pth')
    try:
        params = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 1.13
        params = torch.load(ckpt, map_location=device)
    model.load_state_dict(params)
    model.to(device).eval()
    return model


@torch.no_grad()
def evaluate_rows(scorer, dataset, batch_size: int = 256, device: str = 'cpu'):
    """Run a scorer (the model, or any callable (u_idxs, i_idxs) -> logits) over a row dataset.
    Returns (aggregate metrics, per-row arrays). Divisor = counted rows, asserted (F-S0-03)."""
    loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=0)
    ev = Evaluator(len(dataset))
    for u_idxs, i_idxs, _labels in loader:
        u_idxs, i_idxs = u_idxs.to(device), i_idxs.to(device)
        out = torch.sigmoid(scorer(u_idxs, i_idxs)).cpu().numpy()
        ev.eval_batch(out, sum=False)
    per_row = {k: np.asarray(v, dtype=float) for k, v in ev.get_results(aggregated=False).items()}
    agg = {k: float(v.mean()) for k, v in per_row.items()}
    return agg, per_row


def _find_item_embedding_weight(fe) -> torch.Tensor:
    """Locate the per-item ID embedding table of a CF item branch (the attr-kNN patch target).
    For the double-tie Concatenate wrapper, model_1's base table IS model_2's (tied Parameter),
    so one patch covers both halves. Raises for natively cold-capable branches (no ID table)."""
    if isinstance(fe, ConcatenateFeatureExtractors):
        return _find_item_embedding_weight(fe.model_1)
    if isinstance(fe, PrototypeEmbedding):
        return _find_item_embedding_weight(fe.embedding_ext)
    if isinstance(fe, AnchorBasedCollaborativeFiltering):
        return fe.embedding_ext.embedding_layer.weight
    if isinstance(fe, Embedding):  # covers EmbeddingW
        return fe.embedding_layer.weight
    raise ValueError(f'no per-item ID embedding table found on {type(fe).__name__} — '
                     f'attr-kNN fallback applies to CF rows only')


def _find_feature_embedding(fe):
    """Locate a FeatureEmbedding on the item branch (the ID-column-drop target), or None.
    Handles the direct case (lightfm) and dc01's nested case (fI host since the 2026-07-12
    re-host: PrototypeEmbedding.embedding_ext — the single score surface, so one drop covers
    the whole fI score; verified at dc01 SC.3 / SC.3 re-run, dc_checks/dc01/t11). The
    Concatenate recursion stays for double-tie-shaped branches (the lineage's fUfI merge
    stage), where a shared instance means one drop covers both halves."""
    if isinstance(fe, ConcatenateFeatureExtractors):
        return _find_feature_embedding(fe.model_1) or _find_feature_embedding(fe.model_2)
    if isinstance(fe, PrototypeEmbedding):
        return _find_feature_embedding(fe.embedding_ext)
    if isinstance(fe, FeatureEmbedding):
        return fe
    return None


@torch.no_grad()
def drop_cold_id_rows(model: RecSys, variant_path: str):
    """ID-column-drop convention (S0.3 §5 / S0.4 §2, F-S0-13): zero the per-item ID embedding
    rows of COLD items in a tags+ids FeatureEmbedding item branch, so cold items are scored
    purely from their feature composition (q_cold = Σ_f e_f). Warm rows untouched. Returns an
    info dict, or None when there is nothing to drop (no FeatureEmbedding on the item branch,
    or use_id_feature=False)."""
    fe = _find_feature_embedding(model.item_feature_extractor)
    if fe is None or not fe.use_id_feature:
        return None
    cold = np.sort(pd.read_csv(os.path.join(variant_path, 'cold_items.csv'))['item_id'].to_numpy())
    idx = torch.as_tensor(cold) + fe.n_features
    fe.embedding_layer.weight.data[idx] = 0.
    return {'n_cold_id_rows_zeroed': int(len(cold)),
            'item_branch': type(model.item_feature_extractor).__name__}


def config_expects_user_id_drop(conf: dict) -> bool:
    """dc05 mirror of ``config_expects_id_drop``: True when the config's USER branch is a
    history+ID HistoryFeatureEmbedding composition — ``feature_user_proto`` with
    ``use_id_feature`` enabled (absent key = enabled: the factory defaults to True). Such a run
    must receive the user-side ID-row drop at cold-USER inference (design doc §3.4/§3.7-6,
    F-S0-07 mirror): scoring cold users on their untrained per-user ID rows is the same
    silent-depressant class, one side over."""
    ft_param = conf.get('ft_ext_param', {})
    if ft_param.get('ft_type') != 'feature_user_proto':
        return False
    return bool(ft_param.get('user_ft_ext_param', {}).get('use_id_feature', True))


def _find_history_feature_embedding(fe):
    """Locate a HistoryFeatureEmbedding on the user branch (the user-side ID-drop target), or
    None. Handles dc05's nested case (fU host: PrototypeEmbedding.embedding_ext — the single
    score surface, so one drop covers the whole fU score); the Concatenate recursion is for
    double-tie-shaped branches (the lineage's fUfI merge stage)."""
    if isinstance(fe, ConcatenateFeatureExtractors):
        return _find_history_feature_embedding(fe.model_1) or _find_history_feature_embedding(fe.model_2)
    if isinstance(fe, PrototypeEmbedding):
        return _find_history_feature_embedding(fe.embedding_ext)
    if isinstance(fe, HistoryFeatureEmbedding):
        return fe
    return None


@torch.no_grad()
def drop_cold_user_id_rows(model: RecSys, variant_path: str):
    """dc05 user-side ID-drop convention (design doc §3.4/§3.7-6; F-S0-07/F-S0-13 mirrored to
    users): zero the per-user ID embedding rows of COLD users in a history+ID
    HistoryFeatureEmbedding user branch, so cold users are scored purely from their basket
    composition (q_cold = Σ_f w̄_f·e_f; an empty variant basket then degrades to the verified
    S = B(t) item-baseline path, 4b C3). Warm rows untouched. Returns an info dict, or None when
    there is nothing to drop: no HistoryFeatureEmbedding on the user branch, use_id_feature=False,
    or no ``cold_users.csv`` in the variant dir — the cold-USER testbed is scrutiny-stage work
    (M6.2); this seam is pinned now (dc_checks/dc05/t07) so the future testbed plugs in."""
    fe = _find_history_feature_embedding(model.user_feature_extractor)
    if fe is None or not fe.use_id_feature:
        return None
    cold_users_csv = os.path.join(variant_path, 'cold_users.csv')
    if not os.path.exists(cold_users_csv):
        return None
    cold = np.sort(pd.read_csv(cold_users_csv)['user_id'].to_numpy())
    idx = torch.as_tensor(cold) + fe.n_features
    fe.embedding_layer.weight.data[idx] = 0.
    return {'n_cold_user_id_rows_zeroed': int(len(cold)),
            'user_branch': type(model.user_feature_extractor).__name__}


@torch.no_grad()
def attr_knn_patch(model: RecSys, variant_path: str, n_neighbors: int = 20,
                   fields=None) -> dict:
    """S0.3 §5 attr-kNN fallback: overwrite each cold item's base ID embedding with the mean of
    its top-n attribute-cosine WARM neighbors' trained embeddings. Mutates the model in place."""
    fields = fields or CANONICAL_FIELDS
    X, _, _ = build_attr_multi_hot(variant_path, fields)          # rows all have norm sqrt(F)
    cold = np.sort(pd.read_csv(os.path.join(variant_path, 'cold_items.csv'))['item_id'].to_numpy())
    train = pd.read_csv(os.path.join(variant_path, 'listening_history_train.csv'),
                        usecols=['item_id'])
    warm = np.sort(train['item_id'].unique())                     # trained on the variant

    # equal row norms -> cosine order == dot-product order
    sims = X[cold] @ X[warm].T                                    # (n_cold, n_warm)
    top = torch.topk(sims, k=min(n_neighbors, len(warm)), dim=1).indices  # (n_cold, n)
    neighbor_ids = torch.as_tensor(warm)[top]

    weight = _find_item_embedding_weight(model.item_feature_extractor)
    cold_t = torch.as_tensor(cold)
    weight.data[cold_t] = weight.data[neighbor_ids].mean(dim=1)
    return {'n_cold_patched': int(len(cold)), 'n_neighbors': int(top.shape[1]),
            'n_warm_pool': int(len(warm))}


def _signature_classes(variant_path: str, fields=None) -> np.ndarray:
    fields = fields or CANONICAL_FIELDS
    df = pd.read_csv(os.path.join(variant_path, 'item_features.csv')).sort_values('item_id')
    return df.groupby(fields, sort=True).ngroup().to_numpy()


@torch.no_grad()
def tie_block_diagnostic(model: RecSys, variant_path: str, n_users_sample: int = 5000,
                         k: int = 10, seed: int = SINGLE_SEED, device: str = 'cpu',
                         user_batch: int = 512) -> dict:
    """dc02 §3.6 full-catalog tie-block diagnostic, model-agnostic (S0.3 §5): signature-class
    statistics of the full-catalog top-k for a seeded sample of users with warm history."""
    sig = torch.as_tensor(_signature_classes(variant_path))
    train = pd.read_csv(os.path.join(variant_path, 'listening_history_train.csv'),
                        usecols=['user_id'])
    users_with_history = np.sort(train['user_id'].unique())
    rng = np.random.default_rng(seed)
    n_sample = min(n_users_sample, len(users_with_history))
    users = rng.choice(users_with_history, n_sample, replace=False)

    n_items = sig.shape[0]
    item_repr = model.item_feature_extractor(torch.arange(n_items, device=device))  # (M, D)
    distinct, has_tie, max_class = [], [], []
    for s in range(0, n_sample, user_batch):
        u = torch.as_tensor(users[s:s + user_batch], device=device)
        u_repr = model.user_feature_extractor(u)                                    # (B, D)
        logits = u_repr @ item_repr.T                                               # (B, M)
        top_items = torch.topk(logits, k, dim=1).indices                            # (B, k)
        top_sigs = sig[top_items.cpu()]                                             # (B, k)
        for row in top_sigs:
            _, counts = torch.unique(row, return_counts=True)
            distinct.append(len(counts))
            has_tie.append(bool((counts > 1).any()))
            max_class.append(int(counts.max()))
    return {'n_users_sampled': int(n_sample), 'k': k,
            'mean_distinct_signatures_in_topk': float(np.mean(distinct)),
            'share_users_with_signature_twins_in_topk': float(np.mean(has_tie)),
            'mean_max_signature_class_in_topk': float(np.mean(max_class))}


class PopularityScorer:
    """The popularity reference row: score(u, i) = DENSE variant-train popularity rank of i,
    scaled to [0, 1), plus small seeded uniform tie-break noise (F-S0-08). Rank + noise rather
    than raw counts because (a) equal-popularity ties — ALL cold items on cold-vs-cold — would
    otherwise be resolved deterministically by argpartition internals (the row read 0.00 instead
    of the K/100 chance floor), and (b) raw counts saturate the eval sigmoid at 1.0 for
    pop >= ~17, collapsing warm order. Noise amplitude is half a rank gap, so distinct
    popularity levels are never reordered; cold items (pop 0, the bottom rank) tie-break
    uniformly at random, reading chance on cold-vs-cold and bottom on cold-vs-all."""

    def __init__(self, variant_path: str, n_items: int, seed: int = SINGLE_SEED):
        train = pd.read_csv(os.path.join(variant_path, 'listening_history_train.csv'),
                            usecols=['item_id'])
        pop = np.zeros(n_items, dtype=np.int64)
        counts = train.groupby('item_id').size()
        pop[counts.index.to_numpy()] = counts.to_numpy()
        _, dense_rank = np.unique(pop, return_inverse=True)  # 0..R-1, equal pops share a rank
        n_ranks = int(dense_rank.max()) + 1
        rng = np.random.default_rng(seed)
        score = (dense_rank + rng.uniform(0., 0.5, n_items)) / n_ranks
        self.pop = torch.as_tensor(score, dtype=torch.float32)

    def __call__(self, u_idxs, i_idxs):
        return self.pop[i_idxs]


def per_item_bootstrap_ci(per_row_metric: np.ndarray, row_items: np.ndarray,
                          n_boot: int = 1000, seed: int = SINGLE_SEED, alpha: float = 0.05):
    """Per-item CLUSTER bootstrap (S0.3 §6): resample cold ITEMS with replacement, pool their
    rows. Returns (lo, hi) of the (1-alpha) CI for the row-mean metric."""
    df = pd.DataFrame({'item': row_items, 'v': per_row_metric})
    g = df.groupby('item')['v'].agg(['sum', 'count'])
    sums, counts = g['sum'].to_numpy(), g['count'].to_numpy()
    rng = np.random.default_rng(seed)
    n = len(sums)
    idx = rng.integers(0, n, size=(n_boot, n))
    estimates = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.quantile(estimates, alpha / 2)), float(np.quantile(estimates, 1 - alpha / 2))


def decile_slices(per_row: dict, row_items: np.ndarray, variant_path: str,
                  metrics=('hit_ratio@10', 'ndcg@10')) -> list:
    """Cold metrics by the positive item's ORIGINAL train-popularity decile (from cold_items.csv)."""
    cold = pd.read_csv(os.path.join(variant_path, 'cold_items.csv'))
    dec = dict(zip(cold['item_id'], cold['pop_decile']))
    row_dec = np.array([dec[i] for i in row_items])
    out = []
    for d in sorted(set(row_dec)):
        m = row_dec == d
        entry = {'pop_decile': int(d), 'n_rows': int(m.sum())}
        entry.update({k: float(per_row[k][m].mean()) for k in metrics})
        out.append(entry)
    return out


def run_cold_eval(results_dir: str, variant_path: str, canonical_path: str = None,
                  device: str = 'cpu', batch_size: int = 256, n_neg: int = NEG_VAL,
                  seed: int = SINGLE_SEED, knn: bool = True, tie_diag: bool = True,
                  popularity: bool = True, knn_neighbors: int = 20,
                  canonical_results_dir: str = None) -> dict:
    conf = load_config(results_dir)
    model, n_users, n_items = build_model(conf, variant_path, device)
    load_checkpoint(model, results_dir, device)

    report = {'results_dir': os.path.abspath(results_dir),
              'variant_path': os.path.abspath(variant_path),
              'n_neg': n_neg, 'seed': seed,
              'use_bias': conf.get('rec_sys_param', {}).get('use_bias', None),
              'ft_type': conf.get('ft_ext_param', {}).get('ft_type', None)}

    # ID-column-drop convention (S0.4 §2 / F-S0-13): cold items in a tags+ids FeatureEmbedding
    # branch are scored feature-only — their untrained per-item ID rows are zeroed.
    id_drop = drop_cold_id_rows(model, variant_path)
    if id_drop:
        report['id_column_drop'] = id_drop
        print(f"ID-column drop applied: {id_drop}")
    if config_expects_id_drop(conf) and not id_drop:
        raise RuntimeError(
            'cold eval ABORTED: config expects the ID-column drop (tags+ids FeatureEmbedding '
            'item branch, F-S0-07/F-S0-13) but drop_cold_id_rows found no FeatureEmbedding on '
            'the built model — model structure has drifted from what the drop covers; refusing '
            'to score cold items on untrained per-item ID rows.')

    # dc05 user-side mirror: applies only when the variant ships a cold_users.csv (no cold-USER
    # variant generator exists yet — M6.2 scrutiny-stage work; the convention is pinned now).
    user_id_drop = drop_cold_user_id_rows(model, variant_path)
    if user_id_drop:
        report['user_id_column_drop'] = user_id_drop
        print(f"user-side ID-column drop applied: {user_id_drop}")
    if config_expects_user_id_drop(conf) \
            and os.path.exists(os.path.join(variant_path, 'cold_users.csv')) and not user_id_drop:
        raise RuntimeError(
            'cold eval ABORTED: config expects the user-side ID-row drop (history+ids '
            'HistoryFeatureEmbedding user branch, dc05 §3.4) and the variant ships '
            'cold_users.csv, but drop_cold_user_id_rows found no HistoryFeatureEmbedding on the '
            'built model — structure drift; refusing to score cold users on untrained ID rows.')

    # 1. warm test on the variant (standard machinery; global-RNG negatives, seeded)
    reproducible(seed)
    warm_ds = ProtoRecDataset(variant_path, 'test', n_neg, 'uniform')
    warm_agg, _ = evaluate_rows(model, warm_ds, batch_size, device)
    report['warm_test'] = {'n_rows': len(warm_ds.coo_matrix.row), **warm_agg}

    # 1b. canonical warm reference (F-S0-11): the S0.3 §3 sanity comparison — cold removal must
    # not distort the warm regime.
    if canonical_results_dir:
        with open(os.path.join(canonical_results_dir, 'test_metrics.json')) as f:
            canon = json.load(f)
        report['canonical_warm_reference'] = {
            'results_dir': os.path.abspath(canonical_results_dir),
            'hit_ratio@10': canon.get('hit_ratio@10'), 'ndcg@10': canon.get('ndcg@10'),
            'warm_delta_hit_ratio@10': report['warm_test']['hit_ratio@10'] - canon['hit_ratio@10'],
            'warm_delta_ndcg@10': report['warm_test']['ndcg@10'] - canon['ndcg@10']}

    # 2+3. cold rankings (fresh seeded dataset per ranking -> identical draws across model rows)
    for ranking, key in [('cold', 'cold_vs_cold'), ('all', 'cold_vs_all')]:
        ds = ColdTestDataset(variant_path, ranking=ranking, n_neg=n_neg,
                             canonical_path=canonical_path, seed=seed)
        agg, per_row = evaluate_rows(model, ds, batch_size, device)
        block = {'n_rows': len(ds), **agg,
                 'chance_hit_ratio': ({f'@{k}': k / (1 + n_neg) for k in K_VALUES}
                                      if ranking == 'cold' else None),
                 'by_pop_decile': decile_slices(per_row, ds.items, variant_path)}
        for m in ('hit_ratio@10', 'ndcg@10'):
            block[f'{m}_ci95'] = per_item_bootstrap_ci(per_row[m], ds.items, seed=seed)
        report[key] = block

    # 4. attr-kNN fallback (CF rows): patch, then re-run both cold rankings
    if knn:
        try:
            patch_info = attr_knn_patch(model, variant_path, n_neighbors=knn_neighbors)
            report['attr_knn'] = dict(patch_info)
            for ranking, key in [('cold', 'cold_vs_cold'), ('all', 'cold_vs_all')]:
                ds = ColdTestDataset(variant_path, ranking=ranking, n_neg=n_neg,
                                     canonical_path=canonical_path, seed=seed)
                agg, _ = evaluate_rows(model, ds, batch_size, device)
                report['attr_knn'][key] = agg
            load_checkpoint(model, results_dir, device)   # restore unpatched weights
            if id_drop:  # defensive: re-apply the ID drop after the restore (mutually
                drop_cold_id_rows(model, variant_path)  # exclusive with kNN in practice)
            if user_id_drop:  # dc05 mirror: same defensive re-apply for the user side
                drop_cold_user_id_rows(model, variant_path)
        except ValueError as e:
            report['attr_knn'] = {'skipped': str(e)}

    # 5. tie-block diagnostic (model-agnostic, unpatched weights). Wrapped like attr_knn
    # (F-DC05-16): the diagnostic's signature-class machinery is H&M-field-specific, and an
    # unwrapped raise here would kill the run AFTER the primary metrics but BEFORE the report
    # is written. Full non-H&M support is deferred to the ml-1m_cold enablement work.
    if tie_diag:
        try:
            report['tie_block_diagnostic'] = tie_block_diagnostic(
                model, variant_path, seed=seed, device=device)
        except (ValueError, KeyError) as e:
            report['tie_block_diagnostic'] = {'skipped': str(e)}
            print(f"tie-block diagnostic skipped (F-DC05-16): {e!r}")

    # 6. popularity reference row
    if popularity:
        pop_scorer = PopularityScorer(variant_path, n_items, seed=seed)
        report['popularity_reference'] = {}
        for ranking, key in [('cold', 'cold_vs_cold'), ('all', 'cold_vs_all')]:
            ds = ColdTestDataset(variant_path, ranking=ranking, n_neg=n_neg,
                                 canonical_path=canonical_path, seed=seed)
            agg, _ = evaluate_rows(pop_scorer, ds, batch_size, 'cpu')
            report['popularity_reference'][key] = agg

    out_dir = os.path.join(results_dir, 'cold_eval')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    _write_markdown(report, os.path.join(out_dir, 'report.md'))
    print(f'cold_eval report written to {out_dir}')
    return report


def _write_markdown(report: dict, path: str):
    lines = [f"# Cold eval — {os.path.basename(report['results_dir'])}", '',
             f"variant: `{report['variant_path']}` | n_neg={report['n_neg']} | "
             f"seed={report['seed']} | use_bias={report['use_bias']}", '',
             '| block | rows | HR@10 | NDCG@10 | HR@10 CI95 |', '|---|---:|---:|---:|---|']
    canon = report.get('canonical_warm_reference')
    if canon:
        lines.append(f"| canonical warm reference | | {canon['hit_ratio@10']:.4f} | "
                     f"{canon['ndcg@10']:.4f} | warm Δhr@10 "
                     f"{canon['warm_delta_hit_ratio@10']:+.4f} |")
    for key, label in [('warm_test', 'warm test (variant)'),
                       ('cold_vs_cold', 'cold-vs-cold (PRIMARY)'),
                       ('cold_vs_all', 'cold-vs-all (secondary)')]:
        b = report.get(key)
        if not b:
            continue
        ci = b.get('hit_ratio@10_ci95')
        ci_s = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ''
        lines.append(f"| {label} | {b['n_rows']} | {b['hit_ratio@10']:.4f} | "
                     f"{b['ndcg@10']:.4f} | {ci_s} |")
    if isinstance(report.get('attr_knn'), dict) and 'cold_vs_cold' in report.get('attr_knn', {}):
        for key in ('cold_vs_cold', 'cold_vs_all'):
            b = report['attr_knn'][key]
            lines.append(f"| attr-kNN {key} | | {b['hit_ratio@10']:.4f} | {b['ndcg@10']:.4f} | |")
    if report.get('popularity_reference'):
        for key, b in report['popularity_reference'].items():
            lines.append(f"| popularity {key} | | {b['hit_ratio@10']:.4f} | {b['ndcg@10']:.4f} | |")
    td = report.get('tie_block_diagnostic')
    if td:
        lines += ['', f"tie-block (top-{td['k']}, {td['n_users_sampled']} users): "
                      f"mean distinct sigs {td['mean_distinct_signatures_in_topk']:.2f}, "
                      f"twin share {td['share_users_with_signature_twins_in_topk']:.3f}, "
                      f"mean max class {td['mean_max_signature_class_in_topk']:.2f}"]
    lines += ['', '## cold-vs-cold by original-popularity decile', '',
              '| decile | rows | HR@10 | NDCG@10 |', '|---:|---:|---:|---:|']
    for e in report.get('cold_vs_cold', {}).get('by_pop_decile', []):
        lines.append(f"| {e['pop_decile']} | {e['n_rows']} | {e['hit_ratio@10']:.4f} | "
                     f"{e['ndcg@10']:.4f} |")
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser(description='S0.3 cold-start evaluation runner')
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--variant-path', default=os.path.join(DATA_PATH, 'hm_1_month_cold'))
    ap.add_argument('--canonical-path', default=None)
    ap.add_argument('--canonical-results-dir', default=None,
                    help='completed CANONICAL run dir of the same model: adds the warm-regime '
                         'sanity comparison (F-S0-11) to the report')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--n-neg', type=int, default=NEG_VAL)
    ap.add_argument('--seed', type=int, default=SINGLE_SEED)
    ap.add_argument('--knn-neighbors', type=int, default=20)
    ap.add_argument('--skip-knn', action='store_true')
    ap.add_argument('--skip-tie-diagnostic', action='store_true')
    ap.add_argument('--skip-popularity', action='store_true')
    args = ap.parse_args()
    run_cold_eval(args.results_dir, args.variant_path, args.canonical_path, args.device,
                  args.batch_size, args.n_neg, args.seed, knn=not args.skip_knn,
                  tie_diag=not args.skip_tie_diagnostic, popularity=not args.skip_popularity,
                  knn_neighbors=args.knn_neighbors,
                  canonical_results_dir=args.canonical_results_dir)


if __name__ == '__main__':
    main()
