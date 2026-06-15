"""Scan all baseline result configs and flag recurring exact hyperparameter sets.

Searchable hyperparameters only (the things hyperopt tunes): lr, wd, embedding_dim,
n_prototypes, sim weights, loss, batch, neg_train, optim. Excludes data_path,
n_epochs, patience, device, seed (budget/env, not searched).
"""
import glob
import json
import os

ROOT = "Master/experiments/results"


def walk_numbers(obj, prefix=""):
    """Flatten nested config into searchable leaf hyperparameters."""
    out = {}
    SKIP = {"data_path", "n_epochs", "_max_patience", "_min_delta", "device", "seed",
            "_num_workers", "_optimizing_metric", "val_batch_size"}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in SKIP:
                continue
            out.update(walk_numbers(v, f"{prefix}{k}."))
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        out[prefix.rstrip(".")] = obj
    return out


sigs = {}
rows = []
for cfg_path in sorted(glob.glob(f"{ROOT}/*/config.json")):
    combo = os.path.basename(os.path.dirname(cfg_path))
    with open(cfg_path) as f:
        c = json.load(f)
    flat = walk_numbers(c)
    # signature = the tuned hyperparameters that vary (drop ft_type strings etc.)
    sig_keys = sorted(k for k, v in flat.items()
                      if isinstance(v, (int, float)) and not isinstance(v, bool))
    sig = tuple((k, flat[k]) for k in sig_keys)
    sigs.setdefault(sig, []).append(combo)
    rows.append((combo, flat))

print("=== recurring exact hyperparameter signatures (across combos) ===")
dupe = False
for sig, combos in sigs.items():
    if len(combos) > 1:
        dupe = True
        print("  DUPLICATE:", combos)
if not dupe:
    print("  none — every baseline has a unique tuned-hyperparameter signature.")

print("\n=== key tuned hyperparameters per combo ===")
print(f"{'combo':45s} {'lr':>12s} {'wd':>12s} {'emb':>4s}")
for combo, flat in rows:
    lr = flat.get("optim_param.lr", "")
    wd = flat.get("optim_param.wd", "")
    emb = flat.get("ft_ext_param.embedding_dim", flat.get("ft_ext_param.latent_dimension", ""))
    lr = f"{lr:.6g}" if isinstance(lr, float) else str(lr)
    wd = f"{wd:.6g}" if isinstance(wd, float) else str(wd)
    print(f"{combo:45s} {lr:>12s} {wd:>12s} {str(emb):>4s}")
