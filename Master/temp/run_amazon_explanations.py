"""Run explanation generation for user_item_proto on amazon2014."""
import os, sys, argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

from ray.tune import ExperimentAnalysis
from rec_sys.rec_sys import RecSys
from feature_extraction.feature_extractor_factories import FeatureExtractorFactory
from utilities.explanations_utils import tsne_plot, get_top_k_items, weight_visualization

# --- Config ---
RESULT_DIR = os.path.expanduser('~/ray_results/user_item_proto_amazon2014_DE_38210573_2026-3-31_13-15-21.491510')
DATA_DIR = os.path.join(REPO_ROOT, 'data', 'amazon2014')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'Master', 'experiments', 'replication', 'explanations')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load best trial ---
print("Loading best trial...")
analysis = ExperimentAnalysis(RESULT_DIR)
metric = 'hit_ratio@10'
best_trial = analysis.get_best_trial(metric, 'max', scope='all')
best_config = best_trial.config
best_checkpoint = analysis.get_best_checkpoint(best_trial, metric, 'max')
checkpoint_path = os.path.join(best_checkpoint.to_directory(), 'best_model.pth')
config = argparse.Namespace(**best_config)
print(f"Best val HR@10: {best_trial.last_result['hit_ratio@10']:.4f}")

# --- Build model ---
print("Building model...")
n_users = pd.read_csv(os.path.join(DATA_DIR, 'user_ids.csv')).shape[0]
n_items = pd.read_csv(os.path.join(DATA_DIR, 'item_ids.csv')).shape[0]

user_fe, item_fe = FeatureExtractorFactory.create_models(
    config.ft_ext_param, n_users, n_items
)
model = RecSys(n_users, n_items, config.rec_sys_param, user_fe, item_fe,
               config.loss_func_name, config.loss_func_aggr)
state_dict = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()
print(f"n_users={n_users}, n_items={n_items}")

# --- Extract ---
user_proto_fe = model.user_feature_extractor.model_1
item_proto_fe = model.item_feature_extractor.model_1

user_prototypes = user_proto_fe.prototypes.detach().numpy()
user_embedding = user_proto_fe.embedding_ext.embedding_layer.weight.detach().numpy()
item_prototypes = item_proto_fe.prototypes.detach().numpy()
item_embedding = item_proto_fe.embedding_ext.embedding_layer.weight.detach().numpy()
print(f"User protos: {user_prototypes.shape}, Item protos: {item_prototypes.shape}")

items_info = pd.read_csv(os.path.join(DATA_DIR, 'item_ids.csv'))
print(f"Item info: {items_info.shape}")

# --- 1. TSNE ---
print("\n=== TSNE ===")
np.random.seed(42)

sample_idx = np.random.choice(n_items, min(2000, n_items), replace=False)
tsne_plot(item_embedding[sample_idx], item_prototypes,
          object_legend_text='Items', perplexity=30,
          path_save_fig=os.path.join(OUTPUT_DIR, 'tsne_item_prototypes_amazon.pdf'))
print("Saved tsne_item_prototypes_amazon.pdf")

user_sample_idx = np.random.choice(n_users, min(2000, n_users), replace=False)
tsne_plot(user_embedding[user_sample_idx], user_prototypes,
          object_legend_text='Users', perplexity=30,
          path_save_fig=os.path.join(OUTPUT_DIR, 'tsne_user_prototypes_amazon.pdf'))
print("Saved tsne_user_prototypes_amazon.pdf")

# --- 2. Top-K items per item prototype ---
print("\n=== Top-K items per prototype ===")
with torch.no_grad():
    all_item_ids = torch.arange(n_items)
    item_emb_t = item_proto_fe.embedding_ext(all_item_ids)
    proto_t = item_proto_fe.prototypes
    item_norm = torch.nn.functional.normalize(item_emb_t, dim=1)
    proto_norm = torch.nn.functional.normalize(proto_t, dim=1)
    sim_matrix = (1 + item_norm @ proto_norm.T).numpy()

n_item_protos = item_prototypes.shape[0]
print(f"Similarity matrix: {sim_matrix.shape}")

all_proto_results = []
for p_idx in range(n_item_protos):
    top_items = get_top_k_items(sim_matrix, items_info, proto_idx=p_idx, top_k=10)
    top_items['prototype'] = p_idx
    all_proto_results.append(top_items)
    print(f"\n--- Prototype {p_idx} ---")
    print(top_items.to_string())

all_proto_df = pd.concat(all_proto_results)
all_proto_df.to_csv(os.path.join(OUTPUT_DIR, 'top_k_items_per_prototype_amazon.csv'))
print(f"\nSaved top_k_items_per_prototype_amazon.csv")

# --- 3. Weight visualization ---
print("\n=== Weight visualization ===")
with torch.no_grad():
    sample_user_id = 42
    user_ids_t = torch.tensor([sample_user_id])
    item_ids_t = torch.arange(n_items)
    user_out = model.user_feature_extractor(user_ids_t)
    item_out = model.item_feature_extractor(item_ids_t)
    scores = (user_out @ item_out.T).squeeze()
    top_item_id = scores.argmax().item()
    print(f"User {sample_user_id} -> Top item: {top_item_id}")

    # User similarity to user prototypes
    user_emb = user_proto_fe.embedding_ext(user_ids_t)
    u_norm = torch.nn.functional.normalize(user_emb, dim=1)
    up_norm = torch.nn.functional.normalize(user_proto_fe.prototypes, dim=1)
    u_sim_mtx = (1 + u_norm @ up_norm.T).squeeze().numpy()

    # Item projection onto user prototype space
    item_proj_fe = model.item_feature_extractor.model_2
    i_proj = item_proj_fe(torch.tensor([top_item_id])).squeeze().numpy()

    # Item similarity to item prototypes
    item_emb = item_proto_fe.embedding_ext(torch.tensor([top_item_id]))
    i_norm = torch.nn.functional.normalize(item_emb, dim=1)
    ip_norm = torch.nn.functional.normalize(item_proto_fe.prototypes, dim=1)
    i_sim_mtx = (1 + i_norm @ ip_norm.T).squeeze().numpy()

    # User projection onto item prototype space
    user_proj_fe = model.user_feature_extractor.model_2
    u_proj = user_proj_fe(user_ids_t).squeeze().numpy()

print(f"u_sim_mtx: {u_sim_mtx.shape}, i_proj: {i_proj.shape}")
print(f"i_sim_mtx: {i_sim_mtx.shape}, u_proj: {u_proj.shape}")

weight_visualization(u_sim_mtx, u_proj, i_sim_mtx, i_proj, annotate_top_k=3)
plt.savefig(os.path.join(OUTPUT_DIR, 'weight_viz_user42_amazon.pdf'), format='pdf')
print("Saved weight_viz_user42_amazon.pdf")

print(f"\n=== All outputs in {OUTPUT_DIR} ===")
