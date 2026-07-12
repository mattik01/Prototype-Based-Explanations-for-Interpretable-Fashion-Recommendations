# SUPERSEDED (dc01 SC.5, 2026-07-12): one-off UI-era demo — kept as historical record
# (LLM-usage preservation rule). The live, shared replacement is
# utilities/explanations/breakdown.py (+ the auto-invoked pipeline explainers
# 'breakdown' and 'proto_cards'). This script still assumes the UI-hosted shape
# and is NOT maintained against the fI host.
"""
Generate explanations for the best user_item_proto model on hm_3_month.
Outputs: TSNE plots, top-k items per prototype, weight visualization for a sample recommendation.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, REPO_ROOT)

from ray.tune import ExperimentAnalysis
from rec_sys.rec_sys import RecSys
from feature_extraction.feature_extractor_factories import FeatureExtractorFactory
from utilities.explanations_utils import tsne_plot, get_top_k_items, weight_visualization

# --- Config ---
RESULT_DIR = os.path.expanduser('~/ray_results/user_item_proto_hm_3_month_DE_38210573_2026-3-31_23-44-47.921663')
DATA_DIR = os.path.join(REPO_ROOT, 'data', 'hm_3_month')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'Master', 'experiments', 'hm_baseline', 'explanations')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load best trial ---
print("Loading best trial from Ray Tune...")
analysis = ExperimentAnalysis(RESULT_DIR)
metric = 'hit_ratio@10'
best_trial = analysis.get_best_trial(metric, 'max', scope='all')
best_config = best_trial.config
best_checkpoint = analysis.get_best_checkpoint(best_trial, metric, 'max')
checkpoint_path = os.path.join(best_checkpoint.to_directory(), 'best_model.pth')

config = argparse.Namespace(**best_config)
print(f"Best val HR@10: {best_trial.last_result['hit_ratio@10']:.4f}")

# --- Build model and load weights ---
print("Building model...")
n_users = pd.read_csv(os.path.join(DATA_DIR, 'user_ids.csv')).shape[0]
n_items = pd.read_csv(os.path.join(DATA_DIR, 'item_ids.csv')).shape[0]

user_fe, item_fe = FeatureExtractorFactory.create_feature_extractors(
    config.ft_ext_param, n_users, n_items
)
model = RecSys(n_users, n_items, user_fe, item_fe, config.loss_func_name,
               use_bias=config.rec_sys_param.get('use_bias', 0))

state_dict = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()
device = 'cpu'  # explanations on CPU is fine

print(f"Model loaded. n_users={n_users}, n_items={n_items}")

# --- Extract embeddings and prototypes ---
# For user_item_proto (ConcatenateFeatureExtractors), the structure is:
# user_feature_extractor = ConcatenateFeatureExtractors(PrototypeEmbedding, EmbeddingW)
# item_feature_extractor = ConcatenateFeatureExtractors(PrototypeEmbedding, EmbeddingW)

# User side
user_proto_fe = model.user_feature_extractor.fe_1  # PrototypeEmbedding
user_prototypes = user_proto_fe.prototypes.detach().numpy()  # (n_user_protos, emb_dim)
user_embedding = user_proto_fe.embedding.weight.detach().numpy()  # (n_users, emb_dim)

# Item side
item_proto_fe = model.item_feature_extractor.fe_1  # PrototypeEmbedding
item_prototypes = item_proto_fe.prototypes.detach().numpy()  # (n_item_protos, emb_dim)
item_embedding = item_proto_fe.embedding.weight.detach().numpy()  # (n_items, emb_dim)

print(f"User prototypes: {user_prototypes.shape}, Item prototypes: {item_prototypes.shape}")
print(f"User embeddings: {user_embedding.shape}, Item embeddings: {item_embedding.shape}")

# --- Load item info ---
items_info = pd.read_csv(os.path.join(DATA_DIR, 'item_features.csv'))
print(f"Item features loaded: {items_info.shape}")

# --- 1. TSNE Plots ---
print("\n=== Generating TSNE plots ===")

# Item TSNE (sample 2000 items for speed)
np.random.seed(42)
sample_idx = np.random.choice(n_items, min(2000, n_items), replace=False)
item_emb_sample = item_embedding[sample_idx]

tsne_plot(
    item_emb_sample, item_prototypes,
    object_legend_text='Items',
    perplexity=30,
    path_save_fig=os.path.join(OUTPUT_DIR, 'tsne_item_prototypes.pdf')
)
print(f"Saved: tsne_item_prototypes.pdf")

# User TSNE (sample 2000 users for speed)
user_sample_idx = np.random.choice(n_users, min(2000, n_users), replace=False)
user_emb_sample = user_embedding[user_sample_idx]

tsne_plot(
    user_emb_sample, user_prototypes,
    object_legend_text='Users',
    perplexity=30,
    path_save_fig=os.path.join(OUTPUT_DIR, 'tsne_user_prototypes.pdf')
)
print(f"Saved: tsne_user_prototypes.pdf")

# --- 2. Top-K Items per Item Prototype ---
print("\n=== Top-K items per item prototype ===")

# Compute item-to-item-prototype similarity matrix
with torch.no_grad():
    all_item_ids = torch.arange(n_items)
    item_emb_tensor = item_proto_fe.embedding(all_item_ids)  # (n_items, emb_dim)
    proto_tensor = item_proto_fe.prototypes  # (n_item_protos, emb_dim)

    # Cosine similarity (shifted): 1 + cos(item, proto)
    item_norm = torch.nn.functional.normalize(item_emb_tensor, dim=1)
    proto_norm = torch.nn.functional.normalize(proto_tensor, dim=1)
    sim_matrix = (1 + item_norm @ proto_norm.T).numpy()  # (n_items, n_item_protos)

n_item_protos = item_prototypes.shape[0]
print(f"Item-prototype similarity matrix: {sim_matrix.shape}")

all_proto_results = []
for p_idx in range(n_item_protos):
    top_items = get_top_k_items(sim_matrix, items_info, proto_idx=p_idx, top_k=10)
    top_items['prototype'] = p_idx
    all_proto_results.append(top_items)
    print(f"\nPrototype {p_idx}:")
    # Show key columns
    display_cols = ['product_group_name', 'product_type_name', 'colour_group_name', 'section_name', 'item weight']
    available = [c for c in display_cols if c in top_items.columns]
    print(top_items[available].to_string())

# Save all to CSV
all_proto_df = pd.concat(all_proto_results)
all_proto_df.to_csv(os.path.join(OUTPUT_DIR, 'top_k_items_per_prototype.csv'))
print(f"\nSaved: top_k_items_per_prototype.csv")

# --- 3. Weight Visualization for a sample recommendation ---
print("\n=== Weight visualization for sample recommendation ===")

# Pick a random user and their top recommended item
with torch.no_grad():
    sample_user_id = 42
    user_ids_tensor = torch.tensor([sample_user_id])
    item_ids_tensor = torch.arange(n_items)

    # Get user representation
    user_out = model.user_feature_extractor(user_ids_tensor)  # (1, out_dim)

    # Get all item representations
    item_out = model.item_feature_extractor(item_ids_tensor)  # (n_items, out_dim)

    # Find top recommended item
    scores = (user_out @ item_out.T).squeeze()
    top_item_id = scores.argmax().item()
    print(f"User {sample_user_id} -> Top recommended item: {top_item_id}")
    item_row = items_info[items_info['item_id'] == top_item_id]
    if not item_row.empty:
        print(f"  Item: {item_row.iloc[0].to_dict()}")

    # Extract the components for weight_visualization
    # User side: similarity to user prototypes + item projection
    user_emb_single = user_proto_fe.embedding(user_ids_tensor)  # (1, emb_dim)
    user_emb_norm = torch.nn.functional.normalize(user_emb_single, dim=1)
    user_proto_norm = torch.nn.functional.normalize(user_proto_fe.prototypes, dim=1)
    u_sim_mtx = (1 + user_emb_norm @ user_proto_norm.T).squeeze().numpy()  # (n_user_protos,)

    # Item projection onto user prototype space
    item_emb_single = item_proto_fe.embedding(torch.tensor([top_item_id]))
    # The EmbeddingW projects item embedding to user prototype dimension
    item_proj_fe = model.item_feature_extractor.fe_2  # EmbeddingW
    i_proj = item_proj_fe(torch.tensor([top_item_id])).squeeze().numpy()  # (n_user_protos,)

    # Item side: similarity to item prototypes + user projection
    item_emb_norm = torch.nn.functional.normalize(item_emb_single, dim=1)
    item_proto_norm_t = torch.nn.functional.normalize(item_proto_fe.prototypes, dim=1)
    i_sim_mtx = (1 + item_emb_norm @ item_proto_norm_t.T).squeeze().numpy()  # (n_item_protos,)

    # User projection onto item prototype space
    user_proj_fe = model.user_feature_extractor.fe_2  # EmbeddingW
    u_proj = user_proj_fe(user_ids_tensor).squeeze().numpy()  # (n_item_protos,)

print(f"u_sim_mtx: {u_sim_mtx.shape}, i_proj: {i_proj.shape}")
print(f"i_sim_mtx: {i_sim_mtx.shape}, u_proj: {u_proj.shape}")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

weight_visualization(u_sim_mtx, u_proj, i_sim_mtx, i_proj, annotate_top_k=3)
plt.savefig(os.path.join(OUTPUT_DIR, 'weight_viz_user42.pdf'), format='pdf')
print(f"Saved: weight_viz_user42.pdf")

print(f"\n=== All outputs saved to {OUTPUT_DIR} ===")
