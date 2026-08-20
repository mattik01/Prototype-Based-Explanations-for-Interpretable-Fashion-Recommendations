# Explanation Deep Dive — Observations

> Matteo's observation log for the scrutiny explanation walkthrough.
> **Only Matteo edits this file** (Claude only on explicit instruction).
> All artifacts regenerated 2026-08-17 with the CURRENT pipeline (commit 833919f,
> incl. the `_noid`/`_f0` reachability repair) from the canonical cluster checkpoints
> (`leo5:/scratch/c7031336/protomf_results/`), on the LEO5 login node.
> Local copies: `Master/experiments/expl_deep_dive/<N.M>_<run>/` — folder numbers match section numbers.
> Graphics are hidden by default: each lives in a one-line HTML comment (description + thumbnail);
> remove the `<!--` / `-->` on a line to show it. Preview in VS Code: `Ctrl+Shift+V`.

## 0. General observations (cross-variant)

**findings:**

- 

## 1. U-ProtoMF (`user_proto`)

### 1.1 H&M

run: `Master/experiments/expl_deep_dive/1.1_user_proto_hm_1_month_s38210573`

**lift**  
files: [breakdown_user42_item670.md](../../experiments/expl_deep_dive/1.1_user_proto_hm_1_month_s38210573/explanations/lift/breakdown_user42_item670.md) · [prototype_names_user.csv](../../experiments/expl_deep_dive/1.1_user_proto_hm_1_month_s38210573/explanations/lift/prototype_names_user.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/1.1_user_proto_hm_1_month_s38210573/explanations/lift/prototype_naming_stats_user.csv) · [top_k_items_per_user_prototype.csv](../../experiments/expl_deep_dive/1.1_user_proto_hm_1_month_s38210573/explanations/lift/top_k_items_per_user_prototype.csv)

**raw**  
files: [breakdown_user42_item670.md](../../experiments/expl_deep_dive/1.1_user_proto_hm_1_month_s38210573/explanations/raw/breakdown_user42_item670.md) · [prototype_names_user.csv](../../experiments/expl_deep_dive/1.1_user_proto_hm_1_month_s38210573/explanations/raw/prototype_names_user.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/1.1_user_proto_hm_1_month_s38210573/explanations/raw/prototype_naming_stats_user.csv) · [top_k_items_per_user_prototype.csv](../../experiments/expl_deep_dive/1.1_user_proto_hm_1_month_s38210573/explanations/raw/top_k_items_per_user_prototype.csv)

**findings:**
- prototypes ultra clustered in a few spots
- double checked tsne implementation proper device for vizualizing it works with directions same as our model, trust neighbourhoods, dont trust distances
- Prototype Naming: 1. identify which items get most score from a user being exactly like that prototype 2. take the 9 features for all of them, 3, score the features of how often they appear in top 20 raw vs. lift (lift: frequency normailized by frequency in the entire catalog) 4. some filter (above  catalog avg, above > 1) 5. top 3 become the name. 
- ATTENTION: column cardinality NOT neutral, raw favors columns with few values, lift those with few which pass the filters. NOT IDEAL but irrelevant for now. confirmed in naming table.

Score breakdown:
- 80% unpersonalized item baseline, the DOCUMENTED effect of Userside item popularity channel through the shifted cosine makes itself visible, also explains the collapsed prototypes to an extent.

### 1.2 ML-1M

run: `Master/experiments/expl_deep_dive/1.2_user_proto_ml-1m_s38210573`

**lift**  
files: [breakdown_user42_item73.md](../../experiments/expl_deep_dive/1.2_user_proto_ml-1m_s38210573/explanations/lift/breakdown_user42_item73.md) · [prototype_names_user.csv](../../experiments/expl_deep_dive/1.2_user_proto_ml-1m_s38210573/explanations/lift/prototype_names_user.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/1.2_user_proto_ml-1m_s38210573/explanations/lift/prototype_naming_stats_user.csv) · [top_k_items_per_user_prototype.csv](../../experiments/expl_deep_dive/1.2_user_proto_ml-1m_s38210573/explanations/lift/top_k_items_per_user_prototype.csv)

**raw**  
files: [breakdown_user42_item73.md](../../experiments/expl_deep_dive/1.2_user_proto_ml-1m_s38210573/explanations/raw/breakdown_user42_item73.md) · [prototype_names_user.csv](../../experiments/expl_deep_dive/1.2_user_proto_ml-1m_s38210573/explanations/raw/prototype_names_user.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/1.2_user_proto_ml-1m_s38210573/explanations/raw/prototype_naming_stats_user.csv) · [top_k_items_per_user_prototype.csv](../../experiments/expl_deep_dive/1.2_user_proto_ml-1m_s38210573/explanations/raw/top_k_items_per_user_prototype.csv)

**findings:**
- More Unpersonalized recommendation MASS, yet the prototype space did not degrade here,
- maybe because of richer user histories? degraded prototype space in h and M could be a function of that ?
- user history tes indeed is sharply differnet, but degeneracy of prototype space is way more determined by regularizers turns out. 

- Prototype Naming with tag baskets: 1. same top 20 items per prototype as in H&M, 2. only 2 columns here: genres + tags, both are lists of different length per movie, 3. a movie counts each tag max once, no matter how many tags it has, 4. score = how many of the 20 have that tag, raw vs. lift (lift: divided by how many movies in the whole catalog have it), same filters (at least 2 of the 20, lift > 1), 5. top 3 become the name.
- ATTENTION: the column bias from 1.1 is way stronger here: ~18 genres that are everywhere (lift can barely go above 1) vs. 1000s of rare tags (lift can go into the hundreds) → names are basically only tags, genres never win. confirmed in naming table.
- ATTENTION 2: some movies have many tags, some almost none (obscure movies are tagged less) → a prototype whose top items have few tags has little material to build a name from. so a bad/empty name does not automatically mean the prototype is meaningless — could just be missing tags.
- 

### 1.3 H&M cold

run: `Master/experiments/expl_deep_dive/1.3_user_proto_hm_1_month_cold_s38210573`

**lift**  
files: [breakdown_user42_item334.md](../../experiments/expl_deep_dive/1.3_user_proto_hm_1_month_cold_s38210573/explanations/lift/breakdown_user42_item334.md) · [prototype_names_user.csv](../../experiments/expl_deep_dive/1.3_user_proto_hm_1_month_cold_s38210573/explanations/lift/prototype_names_user.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/1.3_user_proto_hm_1_month_cold_s38210573/explanations/lift/prototype_naming_stats_user.csv) · [top_k_items_per_user_prototype.csv](../../experiments/expl_deep_dive/1.3_user_proto_hm_1_month_cold_s38210573/explanations/lift/top_k_items_per_user_prototype.csv)

**raw**  
files: [breakdown_user42_item334.md](../../experiments/expl_deep_dive/1.3_user_proto_hm_1_month_cold_s38210573/explanations/raw/breakdown_user42_item334.md) · [prototype_names_user.csv](../../experiments/expl_deep_dive/1.3_user_proto_hm_1_month_cold_s38210573/explanations/raw/prototype_names_user.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/1.3_user_proto_hm_1_month_cold_s38210573/explanations/raw/prototype_naming_stats_user.csv) · [top_k_items_per_user_prototype.csv](../../experiments/expl_deep_dive/1.3_user_proto_hm_1_month_cold_s38210573/explanations/raw/top_k_items_per_user_prototype.csv)

**findings:**
- how do the cold runs work? take the normal config, freeze it, take 20% of tiems from the split in train val and test, remove all interaction, retrain. in the test set these items become cold pool, and then evaluate cold vs. cold and cold vs. all which includes items the model has never seen. 
- explanations figures are very similar to 1.1 which is expected. for 2.3 there is the added clound of blue dots that are without interactions but they live in the same universe so be aware of that.

## 2. I-ProtoMF (`item_proto`)

### 2.1 H&M

run: `Master/experiments/expl_deep_dive/2.1_item_proto_hm_1_month_s38210573`

**lift**  
files: [breakdown_user42_item339.md](../../experiments/expl_deep_dive/2.1_item_proto_hm_1_month_s38210573/explanations/lift/breakdown_user42_item339.md) · [prototype_names_item.csv](../../experiments/expl_deep_dive/2.1_item_proto_hm_1_month_s38210573/explanations/lift/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/2.1_item_proto_hm_1_month_s38210573/explanations/lift/prototype_naming_stats_item.csv) · [top_k_items_per_item_prototype.csv](../../experiments/expl_deep_dive/2.1_item_proto_hm_1_month_s38210573/explanations/lift/top_k_items_per_item_prototype.csv)

**raw**  
files: [breakdown_user42_item339.md](../../experiments/expl_deep_dive/2.1_item_proto_hm_1_month_s38210573/explanations/raw/breakdown_user42_item339.md) · [prototype_names_item.csv](../../experiments/expl_deep_dive/2.1_item_proto_hm_1_month_s38210573/explanations/raw/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/2.1_item_proto_hm_1_month_s38210573/explanations/raw/prototype_naming_stats_item.csv) · [top_k_items_per_item_prototype.csv](../../experiments/expl_deep_dive/2.1_item_proto_hm_1_month_s38210573/explanations/raw/top_k_items_per_item_prototype.csv)

**findings:**
- collapsed prototypes even harder, Item channel is not causing it. prototypes are just a vehicle for popularity
- small multiples super redundant in a uuper collapsed space

- 

### 2.2 ML-1M

run: `Master/experiments/expl_deep_dive/2.2_item_proto_ml-1m_s38210573`

**lift**  
files: [breakdown_user42_item30.md](../../experiments/expl_deep_dive/2.2_item_proto_ml-1m_s38210573/explanations/lift/breakdown_user42_item30.md) · [prototype_names_item.csv](../../experiments/expl_deep_dive/2.2_item_proto_ml-1m_s38210573/explanations/lift/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/2.2_item_proto_ml-1m_s38210573/explanations/lift/prototype_naming_stats_item.csv) · [top_k_items_per_item_prototype.csv](../../experiments/expl_deep_dive/2.2_item_proto_ml-1m_s38210573/explanations/lift/top_k_items_per_item_prototype.csv)

**raw**  
files: [breakdown_user42_item30.md](../../experiments/expl_deep_dive/2.2_item_proto_ml-1m_s38210573/explanations/raw/breakdown_user42_item30.md) · [prototype_names_item.csv](../../experiments/expl_deep_dive/2.2_item_proto_ml-1m_s38210573/explanations/raw/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/2.2_item_proto_ml-1m_s38210573/explanations/raw/prototype_naming_stats_item.csv) · [top_k_items_per_item_prototype.csv](../../experiments/expl_deep_dive/2.2_item_proto_ml-1m_s38210573/explanations/raw/top_k_items_per_item_prototype.csv)

**findings:**
-ultra clustered, all items glued to some prototype.
-regularizer working better, it seems for h and m hostores are too small for anything beyond mostly popularity to work properly, designing new h and m higher core set must be necessary. 
- 

### 2.3 H&M cold

run: `Master/experiments/expl_deep_dive/2.3_item_proto_hm_1_month_cold_s38210573`

**lift**  
files: [breakdown_user42_item12218.md](../../experiments/expl_deep_dive/2.3_item_proto_hm_1_month_cold_s38210573/explanations/lift/breakdown_user42_item12218.md) · [prototype_names_item.csv](../../experiments/expl_deep_dive/2.3_item_proto_hm_1_month_cold_s38210573/explanations/lift/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/2.3_item_proto_hm_1_month_cold_s38210573/explanations/lift/prototype_naming_stats_item.csv) · [top_k_items_per_item_prototype.csv](../../experiments/expl_deep_dive/2.3_item_proto_hm_1_month_cold_s38210573/explanations/lift/top_k_items_per_item_prototype.csv)

**raw**  
files: [breakdown_user42_item12218.md](../../experiments/expl_deep_dive/2.3_item_proto_hm_1_month_cold_s38210573/explanations/raw/breakdown_user42_item12218.md) · [prototype_names_item.csv](../../experiments/expl_deep_dive/2.3_item_proto_hm_1_month_cold_s38210573/explanations/raw/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/2.3_item_proto_hm_1_month_cold_s38210573/explanations/raw/prototype_naming_stats_item.csv) · [top_k_items_per_item_prototype.csv](../../experiments/expl_deep_dive/2.3_item_proto_hm_1_month_cold_s38210573/explanations/raw/top_k_items_per_item_prototype.csv)

**findings:**
-irrel, same stuff
- 

## 3. UI-ProtoMF (`user_item_proto`)

### 3.1 H&M

run: `Master/experiments/expl_deep_dive/3.1_user_item_proto_hm_1_month_s38210573`

**lift**  
files: [prototype_names_item.csv](../../experiments/expl_deep_dive/3.1_user_item_proto_hm_1_month_s38210573/explanations/lift/prototype_names_item.csv) · [prototype_names_user.csv](../../experiments/expl_deep_dive/3.1_user_item_proto_hm_1_month_s38210573/explanations/lift/prototype_names_user.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/3.1_user_item_proto_hm_1_month_s38210573/explanations/lift/prototype_naming_stats_item.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/3.1_user_item_proto_hm_1_month_s38210573/explanations/lift/prototype_naming_stats_user.csv) · [top_k_items_per_item_prototype.csv](../../experiments/expl_deep_dive/3.1_user_item_proto_hm_1_month_s38210573/explanations/lift/top_k_items_per_item_prototype.csv) · [top_k_items_per_user_prototype.csv](../../experiments/expl_deep_dive/3.1_user_item_proto_hm_1_month_s38210573/explanations/lift/top_k_items_per_user_prototype.csv)

**raw**  
files: [prototype_names_item.csv](../../experiments/expl_deep_dive/3.1_user_item_proto_hm_1_month_s38210573/explanations/raw/prototype_names_item.csv) · [prototype_names_user.csv](../../experiments/expl_deep_dive/3.1_user_item_proto_hm_1_month_s38210573/explanations/raw/prototype_names_user.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/3.1_user_item_proto_hm_1_month_s38210573/explanations/raw/prototype_naming_stats_item.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/3.1_user_item_proto_hm_1_month_s38210573/explanations/raw/prototype_naming_stats_user.csv) · [top_k_items_per_item_prototype.csv](../../experiments/expl_deep_dive/3.1_user_item_proto_hm_1_month_s38210573/explanations/raw/top_k_items_per_item_prototype.csv) · [top_k_items_per_user_prototype.csv](../../experiments/expl_deep_dive/3.1_user_item_proto_hm_1_month_s38210573/explanations/raw/top_k_items_per_user_prototype.csv)

**findings:**
prototypes stay rather healthy, double job for prototypes punishes duplication. 
breakdown not there yet, on purpose, deffered for fufi stage.

- 

### 3.2 ML-1M

run: `Master/experiments/expl_deep_dive/3.2_user_item_proto_ml-1m_s38210573`

**lift**  
files: [prototype_names_item.csv](../../experiments/expl_deep_dive/3.2_user_item_proto_ml-1m_s38210573/explanations/lift/prototype_names_item.csv) · [prototype_names_user.csv](../../experiments/expl_deep_dive/3.2_user_item_proto_ml-1m_s38210573/explanations/lift/prototype_names_user.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/3.2_user_item_proto_ml-1m_s38210573/explanations/lift/prototype_naming_stats_item.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/3.2_user_item_proto_ml-1m_s38210573/explanations/lift/prototype_naming_stats_user.csv) · [top_k_items_per_item_prototype.csv](../../experiments/expl_deep_dive/3.2_user_item_proto_ml-1m_s38210573/explanations/lift/top_k_items_per_item_prototype.csv) · [top_k_items_per_user_prototype.csv](../../experiments/expl_deep_dive/3.2_user_item_proto_ml-1m_s38210573/explanations/lift/top_k_items_per_user_prototype.csv)

**raw**  
files: [prototype_names_item.csv](../../experiments/expl_deep_dive/3.2_user_item_proto_ml-1m_s38210573/explanations/raw/prototype_names_item.csv) · [prototype_names_user.csv](../../experiments/expl_deep_dive/3.2_user_item_proto_ml-1m_s38210573/explanations/raw/prototype_names_user.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/3.2_user_item_proto_ml-1m_s38210573/explanations/raw/prototype_naming_stats_item.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/3.2_user_item_proto_ml-1m_s38210573/explanations/raw/prototype_naming_stats_user.csv) · [top_k_items_per_item_prototype.csv](../../experiments/expl_deep_dive/3.2_user_item_proto_ml-1m_s38210573/explanations/raw/top_k_items_per_item_prototype.csv) · [top_k_items_per_user_prototype.csv](../../experiments/expl_deep_dive/3.2_user_item_proto_ml-1m_s38210573/explanations/raw/top_k_items_per_user_prototype.csv)

**findings:**

- discussed

### 3.3 H&M cold

run: `Master/experiments/expl_deep_dive/3.3_user_item_proto_hm_1_month_cold_s38210573`

**lift**  
files: [prototype_names_item.csv](../../experiments/expl_deep_dive/3.3_user_item_proto_hm_1_month_cold_s38210573/explanations/lift/prototype_names_item.csv) · [prototype_names_user.csv](../../experiments/expl_deep_dive/3.3_user_item_proto_hm_1_month_cold_s38210573/explanations/lift/prototype_names_user.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/3.3_user_item_proto_hm_1_month_cold_s38210573/explanations/lift/prototype_naming_stats_item.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/3.3_user_item_proto_hm_1_month_cold_s38210573/explanations/lift/prototype_naming_stats_user.csv) · [top_k_items_per_item_prototype.csv](../../experiments/expl_deep_dive/3.3_user_item_proto_hm_1_month_cold_s38210573/explanations/lift/top_k_items_per_item_prototype.csv) · [top_k_items_per_user_prototype.csv](../../experiments/expl_deep_dive/3.3_user_item_proto_hm_1_month_cold_s38210573/explanations/lift/top_k_items_per_user_prototype.csv)

**raw**  
files: [prototype_names_item.csv](../../experiments/expl_deep_dive/3.3_user_item_proto_hm_1_month_cold_s38210573/explanations/raw/prototype_names_item.csv) · [prototype_names_user.csv](../../experiments/expl_deep_dive/3.3_user_item_proto_hm_1_month_cold_s38210573/explanations/raw/prototype_names_user.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/3.3_user_item_proto_hm_1_month_cold_s38210573/explanations/raw/prototype_naming_stats_item.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/3.3_user_item_proto_hm_1_month_cold_s38210573/explanations/raw/prototype_naming_stats_user.csv) · [top_k_items_per_item_prototype.csv](../../experiments/expl_deep_dive/3.3_user_item_proto_hm_1_month_cold_s38210573/explanations/raw/top_k_items_per_item_prototype.csv) · [top_k_items_per_user_prototype.csv](../../experiments/expl_deep_dive/3.3_user_item_proto_hm_1_month_cold_s38210573/explanations/raw/top_k_items_per_user_prototype.csv)

**findings:**

- discussed

## 4. fI-ProtoMF (`feature_item_proto`, dc01)

### 4.1 H&M

run: `Master/experiments/expl_deep_dive/4.1_feature_item_proto_hm_1_month_s38210573`

**cosine**  
files: [breakdown_user42_item670.md](../../experiments/expl_deep_dive/4.1_feature_item_proto_hm_1_month_s38210573/explanations/cosine/breakdown_user42_item670.md) · [prototype_names_item.csv](../../experiments/expl_deep_dive/4.1_feature_item_proto_hm_1_month_s38210573/explanations/cosine/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/4.1_feature_item_proto_hm_1_month_s38210573/explanations/cosine/prototype_naming_stats_item.csv) · [top_k_items_per_item_prototype.csv](../../experiments/expl_deep_dive/4.1_feature_item_proto_hm_1_month_s38210573/explanations/cosine/top_k_items_per_item_prototype.csv)

**lift**  
files: [prototype_names_item.csv](../../experiments/expl_deep_dive/4.1_feature_item_proto_hm_1_month_s38210573/explanations/lift/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/4.1_feature_item_proto_hm_1_month_s38210573/explanations/lift/prototype_naming_stats_item.csv)

**raw**  
files: [prototype_names_item.csv](../../experiments/expl_deep_dive/4.1_feature_item_proto_hm_1_month_s38210573/explanations/raw/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/4.1_feature_item_proto_hm_1_month_s38210573/explanations/raw/prototype_naming_stats_item.csv)

**findings:**

- 

### 4.2 ML-1M

run: `Master/experiments/expl_deep_dive/4.2_feature_item_proto_ml-1m_s38210573`

**cosine**  
files: [breakdown_user42_item107.md](../../experiments/expl_deep_dive/4.2_feature_item_proto_ml-1m_s38210573/explanations/cosine/breakdown_user42_item107.md) · [prototype_names_item.csv](../../experiments/expl_deep_dive/4.2_feature_item_proto_ml-1m_s38210573/explanations/cosine/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/4.2_feature_item_proto_ml-1m_s38210573/explanations/cosine/prototype_naming_stats_item.csv) · [top_k_items_per_item_prototype.csv](../../experiments/expl_deep_dive/4.2_feature_item_proto_ml-1m_s38210573/explanations/cosine/top_k_items_per_item_prototype.csv)

**lift**  
files: [prototype_names_item.csv](../../experiments/expl_deep_dive/4.2_feature_item_proto_ml-1m_s38210573/explanations/lift/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/4.2_feature_item_proto_ml-1m_s38210573/explanations/lift/prototype_naming_stats_item.csv)

**raw**  
files: [prototype_names_item.csv](../../experiments/expl_deep_dive/4.2_feature_item_proto_ml-1m_s38210573/explanations/raw/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/4.2_feature_item_proto_ml-1m_s38210573/explanations/raw/prototype_naming_stats_item.csv)

**findings:**

- 

### 4.3 H&M cold

run: `Master/experiments/expl_deep_dive/4.3_feature_item_proto_hm_1_month_cold_s38210573`

**cosine**  
files: [breakdown_user42_item9580.md](../../experiments/expl_deep_dive/4.3_feature_item_proto_hm_1_month_cold_s38210573/explanations/cosine/breakdown_user42_item9580.md) · [prototype_names_item.csv](../../experiments/expl_deep_dive/4.3_feature_item_proto_hm_1_month_cold_s38210573/explanations/cosine/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/4.3_feature_item_proto_hm_1_month_cold_s38210573/explanations/cosine/prototype_naming_stats_item.csv) · [top_k_items_per_item_prototype.csv](../../experiments/expl_deep_dive/4.3_feature_item_proto_hm_1_month_cold_s38210573/explanations/cosine/top_k_items_per_item_prototype.csv)

**lift**  
files: [prototype_names_item.csv](../../experiments/expl_deep_dive/4.3_feature_item_proto_hm_1_month_cold_s38210573/explanations/lift/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/4.3_feature_item_proto_hm_1_month_cold_s38210573/explanations/lift/prototype_naming_stats_item.csv)

**raw**  
files: [prototype_names_item.csv](../../experiments/expl_deep_dive/4.3_feature_item_proto_hm_1_month_cold_s38210573/explanations/raw/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/4.3_feature_item_proto_hm_1_month_cold_s38210573/explanations/raw/prototype_naming_stats_item.csv)

**findings:**

- 

## 5. fI ablation `_noid`

### 5.1 H&M

run: `Master/experiments/expl_deep_dive/5.1_feature_item_proto_noid_hm_1_month_s38210573`

**cosine**  
files: [breakdown_user42_item384.md](../../experiments/expl_deep_dive/5.1_feature_item_proto_noid_hm_1_month_s38210573/explanations/cosine/breakdown_user42_item384.md) · [prototype_names_item.csv](../../experiments/expl_deep_dive/5.1_feature_item_proto_noid_hm_1_month_s38210573/explanations/cosine/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/5.1_feature_item_proto_noid_hm_1_month_s38210573/explanations/cosine/prototype_naming_stats_item.csv) · [top_k_items_per_item_prototype.csv](../../experiments/expl_deep_dive/5.1_feature_item_proto_noid_hm_1_month_s38210573/explanations/cosine/top_k_items_per_item_prototype.csv)

**lift**  
files: [prototype_names_item.csv](../../experiments/expl_deep_dive/5.1_feature_item_proto_noid_hm_1_month_s38210573/explanations/lift/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/5.1_feature_item_proto_noid_hm_1_month_s38210573/explanations/lift/prototype_naming_stats_item.csv)

**raw**  
files: [prototype_names_item.csv](../../experiments/expl_deep_dive/5.1_feature_item_proto_noid_hm_1_month_s38210573/explanations/raw/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/5.1_feature_item_proto_noid_hm_1_month_s38210573/explanations/raw/prototype_naming_stats_item.csv)

**findings:**

- 

### 5.2 ML-1M

run: `Master/experiments/expl_deep_dive/5.2_feature_item_proto_noid_ml-1m_s38210573`

**cosine**  
files: [breakdown_user42_item66.md](../../experiments/expl_deep_dive/5.2_feature_item_proto_noid_ml-1m_s38210573/explanations/cosine/breakdown_user42_item66.md) · [prototype_names_item.csv](../../experiments/expl_deep_dive/5.2_feature_item_proto_noid_ml-1m_s38210573/explanations/cosine/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/5.2_feature_item_proto_noid_ml-1m_s38210573/explanations/cosine/prototype_naming_stats_item.csv) · [top_k_items_per_item_prototype.csv](../../experiments/expl_deep_dive/5.2_feature_item_proto_noid_ml-1m_s38210573/explanations/cosine/top_k_items_per_item_prototype.csv)

**lift**  
files: [prototype_names_item.csv](../../experiments/expl_deep_dive/5.2_feature_item_proto_noid_ml-1m_s38210573/explanations/lift/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/5.2_feature_item_proto_noid_ml-1m_s38210573/explanations/lift/prototype_naming_stats_item.csv)

**raw**  
files: [prototype_names_item.csv](../../experiments/expl_deep_dive/5.2_feature_item_proto_noid_ml-1m_s38210573/explanations/raw/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/5.2_feature_item_proto_noid_ml-1m_s38210573/explanations/raw/prototype_naming_stats_item.csv)

**findings:**

- 

### 5.3 H&M cold

run: `Master/experiments/expl_deep_dive/5.3_feature_item_proto_noid_hm_1_month_cold_s38210573`

**cosine**  
files: [breakdown_user42_item4627.md](../../experiments/expl_deep_dive/5.3_feature_item_proto_noid_hm_1_month_cold_s38210573/explanations/cosine/breakdown_user42_item4627.md) · [prototype_names_item.csv](../../experiments/expl_deep_dive/5.3_feature_item_proto_noid_hm_1_month_cold_s38210573/explanations/cosine/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/5.3_feature_item_proto_noid_hm_1_month_cold_s38210573/explanations/cosine/prototype_naming_stats_item.csv) · [top_k_items_per_item_prototype.csv](../../experiments/expl_deep_dive/5.3_feature_item_proto_noid_hm_1_month_cold_s38210573/explanations/cosine/top_k_items_per_item_prototype.csv)

**lift**  
files: [prototype_names_item.csv](../../experiments/expl_deep_dive/5.3_feature_item_proto_noid_hm_1_month_cold_s38210573/explanations/lift/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/5.3_feature_item_proto_noid_hm_1_month_cold_s38210573/explanations/lift/prototype_naming_stats_item.csv)

**raw**  
files: [prototype_names_item.csv](../../experiments/expl_deep_dive/5.3_feature_item_proto_noid_hm_1_month_cold_s38210573/explanations/raw/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/5.3_feature_item_proto_noid_hm_1_month_cold_s38210573/explanations/raw/prototype_naming_stats_item.csv)

**findings:**

- 

## 6. fI ablation `_f0`

### 6.1 H&M

run: `Master/experiments/expl_deep_dive/6.1_feature_item_proto_f0_hm_1_month_s38210573`

**cosine**  
files: [breakdown_user42_item2390.md](../../experiments/expl_deep_dive/6.1_feature_item_proto_f0_hm_1_month_s38210573/explanations/cosine/breakdown_user42_item2390.md) · [prototype_names_item.csv](../../experiments/expl_deep_dive/6.1_feature_item_proto_f0_hm_1_month_s38210573/explanations/cosine/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/6.1_feature_item_proto_f0_hm_1_month_s38210573/explanations/cosine/prototype_naming_stats_item.csv) · [top_k_items_per_item_prototype.csv](../../experiments/expl_deep_dive/6.1_feature_item_proto_f0_hm_1_month_s38210573/explanations/cosine/top_k_items_per_item_prototype.csv)

**lift**  
files: [prototype_names_item.csv](../../experiments/expl_deep_dive/6.1_feature_item_proto_f0_hm_1_month_s38210573/explanations/lift/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/6.1_feature_item_proto_f0_hm_1_month_s38210573/explanations/lift/prototype_naming_stats_item.csv)

**raw**  
files: [prototype_names_item.csv](../../experiments/expl_deep_dive/6.1_feature_item_proto_f0_hm_1_month_s38210573/explanations/raw/prototype_names_item.csv) · [prototype_naming_stats_item.csv](../../experiments/expl_deep_dive/6.1_feature_item_proto_f0_hm_1_month_s38210573/explanations/raw/prototype_naming_stats_item.csv)

**findings:**

- 

## 7. fU-ProtoMF (`feature_user_proto`, dc05)

### 7.1 H&M

run: `Master/experiments/expl_deep_dive/7.1_feature_user_proto_hm_1_month_s38210573`

**cosine**  
files: [breakdown_user42_item1424.md](../../experiments/expl_deep_dive/7.1_feature_user_proto_hm_1_month_s38210573/explanations/cosine/breakdown_user42_item1424.md) · [prototype_names_user.csv](../../experiments/expl_deep_dive/7.1_feature_user_proto_hm_1_month_s38210573/explanations/cosine/prototype_names_user.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/7.1_feature_user_proto_hm_1_month_s38210573/explanations/cosine/prototype_naming_stats_user.csv) · [top_k_items_per_user_prototype.csv](../../experiments/expl_deep_dive/7.1_feature_user_proto_hm_1_month_s38210573/explanations/cosine/top_k_items_per_user_prototype.csv)

**lift**  
files: [prototype_names_user.csv](../../experiments/expl_deep_dive/7.1_feature_user_proto_hm_1_month_s38210573/explanations/lift/prototype_names_user.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/7.1_feature_user_proto_hm_1_month_s38210573/explanations/lift/prototype_naming_stats_user.csv)

**raw**  
files: [prototype_names_user.csv](../../experiments/expl_deep_dive/7.1_feature_user_proto_hm_1_month_s38210573/explanations/raw/prototype_names_user.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/7.1_feature_user_proto_hm_1_month_s38210573/explanations/raw/prototype_naming_stats_user.csv)

**findings:**

- 

### 7.2 ML-1M

run: `Master/experiments/expl_deep_dive/7.2_feature_user_proto_ml-1m_s38210573`

**cosine**  
files: [breakdown_user42_item73.md](../../experiments/expl_deep_dive/7.2_feature_user_proto_ml-1m_s38210573/explanations/cosine/breakdown_user42_item73.md) · [prototype_names_user.csv](../../experiments/expl_deep_dive/7.2_feature_user_proto_ml-1m_s38210573/explanations/cosine/prototype_names_user.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/7.2_feature_user_proto_ml-1m_s38210573/explanations/cosine/prototype_naming_stats_user.csv) · [top_k_items_per_user_prototype.csv](../../experiments/expl_deep_dive/7.2_feature_user_proto_ml-1m_s38210573/explanations/cosine/top_k_items_per_user_prototype.csv)

**lift**  
files: [prototype_names_user.csv](../../experiments/expl_deep_dive/7.2_feature_user_proto_ml-1m_s38210573/explanations/lift/prototype_names_user.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/7.2_feature_user_proto_ml-1m_s38210573/explanations/lift/prototype_naming_stats_user.csv)

**raw**  
files: [prototype_names_user.csv](../../experiments/expl_deep_dive/7.2_feature_user_proto_ml-1m_s38210573/explanations/raw/prototype_names_user.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/7.2_feature_user_proto_ml-1m_s38210573/explanations/raw/prototype_naming_stats_user.csv)

**findings:**

- 

### 7.3 H&M cold

run: `Master/experiments/expl_deep_dive/7.3_feature_user_proto_hm_1_month_cold_s38210573`

**cosine**  
files: [breakdown_user42_item334.md](../../experiments/expl_deep_dive/7.3_feature_user_proto_hm_1_month_cold_s38210573/explanations/cosine/breakdown_user42_item334.md) · [prototype_names_user.csv](../../experiments/expl_deep_dive/7.3_feature_user_proto_hm_1_month_cold_s38210573/explanations/cosine/prototype_names_user.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/7.3_feature_user_proto_hm_1_month_cold_s38210573/explanations/cosine/prototype_naming_stats_user.csv) · [top_k_items_per_user_prototype.csv](../../experiments/expl_deep_dive/7.3_feature_user_proto_hm_1_month_cold_s38210573/explanations/cosine/top_k_items_per_user_prototype.csv)

**lift**  
files: [prototype_names_user.csv](../../experiments/expl_deep_dive/7.3_feature_user_proto_hm_1_month_cold_s38210573/explanations/lift/prototype_names_user.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/7.3_feature_user_proto_hm_1_month_cold_s38210573/explanations/lift/prototype_naming_stats_user.csv)

**raw**  
files: [prototype_names_user.csv](../../experiments/expl_deep_dive/7.3_feature_user_proto_hm_1_month_cold_s38210573/explanations/raw/prototype_names_user.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/7.3_feature_user_proto_hm_1_month_cold_s38210573/explanations/raw/prototype_naming_stats_user.csv)

**findings:**

- 

## 8. fU ablation `_noid`

### 8.1 H&M

run: `Master/experiments/expl_deep_dive/8.1_feature_user_proto_noid_hm_1_month_s38210573`

**cosine**  
files: [breakdown_user42_item334.md](../../experiments/expl_deep_dive/8.1_feature_user_proto_noid_hm_1_month_s38210573/explanations/cosine/breakdown_user42_item334.md) · [prototype_names_user.csv](../../experiments/expl_deep_dive/8.1_feature_user_proto_noid_hm_1_month_s38210573/explanations/cosine/prototype_names_user.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/8.1_feature_user_proto_noid_hm_1_month_s38210573/explanations/cosine/prototype_naming_stats_user.csv) · [top_k_items_per_user_prototype.csv](../../experiments/expl_deep_dive/8.1_feature_user_proto_noid_hm_1_month_s38210573/explanations/cosine/top_k_items_per_user_prototype.csv)

**lift**  
files: [prototype_names_user.csv](../../experiments/expl_deep_dive/8.1_feature_user_proto_noid_hm_1_month_s38210573/explanations/lift/prototype_names_user.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/8.1_feature_user_proto_noid_hm_1_month_s38210573/explanations/lift/prototype_naming_stats_user.csv)

**raw**  
files: [prototype_names_user.csv](../../experiments/expl_deep_dive/8.1_feature_user_proto_noid_hm_1_month_s38210573/explanations/raw/prototype_names_user.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/8.1_feature_user_proto_noid_hm_1_month_s38210573/explanations/raw/prototype_naming_stats_user.csv)

**findings:**

- 

### 8.2 ML-1M

run: `Master/experiments/expl_deep_dive/8.2_feature_user_proto_noid_ml-1m_s38210573`

**cosine**  
files: [breakdown_user42_item73.md](../../experiments/expl_deep_dive/8.2_feature_user_proto_noid_ml-1m_s38210573/explanations/cosine/breakdown_user42_item73.md) · [prototype_names_user.csv](../../experiments/expl_deep_dive/8.2_feature_user_proto_noid_ml-1m_s38210573/explanations/cosine/prototype_names_user.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/8.2_feature_user_proto_noid_ml-1m_s38210573/explanations/cosine/prototype_naming_stats_user.csv) · [top_k_items_per_user_prototype.csv](../../experiments/expl_deep_dive/8.2_feature_user_proto_noid_ml-1m_s38210573/explanations/cosine/top_k_items_per_user_prototype.csv)

**lift**  
files: [prototype_names_user.csv](../../experiments/expl_deep_dive/8.2_feature_user_proto_noid_ml-1m_s38210573/explanations/lift/prototype_names_user.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/8.2_feature_user_proto_noid_ml-1m_s38210573/explanations/lift/prototype_naming_stats_user.csv)

**raw**  
files: [prototype_names_user.csv](../../experiments/expl_deep_dive/8.2_feature_user_proto_noid_ml-1m_s38210573/explanations/raw/prototype_names_user.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/8.2_feature_user_proto_noid_ml-1m_s38210573/explanations/raw/prototype_naming_stats_user.csv)

**findings:**

- 

### 8.3 H&M cold

run: `Master/experiments/expl_deep_dive/8.3_feature_user_proto_noid_hm_1_month_cold_s38210573`

**cosine**  
files: [breakdown_user42_item334.md](../../experiments/expl_deep_dive/8.3_feature_user_proto_noid_hm_1_month_cold_s38210573/explanations/cosine/breakdown_user42_item334.md) · [prototype_names_user.csv](../../experiments/expl_deep_dive/8.3_feature_user_proto_noid_hm_1_month_cold_s38210573/explanations/cosine/prototype_names_user.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/8.3_feature_user_proto_noid_hm_1_month_cold_s38210573/explanations/cosine/prototype_naming_stats_user.csv) · [top_k_items_per_user_prototype.csv](../../experiments/expl_deep_dive/8.3_feature_user_proto_noid_hm_1_month_cold_s38210573/explanations/cosine/top_k_items_per_user_prototype.csv)

**lift**  
files: [prototype_names_user.csv](../../experiments/expl_deep_dive/8.3_feature_user_proto_noid_hm_1_month_cold_s38210573/explanations/lift/prototype_names_user.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/8.3_feature_user_proto_noid_hm_1_month_cold_s38210573/explanations/lift/prototype_naming_stats_user.csv)

**raw**  
files: [prototype_names_user.csv](../../experiments/expl_deep_dive/8.3_feature_user_proto_noid_hm_1_month_cold_s38210573/explanations/raw/prototype_names_user.csv) · [prototype_naming_stats_user.csv](../../experiments/expl_deep_dive/8.3_feature_user_proto_noid_hm_1_month_cold_s38210573/explanations/raw/prototype_naming_stats_user.csv)

**findings:**

- 

---

## Historical explanation folders (older pipeline versions, kept as-is)

- `Master/experiments/s07_reference/` — S0.7 reference pulls (hosts, H&M)
- `Master/experiments/sc6_runs/` — SC.6-era pulls (candidates + ml-1m hosts)
- `Master/experiments/results/` — mixed older runs; NOTE: `item_proto`/`user_item_proto`
  H&M and `item_proto` ml-1m here are DIFFERENT runs than the canonical cluster ones
  (md5-verified 2026-08-17)
- `Master/experiments/replication/` — paper replication
