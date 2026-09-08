import argparse
import os

from confs.hyper_params import mf_hyper_params, anchor_hyper_params, user_proto_chose_original_hyper_params, \
    item_proto_chose_original_hyper_params, proto_double_tie_chose_original_hyper_params, debug_hyper_params, \
    wandb_test_hyper_params, feature_item_proto_hyper_params, feature_item_proto_noid_hyper_params, \
    feature_item_proto_f0_hyper_params, feature_item_proto_c_hyper_params, \
    feature_item_proto_c_noid_hyper_params, feature_item_proto_c_noid_iw_hyper_params, \
    feature_item_proto_alpha_hyper_params, attr_item_proto_hyper_params, attr_item_proto_debug_hyper_params, \
    attr_item_proto_debug_knobs_hyper_params, lightfm_tags_hyper_params, lightfm_tags_ids_hyper_params, \
    feature_user_proto_hyper_params, feature_user_proto_noid_hyper_params, feature_user_proto_debug_hyper_params, \
    lightfm_hist_hyper_params, lightfm_hist_ids_hyper_params, \
    user_proto_sm_hyper_params, item_proto_sm_hyper_params, user_proto_sg_hyper_params, \
    item_proto_sg_hyper_params, feature_item_proto_sm_hyper_params, feature_item_proto_noid_sm_hyper_params, \
    feature_user_proto_sm_hyper_params, feature_user_proto_noid_sm_hyper_params
from experiment_helper import start_hyper, start_multiple_hyper
from utilities.consts import SINGLE_SEED

os.environ['WANDB_START_METHOD'] = 'thread'
os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser(description='Start an experiment')

parser.add_argument('--model', '-m', type=str, help='Recommender System model',
                    choices=['mf', 'acf', 'user_proto', 'item_proto', 'user_item_proto', 'feature_item_proto',
                             'feature_item_proto_noid', 'feature_item_proto_f0', 'feature_item_proto_c',
                             'feature_item_proto_c_noid', 'feature_item_proto_c_noid_iw',
                             'feature_item_proto_alpha', 'attr_item_proto',
                             'attr_item_proto_debug', 'attr_item_proto_debug_knobs', 'lightfm_tags',
                             'lightfm_tags_ids', 'feature_user_proto', 'feature_user_proto_noid',
                             'feature_user_proto_debug', 'lightfm_hist', 'lightfm_hist_ids',
                             'user_proto_sm', 'item_proto_sm', 'user_proto_sg', 'item_proto_sg',
                             'feature_item_proto_sm', 'feature_item_proto_noid_sm',
                             'feature_user_proto_sm', 'feature_user_proto_noid_sm',
                             'debug', 'wandb_test'])

parser.add_argument('--dataset', '-d', type=str, help='Recommender System Dataset',
                    choices=['amazon2014', 'ml-1m', 'ml-1m_cold', 'lfm2b-1mon', 'hm_full', 'hm_3_month',
                             'hm_1_month', 'hm_1_month_cold'])

parser.add_argument('--multiple', '-mp',
                    help='Whether to run the experiment across all seeds (see utilities/consts.py)',
                    action='store_true', default=False, required=False)
parser.add_argument('--seed', '-s', help='Seed to set for the experiments', type=int, default=SINGLE_SEED,
                    required=False)

args = parser.parse_args()

model = args.model
dataset = args.dataset
multiple = args.multiple
seed = args.seed

conf_dict = None
if model == 'mf':
    conf_dict = mf_hyper_params
elif model == 'acf':
    conf_dict = anchor_hyper_params
elif model == 'user_proto':
    conf_dict = user_proto_chose_original_hyper_params
elif model == 'item_proto':
    conf_dict = item_proto_chose_original_hyper_params
elif model == 'user_item_proto':
    conf_dict = proto_double_tie_chose_original_hyper_params
elif model == 'feature_item_proto':
    conf_dict = feature_item_proto_hyper_params
elif model == 'feature_item_proto_noid':
    conf_dict = feature_item_proto_noid_hyper_params
elif model == 'feature_item_proto_f0':
    conf_dict = feature_item_proto_f0_hyper_params
elif model == 'feature_item_proto_c':
    conf_dict = feature_item_proto_c_hyper_params
elif model == 'feature_item_proto_c_noid':
    conf_dict = feature_item_proto_c_noid_hyper_params
elif model == 'feature_item_proto_c_noid_iw':
    conf_dict = feature_item_proto_c_noid_iw_hyper_params
elif model == 'feature_item_proto_alpha':
    conf_dict = feature_item_proto_alpha_hyper_params
elif model == 'attr_item_proto':
    conf_dict = attr_item_proto_hyper_params
elif model == 'attr_item_proto_debug':
    conf_dict = attr_item_proto_debug_hyper_params
elif model == 'attr_item_proto_debug_knobs':
    conf_dict = attr_item_proto_debug_knobs_hyper_params
elif model == 'lightfm_tags':
    conf_dict = lightfm_tags_hyper_params
elif model == 'lightfm_tags_ids':
    conf_dict = lightfm_tags_ids_hyper_params
elif model == 'feature_user_proto':
    conf_dict = feature_user_proto_hyper_params
elif model == 'feature_user_proto_noid':
    conf_dict = feature_user_proto_noid_hyper_params
elif model == 'feature_user_proto_debug':
    conf_dict = feature_user_proto_debug_hyper_params
elif model == 'lightfm_hist':
    conf_dict = lightfm_hist_hyper_params
elif model == 'lightfm_hist_ids':
    conf_dict = lightfm_hist_ids_hyper_params
elif model == 'user_proto_sm':
    conf_dict = user_proto_sm_hyper_params
elif model == 'item_proto_sm':
    conf_dict = item_proto_sm_hyper_params
elif model == 'user_proto_sg':
    conf_dict = user_proto_sg_hyper_params
elif model == 'item_proto_sg':
    conf_dict = item_proto_sg_hyper_params
elif model == 'feature_item_proto_sm':
    conf_dict = feature_item_proto_sm_hyper_params
elif model == 'feature_item_proto_noid_sm':
    conf_dict = feature_item_proto_noid_sm_hyper_params
elif model == 'feature_user_proto_sm':
    conf_dict = feature_user_proto_sm_hyper_params
elif model == 'feature_user_proto_noid_sm':
    conf_dict = feature_user_proto_noid_sm_hyper_params
elif model == 'debug':
    conf_dict = debug_hyper_params
elif model == 'wandb_test':
    conf_dict = wandb_test_hyper_params

if multiple:
    start_multiple_hyper(conf_dict, model, dataset)
else:
    start_hyper(conf_dict, model, dataset, seed)
