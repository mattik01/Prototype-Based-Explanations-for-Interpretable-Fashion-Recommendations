"""Collector for the dc09 SC.8 one-off measurements (2026-10-30).

Reads <readout_root>/<run>/readout.json written by Master/scripts/sc8_fufi_readout.py on the eight dc09
100-trial checkpoints plus the UI host (A only) and fI (C, E only), and prints one table per instrument
and dataset. Working evidence (F-DC01-06 restriction), not thesis numbers.
Usage: python3 collect_sc8_readout.py <readout_root> > sc8_readout.md   (stdlib only)
"""
import json
import os
import sys

ROOT = sys.argv[1]
ARMS = ['feature_user_item_proto', 'feature_user_item_proto_noid', 'feature_user_item_proto_sep',
        'feature_user_item_proto_noid_sep']
SHORT = {'feature_user_item_proto': 'fUfI', 'feature_user_item_proto_noid': 'fUfI-noid',
         'feature_user_item_proto_sep': 'fUfI-sep', 'feature_user_item_proto_noid_sep': 'fUfI-noid-sep',
         'user_item_proto': 'UI host', 'feature_item_proto': 'fI'}


def load(m, ds):
    p = os.path.join(ROOT, f'{m}_{ds}_s38210573_n100', 'readout.json')
    return json.load(open(p)) if os.path.exists(p) else None


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def f(x, n=3):
    return '—' if x is None else f'{x:.{n}f}'


print('# dc09 SC.8 one-off measurements — collected (working evidence, not thesis numbers)\n')
print('Generator: `Master/scripts/sc8_fufi_readout.py` (LEO5 login node, CPU, scratch code copy at 921d74a, '
      'checkpoints read only; launcher `/scratch/c7031336/smoke_runs/dc09/sc8_real/readout.sh`); '
      'collector: `Master/temp/dc_checks/dc09/collect_sc8_readout.py`.\n')
for ds in ('hm_1_month', 'ml-1m'):
    print(f'## {ds}\n')
    print('**A — common-mode decomposition (exact, all users × all items; fixed by the scores).** '
          'Common share = SS_item / (SS_item + SS_int); sd columns for the personalised part.\n')
    print('| model | personalised common | U-half common | I-half common | sd item / user / interaction (pers.) '
          '| ρ(pers. common, log pop) | B_item ρ(log pop) | score common |')
    print('|---|---|---|---|---|---|---|---|')
    for m in ARMS + ['user_item_proto']:
        d = load(m, ds)
        A = g(d, 'A_common_mode') or {}
        p = A.get('personalised', {})
        print(f"| {SHORT[m]} | {f(p.get('common_share'))} | {f(g(A, 'U_half', 'common_share'))} | "
              f"{f(g(A, 'I_half', 'common_share'))} | {f(p.get('sd_item_main'), 2)} / {f(p.get('sd_user_main'), 2)} / "
              f"{f(p.get('sd_interaction'), 2)} | {f(p.get('spearman_common_part_vs_log_popularity'), 2)} | "
              f"{f(g(A, 'B_item', 'spearman_vs_log_popularity'), 2)} | {f(g(A, 'score', 'common_share'))} |")
    print('\nUser-ID line (ids arms; 3b-xii comparator — NOT fixed by the scores, F-DC09-13/-21; where training '
          'left the mass): rank-SS ratio of the ID line / its common share, personalised part.\n')
    for m in ('feature_user_item_proto', 'feature_user_item_proto_sep'):
        u = g(load(m, ds), 'A_common_mode', 'user_id_line', 'personalised') or {}
        print(f"- {SHORT[m]}: ID-line rank SS / personalised rank SS {f(u.get('id_line_rank_ss_ratio'))}, "
              f"ID-line common share {f(u.get('id_line_common_share'))}")
    print('\n**B — same-word vs different-word pairs in the personalised part (exact per pair; noid arms fixed by '
          'the scores, ids arms relative to the field-mean read-out).** Means over pairs; same-word / mean = '
          'share of the mean personalised score carried by same-word pairs.\n')
    print('| model | set | mean pers. | same-word | different-word | ID-involving | same-word / mean | pairs with overlap |')
    print('|---|---|---|---|---|---|---|---|')
    for m in ARMS:
        B = g(load(m, ds), 'B_word_pairs') or {}
        for s in ('top10_unseen', 'test_item', 'random10_unseen'):
            b = B.get(s, {})
            print(f"| {SHORT[m]} | {s} | {f(b.get('mean_personalised'), 2)} | {f(b.get('mean_same_word'), 2)} | "
                  f"{f(b.get('mean_different_word'), 2)} | {f(b.get('mean_id_involving'), 2)} | "
                  f"{f(b.get('share_same_word_of_mean'), 2)} | {f(b.get('frac_pairs_with_overlap'))} |")
    c = g(load('feature_user_item_proto', ds), 'C_basket_twins') or {}
    print(f"\n**C — basket twins (data only):** train positives with a word-identical basket twin "
          f"{f(c.get('frac_train_positives_with_basket_twin'))} of {c.get('n_train_positives')}; catalogue items "
          f"with a twin {f(c.get('frac_catalogue_items_with_twin'))}.\n")
    print('**D — cross-prototype cosine, row-max medians [p5, p95] (fixed by the scores; random reference '
          'isotropic in R^d, loose).**\n')
    print('| model | K_u / K_t / d | shared | user→item | item→user | within user | within item | random |')
    print('|---|---|---|---|---|---|---|---|')
    for m in ARMS:
        D = g(load(m, ds), 'D_cross_prototype_cosine') or {}

        def q(k):
            x = D.get(k) or {}
            return f"{f(x.get('p50'), 2)} [{f(x.get('p5'), 2)}, {f(x.get('p95'), 2)}]"
        print(f"| {SHORT[m]} | {D.get('K_u')} / {D.get('K_t')} / {D.get('d')} | {D.get('shared_table')} | "
              f"{q('user_proto_rowmax_vs_item_protos')} | {q('item_proto_rowmax_vs_user_protos')} | "
              f"{q('within_user_protos_rowmax')} | {q('within_item_protos_rowmax')} | {q('random_reference_user_rowmax')} |")
    print('\n**E — item side, across-item variance split words / ID / 2·cov (convention-free; ids arms only).**\n')
    print('| model | q_i | t̂ | B_item |')
    print('|---|---|---|---|')
    for m in ('feature_user_item_proto', 'feature_user_item_proto_sep', 'feature_item_proto'):
        E = g(load(m, ds), 'E_item_words_vs_id') or {}

        def e(k):
            x = E.get(k)
            return '—' if not x else f"{f(x['words'])} / {f(x['id'])} / {f(x['cross'])}"
        print(f"| {SHORT[m]} | {e('q')} | {e('t_hat')} | {e('B_item')} |")
    print()
