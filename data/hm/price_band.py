"""dc01 P7 — H&M price-band feature (decile of per-article mean transaction price).

The categorical H&M ``item_features.csv`` carries no price, so dc01 §3.5's 6th field (a price
decile) is computed here from the raw transactions. This module is the pure, testable core; wiring
it into the full ``hm_splitter.py`` regeneration is GATED (the raw transactions join is heavy and a
fresh split would not be byte-identical to the canonical one — see the implementation plan P0/P7).
``price_band`` is OPTIONAL for the base model, which runs on the 5 categorical fields already present.

Usage (pure function):
    from data.hm.price_band import compute_price_band
    bands = compute_price_band(transactions_df, item_ids_df)   # DataFrame[item_id, item, price_band]
"""
import pandas as pd


def compute_price_band(transactions: pd.DataFrame, item_ids: pd.DataFrame, n_bins: int = 10,
                       price_col: str = 'price', article_col: str = 'article_id') -> pd.DataFrame:
    """Per-article mean price → equal-frequency decile, aligned to the 0-based ``item_id``.

    Equal-frequency binning uses a first-tie rank before ``qcut`` so the result is always the clean
    integer set ``0..n_bins-1`` (avoids ``qcut``'s "non-unique bin edges" failure when many articles
    share a price — common in fast fashion).

    :param transactions: raw transactions with at least ``article_col`` and ``price_col``.
    :param item_ids: the split's ``item_ids.csv`` (columns ``item_id`` 0-based and ``item`` = article_id).
    :param n_bins: number of price bands (10 = deciles).
    :return: DataFrame ``[item_id, item, price_band]`` sorted by ``item_id`` (length == n_items).
    """
    mean_price = transactions.groupby(article_col)[price_col].mean()

    out = item_ids.copy()
    out['_mean_price'] = out['item'].astype(str).map(mean_price)
    ranks = out['_mean_price'].rank(method='first')
    out['price_band'] = pd.qcut(ranks, n_bins, labels=False).astype('Int64')

    out = out.sort_values('item_id').reset_index(drop=True)
    return out[['item_id', 'item', 'price_band']]
