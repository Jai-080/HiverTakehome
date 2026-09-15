"""
Shared stratified sampling helper -- used by draft_reply.py's test harness
and (later) the Hour 6.5-8.5 golden-set sampler, so both use one correct
implementation of distributing a fixed total across intent groups.

Plain integer division (n_total // n_groups) silently drops the remainder
-- e.g. 15 // 8 == 1, producing only 8 rows instead of 15. divmod
distributes that remainder across the first `remainder` groups instead of
discarding it.
"""
import pandas as pd


def stratified_sample(df: pd.DataFrame, group_col: str, n_total: int, random_state: int = 42) -> pd.DataFrame:
    groups = sorted(df[group_col].unique())
    base, remainder = divmod(n_total, len(groups))

    parts = []
    for i, group_value in enumerate(groups):
        n_this = base + (1 if i < remainder else 0)
        group_df = df[df[group_col] == group_value]
        n_this = min(n_this, len(group_df))
        if n_this > 0:
            parts.append(group_df.sample(n=n_this, random_state=random_state))

    return pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
