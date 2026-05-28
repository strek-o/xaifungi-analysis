import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

VIZ_COLUMNS = [
    "descriptive statistics",
    "counterfactual analysis",
    "anchor",
    "LIME",
    "waterfall",
    "bee swarm",
    "text slides",
    "box plots",
    "histograms",
    "distribution of features",
]


PAIR_MERGE_MAP = {
    "PK_DE_11": "PK_DE_11-12",
    "PK_DE_12": "PK_DE_11-12",
}


def load_maxqda_summary(path="data/MAXQDA_SUMMARY.csv", merge_pairs=True):
    df = pd.read_csv(path)
    if merge_pairs:
        df["participant_id"] = df["participant_id"].replace(PAIR_MERGE_MAP)
        df = df.groupby(["participant_id", "code"], as_index=False)[VIZ_COLUMNS].sum()
    return df


def code_count_features(maxqda_df, normalize=True):
    long_df = maxqda_df.copy()
    long_df["total"] = long_df[VIZ_COLUMNS].sum(axis=1)
    wide = long_df.pivot_table(
        index="participant_id",
        columns="code",
        values="total",
        aggfunc="sum",
        fill_value=0,
    ).astype(float)
    if normalize:
        row_sums = wide.sum(axis=1)
        wide = wide.div(row_sums.replace(0, np.nan), axis=0).fillna(0)
    wide.columns.name = None
    return wide


def viz_use_features(maxqda_df, normalize=True):
    grouped = maxqda_df.groupby("participant_id")[VIZ_COLUMNS].sum()
    if normalize:
        row_sums = grouped.sum(axis=1)
        grouped = grouped.div(row_sums.replace(0, np.nan), axis=0).fillna(0)
    return grouped


def maxqda_svd_features(maxqda_df, n_components=15, normalize=True, random_state=42):
    wide = code_count_features(maxqda_df, normalize=normalize)
    max_comp = max(1, min(wide.shape) - 1)
    n_components = min(n_components, max_comp)
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_red = svd.fit_transform(wide.values)
    cols = [f"maxqda_svd_{i + 1}" for i in range(n_components)]
    return pd.DataFrame(X_red, columns=cols, index=wide.index)


def top_codes_per_group(maxqda_df, groups, top_k=10, normalize=True):
    wide = code_count_features(maxqda_df, normalize=normalize)
    grp = pd.Series(groups).reindex(wide.index)
    grouped = wide.groupby(grp).mean()
    return {
        g: grouped.loc[g].sort_values(ascending=False).head(top_k)
        for g in grouped.index
    }
