"""
Feature extraction for the XAI-FUNGI clustering study.
"""

import re
import numpy as np
import pandas as pd
import urllib.request
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

STOPWORDS_URL = (
    "https://raw.githubusercontent.com/stopwords-iso/stopwords-pl/master/stopwords-pl.txt"
)
STOPWORDS_CACHE = Path(__file__).resolve().parent.parent / "data" / "stopwords_pl.txt"

DOMAIN_WORDS = ["anonimizacja", "ns"]

UNCERTAINTY_WORDS = [
    "chyba", "może", "raczej", "pewnie", "niby", "wydaje", "myślę",
    "sądzę", "uważam", "przypuszczam", "podejrzewam", "domyślam", "zdaje",
    "prawdopodobnie", "zapewne", "możliwe", "być może",
    "jakby", "jakoś", "niekoniecznie",
]

IGNORANCE_WORDS = [
    "nie wiem", "nie rozumiem", "nie potrafię", "nie umiem",
    "nie mam pojęcia", "trudno mi", "nie jestem pewien", "nie jestem pewna",
    "ciężko powiedzieć", "ciężko mi",
]


def get_stopwords(url=STOPWORDS_URL, cache_path=STOPWORDS_CACHE, timeout=15):
    """Return Polish stop_words from an external source."""
    if cache_path is not None and Path(cache_path).exists():
        words = Path(cache_path).read_text(encoding="utf-8").split()
    else:
        raw = urllib.request.urlopen(url, timeout=timeout).read().decode("utf-8")
        words = [w.strip() for w in raw.splitlines() if w.strip()]
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_text("\n".join(words), encoding="utf-8")

    return sorted(set(words) | set(DOMAIN_WORDS))


def count_phrases(text, phrases):
    text_lower = text.lower()
    return sum(text_lower.count(p) for p in phrases)


def simple_features(df_text):
    """Selected linguistic statistics, one row per participant."""
    rows = []
    for _, r in df_text.iterrows():
        text = r["text"]
        words = re.findall(r"\w+", text.lower())
        n_words = len(words)
        unique_words = set(words)

        n_questions = text.count("?")
        uncertainty_count = count_phrases(text, UNCERTAINTY_WORDS)
        ignorance_count = count_phrases(text, IGNORANCE_WORDS)

        per_1k = 1000 / n_words if n_words > 0 else 0
        rows.append(
            {
                "participant_id": r["participant_id"],
                "n_words": n_words,
                "ttr": len(unique_words) / n_words if n_words > 0 else 0,
                "avg_word_len": (
                    sum(len(w) for w in words) / n_words if n_words > 0 else 0
                ),
                "questions_per_1k": n_questions * per_1k,
                "uncertainty_per_1k": uncertainty_count * per_1k,
                "ignorance_per_1k": ignorance_count * per_1k,
            }
        )

    return pd.DataFrame(rows).set_index("participant_id")


_PAREN_PATTERN = re.compile(r"\([^)]*\)")


def clean_transcript_text(text):
    if not isinstance(text, str):
        return ""
    return _PAREN_PATTERN.sub(" ", text).lower()


def build_tfidf(text_df, max_features=2000, ngram_range=(1, 1), min_df=2, max_df=0.95, stop_words=None):
    """Fit a TF-IDF vectorizer over one concatenated document per participant."""
    if stop_words is None:
        stop_words = get_stopwords()
    vectorizer = TfidfVectorizer(
        preprocessor=clean_transcript_text,
        lowercase=False,
        token_pattern=r"(?u)\b[^\W\d_]{2,}\b",
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words=list(stop_words),
    )
    X = vectorizer.fit_transform(text_df["text"].tolist())
    return X, vectorizer


def tfidf_svd_features(text_df, n_components=15, random_state=42, **tfidf_kwargs):
    """TF-IDF reduced to a dense ``n_components``- dim latent space (LSA)."""
    X, _ = build_tfidf(text_df, **tfidf_kwargs)
    max_comp = max(1, min(X.shape) - 1)
    n_components = min(n_components, max_comp)
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_reduced = svd.fit_transform(X)

    cols = [f"tfidf_svd_{i + 1}" for i in range(n_components)]
    out = pd.DataFrame(X_reduced, columns=cols, index=text_df["participant_id"].values)
    out.index.name = "participant_id"
    return out


def top_terms_per_cluster(text_df, clusters, top_k=10, **tfidf_kwargs):
    """Most characteristic TF-IDF terms per cluster."""
    X, vectorizer = build_tfidf(text_df, **tfidf_kwargs)
    terms = np.array(vectorizer.get_feature_names_out())
    X_dense = X.toarray()
    clusters = np.asarray(clusters)

    out = {}
    for cluster_id in sorted(set(clusters.tolist())):
        mask = clusters == cluster_id
        mean_weights = X_dense[mask].mean(axis=0)
        top_idx = np.argsort(mean_weights)[::-1][:top_k]
        out[cluster_id] = list(
            zip(terms[top_idx].tolist(), mean_weights[top_idx].tolist())
        )
    return out


def _slide_theme_map(slides_path="data/SLIDES.csv"):
    slides = pd.read_csv(slides_path)
    slides = slides.dropna(subset=["maxqda_theme"])
    return slides.drop_duplicates("slide_id").set_index("slide_id")["maxqda_theme"].to_dict()


def slide_topic_features(df_part, slides_path="data/SLIDES.csv", normalize=True):
    """Words spoken per explanation theme, one row per participant."""
    theme_map = _slide_theme_map(slides_path)
    df = df_part.copy()
    df["n_words"] = df["text"].fillna("").str.split().str.len()
    df["topic"] = df["slide_id"].map(theme_map).fillna("none")

    wide = (
        df.groupby(["participant_id", "topic"])["n_words"]
        .sum()
        .unstack(fill_value=0)
        .add_prefix("topic_")
    )
    if normalize:
        row_sums = wide.sum(axis=1)
        wide = wide.div(row_sums.replace(0, np.nan), axis=0).fillna(0)
    wide.columns.name = None
    return wide


def topic_none_share(df_part, groups=None):
    """Share of each participant's words left untagged (``topic_none``)."""
    df = df_part.copy()
    df["n_words"] = df["text"].fillna("").str.split().str.len()
    df["tagged"] = df["slide_id"].notna()

    per_part = df.groupby("participant_id").apply(
        lambda x: pd.Series(
            {
                "words_total": int(x["n_words"].sum()),
                "words_tagged": int(x.loc[x["tagged"], "n_words"].sum()),
            }
        ),
        include_groups=False,
    )
    per_part["share_none"] = (
            1 - per_part["words_tagged"] / per_part["words_total"].replace(0, np.nan)
    ).fillna(0)

    if groups is None:
        return per_part

    grp = pd.Series(groups).reindex(per_part.index)
    agg = per_part.groupby(grp).agg(
        words_total=("words_total", "sum"),
        words_tagged=("words_tagged", "sum"),
    )
    agg["share_none"] = (1 - agg["words_tagged"] / agg["words_total"]).round(3)
    return agg
