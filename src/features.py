import re
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

STOP_WORDS = [
    "ach", "acz", "aczkolwiek", "aj", "albo", "ale", "ależ", "ani", "aż",
    "bardziej", "bardzo", "bez", "bo", "bowiem", "by", "byli", "bym", "był",
    "była", "było", "były", "być", "będzie", "będą",
    "ci", "cię", "ciebie", "co", "cokolwiek", "coś", "czasami", "czasem",
    "czemu", "czy", "czyli",
    "daleko", "dla", "dlaczego", "dlatego", "do", "dobrze", "dokąd", "dość",
    "dużo", "dwa", "dwaj", "dwie", "dwoje", "dziś", "dzisiaj",
    "gdy", "gdyby", "gdyż", "gdzie", "gdziekolwiek", "gdzieś", "go",
    "ich", "ile", "im", "inna", "inne", "inny", "innych", "iż",
    "ja", "ją", "jak", "jakaś", "jakby", "jaki", "jakichś", "jakie", "jakiś",
    "jakiż", "jakkolwiek", "jako", "je", "jeden", "jedna", "jedno", "jednak",
    "jednakże", "jego", "jej", "jemu", "jest", "jestem", "jeszcze", "jeśli",
    "jeżeli", "już",
    "każdy", "kiedy", "kilka", "kimś", "kto", "ktokolwiek", "ktoś", "która",
    "które", "którego", "której", "który", "których", "którym", "którzy", "ku",
    "lat", "lecz", "lub",
    "ma", "mają", "mało", "mam", "mi", "mimo", "między", "mnie", "mną", "moi",
    "moim", "moja", "moje", "mój", "mu", "musi", "my",
    "na", "nad", "nam", "nami", "nas", "nasi", "nasz", "nasza", "nasze",
    "naszego", "naszych", "natomiast", "natychmiast", "nawet", "nią", "nic",
    "nich", "nie", "niech", "niego", "niej", "niemu", "nigdy", "nim", "nimi",
    "niż", "no",
    "obok", "od", "około", "on", "ona", "one", "oni", "ono", "oraz", "oto",
    "owszem",
    "pan", "pana", "pani", "po", "pod", "podczas", "pomimo", "ponad",
    "ponieważ", "powinien", "powinna", "powinni", "powinno", "poza", "prawie",
    "przecież", "przed", "przede", "przedtem", "przez", "przy",
    "raz", "razie", "również",
    "skąd", "sobie", "sobą", "sposób", "swoje", "są",
    "ta", "tak", "taka", "taki", "takie", "także", "tam", "te", "tego", "tej",
    "temu", "ten", "teraz", "też", "to", "tobą", "tobie", "toteż", "trzeba",
    "tu", "tutaj", "twoi", "twoim", "twoja", "twoje", "twym", "twój", "ty",
    "tych", "tylko", "tym",
    "wam", "wami", "was", "wasz", "wasza", "wasze", "we", "według", "wiele",
    "wielu", "więc", "więcej", "wszyscy", "wszystkich", "wszystkie",
    "wszystkim", "wszystko", "wtedy", "wy",
    "za", "zapewne", "zawsze", "ze", "znowu", "znów", "został",
    "żaden", "żadna", "żadne", "żadnych", "że", "żeby",
    "anonimizacja", "ns",
]


UNCERTAINTY_WORDS = [
    "chyba", "może", "raczej", "pewnie", "niby",
    "wydaje", "myślę", "sądzę", "uważam", "przypuszczam", "podejrzewam",
    "domyślam", "zdaje",
    "prawdopodobnie", "zapewne", "możliwe", "być może",
    "jakby", "jakoś", "niekoniecznie",
]


IGNORANCE_WORDS = [
    "nie wiem", "nie rozumiem", "nie potrafię", "nie umiem",
    "nie mam pojęcia", "trudno mi", "nie jestem pewien", "nie jestem pewna",
    "ciężko powiedzieć", "ciężko mi",
]


def count_phrases(text, phrases):
    text_lower = text.lower()
    return sum(text_lower.count(p) for p in phrases)


def simple_features(df_text):
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


def build_tfidf(
    text_df,
    max_features=2000,
    ngram_range=(1, 1),
    min_df=2,
    max_df=0.95,
    stop_words=None,
):
    if stop_words is None:
        stop_words = STOP_WORDS
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
