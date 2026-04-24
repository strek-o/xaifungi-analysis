import re
import pandas as pd


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
        rows.append({
            "participant_id": r["participant_id"],
            "n_words": n_words,
            "ttr": len(unique_words) / n_words if n_words > 0 else 0,
            "avg_word_len": sum(len(w) for w in words) / n_words if n_words > 0 else 0,
            "questions_per_1k": n_questions * per_1k,
            "uncertainty_per_1k": uncertainty_count * per_1k,
            "ignorance_per_1k": ignorance_count * per_1k,
        })

    return pd.DataFrame(rows).set_index("participant_id")
