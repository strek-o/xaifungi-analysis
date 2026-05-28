import pandas as pd

CERTAINTY = {
    "definitely certain": 2,
    "moderately certain": 1,
    "I can't assess": 0,
    "moderately uncertain": -1,
    "definitely uncertain": -2,
}


DECISION = {
    "eadible": "edible",
    "non-eadible": "poisonous",
    "jadalny": "edible",
    "trujący": "poisonous",
}


def load_problems_responses(path="data/PROBLEMS_RESPONSES.csv"):
    df = pd.read_csv(path)
    df["participant_certaintity_en"] = df["participant_certaintity_en"].str.strip()
    df["participant_decision_en"] = (
        df["participant_decision_en"].astype("string").str.strip()
    )
    df["certainty_score"] = df["participant_certaintity_en"].map(CERTAINTY)
    df["decision_canon"] = df["participant_decision_en"].map(DECISION)
    return df


def load_problems(path="data/PROBLEMS_en.csv"):
    df = pd.read_csv(path)
    df["model_class"] = df["model_class"].astype("string").str.strip()
    return df


def response_features(responses_df, problems_df):
    truth = problems_df.set_index("problem_id")["model_class"]

    df = responses_df.copy()
    df["truth"] = df["problem_id"].map(truth)
    df["correct"] = df["decision_canon"].eq(df["truth"])
    df.loc[df["decision_canon"].isna(), "correct"] = pd.NA

    grouped = df.groupby("participant_id")
    feats = pd.DataFrame(
        {
            "accuracy": grouped["correct"].mean(),
            "mean_certainty": grouped["certainty_score"].mean(),
            "frac_cant_assess": grouped["certainty_score"].apply(
                lambda s: (s == 0).mean()
            ),
        }
    )

    df["overconfident"] = df["correct"].eq(False) & (df["certainty_score"] >= 1)
    feats["overconfidence"] = grouped["overconfident"].sum().astype(int)

    return feats
