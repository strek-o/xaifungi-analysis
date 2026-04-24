import re
from pathlib import Path
import pandas as pd


def get_group(participant_id):
    parts = participant_id.split("_")
    if len(parts) >= 2 and parts[1] in ("DE", "SSH", "IT"):
        return parts[1]
    return None


def is_participant_id(speaker_id):
    if not isinstance(speaker_id, str):
        return False
    return bool(re.match(r"^[A-Z]{2}_(DE|IT|SSH)_\d", speaker_id.strip()))


def load_transcript(path):
    df = pd.read_csv(path)
    df["text"] = df["text"].fillna("").astype(str)
    df["participant_id"] = path.stem
    df["group"] = get_group(path.stem)
    return df


def load_all_transcripts(transcripts_dir="data/TRANSCRIPTS"):
    files = sorted(Path(transcripts_dir).glob("*.csv"))
    frames = [load_transcript(p) for p in files]
    return pd.concat(frames, ignore_index=True)


def filter_only_participants(df):
    mask = df["speaker_id"].apply(is_participant_id)
    return df[mask].copy()


def concat_text_per_participant(df):
    grouped = (
        df.groupby(["participant_id", "group"])["text"]
        .apply(lambda s: " ".join(s.tolist()))
        .reset_index()
    )
    return grouped
