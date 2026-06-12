"""
Download and unpack the XAI-FUNGI dataset from Zenodo.
"""

import io
import json
import urllib.request
import zipfile
from pathlib import Path

ZENODO_RECORD = "15222484"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"

ARCHIVES = {
    "TRANSCRIPTS.zip": "TRANSCRIPTS",
    "VISUALIZATION_MODIFICATIONS.zip": "VISUALIZATION_MODIFICATIONS",
}


def _list_files(timeout=30):
    """Return ``{filename: download_url}`` for every file in the Zenodo record."""
    with urllib.request.urlopen(ZENODO_API, timeout=timeout) as resp:
        meta = json.loads(resp.read().decode("utf-8"))
    return {f["key"]: f["links"]["self"] for f in meta.get("files", [])}


def _download(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def download_dataset(data_dir="data", overwrite=False, timeout=120, verbose=True):
    """Fetch the XAI-FUNGI record into ``data_dir`` and extract the archives."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    files = _list_files(timeout=timeout)
    actions = []

    for name, url in sorted(files.items()):
        if name in ARCHIVES:
            target_dir = data_path / ARCHIVES[name]
            if target_dir.exists() and any(target_dir.iterdir()) and not overwrite:
                actions.append(f"skip\t{ARCHIVES[name]}/ (already extracted)")
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            blob = _download(url, timeout=timeout)
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                zf.extractall(target_dir)
            actions.append(f"unzip\t{name} -> {ARCHIVES[name]}/")
        else:
            target = data_path / name
            if target.exists() and not overwrite:
                actions.append(f"skip\t{name} (already present)")
                continue
            target.write_bytes(_download(url, timeout=timeout))
            actions.append(f"save\t{name}")

    if verbose:
        for line in actions:
            print(line)
