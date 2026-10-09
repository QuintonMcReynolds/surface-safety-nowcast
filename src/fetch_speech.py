"""Download the ATC speech corpora from Hugging Face: UWB-ATCC (CC BY-NC-SA 4.0; train + test)
and the ATCO2 1-hour test set (other airports; used only for testing)."""
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    for repo in ["Jzuluaga/uwb_atcc", "Jzuluaga/atco2_corpus_1h"]:
        for f in list_repo_files(repo, repo_type="dataset"):
            if f.endswith(".parquet"):
                print(hf_hub_download(repo, f, repo_type="dataset", local_dir=ROOT / f"data/speech/{repo.split('/')[1]}"))
