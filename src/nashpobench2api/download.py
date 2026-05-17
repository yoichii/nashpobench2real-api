import os
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    package_dir = Path(__file__).parent
    target_dir = package_dir / "apidata"

    snapshot_download(
        repo_id="yoichii/nashpobench2api",
        repo_type="dataset",
        allow_patterns=["*.pt", "*.json", "*.parquet"],
        local_dir=str(target_dir),
    )

if __name__ == "__main__":
    main()
