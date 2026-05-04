"""Download the 2WikiMultiHopQA dataset.

Downloads from the official repository and extracts to data/raw/.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fair_kg_rag.utils.logging_utils import setup_logging

logger = setup_logging("download_data")

# 2WikiMultiHopQA download URL (from the official GitHub repo)
DATASET_URL = (
    "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1"
)
PARA_URL = (
    "https://www.dropbox.com/s/7ep3fjhllqsrd0r/para_with_hyperlink.zip?dl=1"
)

DEFAULT_RAW_DIR = Path("data/raw")


def download_file(url: str, dest: Path, description: str = "Downloading") -> None:
    """Download a file with progress bar.

    Args:
        url: URL to download from.
        dest: Destination file path.
        description: Progress bar description.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=description) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip file.

    Args:
        zip_path: Path to the zip file.
        dest_dir: Destination directory.
    """
    logger.info(f"Extracting {zip_path} to {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def download_2wikimultihopqa(raw_dir: Path = DEFAULT_RAW_DIR) -> None:
    """Download and extract the 2WikiMultiHopQA dataset.

    Args:
        raw_dir: Directory to store raw data files.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if (raw_dir / "dev.json").exists():
        logger.info("Dataset already downloaded. Skipping.")
        return

    # Download main dataset with entity IDs
    data_zip = raw_dir / "data_ids.zip"
    if not data_zip.exists():
        logger.info("Downloading 2WikiMultiHopQA dataset (with entity IDs)...")
        try:
            download_file(DATASET_URL, data_zip, "2WikiMultiHopQA data")
        except requests.RequestException as e:
            logger.error(f"Download failed: {e}")
            logger.info(
                "Please download manually from: "
                "https://github.com/Alab-NII/2wikimultihop"
            )
            return

    # Extract
    extract_zip(data_zip, raw_dir)

    # Download paragraphs with hyperlinks (optional, for richer KG)
    para_zip = raw_dir / "para_with_hyperlink.zip"
    if not para_zip.exists():
        logger.info("Downloading paragraph data with hyperlinks...")
        try:
            download_file(PARA_URL, para_zip, "Paragraph hyperlinks")
            extract_zip(para_zip, raw_dir)
        except requests.RequestException as e:
            logger.warning(f"Hyperlink data download failed (optional): {e}")

    # Verify files exist
    expected_files = ["train.json", "dev.json", "test.json"]
    for fname in expected_files:
        # Check in raw_dir and subdirectories
        found = list(raw_dir.rglob(fname))
        if found:
            # Move to raw_dir root if in a subdirectory
            if found[0].parent != raw_dir:
                found[0].rename(raw_dir / fname)
            logger.info(f"Found: {fname}")
        else:
            logger.warning(f"Missing: {fname}")

    logger.info(f"Dataset ready at {raw_dir}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download 2WikiMultiHopQA dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory to store raw data",
    )
    args = parser.parse_args()

    download_2wikimultihopqa(args.output_dir)


if __name__ == "__main__":
    main()
