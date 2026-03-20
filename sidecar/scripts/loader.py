"""
Model loader sidecar — downloads HuggingFace model to shared volume,
writes a .ready sentinel file, then polls for updates.
"""
import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [loader] %(message)s")
logger = logging.getLogger("model-loader")

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/qwen")
READY_FILE = os.environ.get("READY_FILE", "/models/.ready")
METADATA_FILE = os.environ.get("METADATA_FILE", "/models/metadata.json")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "300"))


def download_model():
    """Download model from HuggingFace Hub to local directory."""
    from huggingface_hub import snapshot_download

    logger.info(f"Downloading {MODEL_ID} to {MODEL_DIR}")
    start = time.monotonic()

    path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
    )

    elapsed = time.monotonic() - start
    logger.info(f"Download complete in {elapsed:.1f}s -> {path}")
    return path


def write_metadata():
    """Write metadata about the loaded model for other sidecars to read."""
    meta = {
        "model_id": MODEL_ID,
        "model_dir": MODEL_DIR,
        "loaded_at": datetime.now(timezone.utc).isoformat(),
        "loader_version": "v1.0.0",
    }
    with open(METADATA_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata written to {METADATA_FILE}")


def signal_ready():
    """Write the ready file that the inference server's readiness probe checks."""
    Path(READY_FILE).write_text("ready")
    logger.info(f"Ready file written: {READY_FILE}")


def check_for_update():
    """
    Check if the remote model has been updated.
    Returns True if a new version is available.
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.model_info(MODEL_ID)
        remote_sha = info.sha

        local_meta_path = Path(METADATA_FILE)
        if local_meta_path.exists():
            with open(local_meta_path) as f:
                local_meta = json.load(f)
            if local_meta.get("remote_sha") == remote_sha:
                return False

        logger.info(f"New version detected: {remote_sha}")
        return True
    except Exception as e:
        logger.warning(f"Update check failed: {e}")
        return False


def main():
    # Initial download
    download_model()
    write_metadata()
    signal_ready()

    # Poll loop
    logger.info(f"Entering poll loop (interval: {POLL_INTERVAL}s)")
    while True:
        time.sleep(POLL_INTERVAL)
        try:
            if check_for_update():
                logger.info("Downloading updated model...")
                download_model()
                write_metadata()
                logger.info("Model hot-reloaded")
        except Exception as e:
            logger.error(f"Poll cycle error: {e}")


if __name__ == "__main__":
    main()
