"""
Metrics exporter sidecar — scrapes vLLM's native metrics,
adds ML-specific alerting metrics, and exposes them for Prometheus.
"""
import os
import time
import logging

import requests
from prometheus_client import (
    start_http_server,
    Gauge,
    Counter,
    Histogram,
    Info,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [metrics] %(message)s")
logger = logging.getLogger("metrics-exporter")

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen")
METRICS_PORT = int(os.environ.get("METRICS_PORT", "9090"))
INTERVAL = int(os.environ.get("COLLECTION_INTERVAL", "15"))

# ─── Custom metrics (on top of what vLLM already exposes) ────────────
MODEL_HEALTHY = Gauge(
    "ml_model_healthy",
    "Whether the model server is healthy and serving",
    ["model_name"],
)
HEALTH_CHECK_LATENCY = Histogram(
    "ml_health_check_latency_seconds",
    "Latency of health checks to the inference server",
    ["model_name"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
EXPORTER_ERRORS = Counter(
    "ml_exporter_errors_total",
    "Total errors in the metrics exporter sidecar",
    ["error_type"],
)
MODEL_INFO = Info(
    "ml_model",
    "Metadata about the served model",
)

# ─── Parse vLLM's /metrics for key signals ───────────────────────────
VLLM_GAUGE_PATTERNS = {
    "vllm:num_requests_running": Gauge(
        "ml_requests_running", "Number of requests currently being processed", ["model_name"]
    ),
    "vllm:num_requests_waiting": Gauge(
        "ml_requests_waiting", "Number of requests waiting in queue", ["model_name"]
    ),
    "vllm:gpu_cache_usage_perc": Gauge(
        "ml_gpu_kv_cache_usage", "GPU KV cache usage percentage", ["model_name"]
    ),
    "vllm:cpu_cache_usage_perc": Gauge(
        "ml_cpu_kv_cache_usage", "CPU KV cache usage percentage", ["model_name"]
    ),
}


def scrape_vllm_metrics():
    """Scrape vLLM's Prometheus endpoint and extract key metrics."""
    try:
        resp = requests.get(f"{VLLM_URL}/metrics", timeout=5)
        if resp.status_code != 200:
            return

        for line in resp.text.split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            for pattern, gauge in VLLM_GAUGE_PATTERNS.items():
                if line.startswith(pattern):
                    try:
                        value = float(line.split()[-1])
                        gauge.labels(model_name=MODEL_NAME).set(value)
                    except (ValueError, IndexError):
                        pass
    except Exception as e:
        EXPORTER_ERRORS.labels(error_type="scrape").inc()
        logger.debug(f"Scrape error: {e}")


def check_health():
    """Check inference server health and record latency."""
    start = time.monotonic()
    try:
        resp = requests.get(f"{VLLM_URL}/health", timeout=5)
        elapsed = time.monotonic() - start
        HEALTH_CHECK_LATENCY.labels(model_name=MODEL_NAME).observe(elapsed)

        is_healthy = resp.status_code == 200
        MODEL_HEALTHY.labels(model_name=MODEL_NAME).set(1 if is_healthy else 0)
        return is_healthy
    except requests.exceptions.ConnectionError:
        MODEL_HEALTHY.labels(model_name=MODEL_NAME).set(0)
        return False
    except Exception as e:
        EXPORTER_ERRORS.labels(error_type="health_check").inc()
        MODEL_HEALTHY.labels(model_name=MODEL_NAME).set(0)
        return False


def load_model_metadata():
    """Read metadata written by the model-loader sidecar."""
    try:
        import json
        with open("/models/metadata.json") as f:
            meta = json.load(f)
        MODEL_INFO.info({
            "model_id": meta.get("model_id", ""),
            "loaded_at": meta.get("loaded_at", ""),
            "loader_version": meta.get("loader_version", ""),
        })
    except Exception:
        pass


def main():
    start_http_server(METRICS_PORT)
    logger.info(f"Metrics exporter started on :{METRICS_PORT}")
    logger.info(f"Scraping vLLM at {VLLM_URL} every {INTERVAL}s")

    load_model_metadata()

    while True:
        check_health()
        scrape_vllm_metrics()
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
