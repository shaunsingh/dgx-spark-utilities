#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import statistics
import time
from pathlib import Path
from datetime import datetime
from urllib.request import Request, urlopen


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


HEADERS = {"Content-Type": "application/json"}


def load_env_config() -> dict:
    """Load required runtime configuration from environment variables."""
    return {
        "backends": json.loads(require_env("BACKENDS_JSON")),
        "model_id": require_env("MODEL_ID"),
        "num_prompts": int(require_env("NUM_PROMPTS")),
        "concurrency": int(require_env("CONCURRENCY")),
        "dataset_path": require_env("DATASET_PATH"),
        "output_file": os.environ.get("OUTPUT_FILE") or "",
        "text_output_file": os.environ.get("TEXT_OUTPUT_FILE") or "",
        "max_output_tokens": int(os.environ.get("MAX_OUTPUT_TOKENS", "128")),
        "temperature": float(os.environ.get("TEMPERATURE", "0.0")),
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = pct * (len(values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    weight = pos - lo
    return values[lo] * (1 - weight) + values[hi] * weight


def download_dataset(url: str, dest: Path) -> None:
    """Download dataset to dest, overwriting if present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url)
    with urlopen(req, timeout=120) as resp, dest.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)


def load_sharegpt_prompts(path: str, limit: int) -> list[str]:
    dataset_path = Path(path).expanduser()

    def _load_json() -> list:
        with dataset_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    try:
        data = _load_json()
    except json.JSONDecodeError as exc:
        dataset_url = os.environ.get("DATASET_URL") or ""
        default_dataset = os.environ.get("DEFAULT_DATASET") or ""
        dataset_env = os.environ.get("DATASET_PATH") or ""
        is_default = (
            dataset_url
            and default_dataset
            and dataset_path.name == default_dataset
            and os.path.abspath(dataset_env) == os.path.abspath(str(dataset_path))
        )

        if is_default:
            print(
                f"[benchmark_runner] Dataset JSON at {dataset_path} is invalid; "
                f"re-downloading from {dataset_url}...",
                flush=True,
            )
            try:
                download_dataset(dataset_url, dataset_path)
                data = _load_json()
            except Exception as retry_exc:
                raise ValueError(
                    f"Dataset at {dataset_path} is invalid even after re-download from {dataset_url}"
                ) from retry_exc
        else:
            raise ValueError(
                f"Failed to parse dataset JSON at {dataset_path}. "
                "If using the default ShareGPT dataset, delete the file to force re-download "
                "or provide a valid dataset via --dataset."
            ) from exc
    prompts: list[str] = []
    for item in data:
        conv = item.get("conversations") or item.get("messages") or []
        text = None
        for msg in conv:
            role = msg.get("role") or msg.get("from")
            if role in ("user", "human", "prompter"):
                text = (msg.get("value") or msg.get("content") or "").strip()
                if text:
                    break
        if text:
            prompts.append(text)
        if len(prompts) >= limit:
            break
    if not prompts:
        raise ValueError("No usable prompts found in dataset")
    return prompts[:limit]


def check_health(url: str) -> bool:
    try:
        urlopen(f"{url}/health", timeout=3)
        return True
    except Exception:
        return False


def send_request(
    api_url: str, model: str, prompt: str, max_output_tokens: int, temperature: float
) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            "stream": False,
        }
    ).encode("utf-8")

    req = Request(f"{api_url}/v1/chat/completions", data=body, headers=HEADERS)
    start = time.perf_counter()
    try:
        with urlopen(req, timeout=60) as resp:
            payload = json.load(resp)
        latency_ms = (time.perf_counter() - start) * 1000
        usage = payload.get("usage") or {}
        return {
            "ok": True,
            "latency_ms": latency_ms,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "ok": False,
            "latency_ms": latency_ms,
            "error": str(exc),
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }


def run_backend(
    backend: dict,
    prompts: list[str],
    concurrency: int,
    model_id: str,
    max_output_tokens: int,
    temperature: float,
) -> dict:
    api_url = backend["url"].rstrip("/")
    if not check_health(api_url):
        return {
            "backend": backend["name"],
            "ok": False,
            "error": f"Health check failed at {api_url}/health",
        }

    latencies: list[float] = []
    prompt_tokens = 0
    completion_tokens = 0
    successes = 0
    start = time.perf_counter()

    workers = max(1, min(concurrency, len(prompts)))
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                send_request, api_url, model_id, prompt, max_output_tokens, temperature
            )
            for prompt in prompts
        ]
        for fut in cf.as_completed(futures):
            result = fut.result()
            latencies.append(result["latency_ms"])
            if result["ok"]:
                successes += 1
                prompt_tokens += result["prompt_tokens"]
                completion_tokens += result["completion_tokens"]

    duration = time.perf_counter() - start
    total_requests = len(prompts)
    failures = total_requests - successes
    req_throughput = successes / duration if duration > 0 else 0
    output_tps = completion_tokens / duration if duration > 0 else 0
    total_tps = (prompt_tokens + completion_tokens) / duration if duration > 0 else 0

    return {
        "backend": backend["name"],
        "url": api_url,
        "ok": True,
        "duration_s": round(duration, 2),
        "total_requests": total_requests,
        "successful_requests": successes,
        "failed_requests": failures,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "output_throughput_tps": round(output_tps, 2),
        "total_throughput_tps": round(total_tps, 2),
        "request_throughput_rps": round(req_throughput, 2),
        "mean_latency_ms": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        "p50_latency_ms": round(percentile(latencies, 0.5), 2),
        "p99_latency_ms": round(percentile(latencies, 0.99), 2),
    }


def build_text_report(summary: dict) -> str:
    """Render a simple nvidia-smi style text table for benchmark results."""
    inner_width = 95

    def line(char: str = "-") -> str:
        return f"+{char * inner_width}+"

    def pad(text: str) -> str:
        return f"| {text.ljust(inner_width - 2)} |"

    columns = [
        ("Backend", 18),
        ("Req/s", 10),
        ("Out tok/s", 14),
        ("p50 ms", 14),
        ("p99 ms", 14),
        ("Status", 8),
    ]

    def row(values: list[str]) -> str:
        padded = [val.ljust(width)[:width] for val, width in zip(values, (w for _, w in columns))]
        return "| " + " | ".join(padded) + " |"

    rows = [
        line("="),
        pad(f"Benchmark-SMI {datetime.now():%Y-%m-%d %H:%M:%S}"),
        line("="),
        pad(f"Model: {summary.get('model','')}"),
        pad(
            f"Prompts: {summary.get('num_prompts',0)}  "
            f"Concurrency: {summary.get('concurrency',0)}  "
            f"Max output tokens: {summary.get('max_output_tokens',0)}  "
            f"Temp: {summary.get('temperature',0.0)}"
        ),
        line("-"),
        row([title for title, _ in columns]),
        line("-"),
    ]

    for result in summary.get("results", []):
        if result.get("ok"):
            values = [
                result.get("backend", ""),
                f"{result.get('request_throughput_rps', 0):.2f}",
                f"{result.get('output_throughput_tps', 0):.2f}",
                f"{result.get('p50_latency_ms', 0):.2f}",
                f"{result.get('p99_latency_ms', 0):.2f}",
                "OK",
            ]
        else:
            values = [
                result.get("backend", ""),
                "FAIL",
                "n/a",
                "n/a",
                "n/a",
                "ERROR",
            ]
        rows.append(row(values))

    rows.append(line("-"))

    failures = [r for r in summary.get("results", []) if not r.get("ok")]
    if failures:
        rows.append(pad("Errors:"))
        for item in failures:
            err = (item.get("error") or "unknown").replace("\n", " ")[: inner_width - 6]
            rows.append(pad(f"- {item.get('backend','unknown')}: {err}"))
        rows.append(line("-"))

    return "\n".join(rows) + "\n"


def write_text_report(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = build_text_report(summary)
    path.write_text(report, encoding="utf-8")


def run_benchmarks_from_env() -> int:
    cfg = load_env_config()
    prompts = load_sharegpt_prompts(cfg["dataset_path"], cfg["num_prompts"])
    results = []

    for backend in cfg["backends"]:
        print(f"\n▶ Running {backend['name']} at {backend['url']} ...", flush=True)
        res = run_backend(
            backend,
            prompts,
            cfg["concurrency"],
            cfg["model_id"],
            cfg["max_output_tokens"],
            cfg["temperature"],
        )
        results.append(res)
        if res.get("ok"):
            print(
                f"  ✓ {backend['name']}: {res['request_throughput_rps']:.2f} req/s, "
                f"output {res['output_throughput_tps']:.2f} tok/s, "
                f"p50 {res['p50_latency_ms']:.1f} ms, p99 {res['p99_latency_ms']:.1f} ms",
                flush=True,
            )
        else:
            print(f"  ✗ {backend['name']}: {res.get('error','unknown error')}", flush=True)

    summary = {
        "model": cfg["model_id"],
        "num_prompts": cfg["num_prompts"],
        "concurrency": cfg["concurrency"],
        "max_output_tokens": cfg["max_output_tokens"],
        "temperature": cfg["temperature"],
        "results": results,
    }

    if cfg["output_file"]:
        path = Path(cfg["output_file"]).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nResults saved to {path}")

    if cfg["text_output_file"]:
        text_path = Path(cfg["text_output_file"]).expanduser().resolve()
        write_text_report(text_path, summary)
        print(f"Text report saved to {text_path}")

    failed = [r for r in results if not r.get("ok")]
    if failed:
        return 1
    return 0


def aggregate_results(dest: Path, parts: list[Path]) -> None:
    cfg = {
        "model": os.environ.get("MODEL_ID", ""),
        "num_prompts": int(os.environ.get("NUM_PROMPTS", "0") or 0),
        "concurrency": int(os.environ.get("CONCURRENCY", "0") or 0),
        "max_output_tokens": int(os.environ.get("MAX_OUTPUT_TOKENS", "0") or 0),
        "temperature": float(os.environ.get("TEMPERATURE", "0.0") or 0.0),
    }

    aggregate = {**cfg, "results": []}

    for path in parts:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        aggregate["results"].extend(data.get("results", []))

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"Results saved to {dest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark runner and aggregator")
    parser.add_argument(
        "--aggregate",
        nargs="+",
        metavar=("DEST", "FILES"),
        help="Aggregate multiple result files into DEST (DEST followed by input files)",
    )
    parser.add_argument(
        "--text-report",
        metavar="PATH",
        help="Write text report (nvidia-smi style) to PATH when aggregating",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.aggregate:
        if len(args.aggregate) < 2:
            raise SystemExit("ERROR: --aggregate requires DEST and at least one input file")
        dest = Path(args.aggregate[0]).expanduser().resolve()
        inputs = [Path(p) for p in args.aggregate[1:]]
        aggregate_results(dest, inputs)
        if args.text_report:
            write_text_report(
                Path(args.text_report).expanduser().resolve(),
                json.loads(dest.read_text(encoding="utf-8")),
            )
        return

    exit_code = run_benchmarks_from_env()
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()


