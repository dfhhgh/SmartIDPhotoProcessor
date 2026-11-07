"""Phase E — Load & Performance Benchmark for SmartIDPhotoProcessor."""

from __future__ import annotations

import json
import os
import subprocess
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

IMAGE_NAME = "smartidphotoprocessor"
CONTAINER_PREFIX = "smartidphoto-bench"
WARMUP = 5
MEASURED = 30
CONCURRENCIES = [1, 2, 4, 8]
WORKER_CONFIGS = [1, 2]
TIMEOUT = 120
REQ_TIMEOUT = 60
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "phase_e" / "results"
BENCHMARK_IMAGES = [
    "test_images/experiments/09e1122b113e4b2ce108f45d622915ae.jpg",
    "test_images/experiments/c54fd7f979120afea790e3390b6e1a66.jpg",
    "test_images/experiments/glasses.jpg",
    "test_images/experiments/hijab2.jpg",
]


@dataclass
class RequestResult:
    success: bool
    status_code: int
    latency_ms: float
    error: str | None = None
    is_valid: bool | None = None
    total_metrics: int | None = None
    gender: str | None = None


@dataclass
class BenchResult:
    timestamp: str
    workers: int
    concurrency: int
    warmup_count: int
    measured_count: int
    total_duration_s: float
    latencies_ms: list[float] = field(default_factory=list)
    successes: int = 0
    errors: int = 0
    error_details: list[str] = field(default_factory=list)
    http_2xx: int = 0
    http_4xx: int = 0
    http_5xx: int = 0
    conn_failures: int = 0
    mem_samples_mb: list[float] = field(default_factory=list)
    cpu_samples_pct: list[float] = field(default_factory=list)
    correctness: list[dict] = field(default_factory=list)

def docker_run(workers: int, port: int, ml_concurrency: int = 2) -> str:
    name = f"{CONTAINER_PREFIX}-w{workers}-e{ml_concurrency}"
    cmd = [
        "docker", "run", "-d", "--name", name,
        "-p", f"{port}:8000",
        "-e", "PYTHONUNBUFFERED=1",
        "-e", f"ML_MAX_CONCURRENCY={ml_concurrency}",
        IMAGE_NAME,
        "uvicorn", "smartidphotoprocessor.api.app:app",
        "--host", "0.0.0.0", "--port", "8000", "--workers", str(workers),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"docker run failed: {r.stderr.strip()}")
    return r.stdout.strip()[:12]


def docker_stop(container: str) -> None:
    subprocess.run(["docker", "stop", container], capture_output=True, timeout=30)
    subprocess.run(["docker", "rm", container], capture_output=True, timeout=30)


def docker_stats(container: str) -> dict[str, float]:
    cmd = [
        "docker", "stats", container, "--no-stream", "--format",
        "{{.CPUPerc}}|{{.MemUsage}}|{{.MemPerc}}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        return {"cpu_pct": 0.0, "mem_mb": 0.0}
    parts = r.stdout.strip().split("|")
    if len(parts) < 2:
        return {"cpu_pct": 0.0, "mem_mb": 0.0}

    def pc(s):
        return float(s.strip().rstrip("%"))

    def pm(s):
        # MemUsage format: "98.46MiB / 7.609GiB" — take the first part (used)
        s = s.strip().split("/")[0].strip() if "/" in s else s.strip()
        if "GiB" in s:
            return float(s.replace("GiB", "").strip()) * 1024
        if "MiB" in s:
            return float(s.replace("MiB", "").strip())
        if "KiB" in s:
            return float(s.replace("KiB", "").strip()) / 1024
        return 0.0

    return {"cpu_pct": pc(parts[0]), "mem_mb": pm(parts[1])}


def http_get(url: str, timeout: int = 10) -> tuple[int, str]:
    try:
        with urlopen(Request(url), timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except HTTPError as e:
        return e.code, str(e)
    except (URLError, OSError) as e:
        return 0, str(e)


def http_post(url: str, image_path: str, timeout: int = REQ_TIMEOUT) -> tuple[int, str]:
    boundary = "----BenchBound"
    with open(image_path, "rb") as f:
        img = f.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(image_path)}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + img + f"\r\n--{boundary}--\r\n".encode()
    req = Request(url, data=body,
                  headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                  method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except HTTPError as e:
        return e.code, e.read().decode() if hasattr(e, "read") else str(e)
    except (URLError, OSError) as e:
        return 0, str(e)


def wait_ready(port: int, timeout: int = TIMEOUT) -> float:
    url = f"http://localhost:{port}/api/v1/ready"
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        s, b = http_get(url, timeout=5)
        if s == 200 and '"ready"' in b:
            return time.perf_counter() - start
        time.sleep(1)
    raise TimeoutError(f"Not ready after {timeout}s")

def single_req(port: int, image_path: str) -> RequestResult:
    t0 = time.perf_counter()
    s, b = http_post(f"http://localhost:{port}/api/v1/validate", image_path)
    lat = (time.perf_counter() - t0) * 1000
    if s == 0:
        return RequestResult(success=False, status_code=0, latency_ms=lat, error=b)
    try:
        d = json.loads(b)
        return RequestResult(
            success=200 <= s < 300, status_code=s, latency_ms=lat,
            is_valid=d.get("is_valid"),
            total_metrics=d.get("validation", {}).get("total_metrics"),
            gender=d.get("face", {}).get("gender"),
        )
    except json.JSONDecodeError:
        return RequestResult(success=False, status_code=s, latency_ms=lat, error="bad JSON")


def run_bench(port: int, concurrency: int, image_path: str,
              container: str | None = None) -> BenchResult:
    res = BenchResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        workers=0, concurrency=concurrency,
        warmup_count=WARMUP, measured_count=MEASURED, total_duration_s=0,
    )
    # Warmup
    for _ in range(WARMUP):
        single_req(port, image_path)

    if container:
        st = docker_stats(container)
        res.mem_samples_mb.append(st["mem_mb"])
        res.cpu_samples_pct.append(st["cpu_pct"])

    all_r: list[RequestResult] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(single_req, port, image_path) for _ in range(MEASURED)]
        for f in as_completed(futs):
            try:
                all_r.append(f.result())
            except Exception as e:
                all_r.append(RequestResult(False, 0, 0, str(e)))
            if container and len(all_r) % 5 == 0:
                st = docker_stats(container)
                res.mem_samples_mb.append(st["mem_mb"])
                res.cpu_samples_pct.append(st["cpu_pct"])

    res.total_duration_s = round(time.perf_counter() - t0, 3)
    if container:
        st = docker_stats(container)
        res.mem_samples_mb.append(st["mem_mb"])
        res.cpu_samples_pct.append(st["cpu_pct"])

    for r in all_r:
        if r.success:
            res.successes += 1
            res.latencies_ms.append(r.latency_ms)
            res.http_2xx += 1
            if r.is_valid is not None:
                res.correctness.append({"is_valid": r.is_valid, "metrics": r.total_metrics, "gender": r.gender})
        else:
            res.errors += 1
            if r.status_code >= 500:
                res.http_5xx += 1
            elif r.status_code >= 400:
                res.http_4xx += 1
            elif r.status_code == 0:
                res.conn_failures += 1
            if r.error:
                res.error_details.append(r.error[:200])
    return res


def lat_stats(lats: list[float]) -> dict[str, float]:
    if not lats:
        return {"min": 0, "mean": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    s = sorted(lats)
    n = len(s)
    return {
        "min": round(s[0], 1), "mean": round(statistics.mean(s), 1),
        "p50": round(statistics.median(s), 1),
        "p90": round(s[min(int(n * 0.9), n - 1)], 1),
        "p95": round(s[min(int(n * 0.95), n - 1)], 1),
        "p99": round(s[min(int(n * 0.99), n - 1)], 1),
        "max": round(s[-1], 1),
    }


def measure_cold_start(port: int, workers: int, ml_concurrency: int = 2) -> dict[str, Any]:
    name = f"{CONTAINER_PREFIX}-w{workers}-e{ml_concurrency}"
    docker_stop(name)
    t0 = time.perf_counter()
    docker_run(workers, port, ml_concurrency)
    t_health = None
    for _ in range(TIMEOUT):
        s, _ = http_get(f"http://localhost:{port}/api/v1/health", timeout=5)
        if s == 200:
            t_health = round(time.perf_counter() - t0, 3)
            break
        time.sleep(0.5)
    t_ready = None
    try:
        rt = wait_ready(port)
        t_ready = round(t_health + rt, 3) if t_health else None
    except TimeoutError:
        pass
    mem = docker_stats(name)
    return {"health_s": t_health, "ready_s": t_ready, "baseline_mem_mb": round(mem["mem_mb"], 1)}

def run_worker_config(workers: int, base_port: int, ml_concurrency: int = 2) -> dict[str, Any]:
    port = base_port + workers * 100 + ml_concurrency
    name = f"{CONTAINER_PREFIX}-w{workers}-e{ml_concurrency}"
    print(f"\n{'='*60}")
    print(f"  WORKERS: {workers}  EXECUTOR: {ml_concurrency}")
    print(f"{'='*60}")

    cold = measure_cold_start(port, workers, ml_concurrency)
    print(f"  Cold start health={cold['health_s']}s ready={cold['ready_s']}s mem={cold['baseline_mem_mb']}MB")

    image = BENCHMARK_IMAGES[0]
    results = []
    peak_mem = cold["baseline_mem_mb"]

    for c in CONCURRENCIES:
        print(f"\n  Concurrency={c} ... ", end="", flush=True)
        r = run_bench(port, c, image, container=name)
        r.workers = workers
        ls = lat_stats(r.latencies_ms)
        rps = round(r.successes / r.total_duration_s, 2) if r.total_duration_s > 0 else 0
        pm = max(r.mem_samples_mb) if r.mem_samples_mb else 0
        if pm > peak_mem:
            peak_mem = pm
        avg_cpu = round(statistics.mean(r.cpu_samples_pct), 1) if r.cpu_samples_pct else 0
        print(f"P50={ls['p50']}ms P95={ls['p95']}ms P99={ls['p99']}ms "
              f"RPS={rps} err={r.errors} mem={pm}MB cpu={avg_cpu}%")
        results.append({
            "concurrency": c,
            "latency": ls,
            "rps": rps,
            "successes": r.successes,
            "errors": r.errors,
            "http_2xx": r.http_2xx,
            "http_4xx": r.http_4xx,
            "http_5xx": r.http_5xx,
            "conn_failures": r.conn_failures,
            "duration_s": r.total_duration_s,
            "peak_mem_mb": pm,
            "avg_cpu_pct": avg_cpu,
            "correctness_sample": r.correctness[:3],
            "error_details": r.error_details[:5],
            "all_latencies_ms": r.latencies_ms,
        })

    # Get final memory
    final_mem = docker_stats(name)
    return {
        "workers": workers,
        "ml_concurrency": ml_concurrency,
        "cold_start": cold,
        "concurrency_results": results,
        "peak_mem_mb": round(peak_mem, 1),
        "final_mem_mb": round(final_mem["mem_mb"], 1),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Benchmark matrix: (workers, ml_concurrency)
    configs = [
        (1, 1), (1, 2), (1, 4),
        (2, 2),
    ]

    for w, e in configs:
        try:
            wr = run_worker_config(w, 9100, e)
            all_results.append(wr)
        except Exception as ex:
            print(f"\n  FAILED for {w} workers, {e} executor: {ex}")
            all_results.append({"workers": w, "ml_concurrency": e, "error": str(ex)})
        finally:
            docker_stop(f"{CONTAINER_PREFIX}-w{w}-e{e}")

    # Save raw results
    out = RESULTS_DIR / f"benchmark_{timestamp}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nRaw results saved to {out}")

    # Print summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for wr in all_results:
        if "error" in wr:
            print(f"\n  {wr['workers']}w/{wr.get('ml_concurrency', '?')}e: FAILED - {wr['error']}")
            continue
        print(f"\n  {wr['workers']} workers / {wr.get('ml_concurrency', '?')} executor:")
        print(f"    Cold start: health={wr['cold_start']['health_s']}s ready={wr['cold_start']['ready_s']}s")
        print(f"    Baseline mem: {wr['cold_start']['baseline_mem_mb']}MB")
        print(f"    Peak mem: {wr['peak_mem_mb']}MB")
        for cr in wr["concurrency_results"]:
            print(f"    C={cr['concurrency']}: P50={cr['latency']['p50']}ms "
                  f"P95={cr['latency']['p95']}ms P99={cr['latency']['p99']}ms "
                  f"RPS={cr['rps']} err={cr['errors']} mem={cr['peak_mem_mb']}MB")


if __name__ == "__main__":
    main()
