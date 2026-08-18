#!/usr/bin/env python3
"""
Automated eval runner — calls the live GeneSight API with benchmark queries,
waits for results, computes metrics, and prints a report.

Usage:
    python eval/run_eval.py [--url http://localhost:8001] [--benchmark eval/benchmarks.json]

Prerequisites: backend must be running (python main.py or uvicorn main:app).
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Allow running from project root OR from eval/ subdir
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp not installed. Run: pip install aiohttp")
    sys.exit(1)

from eval.metrics import compute_all

DEFAULT_URL      = "http://localhost:8001"
DEFAULT_TIMEOUT  = 300   # seconds to wait for a job to complete
POLL_INTERVAL    = 5     # seconds between status polls


async def submit_query(session: aiohttp.ClientSession, base_url: str, query: str) -> str | None:
    """POST /query and return job_id, or None on failure."""
    try:
        async with session.post(
            f"{base_url}/query",
            json={"query": query},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            data = await resp.json()
            if resp.status == 200:
                job_id = data.get("job_id")
                cached = data.get("cached", False)
                if cached:
                    print(f"    ⚡ Cached result (job_id={job_id})")
                else:
                    print(f"    ↗ Submitted (job_id={job_id})")
                return job_id
            else:
                print(f"    ✗ HTTP {resp.status}: {data}")
                return None
    except Exception as e:
        print(f"    ✗ Submit error: {e}")
        return None


async def poll_until_done(
    session: aiohttp.ClientSession, base_url: str, job_id: str, timeout: int
) -> dict | None:
    """Poll /status/{job_id} until done, then fetch /result/{job_id}."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            async with session.get(
                f"{base_url}/status/{job_id}",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                status_data = await resp.json()
                status = status_data.get("status", "unknown")
                agent  = status_data.get("current_agent", "")
                print(f"    ⏳ {status}{f' [{agent}]' if agent else ''} …", end="\r")

                if status == "done":
                    print(f"    ✓ Done in {timeout - (deadline - time.monotonic()):.0f}s{' ' * 20}")
                    break
                elif status == "error":
                    print(f"\n    ✗ Pipeline error")
                    return None
        except Exception as e:
            print(f"    ✗ Poll error: {e}")

        await asyncio.sleep(POLL_INTERVAL)
    else:
        print(f"\n    ✗ Timed out after {timeout}s")
        return None

    # Fetch result
    try:
        async with session.get(
            f"{base_url}/result/{job_id}",
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                print(f"    ✗ Result fetch HTTP {resp.status}")
                return None
    except Exception as e:
        print(f"    ✗ Result fetch error: {e}")
        return None


def result_to_eval_dict(api_result: dict) -> dict:
    """Convert the /result API response into the shape eval/metrics.py expects."""
    hypotheses = [
        {
            "genes":      h.get("genes", []),
            "confidence": h.get("confidence", 0),
            "statement":  h.get("statement", ""),
            "title":      h.get("title", ""),
            "status":     h.get("status", ""),
        }
        for h in api_result.get("hypotheses", [])
    ]
    papers = [
        {
            "title":    p.get("title", ""),
            "abstract": p.get("abstract", ""),
            "year":     p.get("year", 0),
            "pmid":     p.get("pmid", ""),
        }
        for p in api_result.get("papers", [])
    ]
    return {
        "hypotheses": hypotheses,
        "papers":     papers,
        "db_data":    [],  # not exposed via API; hallucination guard runs server-side
    }


def print_report(results: list[dict]) -> None:
    """Print a formatted evaluation report to stdout."""
    print("\n" + "═" * 72)
    print("  GENESIGHT EVALUATION REPORT")
    print("═" * 72)

    total_passed = 0
    all_scores: list[float] = []

    for r in results:
        bm      = r["benchmark"]
        metrics = r["metrics"]
        status  = "✅ PASS" if metrics.get("passed") else "❌ FAIL"
        if metrics.get("passed"):
            total_passed += 1
        all_scores.append(metrics.get("overall_score", 0.0))

        print(f"\n  [{status}] {bm['id']}")
        print(f"  Query: {bm['query']}")
        if r.get("error"):
            print(f"  ERROR: {r['error']}")
            continue
        print(f"  ─────────────────────────────────────────────────")
        print(f"  gene_validity_rate           {metrics['gene_validity_rate']:.2%}")
        print(f"  hallucination_rate           {metrics['hallucination_rate']:.2%}")
        print(f"  faithfulness_score           {metrics['faithfulness_score']:.2%}")
        print(f"  paper_relevance_score        {metrics['paper_relevance_score']:.2%}")
        print(f"  hypothesis_disease_relevance {metrics['hypothesis_disease_relevance']:.2%}")
        print(f"  expected_gene_recall         {metrics['expected_gene_recall']:.2%}")
        print(f"  overall_score                {metrics['overall_score']:.2%}")
        checks = metrics.get("checks", {})
        for check, passed in checks.items():
            icon = "✓" if passed else "✗"
            print(f"  {icon} {check}")

    print("\n" + "═" * 72)
    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"  TOTAL: {total_passed}/{len(results)} benchmarks passed")
    print(f"  AVG OVERALL SCORE: {avg:.2%}")
    print("═" * 72 + "\n")


async def run_eval(base_url: str, benchmarks_path: str) -> None:
    benchmarks = json.loads(Path(benchmarks_path).read_text())
    print(f"Running {len(benchmarks)} benchmarks against {base_url}\n")

    report_rows: list[dict] = []

    async with aiohttp.ClientSession() as session:
        # Health check
        try:
            async with session.get(f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status != 200:
                    print(f"WARNING: API health check returned {r.status}")
        except Exception as e:
            print(f"WARNING: Cannot reach {base_url} ({e}). Is the server running?")

        for bm in benchmarks:
            print(f"\n{'─' * 60}")
            print(f"Benchmark: {bm['id']}")
            print(f"Query: {bm['query']}")

            job_id = await submit_query(session, base_url, bm["query"])
            if not job_id:
                report_rows.append({"benchmark": bm, "metrics": {}, "error": "submit failed"})
                continue

            api_result = await poll_until_done(session, base_url, job_id, DEFAULT_TIMEOUT)
            if not api_result:
                report_rows.append({"benchmark": bm, "metrics": {}, "error": "poll/fetch failed"})
                continue

            eval_dict = result_to_eval_dict(api_result)
            # Audit log is server-side; approximate from result structure
            audit_log: list[str] = []

            metrics = compute_all(eval_dict, bm, audit_log)
            report_rows.append({"benchmark": bm, "metrics": metrics})

    print_report(report_rows)

    # Save JSON report
    report_path = Path(__file__).parent / "last_eval_report.json"
    report_path.write_text(json.dumps(report_rows, indent=2))
    print(f"Full report saved to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GeneSight eval runner")
    parser.add_argument("--url",       default=DEFAULT_URL,               help="API base URL")
    parser.add_argument("--benchmark", default="eval/benchmarks.json",    help="Benchmarks JSON path")
    args = parser.parse_args()

    asyncio.run(run_eval(args.url, args.benchmark))


if __name__ == "__main__":
    main()
