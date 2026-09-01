"""Parse StarVLA LIBERO eval logs into quant-vla summary files."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _parse(text: str) -> tuple[int, int, float, list[dict]]:
    eps_matches = re.findall(r"# episodes completed so far:\s*(\d+)", text)
    suc_matches = re.findall(r"# successes:\s*(\d+)\s*\(([0-9.]+)%\)", text)
    task_rates = [float(x) for x in re.findall(r"Current task success rate:\s*([0-9.]+)", text)]
    total_rates = [float(x) for x in re.findall(r"Current total success rate:\s*([0-9.]+)", text)]

    total_success_rate = None
    m = re.search(r"Total success rate:\s*([0-9.]+)", text)
    if m:
        total_success_rate = float(m.group(1))
    elif total_rates:
        total_success_rate = total_rates[-1]

    total_episodes = 0
    total_successes = 0
    m = re.search(r"Total episodes:\s*(\d+)", text)
    if m:
        total_episodes = int(m.group(1))
    elif eps_matches:
        total_episodes = int(eps_matches[-1])

    m = re.search(r"Total successes:\s*(\d+)", text)
    if m:
        total_successes = int(m.group(1))
    elif suc_matches:
        total_successes = int(suc_matches[-1][0])

    if total_success_rate is None and total_episodes:
        total_success_rate = total_successes / max(total_episodes, 1)
    if total_success_rate is None:
        raise ValueError("could not parse StarVLA success rate from eval log")
    if total_episodes and not total_successes:
        total_successes = round(total_success_rate * total_episodes)

    task_results = [
        {"task_id": idx, "success_rate": rate}
        for idx, rate in enumerate(task_rates)
    ]
    return total_episodes, total_successes, total_success_rate, task_results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log-file", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--task-suite-name", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--quantization", required=True)
    p.add_argument("--gpus", default="0")
    args = p.parse_args()

    log_file = Path(args.log_file)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    text = log_file.read_text(encoding="utf-8", errors="replace")
    episodes, successes, rate, task_results = _parse(text)

    summary = {
        "task_suite": args.task_suite_name,
        "benchmark": args.benchmark,
        "quantization": args.quantization,
        "gpus": args.gpus,
        "total_episodes": episodes,
        "total_successes": successes,
        "total_success_rate": rate,
        "task_results": task_results,
        "source_log": str(log_file),
    }
    (output_root / "merged_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        f"# LIBERO {args.benchmark} Summary",
        "",
        f"- Task suite: {args.task_suite_name}",
        f"- GPUs: {args.gpus}",
        f"- Quantization: {args.quantization}",
        f"- Benchmark: {args.benchmark}",
    ]
    if episodes:
        md.append(f"- Episodes: {successes}/{episodes} successes ({100 * rate:.1f}%)")
    else:
        md.append(f"- Overall success rate: {100 * rate:.1f}%")
    md.append(f"- Source log: {log_file}")
    if task_results:
        md += ["", "| task_id | success_rate |", "|---:|---:|"]
        for row in task_results:
            md.append(f"| {row['task_id']} | {100 * row['success_rate']:.1f}% |")
    (output_root / "merged_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{100 * rate:.1f} %")


if __name__ == "__main__":
    main()
