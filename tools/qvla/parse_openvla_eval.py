"""Parse OpenVLA LIBERO text logs into Awesome-quant-vla summary files."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _latest_eval_log(log_dir: Path) -> Path:
    files = sorted(log_dir.glob("EVAL-*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        files = sorted(log_dir.rglob("EVAL-*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"no EVAL-*.txt found under {log_dir}")
    return files[0]


def _parse(text: str) -> tuple[int, int, float]:
    final = re.search(r"Total episodes:\s*(\d+).*?Total successes:\s*(\d+).*?Overall success rate:\s*([0-9.]+)", text, re.S)
    if final:
        episodes = int(final.group(1))
        successes = int(final.group(2))
        return episodes, successes, successes / max(episodes, 1)
    eps_matches = re.findall(r"# episodes completed so far:\s*(\d+)", text)
    suc_matches = re.findall(r"# successes:\s*(\d+)\s*\(([0-9.]+)%\)", text)
    if eps_matches and suc_matches:
        episodes = int(eps_matches[-1])
        successes = int(suc_matches[-1][0])
        return episodes, successes, successes / max(episodes, 1)
    rate_matches = re.findall(r"Current total success rate:\s*([0-9.]+)", text)
    if rate_matches:
        rate = float(rate_matches[-1])
        return 0, 0, rate
    raise ValueError("could not parse success metrics from eval log")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--task-suite-name", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--quantization", required=True)
    p.add_argument("--gpus", default="0")
    args = p.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_file = _latest_eval_log(Path(args.log_dir))
    text = log_file.read_text(encoding="utf-8", errors="replace")
    episodes, successes, rate = _parse(text)

    summary = {
        "task_suite": args.task_suite_name,
        "benchmark": args.benchmark,
        "quantization": args.quantization,
        "gpus": args.gpus,
        "total_episodes": episodes,
        "total_successes": successes,
        "total_success_rate": rate,
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
    (output_root / "merged_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{100 * rate:.1f} %")


if __name__ == "__main__":
    main()
