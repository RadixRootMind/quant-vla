#!/usr/bin/env python
"""
Visualize the effect of OHB scaling on o_proj activations (RMS).

This script runs the quantized GR00T policy on a Libero dataset, records per-layer
RMS of the o_proj outputs before scaling, and then applies the beta scalars from
the provided JSON to illustrate how OHB changes activation energy.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch
from tqdm import tqdm

from gr00t.atm import clear_atm_capture, ensure_dit_attention_patch, register_ohb_capture
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy

from examples.Libero.eval.utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
)
from libero.libero import benchmark

# Load Comic Sans font
# font_path = Path.home() / '.local/share/fonts/ComingSoon.ttf'
# if font_path.exists():
#     fm.fontManager.addfont(str(font_path))

_BASE_FONT = plt.rcParams.get("font.size", 10)
plt.rcParams.update({"font.size": _BASE_FONT + 5})


class RMSCollector:
    """Collects per-layer RMS values."""

    def __init__(self) -> None:
        self.sum: Dict[str, float] = {}
        self.count: Dict[str, int] = {}

    def __call__(self, layer_name: str, rms_tensor: torch.Tensor) -> None:
        value = float(rms_tensor.detach().to(torch.float32).mean().cpu())
        if layer_name not in self.sum:
            self.sum[layer_name] = value
            self.count[layer_name] = 1
        else:
            self.sum[layer_name] += value
            self.count[layer_name] += 1

    def finalize(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for name, total in self.sum.items():
            denom = max(self.count.get(name, 1), 1)
            result[name] = total / denom
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize OHB scaling on o_proj activations using beta JSON."
    )
    parser.add_argument("--checkpoint", required=True, help="Quantized model checkpoint or HF repo.")
    parser.add_argument(
        "--task-suite",
        default="libero_goal",
        help="Libero task suite name (default: libero_goal).",
    )
    parser.add_argument("--steps", type=int, default=64, help="Number of steps to replay.")
    parser.add_argument(
        "--max-trials-per-task",
        type=int,
        default=5,
        help="Max trials per task while gathering observations.",
    )
    parser.add_argument("--atm-json", required=True, help="Path to atm_alpha_beta JSON.")
    parser.add_argument(
        "--output-dir",
        default="ohb_viz",
        help="Directory to store generated figures (default: ohb_viz).",
    )
    parser.add_argument(
        "--max-plots",
        type=int,
        default=8,
        help="Maximum number of layer plots to generate (default: 8).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for running inference.",
    )
    parser.add_argument(
        "--duquant-packdir",
        default=None,
        help="Optional duquant pack directory override (sets GR00T_DUQUANT_PACKDIR).",
    )
    return parser.parse_args()


def _convert_observation(obs, language: str):
    xyz = obs["robot0_eef_pos"]
    rpy = quat2axisangle(obs["robot0_eef_quat"].copy())
    gripper = obs["robot0_gripper_qpos"].copy()
    img, wrist_img = get_libero_image(obs)
    converted = {
        "video.image": np.expand_dims(img, axis=0).astype(np.uint8),
        "video.wrist_image": np.expand_dims(wrist_img, axis=0).astype(np.uint8),
        "state.x": np.array([[xyz[0]]], dtype=np.float32),
        "state.y": np.array([[xyz[1]]], dtype=np.float32),
        "state.z": np.array([[xyz[2]]], dtype=np.float32),
        "state.roll": np.array([[rpy[0]]], dtype=np.float32),
        "state.pitch": np.array([[rpy[1]]], dtype=np.float32),
        "state.yaw": np.array([[rpy[2]]], dtype=np.float32),
        "state.gripper": np.expand_dims(gripper.astype(np.float32), axis=0),
        "annotation.human.action.task_description": [language],
    }
    return converted


def gather_observations(task_suite_name: str, max_steps: int, max_trials_per_task: int):
    torch.set_grad_enabled(False)
    dataset: List[Dict[str, np.ndarray]] = []

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        env, _ = get_libero_env(task, resolution=256)
        initial_states = task_suite.get_task_init_states(task_id)

        trials = min(len(initial_states), max_trials_per_task)

        for trial in range(trials):
            env.reset()
            obs = env.set_init_state(initial_states[trial])
            steps = 0
            while steps < max_steps:
                dataset.append(_convert_observation(obs, task.language))
                obs, _, done, _ = env.step(get_libero_dummy_action())
                steps += 1
                if done or len(dataset) >= max_steps:
                    break
            if len(dataset) >= max_steps:
                env.close()
                return dataset
        env.close()
    return dataset


def run_policy(policy: Gr00tPolicy, dataset: List[Dict[str, np.ndarray]], scope: str) -> Dict[str, float]:
    model = policy.model
    ensure_dit_attention_patch(model, scope=scope)
    collector = RMSCollector()
    register_ohb_capture(model, collector, scope=scope)
    torch.set_grad_enabled(False)
    for obs in tqdm(dataset, desc="Collecting o_proj RMS", leave=False):
        policy.get_action(obs)
    clear_atm_capture(model)
    return collector.finalize()


def safe_name(name: str) -> str:
    return name.replace(".", "__").replace("/", "_")


def _short_label(name: str) -> str:
    parts = name.split(".")
    for idx, part in enumerate(parts):
        if part == "transformer_blocks" and idx + 1 < len(parts):
            return f"Block{parts[idx + 1]}"
    return name.split(".")[-1]


def _block_index(name: str) -> int:
    parts = name.split(".")
    for idx, part in enumerate(parts):
        if part == "transformer_blocks" and idx + 1 < len(parts):
            try:
                return int(parts[idx + 1])
            except ValueError:
                return idx
    return 999


def select_layers(stats: Dict[str, float], beta_map: Dict[str, float], max_layers: int):
    scored = []
    for name, rms in stats.items():
        if name not in beta_map:
            continue
        beta = beta_map[name]
        diff = abs(beta * rms - rms)
        scored.append((diff, name))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [name for _, name in scored[:max_layers]]


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.duquant_packdir:
        os.environ["GR00T_DUQUANT_PACKDIR"] = args.duquant_packdir

    dataset = gather_observations(args.task_suite, args.steps, args.max_trials_per_task)
    if not dataset:
        raise RuntimeError("Failed to gather any observations.")

    original_atm_env = {k: os.environ.pop(k) for k in list(os.environ.keys()) if k.startswith("GR00T_ATM_")}

    data_config = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    policy_quant = Gr00tPolicy(
        model_path=args.checkpoint,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        denoising_steps=None,
        device=args.device,
    )

    print("[OHB-VIZ] Running quantized model without OHB to capture RMS...")
    rms_quant = run_policy(policy_quant, dataset, scope="dit")

    saved_duquant_env = {k: os.environ.pop(k) for k in list(os.environ.keys()) if k.startswith("GR00T_DUQUANT_")}
    policy_fp = Gr00tPolicy(
        model_path=args.checkpoint,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        denoising_steps=None,
        device=args.device,
    )
    rms_fp = run_policy(policy_fp, dataset, scope="dit")
    os.environ.update(saved_duquant_env)

    with open(args.atm_json, "r", encoding="utf-8") as f:
        alpha_json = json.load(f)

    beta_map = {}
    for name, entry in alpha_json.items():
        beta = entry.get("beta")
        if beta is not None:
            beta_map[name] = float(beta)

    if not beta_map:
        print("[OHB-VIZ] No beta values found in JSON.")
        return

    layers = select_layers(rms_quant, beta_map, args.max_plots)
    if not layers:
        print("[OHB-VIZ] No overlapping layers between stats and beta JSON.")
        return

    baseline_means = []
    quant_means = []
    ohb_means = []
    filtered_layers = []

    for layer in layers:
        if layer not in rms_fp or layer not in beta_map:
            continue
        beta = beta_map[layer]
        baseline_means.append(rms_fp[layer])
        quant_means.append(rms_quant[layer])
        ohb_means.append(rms_quant[layer] * beta)
        filtered_layers.append(layer)

    if not filtered_layers:
        print("[OHB-VIZ] No qualifying layers for plotting.")
        return

    indices = np.arange(len(filtered_layers))
    width = 0.25
    ordering = sorted(range(len(filtered_layers)), key=lambda i: _block_index(filtered_layers[i]))
    filtered_layers = [filtered_layers[i] for i in ordering]
    baseline_means = [baseline_means[i] for i in ordering]
    quant_means = [quant_means[i] for i in ordering]
    ohb_means = [ohb_means[i] for i in ordering]
    quant_ratios = []
    ohb_ratios = []
    print("\n[OHB-VIZ] Output RMS ratios (Quant/Teacher, OHB/Teacher):")
    for name, base_val, quant_val, ohb_val in zip(filtered_layers, baseline_means, quant_means, ohb_means):
        if base_val == 0:
            quant_ratio = float("inf")
            ohb_ratio = float("inf")
        else:
            quant_ratio = quant_val / base_val
            ohb_ratio = ohb_val / base_val
        quant_ratios.append(quant_ratio)
        ohb_ratios.append(ohb_ratio)
        print(f"  {_short_label(name)}: w/o OHB={quant_ratio:.3f}×, w/ OHB={ohb_ratio:.3f}×")
    if quant_ratios:
        quant_avg = sum(quant_ratios) / len(quant_ratios)
        ohb_avg = sum(ohb_ratios) / len(ohb_ratios)
        print(f"  Overall: w/o OHB={quant_avg:.3f}× (+{quant_avg-1:.3f} vs baseline), "
              f"w/ OHB={ohb_avg:.3f}× (+{ohb_avg-1:.3f} vs baseline)")

    plt.figure(figsize=(max(10, len(filtered_layers) * 0.6), 5), dpi=500)
    plt.bar(indices - width, baseline_means, width, label="Teacher(FP)", color="#63a4f8")
    plt.bar(indices, quant_means, width, label="QuantVLA without OHB", color="#ef8e8b")
    plt.bar(indices + width, ohb_means, width, label="QuantVLA", color="#f7cd55")
    ax = plt.gca()
    ax.grid(True, axis="y", alpha=0.4, linestyle="--")
    plt.ylabel("Attention Output RMS")
    plt.xticks(indices, [_short_label(name) for name in filtered_layers], rotation=45, ha="right")
    plt.title("OHB Effect Across Attention Blocks")
    plt.legend(loc="upper left")
    plt.tight_layout()
    out_path = out_dir / "ohb_summary.png"
    plt.savefig(out_path, dpi=500)
    plt.close()
    print(f"[OHB-VIZ] Saved {out_path}")

    os.environ.update(original_atm_env)


if __name__ == "__main__":
    main()
