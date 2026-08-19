#!/usr/bin/env python
"""
Visualize logits std for Baseline (FP), Post-DuQuant, and Post-ATM paths.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from gr00t.atm import clear_atm_capture, ensure_dit_attention_patch, register_atm_capture
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


class StdCollector:
    def __init__(self) -> None:
        self.sum: Dict[str, torch.Tensor] = {}
        self.count: Dict[str, int] = {}

    def __call__(self, layer_name: str, std_tensor: torch.Tensor) -> None:
        if std_tensor.ndim == 4:
            std_tensor = std_tensor.mean(dim=(2, 3))
        elif std_tensor.ndim == 3:
            std_tensor = std_tensor.mean(dim=2)
        std_tensor = std_tensor.mean(dim=0)
        std_cpu = std_tensor.detach().to(torch.float32)
        if layer_name not in self.sum:
            self.sum[layer_name] = std_cpu.clone()
            self.count[layer_name] = 1
        else:
            self.sum[layer_name] += std_cpu
            self.count[layer_name] += 1

    def finalize(self) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}
        for name, tensor in self.sum.items():
            denom = max(self.count.get(name, 1), 1)
            result[name] = tensor / denom
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize logits std before/after ATM.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--atm-json", required=True)
    parser.add_argument("--task-suite", default="libero_goal")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--max-trials-per-task", type=int, default=5)
    parser.add_argument("--max-plots", type=int, default=8)
    parser.add_argument("--output-dir", default="atm_logits_viz")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--duquant-packdir", default=None)
    return parser.parse_args()


def _convert_observation(obs, language: str):
    xyz = obs["robot0_eef_pos"]
    rpy = quat2axisangle(obs["robot0_eef_quat"].copy())
    gripper = obs["robot0_gripper_qpos"].copy()
    img, wrist_img = get_libero_image(obs)
    return {
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


def run_policy(policy: Gr00tPolicy, dataset, scope: str) -> Dict[str, torch.Tensor]:
    model = policy.model
    ensure_dit_attention_patch(model, scope=scope)
    collector = StdCollector()
    register_atm_capture(model, collector, scope=scope)
    torch.set_grad_enabled(False)
    for obs in tqdm(dataset, desc="Collecting logits std", leave=False):
        policy.get_action(obs)
    clear_atm_capture(model)
    return collector.finalize()


def _short_label(name: str) -> str:
    parts = name.split(".")
    for idx, part in enumerate(parts):
        if part == "transformer_blocks" and idx + 1 < len(parts):
            return f"blk{parts[idx + 1]}"
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


def main():
    args = parse_args()
    if args.duquant_packdir:
        os.environ["GR00T_DUQUANT_PACKDIR"] = args.duquant_packdir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = gather_observations(args.task_suite, args.steps, args.max_trials_per_task)
    if not dataset:
        raise RuntimeError("Failed to gather observations.")

    saved_duquant = {k: os.environ.pop(k) for k in list(os.environ.keys()) if k.startswith("GR00T_DUQUANT_")}
    saved_atm = {k: os.environ.pop(k) for k in list(os.environ.keys()) if k.startswith("GR00T_ATM_")}

    data_cfg = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    print("[ATM-VIZ] Baseline (FP) run...")
    policy_fp = Gr00tPolicy(
        model_path=args.checkpoint,
        modality_config=data_cfg.modality_config(),
        modality_transform=data_cfg.transform(),
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        denoising_steps=None,
        device=args.device,
    )
    baseline_stats = run_policy(policy_fp, dataset, scope="dit")

    print("[ATM-VIZ] Post-DuQuant run...")
    os.environ.update(saved_duquant)
    for k in list(os.environ.keys()):
        if k.startswith("GR00T_ATM_"):
            os.environ.pop(k)
    policy_quant = Gr00tPolicy(
        model_path=args.checkpoint,
        modality_config=data_cfg.modality_config(),
        modality_transform=data_cfg.transform(),
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        denoising_steps=None,
        device=args.device,
    )
    quant_stats = run_policy(policy_quant, dataset, scope="dit")

    print("[ATM-VIZ] Post-ATM run...")
    os.environ["GR00T_ATM_ENABLE"] = "1"
    os.environ["GR00T_ATM_ALPHA_PATH"] = args.atm_json
    os.environ.setdefault("GR00T_ATM_SCOPE", "dit")
    policy_atm = Gr00tPolicy(
        model_path=args.checkpoint,
        modality_config=data_cfg.modality_config(),
        modality_transform=data_cfg.transform(),
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        denoising_steps=None,
        device=args.device,
    )
    atm_stats = run_policy(policy_atm, dataset, scope="dit")

    os.environ.update(saved_atm)

    layers = sorted(baseline_stats.keys(), key=_block_index)
    if args.max_plots > 0:
        layers = layers[: args.max_plots]

    baseline_means = []
    quant_means = []
    atm_means = []
    filtered_layers = []
    for layer in layers:
        if layer not in quant_stats or layer not in atm_stats:
            continue
        baseline_means.append(baseline_stats[layer].mean().item())
        quant_means.append(quant_stats[layer].mean().item())
        atm_means.append(atm_stats[layer].mean().item())
        filtered_layers.append(layer)

    if not filtered_layers:
        print("[ATM-VIZ] No overlapping layers to plot.")
        return

    idx = np.arange(len(filtered_layers))
    width = 0.25
    plt.figure(figsize=(max(10, len(filtered_layers) * 0.6), 5), dpi=220)
    plt.bar(idx - width, baseline_means, width, label="Baseline (FP)", color="#4C72B0")
    plt.bar(idx, quant_means, width, label="Post-DuQuant", color="#DD8452")
    plt.bar(idx + width, atm_means, width, label="Post-ATM", color="#55A868")
    plt.ylabel("Mean Logits Std (pre-softmax)")
    plt.xticks(idx, [_short_label(n) for n in filtered_layers], rotation=45, ha="right")
    plt.title("ATM: Logits Temperature Across Attention Blocks")
    plt.legend(loc="upper left")
    plt.tight_layout()
    out_path = out_dir / "atm_logits_summary.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"[ATM-VIZ] Saved {out_path}")


if __name__ == "__main__":
    main()
