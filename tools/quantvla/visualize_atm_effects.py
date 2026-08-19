#!/usr/bin/env python
"""
Visualize how ATM realigns attention logits temperature.

The script replays a Libero dataset through GR00T three times:
  1) Teacher (FP) to obtain the reference logits std per attention block.
  2) Quantized policy without ATM to measure the drift introduced by DuQuant.
  3) Quantized policy with ATM enabled (using the provided alpha JSON).

Collected statistics are summarized in bar charts showing how ATM pulls the
logits std back toward the teacher reference.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from gr00t.atm import (
    clear_atm_capture,
    ensure_dit_attention_patch,
    register_atm_capture,
)
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

# Optional Comic Sans font (keeps visuals consistent with OHB plots if font is present)
# comic_font = Path.home() / ".local/share/fonts/ComingSoon.ttf"
# if comic_font.exists():
#     fm.fontManager.addfont(str(comic_font))

_BASE_FONT = plt.rcParams.get("font.size", 10)
plt.rcParams.update({"font.size": _BASE_FONT + 5,})


class StdCollector:
    """Accumulates per-layer, per-head logits std."""

    def __init__(self) -> None:
        self.sum: Dict[str, torch.Tensor] = {}
        self.count: Dict[str, int] = {}

    def __call__(self, layer_name: str, std_tensor: torch.Tensor) -> None:
        tensor = std_tensor.detach().to(torch.float32)
        if tensor.ndim == 4:
            tensor = tensor.mean(dim=(2, 3))
        elif tensor.ndim == 3:
            tensor = tensor.mean(dim=2)
        tensor = tensor.mean(dim=0)
        if layer_name not in self.sum:
            self.sum[layer_name] = tensor.clone()
            self.count[layer_name] = 1
        else:
            self.sum[layer_name] += tensor
            self.count[layer_name] += 1

    def finalize(self) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}
        for name, total in self.sum.items():
            result[name] = total / max(self.count.get(name, 1), 1)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ATM temperature calibration effects.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint or HF repo.")
    parser.add_argument("--atm-json", required=True, help="Path to atm_alpha_beta JSON.")
    parser.add_argument(
        "--task-suite",
        default="libero_goal",
        help="Libero task suite to replay (default: libero_goal).",
    )
    parser.add_argument("--steps", type=int, default=64, help="Maximum replay steps.")
    parser.add_argument(
        "--max-trials-per-task",
        type=int,
        default=5,
        help="Trials per Libero task when gathering rollouts.",
    )
    parser.add_argument(
        "--output-dir",
        default="atm_viz",
        help="Directory for figures (default: atm_viz).",
    )
    parser.add_argument(
        "--max-plots",
        type=int,
        default=8,
        help="Maximum number of attention layers to visualize.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--duquant-packdir",
        default=None,
        help="Optional DuQuant pack dir (sets GR00T_DUQUANT_PACKDIR).",
    )
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
    dataset: List[Dict[str, np.ndarray]] = []
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        env, _ = get_libero_env(task, resolution=256)
        init_states = task_suite.get_task_init_states(task_id)
        trials = min(len(init_states), max_trials_per_task)
        for trial in range(trials):
            env.reset()
            obs = env.set_init_state(init_states[trial])
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


def run_with_capture(policy: Gr00tPolicy, dataset, scope: str) -> Dict[str, float]:
    model = policy.model
    ensure_dit_attention_patch(model, scope=scope)
    collector = StdCollector()
    register_atm_capture(model, collector, scope=scope)
    torch.set_grad_enabled(False)
    for obs in tqdm(dataset, desc="Capturing logits std", leave=False):
        policy.get_action(obs)
    clear_atm_capture(model)
    return collector.finalize()


def _block_index(layer_name: str) -> int:
    parts = layer_name.split(".")
    for idx, part in enumerate(parts):
        if part == "transformer_blocks" and idx + 1 < len(parts):
            try:
                return int(parts[idx + 1])
            except ValueError:
                return idx
    return 999


def _short_label(layer_name: str) -> str:
    parts = layer_name.split(".")
    if "transformer_blocks" in parts:
        idx = parts.index("transformer_blocks")
        if idx + 1 < len(parts):
            return f"Block{parts[idx + 1]}"
    return parts[-1]


def select_layers(baseline: Dict[str, torch.Tensor], quant: Dict[str, torch.Tensor], max_plots: int) -> List[str]:
    scored = []
    for name, base in baseline.items():
        if name not in quant:
            continue
        diff_tensor = torch.abs(quant[name] - base)
        diff = float(diff_tensor.mean().item())
        scored.append((diff, name))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [name for _, name in scored[:max_plots]]


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.duquant_packdir:
        os.environ["GR00T_DUQUANT_PACKDIR"] = args.duquant_packdir

    dataset = gather_observations(args.task_suite, args.steps, args.max_trials_per_task)
    if not dataset:
        raise RuntimeError("Failed to gather any observations.")

    saved_atm_env = {k: os.environ.pop(k) for k in list(os.environ.keys()) if k.startswith("GR00T_ATM_")}

    data_cfg = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")

    def make_policy():
        return Gr00tPolicy(
            model_path=args.checkpoint,
            modality_config=data_cfg.modality_config(),
            modality_transform=data_cfg.transform(),
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            denoising_steps=None,
            device=args.device,
        )

    print("[ATM-VIZ] Capturing Teacher (FP) logits std...")
    saved_duquant_env = {k: os.environ.pop(k) for k in list(os.environ.keys()) if k.startswith("GR00T_DUQUANT_")}
    policy_fp = make_policy()
    std_fp = run_with_capture(policy_fp, dataset, scope="dit")
    os.environ.update(saved_duquant_env)

    print("[ATM-VIZ] Capturing quantized logits std (no ATM)...")
    policy_quant = make_policy()
    std_quant = run_with_capture(policy_quant, dataset, scope="dit")

    print("[ATM-VIZ] Applying ATM scalars from JSON...")
    # Clean ATM env vars
    for key in list(os.environ.keys()):
        if key.startswith("GR00T_ATM_"):
            os.environ.pop(key)
    os.environ.update(saved_atm_env)

    with open(args.atm_json, "r", encoding="utf-8") as f:
        alpha_json = json.load(f)

    def apply_alpha(layer: str, tensor: torch.Tensor) -> torch.Tensor:
        entry = alpha_json.get(layer)
        if not entry:
            return tensor
        alpha_vals = entry.get("all")
        if not alpha_vals:
            return tensor
        alpha = torch.tensor(alpha_vals, dtype=tensor.dtype, device=tensor.device)
        if alpha.shape[0] != tensor.shape[0]:
            return tensor
        return tensor * alpha

    std_atm = {layer: apply_alpha(layer, stats) for layer, stats in std_quant.items()}

    layers = select_layers(std_fp, std_quant, args.max_plots)
    if not layers:
        print("[ATM-VIZ] No overlapping layers for visualization.")
        return

    valid_layers = []
    teacher_vals = []
    quant_vals = []
    atm_vals = []
    for name in layers:
        if name not in std_quant or name not in std_atm:
            continue
        valid_layers.append(name)
        teacher_vals.append(std_fp.get(name, torch.tensor(0.0)).mean().item())
        quant_vals.append(std_quant.get(name, torch.tensor(0.0)).mean().item())
        atm_vals.append(std_atm.get(name, torch.tensor(0.0)).mean().item())

    if not valid_layers:
        print("[ATM-VIZ] Layer stats incomplete; nothing to plot.")
        return

    ordering = sorted(range(len(valid_layers)), key=lambda i: _block_index(valid_layers[i]))
    labels = [_short_label(valid_layers[i]) for i in ordering]
    teacher_vals = [teacher_vals[i] for i in ordering]
    quant_vals = [quant_vals[i] for i in ordering]
    atm_vals = [atm_vals[i] for i in ordering]
    quant_ratios = []
    atm_ratios = []
    print("\n[ATM-VIZ] Logits std ratios (Quant/Teacher, ATM/Teacher):")
    for label, t_val, q_val, a_val in zip(labels, teacher_vals, quant_vals, atm_vals):
        if t_val == 0:
            quant_ratio = float("inf")
            atm_ratio = float("inf")
        else:
            quant_ratio = q_val / t_val
            atm_ratio = a_val / t_val
        quant_ratios.append(quant_ratio)
        atm_ratios.append(atm_ratio)
        print(f"  {label}: w/o ATM={quant_ratio:.3f}×, w/ ATM={atm_ratio:.3f}×")
    if quant_ratios:
        quant_avg = sum(quant_ratios) / len(quant_ratios)
        atm_avg = sum(atm_ratios) / len(atm_ratios)
        print(f"  Overall: w/o ATM={quant_avg:.3f}× (+{quant_avg-1:.3f} vs baseline), "
              f"w/ ATM={atm_avg:.3f}× (+{atm_avg-1:.3f} vs baseline)")

    indices = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(max(10, len(labels) * 0.6), 5), dpi=220)
    plt.bar(indices - width, teacher_vals, width, label="Teacher (FP)", color="#63a4f8")
    plt.bar(indices, quant_vals, width, label="QuantVLA without ATM", color="#ef8e8b")
    plt.bar(indices + width, atm_vals, width, label="QuantVLA", color="#f7cd55")
    ax = plt.gca()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.ylabel("Mean Logits Std")
    plt.xticks(indices, labels, rotation=45, ha="right")
    plt.title("ATM Effect Across Attention Blocks")
    plt.legend(loc="upper left")
    plt.tight_layout()
    out_path = out_dir / "atm_summary.png"
    plt.savefig(out_path, dpi=500)
    plt.close()
    print(f"[ATM-VIZ] Saved {out_path}")


if __name__ == "__main__":
    main()
