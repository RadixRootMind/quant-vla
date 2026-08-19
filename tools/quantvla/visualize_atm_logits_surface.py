#!/usr/bin/env python3
"""
3D surface visualization of logits before/after ATM.

For a specified attention layer, capture the raw logits tensor (QK^T/sqrt(d))
for three paths:
  1) Baseline (full-precision, no DuQuant)
  2) Post-DuQuant (DuQuant enabled, ATM disabled)
  3) Post-ATM (DuQuant + ATM)
and plot them as 3D surfaces (token vs channel vs logits value).
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import tqdm

from gr00t.atm import (
    clear_atm_capture,
    ensure_dit_attention_patch,
    register_atm_logits_capture,
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


_BASE_FONT = plt.rcParams.get("font.size", 10)
plt.rcParams.update({"font.size": _BASE_FONT + 4})


class LogitsCollector:
    def __init__(self, target_layer: str, max_samples: int = 1):
        self.target_layer = target_layer
        self.max_samples = max_samples
        self.storage: List[torch.Tensor] = []

    def callback(self, layer_name: str, logits: torch.Tensor) -> None:
        if layer_name != self.target_layer:
            return
        if len(self.storage) >= self.max_samples:
            return
        self.storage.append(logits.detach().cpu())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize logits surfaces before/after ATM.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint or HF repo.")
    parser.add_argument("--atm-json", required=True, help="Path to atm_alpha_beta JSON.")
    parser.add_argument("--layer-name", required=True, help="Full module name of the attention layer.")
    parser.add_argument("--task-suite", default="libero_goal")
    parser.add_argument("--steps", type=int, default=16, help="Number of Libero steps to replay.")
    parser.add_argument("--max-trials-per-task", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--duquant-packdir", default=None)
    parser.add_argument("--head-index", type=int, default=0, help="Attention head index to visualize.")
    parser.add_argument("--batch-index", type=int, default=0, help="Batch index to visualize.")
    parser.add_argument("--token-range", default=None, help="Optional token slice start:end")
    parser.add_argument("--channel-range", default=None, help="Optional channel slice start:end")
    parser.add_argument("--stats-output", default=None, help="Optional JSON to store summary stats")
    parser.add_argument("--output", default="atm_logits_surface.png")
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


def run_policy_with_capture(
    policy: Gr00tPolicy,
    dataset,
    scope: str,
    collector: LogitsCollector,
) -> torch.Tensor:
    model = policy.model
    ensure_dit_attention_patch(model, scope=scope)
    register_atm_logits_capture(model, collector.callback, scope=scope)
    torch.set_grad_enabled(False)
    for obs in tqdm(dataset, desc="Capturing logits", leave=False):
        policy.get_action(obs)
        if collector.storage:
            break
    clear_atm_capture(model)
    if not collector.storage:
        raise RuntimeError(f"No logits captured for layer {collector.target_layer}")
    return collector.storage[0]


def _parse_range(txt: str | None, upper: int) -> slice:
    if not txt:
        return slice(None)
    start, end = txt.split(":")
    start = int(start) if start else None
    end = int(end) if end else None
    return slice(start, end)


def extract_surface(
    logits_tensor: torch.Tensor,
    batch_idx: int,
    head_idx: int,
    token_slice: slice,
    channel_slice: slice,
) -> np.ndarray:
    if logits_tensor.ndim != 4:
        raise ValueError(f"Expected logits tensor with shape [B, H, T, T], got {logits_tensor.shape}")
    b = logits_tensor.shape[0]
    h = logits_tensor.shape[1]
    if batch_idx >= b or head_idx >= h:
        raise IndexError(f"batch_idx/head_idx out of range: tensor shape {logits_tensor.shape}")
    surface = logits_tensor[batch_idx, head_idx].to(torch.float32)
    surface = surface[token_slice, channel_slice]
    return surface.numpy()


def plot_surface(ax, data: np.ndarray, title: str, *, zlim=None, cmap="coolwarm") -> None:
    tokens = np.arange(data.shape[0])
    channels = np.arange(data.shape[1])
    T, C = np.meshgrid(tokens, channels, indexing="ij")
    surf = ax.plot_surface(C, T, data, cmap=cmap, linewidth=0, antialiased=False, alpha=0.95)
    if zlim is not None:
        ax.set_zlim(*zlim)
        norm = plt.Normalize(vmin=zlim[0], vmax=zlim[1])
        surf.set_norm(norm)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Token")
    ax.set_zlabel("Logits")
    ax.set_title(title)
    ax.view_init(elev=30, azim=-120)
    plt.colorbar(surf, ax=ax, shrink=0.45, pad=0.05)


def main():
    args = parse_args()
    if args.duquant_packdir:
        os.environ["GR00T_DUQUANT_PACKDIR"] = args.duquant_packdir

    dataset = gather_observations(args.task_suite, args.steps, args.max_trials_per_task)
    if not dataset:
        raise RuntimeError("Failed to gather observations for visualization.")

    saved_duquant = {k: os.environ.pop(k) for k in list(os.environ.keys()) if k.startswith("GR00T_DUQUANT_")}
    saved_atm = {k: os.environ.pop(k) for k in list(os.environ.keys()) if k.startswith("GR00T_ATM_")}

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

    # Baseline (FP)
    print("[ATM-SURF] Capturing baseline (FP) logits...")
    baseline_collector = LogitsCollector(args.layer_name)
    baseline_policy = make_policy()
    baseline_logits = run_policy_with_capture(baseline_policy, dataset, scope="dit", collector=baseline_collector)

    # Post-DuQuant
    print("[ATM-SURF] Capturing quantized logits...")
    os.environ.update(saved_duquant)
    for key in list(os.environ.keys()):
        if key.startswith("GR00T_ATM_"):
            os.environ.pop(key)
    quant_collector = LogitsCollector(args.layer_name)
    quant_policy = make_policy()
    quant_logits = run_policy_with_capture(quant_policy, dataset, scope="dit", collector=quant_collector)

    # Post-ATM
    print("[ATM-SURF] Capturing ATM logits...")
    os.environ["GR00T_ATM_ENABLE"] = "1"
    os.environ["GR00T_ATM_ALPHA_PATH"] = args.atm_json
    os.environ.setdefault("GR00T_ATM_SCOPE", "dit")
    atm_collector = LogitsCollector(args.layer_name)
    atm_policy = make_policy()
    atm_logits = run_policy_with_capture(atm_policy, dataset, scope="dit", collector=atm_collector)

    # Restore environment
    for key in list(os.environ.keys()):
        if key.startswith("GR00T_DUQUANT_"):
            os.environ.pop(key)
        if key.startswith("GR00T_ATM_"):
            os.environ.pop(key)
    os.environ.update(saved_duquant)
    os.environ.update(saved_atm)

    token_slice = _parse_range(args.token_range, baseline_logits.shape[-1])
    channel_slice = _parse_range(args.channel_range, baseline_logits.shape[-1])

    baseline_surface = extract_surface(baseline_logits, args.batch_index, args.head_index, token_slice, channel_slice)
    quant_surface = extract_surface(quant_logits, args.batch_index, args.head_index, token_slice, channel_slice)
    atm_surface = extract_surface(atm_logits, args.batch_index, args.head_index, token_slice, channel_slice)

    diff_quant = quant_surface - baseline_surface
    diff_atm = atm_surface - baseline_surface

    fig = plt.figure(figsize=(18, 6))
    titles = ["Baseline (FP)", "Post-DuQuant", "Post-ATM"]
    surfaces = [baseline_surface, quant_surface, atm_surface]
    common_min = min(np.min(baseline_surface), np.min(quant_surface), np.min(atm_surface))
    common_max = max(np.max(baseline_surface), np.max(quant_surface), np.max(atm_surface))

    def summarize(name: str, array: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(array.mean()),
            "std": float(array.std()),
            "min": float(array.min()),
            "max": float(array.max()),
        }

    stats_msg = [
        f"{name}\nμ={vals['mean']:.3f} σ={vals['std']:.3f}\nmin={vals['min']:.2f} max={vals['max']:.2f}"
        for name, vals in zip(titles, [summarize("baseline", baseline_surface), summarize("quant", quant_surface), summarize("atm", atm_surface)])
    ]

    for i, (title, surface, msg) in enumerate(zip(titles, surfaces, stats_msg), start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        plot_surface(ax, surface, title, zlim=(common_min, common_max))
        ax.text2D(0.05, 0.02, msg, transform=ax.transAxes, fontsize=_BASE_FONT + 1, bbox=dict(facecolor="white", alpha=0.4))
    plt.tight_layout()
    out_path = Path(args.output)
    plt.savefig(out_path, dpi=240)
    plt.close()
    print(f"[ATM-SURF] Saved figure to {out_path}")

    def summarize(name: str, array: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(array.mean()),
            "std": float(array.std()),
            "min": float(array.min()),
            "max": float(array.max()),
            "abs_mean": float(np.abs(array).mean()),
        }

    stats = {
        "baseline": summarize("baseline", baseline_surface),
        "quant": summarize("quant", quant_surface),
        "atm": summarize("atm", atm_surface),
        "quant_minus_baseline": summarize("quant_minus_baseline", diff_quant),
        "atm_minus_baseline": summarize("atm_minus_baseline", diff_atm),
    }
    print("[ATM-SURF] Summary statistics:")
    for key, val in stats.items():
        print(f"  {key}: {json.dumps(val, indent=2)}")

    if args.stats_output:
        stats_path = Path(args.stats_output)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"[ATM-SURF] Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
