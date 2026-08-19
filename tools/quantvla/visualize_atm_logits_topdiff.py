#!/usr/bin/env python3
"""
Find the attention layer whose logits benefit most from ATM and visualize it.
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


class MultiLogitsCollector:
    def __init__(self) -> None:
        self.storage: Dict[str, torch.Tensor] = {}

    def callback(self, layer_name: str, logits: torch.Tensor) -> None:
        if layer_name not in self.storage:
            self.storage[layer_name] = logits.detach().cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ATM improvement on the most impacted layer.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--atm-json", required=True)
    parser.add_argument("--layer-name", default=None, help="Optional layer name to visualize directly.")
    parser.add_argument("--task-suite", default="libero_goal")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--max-trials-per-task", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--duquant-packdir", default=None)
    parser.add_argument("--head-index", type=int, default=0)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--token-range", default=None)
    parser.add_argument("--channel-range", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--normalize", action="store_true", help="Normalize all surfaces using baseline mean/std for better contrast.")
    parser.add_argument("--output", default="atm_logits_topdiff.png")
    parser.add_argument(
        "--comparison-output",
        default=None,
        help="Optional extra figure comparing Baseline↔Post-DuQuant and Baseline↔Post-ATM.",
    )
    parser.add_argument("--stats-output", default=None)
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


def run_with_capture(policy: Gr00tPolicy, dataset, scope: str) -> Dict[str, torch.Tensor]:
    model = policy.model
    ensure_dit_attention_patch(model, scope=scope)
    collector = MultiLogitsCollector()
    register_atm_logits_capture(model, collector.callback, scope=scope)
    torch.set_grad_enabled(False)
    for obs in tqdm(dataset, desc="Capturing logits", leave=False):
        policy.get_action(obs)
        if len(collector.storage) > 0:
            break
    clear_atm_capture(model)
    return collector.storage


def _parse_range(txt: str | None, upper: int) -> slice:
    if not txt:
        return slice(None)
    start, end = txt.split(":")
    start = int(start) if start else None
    end = int(end) if end else None
    return slice(start, end)


def extract_surface(tensor: torch.Tensor, batch_idx: int, head_idx: int, token_slice: slice, channel_slice: slice) -> np.ndarray:
    if tensor.ndim != 4:
        raise ValueError(f"Expected tensor shape [B, H, T, T], got {tensor.shape}")
    if batch_idx >= tensor.shape[0] or head_idx >= tensor.shape[1]:
        raise IndexError("batch/head index out of range")
    surface = tensor[batch_idx, head_idx].to(torch.float32)
    return surface[token_slice, channel_slice].numpy()


def plot_surface(ax, data: np.ndarray, title: str, *, zlim=None, cmap="coolwarm") -> None:
    tokens = np.arange(data.shape[0])
    channels = np.arange(data.shape[1])
    T, C = np.meshgrid(tokens, channels, indexing="ij")
    surf = ax.plot_surface(C, T, data, cmap=cmap, linewidth=0, antialiased=False, alpha=0.95)
    if zlim is not None:
        ax.set_zlim(*zlim)
        surf.set_norm(plt.Normalize(vmin=zlim[0], vmax=zlim[1]))
    ax.set_xlabel("Channel")
    ax.set_ylabel("Token")
    ax.set_zlabel("Logits")
    ax.set_title(title)


def summarize(array: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main():
    args = parse_args()
    if args.duquant_packdir:
        os.environ["GR00T_DUQUANT_PACKDIR"] = args.duquant_packdir

    dataset = gather_observations(args.task_suite, args.steps, args.max_trials_per_task)
    if not dataset:
        raise RuntimeError("Failed to gather observations.")

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

    print("[ATM-TOP] Capturing baseline logits...")
    policy_fp = make_policy()
    baseline_dict = run_with_capture(policy_fp, dataset, scope="dit")

    print("[ATM-TOP] Capturing post-DuQuant logits...")
    os.environ.update(saved_duquant)
    for key in list(os.environ.keys()):
        if key.startswith("GR00T_ATM_"):
            os.environ.pop(key)
    policy_quant = make_policy()
    quant_dict = run_with_capture(policy_quant, dataset, scope="dit")

    print("[ATM-TOP] Capturing post-ATM logits...")
    os.environ["GR00T_ATM_ENABLE"] = "1"
    os.environ["GR00T_ATM_ALPHA_PATH"] = args.atm_json
    os.environ.setdefault("GR00T_ATM_SCOPE", "dit")
    policy_atm = make_policy()
    atm_dict = run_with_capture(policy_atm, dataset, scope="dit")

    # restore env
    for key in list(os.environ.keys()):
        if key.startswith("GR00T_DUQUANT_") or key.startswith("GR00T_ATM_"):
            os.environ.pop(key)
    os.environ.update(saved_duquant)
    os.environ.update(saved_atm)

    metrics = []
    for layer, base_tensor in baseline_dict.items():
        if layer not in quant_dict or layer not in atm_dict:
            continue
        base = base_tensor.to(torch.float32)
        quant = quant_dict[layer].to(torch.float32)
        atm = atm_dict[layer].to(torch.float32)
        quant_err = torch.mean(torch.abs(quant - base)).item()
        atm_err = torch.mean(torch.abs(atm - base)).item()
        improvement = quant_err - atm_err
        metrics.append((layer, quant_err, atm_err, improvement))

    metrics.sort(key=lambda x: x[3], reverse=True)
    print("[ATM-TOP] Top layers by improvement:")
    for layer, q_err, a_err, imp in metrics[: args.top_k]:
        print(f"  {layer}: quant_err={q_err:.6f} atm_err={a_err:.6f} improvement={imp:.6f}")

    target_layer = args.layer_name or (metrics[0][0] if metrics else None)
    if target_layer is None:
        raise RuntimeError("No layer found for visualization.")
    print(f"[ATM-TOP] Visualizing layer: {target_layer}")

    token_slice = _parse_range(args.token_range, baseline_dict[target_layer].shape[-1])
    channel_slice = _parse_range(args.channel_range, baseline_dict[target_layer].shape[-1])

    baseline_surface = extract_surface(baseline_dict[target_layer], args.batch_index, args.head_index, token_slice, channel_slice)
    quant_surface = extract_surface(quant_dict[target_layer], args.batch_index, args.head_index, token_slice, channel_slice)
    atm_surface = extract_surface(atm_dict[target_layer], args.batch_index, args.head_index, token_slice, channel_slice)
    if args.normalize:
        ref_mean = float(baseline_surface.mean())
        ref_std = float(baseline_surface.std())
        if ref_std < 1e-6:
            ref_std = 1.0
        baseline_surface = (baseline_surface - ref_mean) / ref_std
        quant_surface = (quant_surface - ref_mean) / ref_std
        atm_surface = (atm_surface - ref_mean) / ref_std

    quant_minus_atm = quant_surface - atm_surface

    fig = plt.figure(figsize=(22, 6))
    common_min = min(np.min(baseline_surface), np.min(quant_surface), np.min(atm_surface))
    common_max = max(np.max(baseline_surface), np.max(quant_surface), np.max(atm_surface))
    diff_bound = np.abs(quant_minus_atm).max()

    panels = [
        ("Baseline (FP)", baseline_surface, (common_min, common_max), "coolwarm"),
        ("Post-DuQuant", quant_surface, (common_min, common_max), "coolwarm"),
        ("Post-ATM", atm_surface, (common_min, common_max), "coolwarm"),
    ]
    grid = fig.add_gridspec(1, 3, left=0.04, right=0.98, top=0.92, bottom=0.12, wspace=0.15)
    for idx, (title, data, zlim, cmap) in enumerate(panels):
        ax = fig.add_subplot(grid[0, idx], projection="3d")
        plot_surface(ax, data, title, zlim=zlim, cmap=cmap)
        stats = summarize(data)
        ax.text2D(
            0.72,
            0.9,
            f"μ={stats['mean']:.3f}\nσ={stats['std']:.3f}\nmin={stats['min']:.2f}\nmax={stats['max']:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.82, linewidth=0.4),
            clip_on=True,
        )

    plt.tight_layout()
    out_path = Path(args.output)
    plt.savefig(out_path, dpi=240)
    plt.close()
    print(f"[ATM-TOP] Saved figure to {out_path}")

    summary = {
        "layer": target_layer,
        "normalized": args.normalize,
        "baseline": summarize(baseline_surface),
        "post_duquant": summarize(quant_surface),
        "post_atm": summarize(atm_surface),
        "quant_minus_atm": summarize(quant_minus_atm),
        "top_layers": [{"name": name, "quant_err": q, "atm_err": a, "improvement": imp} for name, q, a, imp in metrics[: args.top_k]],
    }

    if args.comparison_output:
        cmp_fig = plt.figure(figsize=(18, 10))
        rows = 2
        cmp_grid = cmp_fig.add_gridspec(rows, 2, left=0.04, right=0.98, top=0.94, bottom=0.08, hspace=0.18, wspace=0.18)
        pair_specs = [
            ("Baseline (FP)", baseline_surface, "Post-DuQuant", quant_surface, (common_min, common_max)),
            ("Baseline (FP)", baseline_surface, "Post-ATM", atm_surface, (common_min, common_max)),
        ]
        for row_idx, (title_a, data_a, title_b, data_b, zlim) in enumerate(pair_specs):
            ax_a = cmp_fig.add_subplot(cmp_grid[row_idx, 0], projection="3d")
            plot_surface(ax_a, data_a, title_a, zlim=zlim, cmap="coolwarm")
            stats_a = summarize(data_a)
            ax_a.text2D(
                0.7,
                0.9,
                f"μ={stats_a['mean']:.3f}\nσ={stats_a['std']:.3f}\nmin={stats_a['min']:.2f}\nmax={stats_a['max']:.2f}",
                transform=ax_a.transAxes,
                va="top",
                ha="left",
                bbox=dict(facecolor="white", alpha=0.82, linewidth=0.4),
            )
            ax_b = cmp_fig.add_subplot(cmp_grid[row_idx, 1], projection="3d")
            plot_surface(ax_b, data_b, title_b, zlim=zlim, cmap="coolwarm")
            stats_b = summarize(data_b)
            ax_b.text2D(
                0.7,
                0.9,
                f"μ={stats_b['mean']:.3f}\nσ={stats_b['std']:.3f}\nmin={stats_b['min']:.2f}\nmax={stats_b['max']:.2f}",
                transform=ax_b.transAxes,
                va="top",
                ha="left",
                bbox=dict(facecolor="white", alpha=0.82, linewidth=0.4),
            )
        cmp_path = Path(args.comparison_output)
        cmp_fig.savefig(cmp_path, dpi=240)
        plt.close(cmp_fig)
        print(f"[ATM-TOP] Saved comparison figure to {cmp_path}")

    if args.stats_output:
        stats_path = Path(args.stats_output)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[ATM-TOP] Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
