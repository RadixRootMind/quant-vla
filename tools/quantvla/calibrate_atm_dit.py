#!/usr/bin/env python
"""
Calibration script for GR00T DuQuant + ATM on DiT attention.

This script collects a set of LIBERO observations, runs the teacher (FP32)
and quantized (DuQuant) policies, measures per-head attention logits std,
and exports an alpha JSON for inference-time scaling.
"""

import argparse
import copy
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

DEFAULT_DUQUANT_ENVS = {
    "GR00T_DUQUANT_SCOPE": "",
    "GR00T_DUQUANT_INCLUDE": r".*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)).*",
    "GR00T_DUQUANT_EXCLUDE": r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head)(?:\.|$)",
    "GR00T_DUQUANT_WBITS_DEFAULT": "4",
    "GR00T_DUQUANT_ABITS": "8",
    "GR00T_DUQUANT_BLOCK": "64",
    "GR00T_DUQUANT_PERMUTE": "0",
    "GR00T_DUQUANT_ROW_ROT": "restore",
    "GR00T_DUQUANT_ACT_PCT": "99.9",
    "GR00T_DUQUANT_CALIB_STEPS": "32",
    "GR00T_DUQUANT_LS": "0.15",
}

from gr00t.atm import (
    clear_atm_capture,
    ensure_dit_attention_patch,
    register_atm_capture,
    register_ohb_capture,
)
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

from examples.Libero.eval.utils import get_libero_env, get_libero_image, quat2axisangle, get_libero_dummy_action
from libero.libero import benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate ATM alphas for GR00T DiT attention.")
    parser.add_argument("--teacher-checkpoint", type=str, required=True, help="Teacher (FP) model path or HF repo.")
    parser.add_argument("--quant-checkpoint", type=str, required=True, help="Quantized model path or HF repo.")
    parser.add_argument("--task-suite", type=str, default="libero_spatial", help="LIBERO task suite name.")
    parser.add_argument(
        "--steps",
        type=int,
        default=32,
        help="Number of observation steps to collect for calibration.",
    )
    parser.add_argument("--out", type=str, required=True, help="Output JSON path for alpha coefficients.")
    parser.add_argument(
        "--max-trials-per-task",
        type=int,
        default=5,
        help="Maximum trials per task when gathering observations.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for model inference.",
    )
    parser.add_argument(
        "--calibrate-ohb",
        type=int,
        default=0,
        help="Enable Output Head Balancing (OHB) calibration alongside ATM.",
    )
    parser.add_argument(
        "--ohb-log-clamp",
        type=float,
        default=0.30,
        help="Log-domain clamp (delta) for OHB beta computation.",
    )
    parser.add_argument(
        "--ohb-neutral",
        type=float,
        default=0.03,
        help="Neutral threshold in log-domain; |log(beta)| below this is set to beta=1.",
    )
    parser.add_argument(
        "--ohb-scope",
        type=str,
        default="dit",
        help="Scope prefix for OHB (default dit).",
    )
    return parser.parse_args()


class StdCollector:
    """Aggregates per-layer per-head statistics."""

    def __init__(self) -> None:
        self.sum: Dict[str, torch.Tensor] = {}
        self.count: Dict[str, int] = defaultdict(int)

    def __call__(self, layer_name: str, std_tensor: torch.Tensor) -> None:
        if std_tensor.ndim == 4:
            # (batch, heads, q_len, k_len) - reduce last dims
            std_tensor = std_tensor.mean(dim=(2, 3))
        elif std_tensor.ndim == 3:
            std_tensor = std_tensor.mean(dim=2)
        std_tensor = std_tensor.mean(dim=0)  # average over batch
        std_cpu = std_tensor.detach().to(torch.float32)
        if layer_name not in self.sum:
            self.sum[layer_name] = std_cpu.clone()
        else:
            self.sum[layer_name] += std_cpu
        self.count[layer_name] += 1

    def finalize(self) -> Dict[str, torch.Tensor]:
        result = {}
        for name, tensor in self.sum.items():
            denom = max(self.count[name], 1)
            result[name] = tensor / denom
        return result


class RMSCollector:
    """Aggregates per-layer RMS values (scalar)."""

    def __init__(self) -> None:
        self.sum: Dict[str, float] = defaultdict(float)
        self.count: Dict[str, int] = defaultdict(int)

    def __call__(self, layer_name: str, rms_tensor: torch.Tensor) -> None:
        value = float(rms_tensor.detach().to(torch.float32).mean().cpu())
        self.sum[layer_name] += value
        self.count[layer_name] += 1

    def finalize(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for name, total in self.sum.items():
            denom = max(self.count[name], 1)
            result[name] = total / denom
        return result


def _strip_duquant_env() -> Dict[str, str]:
    backup = {}
    for key in list(os.environ.keys()):
        if key.startswith("GR00T_DUQUANT_"):
            backup[key] = os.environ.pop(key)
    return backup


def _restore_env(env_map: Dict[str, str]) -> None:
    for key, value in env_map.items():
        os.environ[key] = value


def _ensure_duquant_defaults() -> None:
    for key, value in DEFAULT_DUQUANT_ENVS.items():
        os.environ.setdefault(key, value)


def _disable_atm_env() -> Dict[str, str]:
    keys = [k for k in os.environ.keys() if k.startswith("GR00T_ATM_")]
    backup = {k: os.environ.pop(k) for k in keys}
    return backup


def policy_modality_config():
    from gr00t.experiment.data_config import load_data_config

    data_config = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    return data_config.modality_config()


def policy_modality_transform():
    from gr00t.experiment.data_config import load_data_config

    data_config = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    return data_config.transform()


def _convert_observation(obs: Dict[str, np.ndarray], language: str) -> Dict[str, np.ndarray]:
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


def gather_observations(
    task_suite_name: str,
    max_steps: int,
    max_trials_per_task: int,
) -> List[Dict[str, np.ndarray]]:
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


def run_policy_with_capture(
    policy: Gr00tPolicy,
    dataset: List[Dict[str, np.ndarray]],
    collector: StdCollector,
    *,
    ohb_collector: RMSCollector | None = None,
    atm_scope: str = "dit",
    ohb_scope: str = "dit",
) -> None:
    model = policy.model
    ensure_dit_attention_patch(model)
    register_atm_capture(model, collector, scope=atm_scope)
    if ohb_collector is not None:
        register_ohb_capture(model, ohb_collector, scope=ohb_scope)
    torch.set_grad_enabled(False)
    for obs in tqdm(dataset, desc="Running policy", leave=False):
        policy.get_action(obs)
    clear_atm_capture(model)


def compute_alpha_json(
    teacher_stats: Dict[str, torch.Tensor],
    quant_stats: Dict[str, torch.Tensor],
    min_alpha: float = 0.7,
    max_alpha: float = 1.4,
    neutral_threshold: float = 0.02,  # Lowered from 0.05 to preserve more subtle corrections
) -> Dict[str, Dict[str, List[float]]]:
    alpha_data: Dict[str, Dict[str, List[float]]] = {}
    for name in sorted(teacher_stats.keys()):
        if name not in quant_stats:
            continue
        teacher_std = teacher_stats[name].to(torch.float32)
        quant_std = quant_stats[name].to(torch.float32)
        alpha = torch.where(
            quant_std > 0,
            teacher_std / (quant_std + 1e-6),
            torch.ones_like(teacher_std),
        )
        alpha = alpha.clamp(min_alpha, max_alpha)
        alpha = torch.where((alpha - 1.0).abs() < neutral_threshold, torch.ones_like(alpha), alpha)
        alpha_data[name] = {"all": alpha.tolist()}
    return alpha_data


def compute_beta_values(
    teacher_rms: Dict[str, float],
    quant_rms: Dict[str, float],
    *,
    log_clamp: float,
    neutral: float,
) -> Dict[str, float]:
    beta_map: Dict[str, float] = {}
    for name, teacher_val in teacher_rms.items():
        if name not in quant_rms:
            continue
        t = max(teacher_val, 1e-8)
        q = max(quant_rms[name], 1e-8)
        rho = q / t
        log_beta = -math.log(max(rho, 1e-8))
        log_beta = max(-log_clamp, min(log_clamp, log_beta))
        if abs(log_beta) < neutral:
            beta = 1.0
        else:
            beta = math.exp(log_beta)
        beta_map[name] = beta
    return beta_map


def log_beta_summary(beta_map: Dict[str, float], log_clamp: float) -> None:
    if not beta_map:
        print("[OHB] No layers produced beta values.")
        return
    deviations = [abs(math.log(max(b, 1e-8))) for b in beta_map.values()]
    max_dev = max(deviations)
    mean_dev = sum(deviations) / len(deviations)
    print(
        f"[OHB] beta count={len(beta_map)} mean|log_beta|={mean_dev:.4f} "
        f"max|log_beta|={max_dev:.4f} (clamp={log_clamp})"
    )
    if max_dev >= log_clamp * 0.95:
        print("[OHB] ⚠️ Some layers hit the log clamp; consider deeper inspection.")


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    original_duquant_env = _strip_duquant_env()
    original_atm_env = _disable_atm_env()
    ohb_enabled = bool(args.calibrate_ohb)

    print("[ATM] Loading teacher policy...")
    teacher_policy = Gr00tPolicy(
        model_path=args.teacher_checkpoint,
        modality_config=copy.deepcopy(policy_modality_config()),
        modality_transform=policy_modality_transform(),
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        denoising_steps=8,
        device=args.device,
    )

    teacher_collector = StdCollector()
    teacher_ohb_collector = RMSCollector() if ohb_enabled else None
    dataset = gather_observations(
        task_suite_name=args.task_suite,
        max_steps=args.steps,
        max_trials_per_task=args.max_trials_per_task,
    )
    if not dataset:
        raise RuntimeError("Failed to gather any observations for calibration.")
    print(f"[ATM] Collected {len(dataset)} observation steps for calibration.")
    run_policy_with_capture(
        teacher_policy,
        dataset,
        teacher_collector,
        ohb_collector=teacher_ohb_collector,
        atm_scope="dit",
        ohb_scope=args.ohb_scope,
    )
    teacher_stats = teacher_collector.finalize()
    teacher_ohb_stats = teacher_ohb_collector.finalize() if teacher_ohb_collector else {}

    # Prepare quant policy with DuQuant
    _restore_env(original_duquant_env)
    os.environ.setdefault("GR00T_ATM_ENABLE", "0")
    print("[ATM] Loading quant policy with DuQuant...")
    _ensure_duquant_defaults()
    quant_policy = Gr00tPolicy(
        model_path=args.quant_checkpoint,
        modality_config=copy.deepcopy(policy_modality_config()),
        modality_transform=policy_modality_transform(),
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        denoising_steps=8,
        device=args.device,
    )

    quant_collector = StdCollector()
    quant_ohb_collector = RMSCollector() if ohb_enabled else None
    run_policy_with_capture(
        quant_policy,
        dataset,
        quant_collector,
        ohb_collector=quant_ohb_collector,
        atm_scope="dit",
        ohb_scope=args.ohb_scope,
    )
    quant_stats = quant_collector.finalize()
    quant_ohb_stats = quant_ohb_collector.finalize() if quant_ohb_collector else {}

    alpha_json = compute_alpha_json(teacher_stats, quant_stats)
    if ohb_enabled:
        beta_map = compute_beta_values(
            teacher_ohb_stats,
            quant_ohb_stats,
            log_clamp=args.ohb_log_clamp,
            neutral=args.ohb_neutral,
        )
        log_beta_summary(beta_map, args.ohb_log_clamp)
        for name, beta in beta_map.items():
            alpha_json.setdefault(name, {})
            alpha_json[name]["beta"] = beta

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(alpha_json, f, indent=2)
    print(f"[ATM] Wrote alpha JSON to {out_path}")

    # restore environment
    _restore_env(original_atm_env)


if __name__ == "__main__":
    main()
