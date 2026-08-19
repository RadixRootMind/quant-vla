#!/usr/bin/env python
"""
Calibration script for GR00T DuQuant + ATM + Per-Head OHB on DiT attention for SimplerEnv.

This script is similar to calibrate_atm_simpler.py but computes PER-HEAD beta values
instead of per-layer beta values for OHB (Output Head Balancing).

Per-layer OHB: 16 beta values (1 per transformer block)
Per-head OHB:  16 * 32 = 512 beta values (32 heads per transformer block)

Usage:
    python tools/calibrate_atm_simpler_perhead_ohb.py \
        --teacher-checkpoint youliangtan/gr00t-n1.5-fractal-posttrain \
        --quant-checkpoint youliangtan/gr00t-n1.5-fractal-posttrain \
        --env google_robot_pick_coke_can \
        --steps 32 \
        --out atm_alphas_fractal_perhead_ohb.json \
        --calibrate-ohb 1
"""

import argparse
import copy
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# Add SimplerEnv to path
SIMPLER_DIR = REPO_DIR / "SimplerEnv"
if str(SIMPLER_DIR) not in sys.path:
    sys.path.insert(0, str(SIMPLER_DIR))

DEFAULT_DUQUANT_ENVS = {
    "GR00T_DUQUANT_SCOPE": "",
    "GR00T_DUQUANT_INCLUDE": r".*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)).*",
    "GR00T_DUQUANT_EXCLUDE": r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|attn1)(?:\.|$)",
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
    register_ohb_perhead_capture,
)
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import load_data_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate ATM alphas and Per-Head OHB betas for GR00T DiT attention on SimplerEnv."
    )
    parser.add_argument("--teacher-checkpoint", type=str, required=True, help="Teacher (FP) model path or HF repo.")
    parser.add_argument("--quant-checkpoint", type=str, required=True, help="Quantized model path or HF repo.")
    parser.add_argument("--env", type=str, default="google_robot_pick_coke_can", help="SimplerEnv environment name.")
    parser.add_argument("--data-config", type=str, default="examples.SimplerEnv.custom_data_config:FractalDataConfig",
                        help="Data config module path.")
    parser.add_argument("--steps", type=int, default=32, help="Number of observation steps to collect for calibration.")
    parser.add_argument("--out", type=str, required=True, help="Output JSON path for alpha/beta coefficients.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for model inference.")
    parser.add_argument("--calibrate-ohb", type=int, default=1,
                        help="Enable Per-Head OHB calibration alongside ATM (default: 1).")
    parser.add_argument("--ohb-log-clamp", type=float, default=0.30,
                        help="Log-domain clamp (delta) for OHB beta computation.")
    parser.add_argument("--ohb-neutral", type=float, default=0.03,
                        help="Neutral threshold in log-domain; |log(beta)| below this is set to beta=1.")
    parser.add_argument("--ohb-scope", type=str, default="dit", help="Scope prefix for OHB (default dit).")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode.")
    parser.add_argument("--policy-driven", action="store_true",
                        help="Use teacher policy to generate actions (default: dummy action like LIBERO).")
    parser.add_argument("--alpha-min", type=float, default=0.7,
                        help="Minimum alpha clamp value (default 0.7, use 0.9 for conservative).")
    parser.add_argument("--alpha-max", type=float, default=1.4,
                        help="Maximum alpha clamp value (default 1.4, use 1.1 for conservative).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    return parser.parse_args()


class StdCollector:
    """Aggregates per-layer per-head statistics for ATM alpha."""

    def __init__(self) -> None:
        self.sum: Dict[str, torch.Tensor] = {}
        self.count: Dict[str, int] = defaultdict(int)

    def __call__(self, layer_name: str, std_tensor: torch.Tensor) -> None:
        if std_tensor.ndim == 4:
            std_tensor = std_tensor.mean(dim=(2, 3))
        elif std_tensor.ndim == 3:
            std_tensor = std_tensor.mean(dim=2)
        std_tensor = std_tensor.mean(dim=0)
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


class RMSPerHeadCollector:
    """Aggregates per-layer per-head RMS values for per-head OHB."""

    def __init__(self) -> None:
        self.sum: Dict[str, torch.Tensor] = {}
        self.count: Dict[str, int] = defaultdict(int)

    def __call__(self, layer_name: str, rms_tensor: torch.Tensor) -> None:
        """
        Args:
            layer_name: Name of the attention layer
            rms_tensor: Shape (heads,) - per-head RMS values
        """
        rms_cpu = rms_tensor.detach().to(torch.float32).cpu()
        if layer_name not in self.sum:
            self.sum[layer_name] = rms_cpu.clone()
        else:
            self.sum[layer_name] += rms_cpu
        self.count[layer_name] += 1

    def finalize(self) -> Dict[str, torch.Tensor]:
        """Returns dict mapping layer_name -> (heads,) averaged RMS tensor."""
        result: Dict[str, torch.Tensor] = {}
        for name, tensor in self.sum.items():
            denom = max(self.count[name], 1)
            result[name] = tensor / denom
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


def get_simpler_env(env_name: str, headless: bool = True):
    """Create a SimplerEnv environment with proper wrappers."""
    import simpler_env
    import cv2
    import gym
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    # Determine robot type
    if "google" in env_name or env_name.startswith("Grasp") or env_name.startswith("Move") or env_name.startswith("Open") or env_name.startswith("Close"):
        robot_type = "google"
    else:
        robot_type = "widowx"

    # Create base environment using simpler_env.make()
    base_env = simpler_env.make(env_name)

    class SimplerObsWrapper(gym.Wrapper):
        """Wrapper that provides image_primary and proprio observations."""
        def __init__(self, env, image_size=(320, 256)):
            super().__init__(env)
            self.image_size = image_size
            self.robot_type = robot_type

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            obs = self._process_obs(obs)
            return obs, info

        def step(self, action):
            obs, reward, done, truncated, info = self.env.step(action)
            obs = self._process_obs(obs)
            return obs, reward, done, truncated, info

        def _process_obs(self, obs):
            img = get_image_from_maniskill2_obs_dict(self.env, obs, camera_name=None)
            img = cv2.resize(img, self.image_size)
            proprio = obs["agent"]["eef_pos"]
            return {
                "image_primary": img,
                "proprio": proprio,
            }

    wrapped_env = SimplerObsWrapper(base_env)
    return wrapped_env, robot_type


def convert_simpler_observation(obs: Dict, language: str, robot_type: str = "google") -> Dict[str, np.ndarray]:
    """Convert wrapped SimplerEnv observation to GR00T format."""
    img = obs["image_primary"]
    proprio = obs["proprio"]

    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)

    if robot_type == "google":
        xyz = proprio[:3]
        quat = proprio[3:7]
        gripper = proprio[7] if len(proprio) > 7 else 0.0

        converted = {
            "video.image": img.astype(np.uint8),
            "state.x": np.array([[xyz[0]]], dtype=np.float32),
            "state.y": np.array([[xyz[1]]], dtype=np.float32),
            "state.z": np.array([[xyz[2]]], dtype=np.float32),
            "state.rx": np.array([[quat[0]]], dtype=np.float32),
            "state.ry": np.array([[quat[1]]], dtype=np.float32),
            "state.rz": np.array([[quat[2]]], dtype=np.float32),
            "state.rw": np.array([[quat[3]]], dtype=np.float32),
            "state.gripper": np.array([[gripper]], dtype=np.float32),
            "annotation.human.action.task_description": [language],
        }
    else:
        from transforms3d.euler import quat2euler
        xyz = proprio[:3]
        quat = proprio[3:7]
        gripper = proprio[7] if len(proprio) > 7 else 0.0
        euler = quat2euler(quat)

        converted = {
            "video.image_0": img.astype(np.uint8),
            "state.x": np.array([[xyz[0]]], dtype=np.float32),
            "state.y": np.array([[xyz[1]]], dtype=np.float32),
            "state.z": np.array([[xyz[2]]], dtype=np.float32),
            "state.roll": np.array([[euler[0]]], dtype=np.float32),
            "state.pitch": np.array([[euler[1]]], dtype=np.float32),
            "state.yaw": np.array([[euler[2]]], dtype=np.float32),
            "state.pad": np.array([[0.0]], dtype=np.float32),
            "state.gripper": np.array([[gripper]], dtype=np.float32),
            "annotation.human.action.task_description": [language],
        }

    return converted


def gather_simpler_observations(
    env_name: str,
    max_steps: int,
    headless: bool = True,
    teacher_policy: Optional[Gr00tPolicy] = None,
) -> List[Dict[str, np.ndarray]]:
    """Gather observations from SimplerEnv for calibration."""
    torch.set_grad_enabled(False)
    dataset: List[Dict[str, np.ndarray]] = []

    LANGUAGE_MAP = {
        "google_robot_pick_coke_can": "pick coke can",
        "google_robot_pick_object": "pick up the object",
        "google_robot_move_near": "move the object near the target",
        "google_robot_open_drawer": "open the drawer",
        "google_robot_close_drawer": "close the drawer",
        "widowx_spoon_on_towel": "put the spoon on the towel",
        "widowx_carrot_on_plate": "put the carrot on the plate",
    }

    print(f"[SimplerEnv] Creating environment: {env_name}")

    env, robot_type = get_simpler_env(env_name, headless=headless)

    if env_name in LANGUAGE_MAP:
        language = LANGUAGE_MAP[env_name]
    elif robot_type == "google":
        language = env_name.replace("google_robot_", "").replace("_", " ")
    else:
        language = env_name.replace("widowx_", "").replace("_", " ")

    if teacher_policy is not None:
        action_mode = "policy-driven (teacher)"
    else:
        action_mode = "dummy (no-op, like LIBERO)"
    print(f"[SimplerEnv] Robot type: {robot_type}, Language: {language}")
    print(f"[SimplerEnv] Action mode: {action_mode}")

    episodes = 0
    max_episodes = 10

    while len(dataset) < max_steps and episodes < max_episodes:
        obs, _ = env.reset()
        episodes += 1
        step = 0
        max_ep_steps = 200

        while step < max_ep_steps and len(dataset) < max_steps:
            try:
                converted = convert_simpler_observation(obs, language, robot_type)
                dataset.append(converted)
            except Exception as e:
                print(f"[SimplerEnv] Warning: Failed to convert observation: {e}")
                break

            if teacher_policy is not None:
                try:
                    action_dict = teacher_policy.get_action(converted)
                    action = np.concatenate([
                        action_dict.get("action.x", [[0]])[0],
                        action_dict.get("action.y", [[0]])[0],
                        action_dict.get("action.z", [[0]])[0],
                        action_dict.get("action.roll", [[0]])[0],
                        action_dict.get("action.pitch", [[0]])[0],
                        action_dict.get("action.yaw", [[0]])[0],
                        action_dict.get("action.gripper", [[0]])[0],
                    ])
                except Exception as e:
                    print(f"[SimplerEnv] Warning: Policy action failed: {e}, using dummy action")
                    action = np.zeros(7)
                    action[-1] = -1
            else:
                action = np.zeros(7)
                action[-1] = -1

            obs, _, done, truncated, _ = env.step(action)
            step += 1

            if done or truncated:
                break

    env.close()
    print(f"[SimplerEnv] Collected {len(dataset)} observations from {episodes} episodes")
    return dataset


def run_policy_with_capture(
    policy: Gr00tPolicy,
    dataset: List[Dict[str, np.ndarray]],
    collector: StdCollector,
    *,
    ohb_perhead_collector: Optional[RMSPerHeadCollector] = None,
    atm_scope: str = "dit",
    ohb_scope: str = "dit",
) -> None:
    model = policy.model
    ensure_dit_attention_patch(model)
    register_atm_capture(model, collector, scope=atm_scope)
    if ohb_perhead_collector is not None:
        register_ohb_perhead_capture(model, ohb_perhead_collector, scope=ohb_scope)
    torch.set_grad_enabled(False)
    for obs in tqdm(dataset, desc="Running policy", leave=False):
        try:
            policy.get_action(obs)
        except Exception as e:
            print(f"[Warning] Policy inference failed: {e}")
            continue
    clear_atm_capture(model)


def compute_alpha_json(
    teacher_stats: Dict[str, torch.Tensor],
    quant_stats: Dict[str, torch.Tensor],
    min_alpha: float = 0.7,
    max_alpha: float = 1.4,
    neutral_threshold: float = 0.02,
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


def compute_beta_perhead_values(
    teacher_rms: Dict[str, torch.Tensor],
    quant_rms: Dict[str, torch.Tensor],
    *,
    log_clamp: float,
    neutral: float,
) -> Dict[str, List[float]]:
    """Compute per-head beta values for OHB.

    Args:
        teacher_rms: Dict mapping layer_name -> (heads,) tensor of teacher RMS
        quant_rms: Dict mapping layer_name -> (heads,) tensor of quant RMS
        log_clamp: Log-domain clamp for beta computation
        neutral: Neutral threshold below which beta is set to 1.0

    Returns:
        Dict mapping layer_name -> list of per-head beta values
    """
    beta_map: Dict[str, List[float]] = {}
    for name, teacher_tensor in teacher_rms.items():
        if name not in quant_rms:
            continue
        quant_tensor = quant_rms[name]
        num_heads = teacher_tensor.shape[0]
        beta_list = []

        for h in range(num_heads):
            t = max(float(teacher_tensor[h]), 1e-8)
            q = max(float(quant_tensor[h]), 1e-8)
            rho = q / t
            log_beta = -math.log(max(rho, 1e-8))
            log_beta = max(-log_clamp, min(log_clamp, log_beta))
            if abs(log_beta) < neutral:
                beta = 1.0
            else:
                beta = math.exp(log_beta)
            beta_list.append(beta)

        beta_map[name] = beta_list
    return beta_map


def log_beta_perhead_summary(beta_map: Dict[str, List[float]], log_clamp: float) -> None:
    if not beta_map:
        print("[OHB-PerHead] No layers produced beta values.")
        return

    total_heads = 0
    all_deviations = []
    for name, betas in beta_map.items():
        total_heads += len(betas)
        for b in betas:
            all_deviations.append(abs(math.log(max(b, 1e-8))))

    max_dev = max(all_deviations)
    mean_dev = sum(all_deviations) / len(all_deviations)
    num_layers = len(beta_map)

    print(
        f"[OHB-PerHead] layers={num_layers} total_heads={total_heads} "
        f"mean|log_beta|={mean_dev:.4f} max|log_beta|={max_dev:.4f} (clamp={log_clamp})"
    )
    if max_dev >= log_clamp * 0.95:
        print("[OHB-PerHead] Warning: Some heads hit the log clamp; consider adjusting parameters.")


def log_rms_perhead_debug(
    teacher_rms: Dict[str, torch.Tensor],
    quant_rms: Dict[str, torch.Tensor],
) -> None:
    """Print debug info about per-head RMS values."""
    print("\n[OHB-PerHead DEBUG] Per-head RMS statistics:")
    for name in sorted(teacher_rms.keys())[:3]:  # Show first 3 layers
        if name not in quant_rms:
            continue
        t_rms = teacher_rms[name]
        q_rms = quant_rms[name]
        print(f"  {name}:")
        print(f"    Teacher RMS: min={t_rms.min():.6f} max={t_rms.max():.6f} std={t_rms.std():.6f}")
        print(f"    Quant RMS:   min={q_rms.min():.6f} max={q_rms.max():.6f} std={q_rms.std():.6f}")
        print(f"    Ratio q/t:   min={(q_rms/t_rms).min():.4f} max={(q_rms/t_rms).max():.4f}")
    print()


def main():
    args = parse_args()

    # Set random seed for reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"[ATM] Random seed: {args.seed}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    original_duquant_env = _strip_duquant_env()
    original_atm_env = _disable_atm_env()
    ohb_enabled = bool(args.calibrate_ohb)

    # Load data config
    data_config = load_data_config(args.data_config)

    print("[ATM] Loading teacher policy (FP16)...")
    teacher_policy = Gr00tPolicy(
        model_path=args.teacher_checkpoint,
        modality_config=copy.deepcopy(data_config.modality_config()),
        modality_transform=data_config.transform(),
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        denoising_steps=8,
        device=args.device,
    )

    print("[ATM] Gathering observations from SimplerEnv...")
    dataset = gather_simpler_observations(
        env_name=args.env,
        max_steps=args.steps,
        headless=args.headless,
        teacher_policy=teacher_policy if args.policy_driven else None,
    )
    if not dataset:
        raise RuntimeError("Failed to gather any observations for calibration.")
    print(f"[ATM] Collected {len(dataset)} observation steps for calibration.")

    teacher_collector = StdCollector()
    teacher_ohb_collector = RMSPerHeadCollector() if ohb_enabled else None
    print("[ATM] Running teacher policy...")
    run_policy_with_capture(
        teacher_policy,
        dataset,
        teacher_collector,
        ohb_perhead_collector=teacher_ohb_collector,
        atm_scope="dit",
        ohb_scope=args.ohb_scope,
    )
    teacher_stats = teacher_collector.finalize()
    teacher_ohb_stats = teacher_ohb_collector.finalize() if teacher_ohb_collector else {}

    # Free teacher model
    del teacher_policy
    torch.cuda.empty_cache()

    # Prepare quant policy with DuQuant
    _restore_env(original_duquant_env)
    os.environ.setdefault("GR00T_ATM_ENABLE", "0")
    print("[ATM] Loading quant policy with DuQuant...")
    _ensure_duquant_defaults()
    quant_policy = Gr00tPolicy(
        model_path=args.quant_checkpoint,
        modality_config=copy.deepcopy(data_config.modality_config()),
        modality_transform=data_config.transform(),
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        denoising_steps=8,
        device=args.device,
    )

    quant_collector = StdCollector()
    quant_ohb_collector = RMSPerHeadCollector() if ohb_enabled else None
    print("[ATM] Running quantized policy...")
    run_policy_with_capture(
        quant_policy,
        dataset,
        quant_collector,
        ohb_perhead_collector=quant_ohb_collector,
        atm_scope="dit",
        ohb_scope=args.ohb_scope,
    )
    quant_stats = quant_collector.finalize()
    quant_ohb_stats = quant_ohb_collector.finalize() if quant_ohb_collector else {}

    alpha_json = compute_alpha_json(
        teacher_stats, quant_stats,
        min_alpha=args.alpha_min,
        max_alpha=args.alpha_max,
    )
    if ohb_enabled:
        beta_perhead_map = compute_beta_perhead_values(
            teacher_ohb_stats,
            quant_ohb_stats,
            log_clamp=args.ohb_log_clamp,
            neutral=args.ohb_neutral,
        )
        log_beta_perhead_summary(beta_perhead_map, args.ohb_log_clamp)
        for name, beta_list in beta_perhead_map.items():
            alpha_json.setdefault(name, {})
            alpha_json[name]["beta_perhead"] = beta_list

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(alpha_json, f, indent=2)
    print(f"[ATM] Wrote alpha/beta JSON to {out_path}")

    # restore environment
    _restore_env(original_atm_env)


if __name__ == "__main__":
    main()
