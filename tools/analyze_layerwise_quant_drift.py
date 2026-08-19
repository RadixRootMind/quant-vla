#!/usr/bin/env python
"""
Layer-wise DuQuant drift analysis on LIBERO samples.

This script compares a full-precision GR00T policy against per-scenario
quantized policies built with DuQuant. It supports:

1. Full-precision baseline metrics
2. Individual layer quantization sweeps
3. Cumulative quantization sweeps
4. Per-layer weight drift (W)
5. Per-layer output drift (WX) under the same inputs and random seeds

The default data source is a LeRobot-format LIBERO dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import site
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_ROOT = Path(
    os.environ.get(
        "AWESOME_QVLA_CACHE_ROOT",
        os.environ.get("QUANTVLA_CACHE_ROOT", str(Path.home() / ".cache" / "awesome_quant_vla")),
    )
)
DEFAULT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(DEFAULT_CACHE_ROOT / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DEFAULT_CACHE_ROOT / "huggingface" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(DEFAULT_CACHE_ROOT / "huggingface" / "transformers"))
os.environ.setdefault("HF_MODULES_CACHE", str(DEFAULT_CACHE_ROOT / "huggingface" / "modules"))
os.environ.setdefault("TORCH_HOME", str(DEFAULT_CACHE_ROOT / "torch"))
os.environ.setdefault("LIBERO_CONFIG_PATH", str(Path.home() / ".libero"))
os.environ.setdefault("LIBERO_ROOT", str(Path.home() / "LIBERO"))

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy, unsqueeze_dict_values
from gr00t.quantization.duquant_layers import DuQuantLinear, select_targets
from gr00t.quantization.duquant_preprocess import fake_quantize_sym


def normalized_input_no_inference(policy, sample_obs):
    """Mimic policy.get_action's preprocessing without inference_mode.

    Generic LIBERO calibration helper shared by the offline pack builders and
    the SVD diagnostics. (Lives here because analyze_layerwise_quant_drift is
    the common import hub for those tools.)
    """
    obs_copy = sample_obs.copy()
    obs_copy = unsqueeze_dict_values(obs_copy)
    for k, v in obs_copy.items():
        if not isinstance(v, np.ndarray):
            obs_copy[k] = np.array(v)
    normalized_input = policy.apply_transforms(obs_copy)
    return normalized_input


DEFAULT_INCLUDE_REGEX = (
    r".*("
    r"backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    r"|"
    r"action_head\.model\.transformer_blocks\.\d+\.(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2))"
    r").*"
)
DEFAULT_EXCLUDE_REGEX = (
    r"(?:^|\.)"
    r"(vision|radio|norm|ln|layernorm|embed|lm_head|timestep_encoder|state_encoder|"
    r"action_encoder|action_decoder|pos_embed|vl_self_attention|vlln|future_tokens)"
    r"(?:\.|$)"
)


@dataclass
class Scenario:
    name: str
    mode: str
    quantized_layers: list[str]
    focus_layer: str | None = None


class RunningDiffStats:
    """Streaming tensor difference statistics."""

    def __init__(self) -> None:
        self.numel = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.ref_sq = 0.0
        self.other_sq = 0.0
        self.dot = 0.0
        self.max_abs = 0.0

    def update(self, ref: np.ndarray | torch.Tensor, other: np.ndarray | torch.Tensor) -> None:
        ref_t = torch.as_tensor(ref, dtype=torch.float32)
        other_t = torch.as_tensor(other, dtype=torch.float32)
        if ref_t.shape != other_t.shape:
            raise ValueError(f"Shape mismatch: {tuple(ref_t.shape)} vs {tuple(other_t.shape)}")

        diff = other_t - ref_t
        self.numel += ref_t.numel()
        self.sum_abs += float(diff.abs().sum().item())
        self.sum_sq += float(diff.square().sum().item())
        self.ref_sq += float(ref_t.square().sum().item())
        self.other_sq += float(other_t.square().sum().item())
        self.dot += float((ref_t * other_t).sum().item())
        self.max_abs = max(self.max_abs, float(diff.abs().max().item()))

    def to_dict(self, prefix: str) -> dict[str, float]:
        denom = max(self.numel, 1)
        rmse = (self.sum_sq / denom) ** 0.5
        ref_norm = self.ref_sq**0.5
        other_norm = self.other_sq**0.5
        cosine = self.dot / max(ref_norm * other_norm, 1e-12)
        rel_l2 = (self.sum_sq**0.5) / max(ref_norm, 1e-12)
        return {
            f"{prefix}_mae": self.sum_abs / denom,
            f"{prefix}_mse": self.sum_sq / denom,
            f"{prefix}_rmse": rmse,
            f"{prefix}_rel_l2": rel_l2,
            f"{prefix}_cosine": cosine,
            f"{prefix}_max_abs": self.max_abs,
            f"{prefix}_numel": float(self.numel),
        }


def sample_task_id(sample: dict[str, Any]) -> int | None:
    value = sample.get("task_id")
    if value is None:
        return None
    return int(value)


def sample_task_label(sample: dict[str, Any]) -> str:
    task_id = sample_task_id(sample)
    if task_id is None:
        return "all_samples"
    return f"task_{task_id:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze layer-wise DuQuant drift on LIBERO data.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path or HF repo id.")
    parser.add_argument(
        "--data-source",
        default="libero",
        choices=["libero", "dataset"],
        help="Use fixed observations from LIBERO simulator or from a LeRobot dataset.",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(Path(os.environ.get("LIBERO_ROOT", str(Path.home() / "LIBERO"))) / "datasets" / "lerobot_libero_10"),
        help="LeRobot-format dataset path.",
    )
    parser.add_argument(
        "--task-suite-name",
        default="libero_10",
        help="LIBERO task suite used when --data-source=libero.",
    )
    parser.add_argument(
        "--libero-num-trials-per-task",
        type=int,
        default=5,
        help="Max LIBERO trials to traverse per task while gathering observations.",
    )
    parser.add_argument(
        "--libero-num-steps-wait",
        type=int,
        default=10,
        help="Initial no-op steps in LIBERO to let objects settle.",
    )
    parser.add_argument(
        "--libero-sampling-mode",
        default="sequential",
        choices=["sequential", "one_per_task"],
        help="How LIBERO observations are gathered. one_per_task collects at most one sample per task.",
    )
    parser.add_argument(
        "--libero-resolution",
        type=int,
        default=256,
        help="Camera resolution for LIBERO observations.",
    )
    parser.add_argument(
        "--data-config",
        default="examples.Libero.custom_data_config:LiberoDataConfig",
        help="Data config used to build the policy and dataset modality config.",
    )
    parser.add_argument(
        "--embodiment-tag",
        default="new_embodiment",
        help="Embodiment tag used by the policy and dataset.",
    )
    parser.add_argument(
        "--video-backend",
        default="torchvision_av",
        choices=["torchcodec", "decord", "torchvision_av"],
        help="Video backend for LeRobot dataset loading.",
    )
    parser.add_argument(
        "--denoising-steps",
        type=int,
        default=8,
        help="Number of action denoising steps for inference.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of dataset steps to replay.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=0,
        help="Optional fixed stride between samples. 0 means uniform spacing across the dataset.",
    )
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="Start offset into the dataset steps.",
    )
    parser.add_argument(
        "--modes",
        default="individual,cumulative",
        help="Comma-separated list from: individual,cumulative,all.",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=0,
        help="Limit the number of target layers to analyze. 0 means all matched layers.",
    )
    parser.add_argument(
        "--start-layer",
        type=int,
        default=0,
        help="Start index within the matched target layer list.",
    )
    parser.add_argument(
        "--scenario-start",
        type=int,
        default=0,
        help="Start index within the full scenario list after modes are expanded.",
    )
    parser.add_argument(
        "--scenario-count",
        type=int,
        default=0,
        help="How many scenarios to run from --scenario-start. 0 means all remaining.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the full-precision baseline summary row for this shard.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Only resolve target layers and scenario plan, then exit.",
    )
    parser.add_argument(
        "--include-regex",
        default=DEFAULT_INCLUDE_REGEX,
        help="Regex for selecting quantization target linear layers.",
    )
    parser.add_argument(
        "--exclude-regex",
        default=DEFAULT_EXCLUDE_REGEX,
        help="Regex for excluding target linear layers.",
    )
    parser.add_argument("--scope", default="", help="Optional module name prefix filter.")
    parser.add_argument("--wbits", type=int, default=4, help="Weight quantization bits.")
    parser.add_argument("--abits", type=int, default=8, help="Activation quantization bits.")
    parser.add_argument("--block-size", type=int, default=64, help="DuQuant block size.")
    parser.add_argument(
        "--block-out-size",
        type=int,
        default=64,
        help="DuQuant output block size.",
    )
    parser.add_argument(
        "--permute",
        type=int,
        default=0,
        choices=[0, 1],
        help="Enable DuQuant permutation.",
    )
    parser.add_argument(
        "--row-rot",
        default="restore",
        choices=["0", "restore", "propagate"],
        help="DuQuant row rotation mode.",
    )
    parser.add_argument("--act-pct", type=float, default=99.9, help="Activation percentile.")
    parser.add_argument(
        "--calib-steps",
        type=int,
        default=32,
        help="DuQuant activation calibration steps.",
    )
    parser.add_argument("--lambda-smooth", type=float, default=0.15, help="DuQuant lambda smoothing.")
    parser.add_argument(
        "--packdir",
        default=None,
        help="Optional DuQuant pack dir. Reusing a pack dir speeds up repeated runs.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed. Each sample uses seed + sample_idx.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/layerwise_quant_analysis",
        help="Directory for metrics and metadata.",
    )
    parser.add_argument(
        "--split-by-task",
        action="store_true",
        help="Also write task-specific summary rows and per-layer metrics.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def classify_layer(name: str) -> str:
    if "backbone.eagle_model.language_model" in name:
        return "llm"
    if name.startswith("action_head.model."):
        return "dit"
    return "other"


def ensure_libero_runtime() -> None:
    candidates = []
    extra = os.environ.get("QUANTVLA_EXTRA_SITE_DIRS", "")
    if extra:
        candidates.extend([path for path in extra.split(":") if path])
    conda_root = os.environ.get("CONDA_ROOT", str(Path.home() / "miniconda3"))
    default_env = os.environ.get("AWESOME_QVLA_CONDA_ENV", os.environ.get("QUANTVLA_CONDA_ENV", "awesome_quant_vla"))
    quant_env = os.environ.get("QUANTVLA_CONDA_ENV", default_env)
    libero_env = os.environ.get("LIBERO_CONDA_ENV", default_env)
    candidates.extend(
        [
            f"{conda_root}/envs/{quant_env}/lib/python3.10/site-packages",
            f"{conda_root}/envs/{libero_env}/lib/python3.10/site-packages",
        ]
    )
    for path in candidates:
        if path and Path(path).exists():
            site.addsitedir(path)


def choose_sample_indices(total_steps: int, num_samples: int, stride: int, offset: int) -> list[int]:
    if total_steps <= 0:
        return []
    offset = max(offset, 0)
    if offset >= total_steps:
        raise ValueError(f"sample_offset={offset} exceeds dataset length {total_steps}")

    if stride > 0:
        indices = list(range(offset, total_steps, stride))[:num_samples]
    else:
        remaining = total_steps - offset
        if num_samples >= remaining:
            indices = list(range(offset, total_steps))
        else:
            positions = np.linspace(offset, total_steps - 1, num=num_samples, dtype=int)
            indices = sorted(set(int(i) for i in positions))

    if not indices:
        raise RuntimeError("No sample indices were selected.")
    return indices


def flatten_action_dict(action_dict: dict[str, Any], action_keys: Sequence[str]) -> np.ndarray:
    parts = []
    for key in action_keys:
        value = np.asarray(action_dict[key], dtype=np.float32)
        if value.ndim == 1:
            value = value[:, None]
        parts.append(value)
    return np.concatenate(parts, axis=-1)


def _convert_libero_observation(obs: dict[str, Any], language: str) -> dict[str, np.ndarray]:
    from examples.Libero.eval.utils import get_libero_image, quat2axisangle

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


def load_libero_samples(args: argparse.Namespace, data_config):
    ensure_libero_runtime()
    from libero.libero import benchmark

    from examples.Libero.eval.utils import get_libero_dummy_action, get_libero_env

    samples = []
    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite_name not in benchmark_dict:
        raise ValueError(
            f"Unknown LIBERO task suite '{args.task_suite_name}'. "
            f"Available: {sorted(benchmark_dict.keys())}"
        )

    # Hold-out calibration support: start init_state index at offset.
    # Read from env var so we don't have to add CLI arg to every caller.
    init_offset = int(os.environ.get("GR00T_LIBERO_INIT_OFFSET", "0"))
    # Optional task filter: GR00T_LIBERO_TASK_FILTER="5" or "0,2,5" → restrict
    # calibration to a subset of tasks (e.g. for per-task diagnostic builds).
    task_filter_env = os.environ.get("GR00T_LIBERO_TASK_FILTER", "").strip()
    task_filter = (
        {int(x) for x in task_filter_env.split(",") if x.strip()}
        if task_filter_env
        else None
    )

    task_suite = benchmark_dict[args.task_suite_name]()
    for task_id in range(task_suite.n_tasks):
        if len(samples) >= args.num_samples:
            break
        if task_filter is not None and task_id not in task_filter:
            continue
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, _ = get_libero_env(task, resolution=args.libero_resolution)
        try:
            max_avail = max(0, len(initial_states) - init_offset)
            max_trials = min(args.libero_num_trials_per_task, max_avail)
            for offset_in_window in range(max_trials):
                trial_idx = init_offset + offset_in_window
                env.reset()
                obs = env.set_init_state(initial_states[trial_idx])
                for wait_step in range(args.libero_num_steps_wait):
                    obs, _, done, _ = env.step(get_libero_dummy_action())
                    if done:
                        break

                step_idx = 0
                # In task-filter mode, treat 'one_per_task' as 'one_per_trial'
                # so we collect num_samples init_states of the same task instead
                # of stopping after 1.
                effective_mode = args.libero_sampling_mode
                if task_filter is not None and effective_mode == "one_per_task":
                    effective_mode = "one_per_trial"
                if effective_mode == "one_per_task" or effective_mode == "one_per_trial":
                    converted = _convert_libero_observation(obs, task.language)
                    samples.append(
                        {
                            "dataset_index": int(len(samples)),
                            "trajectory_id": int(task_id),
                            "base_index": int(step_idx),
                            "seed": int(args.seed + len(samples)),
                            "obs": converted,
                            "gt_action": None,
                            "task_suite": args.task_suite_name,
                            "task_id": int(task_id),
                            "trial_idx": int(trial_idx),
                            "language": task.language,
                        }
                    )
                    if effective_mode == "one_per_task":
                        # Original behaviour: 1 sample per task across many tasks
                        break
                    # 'one_per_trial': continue to next init_state for same task
                    continue

                while len(samples) < args.num_samples:
                    converted = _convert_libero_observation(obs, task.language)
                    samples.append(
                        {
                            "dataset_index": int(len(samples)),
                            "trajectory_id": int(task_id),
                            "base_index": int(step_idx),
                            "seed": int(args.seed + len(samples)),
                            "obs": converted,
                            "gt_action": None,
                            "task_suite": args.task_suite_name,
                            "task_id": int(task_id),
                            "trial_idx": int(trial_idx),
                            "language": task.language,
                        }
                    )
                    if len(samples) >= args.num_samples:
                        break

                    obs, _, done, _ = env.step(get_libero_dummy_action())
                    step_idx += 1
                    if done:
                        break

                if len(samples) >= args.num_samples:
                    break
            if len(samples) >= args.num_samples:
                break
        finally:
            env.close()

    if len(samples) < args.num_samples:
        raise RuntimeError(
            f"Only collected {len(samples)} LIBERO observations, fewer than requested {args.num_samples}"
        )
    return f"LIBERO-{args.task_suite_name}", data_config, samples


def load_dataset_samples(args: argparse.Namespace, data_config):
    embodiment_tag = EmbodimentTag(args.embodiment_tag)
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=data_config.modality_config(),
        embodiment_tag=embodiment_tag,
        video_backend=args.video_backend,
    )
    sample_indices = choose_sample_indices(
        total_steps=len(dataset),
        num_samples=args.num_samples,
        stride=args.sample_stride,
        offset=args.sample_offset,
    )
    samples = []
    for sample_index in sample_indices:
        traj_id, base_index = dataset.all_steps[sample_index]
        obs = dataset.get_step_data(traj_id, base_index)
        gt_action = flatten_action_dict(obs, data_config.action_keys)
        samples.append(
            {
                "dataset_index": int(sample_index),
                "trajectory_id": int(traj_id),
                "base_index": int(base_index),
                "seed": int(args.seed + len(samples)),
                "obs": obs,
                "gt_action": gt_action,
            }
        )
    return dataset, data_config, samples


def load_samples(args: argparse.Namespace, data_config):
    if args.data_source == "libero":
        return load_libero_samples(args, data_config)
    return load_dataset_samples(args, data_config)


@contextmanager
def isolated_quant_env(new_env: dict[str, str] | None = None):
    prefixes = ("GR00T_DUQUANT_", "GR00T_ATM_", "GR00T_OHB_")
    touched = {k: v for k, v in os.environ.items() if k.startswith(prefixes)}
    for key in list(os.environ.keys()):
        if key.startswith(prefixes):
            os.environ.pop(key, None)
    try:
        if new_env:
            os.environ.update(new_env)
        yield
    finally:
        for key in list(os.environ.keys()):
            if key.startswith(prefixes):
                os.environ.pop(key, None)
        os.environ.update(touched)


def build_duquant_env(args: argparse.Namespace, layer_names: Sequence[str]) -> dict[str, str]:
    env = {
        "GR00T_DUQUANT_SCOPE": args.scope,
        "GR00T_DUQUANT_INCLUDE": args.include_regex,
        "GR00T_DUQUANT_EXCLUDE": args.exclude_regex,
        "GR00T_DUQUANT_WBITS_DEFAULT": str(args.wbits),
        "GR00T_DUQUANT_ABITS": str(args.abits),
        "GR00T_DUQUANT_BLOCK": str(args.block_size),
        "GR00T_DUQUANT_BLOCK_OUT": str(args.block_out_size),
        "GR00T_DUQUANT_PERMUTE": str(args.permute),
        "GR00T_DUQUANT_ROW_ROT": args.row_rot,
        "GR00T_DUQUANT_ACT_PCT": str(args.act_pct),
        "GR00T_DUQUANT_CALIB_STEPS": str(args.calib_steps),
        "GR00T_DUQUANT_LS": str(args.lambda_smooth),
        "GR00T_DUQUANT_LAYERS": ",".join(layer_names),
    }
    if args.packdir:
        env["GR00T_DUQUANT_PACKDIR"] = args.packdir
    return env


def load_policy(args: argparse.Namespace, data_config, quantized_layers: Sequence[str] | None) -> Gr00tPolicy:
    quant_env = None
    if quantized_layers:
        quant_env = build_duquant_env(args, quantized_layers)
    with isolated_quant_env(quant_env):
        policy = Gr00tPolicy(
            model_path=args.checkpoint,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            embodiment_tag=EmbodimentTag(args.embodiment_tag),
            denoising_steps=args.denoising_steps,
            device=args.device,
        )
    return policy


def get_target_layers(args: argparse.Namespace, data_config) -> list[dict[str, Any]]:
    policy = load_policy(args, data_config, quantized_layers=None)
    try:
        targets = select_targets(
            policy.model,
            include_regex=args.include_regex,
            exclude_regex=args.exclude_regex,
            scope_prefix=args.scope or None,
            whitelist=None,
            blacklist=None,
        )
        layer_names = [name for name, _ in targets]
    finally:
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selected = layer_names[args.start_layer :]
    if args.max_layers > 0:
        selected = selected[: args.max_layers]

    return [
        {
            "layer_index": idx,
            "layer_name": name,
            "family": classify_layer(name),
        }
        for idx, name in enumerate(selected, start=args.start_layer)
    ]


def build_scenarios(layer_names: Sequence[str], modes: set[str]) -> list[Scenario]:
    scenarios: list[Scenario] = []
    if "individual" in modes:
        for idx, layer_name in enumerate(layer_names):
            scenarios.append(
                Scenario(
                    name=f"individual_{idx:03d}",
                    mode="individual",
                    quantized_layers=[layer_name],
                    focus_layer=layer_name,
                )
            )
    if "cumulative" in modes:
        running: list[str] = []
        for idx, layer_name in enumerate(layer_names):
            running.append(layer_name)
            scenarios.append(
                Scenario(
                    name=f"cumulative_{idx + 1:03d}",
                    mode="cumulative",
                    quantized_layers=list(running),
                    focus_layer=layer_name,
                )
            )
    if "all" in modes and layer_names:
        scenarios.append(
            Scenario(
                name="all_selected",
                mode="all",
                quantized_layers=list(layer_names),
                focus_layer=layer_names[-1],
            )
        )
    return scenarios


def slice_scenarios(
    scenarios: Sequence[Scenario],
    start: int,
    count: int,
) -> list[Scenario]:
    start = max(start, 0)
    if start >= len(scenarios):
        return []
    if count <= 0:
        return list(scenarios[start:])
    return list(scenarios[start : start + count])


def get_named_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(f"Module not found: {name}")
    return modules[name]


def capture_policy_outputs(
    policy: Gr00tPolicy,
    samples: Sequence[dict[str, Any]],
    layer_names: Sequence[str],
    action_keys: Sequence[str],
) -> tuple[list[np.ndarray], list[dict[str, torch.Tensor]]]:
    module_map = {name: get_named_module(policy.model, name) for name in layer_names}
    action_outputs: list[np.ndarray] = []
    layer_outputs: list[dict[str, torch.Tensor]] = []
    current_capture: dict[str, torch.Tensor] = {}

    def make_hook(layer_name: str):
        def hook(_module, _inputs, output):
            current_capture[layer_name] = output.detach().to(torch.float32, copy=True).cpu()
        return hook

    handles = [module.register_forward_hook(make_hook(name)) for name, module in module_map.items()]
    try:
        for sample in tqdm(samples, desc="Capturing outputs", leave=False):
            current_capture = {}
            seed_everything(sample["seed"])
            action_dict = policy.get_action(sample["obs"])
            action_outputs.append(flatten_action_dict(action_dict, action_keys))
            layer_outputs.append(current_capture)
    finally:
        for handle in handles:
            handle.remove()
    return action_outputs, layer_outputs


def get_quant_weight(module: DuQuantLinear) -> torch.Tensor:
    module._maybe_update_weight_cache()
    if module.weight_bits <= 0:
        return module._W_t.detach().to(torch.float32).cpu()
    if module._weight_quantized_cached:
        return module._W_t_quantized.detach().to(torch.float32).cpu()
    quant_weight = fake_quantize_sym(
        module._W_t,
        module._w_scales[:, None],
        module.weight_bits,
        label="weight_analysis",
    )
    return quant_weight.detach().to(torch.float32).cpu()


def compute_weight_metrics(policy: Gr00tPolicy, layer_names: Sequence[str]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for layer_name in layer_names:
        module = get_named_module(policy.model, layer_name)
        if not isinstance(module, DuQuantLinear):
            continue
        module._maybe_update_weight_cache()
        ref_weight = module._W_t.detach().to(torch.float32).cpu()
        quant_weight = get_quant_weight(module)
        stats = RunningDiffStats()
        stats.update(ref_weight, quant_weight)
        values = stats.to_dict("w")
        values["w_ref_domain"] = "duquant_transformed_weight"
        values["wbits"] = float(module.weight_bits)
        values["abits"] = float(module.cfg.act_bits)
        metrics[layer_name] = values
    return metrics


def scenario_summary_row(
    scenario: Scenario,
    fp_vs_gt: RunningDiffStats | None,
    quant_vs_gt: RunningDiffStats | None = None,
    quant_vs_fp: RunningDiffStats | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scenario": scenario.name,
        "mode": scenario.mode,
        "num_quantized_layers": len(scenario.quantized_layers),
        "focus_layer": scenario.focus_layer or "",
    }
    if fp_vs_gt is not None:
        row["fp_gt_mse"] = fp_vs_gt.to_dict("fp_gt")["fp_gt_mse"]
        row["fp_gt_rmse"] = fp_vs_gt.to_dict("fp_gt")["fp_gt_rmse"]
    if quant_vs_gt is not None:
        q_gt = quant_vs_gt.to_dict("quant_gt")
        row.update(q_gt)
        if "fp_gt_mse" in row:
            row["delta_gt_mse_vs_fp"] = q_gt["quant_gt_mse"] - row["fp_gt_mse"]
    if quant_vs_fp is not None:
        row.update(quant_vs_fp.to_dict("action_fp"))
    return row


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = {mode.strip() for mode in args.modes.split(",") if mode.strip()}
    invalid_modes = modes - {"individual", "cumulative", "all"}
    if invalid_modes:
        raise ValueError(f"Unsupported modes: {sorted(invalid_modes)}")

    data_config = load_data_config(args.data_config)
    target_layers = get_target_layers(args, data_config)
    target_layer_names = [row["layer_name"] for row in target_layers]
    all_scenarios = build_scenarios(target_layer_names, modes)
    if not all_scenarios:
        raise RuntimeError("No scenarios were generated. Check --modes and target layer selection.")
    scenarios = slice_scenarios(all_scenarios, args.scenario_start, args.scenario_count)
    if not scenarios and not args.plan_only:
        raise RuntimeError(
            "No scenarios selected after applying --scenario-start/--scenario-count."
        )

    run_config = {
        "checkpoint": args.checkpoint,
        "data_source": args.data_source,
        "dataset_path": str(args.dataset_path),
        "task_suite_name": args.task_suite_name,
        "data_config": args.data_config,
        "embodiment_tag": args.embodiment_tag,
        "video_backend": args.video_backend,
        "denoising_steps": args.denoising_steps,
        "modes": sorted(modes),
        "start_layer": args.start_layer,
        "max_layers": args.max_layers,
        "scenario_start": args.scenario_start,
        "scenario_count": args.scenario_count,
        "skip_baseline": args.skip_baseline,
        "plan_only": args.plan_only,
        "quant_config": {
            "wbits": args.wbits,
            "abits": args.abits,
            "block_size": args.block_size,
            "block_out_size": args.block_out_size,
            "permute": args.permute,
            "row_rot": args.row_rot,
            "act_pct": args.act_pct,
            "calib_steps": args.calib_steps,
            "lambda_smooth": args.lambda_smooth,
            "packdir": args.packdir,
            "include_regex": args.include_regex,
            "exclude_regex": args.exclude_regex,
            "scope": args.scope,
        },
        "libero_config": {
            "num_trials_per_task": args.libero_num_trials_per_task,
            "num_steps_wait": args.libero_num_steps_wait,
            "sampling_mode": args.libero_sampling_mode,
            "resolution": args.libero_resolution,
        },
        "split_by_task": args.split_by_task,
    }
    write_json(output_dir / "run_config.json", run_config)
    write_json(output_dir / "target_layers.json", target_layers)
    scenario_plan = [
        {
            "scenario_index": idx,
            "scenario": scenario.name,
            "mode": scenario.mode,
            "num_quantized_layers": len(scenario.quantized_layers),
            "focus_layer": scenario.focus_layer or "",
        }
        for idx, scenario in enumerate(all_scenarios)
    ]
    write_json(output_dir / "scenario_plan.json", scenario_plan)

    print(f"[Layerwise] Target layers: {len(target_layer_names)}")
    print(f"[Layerwise] Total scenarios: {len(all_scenarios)}")
    print(f"[Layerwise] Selected scenarios: {len(scenarios)}")

    if args.plan_only:
        write_json(
            output_dir / "plan_summary.json",
            {
                "num_target_layers": len(target_layer_names),
                "num_total_scenarios": len(all_scenarios),
                "num_selected_scenarios": len(scenarios),
                "scenario_start": args.scenario_start,
                "scenario_count": args.scenario_count,
            },
        )
        return

    dataset, _, samples = load_samples(args, data_config)
    run_config["num_samples"] = len(samples)
    write_json(output_dir / "run_config.json", run_config)
    write_json(
        output_dir / "samples.json",
        [
            {
                "dataset_index": sample["dataset_index"],
                "trajectory_id": sample["trajectory_id"],
                "base_index": sample["base_index"],
                "seed": sample["seed"],
            }
            for sample in samples
        ],
    )

    print(f"[Layerwise] Dataset: {dataset}")
    print(f"[Layerwise] Selected samples: {len(samples)}")
    print(f"[Layerwise] Scenarios in this run: {len(scenarios)}")

    has_gt = all(sample.get("gt_action") is not None for sample in samples)
    task_samples: dict[int, dict[str, Any]] = {}
    if args.split_by_task:
        for sample in samples:
            task_id = sample_task_id(sample)
            if task_id is not None and task_id not in task_samples:
                task_samples[task_id] = sample

    fp_vs_gt = RunningDiffStats() if has_gt else None
    scenario_rows: list[dict[str, Any]] = []
    task_scenario_rows: list[dict[str, Any]] = []
    if not args.skip_baseline:
        print("[Layerwise] Capturing full-precision baseline...")
        fp_policy = load_policy(args, data_config, quantized_layers=None)
        try:
            fp_actions, _ = capture_policy_outputs(
                fp_policy,
                samples,
                layer_names=[],
                action_keys=data_config.action_keys,
            )
        finally:
            del fp_policy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if fp_vs_gt is not None:
            for sample, fp_action in zip(samples, fp_actions):
                fp_vs_gt.update(sample["gt_action"], fp_action)

        scenario_rows.append(
            {
                "scenario": "full_precision",
                "mode": "baseline",
                "num_quantized_layers": 0,
                "focus_layer": "",
                **(fp_vs_gt.to_dict("fp_gt") if fp_vs_gt is not None else {}),
            }
        )
        if args.split_by_task:
            for task_id, sample in sorted(task_samples.items()):
                row = {
                    "scenario": "full_precision",
                    "mode": "baseline",
                    "num_quantized_layers": 0,
                    "focus_layer": "",
                    "task_id": task_id,
                    "task_label": sample_task_label(sample),
                    "task_suite": sample.get("task_suite", args.task_suite_name),
                    "language": sample.get("language", ""),
                }
                task_scenario_rows.append(row)
    else:
        print("[Layerwise] Skipping baseline for this shard.")

    per_layer_rows: list[dict[str, Any]] = []
    task_per_layer_rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        print(
            f"[Layerwise] Scenario {scenario.name}: "
            f"{len(scenario.quantized_layers)} quantized layer(s)"
        )

        print("[Layerwise] Capturing FP reference outputs for quantized layers...")
        fp_policy = load_policy(args, data_config, quantized_layers=None)
        try:
            fp_actions_ref, fp_layer_outputs = capture_policy_outputs(
                fp_policy,
                samples,
                layer_names=scenario.quantized_layers,
                action_keys=data_config.action_keys,
            )
        finally:
            del fp_policy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("[Layerwise] Capturing quantized outputs...")
        quant_policy = load_policy(args, data_config, quantized_layers=scenario.quantized_layers)
        try:
            quant_actions, quant_layer_outputs = capture_policy_outputs(
                quant_policy,
                samples,
                layer_names=scenario.quantized_layers,
                action_keys=data_config.action_keys,
            )
            weight_metrics = compute_weight_metrics(quant_policy, scenario.quantized_layers)
        finally:
            del quant_policy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        action_vs_fp = RunningDiffStats()
        quant_vs_gt = RunningDiffStats() if has_gt else None
        task_action_vs_fp: dict[int, RunningDiffStats] = {}
        task_quant_vs_gt: dict[int, RunningDiffStats] = {}
        for sample, fp_action, quant_action in zip(samples, fp_actions_ref, quant_actions):
            action_vs_fp.update(fp_action, quant_action)
            if quant_vs_gt is not None:
                quant_vs_gt.update(sample["gt_action"], quant_action)
            if args.split_by_task:
                task_id = sample_task_id(sample)
                if task_id is not None:
                    task_action_vs_fp.setdefault(task_id, RunningDiffStats()).update(fp_action, quant_action)
                    if quant_vs_gt is not None:
                        task_quant_vs_gt.setdefault(task_id, RunningDiffStats()).update(
                            sample["gt_action"], quant_action
                        )

        if args.skip_baseline:
            fp_vs_gt = RunningDiffStats() if has_gt else None
            if fp_vs_gt is not None:
                for sample, fp_action in zip(samples, fp_actions_ref):
                    fp_vs_gt.update(sample["gt_action"], fp_action)

        scenario_rows.append(scenario_summary_row(scenario, fp_vs_gt, quant_vs_gt, action_vs_fp))

        layer_stats = {name: RunningDiffStats() for name in scenario.quantized_layers}
        task_layer_stats: dict[int, dict[str, RunningDiffStats]] = {}
        for sample, fp_capture, quant_capture in zip(samples, fp_layer_outputs, quant_layer_outputs):
            task_id = sample_task_id(sample) if args.split_by_task else None
            if task_id is not None and task_id not in task_layer_stats:
                task_layer_stats[task_id] = {
                    name: RunningDiffStats() for name in scenario.quantized_layers
                }
            for layer_name in scenario.quantized_layers:
                if layer_name not in fp_capture or layer_name not in quant_capture:
                    raise KeyError(f"Missing captured output for layer {layer_name}")
                layer_stats[layer_name].update(fp_capture[layer_name], quant_capture[layer_name])
                if task_id is not None:
                    task_layer_stats[task_id][layer_name].update(
                        fp_capture[layer_name], quant_capture[layer_name]
                    )

        for layer_name in scenario.quantized_layers:
            layer_row = {
                "scenario": scenario.name,
                "mode": scenario.mode,
                "layer_name": layer_name,
                "family": classify_layer(layer_name),
                "quantized": True,
                "focus_layer": scenario.focus_layer == layer_name,
                "num_quantized_layers": len(scenario.quantized_layers),
            }
            layer_row.update(weight_metrics.get(layer_name, {}))
            layer_row.update(layer_stats[layer_name].to_dict("wx"))
            per_layer_rows.append(layer_row)

        if args.split_by_task:
            for task_id, sample in sorted(task_samples.items()):
                task_row = scenario_summary_row(
                    scenario,
                    None,
                    task_quant_vs_gt.get(task_id),
                    task_action_vs_fp.get(task_id),
                )
                task_row.update(
                    {
                        "task_id": task_id,
                        "task_label": sample_task_label(sample),
                        "task_suite": sample.get("task_suite", args.task_suite_name),
                        "language": sample.get("language", ""),
                    }
                )
                task_scenario_rows.append(task_row)

                if task_id not in task_layer_stats:
                    continue
                for layer_name in scenario.quantized_layers:
                    layer_row = {
                        "scenario": scenario.name,
                        "mode": scenario.mode,
                        "layer_name": layer_name,
                        "family": classify_layer(layer_name),
                        "quantized": True,
                        "focus_layer": scenario.focus_layer == layer_name,
                        "num_quantized_layers": len(scenario.quantized_layers),
                        "task_id": task_id,
                        "task_label": sample_task_label(sample),
                        "task_suite": sample.get("task_suite", args.task_suite_name),
                        "language": sample.get("language", ""),
                    }
                    layer_row.update(weight_metrics.get(layer_name, {}))
                    layer_row.update(task_layer_stats[task_id][layer_name].to_dict("wx"))
                    task_per_layer_rows.append(layer_row)

    write_csv(output_dir / "scenario_summary.csv", scenario_rows)
    write_jsonl(output_dir / "per_layer_metrics.jsonl", per_layer_rows)
    write_json(output_dir / "scenario_summary.json", scenario_rows)
    if args.split_by_task:
        write_csv(output_dir / "task_scenario_summary.csv", task_scenario_rows)
        write_json(output_dir / "task_scenario_summary.json", task_scenario_rows)
        write_jsonl(output_dir / "task_per_layer_metrics.jsonl", task_per_layer_rows)
        write_json(
            output_dir / "task_manifest.json",
            [
                {
                    "task_id": task_id,
                    "task_label": sample_task_label(sample),
                    "task_suite": sample.get("task_suite", args.task_suite_name),
                    "language": sample.get("language", ""),
                }
                for task_id, sample in sorted(task_samples.items())
            ],
        )

    print(f"[Layerwise] Wrote results to {output_dir}")
    print(f"[Layerwise] Scenario summary: {output_dir / 'scenario_summary.csv'}")
    print(f"[Layerwise] Per-layer metrics: {output_dir / 'per_layer_metrics.jsonl'}")


if __name__ == "__main__":
    main()
