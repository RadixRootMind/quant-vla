import collections
import os
import pprint
import json
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import tqdm
import tyro
from libero.libero import benchmark

from examples.Libero.eval.utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    normalize_gripper_action,
    quat2axisangle,
    save_rollout_video,
)

DEFAULT_LOG_DIR = "/tmp/logs"
os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)  # ensures directory exists


def summarize_obs(obs_dict):
    summary = {}
    for k, v in obs_dict.items():
        if isinstance(v, torch.Tensor):
            summary[k] = {"shape": tuple(v.shape), "dtype": v.dtype, "device": v.device}
        elif isinstance(v, np.ndarray):
            summary[k] = {"shape": v.shape, "dtype": v.dtype}
        else:
            summary[k] = type(v).__name__
    pprint.pprint(summary)


def show_obs_images_cv2(new_obs):
    # remove batch dim
    img_agent = new_obs["video.image"][0]
    img_wrist = new_obs["video.wrist_image"][0]

    # convert RGB -> BGR for OpenCV
    img_agent_bgr = cv2.cvtColor(img_agent, cv2.COLOR_RGB2BGR)
    img_wrist_bgr = cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR)

    # show in separate windows
    cv2.imshow("Agent View", img_agent_bgr)
    cv2.imshow("Wrist View", img_wrist_bgr)
    cv2.waitKey(1)


@dataclass
class GenerateConfig:
    # fmt: off
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 5                     # Number of rollouts per task
    init_offset: int = 0                             # Skip the first N init_states (used to split a task across shards)
    #################################################################################################################
    # fmt: on
    """Port to connect to."""
    port: int = 5555
    """Headless mode (no GUI)."""
    headless: bool = False
    """Run only the specified task indices (overrides order if provided)."""
    task_ids: list[int] | None = None
    """Run tasks in this explicit order."""
    task_order: list[int] | None = None
    """Directory for text logs."""
    log_dir: str = DEFAULT_LOG_DIR
    """Optional suffix appended to log / rollout names to keep shards separate."""
    log_suffix: str = ""
    """Optional directory to save rollout mp4s."""
    rollout_dir: str | None = None
    """Optional path to dump a machine-readable summary JSON."""
    summary_json: str | None = None
    """Whether to save rollout mp4 videos."""
    save_videos: bool = True
    """Policy backend to query."""
    policy_backend: str = "groot_zmq"
    """How to decode the action payload returned by the inference server."""
    policy_output_format: str = "gr00t_dict"
    """Key to read when the server returns a dict-wrapped vector action."""
    response_action_key: str = "action"
    """Resize used by openpi's LIBERO policy inputs."""
    resize_size: int = 224
    """Number of openpi action-chunk steps to execute before replanning."""
    replan_steps: int = 5
    """Max-step table to use: groot keeps this repo's previous settings, openpi matches upstream openpi."""
    max_steps_profile: str = "groot"


class ExternalPolicy:
    """External policy wrapper for Libero environments."""

    LIBERO_CONFIG = {
        "proprio_size": 8,
        "state_key_mapping": {
            "x": 0,
            "y": 1,
            "z": 2,
            "roll": 3,
            "pitch": 4,
            "yaw": 5,
            "gripper": (6, 8),
        },
    }

    def __init__(
        self,
        host="localhost",
        port=5555,
        headless=False,
        policy_backend: str = "groot_zmq",
        policy_output_format: str = "gr00t_dict",
        response_action_key: str = "action",
        resize_size: int = 224,
        replan_steps: int = 5,
    ):
        self.policy_backend = policy_backend
        if self.policy_backend == "groot_zmq":
            from gr00t.eval.service import ExternalRobotInferenceClient

            self.policy = ExternalRobotInferenceClient(host=host, port=port)
        elif self.policy_backend == "openpi_ws":
            from openpi_client import websocket_client_policy as _websocket_client_policy

            self.policy = _websocket_client_policy.WebsocketClientPolicy(host, port)
        else:
            raise ValueError(f"Unknown policy_backend: {self.policy_backend}")
        self.config = self.LIBERO_CONFIG
        self.action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        self.headless = headless
        self.policy_output_format = policy_output_format
        self.response_action_key = response_action_key
        self.resize_size = resize_size
        self.replan_steps = replan_steps
        self.action_plan = collections.deque()

    def get_action(self, observation_dict, lang: str):
        """Get action from the external policy given observation and language instruction."""
        if self.policy_backend == "openpi_ws":
            return self._get_openpi_action(observation_dict, lang)
        obs_dict = self._process_observation(observation_dict, lang)
        # summarize_obs(obs_dict)
        action_chunk = self.policy.get_action(obs_dict)
        return self._convert_to_libero_action(action_chunk, 0)

    def reset_episode(self):
        self.action_plan.clear()

    def _process_observation(self, obs, lang: str):
        """Convert Libero observation to GR00T format."""
        xyz = obs["robot0_eef_pos"]
        rpy = quat2axisangle(obs["robot0_eef_quat"])
        gripper = obs["robot0_gripper_qpos"]
        img, wrist_img = get_libero_image(obs)
        new_obs = {
            "video.image": np.expand_dims(img, axis=0),
            "video.wrist_image": np.expand_dims(wrist_img, axis=0),
            "state.x": np.array([[xyz[0]]]),
            "state.y": np.array([[xyz[1]]]),
            "state.z": np.array([[xyz[2]]]),
            "state.roll": np.array([[rpy[0]]]),
            "state.pitch": np.array([[rpy[1]]]),
            "state.yaw": np.array([[rpy[2]]]),
            "state.gripper": np.expand_dims(gripper, axis=0),
            "annotation.human.action.task_description": [lang],
        }
        if not self.headless:
            show_obs_images_cv2(new_obs)
        return new_obs

    def _convert_to_libero_action(
        self, action_chunk: dict[str, np.array], idx: int = 0
    ) -> np.ndarray:
        """Convert an external action payload to Libero format.

        Args:
            action_chunk: Action payload returned by the server
            idx: Index of action to extract from chunk (default: 0 for first action)

        Returns:
            7-dim numpy array: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        if self.policy_output_format == "vector7":
            action_array = self._extract_vector_action(action_chunk, idx)
        else:
            action_components = [self._extract_named_component(action_chunk, key, idx) for key in self.action_keys]
            action_array = np.array(action_components, dtype=np.float32)
        action_array = normalize_gripper_action(action_array, binarize=True)
        assert len(action_array) == 7, f"Expected 7-dim action, got {len(action_array)}"
        return action_array

    def _extract_named_component(self, action_chunk, key: str, idx: int) -> float:
        if not isinstance(action_chunk, dict):
            raise TypeError(
                f"Expected dict action payload for format '{self.policy_output_format}', "
                f"got {type(action_chunk).__name__}"
            )
        for candidate in (f"action.{key}", key):
            if candidate in action_chunk:
                return float(np.atleast_1d(action_chunk[candidate])[idx])
        raise KeyError(f"Could not find action component '{key}' in server response")

    def _extract_vector_action(self, action_chunk, idx: int) -> np.ndarray:
        payload = action_chunk
        if isinstance(action_chunk, dict):
            if self.response_action_key in action_chunk:
                payload = action_chunk[self.response_action_key]
            elif "actions" in action_chunk:
                payload = action_chunk["actions"]
            elif len(action_chunk) == 1:
                payload = next(iter(action_chunk.values()))

        action_array = np.asarray(payload, dtype=np.float32)
        if action_array.ndim == 2:
            action_array = action_array[idx]
        elif action_array.ndim > 2:
            raise ValueError(
                f"Expected vector action with rank <= 2, got shape {action_array.shape}"
            )
        if action_array.shape[-1] != 7:
            raise ValueError(f"Expected vector action with final dim 7, got shape {action_array.shape}")
        return action_array

    def _get_openpi_action(self, obs, lang: str) -> np.ndarray:
        """Convert LIBERO observations to openpi's websocket policy format."""
        if not self.action_plan:
            from openpi_client import image_tools

            img, wrist_img = get_libero_image(obs)
            img = np.ascontiguousarray(img)
            wrist_img = np.ascontiguousarray(wrist_img)
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, self.resize_size, self.resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, self.resize_size, self.resize_size)
            )
            state = np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            )
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": state,
                "prompt": str(lang),
            }
            action_chunk = self.policy.infer(element)["actions"]
            if len(action_chunk) < self.replan_steps:
                raise ValueError(
                    f"openpi returned {len(action_chunk)} actions, "
                    f"but replan_steps={self.replan_steps}"
                )
            self.action_plan.extend(action_chunk[: self.replan_steps])
        return np.asarray(self.action_plan.popleft(), dtype=np.float32)


def get_max_steps(task_suite_name: str, profile: str) -> int:
    if profile == "openpi":
        if task_suite_name == "libero_spatial":
            return 220
        if task_suite_name == "libero_object":
            return 280
        if task_suite_name == "libero_goal":
            return 300
        if task_suite_name == "libero_10":
            return 520
        if task_suite_name == "libero_90":
            return 400
    elif profile == "groot":
        if task_suite_name == "libero_spatial":
            return 220
        if task_suite_name == "libero_object":
            return 280
        if task_suite_name == "libero_goal":
            return 600
        if task_suite_name == "libero_10":
            return 1000
        if task_suite_name == "libero_90":
            return 400
    raise ValueError(f"Unknown task suite/profile: {task_suite_name}/{profile}")


def eval_libero(cfg: GenerateConfig) -> None:
    os.makedirs(cfg.log_dir, exist_ok=True)
    log_stem = f"libero_eval_{cfg.task_suite_name}"
    if cfg.log_suffix:
        log_stem = f"{log_stem}_{cfg.log_suffix}"
    log_path = os.path.join(cfg.log_dir, f"{log_stem}.log")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    print(f"Task IDs: {cfg.task_ids if cfg.task_ids is not None else 'all'}")
    print(f"Log path: {log_path}")
    log_file = open(log_path, "w")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    log_file.write(f"Task IDs: {cfg.task_ids if cfg.task_ids is not None else 'all'}\n")
    log_file.write(f"Log path: {log_path}\n")

    # Decide which task indices to run
    if cfg.task_ids:
        task_indices = cfg.task_ids
    elif cfg.task_order:
        task_indices = cfg.task_order
    else:
        task_indices = list(range(num_tasks_in_suite))

    # Clamp indices to valid range and warn if needed
    task_indices = [idx for idx in task_indices if 0 <= idx < num_tasks_in_suite]

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_summaries = []
    for task_id in tqdm.tqdm(task_indices):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, resolution=256)

        policy = ExternalPolicy(
            host="localhost",
            port=cfg.port,
            headless=cfg.headless,
            policy_backend=cfg.policy_backend,
            policy_output_format=cfg.policy_output_format,
            response_action_key=cfg.response_action_key,
            resize_size=cfg.resize_size,
            replan_steps=cfg.replan_steps,
        )

        # Start episodes
        task_episodes, task_successes = 0, 0
        offset = max(0, cfg.init_offset)
        max_trials = min(cfg.num_trials_per_task, len(initial_states) - offset)
        if max_trials <= 0:
            print(f"[skip task {task_id}] init_offset={offset} leaves no trials in pool of {len(initial_states)}")
            continue
        for episode_idx in tqdm.tqdm(range(max_trials)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()
            policy.reset_episode()

            # Set initial states (offset lets us split a task across shards)
            obs = env.set_init_state(initial_states[offset + episode_idx])

            # Setup
            t = 0
            done = False
            top_view = []
            wrist_view = []
            max_steps = get_max_steps(cfg.task_suite_name, cfg.max_steps_profile)

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action())
                        t += 1
                        continue

                    # # Get preprocessed image
                    img, wrist_img = get_libero_image(obs)

                    # # Save preprocessed image for replay video
                    top_view.append(img)
                    wrist_view.append(wrist_img)

                    # Query model to get action
                    action = policy.get_action(
                        obs,
                        task.language,
                    )

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            if cfg.save_videos:
                save_rollout_video(
                    top_view,
                    wrist_view,
                    total_episodes,
                    success=done,
                    task_description=task_description,
                    log_file=log_file,
                    rollout_dir=cfg.rollout_dir,
                )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n"
            )
            log_file.flush()

        # Log final results
        task_success_rate = float(task_successes) / float(task_episodes)
        total_success_rate = float(total_successes) / float(total_episodes)
        print(f"Current task success rate: {task_success_rate}")
        print(f"Current total success rate: {total_success_rate}")
        log_file.write(
            f"Current task success rate: {task_success_rate}\n"
        )
        log_file.write(
            f"Current total success rate: {total_success_rate}\n"
        )
        log_file.flush()
        task_summaries.append(
            {
                "task_id": task_id,
                "task_description": task_description,
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": task_success_rate,
            }
        )

    # Save local log file
    log_file.close()

    summary = {
        "task_suite_name": cfg.task_suite_name,
        "task_ids": task_indices,
        "num_trials_per_task": cfg.num_trials_per_task,
        "num_steps_wait": cfg.num_steps_wait,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_success_rate": float(total_successes) / float(total_episodes)
        if total_episodes
        else 0.0,
        "task_summaries": task_summaries,
        "log_path": log_path,
        "rollout_dir": cfg.rollout_dir,
    }

    if cfg.summary_json:
        summary_dir = os.path.dirname(cfg.summary_json)
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        with open(cfg.summary_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary JSON written to {cfg.summary_json}")


if __name__ == "__main__":
    cfg = tyro.cli(GenerateConfig)
    eval_libero(cfg)
