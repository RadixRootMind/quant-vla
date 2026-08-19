"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os

import imageio
import numpy as np
import tensorflow as tf

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)

import sys 
sys.path.append("../VLN-CE")
from experiments.robot.r2r.config.default import get_config
# from vlnce_baselines.config.default import get_config
import random
import torch

import habitat



def get_env_config(env_config_path, seed):

    env_config = get_config(env_config_path)
    # logger.info(f"env_config: {env_config}")
    
    split = env_config.EVAL.SPLIT
    env_config.defrost()

    env_config.TASK_CONFIG.SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    # turn off RGBD rendering as neither RandomAgent nor HandcraftedAgent use it.
    # config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = []
    # config.TASK_CONFIG.TASK.SENSORS = []
    env_config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    env_config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    env_config.TASK_CONFIG.DATASET.SPLIT = split
    env_config.TASK_CONFIG.TASK.NDTW.SPLIT = split
    env_config.TASK_CONFIG.TASK.SDTW.SPLIT = split
    env_config.freeze()

    return env_config


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_navigation_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["rgb"]
    # img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den



def split_and_sample_dataset(config, num_splits=4, episodes_per_split=25):
    """Split dataset and sample top episodes from each split"""
    
    # Create dataset using make_dataset
    dataset = habitat.make_dataset(
        id_dataset=config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET
    )
    
    # Get splits
    splits = dataset.get_splits(num_splits=num_splits)
    # Create split datasets and sample
    final_episode_ids = []
    for split_indices in splits:
        # final_episodes += split_indices.get_episode(list(range(25)))
        episodes = split_indices.get_episodes(list(range(episodes_per_split)))
        final_episode_ids.extend([ep.episode_id for ep in episodes])
    # Create a filter function that only keeps episodes with matching IDs
    def filter_fn(episode):
        return episode.episode_id in final_episode_ids
    # Create new filtered dataset
    filtered_dataset = dataset.filter_episodes(filter_fn)
    
    return filtered_dataset
