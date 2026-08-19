import json
from collections import defaultdict

import numpy as np
from habitat import Env, logger
from habitat.config.default import Config
from habitat.core.agent import Agent
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from tqdm import tqdm, trange
from habitat_extensions.utils import generate_video, observations_to_image
from habitat.utils.visualizations.utils import append_text_to_image

from vlnce_baselines.common.environments import VLNCEInferenceEnv
import os
import gzip
import json
from PIL import Image


def get_points_region(env):
    regions = env._sim.semantic_annotations().regions
    agent_loc = env._sim.get_agent_state().position

    point_in_region = []
        
    for region in regions:
        center = region.aabb.center
        sizes = region.aabb.sizes


        min_bound = center - sizes/2
        max_bound = center + sizes/2
        
        if (min_bound[0] <= agent_loc[0] <= max_bound[0] and
            min_bound[1] <= agent_loc[1] <= max_bound[1] and
            min_bound[2] <= agent_loc[2] <= max_bound[2]):
            point_in_region.append(region.category.name())
    

    if not point_in_region:
        min_dist = float('inf')
        nearest_region = None
        
        VERTICAL_THRESHOLD = 2.0
        # dists = []
        for region in regions:
            center = region.aabb.center
            sizes = region.aabb.sizes
            min_bound = center - sizes/2
            max_bound = center + sizes/2
            
            # Check vertical (Y-axis) distance first
            y_dist = max(min_bound[1] - agent_loc[1], 0, agent_loc[1] - max_bound[1])
            if y_dist > VERTICAL_THRESHOLD:
                continue
                
            # Then check horizontal distance (X-Z plane)
            dx = max(min_bound[0] - agent_loc[0], 0, agent_loc[0] - max_bound[0])
            dz = max(min_bound[2] - agent_loc[2], 0, agent_loc[2] - max_bound[2])
            
            dist = (dx * dx + (y_dist * 5) * (y_dist * 5) + dz * dz) ** 0.5
            
            if dist < min_dist:
                min_dist = dist
                nearest_region = region
            

        if nearest_region:
            return [nearest_region.category.name()]
    return point_in_region

def evaluate_agent(config: Config) -> None:
    split = config.EVAL.SPLIT
    config.defrost()
    # turn off RGBD rendering as neither RandomAgent nor HandcraftedAgent use it.
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.TASK_CONFIG.DATASET.SPLIT = split
    config.TASK_CONFIG.TASK.NDTW.SPLIT = split
    config.TASK_CONFIG.TASK.SDTW.SPLIT = split
    config.freeze()

    env = Env(config=config.TASK_CONFIG)


    os.makedirs(config.VIDEO_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    assert config.EVAL.NONLEARNING.AGENT in [
        "RandomAgent",
        "HandcraftedAgent",
    ], "EVAL.NONLEARNING.AGENT must be either RandomAgent or HandcraftedAgent."



    # file_path = 'data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz'
    file_path = config.NDTW.GT_PATH
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        data_gt = json.load(f)

    stats = defaultdict(float)
    stats_episodes = defaultdict(dict)
    rgb_frames = defaultdict(list)
    print(len(env.episodes))
    num_episodes = min(config.EVAL.EPISODE_COUNT, len(env.episodes))
    

    episode_data = []
    all_start_end_ids = []
    episode_lengths = []
    language_data = {
        'language': {
            'ann': [],
        },
        'indx': []  # Will store (start_idx, end_idx) for each frame
    }
    # frame_region_info = []
    # frame_region_info_test = []
    current_idx = 0

    for i in trange(num_episodes):
        obs = env.reset()
        ep_id = env.current_episode.episode_id
        actions = data_gt[ep_id]['actions']
        info = env.get_metrics()  
        if config.VIDEO_OPTION:
            frame = observations_to_image(obs, info)
            frame = append_text_to_image(
                frame, 
                env.current_episode.instruction.instruction_text
            )
            rgb_frames[i].append(frame)

        # Store language data for this episode
        instruction_text = env.current_episode.instruction.instruction_text
        language_data['language']['ann'].append(instruction_text)
        
        # First frame
        info = env.get_metrics()

        start_idx = current_idx
        # region_info = []
        for j in range(len(actions)):
            frame_data = {
                'actions': np.array([actions[j]]),
                'info': info,
                'rgb_static': obs['rgb'],
                'depth_static': obs['depth']
            }
            # frame_region_info.append(get_points_region(env))
            # region_info.append(get_points_region(env))
            # Save each frame
            np.savez(
                os.path.join(config.OUTPUT_DIR, f"episode_{str(current_idx).zfill(7)}.npz"),
                **frame_data
            )
            obs = env.step(actions[j])
            info = env.get_metrics()  
            if config.VIDEO_OPTION:
                frame = observations_to_image(obs, info)
                frame = append_text_to_image(
                    frame, 
                    env.current_episode.instruction.instruction_text
                )
                rgb_frames[i].append(frame)

            current_idx += 1
        # frame_region_info_test.append(region_info)
        language_data['indx'].append((start_idx, current_idx - 1))

    

        for m, v in env.get_metrics().items():
            stats[m] += v

    stats = {k: v / num_episodes for k, v in stats.items()}

    os.makedirs(os.path.join(config.OUTPUT_DIR, 'lang_annotations'), exist_ok=True)
    # os.makedirs(os.path.join(config.OUTPUT_DIR, 'region_annotations'), exist_ok=True)

    # Save language data
    np.save(os.path.join(config.OUTPUT_DIR, 'lang_annotations/auto_lang_ann.npy'), language_data)

    # Save region data
    # with open(os.path.join(config.OUTPUT_DIR, 'region_annotations/region_ann.json'), 'w') as f:
        # json.dump(frame_region_info, f, indent=4)
        # json.dump(frame_region_info_test, f, indent=4)

    if config.VIDEO_OPTION:
        for i in range(len(rgb_frames)):
            ep_id = env.episodes[i].episode_id
            generate_video(
                video_option=["disk"],
                video_dir=config.VIDEO_DIR, 
                images=rgb_frames[i],
                episode_id=ep_id,
                checkpoint_idx=0,  
                metrics={},
                tb_writer=None,
            )
            logger.info(f"Generated video for episode {ep_id}")


    logger.info(f"Averaged benchmark for {config.EVAL.NONLEARNING.AGENT}:")
    for stat_key in stats.keys():
        logger.info("{}: {:.3f}".format(stat_key, stats[stat_key]))

    with open(f"stats_{config.EVAL.NONLEARNING.AGENT}_{split}.json", "w") as f:
        json.dump(stats, f, indent=4)


def nonlearning_inference(config: Config) -> None:
    split = config.INFERENCE.SPLIT
    config.defrost()
    # turn off RGBD rendering as neither RandomAgent nor HandcraftedAgent use it.
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = []
    config.TASK_CONFIG.DATASET.SPLIT = config.INFERENCE.SPLIT
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.TASK_CONFIG.TASK.MEASUREMENTS = []
    config.TASK_CONFIG.TASK.SENSORS = []
    config.freeze()

    env = VLNCEInferenceEnv(config=config)

    assert config.INFERENCE.NONLEARNING.AGENT in [
        "RandomAgent",
        "HandcraftedAgent",
    ], "INFERENCE.NONLEARNING.AGENT must be either RandomAgent or HandcraftedAgent."

    if config.INFERENCE.NONLEARNING.AGENT == "RandomAgent":
        agent = RandomAgent()
    else:
        agent = HandcraftedAgent()

    episode_predictions = defaultdict(list)
    for _ in tqdm(range(len(env.episodes)), desc=f"[inference:{split}]"):
        env.reset()
        obs = agent.reset()

        episode_id = env.current_episode.episode_id
        episode_predictions[episode_id].append(env.get_info(obs))

        while not env.get_done(obs):
            obs = env.step(agent.act(obs))
            episode_predictions[episode_id].append(env.get_info(obs))

    with open(config.INFERENCE.PREDICTIONS_FILE, "w") as f:
        json.dump(episode_predictions, f, indent=2)

    logger.info(f"Predictions saved to: {config.INFERENCE.PREDICTIONS_FILE}")

