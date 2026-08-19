import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import draccus
import numpy as np
import tqdm
from tqdm import trange
from collections import defaultdict
from libero.libero import benchmark

import wandb
import json

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.r2r.r2r_utils import (
    get_libero_dummy_action,
    get_navigation_image,
    quat2axisangle,
    save_rollout_video,
    get_env_config,
    split_and_sample_dataset
)
from habitat import Env, logger

from habitat.utils.visualizations.utils import append_text_to_image

from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_latent_action,
    get_vla_action,
    get_image_resize_size,
    get_model,
    set_seed_everywhere
)


# import sys 
# sys.path.append("../VLN-CE")
from habitat_extensions.utils import generate_video, observations_to_image

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
    
    use_pretrained_openvla: bool = False
    action_decoder_path:str = ""
    center_crop: bool = False                        # Center crop? (if trained w/ random crop image aug)
    with_hist: bool = True                           
    add_visual_embed: bool = True
    #################################################################################################################
    # R2R environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "r2r"           
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    window_size: int = 4
    hist_latent_action_num: int = 1
    use_random_policy: bool = False
    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/eval_logs"   # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)
    max_steps: int = 500

    env_config_path: str = "experiments/robot/r2r/config/eval_val_unseen.yaml"
    save_video: bool = True
    video_dir: str = "./experiments/robot/r2r/video"
    # fmt: on

from prismatic.models.policy.transformer_utils import MAPBlock
class ActionDecoder(nn.Module):
    def __init__(self,window_size=5, add_visual_embed=True):
        super().__init__()
        self.attn_pool = MAPBlock(n_latents = 1, vis_dim = 4096, embed_dim = 512, n_heads = 8)

        self.window_size = window_size
        self.add_visual_embed = add_visual_embed
        if self.add_visual_embed:
            self.visual_pool = MAPBlock(n_latents = 1, vis_dim = 4096, embed_dim = 512, n_heads = 8)
            self.proj = nn.Linear(1024, 4 * window_size)
        else:
            self.proj = nn.Linear(512, 4 * window_size)
        
        # don't use in this version
        self.temporal_size = self.window_size
        self.temporal_mask = torch.flip(torch.triu(torch.ones(self.temporal_size, self.temporal_size, dtype=torch.bool)), dims=[1]).numpy()
        
        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 4))
        self.action_buffer_mask = np.zeros(
            (self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_
        )
        # Action chunking with temporal aggregation
        balancing_factor = 0.1
        self.temporal_weights = np.array([np.exp(-1 * balancing_factor * i) for i in range(self.temporal_size)])[:, None]


    def reset(self):
        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 4))
        self.action_buffer_mask = np.zeros(
            (self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_
        )
    
    def forward(self, latent_actions, visual_embed=None, mask=None, shift_action_buffer=True):
        # Forward action decoder
        if self.add_visual_embed:
            visual_embed = self.visual_pool(visual_embed.to(torch.float))
            pred_action = self.proj(torch.cat([self.attn_pool(latent_actions.to(torch.float), init_embed=None), visual_embed], dim=-1)).reshape(-1, self.window_size, 4)
        else:
            pred_action = self.proj(self.attn_pool(latent_actions.to(torch.float), init_embed=None)).reshape(-1, self.window_size, 4)

        pred_action = F.softmax(pred_action, dim=-1)
        pred_action = np.array(pred_action.cpu().tolist())


        if shift_action_buffer:
            # Shift action buffer
            self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
            self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
            self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
            self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
            self.action_buffer_mask = self.action_buffer_mask * self.temporal_mask

            # Add to action buffer
            self.action_buffer[0] = pred_action  
            self.action_buffer_mask[0] = np.array([True] * self.temporal_mask.shape[0], dtype=np.bool_)

            # action_prediction = pred_action[0,0]
            # Ensemble temporally to predict action
            action_prediction = np.sum(self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1] * self.temporal_weights, axis=0) / np.sum(self.action_buffer_mask[:, 0:1] * self.temporal_weights)

            current_pred = action_prediction
            # print("current_pred", current_pred)
            action_prediction = np.argmax(current_pred, axis=-1)
        
        else:
            current_pred = pred_action
            if mask is not None:
                current_pred = np.where(
                    mask,
                    current_pred,
                    np.array([1.0] + [0.0]*3)  
                )
            # print("current_pred", current_pred)

            action_prediction = np.argmax(current_pred, axis=-1)
            action_prediction = action_prediction[0][0]


        return action_prediction


@draccus.wrap()
def eval_r2r(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint and not cfg.use_pretrained_openvla:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    if not cfg.use_random_policy:
        # Load action decoder
        if not cfg.use_pretrained_openvla:
            action_decoder = ActionDecoder(cfg.window_size, cfg.add_visual_embed)
            action_decoder.load_state_dict(torch.load(cfg.action_decoder_path))
            action_decoder.eval().cuda()

        # Load model
        model = get_model(cfg)

        # [OpenVLA] Check that the model contains the action un-normalization key
        if cfg.model_family == "openvla":
            # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
            # with the suffix "_no_noops" in the dataset name)
            if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
                cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
            
            # assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

        # [OpenVLA] Get Hugging Face processor
        processor = None
        if cfg.model_family == "openvla":
            processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )


    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)
    
    env_config = get_env_config(cfg.env_config_path, cfg.seed)

    filtered_dataset = None
    env = Env(config=env_config.TASK_CONFIG, dataset=filtered_dataset)
    # if len(config.VIDEO_OPTION) > 0:
        # os.makedirs(config.VIDEO_DIR, exist_ok=True)
    os.makedirs(env_config.OUTPUT_DIR, exist_ok=True)
    
    stats = defaultdict(float)
    stats_episodes = defaultdict(dict)

    print(len(env.episodes))
    num_episodes = min(env_config.EVAL.EPISODE_COUNT, len(env.episodes))
    
    fail_episodes = []
    info_list = []

    save_dir = os.path.join(cfg.pretrained_checkpoint, "evaluate")

    for i in trange(num_episodes):
        obs = env.reset()
        if not cfg.use_pretrained_openvla and not cfg.use_random_policy:
            action_decoder.reset()
            prev_hist_action = [''] * cfg.hist_latent_action_num

        ep_id = env.current_episode.episode_id
        rgb_frames = []

        task_description = env.current_episode.instruction.instruction_text

        
        print("instruction:", task_description)
        info = env.get_metrics()  
        if cfg.save_video and i < 100:
            frame = observations_to_image(obs, info)
            frame = append_text_to_image(
                frame, 
                env.current_episode.instruction.instruction_text
            )
            rgb_frames.append(frame)

        # Store language data for this episode
        instruction_text = env.current_episode.instruction.instruction_text

        action_list = []
        for step in range(cfg.max_steps):
            # print("distance_to_goal", info['distance_to_goal'])
            img = obs['rgb']

            # Prepare observations dict
            observation = {
                "full_image": img,
            }

            # Query model to get action
            latent_action, visual_embed, generated_ids= get_latent_action(
                cfg,
                model,
                observation,
                task_description,
                processor=processor,
                hist_action=prev_hist_action[-cfg.hist_latent_action_num:],
            )

            latent_action_detokenize = [f'<ACT_{i}>' for i in range(32)]
            if cfg.with_hist:
                hist_action = ''
                for latent_action_ids in generated_ids[0]:
                    if latent_action_ids.item() - 32001 >= 0 and latent_action_ids.item() - 32001 < 32:
                        hist_action += latent_action_detokenize[latent_action_ids.item() - 32001]
                    else:
                        hist_action = ""
                        break
                prev_hist_action.append(hist_action)

            action = action_decoder(latent_action, visual_embed, mask=None)
            print("episodes, step:", i, step)

            
            obs = env.step(action)
            action_list.append(action)

            info = env.get_metrics() 
            

            if cfg.save_video and i < 100:
                frame = observations_to_image(obs, info)
                frame = append_text_to_image(
                    frame, 
                    env.current_episode.instruction.instruction_text
                )
                rgb_frames.append(frame)

            if info['oracle_success'] == 1:
                break
            
            if action_list[-1] == 0:
                break

        is_success = env.get_metrics()['oracle_success']
        if not is_success:
            fail_episodes.append(i)
        info_list.append(env.get_metrics())
        for m, v in env.get_metrics().items():
            stats[m] += v
        # print(i)
        # print("action_list", action_list)

        if cfg.save_video and i < 100:
            video_dir = os.path.join(save_dir, "videos", f"{env_config.EVAL.SPLIT}_hist_{cfg.hist_latent_action_num}")

            generate_video(
                video_option=["disk"],
                video_dir=video_dir, 
                images=rgb_frames,
                episode_id=ep_id,
                checkpoint_idx=0, 
                metrics={'oracle_success': info_list[i]['oracle_success']},
                tb_writer=None,
            )
            print(f"Generated video for episode {ep_id}")
            logger.info(f"Generated video for episode {ep_id}")


    print("*"*50)
    print("unsuccessful_episodes", fail_episodes)
    stats = {k: v / num_episodes for k, v in stats.items()}
    print("stats", stats)

    log_file.write(f"Averaged benchmark for {env_config.EVAL.NONLEARNING.AGENT}:")
    for stat_key in stats.keys():
        print("{}: {:.3f}".format(stat_key, stats[stat_key]))
        log_file.write("{}: {:.3f}".format(stat_key, stats[stat_key]))


    with open(os.path.join(save_dir, f"stats_{env_config.EVAL.NONLEARNING.AGENT}_{env_config.EVAL.SPLIT}_hist_{cfg.hist_latent_action_num}.json"), "w") as f:
        json.dump(stats, f, indent=4)






if __name__ == "__main__":
    eval_r2r()
