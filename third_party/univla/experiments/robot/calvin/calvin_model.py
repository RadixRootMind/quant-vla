import math
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from PIL import Image
from einops import rearrange
import time

import torch.nn as nn

from calvin_agent.models.calvin_base_model import CalvinBaseModel

from experiments.robot.robot_utils import (
    DATE_TIME,
    get_latent_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.openvla_utils import get_processor
from prismatic.models.policy.transformer_utils import MAPBloc



class ActionDecoder(torch.nn.Module):
    def __init__(self, window_size=5, hidden_dim=512):
        super().__init__()
        self.latent_action_pool = MAPBlock(n_latents = 1, vis_dim = 4096, embed_dim = hidden_dim, n_heads = hidden_dim//64)
        self.visual_pool = MAPBlock(n_latents = 1, vis_dim = 4096, embed_dim = hidden_dim, n_heads = hidden_dim//64)


        self.proj = nn.Sequential(
                                nn.Linear(hidden_dim, 7 * window_size),
                                nn.Tanh(),
                    )

    def forward(self, latent_action_tokens, visual_embed):
        latent_action_tokens = latent_action_tokens[:, -4:]
        visual_embed = self.visual_pool(visual_embed)
        action = self.proj(self.latent_action_pool(latent_action_tokens , init_embed=visual_embed))
        
        return action



class ActionDecoderWrapper(nn.Module):
    def __init__(self, window_size=12):
        super().__init__()
        self.net = ActionDecoder(window_size)
        self.action_chunk_size = window_size

    def reset(self):
        pass

    def forward(self, latent_actions, visual_embed):
        # Forward action decoder
        pred_action = self.net(latent_actions.to(torch.float), 
                               visual_embed.to(torch.float)).reshape(-1, self.action_chunk_size, 7)
        pred_action = np.array(pred_action.tolist())[0]

        return pred_action


class WrappedModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Load action decoder
        self.action_decoder = ActionDecoderWrapper(cfg.window_size)
        self.action_decoder.net.load_state_dict(torch.load(cfg.action_decoder_path))

        # Load VLA
        self.vla = get_model(cfg)



class WrappedCalvinEvaluation(CalvinBaseModel):
    def __init__(self, cfg, wrapped_model):
        super().__init__()
        self.cfg = cfg

        self.model = wrapped_model
        # [OpenVLA] Get Hugging Face processor
        self.processor = get_processor(cfg)
        self.prev_hist_action = ['']

        

    def reset(self,):
        """
        This is called
        """ 
        self.model.module.action_decoder.reset()
        self.prev_hist_action = ['']


    def step(self, obs, instruction, step):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        img = obs["rgb_obs"]['rgb_static']

        observation = {
            "full_image": img,
            "state": [],
        }

        # Query model to get latent action
        latent_action, visual_embed, generated_ids = get_latent_action(
            self.cfg,
            self.model.module.vla,
            observation,
            instruction,
            processor=self.processor,
        )

        # Get decoded action
        action = self.model.module.action_decoder(latent_action, visual_embed)

        return action
