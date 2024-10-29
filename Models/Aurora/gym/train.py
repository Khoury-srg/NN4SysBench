# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

''' This script is for training the aurora model'''
import gym
import network_sim
import torch
import random
import numpy as np
from model import CustomNetwork_mid, CustomNetwork_big, CustomNetwork_small

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import os
import sys
import inspect
import argparse

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from common.simple_arg_parse import arg_or_default

K = 10

# Custom MLP policy class
class MyMlpPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule=0.0001,
                 model_type='small', *args, **kwargs):
        self.model_type = model_type
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        if self.model_type == 'small':
            self.mlp_extractor = CustomNetwork_small()
        elif self.model_type == 'mid':
            self.mlp_extractor = CustomNetwork_mid()
        elif self.model_type == 'big':
            self.mlp_extractor = CustomNetwork_big()

# Function to train the selected model
def train_model(model_type):
    env = gym.make('PccNs-v0')

    gamma = arg_or_default("--gamma", default=0.99)
    print(f"gamma = {gamma}")
    
    # Create the model with the selected model type
    model = PPO(
        MyMlpPolicy,
        env,
        policy_kwargs={'model_type': model_type},
        seed=20,
        learning_rate=0.0001,
        verbose=1,
        batch_size=2048,
        n_steps=8192,
        gamma=gamma
    )

    MODEL_PATH = f"./results/pcc_model_{model_type}_{K}_%d.pt"
    for i in range(0, 6):
        model.learn(total_timesteps=(1600 * 410))
        torch.save(model.policy.state_dict(), MODEL_PATH % i)
    print(f"Training completed for {model_type} model.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train Aurora model (small, mid, big, or all).")
    parser.add_argument(
        "--model", 
        choices=["small", "mid", "big", "all"], 
        default="small", 
        help="Choose the model type to train: small, mid, big, or all models."
    )
    args = parser.parse_args()

    # Train the selected model or all models
    if args.model == "all":
        for model_type in ["small", "mid", "big"]:
            print(f"Training {model_type} model...")
            train_model(model_type)
    else:
        print(f"Training {args.model} model...")
        train_model(args.model)

    print(f"[Done] Finished training {args.model} model")
