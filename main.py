import sys
import torch
from tqdm import tqdm
from models import WorldModel, Actor, Critic
from layers import Moments
import robomimic.utils.file_utils as FileUtils
import d4rl
import torch.nn as nn
import h5py
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import imageio
import gym
import copy
    

def imagine_rollout(model, actor, deter, stoch, horizon=16):
    # model.eval()
    actions = []
    logprobs = []
    entropies = []
    deters = []
    stochs = []
    init = {'deter': deter, 'stoch': stoch}
    curr_state = init
    for tau in range(horizon):
        feat = model.dynamics.get_feat(curr_state)
        action, logprob, entropy = actor(feat)
        actions.append(action)
        logprobs.append(logprob)
        entropies.append(entropy)
        curr_state = model.dynamics.img_step(curr_state, action)
        # states should recorded after imagine_step?
        deters.append(curr_state['deter'])
        stochs.append(curr_state['stoch'])
    actions = torch.stack(actions, dim=1)  # (B, H, action_size)
    logprobs = torch.stack(logprobs, dim=1)  # (B, H)
    entropies = torch.stack(entropies, dim=1)  # (B, H)
    deters = torch.stack(deters, dim=1)  # (B, H, deter_size)
    stochs = torch.stack(stochs, dim=1)  # (B, H, stoch_size)
    # model.train()
    return actions, logprobs, entropies, deters, stochs

def critic_targets(rewards, continues, values, discount=0.997, lambda_=0.95):
    # rewards: (B, T)
    # continues: (B, T)
    # values: (B, T)
    T = rewards.shape[1]
    targets = torch.zeros_like(rewards)
    targets[:, -1] = values[:, -1]
    for t in reversed(range(T-1)):
        reward = rewards[:, t]
        continue_ = continues[:, t]
        value = values[:, t+1]
        target = targets[:, t+1]
        bootstrap = (1-lambda_) * value + lambda_ * target
        targets[:, t] = reward + discount * continue_ * bootstrap
    return targets


def collect_trajectory(env_name, model, actor, replay_buffer):
    model.eval()
    actor.eval()
    env = gym.make(env_name)
    env.seed(100)
    num_episodes = 10
    total_score = 0.0
    for ep in range(num_episodes):
        observations = []
        frames = []
        actions = []
        rewards = []
        continues = []

        obs = env.reset()
        post = None
        action = None
        done = False
        total_reward = 0.0
        with torch.no_grad():
            while not done:
                # render frame
                frame = env.render(mode='rgb_array', width=64, height=64)

                # encode observation
                data = {}
                data["obs"] = torch.Tensor(obs).cuda()[None, None, ...]
                data["image"] = torch.Tensor(frame.copy()).cuda().permute(2,0,1)[None, None, ...]
                data["image"] = data["image"] / 255.0 * 2 -1  # normalize to [-1, 1]
                embed = model.encoder(data).squeeze(1)
                
                # update posterior state
                first = torch.Tensor([1.0 if action is None else 0.0]).unsqueeze(0)
                post, _ = model.dynamics.obs_step(post, action, embed, first)

                # choose action
                feat = model.dynamics.get_feat(post)
                action, _, _ = actor(feat)
                action_np = action.squeeze(0).detach().cpu().numpy()

                # step in env
                next_obs, reward, done, _ = env.step(action_np)
                total_reward += reward

                # save data
                observations.append(obs)
                frames.append(frame)
                actions.append(action_np)
                rewards.append(reward)
                continues.append(1.0 - float(done))
                obs = next_obs
        
        d4rl_score = env.get_normalized_score(total_reward) * 100.0
        total_score += d4rl_score

        # stack along time dimension
        frames = np.stack(frames, axis=0).transpose(0, 3, 1, 2)  # (T, C, H, W)
        obs = np.stack(observations, axis=0)  # (T, obs_dim)
        actions = np.stack(actions, axis=0)  # (T, action_dim)
        rewards = np.array(rewards)  # (T,)
        continues = np.array(continues)  # (T,)
        states = {'obs': obs, 'image': frames}

        replay_buffer.insert(
            states,
            actions,
            rewards,
            continues,
        )

    total_score /= num_episodes
    print(f"average d4rl score: {total_score:.2f}")
    model.train()
    actor.train()
    return total_score


def get_seed_episodes(env_name, replay_buffer, num_env_steps=10000):
    env = gym.make(env_name)
    env.seed(52)
    step = 0
    while step < num_env_steps:
        observations = []
        frames = []
        actions = []
        rewards = []
        continues = []

        obs = env.reset()
        done = False
        while not done and step < num_env_steps:
            frame = env.render(mode='rgb_array', width=64, height=64)

            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            observations.append(obs)
            frames.append(frame)
            actions.append(action)
            rewards.append(reward)
            continues.append(1.0 - float(done))

            obs = next_obs
            step += 1
        frames = np.stack(frames, axis=0).transpose(0, 3, 1, 2)  # (T, C, H, W)
        obs = np.stack(observations, axis=0)  # (T, obs_dim)
        actions = np.stack(actions, axis=0)  # (T, action_dim)
        rewards = np.array(rewards)  # (T,)
        continues = np.array(continues)  # (T,)
        data = {'obs': obs, 'image': frames}
        replay_buffer.insert(
            data,
            actions,
            rewards,
            continues,
        )

class ReplayBuffer:
    def __init__(self, config, max_size=1_000_000):
        self.observations = np.empty((max_size, *config['mlp_shapes']["obs"]), dtype=np.float32)
        self.frames = np.empty((max_size, *config["cnn_shapes"]["image"]), dtype=np.uint8)
        self.actions = np.empty((max_size, config["action_size"]), dtype=np.float32)
        self.rewards = np.empty((max_size,))
        self.continues = np.empty((max_size, ))
        self.is_first = np.empty((max_size, ))
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

    def insert(self, data, action, reward, continue_):
        obs = data['obs']
        frame = data['image']
        length = obs.shape[0]
        first = np.zeros((length, ), dtype=np.float32)
        first[0] = 1.0
        if self.ptr + length <= self.max_size:
            self.observations[self.ptr:self.ptr + length] = obs
            self.frames[self.ptr:self.ptr + length] = frame
            self.actions[self.ptr:self.ptr + length] = action
            self.rewards[self.ptr:self.ptr + length] = reward
            self.continues[self.ptr:self.ptr + length] = continue_
            self.is_first[self.ptr:self.ptr + length] = first
            self.ptr += length
        else:
            # truncate
            valid_length = self.max_size - self.ptr
            self.observations[self.ptr:self.max_size] = obs[:valid_length]
            self.frames[self.ptr:self.max_size] = frame[:valid_length]
            self.actions[self.ptr:self.max_size] = action[:valid_length]
            self.rewards[self.ptr:self.max_size] = reward[:valid_length]
            self.continues[self.ptr:self.max_size] = continue_[:valid_length]
            self.is_first[self.ptr:self.max_size] = first[:valid_length]
            self.ptr = 0
        self.size = min(self.size + length, self.max_size)

    def sample(self, batch_size, horizon=16):
        start_idxs = np.random.choice(self.size-horizon, batch_size, replace=False)

        # Create indices for contiguous sequences
        # start_idxs: (batch_size,)
        # indices: (batch_size, horizon)
        indices = start_idxs[:, None] + np.arange(horizon)[None, :]

        batch = {
            'obs': torch.Tensor(self.observations[indices]),  # (batch_size, horizon, ...)
            'image': torch.Tensor(self.frames[indices])/255.0 * 2 -1,  # (batch_size, horizon, C, H, W)
            'actions': torch.Tensor(self.actions[indices]),  # (batch_size, horizon, action_dim)
            'rewards': torch.Tensor(self.rewards[indices]),  # (batch_size, horizon)
            'continues': torch.Tensor(self.continues[indices]),  # (batch_size, horizon)
            'is_first': torch.Tensor(self.is_first[indices]),  # (batch_size, horizon)
        }
        return batch


if __name__ == "__main__":
    config = {
        'mlp_shapes': {'obs': (17, )},
        'cnn_shapes': {'image': (3, 64, 64)},
        'state_size': 256,
        'action_size': 6,
    }
    env_name = "walker2d-medium-expert-v2"
    torch.manual_seed(42)
    replay = ReplayBuffer(config, max_size=300000)
    model = WorldModel(config).cuda()
    actor = Actor(config).cuda()
    critic = Critic(config).cuda()
    critic_target = copy.deepcopy(critic).cuda()
    moments = Moments().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-4)

    batch_size = 32
    update_steps = 50
    seq_len = 50
    img_len = 16
    clip_grad_norm = 100.0
    polyak = 0.98
    get_seed_episodes(env_name, replay, num_env_steps=1000)

    num_steps = 0
    for epoch in range(100):
        pbar = tqdm(range(update_steps), desc=f"Epoch {epoch}")
        for _ in pbar:
            # train model
            batch = replay.sample(batch_size, horizon=seq_len)
            batch = {k: v.cuda() for k, v in batch.items()}

            loss, post = model(batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()
            # imagine rollouts
            B, T, _ = post['deter'].shape
            deter = post['deter'].reshape(B*T, -1).detach()
            stoch = post['stoch'].reshape(B*T, -1).detach()
            img_actions, img_action_logprobs, img_action_entropy, img_deters, img_stochs = imagine_rollout(model, actor, deter, stoch, horizon=img_len)
            img_deters = img_deters.detach()
            img_stochs = img_stochs.detach()
            # predict rewards and continuations
            feats = torch.cat([img_stochs, img_deters], dim=-1)
            rewards = model.reward(feats).mean
            continues = model.continuation(feats).mean
            critic_target_dist = critic_target(feats)
            values = critic_target_dist.mean
            # critic targets
            target_values = critic_targets(rewards, continues, values)
            
            # critic loss
            critic_dist = critic(feats)
            critic_loss = -critic_dist.log_prob(target_values).mean()
            # actor loss
            _, scale = moments(target_values)
            advantages = (target_values - values).detach()/scale
            actor_loss = -(img_action_logprobs * advantages).mean() - 3e-4 * img_action_entropy.mean()
            
            actor_opt.zero_grad()
            actor_loss.backward()
            for name, param in actor.named_parameters():
                if param.grad is not None:
                    if param.grad.isnan().any():
                        print(f"NaN detected in actor parameter: {name}")
            nn.utils.clip_grad_norm_(actor.parameters(), max_norm=clip_grad_norm)
            actor_opt.step()

            critic_opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=clip_grad_norm)
            critic_opt.step()
            
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(polyak * target_param.data + (1 - polyak) * param.data)


            num_steps += 1
            pbar.set_postfix({'model_loss': f'{loss.item():.3f}',
                              'actor_loss': f'{actor_loss.item():.3f}',
                              'critic_loss': f'{critic_loss.item():.3f}'})
        
        # collect more episodes
        d4rl_score = collect_trajectory(env_name, model, actor, replay)
