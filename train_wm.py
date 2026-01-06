# train the world model from expert trajectories

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from models import WorldModel
import imageio
from utils import ReplayBuffer
from collections import OrderedDict

def imagine_rollout(model, data):
    batch_size = 1
    with torch.no_grad():
        embed = model.encoder(data)
        post, _ = model.dynamics.observe(
            embed[:batch_size, :5], data["actions"][:batch_size, :5], data["is_first"][:batch_size, :5]
        )
        feat = model.dynamics.get_feat(post)
        recon = model.decoder(feat)["image"].mode

        # open loop prediction, uses prior
        init = {k: v[:, -1] for k, v in post.items()}
        prior = model.dynamics.imagine_with_action(data["actions"][:batch_size, 5:], init)
        feat = model.dynamics.get_feat(prior)
        openl = model.decoder(feat)["image"].mode
        
        # observed image is given until 5 steps
        pred = torch.cat([recon[:, :5], openl], 1)
        pred = ((pred +1) /2 * 255.0).clamp(0, 255)
        pred = pred.cpu().numpy().astype(np.uint8)  # (B, T, C, H, W)
    
    # [-1, 1] -> [0, 255]
    gt_images = ((data["image"][:batch_size] +1) /2 * 255.0).clamp(0, 255)
    gt_images = gt_images.cpu().numpy().astype(np.uint8)
    pred_images = pred

    # Sample every 5 frames
    frame_indices = np.arange(0, 50, 5)  # [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    gt_sampled = gt_images[0, frame_indices]  # (10, 3, 64, 64)
    pred_sampled = pred_images[0, frame_indices]  # (10, 3, 64, 64)

    # Convert from (10, 3, 64, 64) to (10, 64, 64, 3) for imageio
    gt_sampled = np.transpose(gt_sampled, (0, 2, 3, 1))  # (10, 64, 64, 3)
    pred_sampled = np.transpose(pred_sampled, (0, 2, 3, 1))  # (10, 64, 64, 3)

    # Create grid: concatenate horizontally for each row, then vertically
    gt_row = np.concatenate(list(gt_sampled), axis=1)  # (64, 640, 3)
    pred_row = np.concatenate(list(pred_sampled), axis=1)  # (64, 640, 3)
    grid = np.concatenate([gt_row, pred_row], axis=0)  # (128, 640, 3)

    # Save the grid
    imageio.imwrite('rollout_grid.png', grid)


if __name__ == '__main__':
    env_name = "walker2d-medium-expert-v2"
    # load traj.npz
    file_name = f'rollouts/{env_name}_td3bc_trajectories.npz'
    data = np.load(file_name)
    # fill replay buffer with expert data
    config = {
        'mlp_shapes': {'obs': (17, )},
        'cnn_shapes': {'image': (3, 64, 64)},
        'state_size': 256,
        'action_size': 6,
    }
    replay_buffer = ReplayBuffer(config, max_size=300000)
    length = data['observations'].shape[0]
    replay_buffer.observations[:length] = data['observations']
    replay_buffer.frames[:length] = data['frames']
    replay_buffer.actions[:length] = data['actions']
    replay_buffer.rewards[:length] = data['rewards']
    replay_buffer.continues[:length] = data['continues']
    replay_buffer.is_first[:length] = data['is_first']
    replay_buffer.size = length
    replay_buffer.ptr = length % replay_buffer.max_size
    print(f"loaded {replay_buffer.size} expert transitions into replay buffer")

    # train world model
    batch_size = 64
    horizon = 50
    num_iterations = 1000

    model = WorldModel(config).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    clip_grad_norm = 100.0

    for it in tqdm(range(num_iterations)):
        batch = replay_buffer.sample(batch_size, horizon)
        batch = {k: v.cuda() for k, v in batch.items()}

        loss, post = model(batch)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()

        if it % 50 == 0:
            print(f"Iteration {it + 1}/{num_iterations} completed")
            imagine_rollout(model, batch)