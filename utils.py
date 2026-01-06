import numpy as np
import torch

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
