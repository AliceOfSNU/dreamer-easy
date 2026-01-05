import torch
import torch.nn as nn
import torch.distributions as D
from layers import RSSM, MultiEncoder, MultiDecoder
import torch.nn.functional as F

class RewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 256
        self.net = nn.Sequential(
            nn.Linear(config['state_size']*2, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        mean = self.net(x).squeeze(-1)
        std = torch.ones_like(mean)
        dist = D.Normal(mean, std)
        dist = D.Independent(dist, 1)
        return dist
    
class ContinuationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 256
        self.net = nn.Sequential(
            nn.Linear(config['state_size']*2, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        prob = self.net(x).squeeze(-1)
        dist = D.Bernoulli(prob)
        dist = D.Independent(dist, 1)
        return dist
    
class WorldModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = MultiEncoder(config)
        self.embed_size = self.encoder.out_size
        self.dynamics = RSSM(
            embed=self.embed_size,
            action_size=config['action_size'],
        )
        self.decoder = MultiDecoder(config)
        self.reward = RewardModel(config)
        self.continuation = ContinuationModel(config)

    def forward(self, data):
        obs = data
        actions = data['actions']
        rewards = data['rewards']
        continues = data['continues']
        is_first = data['is_first']

        embed = self.encoder(obs)
        post, prior = self.dynamics.observe(embed, actions, is_first)
        kl_loss = self.dynamics.kl_loss(post, prior)
        feat = self.dynamics.get_feat(post)
        pred = self.decoder(feat)
        reconstruction_loss = 0.0
        for key in pred.keys():
            reconstruction_loss += -pred[key].log_prob(obs[key]).mean()
        reward_loss = -self.reward(feat[:, 1:]).log_prob(rewards[:, :-1]).mean()
        continue_loss = -self.continuation(feat[:, 1:]).log_prob(continues[:, :-1]).mean()
        loss = reconstruction_loss + kl_loss + reward_loss + continue_loss
        return loss, post
    

class Actor(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 256
        self.net = nn.Sequential(
            nn.Linear(config['state_size']*2, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU()
        )
        self.mean = nn.Linear(hidden, config['action_size'])
        self.std = nn.Linear(hidden, config['action_size'])

    def forward(self, x):
        x = self.net(x)
        mean = F.tanh(self.mean(x))
        std = F.softplus(self.std(x)) + 1e-4
        dist = D.Normal(mean, std)
        dist = D.TransformedDistribution(dist, D.TanhTransform())
        dist = D.Independent(dist, 1)
        action = dist.rsample()
        epsilon = float(1e-6)
        action = action.clamp(-1.0 + epsilon, 1.0 - epsilon) # tanh transform compatibility
        logprob = dist.log_prob(action)
        entropy = dist.base_dist.base_dist.entropy().sum(-1)
        if logprob.isinf().any():
            print("Infinite logprob encountered in Actor")
        return action, logprob, entropy
    
class Critic(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 256
        self.net = nn.Sequential(
            nn.Linear(config['state_size']*2, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        mean = self.net(x).squeeze(-1)
        std = torch.ones_like(mean)
        dist = D.Normal(mean, std)
        dist = D.Independent(dist, 1)
        return dist