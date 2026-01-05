import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np

# recurrent state-space model
class RSSM(nn.Module):
    def __init__(self, embed, action_size, stoch=256, deter=256, hidden=64):
        super().__init__()
        self.action_size = action_size
        self.input_layers = nn.Sequential(
            nn.Linear(stoch + action_size, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.SiLU()
        )

        self.cell = nn.GRUCell(hidden, deter)

        self.img_out_layers = nn.Sequential(
            nn.Linear(deter, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.obs_out_layers = nn.Sequential(
            nn.Linear(deter + embed, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        self.img_stat_layer = nn.Linear(hidden, 2 * stoch)
        self.obs_stat_layer = nn.Linear(hidden, 2 * stoch)

        self.W = nn.Parameter(
            torch.zeros((1, deter)), requires_grad=True
        )

    # imagination step does not have observation input
    def img_step(self, prev_state, action):
        prev_stoch, prev_deter = prev_state['stoch'], prev_state['deter']
        x = torch.cat([prev_stoch, action], dim=-1)
        x = self.input_layers(x)
        deter = self.cell(x, prev_deter)
        x = self.img_out_layers(deter)

        mean, std = self.img_stat_layer(x).chunk(2, dim=-1)
        std = F.softplus(std) + 0.1
        dist = D.independent.Independent(D.Normal(mean, std), 1)
        stoch = dist.rsample()
        prior = {"stoch": stoch, "deter": deter, "mean": mean, "std": std}
        return prior
    
    def initialize_state(self, batch_size):
        initial_deter = torch.tanh(self.W).repeat(batch_size, 1)
        initial_stoch = self.img_out_layers(initial_deter)
        initial_stoch, _ = self.img_stat_layer(initial_stoch).chunk(2, dim=-1)
        return {"stoch": initial_stoch, "deter": initial_deter}
    
    # observation step uses observation embedding
    def obs_step(self, prev_state, action, embed, is_first):
        # initialize state if first step in episode
        if prev_state is None or is_first.sum() == is_first.shape[0]:
            prev_state = self.initialize_state(is_first.shape[0])
            action = prev_state["stoch"].new_zeros(is_first.shape[0], self.action_size)
            
        elif torch.sum(is_first) > 0:
            initial_state = self.initialize_state(is_first.shape[0])
            is_first = is_first.unsqueeze(-1)
            # set actions to zero where is_first is True
            action = action * (1 - is_first)
            prev_stoch = prev_state["stoch"] * (1 - is_first) + initial_state["stoch"] * is_first
            prev_deter = prev_state["deter"] * (1 - is_first) + initial_state["deter"] * is_first
            prev_state = {"stoch": prev_stoch, "deter": prev_deter}

        prior = self.img_step(prev_state, action)
        deter = prior["deter"]
        x = torch.cat([deter, embed], dim=-1)
        x = self.obs_out_layers(x)
        mean, std = self.obs_stat_layer(x).chunk(2, dim=-1)
        std = F.softplus(std) + 0.1
        dist = D.independent.Independent(D.Normal(mean, std), 1)
        stoch = dist.rsample()
        post = {"stoch": stoch, "deter": deter, "mean": mean, "std": std}
        return post, prior

    def get_feat(self, state):
        stoch, deter = state["stoch"], state["deter"]
        return torch.cat([stoch, deter], dim=-1)
    
    def observe(self, embed, actions, is_first, state=None):
        # (B, T, ...) => (T, B, ...) time first!
        embed = embed.transpose(0, 1)
        actions = actions.transpose(0, 1)
        is_first = is_first.transpose(0, 1)

        post = []
        prior = []
        for t in range(embed.shape[0]):
            action = None if t == 0 else actions[t-1] # get previous action
            e = embed[t]
            f = is_first[t]
            state, p = self.obs_step(state, action, e, f)
            post.append(state)
            prior.append(p)

        # gather each key's states over time
        post = {key: [s[key] for s in post] for key in post[0].keys()}
        prior = {key: [s[key] for s in prior] for key in prior[0].keys()}

        post = {key: torch.stack(s, dim=0).transpose(0, 1) for key, s in post.items()}
        prior = {key: torch.stack(s, dim=0).transpose(0, 1) for key, s in prior.items()}
        
        return post, prior
    
    def kl_loss(self, post, prior, free=1.0):
        kld = D.kl.kl_divergence
        dist = lambda s: D.independent.Independent(
            D.Normal(s["mean"], s["std"]), 1
        )
        sg = lambda s: {key: val.detach() for key, val in s.items()}

        # kl balancing
        rep_loss = kld(dist(post), dist(sg(prior))).clip(min=free).mean()
        dyn_loss = kld(dist(sg(post)), dist(prior)).clip(min=free).mean()
        loss = 0.2 * rep_loss + 0.8 * dyn_loss
        return loss
    
    def imagine_with_action(self, actions, init_state):
        actions = actions.transpose(0, 1)  # time first
        prior = []
        for t in range(actions.shape[0]):
            a = actions[t]
            state = self.img_step(init_state, a)
            prior.append(state)
            init_state = state
        prior = {key: [s[key] for s in prior] for key in prior[0].keys()}
        prior = {key: torch.stack(s, dim=0).transpose(0, 1) for key, s in prior.items()}
        return prior
    
class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        # initial shape is something like (3, 64, 64)
        # we downsample 4 times with stride 2 convs
        in_channels = 3
        out_channels = 32
        for _ in range(4):
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.SiLU())
            in_channels = out_channels
            out_channels *= 2
        self.conv = nn.Sequential(*layers)
        self.out_size = in_channels * 4 * 4

    def forward(self, x):
        # (B, T, C, H, W) => (B*T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.conv(x)
        x = x.reshape(b, t, -1) # (B, T, features)
        return x
    
class ConvDecoder(nn.Module):
    def __init__(self, feat_size):
        super().__init__()
        self.linear = nn.Linear(feat_size, feat_size)
        layers = []
        in_channels = feat_size // 16
        out_channels = in_channels // 2
        for _ in range(3):
            layers.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.SiLU())
            in_channels = out_channels
            out_channels = max(out_channels // 2, 32)
        # last layer to get back to 3 channels
        layers.append(
            nn.ConvTranspose2d(in_channels, 3, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True)
        )
        self.deconv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.linear(x)
        # (B, T, feat) => (B*T, ch, h, w)
        b, t, f = x.shape
        x = x.view(b*t, f//16, 4, 4)  # assuming final conv output is (B, feat, 4, 4)
        x = self.deconv(x)
        x = x.view(b, t, 3, 64, 64)
        return x
    
class MLP(nn.Module):
    def __init__(self, input_size, mlp_shapes:dict[str, tuple]|None = None):
        super().__init__()
        layers = []
        in_size = input_size
        num_layers = 4
        hidden = 512
        for _ in range(num_layers):
            layers.append(nn.Linear(in_size, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.SiLU())
            in_size = hidden
        self.net = nn.Sequential(*layers)
        self.out_size = hidden

        if mlp_shapes is not None:
            self.mean_heads = nn.ModuleDict()
            self.std_heads = nn.ModuleDict()
            for key, shape in mlp_shapes.items():
                self.mean_heads[key] = nn.Linear(hidden, np.prod(shape))
                self.std_heads[key] = nn.Linear(hidden, np.prod(shape))
    
    def forward(self, x):
        out = self.net(x)
        if not hasattr(self, 'mean_heads'):
            return out
        dists = {}
        for key in self.mean_heads.keys():
            mean = self.mean_heads[key](out)#F.tanh(self.mean_heads[key](out))
            std = F.softplus(self.std_heads[key](out)) + 0.1
            dist = D.independent.Independent(
                D.Normal(mean, std), 1
            )
            dists[key] = dist
        return dists

    
class MultiDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp_shapes = config['mlp_shapes']
        self.cnn_shapes = config['cnn_shapes']
        self.feat_size = config['state_size'] * 2
        self.mlp = MLP(self.feat_size, self.mlp_shapes)
        self.conv_decoder = ConvDecoder(feat_size=self.feat_size)

    def forward(self, x) -> dict[str, D.Distribution]:
        dists = {}
        image_mean = self.conv_decoder(x)
        dists.update(self.make_image_dists(image_mean))
        dists.update(self.mlp(x))
        return dists
    
    def make_image_dists(self, image_mean) -> dict[str, D.Distribution]:
        dist = D.independent.Independent(
            D.Normal(image_mean, 1.0), 3
        )
        return {'image': dist}
    
class MultiEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # cnns
        self.cnn_shapes = config['cnn_shapes']
        self.conv_encoder = ConvEncoder()
        self.out_size = self.conv_encoder.out_size
        # mlps
        self.mlp_shapes = config['mlp_shapes']
        input_size = sum([v[0] for v in self.mlp_shapes.values()])
        self.mlp = MLP(input_size, None)  # assuming conv encoder outputs 512-dim features
        self.out_size += self.mlp.out_size

    def forward(self, obs):
        outputs = []
        cnn_inputs = obs['image']
        outputs.append(self.conv_encoder(cnn_inputs))
        mlp_inputs = torch.cat([obs[k] for k in self.mlp_shapes.keys()], dim=-1)
        outputs.append(self.mlp(mlp_inputs))
        output = torch.cat(outputs, dim=-1)
        return output
    
class Moments(nn.Module):
    def __init__( self, decay = 0.99, min_=1, percentileLow = 0.05, percentileHigh = 0.95):
        super().__init__()
        self._decay = decay
        self._min = torch.tensor(min_).cuda()
        self._percentileLow = percentileLow
        self._percentileHigh = percentileHigh
        self.register_buffer("low", torch.zeros((), dtype=torch.float32).cuda())
        self.register_buffer("high", torch.zeros((), dtype=torch.float32).cuda())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.detach()
        low = torch.quantile(x, self._percentileLow)
        high = torch.quantile(x, self._percentileHigh)
        self.low = self._decay*self.low + (1 - self._decay)*low
        self.high = self._decay*self.high + (1 - self._decay)*high
        inverseScale = torch.max(self._min, self.high - self.low)
        return self.low.detach(), inverseScale.detach()