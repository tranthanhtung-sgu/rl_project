import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        if len(input_shape) == 4:
            frames, h, w, _ = input_shape
            c = frames
        else:
            c, h, w = input_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            self.feature_dim = self.net(dummy).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.features = CNNFeatureExtractor(input_shape)
        self.policy = nn.Sequential(
            nn.Linear(self.features.feature_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(self.features.feature_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        feats = self.features(x)
        logits = self.policy(feats)
        value = self.value(feats).squeeze(-1)
        return logits, value


class RolloutBuffer:
    def __init__(self, capacity, obs_shape, device):
        self.capacity = capacity
        self.device = device
        self.reset(obs_shape)

    def reset(self, obs_shape):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []

    def add(self, obs, action, reward, done, value, log_prob):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def compute_returns_advantages(self, last_value, gamma, gae_lambda):
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - float(self.dones[t])
            next_value = last_value if t == T - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * nonterminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
            advantages[t] = last_gae
        returns = (np.array(self.values, dtype=np.float32) + advantages).astype(np.float32)
        return returns, advantages


class PPOAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 is_continuous,
                 lr=2.5e-4,
                 gamma=0.99,
                 gae_lambda=0.90,
                 clip_range=0.10,
                 ent_coef=0.01,
                 vf_coef=0.5,
                 n_steps=128,
                 batch_size=256,
                 n_epochs=4,
                 max_grad_norm=0.5,
                 device="cpu"):
        assert not is_continuous, "PPOAgent here supports discrete actions (Atari) only"
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.max_grad_norm = max_grad_norm

        if isinstance(state_dim, tuple) and len(state_dim) == 4:
            frames, h, w, _ = state_dim
            input_shape = (frames, h, w)
        elif isinstance(state_dim, tuple) and len(state_dim) == 3:
            input_shape = state_dim
        else:
            # Expect Atari image input
            input_shape = (4, 84, 84)

        self.net = ActorCritic(input_shape, action_dim).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.rollout = None

    def _to_tensor_obs(self, obs):
        if hasattr(obs, "_frames"):
            obs = np.array(obs)
        if isinstance(obs, np.ndarray):
            if obs.ndim == 4:
                obs = obs.squeeze(-1)
            if obs.ndim == 3:
                obs = obs[None, ...]
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def select_action(self, state, stochastic=True, store=True):
        s = self._to_tensor_obs(state)
        logits, value = self.net(s)
        dist = Categorical(logits=logits)
        if stochastic:
            action = dist.sample()
        else:
            action = dist.probs.argmax(dim=1)
        log_prob = dist.log_prob(action)
        if store:
            # Store CPU scalars to keep memory light
            a = int(action.item())
            v = float(value.squeeze(0).item())
            lp = float(log_prob.squeeze(0).item())
            return a, v, lp
        return int(action.item())

    def start_rollout(self, obs_shape):
        self.rollout = RolloutBuffer(self.n_steps, obs_shape, self.device)

    def store_step(self, obs, action, reward, done, value, log_prob):
        self.rollout.add(obs, action, reward, done, value, log_prob)

    def update(self, last_value):
        returns, advantages = self.rollout.compute_returns_advantages(last_value, self.gamma, self.gae_lambda)
        obs_np = np.array(self.rollout.obs)
        acts_np = np.array(self.rollout.actions)
        vals_np = np.array(self.rollout.values, dtype=np.float32)
        old_logp_np = np.array(self.rollout.log_probs, dtype=np.float32)

        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Prepare tensors
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        # Handle shapes:
        # - (N, C, H, W)                -> OK
        # - (N, C, H, W, 1)             -> squeeze last dim
        # - (C, H, W, 1) or (C, H, W)   -> add batch dim earlier; here we expect batched
        if obs_t.ndim == 5 and obs_t.shape[-1] in [1, 3]:
            obs_t = obs_t.squeeze(-1)  # (N, C, H, W)
        elif obs_t.ndim == 4 and obs_t.shape[-1] in [1, 3]:
            # If somehow unbatched but with channel last, move to channel-first by adding batch dim handled earlier
            obs_t = obs_t.permute(0, 3, 1, 2)  # (N, C, H, W)
        acts_t = torch.as_tensor(acts_np, dtype=torch.long, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        old_logp_t = torch.as_tensor(old_logp_np, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)

        N = obs_t.shape[0]
        idxs = np.arange(N)

        for _ in range(self.n_epochs):
            np.random.shuffle(idxs)
            for start in range(0, N, self.batch_size):
                end = start + self.batch_size
                mb_idx = idxs[start:end]
                mb_obs = obs_t[mb_idx]
                mb_act = acts_t[mb_idx]
                mb_ret = returns_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_old_logp = old_logp_t[mb_idx]

                logits, values = self.net(mb_obs)
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(values, mb_ret)
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

        # Clear buffer
        self.rollout = None
        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item())
        }

    def save(self, path):
        torch.save({"net": self.net.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        state = ckpt["net"] if "net" in ckpt else ckpt
        self.net.load_state_dict(state)


