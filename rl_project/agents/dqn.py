import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


class QNetwork(nn.Module):
    """
    Q-network closely following Practical 8 structure (MLP head); used for Atari after CNN features
    are provided by env wrappers (84x84, stacked frames). For simplicity here we implement a small CNN
    + MLP consistent with Practical 8 examples.
    """
    def __init__(self, input_shape, num_actions, hidden_sizes=(512, 512, 256), lr=1e-3):
        super().__init__()

        c, h, w = input_shape  # expecting (C, H, W)
        self.features = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            feat_size = self.features(dummy).shape[1]

        layers = []
        last = feat_size
        for hs in hidden_sizes:
            layers += [nn.Linear(last, hs), nn.ReLU()]
            last = hs
        layers += [nn.Linear(last, num_actions)]
        self.head = nn.Sequential(*layers)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()  # Huber loss (Prac8-friendly improvement)

    def forward(self, x):
        x = self.features(x)
        return self.head(x)

    def update(self, states, targets):
        self.optimizer.zero_grad()
        out = self.forward(states)
        loss = self.criterion(out, targets)
        loss.backward()
        self.optimizer.step()


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, *exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        s, a, r, ns, d = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(ns), np.array(d)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Minimal DQN derived from Practical 8 ideas: online/target networks, replay buffer, epsilon-greedy.
    Designed for discrete Atari (Seaquest).
    """
    def __init__(self, state_dim, action_dim, is_continuous, lr=1e-3, gamma=0.99,
                 replay_size=1000000, batch_size=512, target_update=1000, device="cpu"):
        assert not is_continuous, "DQN supports discrete action spaces only"
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.total_steps = 0

        # Expect state_dim as image shape (C, H, W)
        if isinstance(state_dim, tuple) and len(state_dim) == 3:
            c, h, w = state_dim
        else:
            # Fallback (should not happen for Atari path with wrappers)
            c, h, w = 4, 84, 84

        self.q = QNetwork((c, h, w), action_dim, lr=lr).to(device)
        self.q_target = QNetwork((c, h, w), action_dim, lr=lr).to(device)
        self.q_target.load_state_dict(self.q.state_dict())

        self.buffer = ReplayBuffer(replay_size)

        # Epsilon-greedy
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay = 200000  # steps

    def _epsilon(self):
        t = self.total_steps
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-t / self.eps_decay)

    def _to_tensor(self, x):
        return torch.FloatTensor(x).to(self.device)

    def select_action(self, state, stochastic=True):
        # state from env_wrappers is (4,84,84,1) -> our wrappers convert; ensure (C,H,W)
        if hasattr(state, "_frames"):
            state = np.array(state)
        if isinstance(state, np.ndarray):
            if state.ndim == 4:  # (C,H,W,1)
                state = state.squeeze(-1)
            if state.ndim == 3:
                state = state[None, ...]
        eps = self._epsilon()
        self.total_steps += 1
        if np.random.rand() < eps:
            return np.random.randint(0, self.q.head[-1].out_features)
        with torch.no_grad():
            s = self._to_tensor(state)
            q = self.q(s)
            return int(q.argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        # store raw; convert in update
        if hasattr(state, "_frames"):
            state = np.array(state)
        if hasattr(next_state, "_frames"):
            next_state = np.array(next_state)
        # squeeze last channel if present
        if isinstance(state, np.ndarray) and state.ndim == 4:
            state = state.squeeze(-1)
        if isinstance(next_state, np.ndarray) and next_state.ndim == 4:
            next_state = next_state.squeeze(-1)
        self.buffer.add(state, action, reward, next_state, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return {"loss": 0.0}
        s, a, r, ns, d = self.buffer.sample(self.batch_size)
        # to torch
        s = self._to_tensor(s)
        ns = self._to_tensor(ns)
        r = self._to_tensor(r).unsqueeze(1)
        d = self._to_tensor(d).unsqueeze(1)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)

        with torch.no_grad():
            # Double DQN: action selection from online net, evaluation from target net
            next_online = self.q(ns).argmax(dim=1, keepdim=True)
            target_q_next = self.q_target(ns).gather(1, next_online)
            y = r + (1 - d) * self.gamma * target_q_next

        q_all = self.q(s)
        q = q_all.gather(1, a)
        # Reward clipping: [-1,1]
        y_clipped = torch.clamp(y, -1.0, 1.0)
        # Build full targets (keep other actions unchanged)
        full_targets = q_all.detach()
        full_targets.scatter_(1, a, y_clipped)

        self.q.update(s, full_targets)

        if self.total_steps % self.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return {"loss": float((q - y).pow(2).mean().item())}

    def save(self, path):
        torch.save({"q": self.q.state_dict(), "q_target": self.q_target.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["q"] if "q" in ckpt else ckpt)
        if "q_target" in ckpt:
            self.q_target.load_state_dict(ckpt["q_target"])

