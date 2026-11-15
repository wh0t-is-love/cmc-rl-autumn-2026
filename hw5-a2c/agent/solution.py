"""A2C solution for Practical_RL week06_policy_based / a2c-optional.

Blocks implemented here (same order as in the notebook):
  1. Model       — Nature DQN CNN with two heads (policy logits + state value)
  2. Policy      — samples actions, returns dict for the runner
  3. ComputeValueTargets — n-step value targets with resets masking
  4. MergeTimeBatch      — (T, N, ...) -> (T*N, ...)
  5. A2C         — losses + optimization step
  6. (train.py)  — training loop with RMSprop + linear lr decay
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


# ---------------------------------------------------------------- 1. model
def _init_layer(m):
    """Orthogonal init with gain sqrt(2), zero biases (as the notebook asks)."""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class Model(nn.Module):
    """Nature DQN conv net with two output heads: logits and state value."""

    def __init__(self, n_actions):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
        )
        self.policy_head = nn.Linear(512, n_actions)
        self.value_head = nn.Linear(512, 1)
        self.apply(_init_layer)

    def forward(self, obs):
        # obs: (N, 4, 84, 84) float32 in [0, 1]
        h = self.features(obs)
        logits = self.policy_head(h)              # (N, n_actions)
        values = self.value_head(h).squeeze(-1)   # (N,)
        return logits, values


# ---------------------------------------------------------------- 2. policy
class Policy:
    def __init__(self, model):
        self.model = model

    def reset(self):
        pass  # stateless policy; exists because EnvRunner.reset() calls it

    def act(self, inputs):
        """One forward pass, sample actions, return everything the runner stores.

        No gradients here: these tensors are only used as data (bootstrapping
        values, stored log_probs for reference). The training forward pass is
        done again inside A2C.loss where the graph is needed.
        """
        obs = torch.as_tensor(inputs, dtype=torch.float32)
        with torch.no_grad():
            logits, values = self.model(obs)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
        return {
            "actions": actions.cpu().numpy(),  # must be np.ndarray for env.step
            "logits": logits,
            "log_probs": log_probs,
            "values": values,
        }


# ------------------------------------------------- 3. value targets
class ComputeValueTargets:
    def __init__(self, policy, gamma=0.99):
        self.policy = policy
        self.gamma = gamma

    def __call__(self, trajectory):
        """Modify trajectory inplace: add 'value_targets'.

        v̂(s_t) = r_t + γ r_{t+1} + ... + γ^{T-1} r_{t+T-1} + γ^T v̂(s_{t+T}),
        where the future is cut at every reset (life lost): if resets[t] is
        True, then v̂(s_t) = r_t — no bootstrapping and no rewards beyond it.
        """
        rewards = np.asarray(trajectory["rewards"], dtype=np.float32)  # (T, N)
        resets = np.asarray(trajectory["resets"], dtype=bool)          # (T, N)
        T, N = rewards.shape

        # Bootstrap value of the last observation s_{t+T} for every env.
        last_obs = trajectory["state"]["latest_observation"]  # (N, 4, 84, 84)
        with torch.no_grad():
            _, last_values = self.policy.model(
                torch.as_tensor(last_obs, dtype=torch.float32)
            )
        acc = last_values.cpu().numpy().astype(np.float32).copy()  # (N,)

        value_targets = np.empty((T, N), dtype=np.float32)
        not_done = (~resets).astype(np.float32)
        for t in reversed(range(T)):
            # acc == discounted value of the state *after* step t
            acc = rewards[t] + self.gamma * acc * not_done[t]
            value_targets[t] = acc

        trajectory["value_targets"] = value_targets


# ------------------------------------------------- 4. merge time & batch
class MergeTimeBatch:
    """Merges first two axes typically representing time and env batch."""

    def __call__(self, trajectory):
        # Modify trajectory inplace. Every per-step entry (T of them, each
        # shaped (N, ...)) becomes a single (T*N, ...) array/tensor. All keys
        # flatten in the same (t, n) -> t * N + n order, so observations,
        # actions and value_targets stay aligned.
        T = len(trajectory["observations"])
        for key, val in list(trajectory.items()):
            if key == "state":
                continue
            if isinstance(val, list) and val:
                first = val[0]
                if isinstance(first, torch.Tensor):
                    trajectory[key] = torch.stack(val).reshape(-1, *first.shape[1:])
                elif isinstance(first, np.ndarray):
                    trajectory[key] = np.stack(val).reshape(-1, *first.shape[1:])
            elif isinstance(val, np.ndarray) and val.ndim >= 2 and val.shape[0] == T:
                # keys already stored as (T, N, ...) arrays, e.g. value_targets
                trajectory[key] = val.reshape(-1, *val.shape[2:])


# ---------------------------------------------------------------- 5. A2C
class A2C:
    def __init__(self, policy, optimizer,
                 value_loss_coef=0.25, entropy_coef=0.01, max_grad_norm=0.5):
        self.policy = policy
        self.optimizer = optimizer
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def _training_forward(self, trajectory):
        """Fresh forward pass WITH gradients over the merged batch.

        The log_probs / values stored in the trajectory come from
        policy.act() under no_grad, so they carry no graph — recompute here.
        """
        obs = torch.as_tensor(trajectory["observations"], dtype=torch.float32)
        actions = torch.as_tensor(trajectory["actions"], dtype=torch.int64)
        logits, values = self.policy.model(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)   # log π(a_t | s_t), with grads
        entropy = dist.entropy().mean()
        return values, log_probs, entropy

    def policy_loss(self, trajectory, log_probs):
        # Advantage: how much better the return turned out than predicted.
        # Detached: the value net is trained by value_loss only.
        advantages = (
            torch.as_tensor(trajectory["value_targets"])
            - torch.as_tensor(trajectory["values"])
        ).detach()
        return -(advantages * log_probs).mean()

    def value_loss(self, trajectory, values):
        targets = torch.as_tensor(trajectory["value_targets"])
        return 0.5 * ((values - targets) ** 2).mean()

    def loss(self, trajectory):
        values, log_probs, entropy = self._training_forward(trajectory)
        pl = self.policy_loss(trajectory, log_probs)
        vl = self.value_loss(trajectory, values)
        loss = pl + self.value_loss_coef * vl - self.entropy_coef * entropy
        # metrics for logging (detached)
        with torch.no_grad():
            targets = torch.as_tensor(trajectory["value_targets"])
            ss_res = ((values - targets) ** 2).sum()
            ss_tot = ((targets - targets.mean()) ** 2).sum() + 1e-8
            self.last_metrics = {
                "policy_loss": pl.item(),
                "value_loss": vl.item(),
                "entropy": entropy.item(),
                "loss": loss.item(),
                "r_squared": 1.0 - (ss_res / ss_tot).item(),
                "advantage_mean": (targets - values).mean().item(),
            }
        return loss

    def step(self, trajectory):
        loss = self.loss(trajectory)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.model.parameters(), self.max_grad_norm
        )
        self.optimizer.step()
        self.last_metrics["grad_norm"] = float(grad_norm)
        return self.last_metrics
