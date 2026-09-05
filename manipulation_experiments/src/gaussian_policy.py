#!/usr/bin/env python
"""Diagonal-Gaussian action-chunk actor for manipulation PPO.

Reuses a frozen FlowMatchingPolicy only as vision / state encoder and for
action normalization. The policy itself is π(a|s) = N(μ(s), σ), not a flow.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

from lerobot.common.constants import ACTION


class GaussianChunkPolicy(nn.Module):
    """Chunked Gaussian policy on top of a frozen observation encoder."""

    def __init__(
        self,
        flow_policy: nn.Module,
        init_log_std: float = -2.302585,  # log(0.1)
        hidden_dims: Sequence[int] = (1024, 1024, 1024),
        std_min: float = 1e-3,
        std_max: float = 1.0,
    ):
        super().__init__()
        self.flow = flow_policy
        self.config = flow_policy.config
        self.model = flow_policy.model
        self.normalize_inputs = flow_policy.normalize_inputs
        self.normalize_targets = flow_policy.normalize_targets
        self.unnormalize_outputs = flow_policy.unnormalize_outputs
        self.ema_model = None
        self._ema_update_count = 0

        cond_dim = int(self.model.global_cond_dim)
        action_dim = int(self.model.action_dim)
        horizon = int(self.config.horizon)
        self.action_dim = action_dim
        self.horizon = horizon
        self.std_min = std_min
        self.std_max = std_max

        layers: list[nn.Module] = []
        prev = cond_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, horizon * action_dim))
        self.mean_net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))

        self.num_envs = None
        self.action_buffers = None
        self.log_prob_buffers = None
        self.last_step_log_prob: Tensor | None = None
        self.reset()

    def std(self) -> Tensor:
        return self.log_std.exp().clamp(self.std_min, self.std_max)

    def _mean_and_std(self, obs: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        obs_n = self.normalize_inputs(obs)
        cond = self.model.encode_observations(obs_n)
        mean = self.mean_net(cond).view(cond.shape[0], self.horizon, self.action_dim)
        std = self.std().view(1, 1, -1).expand_as(mean)
        return mean, std, cond

    def evaluate_actions(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Log-prob / entropy of a stored (unnormalized) action chunk.

        batch must contain observations at chunk start plus ``action`` of shape (B, T, A).
        """
        actions = batch[ACTION]
        mean, std, _ = self._mean_and_std(
            {k: v for k, v in batch.items() if k not in {ACTION, "mdp_x_t_path"}}
        )
        t = actions.shape[1]
        mean = mean[:, :t]
        std = std[:, :t]
        actions_n = self.normalize_targets({ACTION: actions})[ACTION]
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions_n).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    @torch.no_grad()
    def sample_chunk(self, batch: dict[str, Tensor], deterministic: bool = False) -> tuple[Tensor, Tensor]:
        mean, std, _ = self._mean_and_std(batch)
        dist = Normal(mean, std)
        z = mean if deterministic else dist.sample()
        log_prob = dist.log_prob(z).sum(dim=-1)
        actions = self.unnormalize_outputs({ACTION: z})[ACTION]
        n = int(self.config.n_action_steps)
        return actions[:, :n], log_prob[:, :n]

    def init_action_buffers(self, num_envs: int):
        self.num_envs = num_envs
        self.action_buffers = {i: deque([], maxlen=self.config.n_action_steps) for i in range(num_envs)}
        self.log_prob_buffers = {i: deque([], maxlen=self.config.n_action_steps) for i in range(num_envs)}

    def reset(self, env_ids: Tensor | None = None):
        if self.num_envs is None:
            self.action_buffers = {}
            self.log_prob_buffers = {}
            return
        if env_ids is None:
            ids = range(self.num_envs)
        else:
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(env_ids)
            ids = env_ids.tolist()
        for env_id in ids:
            self.action_buffers[env_id] = deque([], maxlen=self.config.n_action_steps)
            self.log_prob_buffers[env_id] = deque([], maxlen=self.config.n_action_steps)

    def step_ema(self):
        return

    @torch.no_grad()
    def select_action(
        self,
        batch: dict[str, Tensor],
        zero_sampling: bool = False,
        sde_sampling: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if self.num_envs is None or self.action_buffers is None:
            raise ValueError("Call init_action_buffers first.")
        need = [i for i in range(self.num_envs) if len(self.action_buffers[i]) == 0]
        if need:
            sub = {k: v[need] for k, v in batch.items()}
            chunks, lp = self.sample_chunk(sub, deterministic=bool(zero_sampling))
            for i, env_id in enumerate(need):
                self.action_buffers[env_id].extend(list(chunks[i]))
                self.log_prob_buffers[env_id].extend(list(lp[i]))

        actions = []
        logps = []
        for env_id in range(self.num_envs):
            actions.append(self.action_buffers[env_id].popleft())
            logps.append(self.log_prob_buffers[env_id].popleft())
        actions_t = torch.stack(actions)
        self.last_step_log_prob = torch.stack(logps)
        dummy = torch.zeros(
            (self.num_envs, self.config.sampling_steps, self.action_dim),
            device=actions_t.device,
            dtype=actions_t.dtype,
        )
        return actions_t, dummy

    def save_pretrained(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.flow.save_pretrained(str(path))
        torch.save(
            {
                "mean_net": self.mean_net.state_dict(),
                "log_std": self.log_std.detach().cpu(),
            },
            path / "gaussian_head.pt",
        )

    def load_gaussian_head(self, path: str | Path):
        blob = torch.load(path, map_location="cpu")
        self.mean_net.load_state_dict(blob["mean_net"])
        with torch.no_grad():
            self.log_std.copy_(blob["log_std"].to(self.log_std.device))


def distill_gaussian_from_flow(
    gaussian: GaussianChunkPolicy,
    env,
    epochs: int,
    logger=None,
) -> None:
    """Fit μ(s) to frozen flow BC chunks so PPO does not start from noise."""
    if epochs <= 0:
        return
    opt = torch.optim.Adam(gaussian.mean_net.parameters(), lr=1e-3)
    n_steps = int(gaussian.config.n_action_steps)
    obs, _ = env.reset()
    gaussian.flow.eval()
    gaussian.mean_net.train()
    last = 0.0
    for i in range(epochs):
        with torch.no_grad():
            bc_chunk, _ = gaussian.flow.predict_action_chunk(
                {k: v.clone() for k, v in obs.items()},
                zero_sampling=True,
            )
            bc_chunk = bc_chunk[:, :n_steps]
        mean, _, _ = gaussian._mean_and_std({k: v.clone() for k, v in obs.items()})
        mean = mean[:, :n_steps]
        target = gaussian.normalize_targets({ACTION: bc_chunk})[ACTION]
        loss = F.mse_loss(mean, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.detach())
        for t in range(n_steps):
            obs, _, terminated, truncated, _ = env.step(bc_chunk[:, t])
            done = terminated | truncated
            if bool(done.any()):
                obs, _ = env.reset()
                break
        if logger is not None and (i == 0 or (i + 1) % 10 == 0 or i == epochs - 1):
            logger.info(f"Gaussian BC distill {i + 1}/{epochs} mse={last:.5f}")
    gaussian.reset()
    gaussian.flow.reset()
    env.reset()
    gaussian.reset()
