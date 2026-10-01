from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn

from models import PolicyNetwork, ValueNetwork


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self.data: list[Transition] = []

    def add(self, transition: Transition) -> None:
        self.data.append(transition)
        if len(self.data) > self.capacity:
            del self.data[: len(self.data) - self.capacity]

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[Transition]:
        if not self.data:
            return []
        idx = rng.choice(len(self.data), size=min(batch_size, len(self.data)), replace=False)
        return [self.data[int(i)] for i in idx]

    def __len__(self) -> int:
        return len(self.data)


def flat_parameters(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def set_flat_parameters(model: nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    for p in model.parameters():
        size = p.numel()
        p.data.copy_(flat[offset : offset + size].view_as(p))
        offset += size


def flat_grad(loss: torch.Tensor, model: nn.Module, create_graph: bool = False) -> torch.Tensor:
    grads = torch.autograd.grad(loss, tuple(model.parameters()), create_graph=create_graph, allow_unused=False)
    return torch.cat([g.reshape(-1) for g in grads])


def conjugate_gradient(fvp, b: torch.Tensor, steps: int = 10, residual_tol: float = 1e-10) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(steps):
        z = fvp(p)
        denom = torch.dot(p, z).clamp_min(1e-12)
        alpha = rdotr / denom
        x = x + alpha * p
        r = r - alpha * z
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / rdotr.clamp_min(1e-12)
        p = r + beta * p
        rdotr = new_rdotr
    return x


class ERTRPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float,
        entropy_beta: float,
        homotopy_rho: float,
        gamma: float = 0.99,
        max_kl: float = 0.01,
        damping: float = 0.1,
        device: str = "cpu",
        dropout: float = 0.0,
    ) -> None:
        self.device = torch.device(device)
        self.policy = PolicyNetwork(state_dim, action_dim, dropout=dropout).to(self.device)
        self.value = ValueNetwork(state_dim).to(self.device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=learning_rate)
        self.entropy_beta = float(entropy_beta)
        self.homotopy_rho = float(homotopy_rho)
        self.gamma = float(gamma)
        self.max_kl = float(max_kl)
        self.damping = float(damping)

    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.policy.distribution(s)
        if deterministic:
            return int(torch.argmax(dist.logits, dim=-1).item())
        return int(dist.sample().item())

    def _batch_tensors(self, transitions: Iterable[Transition]):
        items = list(transitions)
        states = torch.as_tensor(np.stack([t.state for t in items]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([t.action for t in items], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([t.reward for t in items], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([t.next_state for t in items]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([t.done for t in items], dtype=torch.float32, device=self.device)
        return states, actions, rewards, next_states, dones

    def update(self, transitions: Iterable[Transition]) -> dict[str, float]:
        items = list(transitions)
        if len(items) < 4:
            return {"policy_objective": 0.0, "kl": 0.0, "value_loss": 0.0}
        states, actions, rewards, next_states, dones = self._batch_tensors(items)

        with torch.no_grad():
            old_dist = self.policy.distribution(states)
            old_logp = old_dist.log_prob(actions)
            old_probs = old_dist.probs
            values = self.value(states)
            next_values = self.value(next_states)
            returns = rewards + self.gamma * (1.0 - dones) * next_values
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        for _ in range(5):
            pred_values = self.value(states)
            value_loss = torch.mean((pred_values - returns) ** 2)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        def surrogate() -> torch.Tensor:
            dist = self.policy.distribution(states)
            ratio = torch.exp(dist.log_prob(actions) - old_logp)
            entropy = dist.entropy().mean()
            base = torch.mean(ratio * advantages)
            return base + self.homotopy_rho * self.entropy_beta * entropy

        def mean_kl() -> torch.Tensor:
            new_dist = self.policy.distribution(states)
            kl = torch.sum(old_probs * (torch.log(old_probs + 1e-8) - torch.log(new_dist.probs + 1e-8)), dim=1)
            return kl.mean()

        objective = surrogate()
        g = flat_grad(objective, self.policy).detach()

        def fvp(v: torch.Tensor) -> torch.Tensor:
            kl = mean_kl()
            grad_kl = flat_grad(kl, self.policy, create_graph=True)
            kl_v = torch.dot(grad_kl, v)
            hvp = flat_grad(kl_v, self.policy).detach()
            return hvp + self.damping * v

        step_dir = conjugate_gradient(fvp, g, steps=10)
        shs = 0.5 * torch.dot(step_dir, fvp(step_dir))
        scale = torch.sqrt(self.max_kl / (shs + 1e-8))
        full_step = step_dir * scale
        old_params = flat_parameters(self.policy)
        old_obj = float(objective.detach().item())

        accepted = False
        final_kl = 0.0
        for fraction in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]:
            candidate = old_params + fraction * full_step
            set_flat_parameters(self.policy, candidate)
            new_obj = float(surrogate().detach().item())
            final_kl = float(mean_kl().detach().item())
            if np.isfinite(new_obj) and new_obj > old_obj and final_kl <= self.max_kl:
                accepted = True
                break
        if not accepted:
            set_flat_parameters(self.policy, old_params)
            final_kl = float(mean_kl().detach().item())

        return {
            "policy_objective": float(surrogate().detach().item()),
            "kl": final_kl,
            "value_loss": float(value_loss.detach().item()),
        }
