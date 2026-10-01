from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    LimeTabularExplainer = None

from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KernelDensity

from data_pipeline import DatasetSplit, split_labeled_unlabeled
from er_trpo import ERTRPOAgent, ReplayBuffer, Transition
from models import PriceFeatureModel


@dataclass
class AgentConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    activation: str
    dropout: float
    beta: float


@dataclass
class SearchConfig:
    agent1: AgentConfig
    agent2: AgentConfig
    delta_init: float
    mu: float
    vartheta: float
    lstm_layers: int
    mlp_layers: int


@dataclass
class Metrics:
    rmse: float
    mape: float
    rmsle: float
    r2: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    mape = float(100.0 * np.mean(np.abs((y_true - y_pred) / denom)))
    yt = np.maximum(y_true, 0.0)
    yp = np.maximum(y_pred, 0.0)
    rmsle = float(np.sqrt(np.mean((np.log1p(yt) - np.log1p(yp)) ** 2)))
    r2 = float(r2_score(y_true, y_pred))
    return Metrics(rmse, mape, rmsle, r2)


class LIMEFeatureScorer:
    def __init__(self, background: np.ndarray, model: PriceFeatureModel, device: str, seed: int) -> None:
        self.model = model
        self.device = torch.device(device)
        self.dim = background.shape[1]
        self.background = background.astype(np.float64)
        self.rng = np.random.default_rng(seed)
        self.explainer = None
        if LimeTabularExplainer is not None:
            self.explainer = LimeTabularExplainer(
                training_data=background,
                mode="regression",
                feature_names=[f"latent_{i}" for i in range(self.dim)],
                discretize_continuous=False,
                random_state=seed,
            )

    def _predict(self, z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            latent = torch.as_tensor(z, dtype=torch.float32, device=self.device)
            mask = torch.ones_like(latent)
            return self.model.predict_from_masked_latent(latent, mask).cpu().numpy()

    def scores(self, latent: np.ndarray, num_samples: int = 256) -> np.ndarray:
        if self.explainer is not None:
            explanation = self.explainer.explain_instance(
                latent.astype(np.float64),
                self._predict,
                num_features=self.dim,
                num_samples=num_samples,
            )
            values = np.zeros(self.dim, dtype=np.float32)
            for idx, weight in explanation.local_exp[1]:
                if 0 <= idx < self.dim:
                    values[idx] = abs(float(weight))
        else:
            std = self.background.std(axis=0) + 1e-6
            noise = self.rng.normal(0.0, std, size=(num_samples, self.dim))
            perturb = latent.astype(np.float64)[None, :] + noise
            perturb[0] = latent
            distances = np.linalg.norm((perturb - latent[None, :]) / std[None, :], axis=1)
            kernel_width = 0.75 * np.sqrt(self.dim)
            weights = np.exp(-(distances ** 2) / (kernel_width ** 2 + 1e-12))
            targets = self._predict(perturb)
            surrogate = Ridge(alpha=1.0)
            surrogate.fit(perturb, targets, sample_weight=weights)
            values = np.abs(surrogate.coef_).astype(np.float32)
        maximum = float(values.max())
        if maximum > 0:
            values /= maximum
        return values


class DualAgentTrainer:
    def __init__(
        self,
        input_dim: int,
        config: SearchConfig,
        device: str,
        rho: float,
        seed: int,
        labeled_fraction: float,
        clusters: int = 8,
    ) -> None:
        self.config = config
        self.device = device
        self.rho = rho
        self.seed = seed
        self.labeled_fraction = labeled_fraction
        self.clusters = clusters
        self.rng = np.random.default_rng(seed)
        hidden_dim = 64
        self.model = PriceFeatureModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            lstm_layers=config.lstm_layers,
            mlp_layers=config.mlp_layers,
            activation=config.agent2.activation,
            dropout=config.agent2.dropout,
        ).to(device)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=config.agent2.learning_rate)
        self.agent1 = ERTRPOAgent(
            state_dim=hidden_dim + 1,
            action_dim=2,
            learning_rate=config.agent1.learning_rate,
            entropy_beta=config.agent1.beta,
            homotopy_rho=rho,
            device=device,
            dropout=config.agent1.dropout,
        )
        self.agent2 = ERTRPOAgent(
            state_dim=hidden_dim * 3,
            action_dim=2,
            learning_rate=config.agent2.learning_rate,
            entropy_beta=config.agent2.beta,
            homotopy_rho=rho,
            device=device,
            dropout=config.agent2.dropout,
        )
        self.buffer1 = ReplayBuffer()
        self.buffer2 = ReplayBuffer()
        self.hidden_dim = hidden_dim

    def _train_predictor(self, x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int) -> None:
        self.model.train()
        n = len(x)
        for _ in range(max(1, epochs)):
            order = self.rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                xb = torch.as_tensor(x[idx], dtype=torch.float32, device=self.device)
                yb = torch.as_tensor(y[idx], dtype=torch.float32, device=self.device)
                pred, _, _ = self.model(xb)
                loss = torch.mean((pred - yb) ** 2)
                self.model_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.model_optimizer.step()

    @torch.no_grad()
    def _latent_prediction(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        xb = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        pred, relevance, latent = self.model(xb)
        return latent.cpu().numpy(), pred.cpu().numpy(), relevance.cpu().numpy()

    def _fit_kmeans(self, x_l: np.ndarray) -> KMeans:
        latent, _, _ = self._latent_prediction(x_l)
        k = min(self.clusters, max(2, len(latent) // 4), len(latent))
        return KMeans(n_clusters=k, random_state=self.seed, n_init=10).fit(latent)

    @staticmethod
    def _cluster_entropy(latent: np.ndarray, centers: np.ndarray) -> float:
        distances = np.linalg.norm(centers - latent[None, :], axis=1)
        logits = -distances
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum() + 1e-12
        return float(-np.sum(probs * np.log(probs + 1e-12)))

    def _run_ssl_epoch(
        self,
        x_l: np.ndarray,
        y_l: np.ndarray,
        x_u: np.ndarray,
        epoch: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        kmeans = self._fit_kmeans(x_l)
        delta = self.config.delta_init * math.exp(-self.config.mu * epoch)
        remaining: list[np.ndarray] = []
        accepted_x: list[np.ndarray] = []
        accepted_y: list[float] = []
        order = self.rng.permutation(len(x_u))
        for pos, idx in enumerate(order):
            sample = x_u[idx : idx + 1]
            latent, pred, _ = self._latent_prediction(sample)
            state = np.concatenate([latent[0], [pred[0]]]).astype(np.float32)
            action = self.agent1.act(state)
            entropy = self._cluster_entropy(latent[0], kmeans.cluster_centers_)
            if action == 1 and entropy <= delta:
                reward = 1.0
                accepted_x.append(sample[0])
                accepted_y.append(float(pred[0]))
            elif action == 1:
                reward = -1.0
                remaining.append(sample[0])
            else:
                reward = 0.0
                remaining.append(sample[0])
            next_state = state.copy()
            done = pos == len(order) - 1
            self.buffer1.add(Transition(state, action, reward, next_state, done))

        if accepted_x:
            x_l = np.concatenate([x_l, np.asarray(accepted_x, dtype=np.float32)], axis=0)
            y_l = np.concatenate([y_l, np.asarray(accepted_y, dtype=np.float32)], axis=0)
            self._train_predictor(x_l, y_l, epochs=1, batch_size=self.config.agent2.batch_size)
        batch = self.buffer1.sample(self.config.agent1.batch_size, self.rng)
        self.agent1.update(batch)
        new_u = np.asarray(remaining, dtype=np.float32) if remaining else np.empty((0,) + x_u.shape[1:], dtype=np.float32)
        return x_l, y_l, new_u

    def _run_agent2_epoch(self, x_l: np.ndarray, y_l: np.ndarray, lime_samples: int = 128) -> None:
        latents, _, _ = self._latent_prediction(x_l)
        bg_idx = self.rng.choice(len(latents), size=min(512, len(latents)), replace=False)
        lime = LIMEFeatureScorer(latents[bg_idx], self.model, self.device, self.seed)
        order = self.rng.permutation(len(x_l))
        max_instances = min(len(order), max(32, self.config.agent2.batch_size * 2))
        for idx in order[:max_instances]:
            x = x_l[idx : idx + 1]
            y = float(y_l[idx])
            latent, _, relevance = self._latent_prediction(x)
            z = latent[0].astype(np.float32)
            rel = relevance[0].astype(np.float32)
            lime_scores = lime.scores(z, num_samples=lime_samples)
            active = np.zeros(self.hidden_dim, dtype=np.float32)
            done = False
            steps = 0
            while not done and steps <= self.hidden_dim:
                state = np.concatenate([z, rel, active]).astype(np.float32)
                if active.sum() >= self.hidden_dim:
                    action = 1
                else:
                    action = self.agent2.act(state)
                if action == 0:
                    masked_scores = rel.copy()
                    masked_scores[active > 0] = -np.inf
                    feature_idx = int(np.argmax(masked_scores))
                    active[feature_idx] = 1.0
                    reward = self.config.vartheta * float(lime_scores[feature_idx])
                    next_state = np.concatenate([z, rel, active]).astype(np.float32)
                else:
                    mask_t = torch.as_tensor(active, dtype=torch.float32, device=self.device).unsqueeze(0)
                    latent_t = torch.as_tensor(z, dtype=torch.float32, device=self.device).unsqueeze(0)
                    if active.sum() == 0:
                        mask_t = torch.ones_like(mask_t)
                    pred = float(self.model.predict_from_masked_latent(latent_t, mask_t).detach().cpu().item())
                    reward = -float((y - pred) ** 2)
                    next_state = state.copy()
                    done = True
                self.buffer2.add(Transition(state, action, reward, next_state, done))
                steps += 1
        batch = self.buffer2.sample(self.config.agent2.batch_size, self.rng)
        self.agent2.update(batch)

    def fit(self, split: DatasetSplit, budget_epochs: int) -> None:
        x_l, y_l, x_u, _ = split_labeled_unlabeled(
            split.x_train, split.y_train, self.labeled_fraction, self.seed
        )
        warmup = max(1, min(5, budget_epochs // 5))
        self._train_predictor(x_l, y_l, epochs=warmup, batch_size=self.config.agent2.batch_size)
        total_epochs = min(
            int(budget_epochs),
            int(self.config.agent1.epochs),
            int(self.config.agent2.epochs),
        )
        for epoch in range(total_epochs):
            if len(x_u) > 0:
                x_l, y_l, x_u = self._run_ssl_epoch(x_l, y_l, x_u, epoch)
            self._run_agent2_epoch(x_l, y_l)
            self._train_predictor(x_l, y_l, epochs=1, batch_size=self.config.agent2.batch_size)

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        preds = []
        for start in range(0, len(x), 2048):
            xb = torch.as_tensor(x[start : start + 2048], dtype=torch.float32, device=self.device)
            pred, _, _ = self.model(xb)
            preds.append(pred.cpu().numpy())
        return np.concatenate(preds)


class SearchSpace:
    activations = ["Tanh", "Leaky ReLU", "Linear", "ReLU", "Sigmoid"]

    def sample(self, rng: np.random.Generator) -> SearchConfig:
        def agent() -> AgentConfig:
            return AgentConfig(
                batch_size=int(rng.integers(8, 257)),
                epochs=int(rng.integers(32, 1025)),
                learning_rate=float(np.exp(rng.uniform(np.log(1e-4), np.log(5e-2)))),
                activation=str(rng.choice(self.activations)),
                dropout=float(rng.uniform(0.0, 1.0)),
                beta=float(np.exp(rng.uniform(np.log(1e-4), np.log(1e-1)))),
            )
        return SearchConfig(
            agent1=agent(),
            agent2=agent(),
            delta_init=float(rng.uniform(0.1, 1.0)),
            mu=float(np.exp(rng.uniform(np.log(1e-4), np.log(1e-1)))),
            vartheta=float(rng.uniform(0.0, 1.0)),
            lstm_layers=int(rng.integers(1, 9)),
            mlp_layers=int(rng.integers(1, 9)),
        )

    @staticmethod
    def vector(config: SearchConfig) -> np.ndarray:
        activation_map = {name: i / 4.0 for i, name in enumerate(SearchSpace.activations)}
        return np.array([
            config.agent1.batch_size / 256,
            np.log(config.agent1.epochs) / np.log(1024),
            np.log(config.agent1.learning_rate / 1e-4) / np.log(5e-2 / 1e-4),
            activation_map[config.agent1.activation],
            config.agent1.dropout,
            np.log(config.agent1.beta / 1e-4) / np.log(1e-1 / 1e-4),
            config.agent2.batch_size / 256,
            np.log(config.agent2.epochs) / np.log(1024),
            np.log(config.agent2.learning_rate / 1e-4) / np.log(5e-2 / 1e-4),
            activation_map[config.agent2.activation],
            config.agent2.dropout,
            np.log(config.agent2.beta / 1e-4) / np.log(1e-1 / 1e-4),
            config.delta_init,
            np.log(config.mu / 1e-4) / np.log(1e-1 / 1e-4),
            config.vartheta,
            config.lstm_layers / 8,
            config.mlp_layers / 8,
        ], dtype=np.float64)


class HomotopyBOHB:
    def __init__(
        self,
        evaluator: Callable[[SearchConfig, int, float], float],
        seed: int,
        min_budget: int = 10,
        max_budget: int = 90,
        eta: int = 3,
        initial_configs: int = 9,
    ) -> None:
        self.evaluator = evaluator
        self.rng = np.random.default_rng(seed)
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.eta = eta
        self.initial_configs = initial_configs
        self.space = SearchSpace()
        self.history: list[tuple[SearchConfig, float, int, float]] = []

    def _model_based_candidate(self) -> SearchConfig:
        if len(self.history) < 8:
            return self.space.sample(self.rng)
        recent = sorted(self.history, key=lambda z: z[1])
        n_good = max(3, int(math.ceil(0.15 * len(recent))))
        good = np.stack([self.space.vector(c) for c, _, _, _ in recent[:n_good]])
        bad = np.stack([self.space.vector(c) for c, _, _, _ in recent[n_good:]])
        if len(bad) < 2:
            return self.space.sample(self.rng)
        good_kde = KernelDensity(kernel="gaussian", bandwidth=0.15).fit(good)
        bad_kde = KernelDensity(kernel="gaussian", bandwidth=0.15).fit(bad)
        candidates = [self.space.sample(self.rng) for _ in range(64)]
        vectors = np.stack([self.space.vector(c) for c in candidates])
        score = good_kde.score_samples(vectors) - bad_kde.score_samples(vectors)
        return candidates[int(np.argmax(score))]

    def _new_candidates(self, n: int) -> list[SearchConfig]:
        out = []
        for _ in range(n):
            if self.history and self.rng.random() > 1 / 3:
                out.append(self._model_based_candidate())
            else:
                out.append(self.space.sample(self.rng))
        return out

    def optimize(self) -> SearchConfig:
        warm: list[SearchConfig] = []
        for rho in [0.0, 0.25, 0.50, 0.75, 1.0]:
            candidates = warm[: min(len(warm), self.initial_configs // 3)] + self._new_candidates(self.initial_configs)
            budget = self.min_budget
            while candidates:
                scored: list[tuple[SearchConfig, float]] = []
                for config in candidates:
                    loss = float(self.evaluator(config, budget, rho))
                    self.history.append((config, loss, budget, rho))
                    scored.append((config, loss))
                scored.sort(key=lambda z: z[1])
                if budget >= self.max_budget or len(scored) == 1:
                    warm = [x[0] for x in scored[: max(1, min(5, len(scored)))]]
                    break
                keep = max(1, len(scored) // self.eta)
                candidates = [x[0] for x in scored[:keep]]
                budget = min(self.max_budget, budget * self.eta)
        final = [h for h in self.history if h[3] == 1.0]
        if not final:
            final = self.history
        return min(final, key=lambda z: z[1])[0]


def evaluate_config(
    config: SearchConfig,
    split: DatasetSplit,
    budget: int,
    rho: float,
    device: str,
    seed: int,
    labeled_fraction: float,
    max_search_rows: int,
) -> float:
    if max_search_rows > 0 and len(split.x_train) > max_search_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(split.x_train), size=max_search_rows, replace=False)
        small = DatasetSplit(
            split.x_train[idx], split.y_train[idx],
            split.x_val, split.y_val,
            split.x_test, split.y_test,
            split.feature_names,
        )
    else:
        small = split
    trainer = DualAgentTrainer(
        input_dim=small.x_train.shape[-1],
        config=config,
        device=device,
        rho=rho,
        seed=seed,
        labeled_fraction=labeled_fraction,
    )
    trainer.fit(small, budget_epochs=budget)
    pred = trainer.predict(small.x_val)
    return float(np.sqrt(mean_squared_error(small.y_val, pred)))


def train_final_and_evaluate(
    config: SearchConfig,
    split: DatasetSplit,
    device: str,
    seed: int,
    labeled_fraction: float,
) -> tuple[DualAgentTrainer, Metrics]:
    trainer = DualAgentTrainer(
        input_dim=split.x_train.shape[-1],
        config=config,
        device=device,
        rho=1.0,
        seed=seed,
        labeled_fraction=labeled_fraction,
    )
    final_budget = min(90, config.agent1.epochs, config.agent2.epochs)
    trainer.fit(split, budget_epochs=final_budget)
    pred = trainer.predict(split.x_test)
    return trainer, regression_metrics(split.y_test, pred)
