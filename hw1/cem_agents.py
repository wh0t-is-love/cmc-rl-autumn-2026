"""Picklable agents for deep / continuous CEM homework."""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor


def mountain_features(s):
    x, v = np.asarray(s, dtype=np.float64)
    return np.array(
        [
            x,
            v,
            x * v,
            x * x,
            v * v,
            np.sin(3 * x),
            np.cos(3 * x),
            v * np.cos(3 * x),
            np.sign(v),
            abs(v),
        ],
        dtype=np.float64,
    )


class FeatureMLPAgent:
    """MLPClassifier on hand-crafted MountainCar features."""

    def __init__(self, hidden=(200, 200), n_actions=3):
        self.n_actions = n_actions
        self.mlp = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="tanh",
            alpha=1e-5,
            learning_rate_init=1e-3,
        )
        self._fitted = False

    def _X(self, states):
        return np.asarray([mountain_features(s) for s in states], dtype=np.float64)

    def partial_fit(self, states, actions):
        X = self._X(states)
        y = np.asarray(actions)
        if not self._fitted:
            self.mlp.partial_fit(X, y, classes=np.arange(self.n_actions))
            self._fitted = True
        else:
            self.mlp.partial_fit(X, y)
        return self

    def predict_proba(self, states):
        return self.mlp.predict_proba(self._X(states))


class SoftmaxLinearAgent:
    """Softmax(W @ features(s)). Used as CEM search policy, then distilled."""

    def __init__(self, W):
        self.W = np.asarray(W, dtype=np.float64)

    def predict_proba(self, states):
        X = np.asarray([mountain_features(s) for s in states], dtype=np.float64)
        logits = X @ self.W.T
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def partial_fit(self, states, actions):
        # not trained via partial_fit; CEM updates W externally
        return self


class ContinuousNoiseAgent:
    """MLPRegressor mean action + Gaussian exploration noise."""

    def __init__(self, state_dim=2, action_dim=1, hidden=(128, 128), noise_std=0.3):
        self.action_dim = action_dim
        self.noise_std = noise_std
        self.mlp = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="tanh",
            alpha=1e-4,
            learning_rate_init=1e-3,
        )
        dummy_x = np.zeros((max(2, action_dim), state_dim))
        dummy_y = np.zeros((max(2, action_dim), action_dim))
        self.mlp.partial_fit(dummy_x, dummy_y)

    def act(self, s, deterministic=False):
        mu = np.asarray(self.mlp.predict(np.asarray(s, dtype=np.float64).reshape(1, -1)), dtype=np.float64).reshape(-1)
        if not deterministic:
            mu = mu + self.noise_std * np.random.randn(self.action_dim)
        return np.clip(mu, -1.0, 1.0).astype(np.float64)

    def predict(self, states, deterministic=False):
        X = np.asarray(states, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        mu = np.asarray(self.mlp.predict(X), dtype=np.float64)
        if mu.ndim == 1:
            mu = mu.reshape(-1, self.action_dim)
        if deterministic:
            return mu
        return mu + self.noise_std * np.random.randn(*mu.shape)

    def partial_fit(self, states, actions):
        X = np.asarray(states, dtype=np.float64)
        Y = np.asarray(actions, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, self.action_dim)
        self.mlp.partial_fit(X, Y)
        return self
