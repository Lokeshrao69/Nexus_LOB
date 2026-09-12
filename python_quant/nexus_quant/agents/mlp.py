"""Tiny NumPy neural-net core for the execution agent (zero external deps).

The whole RL stack (MLP forward/backward, Adam, the PPO learner) is pure
NumPy so that:

* training is byte-reproducible across machines (no GPU nondeterminism,
  no library-version drift), and
* the actor/critic are a handful of small arrays — inspectable, dumpable,
  and swappable for a torch/GRPO variant later without touching the env.

Two network types are used by ``ppo.PPOPolicy``:

* **actor** — ``out_act="tanh"``: outputs a mean action in (-1,1).
* **critic** — ``out_act="linear"``: outputs a scalar value estimate.

Gradients are hand-derived (no autograd dependency); correctness is covered
by a finite-difference check in ``test_ppo_agent.py``.
"""

from __future__ import annotations

import numpy as np

Activation = str


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _tanh_grad(z: np.ndarray) -> np.ndarray:
    # derivative of tanh wrt PRE-activation z, in terms of z.
    return 1.0 - np.tanh(z) ** 2


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(np.float64)


def _identity(x: np.ndarray) -> np.ndarray:
    return x


def _identity_grad(_z: np.ndarray) -> np.ndarray:
    return np.ones_like(_z)


_FWD: dict[str, Activation] = {"tanh": _tanh, "relu": _relu, "linear": _identity}
_GRAD: dict[str, Activation] = {
    "tanh": _tanh_grad,
    "relu": _relu_grad,
    "linear": _identity_grad,
}


class MLP:
    """Fully-connected feed-forward net with manual forward/backward.

    ``forward`` caches per-layer activations; ``backward`` consumes a
    gradient of a scalar loss with respect to the *output* and returns
    ``(dW, db)`` per layer (same order as ``self.params``), updating nothing.
    """

    def __init__(
        self,
        dims: tuple[int, ...],
        hidden_act: str = "tanh",
        *,
        out_act: str = "linear",
        out_scale: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.dims = tuple(int(d) for d in dims)
        self.hidden_act = hidden_act
        self.out_act = out_act
        self.out_scale = float(out_scale)
        rng = np.random.default_rng(seed)
        self.params: list[list[np.ndarray]] = []
        for i in range(len(self.dims) - 1):
            fan_in = self.dims[i]
            scale = 1.0 / np.sqrt(fan_in)  # Glorot-ish; keeps tanh(0)≈0 warm start
            W = rng.normal(0.0, scale, size=(self.dims[i + 1], fan_in))
            b = np.zeros(self.dims[i + 1], dtype=np.float64)
            self.params.append([W, b])
        self._cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # ------------------------------------------------------------------ #
    def param_list(self) -> list[np.ndarray]:
        """Flat, *in-place-aliased* list of every W and b (for Adam)."""
        return [p for layer in self.params for p in layer]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """``out = out_scale * act_last(...)`` for input shape (..., in).

        Stashes ``(x_before, z, a)`` per layer for ``backward``. Bounds the
        last layer with ``out_act``; hidden layers use ``hidden_act``.
        """
        x = np.asarray(x, dtype=np.float64)
        self._cache = []
        h = x
        n_layers = len(self.params)
        for i, (W, b) in enumerate(self.params):
            z = h @ W.T + b
            act = _FWD[self.out_act] if i == n_layers - 1 else _FWD[self.hidden_act]
            a = act(z)
            self._cache.append((h, z, a))
            h = a
        return h * self.out_scale

    def backward(self, d_out: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Backprop ``d_out`` = d(loss)/d(out) for each sample in the batch.

        Returned gradient is the *summed* contribution of the batch —
        the caller scales by ``1/n`` if the loss is a mean.
        """
        d_out = np.asarray(d_out, dtype=np.float64)
        # Accept a per-sample 1-D gradient (shape (batch,)) for a 1-column
        # output and broadcast it to the network's actual output shape.
        if self._cache and d_out.ndim < self._cache[-1][2].ndim:
            d_out = d_out.reshape(self._cache[-1][2].shape)
        d_out = d_out * self.out_scale
        grads: list[tuple[np.ndarray, np.ndarray]] = [None] * len(self.params)  # type: ignore[list-item]
        d = d_out
        for i in range(len(self.params) - 1, -1, -1):
            W, _ = self.params[i]
            x_before, z, _a = self._cache[i]
            act_deriv = _GRAD[self.out_act] if i == len(self.params) - 1 else _GRAD[self.hidden_act]
            dz = d * act_deriv(z)
            db = dz.sum(axis=0)
            dW = dz.T @ x_before
            grads[i] = (dW, db)
            if i > 0:
                d = dz @ W
        return grads  # type: ignore[return-value]  # mypy: list is fully populated


def clip_grad_norm(param_grads: list[np.ndarray], max_norm: float) -> None:
    """In-place global-norm clipping on a flat list of gradient arrays."""
    total = 0.0
    for g in param_grads:
        total += float(np.sum(g * g))
    norm = float(np.sqrt(total))
    if norm > max_norm > 0.0:
        for g in param_grads:
            g *= max_norm / norm


class Adam:
    """Adam optimizer updating ``params`` in place (BiasCorrected, β1=0.9)."""

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.params = params
        self.lr = float(lr)
        self.b1, self.b2 = betas
        self.eps = float(eps)
        self.m: list[np.ndarray] = [np.zeros_like(p) for p in self.params]
        self.v: list[np.ndarray] = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1
        t = self.t
        b1t = 1.0 - self.b1**t
        b2t = 1.0 - self.b2**t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * (g * g)
            mhat = m / b1t
            vhat = v / b2t
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)