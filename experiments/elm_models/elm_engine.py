"""
elm_engine.py
High-performance, low-RAM, GPU-accelerated Extreme Learning Machine engine.
Supports chunked ridge accumulation in float64 for exact precision with < 100 MB RAM footprint.
"""

import os
import gc
import pickle
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GPUChunkedELM:
    def __init__(self, n_hidden=512, alpha=1e-2, activation="tanh", seed=42):
        self.n_hidden = n_hidden
        self.alpha = float(alpha)
        self.activation = activation
        self.seed = seed
        self.input_dim = None
        self.W = None  # (d, L)
        self.b = None  # (1, L)
        self.beta = None  # (L, 1)

    def _init_weights(self, input_dim):
        self.input_dim = input_dim
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(input_dim)
        w_np = rng.uniform(-scale, scale, size=(input_dim, self.n_hidden)).astype(np.float32)
        b_np = rng.uniform(-scale, scale, size=(1, self.n_hidden)).astype(np.float32)
        self.W = torch.from_numpy(w_np).to(device)
        self.b = torch.from_numpy(b_np).to(device)

    def _activate(self, z):
        if self.activation == "tanh":
            return torch.tanh(z)
        elif self.activation == "elu":
            return torch.nn.functional.elu(z)
        elif self.activation == "relu":
            return torch.relu(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def fit_chunked(self, X_array, y_array, chunk_size=50_000):
        """
        Fit ridge regression coefficients beta = (H^T H + alpha * I)^-1 H^T Y
        accumulating in float64 chunks for zero RAM explosion and high numerical precision.
        """
        N, d = X_array.shape
        if self.W is None:
            self._init_weights(d)

        L = self.n_hidden
        HtH = torch.zeros((L, L), dtype=torch.float64, device=device)
        HtY = torch.zeros((L, 1), dtype=torch.float64, device=device)

        n_chunks = int(np.ceil(N / chunk_size))
        for c in range(n_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, N)

            # Load slice from memory map
            xc = torch.from_numpy(X_array[start:end].astype(np.float32)).to(device)
            yc = torch.from_numpy(y_array[start:end].astype(np.float32)).to(device)
            if yc.ndim == 1:
                yc = yc.unsqueeze(1)

            # Hidden activations
            hc = self._activate(xc @ self.W + self.b).to(torch.float64)
            HtH += hc.T @ hc
            HtY += hc.T @ yc.to(torch.float64)

            del xc, yc, hc

        # Regularized solve: beta = (HtH + alpha * I)^-1 HtY
        I = torch.eye(L, dtype=torch.float64, device=device)
        A = HtH + self.alpha * I
        self.beta = torch.linalg.solve(A, HtY).to(torch.float32)

        del HtH, HtY, A, I
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, X_array, chunk_size=50_000):
        """Predict physical or standardized outputs in chunks."""
        N = X_array.shape[0]
        preds = np.empty(N, dtype=np.float32)

        n_chunks = int(np.ceil(N / chunk_size))
        with torch.no_grad():
            for c in range(n_chunks):
                start = c * chunk_size
                end = min(start + chunk_size, N)

                xc = torch.from_numpy(X_array[start:end].astype(np.float32)).to(device)
                hc = self._activate(xc @ self.W + self.b)
                yc = (hc @ self.beta).squeeze(-1)
                preds[start:end] = yc.cpu().numpy()
                del xc, hc, yc

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return preds

    def save(self, filepath):
        state = {
            "n_hidden": self.n_hidden,
            "alpha": self.alpha,
            "activation": self.activation,
            "seed": self.seed,
            "input_dim": self.input_dim,
            "W": self.W.cpu().numpy(),
            "b": self.b.cpu().numpy(),
            "beta": self.beta.cpu().numpy()
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        elm = cls(n_hidden=state["n_hidden"], alpha=state["alpha"],
                  activation=state["activation"], seed=state["seed"])
        elm.input_dim = state["input_dim"]
        elm.W = torch.from_numpy(state["W"]).to(device)
        elm.b = torch.from_numpy(state["b"]).to(device)
        elm.beta = torch.from_numpy(state["beta"]).to(device)
        return elm

