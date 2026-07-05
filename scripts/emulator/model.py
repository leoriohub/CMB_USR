"""PSEmulator: PyTorch-based P_S(k) emulator with GPU acceleration.

Input params (5D): [x_c, c, log10(beta), chi0, n_star]
Output: P_S(k) at 200-point fixed k-grid.
"""
import argparse
import os

import joblib
import numpy as np
import torch
import torch.nn as nn

from scripts.emulator.grid import FIXED_K_GRID


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PSNet(nn.Module):
    """Simple MLP for P_S(k) prediction: 5 → 128 → 128 → 64 → n_k."""

    def __init__(self, n_out=200, hidden_sizes=(128, 128, 64)):
        super().__init__()
        layers = []
        prev = 5
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PSEmulator:
    """PyTorch-based emulator for P_S(k) from Ezquiaga CHI parameters.

    Uses GPU if available. Normalizes inputs/outputs with sklearn StandardScaler.
    """

    def __init__(self, hidden_sizes=(128, 128, 64), lr=1e-3, batch_size=64,
                 max_epochs=2000, patience=50, val_split=0.15, verbose=False):
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_split = val_split
        self.verbose = verbose

        self.device = _get_device()
        self._net = None
        self._X_scaler = None
        self._y_scaler = None
        self._is_trained = False
        self._k_grid = None

    def train(self, params, ps, k_grid=None):
        """Train the emulator.

        Args:
            params: (n_samples, 5) array of [x_c, c, log10(beta), chi0, n_star].
            ps: (n_samples, n_k) array of P_S(k) values.
            k_grid: k-grid corresponding to ps columns. If None, uses FIXED_K_GRID.
        """
        params = np.asarray(params, dtype=np.float32)
        ps = np.asarray(ps, dtype=np.float32)

        if params.ndim != 2 or params.shape[1] != 5:
            raise ValueError(f"params must be (n_samples, 5), got {params.shape}")
        if ps.ndim != 2:
            raise ValueError(f"ps must be 2D, got shape {ps.shape}")
        if params.shape[0] != ps.shape[0]:
            raise ValueError(f"params ({params.shape[0]}) and ps ({ps.shape[0]}) "
                             f"must have same number of samples")

        n_k = ps.shape[1]
        self._k_grid = FIXED_K_GRID if k_grid is None else np.asarray(k_grid, dtype=float)

        valid = np.all(np.isfinite(ps), axis=1) & np.all(ps > 0, axis=1)
        X_train = params[valid]
        y_train = np.log(ps[valid])  # train in log-space

        if len(X_train) < 10:
            raise ValueError(f"Only {len(X_train)} valid samples, need at least 10")

        from sklearn.preprocessing import StandardScaler
        self._X_scaler = StandardScaler().fit(X_train)
        self._y_scaler = StandardScaler().fit(y_train)

        X_scaled = self._X_scaler.transform(X_train).astype(np.float32)
        y_scaled = self._y_scaler.transform(y_train).astype(np.float32)

        n = len(X_scaled)
        n_val = max(1, int(n * self.val_split))
        idx = np.random.RandomState(42).permutation(n)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        X_val_t = torch.from_numpy(X_scaled[val_idx]).to(self.device)
        y_val_t = torch.from_numpy(y_scaled[val_idx]).to(self.device)
        X_t = torch.from_numpy(X_scaled[train_idx]).to(self.device)
        y_t = torch.from_numpy(y_scaled[train_idx]).to(self.device)

        self._net = PSNet(n_out=n_k, hidden_sizes=self.hidden_sizes).to(self.device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best_loss = float("inf")
        best_state = None
        no_improve = 0

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=min(self.batch_size, len(X_t)),
                                              shuffle=True)

        for epoch in range(self.max_epochs):
            self._net.train()
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_fn(self._net(batch_X), batch_y)
                loss.backward()
                optimizer.step()

            self._net.eval()
            with torch.no_grad():
                val_loss = loss_fn(self._net(X_val_t), y_val_t).item()

            if epoch % 100 == 99 and self.verbose:
                print(f"  Epoch {epoch+1}, val_loss = {val_loss:.6e}")

            if val_loss < best_loss - 1e-8:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    if self.verbose:
                        print(f"  Early stop at epoch {epoch+1} (val_loss={val_loss:.6e})")
                    break

        self._net.load_state_dict(best_state)
        self._net.eval()
        self._is_trained = True

    @torch.no_grad()
    def _predict_numpy(self, X):
        """Internal: predict on (N, 5) numpy array, return (N, n_k) numpy."""
        self._net.eval()
        X_scaled = self._X_scaler.transform(X).astype(np.float32)
        X_t = torch.from_numpy(X_scaled).to(self.device)
        y_pred = self._net(X_t).cpu().numpy()
        log_ps = self._y_scaler.inverse_transform(y_pred)
        return np.exp(np.maximum(log_ps, -700))

    def predict(self, xc, c, beta, chi0, n_star):
        """Predict P_S(k) for a single parameter set.

        Returns (k_grid, P_S) tuple.
        """
        if not self._is_trained:
            raise RuntimeError("Emulator not trained. Call train() first.")
        log_beta = np.log10(max(float(beta), 1e-300))
        X = np.array([[xc, c, log_beta, chi0, n_star]], dtype=np.float32)
        ps = self._predict_numpy(X)[0]
        return self._k_grid.copy(), ps

    def predict_batch(self, param_grid):
        """Predict P_S(k) for multiple parameter sets.

        Args:
            param_grid: List of dicts, each with keys
                        'x_c', 'c', 'beta', 'chi0', 'n_star'.

        Returns:
            (k_grid, P_S_array) where P_S_array has shape (n, n_k).
        """
        if not self._is_trained:
            raise RuntimeError("Emulator not trained. Call train() first.")
        rows = []
        for entry in param_grid:
            beta = float(entry["beta"])
            log_beta = np.log10(max(beta, 1e-300))
            rows.append([entry["x_c"], entry["c"], log_beta,
                         entry["chi0"], entry["n_star"]])
        X = np.array(rows, dtype=np.float32)
        ps = self._predict_numpy(X)
        return self._k_grid.copy(), ps

    def score(self, params, ps):
        """R² score on held-out data."""
        if not self._is_trained:
            raise RuntimeError("Emulator not trained.")
        params = np.asarray(params, dtype=np.float32)
        ps = np.asarray(ps, dtype=np.float32)
        valid = np.all(np.isfinite(ps), axis=1) & np.all(ps > 0, axis=1)
        X = params[valid]
        y = np.log(ps[valid])
        if len(X) == 0:
            return float("nan")
        pred = self._predict_numpy(X)
        log_pred = np.log(np.maximum(pred, 1e-300))
        ss_res = np.sum((y - log_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        if ss_tot == 0:
            return float("nan")
        return float(1.0 - ss_res / ss_tot)

    def save(self, path):
        """Serialize the emulator to a joblib file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "net_state": self._net.state_dict() if self._net else None,
            "X_scaler": self._X_scaler,
            "y_scaler": self._y_scaler,
            "is_trained": self._is_trained,
            "k_grid": self._k_grid,
            "hidden_sizes": self.hidden_sizes,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
        }
        joblib.dump(data, path)
        return path

    @classmethod
    def load(cls, path):
        """Deserialize a saved emulator from a joblib file."""
        data = joblib.load(path)
        emu = cls(
            hidden_sizes=data.get("hidden_sizes", (128, 128, 64)),
            lr=data.get("lr", 1e-3),
            batch_size=data.get("batch_size", 64),
            max_epochs=data.get("max_epochs", 2000),
            patience=data.get("patience", 50),
        )
        emu._X_scaler = data["X_scaler"]
        emu._y_scaler = data["y_scaler"]
        emu._is_trained = data["is_trained"]
        emu._k_grid = data["k_grid"]

        state = data["net_state"]
        if state is not None:
            n_k = emu._k_grid.shape[0] if emu._k_grid is not None else 200
            emu._net = PSNet(n_out=n_k, hidden_sizes=emu.hidden_sizes).to(emu.device)
            emu._net.load_state_dict(state)
            emu._net.eval()
        return emu

    @property
    def is_trained(self):
        return self._is_trained

    def __repr__(self):
        status = "trained" if self._is_trained else "untrained"
        return f"<PSEmulator {status} torch({','.join(str(h) for h in self.hidden_sizes)})>"


def train_emulator(data_path, verbose=False):
    """Convenience function: load .npz, train, return PSEmulator."""
    data = np.load(data_path)
    params = data["params"]
    ps = data["P_S"]
    k_grid = data.get("k_grid", FIXED_K_GRID)
    emu = PSEmulator(verbose=verbose)
    device_name = emu.device.type
    print(f"Training on {device_name.upper()} — {params.shape[0]} samples, {ps.shape[1]} k-points...")
    emu.train(params, ps, k_grid=k_grid)
    r2 = emu.score(params, ps)
    print(f"Training R²: {r2:.6f}")
    return emu


def main():
    parser = argparse.ArgumentParser(
        description="Train PSEmulator from .npz data file."
    )
    parser.add_argument("data_path", type=str,
                        help="Path to .npz training data (keys: params, P_S, k_grid)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path for model (default: auto from data_path)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show training progress")
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.data_path))[0]
        args.output = os.path.join(os.path.dirname(args.data_path), f"{base}_model.joblib")
    emu = train_emulator(args.data_path, verbose=args.verbose)
    emu.save(args.output)
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
