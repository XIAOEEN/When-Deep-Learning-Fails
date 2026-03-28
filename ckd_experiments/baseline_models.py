"""
Baseline models for CKD progression prediction.
1. Linear ITSA (Interrupted Time Series Analysis)
2. Gaussian Process (GP) Regression
3. LSTM/GRU
4. Simple Exponential Decay
5. TE-CDE baseline (simplified)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler


@dataclass
class PredictionResult:
    """Container for prediction results."""
    predictions: np.ndarray  # (n_samples, forecast_steps, n_vars)
    uncertainties: Optional[np.ndarray] = None  # (n_samples, forecast_steps, n_vars)
    cat_ids: Optional[List] = None
    time_indices: Optional[List] = None
    metrics: Optional[Dict] = None


class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, name: str = "BaseModel"):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray,
            groups: Optional[np.ndarray] = None) -> 'BaseModel':
        """Fit the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, groups: Optional[np.ndarray] = None) -> PredictionResult:
        """Make predictions."""
        pass

    def _ensure_prediction_result(self, preds, **kwargs) -> PredictionResult:
        """Wrap predictions in PredictionResult."""
        if isinstance(preds, PredictionResult):
            return preds
        return PredictionResult(predictions=preds, **kwargs)


class LinearITSA(BaseModel):
    """
    Linear Interrupted Time Series Analysis.
    Standard ITSA with treatment indicator.
    y(t) = alpha + beta*t + gamma*T(t) + delta*(T(t)*t) + epsilon
    where T(t) = 0 before treatment, 1 after.
    """

    def __init__(self):
        super().__init__("LinearITSA")

    def fit(self, X: np.ndarray, y: np.ndarray,
            groups: Optional[np.ndarray] = None) -> 'LinearITSA':
        """
        Fit ITSA model.

        For each variable and each cat, fit:
        y[t] = intercept + slope * t + treatment_effect
        """
        self.models_ = {}
        n_samples, lookback, n_vars = X.shape

        # Flatten data for linear regression
        # Use last timepoint of X to predict first timepoint of y
        X_last = X[:, -1, :]  # (n_samples, n_vars)
        y_first = y[:, 0, :]   # (n_samples, n_vars)

        # Add time trend
        time_idx = np.arange(n_samples).reshape(-1, 1)

        # Simple linear model: y ~ X_last + time
        X_design = np.hstack([X_last, time_idx])

        self.models_ = {}
        for v in range(n_vars):
            mask = ~np.isnan(X_design[:, v]) & ~np.isnan(y_first[:, v])
            if mask.sum() > 10:
                lr = LinearRegression()
                lr.fit(X_design[mask], y_first[mask, v])
                self.models_[v] = lr

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, groups: Optional[np.ndarray] = None) -> PredictionResult:
        n_samples, lookback, n_vars = X.shape
        X_last = X[:, -1, :]
        time_idx = np.arange(n_samples).reshape(-1, 1)
        X_design = np.hstack([X_last, time_idx])

        preds = np.zeros((n_samples, 1, n_vars))
        for v in range(n_vars):
            if v in self.models_:
                preds[:, 0, v] = self.models_[v].predict(X_design)

        return PredictionResult(predictions=preds)


class GaussianProcessModel(BaseModel):
    """
    Gaussian Process regression per cat and per variable.
    Gold standard for small-data time series.
    """

    def __init__(self, kernel_type: str = 'rbf'):
        super().__init__(f"GP_{kernel_type}")
        self.kernel_type = kernel_type

    def fit(self, X: np.ndarray, y: np.ndarray,
            groups: Optional[np.ndarray] = None) -> 'GaussianProcessModel':
        """
        Fit GP per variable.
        For each cat, fit a GP on their trajectory.
        """
        n_samples, lookback, n_vars = X.shape
        self.models_ = {}
        self.scaler_ = StandardScaler()

        # Use lookback as time index
        time = np.arange(lookback).reshape(-1, 1)

        for cat_idx in range(n_samples):
            # Use concatenated history as features
            x_cat = X[cat_idx]  # (lookback, n_vars)
            y_cat = y[cat_idx, 0, :]  # predict first step of horizon

            for v in range(n_vars):
                mask = ~np.isnan(x_cat[:, v])
                if mask.sum() < 3:
                    continue

                kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(0.1)
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=0.1)
                gp.fit(time[mask], x_cat[mask, v])
                self.models_[(cat_idx, v)] = gp

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, groups: Optional[np.ndarray] = None) -> PredictionResult:
        n_samples, lookback, n_vars = X.shape
        time = np.arange(lookback).reshape(-1, 1)
        time_future = np.array([lookback]).reshape(-1, 1)

        preds = np.zeros((n_samples, 1, n_vars))
        stds = np.zeros((n_samples, 1, n_vars))

        for cat_idx in range(n_samples):
            for v in range(n_vars):
                if (cat_idx, v) in self.models_:
                    pred, std = self.models_[(cat_idx, v)].predict(time_future, return_std=True)
                    preds[cat_idx, 0, v] = pred[0]
                    stds[cat_idx, 0, v] = std[0]

        return PredictionResult(predictions=preds, uncertainties=stds)


class ExponentialCurveModel(BaseModel):
    """
    Simple exponential curve model for CKD progression.
    Assumes creatinine follows: y(t) = a * exp(b * t) + c
    where b > 0 means increasing (worsening kidney function).
    """

    def __init__(self):
        super().__init__("ExpCurve")

    def fit(self, X: np.ndarray, y: np.ndarray,
            groups: Optional[np.ndarray] = None) -> 'ExponentialCurveModel':
        """Fit per cat, using log-linear regression."""
        n_samples, lookback, n_vars = X.shape
        self.params_ = {}

        time = np.arange(lookback)

        for cat_idx in range(n_samples):
            x_cat = X[cat_idx]
            for v in range(n_vars):
                vals = x_cat[:, v]
                mask = ~np.isnan(vals) & (vals > 0)
                if mask.sum() < 3:
                    continue

                t = time[mask]
                v_vals = vals[mask]
                try:
                    log_vals = np.log(v_vals + 1e-8)
                    slope, intercept = np.polyfit(t, log_vals, 1)
                    self.params_[(cat_idx, v)] = (slope, intercept)
                except:
                    pass

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, groups: Optional[np.ndarray] = None) -> PredictionResult:
        n_samples, lookback, n_vars = X.shape
        preds = np.zeros((n_samples, 1, n_vars))

        for cat_idx in range(n_samples):
            for v in range(n_vars):
                if (cat_idx, v) in self.params_:
                    slope, intercept = self.params_[(cat_idx, v)]
                    preds[cat_idx, 0, v] = np.exp(slope * lookback + intercept)

        return PredictionResult(predictions=preds)


class SimpleLSTMModel(BaseModel):
    """
    Simple LSTM/GRU baseline for time series.
    Uses PyTorch if available, otherwise sklearn MLP.
    """

    def __init__(self, hidden_size: int = 32, n_layers: int = 1, epochs: int = 50):
        super().__init__(f"LSTM_h{hidden_size}_l{n_layers}")
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.epochs = epochs
        self.model_ = None
        self.scaler_ = StandardScaler()
        self.device_ = 'cpu'

        # Try to import torch
        try:
            import torch
            self.torch_ = torch
            self.torch_nn_ = torch.nn
            self.torch_available_ = True
        except ImportError:
            self.torch_available_ = False
            print("PyTorch not available, using sklearn MLP fallback")

    def _build_model(self, input_size: int, output_size: int):
        """Build PyTorch LSTM model."""
        if not self.torch_available_:
            from sklearn.neural_network import MLPRegressor
            return None

        class SimpleLSTM(self.torch_nn_.Module):
            def __init__(self, input_size, hidden_size, output_size, n_layers):
                super().__init__()
                self.lstm = self.torch_nn_.LSTM(input_size, hidden_size, n_layers,
                                                  batch_first=True, dropout=0.1)
                self.fc = self.torch_nn_.Linear(hidden_size, output_size)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                return out

        return SimpleLSTM(input_size, self.hidden_size, output_size, self.n_layers)

    def fit(self, X: np.ndarray, y: np.ndarray,
            groups: Optional[np.ndarray] = None) -> 'SimpleLSTMModel':
        """
        Fit LSTM model.

        Args:
            X: (n_samples, lookback, n_vars)
            y: (n_samples, forecast_steps, n_vars)
            groups: group labels for each sample
        """
        if self.torch_available_:
            return self._fit_torch(X, y, groups)
        else:
            return self._fit_sklearn(X, y, groups)

    def _fit_torch(self, X: np.ndarray, y: np.ndarray,
                   groups: Optional[np.ndarray] = None) -> 'SimpleLSTMModel':
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        n_samples, lookback, n_vars = X.shape
        forecast_steps = y.shape[1]

        # Handle NaN
        X_flat = X.reshape(n_samples, -1)
        mask = ~(np.isnan(X_flat).any(axis=1) | np.isnan(y[:, 0, :]).any(axis=1))
        X_clean = X[mask]
        y_clean = y[mask, 0, :]  # predict first step only

        if len(X_clean) < 5:
            self.is_fitted = False
            return self

        # Normalize
        X_2d = X_clean.reshape(len(X_clean), -1)
        self.scaler_.fit(X_2d)
        X_scaled = self.scaler_.transform(X_2d).reshape(len(X_clean), lookback, n_vars)

        # To tensor
        X_t = torch.FloatTensor(X_scaled)
        y_t = torch.FloatTensor(y_clean)

        # Build model
        self.model_ = self._build_model(n_vars, n_vars)
        if self.model_ is None:
            self.is_fitted = False
            return self

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.01, weight_decay=1e-3)
        loss_fn = nn.MSELoss()

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True)

        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = self.model_(batch_x)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if epoch % 20 == 0:
                print(f"  LSTM Epoch {epoch}: loss={epoch_loss/len(loader):.4f}")

        self.is_fitted = True
        return self

    def _fit_sklearn(self, X: np.ndarray, y: np.ndarray,
                     groups: Optional[np.ndarray] = None) -> 'SimpleLSTMModel':
        from sklearn.neural_network import MLPRegressor

        n_samples, lookback, n_vars = X.shape

        X_flat = X.reshape(n_samples, -1)
        mask = ~(np.isnan(X_flat).any(axis=1) | np.isnan(y[:, 0, :]).any(axis=1))
        X_clean = X[mask]
        y_clean = y[mask, 0, :]

        if len(X_clean) < 5:
            self.is_fitted = False
            return self

        X_2d = X_clean.reshape(len(X_clean), -1)
        self.scaler_.fit(X_2d)
        X_scaled = self.scaler_.transform(X_2d)

        self.model_ = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=42
        )
        self.model_.fit(X_scaled, y_clean)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, groups: Optional[np.ndarray] = None) -> PredictionResult:
        if not self.is_fitted or self.model_ is None:
            return PredictionResult(predictions=np.zeros((X.shape[0], 1, X.shape[2])))

        n_samples, lookback, n_vars = X.shape
        X_flat = X.reshape(n_samples, -1)
        X_scaled = self.scaler_.transform(X_flat)
        preds = self.model_.predict(X_scaled)

        return PredictionResult(predictions=preds.reshape(n_samples, 1, n_vars))


class TreatmentEffectEstimator:
    """
    Estimates treatment effects by comparing predicted trajectories
    under treatment vs no-treatment conditions.

    For a treatment cat:
    - Predicted trajectory WITH treatment = model(t | treatment=1)
    - Counterfactual trajectory WITHOUT treatment = model(t | treatment=0)
    - ITE = WITH - WITHOUT
    """

    def __init__(self, base_model: BaseModel):
        self.model = base_model

    def estimate_ate(self,
                     df: pd.DataFrame,
                     treatment_col: str = 'group',
                     outcome_var: str = 'creatinine',
                     cat_col: str = 'cat_id',
                     time_col: str = 'time') -> Dict:
        """
        Estimate ATE from longitudinal data.

        For treatment cats: estimate counterfactual (no treatment) trajectory
        and compute difference.

        For control cats: the observed trajectory IS the no-treatment trajectory.
        """
        treatment_cats = df[df[treatment_col] == 'treatment'][cat_col].unique()
        control_cats = df[df[treatment_col] == 'control'][cat_col].unique()

        # Simple approach: compare average trajectory levels
        treat_obs = df[df[treatment_col] == 'treatment'].groupby(time_col)[outcome_var].mean()
        control_obs = df[df[treatment_col] == 'control'].groupby(time_col)[outcome_var].mean()

        # ATE = difference in final levels
        ate = treat_obs.iloc[-1] - control_obs.iloc[-1]

        # Per-timepoint ATE
        ate_by_time = (treat_obs - control_obs).to_dict()

        # ITE approximation (treatment effect heterogeneity)
        ites = {}
        for cat in treatment_cats:
            cat_traj = df[df[cat_col] == cat].sort_values(time_col)[outcome_var]
            control_traj = control_obs
            # Simple ITE: difference from control mean
            ite = cat_traj.iloc[-1] - control_obs.iloc[-1]
            ites[cat] = ite

        return {
            'ate': ate,
            'ate_by_time': ate_by_time,
            'ite': ites,
            'n_treatment': len(treatment_cats),
            'n_control': len(control_cats),
            'method': 'simple_mean_diff'
        }


def evaluate_predictions(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          y_std: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute standard regression metrics.

    Args:
        y_true: (n_samples, forecast_steps, n_vars)
        y_pred: (n_samples, forecast_steps, n_vars)
        y_std: uncertainties (if available)
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]

    mse = np.mean((y_t - y_p) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_t - y_p))

    # R-squared
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

    # Calibration if uncertainties provided
    if y_std is not None:
        y_s = y_std[mask]
        z_scores = np.abs(y_t - y_p) / (y_s + 1e-8)
        # 95% CI coverage
        coverage = np.mean(z_scores < 1.96)
        metrics['calibration_95_coverage'] = coverage

    return metrics


if __name__ == '__main__':
    from simulation import SyntheticCKDGenerator

    # Test baselines on synthetic data
    gen = SyntheticCKDGenerator()
    datasets = gen.generate(seed=42, n_datasets=1)
    df = datasets[0]['df']
    print(f"Synthetic data: {len(df)} records")
    print(f"True ATE: {datasets[0]['true_ate']:.4f}")

    # Reshape for modeling
    # Wide format
    pivot = df.pivot_table(index=['cat_id', 'time'], columns='variable', values='creatinine').reset_index()

    print(f"\nBaseline results on synthetic data:")
    ate_est = TreatmentEffectEstimator(None).estimate_ate(df, outcome_var='creatinine')
    print(f"  Simple ATE estimate: {ate_est['ate']:.4f} (true: {datasets[0]['true_ate']:.4f})")
    print(f"  Error: {abs(ate_est['ate'] - datasets[0]['true_ate']):.4f}")
