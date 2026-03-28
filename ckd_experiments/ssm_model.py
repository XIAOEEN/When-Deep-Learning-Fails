"""
SSM (Mamba) Treatment Effect Model.
Uses Mamba (or S4) for longitudinal trajectory modeling with treatment conditioning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from baseline_models import BaseModel, PredictionResult, evaluate_predictions
from sklearn.preprocessing import StandardScaler


@dataclass
class SSMConfig:
    """Configuration for SSM treatment effect model."""
    hidden_size: int = 64
    n_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 16
    treatment_embedding_dim: int = 8
    use_bayesian: bool = True  # Whether to use MC dropout for uncertainty


class SSMTreatmentEffectModel(BaseModel):
    """
    SSM-based treatment effect estimator.

    Architecture:
    1. Mamba/S4 backbone for sequence modeling
    2. Treatment embedding layer
    3. Counterfactual prediction head

    For treatment cats, we predict:
    - Observed trajectory (with treatment)
    - Counterfactual trajectory (without treatment)
    ITE = observed - counterfactual
    """

    def __init__(self,
                 config: Optional[SSMConfig] = None,
                 model_type: str = 'mamba'):
        super().__init__(f"SSM_TE_{model_type}")
        self.config = config or SSMConfig()
        self.model_type = model_type
        self.model_ = None
        self.scaler_ = None
        self.torch_available_ = False

        # Try to import torch
        try:
            import torch
            import torch.nn as nn
            self.torch_ = torch
            self.nn_ = nn
            self.torch_available_ = True
        except ImportError:
            print("PyTorch not available, using numpy fallback")
            self.torch_available_ = False

    def _build_mamba_model(self, n_vars: int, n_treatments: int = 2):
        """Build Mamba-based model using mamba-ssm or fallback."""
        if not self.torch_available_:
            return None

        try:
            from mamba_ssm import Mamba
            has_mamba = True
        except ImportError:
            has_mamba = False

        # Capture outer scope variables
        torch_nn = self.nn_
        hidden_size = self.config.hidden_size
        n_layers = self.config.n_layers
        dropout = self.config.dropout

        class SSMTreatmentModel(torch_nn.Module):
            """
            SSM-based treatment effect model.

            Uses Mamba (or SSM) to encode trajectory,
            with treatment as a conditioning variable.
            """

            def __init__(self, n_vars, hidden_size, n_layers, n_treatments, dropout):
                super().__init__()
                self.n_vars = n_vars

                # Input projection
                self.input_proj = torch_nn.Linear(n_vars, hidden_size)

                # Treatment embedding
                self.treatment_emb = torch_nn.Embedding(n_treatments, hidden_size)

                if has_mamba:
                    # Mamba backbone
                    self.ssm_layers = torch_nn.ModuleList([
                        Mamba(hidden_size, state_dim=hidden_size)
                        for _ in range(n_layers)
                    ])
                else:
                    # Fallback: use LSTM as SSM-like alternative
                    self.lstm = torch_nn.LSTM(hidden_size, hidden_size, n_layers,
                                              batch_first=True, dropout=dropout)

                # Layer norm
                self.ln = torch_nn.LayerNorm(hidden_size)

                # Output: predicts next step for each treatment condition
                self.treatment_head = torch_nn.Linear(hidden_size, n_treatments * n_vars)

                self.dropout = torch_nn.Dropout(dropout)

            def forward(self, x, treatment):
                """
                Args:
                    x: (batch, seq_len, n_vars) - input trajectory
                    treatment: (batch,) - treatment indicator (0=control, 1=treatment)
                Returns:
                    predictions for each treatment condition
                """
                batch_size, seq_len, _ = x.shape

                # Project input
                h = self.input_proj(x)  # (batch, seq_len, hidden)

                # Add treatment embedding
                t_emb = self.treatment_emb(treatment)  # (batch, hidden)
                h = h + t_emb.unsqueeze(1)  # broadcast

                # SSM forward
                if has_mamba:
                    for layer in self.ssm_layers:
                        h = layer(h) + h  # residual
                else:
                    h, _ = self.lstm(h)

                h = self.dropout(self.ln(h))

                # Take last timestep
                h = h[:, -1, :]  # (batch, hidden)

                # Predict for both treatment conditions
                out = self.treatment_head(h)  # (batch, n_treatments * n_vars)
                out = out.view(batch_size, 2, n_vars)  # (batch, 2, n_vars)

                return out

            def predict_counterfactual(self, x, treatment):
                """
                Predict counterfactual: what if treatment condition was different?
                """
                out = self.forward(x, treatment)
                batch_size = x.shape[0]

                # For treatment cats: predict counterfactual (control=0)
                # For control cats: predict counterfactual (treatment=1)
                cf_treatment = 1 - treatment  # flip treatment

                out_cf = self.forward(x, cf_treatment)

                # Return observed and counterfactual
                treatment_idx = treatment.long().unsqueeze(1).unsqueeze(2).expand(-1, 1, self.n_vars)
                cf_idx = cf_treatment.long().unsqueeze(1).unsqueeze(2).expand(-1, 1, self.n_vars)

                observed = out.gather(1, treatment_idx).squeeze(1)
                counterfactual = out_cf.gather(1, cf_idx).squeeze(1)

                return observed, counterfactual

        return SSMTreatmentModel(
            n_vars=n_vars,
            hidden_size=self.config.hidden_size,
            n_layers=self.config.n_layers,
            n_treatments=2,
            dropout=self.config.dropout
        )

    def fit(self, X: np.ndarray, y: np.ndarray,
            groups: Optional[np.ndarray] = None) -> 'SSMTreatmentEffectModel':
        """
        Fit the SSM treatment effect model.

        Args:
            X: (n_samples, lookback, n_vars) - input sequences
            y: (n_samples, forecast_steps, n_vars) - target sequences
            groups: (n_samples,) - group labels (0=control, 1=treatment)
        """
        if not self.torch_available_:
            print("PyTorch not available, cannot fit SSM model")
            self.is_fitted = False
            return self

        import torch
        from torch.utils.data import DataLoader, TensorDataset

        n_samples, lookback, n_vars = X.shape

        # Handle NaN - remove samples with NaN
        X_flat = X.reshape(n_samples, -1)
        y_flat = y[:, 0, :]  # predict first step
        mask = ~(np.isnan(X_flat).any(axis=1) | np.isnan(y_flat).any(axis=1))

        if groups is None:
            groups = np.zeros(n_samples)
        groups = np.array(groups)[mask].astype(int)

        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) < 5:
            print(f"Too few samples after filtering: {len(X_clean)}")
            self.is_fitted = False
            return self

        # Normalize
        X_2d = X_clean.reshape(len(X_clean), -1)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X_2d)
        X_scaled = self.scaler_.transform(X_2d).reshape(len(X_clean), lookback, n_vars)
        y_mean = np.nanmean(y_clean.reshape(len(y_clean), -1), axis=1)
        y_std = np.nanstd(y_clean.reshape(len(y_clean), -1), axis=1) + 1e-8

        # Build model
        self.model_ = self._build_mamba_model(n_vars)
        if self.model_ is None:
            self.is_fitted = False
            return self

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-3
        )
        loss_fn = torch.nn.MSELoss()

        # To tensors
        X_t = torch.FloatTensor(X_scaled)
        y_t = torch.FloatTensor(y_clean[:, 0, :])  # first step
        groups_t = torch.LongTensor(groups)

        dataset = TensorDataset(X_t, y_t, groups_t)
        loader = DataLoader(dataset, batch_size=min(self.config.batch_size, len(dataset)), shuffle=True)

        self.model_.train()
        for epoch in range(self.config.epochs):
            epoch_loss = 0
            for batch_x, batch_y, batch_g in loader:
                optimizer.zero_grad()

                # Forward pass
                out = self.model_(batch_x, batch_g)  # (batch, 2, n_vars)

                # Predict for the actual treatment condition
                g_idx = batch_g.long().unsqueeze(1).unsqueeze(2).expand(-1, 1, n_vars)
                pred = out.gather(1, g_idx).squeeze(1)

                loss = loss_fn(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if epoch % 20 == 0:
                print(f"  SSM Epoch {epoch}: loss={epoch_loss/len(loader):.4f}")

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray,
                groups: Optional[np.ndarray] = None) -> PredictionResult:
        """
        Predict with counterfactual estimation.
        """
        if not self.is_fitted or self.model_ is None:
            return PredictionResult(predictions=np.zeros((X.shape[0], 1, X.shape[2])))

        import torch

        n_samples, lookback, n_vars = X.shape

        self.model_.eval()
        with torch.no_grad():
            X_flat = X.reshape(n_samples, -1)
            X_scaled = self.scaler_.transform(X_flat).reshape(n_samples, lookback, n_vars)
            X_t = torch.FloatTensor(X_scaled)

            if groups is None:
                groups = np.zeros(n_samples)
            groups_t = torch.LongTensor(np.array(groups).astype(int))

            # Get predictions for both treatment conditions
            out = self.model_(X_t, groups_t)  # (batch, 2, n_vars)

            # For each sample, return the observed prediction
            g_idx = groups_t.long().unsqueeze(1).unsqueeze(2).expand(-1, 1, n_vars)
            preds = out.gather(1, g_idx).squeeze(1).numpy()

            # Also get counterfactual predictions for treatment effect estimation
            cf_groups = 1 - groups_t
            cf_idx = cf_groups.long().unsqueeze(1).unsqueeze(2).expand(-1, 1, n_vars)
            cf_preds = out.gather(1, cf_idx).squeeze(1).numpy()

        return PredictionResult(
            predictions=preds.reshape(n_samples, 1, n_vars),
            uncertainties=np.abs(preds - cf_preds).reshape(n_samples, 1, n_vars)  # ITE magnitude as uncertainty
        )

    def estimate_treatment_effects(self,
                                   X: np.ndarray,
                                   groups: np.ndarray) -> Dict[str, float]:
        """
        Estimate individualized and average treatment effects.

        Returns:
            ate: Average treatment effect
            ites: Dict of ITE per cat
        """
        if not self.is_fitted:
            return {'ate': 0.0, 'ite': {}, 'method': 'ssm'}

        import torch

        n_samples, lookback, n_vars = X.shape

        self.model_.eval()
        with torch.no_grad():
            X_flat = X.reshape(n_samples, -1)
            X_scaled = self.scaler_.transform(X_flat).reshape(n_samples, lookback, n_vars)
            X_t = torch.FloatTensor(X_scaled)

            groups_t = torch.LongTensor(groups.astype(int))

            # Get predictions for both treatment conditions
            out = self.model_(X_t, groups_t)  # (batch, 2, n_vars)

            # Observed prediction (for treatment cats)
            obs_idx = groups_t.long().unsqueeze(1).unsqueeze(2).expand(-1, 1, n_vars)
            obs_preds = out.gather(1, obs_idx).squeeze(1).numpy()

            # Counterfactual (flip treatment)
            cf_groups = 1 - groups_t
            cf_idx = cf_groups.long().unsqueeze(1).unsqueeze(2).expand(-1, 1, n_vars)
            cf_preds = out.gather(1, cf_idx).squeeze(1).numpy()

        # ITE = observed - counterfactual
        ites = {}
        ate_components = []

        for i in range(n_samples):
            if groups[i] == 1:  # treatment cat
                ite = obs_preds[i].mean() - cf_preds[i].mean()
                ites[f'treat_{i}'] = ite
                ate_components.append(ite)
            else:  # control cat
                # For control cats, we can also estimate what-if-treated
                ite = cf_preds[i].mean() - obs_preds[i].mean()
                ites[f'control_{i}'] = ite

        ate = np.mean(ate_components) if ate_components else 0.0

        return {
            'ate': ate,
            'ite': ites,
            'method': 'ssm_mamba',
            'observed_preds': obs_preds,
            'counterfactual_preds': cf_preds
        }


class S4ModelWrapper(BaseModel):
    """
    Wrapper for S4 (Structured State Space Sequence Model) from S4D.

    Reference: "Counterfactual Outcome Prediction using Structured State Space Model" (2024)
    """

    def __init__(self, d_model: int = 64, n_layers: int = 2):
        super().__init__(f"S4_d{d_model}_l{n_layers}")
        self.d_model = d_model
        self.n_layers = n_layers
        self.model_ = None

        try:
            import torch
            self.torch_ = torch
            self.torch_available_ = True
        except ImportError:
            self.torch_available_ = False

    def _build_model(self, n_vars: int, n_treatments: int = 2):
        """Build S4-based model."""
        if not self.torch_available_:
            return None

        try:
            from mamba_ssm import Mamba
            has_mamba = True
        except ImportError:
            has_mamba = False

        # Capture outer scope variables
        torch_nn = self.torch_.nn
        d_model = self.d_model
        n_layers = self.n_layers

        class S4TreatmentModel(torch_nn.Module):
            def __init__(self, n_vars, d_model, n_layers, n_treatments):
                super().__init__()
                self.n_vars = n_vars

                self.input_proj = torch_nn.Linear(n_vars, d_model)
                self.treatment_emb = torch_nn.Embedding(n_treatments, d_model)

                if has_mamba:
                    self.ssm = Mamba(d_model, state_dim=d_model)
                else:
                    self.ssm = torch_nn.RNN(d_model, d_model, n_layers,
                                                    batch_first=True)

                self.head = torch_nn.Linear(d_model, n_treatments * n_vars)

            def forward(self, x, treatment):
                h = self.input_proj(x)
                t_emb = self.treatment_emb(treatment)
                h = h + t_emb.unsqueeze(1)

                if has_mamba:
                    h = self.ssm(h) + h
                else:
                    h, _ = self.ssm(h)

                h = h[:, -1, :]
                out = self.head(h)
                return out.view(-1, 2, n_vars)

        return S4TreatmentModel(n_vars, d_model, n_layers, 2)

    def fit(self, X: np.ndarray, y: np.ndarray,
            groups: Optional[np.ndarray] = None) -> 'S4ModelWrapper':
        """Same interface as SSMTreatmentEffectModel."""
        return self._fit_common(X, y, groups)

    def _fit_common(self, X: np.ndarray, y: np.ndarray,
                    groups: Optional[np.ndarray] = None) -> 'S4ModelWrapper':
        """Common fitting logic shared between S4 and SSM."""
        if not self.torch_available_:
            self.is_fitted = False
            return self

        import torch
        from torch.utils.data import DataLoader, TensorDataset

        n_samples, lookback, n_vars = X.shape

        X_flat = X.reshape(n_samples, -1)
        mask = ~(np.isnan(X_flat).any(axis=1) | np.isnan(y[:, 0, :]).any(axis=1))

        if groups is None:
            groups = np.zeros(n_samples)
        groups = np.array(groups)[mask].astype(int)

        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) < 5:
            self.is_fitted = False
            return self

        from sklearn.preprocessing import StandardScaler
        self.scaler_ = StandardScaler()
        X_2d = X_clean.reshape(len(X_clean), -1)
        self.scaler_.fit(X_2d)
        X_scaled = self.scaler_.transform(X_2d).reshape(len(X_clean), lookback, n_vars)

        self.model_ = self._build_model(n_vars)
        if self.model_ is None:
            self.is_fitted = False
            return self

        X_t = torch.FloatTensor(X_scaled)
        y_t = torch.FloatTensor(y_clean[:, 0, :])
        groups_t = torch.LongTensor(groups)

        dataset = TensorDataset(X_t, y_t, groups_t)
        loader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=1e-3, weight_decay=1e-3)
        loss_fn = torch.nn.MSELoss()

        self.model_.train()
        for epoch in range(100):
            for batch_x, batch_y, batch_g in loader:
                optimizer.zero_grad()
                out = self.model_(batch_x, batch_g)
                g_idx = batch_g.long().unsqueeze(1).unsqueeze(2).expand(-1, 1, n_vars)
                pred = out.gather(1, g_idx).squeeze(1)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray,
                groups: Optional[np.ndarray] = None) -> PredictionResult:
        """Predict."""
        if not self.is_fitted:
            return PredictionResult(predictions=np.zeros((X.shape[0], 1, X.shape[2])))

        import torch
        n_samples, lookback, n_vars = X.shape

        self.model_.eval()
        with torch.no_grad():
            X_flat = X.reshape(n_samples, -1)
            X_scaled = self.scaler_.transform(X_flat).reshape(n_samples, lookback, n_vars)
            X_t = torch.FloatTensor(X_scaled)
            groups_t = torch.LongTensor(np.zeros(n_samples).astype(int) if groups is None else groups.astype(int))
            out = self.model_(X_t, groups_t)
            preds = out[:, 0, :].numpy()

        return PredictionResult(predictions=preds.reshape(n_samples, 1, n_vars))


if __name__ == '__main__':
    from simulation import SyntheticCKDGenerator
    from sklearn.preprocessing import StandardScaler

    print("Testing SSM model on synthetic data...")

    gen = SyntheticCKDGenerator()
    datasets = gen.generate(seed=42, n_datasets=1)
    df = datasets[0]['df']

    # Reshape to (n_cats * (T-lookback), lookback, n_vars)
    pivot = df.pivot_table(index=['cat_id', 'time'], columns='variable', values='creatinine')
    pivot = pivot.reset_index()
    pivot['group'] = pivot['cat_id'].apply(lambda x: 1 if 't' in x else 0)

    cats = pivot['cat_id'].unique()
    lookback = 5

    X_list, y_list, groups_list = [], [], []
    for cat in cats:
        cat_df = pivot[pivot['cat_id'] == cat].sort_values('time')
        vals = cat_df[['creatinine', 'bun', 'weight']].values
        group = cat_df['group'].values[0]

        for t in range(lookback, len(vals)):
            X_list.append(vals[t-lookback:t])
            y_list.append(vals[t:t+1])
            groups_list.append(group)

    X = np.array(X_list)
    y = np.array(y_list)
    groups = np.array(groups_list)

    print(f"Data shape: X={X.shape}, y={y.shape}, groups={groups.shape}")

    # Fit SSM model
    model = SSMTreatmentEffectModel(config=SSMConfig(hidden_size=32, n_layers=1, epochs=50))
    model.fit(X, y, groups)

    if model.is_fitted:
        print("SSM model fitted successfully!")
        result = model.estimate_treatment_effects(X, groups)
        print(f"  Estimated ATE: {result['ate']:.4f}")
        print(f"  True ATE: {0.015:.4f}")
    else:
        print("SSM model fitting failed")
