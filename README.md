# When Deep Learning Fails: Treatment Effect Estimation in Small-Animal Longitudinal Studies

This repository contains the code for the paper:

> **When Deep Learning Fails: Treatment Effect Estimation in Small-Animal Longitudinal Studies**
>
> We demonstrate that simple statistical models (Gaussian Process, Exponential Curve) outperform complex SSM architectures for treatment effect estimation in small-N veterinary studies, using feline chronic kidney disease (CKD) as a case study.

## Key Finding

Despite the popularity of state-space models (SSM/Mamba) for longitudinal causal inference, we show that on N<10 subjects:
- **Gaussian Process**: R² ≈ 0.24 (best)
- **Exponential Curve**: R² ≈ 0.23
- **SSM/Mamba**: R² ≈ -0.63 (severely overfit)

This finding challenges the assumption that deep learning methods are universally superior.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── setup.py
├── ckd_experiments/
│   ├── __init__.py
│   ├── main.py              # Main entry point
│   ├── run_experiments.py   # Experiment runner
│   ├── data_loader.py       # Data loading from Excel files
│   ├── preprocessing.py      # Data preprocessing
│   ├── simulation.py         # Synthetic data generation
│   ├── baseline_models.py   # GP, ITSA, ExpCurve models
│   └── ssm_model.py         # SSM/Mamba treatment effect model
└── data/                    # Place your data here
```

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+ (for SSM models, optional)
- NumPy, Pandas, SciPy
- scikit-learn
- openpyxl (for Excel file reading)
- GPyTorch (optional, for advanced GP models)

## Usage

### 1. Prepare Your Data

Place your longitudinal data in the `data/` directory. The expected format is:
- Excel files with sheet names representing dates
- Columns for different subjects/cats
- Rows for different variables

### 2. Run All Experiments

```bash
cd ckd_experiments
python main.py
```

This will:
1. Load and preprocess your data
2. Run Leave-One-Subject-Out Cross-Validation (LOSO-CV)
3. Estimate treatment effects with bootstrap confidence intervals
4. Run sensitivity analyses
5. Generate simulation studies

### 3. Individual Components

```python
from ckd_experiments import load_all_data, create_unified_dataframe
from ckd_experiments.preprocessing import select_core_variables, align_to_common_timepoints
from ckd_experiments.baseline_models import GaussianProcessModel, LinearITSA, ExponentialCurveModel
from ckd_experiments.ssm_model import SSMTreatmentEffectModel, S4ModelWrapper
from ckd_experiments.simulation import SyntheticCKDGenerator

# Load data
data = load_all_data('path/to/data')
unified = create_unified_dataframe(data)

# Select variables and align
df_sel = select_core_variables(unified)
df_aligned = align_to_common_timepoints(df_sel)

# Train models
gp = GaussianProcessModel()
gp.fit(X_train, y_train, groups_train)
predictions = gp.predict(X_test)

# Estimate treatment effects with bootstrap
from ckd_experiments.run_experiments import run_bootstrap_ate
bootstrap_results = run_bootstrap_ate(X, y, groups, n_bootstrap=1000)
```

## Methods Implemented

1. **Linear ITSA** - Interrupted Time Series Analysis
2. **Gaussian Process** - RBF kernel regression
3. **Exponential Curve** - Log-linear trajectory modeling
4. **Simple LSTM** - PyTorch LSTM baseline
5. **SSM (Mamba)** - State-space model with treatment conditioning
6. **S4 Model** - Structured State Space Sequence Model

## Bootstrap Confidence Intervals

We use subject-level bootstrap resampling (1000 iterations) for ATE uncertainty quantification:

```python
def run_bootstrap_ate(X, y, groups, n_bootstrap=1000):
    """
    Bootstrap confidence intervals for Average Treatment Effect.
    Resamples entire subjects within groups.
    """
    # Implementation in run_experiments.py
```

## Results

On our feline CKD dataset (N=9 cats, 11 timepoints):

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Exponential Curve | 55.90 | 96.12 | 0.234 |
| Gaussian Process | 56.25 | 96.24 | 0.240 |
| Linear ITSA | 56.91 | 91.28 | -0.628 |
| S4 Treatment | 70.72 | 124.35 | -0.549 |
| SSM Mamba | 75.09 | 127.52 | -0.631 |

**Bootstrap ATE**: -61.13 mg/dL (95% CI [-80.79, -42.29])

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ckd2026treatment,
  title={When Deep Learning Fails: Treatment Effect Estimation in Small-Animal Longitudinal Studies},
  author={},
  journal={},
  year={2026}
}
```

## License

MIT License

## Contact

For questions about the code, please open an issue on GitHub.
