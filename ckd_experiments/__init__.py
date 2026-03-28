"""
CKD Experiments Package

Treatment Effect Estimation in Small-Animal Longitudinal Studies.
"""

__version__ = "1.0.0"
__author__ = ""

from .baseline_models import (
    BaseModel,
    PredictionResult,
    LinearITSA,
    GaussianProcessModel,
    ExponentialCurveModel,
    SimpleLSTMModel,
    TreatmentEffectEstimator,
    evaluate_predictions,
)

from .ssm_model import (
    SSMTreatmentEffectModel,
    S4ModelWrapper,
    SSMConfig,
)

from .simulation import (
    SyntheticCKDGenerator,
    CKDParameters,
    run_simulation_study,
)

__all__ = [
    # Version
    "__version__",
    # Base
    "BaseModel",
    "PredictionResult",
    "evaluate_predictions",
    # Models
    "LinearITSA",
    "GaussianProcessModel",
    "ExponentialCurveModel",
    "SimpleLSTMModel",
    "SSMTreatmentEffectModel",
    "S4ModelWrapper",
    "SSMConfig",
    "TreatmentEffectEstimator",
    # Simulation
    "SyntheticCKDGenerator",
    "CKDParameters",
    "run_simulation_study",
]
