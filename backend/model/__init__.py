"""Model architecture modules - Prediction only"""
from .fedfusionnet_simple import FedFusionNetPlus
from .tabular_model import OralCancerTabularModel

__all__ = [
    'FedFusionNetPlus',
    'OralCancerTabularModel'
]
