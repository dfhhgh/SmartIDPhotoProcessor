"""Experiment A: head-only fine-tuning with frozen backbone."""

from .config import ExperimentAConfig
from .trainer import train_experiment_a

__all__ = ["ExperimentAConfig", "train_experiment_a"]
