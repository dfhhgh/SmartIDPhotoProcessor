"""
Eyewear classifier service contract.
"""

from abc import ABC, abstractmethod

import numpy as np
from insightface.app.common import Face

from models.eyewear_prediction import EyewearPrediction


class EyewearClassifier(ABC):
    """
    Abstract contract for eyewear classification services.

    Defines the contract for eyewear classification services.

    Concrete implementations may use different machine learning models while exposing
    the same interface to the validation layer.
    This abstract service decouples the validation layer from any specific
    machine learning technology, framework, or model implementation (e.g.
    ONNX Runtime, PyTorch, TensorFlow, YOLO, Vision Transformers).
    """

    @abstractmethod
    def classify(
        self,
        image: np.ndarray,
        face: Face,
    ) -> EyewearPrediction:
        """
        Classify eyewear on the provided face region.

        Receives the full original image and the detected Face object containing
        facial landmarks, bounding box coordinates, and metadata to enable flexible
        preprocessing and feature extraction.

        Args:
            image: Full original image as a NumPy array.
            face: Detected face object containing bounding box and landmark metadata.

        Returns:
            An EyewearPrediction containing the predicted eyewear category and
            associated confidence score.
        """
        raise NotImplementedError
