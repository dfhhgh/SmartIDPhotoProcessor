import logging

import numpy as np
from insightface.app.common import Face

from services.face_service import FaceService
from exceptions.face_exceptions import FaceDetectionError
logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Detect faces in an image using InsightFace.
    """

    def __init__(self) -> None:
        self._model = FaceService().get_model()

    def detect(
        self,
        image: np.ndarray
    ) -> list[Face]:
        """
        Detect all faces in an image.

        Args:
            image:
                Input image as an OpenCV BGR NumPy array.

        Returns:
            A list of detected InsightFace Face objects.
        """
        if image is None:
            raise ValueError("Input image cannot be None.")

        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a NumPy ndarray.")

        if image.size == 0:
            raise ValueError("Input image cannot be empty.")

        if image.ndim not in (2, 3):
            raise ValueError(
                "Input image must have 2 or 3 dimensions."
            )
        logger.info("Starting face detection.")
        try:
            faces: list[Face] = self._model.get(image)

            logger.info("Detected %d face(s).", len(faces))

            return faces

        except Exception as e:
            logger.exception("Face detection failed.")
            raise FaceDetectionError(
                "Failed to detect faces."
            ) from e