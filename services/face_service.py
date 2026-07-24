import logging
from config.settings import Settings

from insightface.app import FaceAnalysis


logger=logging.getLogger(__name__)
class FaceService:
    """
    Service responsible for managing the InsightFace model.

    Responsibilities:
    - Read application settings.
    - Load the model (later).
    - Provide a single model instance.
    """
    _instance = None
    _initialized = False
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self) -> None:

        if self._initialized:
            return

        self._model: FaceAnalysis| None = None

        self.model_name = Settings.MODEL_NAME
        self.model_root = Settings.MODEL_ROOT

        self.use_gpu = Settings.USE_GPU
        self.gpu_id = Settings.GPU_ID

        self.detection_size = Settings.DETECTION_SIZE
        self.detection_threshold = Settings.DETECTION_THRESHOLD

        self._initialized = True

    def _load_model(self) -> None:
        """
        Load and prepare the InsightFace model.
        """

        logger.info("Loading InsightFace model...")

        ctx_id = self.gpu_id if self.use_gpu else -1

        logger.info(
            "Using %s",
            f"GPU ({ctx_id})" if ctx_id >= 0 else "CPU"
        )

        try:
            self._model = FaceAnalysis(
                name=self.model_name,
                root=str(self.model_root)
            )

            self._model.prepare(
                ctx_id=ctx_id,
                det_size=self.detection_size,
                det_thresh=self.detection_threshold
            )

            logger.info("InsightFace model loaded successfully.")

        except Exception as e:
            logger.exception("Failed to load InsightFace model.")
            raise RuntimeError("Could not initialize InsightFace.") from e



    def get_model(self) -> FaceAnalysis:
        """
        Return the loaded InsightFace model.
        Load it only once if it has not been loaded yet.
        """

        if self._model is None:
            self._load_model()

        return self._model