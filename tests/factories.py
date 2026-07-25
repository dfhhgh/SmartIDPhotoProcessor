import numpy as np
from insightface.app.common import Face


from typing import Sequence


def create_face(
    bbox: Sequence[float] | None = (
        100,
        100,
        400,
        400,
    ),
    det_score: float | None = 0.95,
) -> Face:
    """
    Create a valid InsightFace Face object for testing.
    """

    return Face(
    {
        "bbox": (
            None
            if bbox is None
            else np.asarray(
                bbox,
                dtype=np.float32,
            )
        ),
        "det_score": det_score,
    }
)


