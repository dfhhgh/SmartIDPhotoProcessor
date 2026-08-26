#!/usr/bin/env python3
"""
Model loading verification script.

Instantiates and tests FaceService and FaceParserService,
forces lazy-loaded models to initialize, measures and prints loading time, detects
whether GPU or CPU is used, and reports overall status.
"""

from __future__ import annotations

import sys
import time
import logging

from services.face_service import FaceService
from services.face_parser_service import FaceParserService

logger = logging.getLogger(__name__)


def check_insightface() -> tuple[bool, str, float]:
    """Test loading FaceService (InsightFace)."""
    start_time = time.perf_counter()
    try:
        service = FaceService()
        service.get_model()
        elapsed = time.perf_counter() - start_time
        
        ctx_id = service.gpu_id if service.use_gpu else -1
        device_str = "GPU" if ctx_id >= 0 else "CPU"
        return True, device_str, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        return False, str(e), elapsed


def check_face_parser() -> tuple[bool, str, float]:
    """Test loading FaceParserService (BiSeNet)."""
    start_time = time.perf_counter()
    try:
        service = FaceParserService()
        session = service._ensure_loaded()
        elapsed = time.perf_counter() - start_time

        providers = session.get_providers()
        is_gpu = any("CUDA" in p for p in providers)
        device_str = "GPU" if is_gpu else "CPU"
        return True, device_str, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        return False, str(e), elapsed


def main() -> None:
    """Main entry point for checking AI models."""
    print("========================================")
    print("MODEL LOADING REPORT")
    print("========================================")
    print()

    models_to_check = [
        ("InsightFace", check_insightface),
        ("Face Parser", check_face_parser),
    ]

    success_all = True

    for name, checker in models_to_check:
        success, info_or_error, elapsed = checker()
        
        if success:
            print(f"{name:<22} ............ OK ({info_or_error})  {elapsed:.1f} s")
        else:
            success_all = False
            print(f"{name:<22} ............ FAILED: {info_or_error}")
        print()

    print("========================================")
    if success_all:
        print("All AI models loaded successfully.")
        print()
        sys.exit(0)
    else:
        print("One or more AI models failed to load.")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
