# Project Health & Evaluation Report

## 1. Executive Summary

This report establishes the comprehensive evaluation findings of the Smart ID Photo Processor evaluation framework. Operating on test datasets and real-world student ID photographs (including individuals wearing hijabs or prescription eyeglasses), the framework observes, explains, and visualizes system behavior without altering production inference.

---

## 2. Key Questions Answered

### 1. Which validator causes the most failures?
- **FaceVisibilityValidator** causes the highest frequency of failures, primarily due to strict pixel-area thresholds and parser false negatives on eyebrow and eye regions.

### 2. Which semantic regions fail most frequently?
- **LEFT_BROW** and **RIGHT_BROW** fail most frequently, followed by **LEFT_EYE** and **RIGHT_EYE**. This is primarily driven by BiSeNet's lower segmentation sensitivity on thin eyebrow structures in CelebAMask-HQ.

### 3. Are failures primarily caused by parser limitations or by validation thresholds?
- Failures are **primarily caused by parser limitations** (BiSeNet false negatives on eyebrows and broad `FacePart.HAT` occlusion grouping), rather than inappropriate validation thresholds.

### 4. Which thresholds appear overly strict?
- None of the thresholds appear overly strict; continuous weighted evidence fusion successfully mitigates minor segmentation dropouts. However, `FACE_VISIBILITY_REQUIRED_PART_THRESHOLDS` for eyes and nose operate close to empirical limits.

### 5. Which failures are genuine and should remain rejected?
- Failures caused by prohibited headwear (caps, helmets), extreme head pose deviations (yaw/pitch > 15°), severe blur, and incorrect lighting (dark/washed-out) are genuine and correctly rejected.

### 6. Which failures are most likely recoverable by improving the parser?
- Eyebrow missing-detections (false negatives) and false-positive occlusion classifications on religious head coverings (hijabs) are fully recoverable by improving the parser.

### 7. Which failures are most likely recoverable by threshold calibration?
- Minor borderline face size ratios and contrast variations near boundary conditions.

### 8. Which failures require collecting new training data?
- Diverse real-world student headwear variations (various hijab styles, textured headscarves, varied lighting conditions) not fully represented in CelebAMask-HQ.

### 9. Which parser classes should be prioritized during future fine-tuning?
- 1. **LEFT_BROW** / **RIGHT_BROW** (eyebrows)
- 2. **HAT** (subclassifying religious head coverings from caps/helmets)
- 3. **LEFT_EYE** / **RIGHT_EYE** (eye segmentation under glasses)

### 10. Prioritized Roadmap for System Improvement
1. **Phase 2.1 (Dataset Collection & Annotation)**: Gather a curated set of 500+ diverse student ID photos (including hijabs and glasses).
2. **Phase 2.2 (BiSeNet Fine-Tuning)**: Fine-tune BiSeNet on the target ID photo distribution, prioritizing eyebrow classes and hierarchical headwear classes.
3. **Phase 2.3 (Threshold Calibration)**: Calibrate visibility and pose thresholds on the fine-tuned parser outputs.
4. **Phase 2.4 (Production Release)**: Deploy the retrained parser into the stable architecture.
