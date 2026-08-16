# Experiment A: Head-Only BiSeNet Fine-Tuning

## Hypothesis
Catastrophic forgetting in the previous full-network fine-tuning was caused by
updating the backbone on a small dataset. Freezing feature extraction and
training only the output heads should preserve non-target class performance.

## Architecture Freezing

| Component | Status | Parameters |
|-----------|--------|------------|
| ContextPath (backbone + ARM + conv_heads) | FROZEN | cp (12,456,256 params) |
| FeatureFusionModule | FROZEN | ffm (98,816 params) |
| conv_out (main head) | TRAINABLE | conv_out (595,200/595,200 trainable) |
| conv_out16 (aux head) | TRAINABLE | conv_out16 (75,072/75,072 trainable) |
| conv_out32 (aux head) | TRAINABLE | conv_out32 (75,072/75,072 trainable) |

## Parameter Counts

| Metric | Value |
|--------|------:|
| Total parameters | 13,300,416 |
| Frozen parameters | 12,555,072 |
| Trainable parameters | 745,344 |
| Trainable % | 5.60% |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 20 |
| Batch size | 4 |
| Learning rate | 1e-05 |
| Optimizer | adamw |
| Scheduler | cosine |
| Class weights | {4: 2.0, 5: 2.0, 6: 1.0} |
| Aux16 weight | 0.4 |
| Aux32 weight | 0.4 |
| Checkpoint criterion | **Full 19-class val mIoU** |
| Target classes | LEFT_BROW, RIGHT_BROW, LEFT_EYE, RIGHT_EYE, EYE_GLASS, HAT |

## Results

| Metric | Value |
|--------|------:|
| Best epoch | 18 |
| Best val mIoU (19-class) | 0.767996 |
| Best val target mIoU | 0.682147 |
| Best val non-target mIoU | 0.814823 |

## Per-Class IoU (Best Checkpoint, Validation)

| Class | IoU |
|-------|----:|
| BACKGROUND | 0.914111 |
| SKIN | 0.947279 |
| LEFT_BROW | 0.722586 |
| RIGHT_BROW | 0.610987 |
| LEFT_EYE | 0.548332 |
| RIGHT_EYE | 0.523824 |
| EYE_GLASS | 0.853435 |
| LEFT_EAR | 0.829331 |
| RIGHT_EAR | 0.817304 |
| EAR_RING | N/A |
| NOSE | 0.912180 |
| MOUTH | 0.832448 |
| UPPER_LIP | 0.821648 |
| LOWER_LIP | 0.853468 |
| NECK | 0.736973 |
| NECKLACE | N/A |
| CLOTH | 0.377197 |
| HAIR | 0.921118 |
| HAT | 0.833717 |


## Dataset

| Split | Samples |
|-------|--------:|
| Train | 107 |
| Val | 20 |
| Test | 26 |

## Reproducibility

- Seed: 42
- Python: 3.12.2
- PyTorch: 2.13.0+cpu
- CUDA available: False
- Device: cpu
