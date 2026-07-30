# v1.0.0

First public weight-only release of the integrated RM radar vision models.

Included:

- car detector
- armor detector and red/blue/dead classification
- armor digit/role classifier
- drone detector in PyTorch and ONNX formats
- laser-module box and center-keypoint ONNX model
- per-file SHA256 checksums
- model card and third-party license notices

Not included:

- training or validation datasets
- images or labels
- training caches and run directories
- local camera calibration
- referee-system logs or credentials
- historical/rollback weights

Recommended drone confidence threshold: 0.60. Recommended laser-module threshold: 0.50.
