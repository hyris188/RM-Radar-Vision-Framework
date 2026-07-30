# Model Card: RM Radar Integrated Models v1.0.0

## 概述

该发布包由五个视觉任务、六个权重文件组成，用于一条相机画面上的车辆、装甲板、装甲板数字、无人机和激光模块识别。训练和验证数据不随模型发布。

## 模型规格

| 模型 | 任务与类别 | 部署输入 | 记录指标 |
| --- | --- | --- | --- |
| `car_detector_best.pt` | Detect；`car` | 综合流水线默认 640，原训练配置 1280 | mAP50 0.9588 |
| `armor_detector_best.pt` | Detect；`dead_armor`, `red_armor`, `blue_armor` | 192 | mAP50 0.900 |
| `armor_digit_classifier_best.pth` | MobileNet；`1`, `2`, `3`, `4`, `S`, `Q` | RGB 64×64 | accuracy 0.9179 |
| `drone_detector_best.pt/.onnx` | Detect；`drone` | 640×640 | P 0.9866, R 0.9940, mAP50 0.9942, mAP50-95 0.8117 |
| `laser_module_pose_best.onnx` | Pose；`laser_module`，一个中心关键点 | 640×640 | box/keypoint mAP50 0.995 |

无人机指标是在当前验证划分上重新评估得到。不同场景、相机、曝光或数据划分不能直接保证相同结果。

## v1.0.0 模型选择

无人机权重采用早期使用完整、未人工删减 Hikrobot/MVS 数据集训练的 `drone_mvs_gpu_e30` 最佳检查点。该版本在当前验证划分上的自动指标较高，但在复杂背景中仍可能把周围图案识别为无人机。

推荐无人机阈值为 0.60；低于该阈值不显示，并设置 `max_det=1`。阈值不是权重的一部分。

## 数据说明

- 本发布不包含任何训练图片、验证图片、标签、缓存或数据集路径。
- 权重使用项目维护者本地整理的数据训练。
- 数据不公开，因此第三方不能只依靠本发布包完全复现训练过程或独立审计全部标签质量。
- 使用者应在自己的目标域数据上重新验证性能、公平性和安全性。

## 预处理与后处理

- YOLO/ONNX 模型使用 Ultralytics 常规 letterbox、归一化和 NMS 流程。
- 无人机：`conf=0.60`, `max_det=1`。
- 激光模块：`conf=0.50`, `max_det=1`；中心关键点推荐阈值 `0.25`。
- 相机 Bayer 图像必须先用正确 Bayer 排列转换为 BGR；错误通道顺序会导致红蓝装甲板颜色错误。

## 测试环境

- Ubuntu 22.04
- Python 3.10.12
- PyTorch 2.13.0+cu130
- Ultralytics 8.4.71
- ONNX Runtime 1.23.2，CUDA Execution Provider
- OpenCV 4.10.0
- NVIDIA GeForce RTX 5060 Laptop GPU

其他版本可能可以运行，但没有包含在本次发布验证范围内。

## 限制与风险

- 无人机模型可能对具有相似轮廓、纹理或颜色的背景杂物产生误检。
- 装甲板颜色依赖相机 Bayer 格式和颜色转换正确。
- 装甲板数字分类器需要与发布它的 MobileNet 网络结构一致，单独加载 state dict 不足以恢复网络定义。
- 模型只输出图像检测结果，不提供真实世界坐标；相机内外参、PnP、场地映射和裁判系统通信均不在该模型包中。
- 模型不应作为无人监督的安全关键决策依据。

## 完整性

发布包内 `weights/SHA256SUMS` 记录每个权重的 SHA256。下载后必须先校验外层压缩包，再校验内部权重。

## 许可证

参见 `LICENSE.md` 与 `THIRD_PARTY_NOTICES.md`。模型使用的框架和上游项目具有不同许可证，不能仅以根项目的 MIT 许可证概括全部文件。
