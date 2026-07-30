# RM Radar Integrated Models

面向 RoboMaster 雷达视觉的综合模型权重发布，包含车辆、装甲板、装甲板数字、无人机和激光模块识别模型。

本仓库只发布模型说明与 Release 权重包，**不包含训练集、验证集、图片或标签**。

## 下载

从 [v1.0.0 Release](https://github.com/hyris188/RM-Radar-Integrated-Models/releases/tag/v1.0.0) 下载：

- [rm_radar_integrated_models-v1.0.0.tar.gz](https://github.com/hyris188/RM-Radar-Integrated-Models/releases/download/v1.0.0/rm_radar_integrated_models-v1.0.0.tar.gz)
- [rm_radar_integrated_models-v1.0.0.tar.gz.sha256](https://github.com/hyris188/RM-Radar-Integrated-Models/releases/download/v1.0.0/rm_radar_integrated_models-v1.0.0.tar.gz.sha256)

验证并解压：

```bash
sha256sum -c rm_radar_integrated_models-v1.0.0.tar.gz.sha256
tar -xzf rm_radar_integrated_models-v1.0.0.tar.gz
cd rm_radar_integrated_models-v1.0.0
sha256sum -c weights/SHA256SUMS
```

## 模型列表

| 文件 | 功能 | 格式 |
| --- | --- | --- |
| `car_detector_best.pt` | 车辆检测 | Ultralytics PyTorch |
| `armor_detector_best.pt` | 装甲板检测与红蓝颜色分类 | Ultralytics PyTorch |
| `armor_digit_classifier_best.pth` | 装甲板数字/兵种分类 | MobileNet PyTorch state dict |
| `drone_detector_best.pt` | 无人机检测训练权重 | Ultralytics PyTorch |
| `drone_detector_best.onnx` | 无人机检测部署权重 | ONNX |
| `laser_module_pose_best.onnx` | 激光模块框与中心点检测 | ONNX Pose |

详细输入、类别、阈值、指标和限制见 [MODEL_CARD.md](MODEL_CARD.md)。许可证与第三方声明见 [LICENSE.md](LICENSE.md) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 推荐部署参数

- 无人机置信度阈值：`0.60`
- 激光模块置信度阈值：`0.50`
- 无人机和激光模块：每帧仅保留最高置信度的一个结果
- ONNX Runtime：优先使用 CUDA Execution Provider

阈值和 `max_det=1` 是推理参数，不写入模型权重，使用者需要在自己的推理程序中设置。

## 相关代码

基础雷达算法项目：<https://github.com/hkustenterprize/RM2025-Radar-Algorithm>

本模型包不承诺在比赛环境中零误检。使用前必须在实际相机、镜头、曝光和场地背景下重新验证。
