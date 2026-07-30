# RM Radar Vision Framework

面向 RoboMaster 雷达站的视觉训练与部署代码框架，覆盖车辆、装甲板、装甲板数字、无人机和激光模块检测链路。

本项目中用于**车辆与装甲板标注、训练、识别和检测**的代码框架，参考并使用了香港科技大学 ENTERPRIZE 战队于 2025 年开源的 [RM2025 Radar Algorithm](https://github.com/hkustenterprize/RM2025-Radar-Algorithm) 项目。感谢原项目团队的开源贡献；相关代码保留原 MIT 版权与许可证声明。

> **开源范围说明**
>
> 本项目仅开放代码框架，**不提供任何训练权重，数据集不公开**。使用者需要自行采集、筛选和标注数据，并从头训练适用于自身相机与赛场环境的模型。

## 公开内容

| 目录 | 内容 |
| --- | --- |
| `training/` | 通用 YOLO 训练入口、装甲板数字分类训练源码 |
| `augmentation/` | 保持 YOLO 框同步的数据增强代码 |
| `preprocessing/` | 视频抽帧、数据划分、标签检查与泄漏检查 |
| `deployment/` | 雷达端海康相机综合推理与可视化代码 |
| `driver/hik_camera/` | 海康 MVS 相机接入代码，不包含厂商 SDK 二进制 |
| `configs/` | 车辆、装甲板、无人机、激光模块的数据和训练 YAML 模板 |
| `docs/ANNOTATION_GUIDE.md` | 检测框、颜色、遮挡、负样本和关键点标注规范 |

## 不公开内容

- 所有训练完成的模型权重及备份，包括 `.pt`、`.pth`、`.onnx`、`.engine`、checkpoint；
- 训练集、验证集、测试集、图片、视频及标签；
- 本地训练输出、日志、缓存、相机标定结果和现场配置。

仓库通过 `.gitignore` 排除上述内容。提交前仍应执行人工复核，禁止使用 `git add -f` 绕过限制。

## 环境安装

```bash
git clone https://github.com/hyris188/RM-Radar-Vision-Framework.git
cd RM-Radar-Vision-Framework
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

海康相机实时推理还需要单独安装 Hikrobot MVS SDK 和 ROS 2 Humble。厂商 SDK 不包含在本仓库中。

## 准备自己的数据

数据目录约定如下，`datasets/` 已被 Git 忽略：

```text
datasets/car/
├── images/train/
├── images/val/
├── labels/train/
└── labels/val/
```

其他任务使用相同结构。类别顺序和标注要求必须遵循 [标注规范](docs/ANNOTATION_GUIDE.md)。

数据预处理示例：

```bash
python preprocessing/extract_video_frames.py \
  --video /path/to/private_video.mp4 \
  --output /path/to/private_frames

python preprocessing/validate_yolo_dataset.py \
  --dataset datasets/drone --classes 1
```

## 从头训练

配置默认使用网络结构 YAML，不附带预训练权重：

```bash
python training/train_yolo.py --config configs/train/car.yaml
python training/train_yolo.py --config configs/train/armor.yaml
python training/train_yolo.py --config configs/train/drone.yaml
python training/train_yolo.py --config configs/train/laser_module_pose.yaml
```

装甲板数字分类：

```bash
python -m training.train_digit_classifier \
  --dataset-path datasets/digit \
  --epochs 100
```

训练产生的权重只保存在本地 `runs/` 或 `weights/`，不得提交到本仓库。

## 雷达端部署

将自行训练得到的权重放在本地 `weights/`，复制并修改相机配置：

```bash
cp configs/deployment/device.example.yaml configs/deployment/device.local.yaml

./run_integrated_realtime.sh \
  --device-config configs/deployment/device.local.yaml \
  --car-weights weights/car_detector.pt \
  --armor-weights weights/armor_detector.pt \
  --digit-weights weights/armor_digit_classifier.pth \
  --drone-weights weights/drone_detector.onnx \
  --laser-weights weights/laser_module_pose.onnx
```

部署代码仅提供流程参考。不同相机 Bayer 排列、曝光、分辨率和场地背景都需要现场重新验证。

## 许可证

本仓库代码保留原项目及第三方许可证声明，详见 [LICENSE.md](LICENSE.md) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
