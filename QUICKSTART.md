# 快速开始指南 (Quick Start Guide)

本指南将帮助您快速上手使用YOLO到RKNN的模型转换工具。

## 步骤1: 安装依赖

```bash
# 安装基础依赖
pip install torch torchvision onnx onnx-simplifier numpy opencv-python

# 安装RKNN Toolkit2 (需要从官方下载wheel包)
# 下载地址: https://github.com/rockchip-linux/rknn-toolkit2/tree/master/rknn-toolkit2/packages
pip install rknn_toolkit2-*.whl
```

## 步骤2: 准备YOLO模型

确保您有一个训练好的YOLO PyTorch模型文件（.pt格式）。

示例:
- `yolov5s.pt` - YOLOv5小型模型
- `yolov5m.pt` - YOLOv5中型模型
- `yolov8n.pt` - YOLOv8纳米模型

## 步骤3: 运行转换

### 基本转换 (推荐新手)

```bash
python convert_yolo_to_rknn.py --pt_model yolov5s.pt
```

这将自动完成以下步骤:
1. 将PT模型转换为ONNX格式
2. 简化ONNX模型
3. 将ONNX转换为RKNN格式（带量化）
4. 输出文件到 `./output` 目录

### 自定义输出目录

```bash
python convert_yolo_to_rknn.py --pt_model yolov5s.pt --output_dir ./my_models
```

### 不同的输入尺寸

```bash
# 320x320 (快速模式)
python convert_yolo_to_rknn.py --pt_model yolov5s.pt --img_size 320

# 1280x1280 (高精度模式)
python convert_yolo_to_rknn.py --pt_model yolov5l.pt --img_size 1280
```

## 步骤4: 查看输出

转换成功后，在输出目录中会找到:

```
output/
├── yolov5s.onnx    # ONNX模型 (中间格式)
└── yolov5s.rknn    # RKNN模型 (最终格式，用于RK3588)
```

## 步骤5: 部署到RK3588设备

将生成的 `.rknn` 文件复制到RK3588/RK3588S设备上，使用RKNN Runtime进行推理。

## 常见使用场景

### 场景1: 实时视频检测 (追求速度)

```bash
python convert_yolo_to_rknn.py \
  --pt_model yolov5n.pt \
  --img_size 416 \
  --target_platform rk3588s
```

### 场景2: 高精度检测 (追求准确度)

```bash
python convert_yolo_to_rknn.py \
  --pt_model yolov5l.pt \
  --img_size 1280 \
  --no_quantize
```

### 场景3: 平衡模式 (速度和精度)

```bash
python convert_yolo_to_rknn.py \
  --pt_model yolov5s.pt \
  --img_size 640
```

### 场景4: 使用自定义量化数据集

```bash
# 创建数据集文件 (dataset.txt)
# 每行一个图像路径
echo "/path/to/image1.jpg" > dataset.txt
echo "/path/to/image2.jpg" >> dataset.txt
echo "/path/to/image3.jpg" >> dataset.txt

# 运行转换
python convert_yolo_to_rknn.py \
  --pt_model yolov5s.pt \
  --dataset dataset.txt
```

## 故障排除

### 问题1: 找不到PT模型

```
错误: FileNotFoundError: 找不到PT模型文件
解决: 检查文件路径是否正确，使用绝对路径或相对路径
```

### 问题2: 内存不足

```
错误: RuntimeError: CUDA out of memory
解决: 减小 --img_size 参数，例如从640改为416或320
```

### 问题3: 缺少依赖库

```
错误: ImportError: No module named 'torch'
解决: 安装缺失的库: pip install torch
```

### 问题4: RKNN Toolkit2未安装

```
错误: ImportError: No module named 'rknn'
解决: 从官方下载并安装rknn-toolkit2的wheel包
```

## 性能参考

不同配置的转换时间和模型大小参考:

| 模型 | 输入尺寸 | 量化 | ONNX大小 | RKNN大小 | 转换时间 |
|------|---------|------|----------|----------|----------|
| YOLOv5n | 640 | 是 | ~7MB | ~2MB | ~2分钟 |
| YOLOv5s | 640 | 是 | ~28MB | ~7MB | ~3分钟 |
| YOLOv5m | 640 | 是 | ~84MB | ~21MB | ~5分钟 |
| YOLOv5l | 640 | 否 | ~180MB | ~180MB | ~8分钟 |

*注: 实际时间取决于硬件配置

## 下一步

1. 了解如何在RK3588上使用RKNN Runtime加载和运行模型
2. 优化模型参数以获得更好的性能
3. 使用自己的数据集训练YOLO模型

## 获取帮助

如果遇到问题，可以:
1. 查看 `README.md` 获取详细文档
2. 查看 `config_example.txt` 了解更多配置选项
3. 运行 `python convert_yolo_to_rknn.py --help` 查看所有参数

---

祝您使用愉快！如有问题欢迎提Issue。
