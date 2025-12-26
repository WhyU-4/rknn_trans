# RKNN Trans - YOLO模型转换工具

基于rknn-toolkit2的YOLO模型转换工具，可将YOLO的PyTorch模型(.pt)转换为ONNX格式，再转换为RKNN格式，用于在RK3588/RK3588S设备上使用NPU加速。

## 功能特性

- ✅ 支持YOLO系列模型转换 (YOLOv5, YOLOv8等)
- ✅ 自动完成PT → ONNX → RKNN的完整转换流程
- ✅ 支持模型量化优化，提升推理速度
- ✅ 支持自定义输入尺寸
- ✅ 支持RK3588/RK3588S平台
- ✅ 完整的错误处理和日志输出
- ✅ 简单易用的命令行接口

## 环境要求

- Python 3.6+
- PyTorch
- ONNX
- rknn-toolkit2

## 安装依赖

### 1. 安装基础依赖

```bash
pip install -r requirements.txt
```

### 2. 安装RKNN Toolkit2

RKNN Toolkit2需要从官方渠道下载安装:

1. 访问 [RKNN Toolkit2 GitHub](https://github.com/rockchip-linux/rknn-toolkit2)
2. 下载对应Python版本的wheel包
3. 安装:

```bash
pip install rknn_toolkit2-*-cp3*-*.whl
```

## 使用方法

### 基本使用

```bash
python convert_yolo_to_rknn.py --pt_model yolov5s.pt
```

这将使用默认参数转换模型:
- 输出目录: `./output`
- 图像尺寸: `640x640`
- 启用量化
- 目标平台: `rk3588`

### 高级使用

#### 指定输出目录和图像尺寸

```bash
python convert_yolo_to_rknn.py --pt_model yolov5s.pt --output_dir ./models --img_size 640
```

#### 不使用量化

```bash
python convert_yolo_to_rknn.py --pt_model yolov5s.pt --no_quantize
```

#### 使用量化数据集

```bash
python convert_yolo_to_rknn.py --pt_model yolov5s.pt --dataset ./dataset.txt
```

量化数据集文件格式(每行一个图像路径):
```
/path/to/image1.jpg
/path/to/image2.jpg
/path/to/image3.jpg
```

#### 针对RK3588S平台

```bash
python convert_yolo_to_rknn.py --pt_model yolov5s.pt --target_platform rk3588s
```

### 完整参数说明

```
--pt_model          输入的PyTorch模型路径 (.pt文件) [必需]
--output_dir        输出目录 (默认: ./output)
--img_size          输入图像尺寸 (默认: 640)
--opset_version     ONNX opset版本 (默认: 12)
--no_simplify       不简化ONNX模型
--target_platform   目标平台 (rk3588/rk3588s, 默认: rk3588)
--no_quantize       不进行量化
--dataset           量化数据集路径 (可选)
```

查看帮助:
```bash
python convert_yolo_to_rknn.py --help
```

## 转换流程

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  YOLO PT    │ ───> │    ONNX     │ ───> │    RKNN     │
│   模型      │      │    模型     │      │    模型     │
└─────────────┘      └─────────────┘      └─────────────┘
    .pt文件            .onnx文件            .rknn文件
                                          (NPU加速)
```

1. **PT → ONNX**: 使用PyTorch的导出功能将模型转换为ONNX格式
2. **ONNX简化**: 使用onnx-simplifier优化模型结构(可选)
3. **ONNX → RKNN**: 使用RKNN Toolkit2转换为RKNN格式，支持量化优化

## 性能优化建议

### 模型选择

- **实时应用**: YOLOv5n/s (速度快)
- **平衡性能**: YOLOv5m
- **高精度**: YOLOv5l/x

### 量化建议

- **实时检测场景**: 建议开启量化，显著提升速度
- **高精度要求**: 建议关闭量化，保持精度
- **提供量化数据集**: 可提高量化模型的精度

### 输入尺寸建议

- **实时视频检测**: 320-416
- **标准检测**: 640
- **高精度检测**: 1280

## 示例

### 示例1: 转换YOLOv5s (标准配置)

```bash
python convert_yolo_to_rknn.py --pt_model yolov5s.pt
```

### 示例2: 转换YOLOv5n (实时优化)

```bash
python convert_yolo_to_rknn.py --pt_model yolov5n.pt --img_size 416
```

### 示例3: 转换YOLOv5l (高精度)

```bash
python convert_yolo_to_rknn.py --pt_model yolov5l.pt --img_size 1280 --no_quantize
```

### 示例4: 使用量化数据集

```bash
python convert_yolo_to_rknn.py --pt_model yolov5s.pt --dataset ./coco_sample.txt
```

## 输出文件

转换完成后，在输出目录中会生成:

- `{model_name}.onnx` - ONNX格式模型
- `{model_name}.rknn` - RKNN格式模型 (可在RK3588/RK3588S上使用)

## 常见问题

### Q: 转换后精度下降怎么办？

A: 尝试以下方法:
- 关闭量化 (`--no_quantize`)
- 提供高质量的量化数据集 (`--dataset`)
- 增加输入尺寸

### Q: 模型文件太大？

A: 尝试以下方法:
- 使用较小的模型 (如YOLOv5n/s)
- 开启量化
- 减小输入尺寸

### Q: 推理速度慢？

A: 尝试以下方法:
- 减小输入尺寸
- 使用量化
- 使用较小的模型

### Q: 内存不足？

A: 减小输入尺寸或使用较小的模型

### Q: 如何在RK3588设备上使用转换后的模型？

A: 使用RKNN Runtime API加载.rknn文件进行推理。参考[RKNN Runtime文档](https://github.com/rockchip-linux/rknn-toolkit2)。

## 技术支持

- RKNN Toolkit2: https://github.com/rockchip-linux/rknn-toolkit2
- YOLOv5: https://github.com/ultralytics/yolov5
- ONNX: https://github.com/onnx/onnx

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！