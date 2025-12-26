#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO模型转换工具
将YOLO的PyTorch模型(.pt)转换为ONNX格式，再转换为RKNN格式，用于RK3588S设备NPU加速

使用方法:
    python convert_yolo_to_rknn.py --pt_model yolov5s.pt --output_dir ./output

依赖:
    - torch
    - onnx
    - rknn-toolkit2
"""

import sys
import argparse
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YoloToRknnConverter:
    """YOLO模型转换器类"""
    
    def __init__(self, pt_model_path, output_dir='./output', img_size=640):
        """
        初始化转换器
        
        Args:
            pt_model_path: PyTorch模型路径 (.pt文件)
            output_dir: 输出目录
            img_size: 输入图像尺寸 (默认640)
        """
        self.pt_model_path = Path(pt_model_path)
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        
        # 检查输入文件
        if not self.pt_model_path.exists():
            raise FileNotFoundError(f"找不到PT模型文件: {self.pt_model_path}")
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置输出文件路径
        model_name = self.pt_model_path.stem
        self.onnx_path = self.output_dir / f"{model_name}.onnx"
        self.rknn_path = self.output_dir / f"{model_name}.rknn"
        
        logger.info(f"输入模型: {self.pt_model_path}")
        logger.info(f"ONNX输出: {self.onnx_path}")
        logger.info(f"RKNN输出: {self.rknn_path}")
    
    def pt_to_onnx(self, opset_version=12, simplify=True):
        """
        将PyTorch模型转换为ONNX格式
        
        Args:
            opset_version: ONNX opset版本 (默认12)
            simplify: 是否简化ONNX模型 (默认True)
        
        Returns:
            str: ONNX模型路径
        """
        try:
            import torch
            import onnx
            
            logger.info("=" * 60)
            logger.info("步骤 1: PT模型转换为ONNX")
            logger.info("=" * 60)
            
            # 加载PyTorch模型
            logger.info(f"加载PyTorch模型: {self.pt_model_path}")
            device = torch.device('cpu')
            
            # 尝试加载模型 - 支持YOLOv5/YOLOv8等不同格式
            try:
                # YOLOv5格式
                model = torch.load(self.pt_model_path, map_location=device)
                if isinstance(model, dict):
                    if 'model' in model:
                        model = model['model']
                    elif 'ema' in model:
                        model = model['ema']
                model = model.float()
                model.eval()
            except Exception as e:
                logger.error(f"加载模型失败: {e}")
                raise
            
            # 创建输入张量
            logger.info(f"创建输入张量 (batch_size=1, channels=3, height={self.img_size}, width={self.img_size})")
            dummy_input = torch.randn(1, 3, self.img_size, self.img_size, device=device)
            
            # 导出为ONNX
            logger.info(f"导出ONNX模型到: {self.onnx_path}")
            torch.onnx.export(
                model,
                dummy_input,
                str(self.onnx_path),
                verbose=False,
                opset_version=opset_version,
                input_names=['images'],
                output_names=['output'],
                dynamic_axes={
                    'images': {0: 'batch'},
                    'output': {0: 'batch'}
                }
            )
            
            # 验证ONNX模型
            logger.info("验证ONNX模型...")
            onnx_model = onnx.load(str(self.onnx_path))
            onnx.checker.check_model(onnx_model)
            logger.info("✓ ONNX模型验证通过")
            
            # 简化ONNX模型 (可选)
            if simplify:
                try:
                    import onnxsim
                    logger.info("简化ONNX模型...")
                    model_simp, check = onnxsim.simplify(onnx_model)
                    if check:
                        onnx.save(model_simp, str(self.onnx_path))
                        logger.info("✓ ONNX模型简化成功")
                    else:
                        logger.warning("ONNX模型简化失败，使用原始模型")
                except ImportError:
                    logger.warning("未安装onnx-simplifier，跳过简化步骤")
                except Exception as e:
                    logger.warning(f"简化ONNX模型时出错: {e}，使用原始模型")
            
            logger.info(f"✓ ONNX转换完成: {self.onnx_path}")
            return str(self.onnx_path)
            
        except ImportError as e:
            logger.error(f"缺少必要的库: {e}")
            logger.error("请安装: pip install torch onnx onnx-simplifier")
            raise
        except Exception as e:
            logger.error(f"PT到ONNX转换失败: {e}")
            raise
    
    def onnx_to_rknn(self, target_platform='rk3588', quantize=True, dataset=None):
        """
        将ONNX模型转换为RKNN格式
        
        Args:
            target_platform: 目标平台 (默认'rk3588')
            quantize: 是否进行量化 (默认True)
            dataset: 量化数据集路径 (可选)
        
        Returns:
            str: RKNN模型路径
        """
        try:
            from rknn.api import RKNN
        except ImportError:
            # 尝试备用导入路径
            try:
                from rknn_toolkit2.api import RKNN
            except ImportError:
                raise ImportError(
                    "无法导入RKNN模块。请确保已正确安装rknn-toolkit2。\n"
                    "参考: https://github.com/rockchip-linux/rknn-toolkit2"
                )
        
        logger.info("=" * 60)
        logger.info("步骤 2: ONNX模型转换为RKNN")
        logger.info("=" * 60)
        
        # 创建RKNN对象
        logger.info("初始化RKNN转换器...")
        rknn = RKNN(verbose=True)
        
        # 配置RKNN
        logger.info(f"配置目标平台: {target_platform}")
        
        # 加载ONNX模型
        logger.info(f"加载ONNX模型: {self.onnx_path}")
        ret = rknn.load_onnx(
            model=str(self.onnx_path)
        )
        if ret != 0:
            raise RuntimeError("加载ONNX模型失败")
        logger.info("✓ ONNX模型加载成功")
        
        # 构建RKNN模型
        logger.info("构建RKNN模型...")
        
        # 配置量化参数
        if quantize:
            logger.info("启用量化优化...")
            ret = rknn.build(
                do_quantization=True,
                dataset=dataset if dataset else None,
                rknn_batch_size=1
            )
        else:
            logger.info("不使用量化...")
            ret = rknn.build(
                do_quantization=False,
                rknn_batch_size=1
            )
        
        if ret != 0:
            raise RuntimeError("构建RKNN模型失败")
        logger.info("✓ RKNN模型构建成功")
        
        # 导出RKNN模型
        logger.info(f"导出RKNN模型到: {self.rknn_path}")
        ret = rknn.export_rknn(str(self.rknn_path))
        if ret != 0:
            raise RuntimeError("导出RKNN模型失败")
        logger.info("✓ RKNN模型导出成功")
        
        # 释放RKNN对象
        rknn.release()
        
        logger.info(f"✓ RKNN转换完成: {self.rknn_path}")
        return str(self.rknn_path)
    
    def convert(self, opset_version=12, simplify=True, target_platform='rk3588', 
                quantize=True, dataset=None):
        """
        执行完整的转换流程: PT -> ONNX -> RKNN
        
        Args:
            opset_version: ONNX opset版本
            simplify: 是否简化ONNX模型
            target_platform: 目标平台
            quantize: 是否进行量化
            dataset: 量化数据集路径
        
        Returns:
            tuple: (onnx_path, rknn_path)
        """
        logger.info("开始YOLO模型转换流程...")
        logger.info(f"输入图像尺寸: {self.img_size}x{self.img_size}")
        
        try:
            # 步骤1: PT -> ONNX
            onnx_path = self.pt_to_onnx(
                opset_version=opset_version,
                simplify=simplify
            )
            
            # 步骤2: ONNX -> RKNN
            rknn_path = self.onnx_to_rknn(
                target_platform=target_platform,
                quantize=quantize,
                dataset=dataset
            )
            
            logger.info("=" * 60)
            logger.info("转换完成！")
            logger.info("=" * 60)
            logger.info(f"ONNX模型: {onnx_path}")
            logger.info(f"RKNN模型: {rknn_path}")
            logger.info(f"模型可在{target_platform}设备上使用NPU加速")
            
            return onnx_path, rknn_path
            
        except Exception as e:
            logger.error(f"转换过程出错: {e}")
            raise


def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(
        description='YOLO模型转换工具 - PT到RKNN格式转换',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用
  python convert_yolo_to_rknn.py --pt_model yolov5s.pt

  # 指定输出目录和图像尺寸
  python convert_yolo_to_rknn.py --pt_model yolov5s.pt --output_dir ./models --img_size 640

  # 不使用量化
  python convert_yolo_to_rknn.py --pt_model yolov5s.pt --no_quantize

  # 指定量化数据集
  python convert_yolo_to_rknn.py --pt_model yolov5s.pt --dataset ./dataset.txt

注意:
  - 需要安装torch, onnx, rknn-toolkit2等依赖
  - 量化可以提高推理速度，但可能略微降低精度
  - 默认目标平台为rk3588，也支持rk3588s
        """
    )
    
    parser.add_argument(
        '--pt_model',
        type=str,
        required=True,
        help='输入的PyTorch模型路径 (.pt文件)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
        help='输出目录 (默认: ./output)'
    )
    
    parser.add_argument(
        '--img_size',
        type=int,
        default=640,
        help='输入图像尺寸 (默认: 640)'
    )
    
    parser.add_argument(
        '--opset_version',
        type=int,
        default=12,
        help='ONNX opset版本 (默认: 12)'
    )
    
    parser.add_argument(
        '--no_simplify',
        action='store_true',
        help='不简化ONNX模型'
    )
    
    parser.add_argument(
        '--target_platform',
        type=str,
        default='rk3588',
        choices=['rk3588', 'rk3588s'],
        help='目标平台 (默认: rk3588)'
    )
    
    parser.add_argument(
        '--no_quantize',
        action='store_true',
        help='不进行量化'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='量化数据集路径 (可选)'
    )
    
    args = parser.parse_args()
    
    try:
        # 创建转换器
        converter = YoloToRknnConverter(
            pt_model_path=args.pt_model,
            output_dir=args.output_dir,
            img_size=args.img_size
        )
        
        # 执行转换
        converter.convert(
            opset_version=args.opset_version,
            simplify=not args.no_simplify,
            target_platform=args.target_platform,
            quantize=not args.no_quantize,
            dataset=args.dataset
        )
        
        return 0
        
    except Exception as e:
        logger.error(f"转换失败: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
