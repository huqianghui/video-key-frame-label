#!/usr/bin/env python3
"""
手动下载并设置各种模型，特别处理PyTorch 2.6+兼容性问题
"""
import os
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np


def print_color(message, color_code="\033[0;32m"):
    """使用颜色打印消息"""
    reset_code = "\033[0m"
    print(f"{color_code}{message}{reset_code}")


def download_file(url, output_path):
    """从URL下载文件到指定路径"""
    try:
        print_color(f"正在下载: {url}", "\033[0;36m")
        urllib.request.urlretrieve(url, output_path)
        print_color(f"下载完成: {output_path}", "\033[0;32m")
        return True
    except Exception as e:
        print_color(f"下载失败: {e}", "\033[0;31m")
        return False


def download_yolo_models():
    """下载YOLO模型文件"""
    # 项目根目录
    base_dir = Path(__file__).parent.parent.absolute()
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True, parents=True)
    
    yolo_pt_path = models_dir / "yolov8s.pt"
    yolo_onnx_path = models_dir / "yolov8s.onnx"
    
    # 检查PyTorch版本
    try:
        import torch
        torch_version = torch.__version__
        print_color(f"检测到PyTorch版本: {torch_version}", "\033[0;36m")
        using_torch_26_plus = torch_version.startswith("2.6") or torch_version.startswith("2.7") or torch_version.startswith("2.8")
    except ImportError:
        using_torch_26_plus = False
        print_color("未检测到PyTorch", "\033[0;33m")
    
    # 下载所有格式的模型文件
    success = False
    
    # 1. 尝试直接下载PyTorch模型
    if not yolo_pt_path.exists():
        print_color("尝试下载PyTorch模型文件...", "\033[0;36m")
        success = download_file(
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt", 
            yolo_pt_path
        )
    else:
        print_color(f"PyTorch模型已存在: {yolo_pt_path}", "\033[0;32m")
        success = True
    
    # 2. 如果是PyTorch 2.6+，也下载ONNX模型
    if using_torch_26_plus and not yolo_onnx_path.exists():
        print_color("检测到PyTorch 2.6+，尝试下载ONNX模型作为备用...", "\033[0;36m")
        onnx_success = download_file(
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.onnx", 
            yolo_onnx_path
        )
        if not onnx_success:
            print_color("尝试其他ONNX模型源...", "\033[0;33m")
            onnx_success = download_file(
                "https://ultralytics.com/assets/yolov8s.onnx", 
                yolo_onnx_path
            )
    elif yolo_onnx_path.exists():
        print_color(f"ONNX模型已存在: {yolo_onnx_path}", "\033[0;32m")
    
    # 3. 尝试创建一个空YAML模型配置
    yaml_path = models_dir / "yolov8s.yaml"
    if not yaml_path.exists():
        try:
            with open(yaml_path, 'w') as f:
                f.write("""
# YOLOv8s YAML
nc: 80  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple

backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 3, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C2f, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, C2f, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 6, C2f, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, C2f, [1024]],
   [-1, 1, SPPF, [1024, 5]],  # 9
  ]

head:
  [[-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C2f, [512]],  # 12
   
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C2f, [256]],  # 15 (P3/8-small)
   
   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 12], 1, Concat, [1]],  # cat head P4
   [-1, 3, C2f, [512]],  # 18 (P4/16-medium)
   
   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 9], 1, Concat, [1]],  # cat head P5
   [-1, 3, C2f, [1024]],  # 21 (P5/32-large)

   [[15, 18, 21], 1, Detect, [nc]],  # Detect(P3, P4, P5)
  ]
                """)
            print_color(f"创建了YOLOv8模型配置文件: {yaml_path}", "\033[0;32m")
        except Exception as e:
            print_color(f"创建YAML文件失败: {e}", "\033[0;31m")
    
    return success


def main():
    """主函数"""
    print_color("开始下载模型文件...", "\033[1;36m")
    
    yolo_success = download_yolo_models()
    
    print_color("\n下载结果摘要:", "\033[1;36m")
    print_color(f"YOLO模型: {'成功' if yolo_success else '失败'}", 
                "\033[0;32m" if yolo_success else "\033[0;31m")
    
    print_color("\n模型准备完成!", "\033[1;36m")
    
    return 0 if yolo_success else 1


if __name__ == "__main__":
    sys.exit(main())
