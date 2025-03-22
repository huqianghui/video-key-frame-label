#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from pathlib import Path


def print_color(message, color_code="\033[0;32m"):
    """使用颜色打印消息"""
    reset_code = "\033[0m"
    print(f"{color_code}{message}{reset_code}")

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_color(f"Python版本正常: {sys.version}")
        return True
    else:
        print_color(f"Python版本过低: {sys.version}. 需要Python 3.8+", "\033[0;31m")
        return False

def check_ffmpeg():
    """检查ffmpeg是否安装"""
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    
    if ffmpeg_path and ffprobe_path:
        print_color(f"ffmpeg已安装: {ffmpeg_path}")
        print_color(f"ffprobe已安装: {ffprobe_path}")
        return True
    else:
        print_color("ffmpeg未完全安装", "\033[0;31m")
        if not ffmpeg_path:
            print_color("找不到ffmpeg命令", "\033[0;31m")
        if not ffprobe_path:
            print_color("找不到ffprobe命令", "\033[0;31m")
        
        print_color("请安装ffmpeg:", "\033[1;33m")
        print_color("Ubuntu/Debian: sudo apt-get install ffmpeg", "\033[1;33m")
        print_color("CentOS/RHEL: sudo yum install ffmpeg", "\033[1;33m")
        print_color("Windows: 下载并安装 https://ffmpeg.org/download.html", "\033[1;33m")
        return False

def check_pip_packages():
    """检查需要的pip包是否安装"""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        print_color(f"PyTorch已安装: {torch.__version__}, CUDA可用: {gpu_available}")
    except ImportError:
        print_color("PyTorch未安装或导入失败", "\033[0;31m")
    
    try:
        import cv2
        print_color(f"OpenCV已安装: {cv2.__version__}")
    except ImportError:
        print_color("OpenCV未安装或导入失败", "\033[0;31m")
    
    try:
        import numpy
        print_color(f"NumPy已安装: {numpy.__version__}")
    except ImportError:
        print_color("NumPy未安装或导入失败", "\033[0;31m")
        
    try:
        from dotenv import load_dotenv
        print_color("python-dotenv已安装")
    except ImportError:
        print_color("python-dotenv未安装或导入失败", "\033[0;31m")
    
    try:
        import requests
        print_color(f"requests已安装: {requests.__version__}")
    except ImportError:
        print_color("requests未安装或导入失败", "\033[0;31m")

def check_yolo_model():
    """检查YOLO模型是否存在"""
    # 项目根目录
    base_dir = Path(__file__).parent.absolute()
    models_dir = base_dir / "models"
    yolo_model_path = models_dir / "yolov8s.pt"
    
    if yolo_model_path.exists():
        print_color(f"YOLO模型文件存在: {yolo_model_path}")
    else:
        print_color(f"YOLO模型文件不存在: {yolo_model_path}", "\033[0;31m")
        print_color("将尝试下载YOLO模型...", "\033[1;33m")
        
        try:
            # 确保models目录存在
            models_dir.mkdir(exist_ok=True)
            
            # 尝试使用ultralytics下载模型
            import torch
            from ultralytics import YOLO
            
            print_color("正在下载YOLO模型...", "\033[1;33m")
            model = YOLO("yolov8s.pt")
            
            # 检查是否下载到当前目录
            current_dir_model = Path("yolov8s.pt")
            if current_dir_model.exists():
                print_color(f"模型下载到当前目录: {current_dir_model}")
                print_color(f"移动到models目录: {yolo_model_path}")
                import shutil
                shutil.move(str(current_dir_model), str(yolo_model_path))
                print_color(f"YOLO模型文件已保存: {yolo_model_path}")
            else:
                print_color("模型下载完成，但找不到文件。请手动下载模型。", "\033[0;31m")
                
        except Exception as e:
            print_color(f"下载YOLO模型失败: {e}", "\033[0;31m")
            print_color("请手动下载YOLO模型:", "\033[1;33m")
            print_color("1. 运行 'pip install ultralytics'", "\033[1;33m")
            print_color("2. 运行 'from ultralytics import YOLO; YOLO(\"yolov8s.pt\")'", "\033[1;33m")
            print_color("3. 将下载的模型移动到 models/yolov8s.pt", "\033[1;33m")

def check_env_file():
    """检查.env文件是否存在"""
    base_dir = Path(__file__).parent.absolute()
    env_file = base_dir / ".env"
    env_example = base_dir / ".env.example"
    
    if env_file.exists():
        print_color(".env文件已存在")
    else:
        print_color(".env文件不存在", "\033[0;31m")
        if env_example.exists():
            print_color("将.env.example复制为.env", "\033[1;33m")
            import shutil
            shutil.copy(str(env_example), str(env_file))
            print_color("请编辑.env文件，填入您的API密钥和端点", "\033[1;33m")
        else:
            print_color(".env.example也不存在，请创建.env文件", "\033[0;31m")

def main():
    """主函数"""
    print_color("正在检查系统依赖...", "\033[1;36m")
    
    py_ok = check_python_version()
    ff_ok = check_ffmpeg()
    
    print_color("\n正在检查Python依赖...", "\033[1;36m")
    check_pip_packages()
    
    print_color("\n正在检查YOLO模型...", "\033[1;36m")
    check_yolo_model()
    
    print_color("\n正在检查配置文件...", "\033[1;36m")
    check_env_file()
    
    print_color("\n检查完成!", "\033[1;36m")
    
    if not py_ok or not ff_ok:
        print_color("\n⚠️ 有一些依赖项缺失，请安装后再运行程序", "\033[0;31m")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
