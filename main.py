import argparse
import os
import subprocess
import sys
from pathlib import Path

from config import DATA_DIR, OUTPUT_DIR


def check_environment():
    """检查运行环境是否满足要求"""
    # 检查ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("✓ ffmpeg 已安装")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("✗ ffmpeg 未安装或不可用")
        install_choice = input("是否尝试自动安装 ffmpeg? (y/n): ")
        if install_choice.lower() == 'y':
            try:
                subprocess.run(['python', 'setup_ffmpeg.py'], check=True)
            except subprocess.SubprocessError:
                print("自动安装失败，请手动安装 ffmpeg")
                print("Ubuntu/Debian: sudo apt-get install ffmpeg")
                print("CentOS/RHEL: sudo yum install ffmpeg")
                return False
        else:
            print("请手动安装 ffmpeg 后再运行程序")
            return False
    
    # 检查配置文件
    from dotenv import load_dotenv
    env_loaded = load_dotenv()
    if not env_loaded:
        print("⚠️ .env 文件不存在或加载失败，将使用默认配置")
    
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="安防视频分析系统")
    # 为 video_path 提供默认值：DATA_DIR
    parser.add_argument("--video_path", type=str, default="/home/azureuser/video-process/video/57e54737-1c62-4605-a1d0-4de4068915d9.mp4", help="输入视频文件或文件夹路径")
    parser.add_argument("--output_dir", type=str, default="/home/azureuser/video-process/output", help="输出目录路径")
    parser.add_argument("--model", type=str, default="phi-4-multi", 
                       choices=["phi-4-multi", "gemma-3"], help="选择初始筛选模型")
    parser.add_argument("--advanced_model", type=str, default="gpt-4o", help="高级分析模型")
    parser.add_argument("--use_yolo", action="store_true", default=True, help="是否使用YOLO进行对象检测")
    parser.add_argument("--batch", action="store_true", help="批处理模式")
    return parser.parse_args()

def main():
    # 检查环境
    if not check_environment():
        print("环境检查失败，程序将退出")
        return 1
        
    args = parse_args()
    
    # 自动决定处理方式：如果传入路径为目录或指定了 batch 模式，则当作批处理
    video_path = Path(args.video_path)
    if args.batch or video_path.is_dir():
        video_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(video_path) 
                      for f in filenames if f.endswith(('.mp4', '.avi', '.mkv'))]
    else:
        video_files = [str(video_path)]
    
    if not video_files:
        print(f"未找到视频文件: {args.video_path}")
        return 1
    
    # 导入依赖，如果出错则打印详细信息
    try:
        from modules.orchestration import VideoAnalysisPipeline
    except ImportError as e:
        print(f"导入必要模块失败: {e}")
        print("请确保已安装所有依赖: pip install -r requirements.txt")
        return 1
    
    # 创建分析管道
    try:
        print("初始化视频分析管道...")
        pipeline = VideoAnalysisPipeline(
            initial_model=args.model,
            advanced_model=args.advanced_model,
            use_yolo=args.use_yolo
        )
    except Exception as e:
        print(f"初始化分析管道失败: {e}")
        return 1
    
    # 分析视频
    for i, video_file in enumerate(video_files):
        print(f"处理视频 [{i+1}/{len(video_files)}]: {video_file}")
        try:
            event_result = pipeline.process_video(video_file, args.output_dir)
            # 输出事件分析结果
            print(f"事件结果: {event_result}")
        except Exception as e:
            print(f"处理视频时发生错误: {e}")
            import traceback
            traceback.print_exc()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
