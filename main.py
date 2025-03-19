import argparse
import os
from pathlib import Path

from config import DATA_DIR, OUTPUT_DIR
from modules.orchestration import VideoAnalysisPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="安防视频分析系统")
    # 为 video_path 提供默认值：DATA_DIR
    parser.add_argument("--video_path", type=str, default="/home/azureuser/video-process/video/57e54737-1c62-4605-a1d0-4de4068915d9.mp4", help="输入视频文件或文件夹路径")
    parser.add_argument("--output_dir", type=str, default="/home/azureuser/video-process/output", help="输出目录路径")
    parser.add_argument("--model", type=str, default="phi-4-multi", 
                       choices=["phi-4-multi", "gemma-3"], help="选择初始筛选模型")
    parser.add_argument("--advanced_model", type=str, default="gpt-4o", help="高级分析模型")
    parser.add_argument("--batch", action="store_true", help="批处理模式")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 自动决定处理方式：如果传入路径为目录或指定了 batch 模式，则当作批处理
    video_path = Path(args.video_path)
    if args.batch or video_path.is_dir():
        video_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(video_path) 
                      for f in filenames if f.endswith(('.mp4', '.avi', '.mkv'))]
    else:
        video_files = [str(video_path)]
    
    # 创建分析管道
    pipeline = VideoAnalysisPipeline(
        initial_model=args.model,
        advanced_model=args.advanced_model
    )
    
    # 分析视频
    for video_file in video_files:
        print(f"处理视频: {video_file}")
        event_result = pipeline.process_video(video_file, args.output_dir)
        # 输出事件分析结果
        print(f"事件结果: {event_result}")

if __name__ == "__main__":
    main()
