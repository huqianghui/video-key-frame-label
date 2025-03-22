#!/usr/bin/env python3
"""
提取视频中的所有帧保存为图片，方便手动检查
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
from tqdm import tqdm


def print_color(message, color_code="\033[0;32m"):
    """使用颜色打印消息"""
    reset_code = "\033[0m"
    print(f"{color_code}{message}{reset_code}")


def extract_all_frames(video_path, output_dir, interval=1, max_frames=None):
    """提取所有帧"""
    # 确保输出目录存在
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print_color(f"无法打开视频: {video_path}", "\033[0;31m")
        return False
        
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print_color(f"视频信息:", "\033[0;36m")
    print_color(f"  - 总帧数: {total_frames}", "\033[0;36m")
    print_color(f"  - 帧率: {fps} fps", "\033[0;36m")
    print_color(f"  - 分辨率: {width}x{height}", "\033[0;36m")
    print_color(f"  - 时长: {total_frames/fps:.2f} 秒", "\033[0;36m")
    
    # 如果设置了最大帧数，计算相应的间隔
    if max_frames and total_frames > max_frames:
        calculated_interval = max(1, int(total_frames / max_frames))
        if calculated_interval > interval:
            print_color(f"由于设置了最大帧数 {max_frames}，调整提取间隔从 {interval} 到 {calculated_interval}", "\033[0;33m")
            interval = calculated_interval
    
    # 提取帧
    frame_count = 0
    saved_count = 0
    
    print_color(f"开始提取帧，间隔: {interval}...", "\033[0;32m")
    with tqdm(total=total_frames) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % interval == 0:
                # 计算时间戳
                timestamp = frame_count / fps
                
                # 生成文件名
                filename = f"frame_{frame_count:06d}_{timestamp:.2f}s.jpg"
                file_path = output_path / filename
                
                # 保存帧
                cv2.imwrite(str(file_path), frame)
                saved_count += 1
                
            frame_count += 1
            pbar.update(1)
            
            # 如果达到最大帧数，提前结束
            if max_frames and saved_count >= max_frames:
                print_color(f"已达到设置的最大帧数 {max_frames}，停止提取", "\033[0;33m")
                break
    
    cap.release()
    print_color(f"提取完成! 共保存 {saved_count} 帧到 {output_path}", "\033[0;32m")
    return True


def main():
    parser = argparse.ArgumentParser(description="提取视频中的所有帧")
    parser.add_argument("video_path", help="输入视频的路径")
    parser.add_argument("--output", "-o", default="./extracted_frames", help="输出目录")
    parser.add_argument("--interval", "-i", type=int, default=1, help="提取间隔(每N帧提取一帧)")
    parser.add_argument("--max-frames", "-m", type=int, help="最大提取帧数")
    
    args = parser.parse_args()
    
    # 如果输出目录是相对路径，则相对于当前目录
    if not os.path.isabs(args.output):
        args.output = os.path.join(os.getcwd(), args.output)
    
    # 提取所有帧
    success = extract_all_frames(args.video_path, args.output, args.interval, args.max_frames)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
