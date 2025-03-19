import os
import cv2
import numpy as np
import ffmpeg
from pathlib import Path
from datetime import datetime
import time
from tqdm import tqdm

from config import VIDEO_CONFIG, TEMP_DIR

class VideoProcessor:
    def __init__(self):
        self.segment_length = VIDEO_CONFIG["segment_length"]
        self.frame_rate = VIDEO_CONFIG["frame_rate"]
        self.resize_dim = VIDEO_CONFIG["resize_dim"]
        
    def segment_video(self, video_path):
        """将视频分段为指定长度的片段"""
        output_dir = TEMP_DIR / "segments" / Path(video_path).stem
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 获取视频信息
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        duration = float(video_info['duration'])
        
        # 计算段数
        num_segments = int(np.ceil(duration / self.segment_length))
        segment_files = []
        
        print(f"将视频分为{num_segments}段")
        
        for i in tqdm(range(num_segments)):
            start_time = i * self.segment_length
            output_file = str(output_dir / f"segment_{i:03d}.mp4")
            
            try:
                (
                    ffmpeg
                    .input(video_path, ss=start_time, t=self.segment_length)
                    .output(output_file, c='copy')
                    .global_args('-loglevel', 'error')
                    .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
                )
                segment_files.append(output_file)
            except ffmpeg.Error as e:
                print(f"分段{i}处理失败: {e}")
        
        return segment_files
    
    def extract_frames(self, video_path):
        """从视频中提取关键帧"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频 {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps / self.frame_rate))
        
        frames = []
        frame_timestamps = []
        frame_index = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_index % frame_interval == 0:
                # 调整尺寸
                if self.resize_dim:
                    frame = cv2.resize(frame, self.resize_dim)
                
                # 获取当前帧的时间戳(秒)
                timestamp = frame_index / fps
                
                frames.append(frame)
                frame_timestamps.append(timestamp)
                
            frame_index += 1
        
        cap.release()
        
        return frames, frame_timestamps
    
    def detect_scene_changes(self, frames, threshold=30):
        """检测场景变化以识别潜在的关键帧"""
        if len(frames) < 2:
            return [0] if len(frames) > 0 else []
        
        scene_change_indices = [0]  # 始终包含第一帧
        
        # 计算连续帧之间的差异
        for i in range(1, len(frames)):
            prev_frame = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # 计算帧差异
            diff = cv2.absdiff(prev_frame, curr_frame)
            non_zero_count = np.count_nonzero(diff > 25)
            
            # 计算像素变化百分比
            change_percent = (non_zero_count * 100) / (prev_frame.shape[0] * prev_frame.shape[1])
            
            if change_percent > threshold:
                scene_change_indices.append(i)
        
        return scene_change_indices
    
    def save_frame(self, frame, output_dir, prefix, index):
        """保存帧到磁盘"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}_{index:04d}.jpg"
        output_path = output_dir / filename
        
        cv2.imwrite(str(output_path), frame)
        return str(output_path)
