import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import ffmpeg
import numpy as np
from tqdm import tqdm

from config import TEMP_DIR, VIDEO_CONFIG


class VideoProcessor:
    def __init__(self):
        self.segment_length = VIDEO_CONFIG["segment_length"]
        self.frame_rate = VIDEO_CONFIG["frame_rate"]
        self.resize_dim = VIDEO_CONFIG["resize_dim"]
        self.enable_debug = VIDEO_CONFIG.get("enable_debug", False)
        self.scene_threshold = VIDEO_CONFIG.get("scene_threshold", 30)  # 新增场景变化阈值设置
        self.fixed_interval = VIDEO_CONFIG.get("fixed_interval", 10)  # 新增固定间隔帧提取
        self._check_ffmpeg()
        
    def _check_ffmpeg(self):
        """检查ffmpeg是否安装"""
        try:
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(['ffprobe', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("ffmpeg和ffprobe已安装")
        except FileNotFoundError:
            print("警告: ffmpeg或ffprobe未安装，视频分段功能将不可用")
            print("请安装ffmpeg: apt-get install ffmpeg")
            
    def segment_video(self, video_path):
        """将视频分段为指定长度的片段"""
        output_dir = TEMP_DIR / "segments" / Path(video_path).stem
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 检查ffprobe是否可用
        try:
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
        except (ffmpeg.Error, FileNotFoundError) as e:
            print(f"ffmpeg处理失败: {e}")
            print("直接处理完整视频文件...")
            # 如果ffmpeg不可用，返回原始视频
            return [video_path]
        except Exception as e:
            print(f"视频分段错误: {e}")
            return [video_path]
            
        if not segment_files:
            print("分段失败，使用原始视频")
            return [video_path]
            
        return segment_files
    
    def extract_frames(self, video_path, save_intermediate=False):
        """从视频中提取关键帧，增强版本"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频 {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps / self.frame_rate))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        frame_timestamps = []
        frame_index = 0
        
        # 保存中间帧的目录
        if save_intermediate:
            video_name = Path(video_path).stem
            intermediate_dir = TEMP_DIR / "intermediate" / video_name
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            print(f"中间帧将保存到: {intermediate_dir}")
        
        print(f"视频FPS: {fps}, 提取间隔: {frame_interval}, 总帧数: {total_frames}")
        
        # 读取所有I帧和固定间隔帧
        all_frames = []
        all_timestamps = []
        
        # 首先尝试使用ffmpeg提取I帧
        try:
            i_frames = self._extract_i_frames(video_path)
            if i_frames:
                print(f"成功提取 {len(i_frames)} 个I帧")
                for i, (i_frame, ts) in enumerate(i_frames):
                    all_frames.append(i_frame)
                    all_timestamps.append(ts)
                    if save_intermediate:
                        i_frame_path = str(intermediate_dir / f"i_frame_{i:04d}_{ts:.2f}s.jpg")
                        cv2.imwrite(i_frame_path, i_frame)
        except Exception as e:
            print(f"提取I帧失败，继续使用常规方法: {e}")
        
        # 重置视频读取
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_index = 0
        
        # 逐帧读取
        print("开始提取所有采样帧...")
        with tqdm(total=total_frames) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 保存固定间隔帧和其他关键帧
                is_sample_frame = frame_index % frame_interval == 0
                is_fixed_interval = frame_index % self.fixed_interval == 0
                
                if is_sample_frame or is_fixed_interval:
                    # 调整尺寸
                    if self.resize_dim:
                        frame = cv2.resize(frame, self.resize_dim)
                    
                    # 获取当前帧的时间戳(秒)
                    timestamp = frame_index / fps
                    
                    # 添加到总帧列表
                    frames.append(frame)
                    frame_timestamps.append(timestamp)
                    
                    # 保存中间帧
                    if save_intermediate:
                        frame_type = "sample" if is_sample_frame else "fixed"
                        frame_path = str(intermediate_dir / f"{frame_type}_{frame_index:06d}_{timestamp:.2f}s.jpg")
                        cv2.imwrite(frame_path, frame)
                
                frame_index += 1
                pbar.update(1)
        
        cap.release()
        
        print(f"总共提取 {len(frames)} 帧进行分析")
        return frames, frame_timestamps
    
    def _extract_i_frames(self, video_path):
        """尝试使用ffmpeg提取I帧"""
        print("尝试提取I帧...")
        i_frames = []
        
        try:
            # 创建临时目录存储I帧
            temp_i_frames_dir = TEMP_DIR / "i_frames" / Path(video_path).stem
            temp_i_frames_dir.mkdir(parents=True, exist_ok=True)
            
            # 使用ffmpeg提取I帧
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', 'select=eq(pict_type\,I)',
                '-vsync', '0',
                '-f', 'image2',
                f'{temp_i_frames_dir}/i_frame_%04d.jpg'
            ]
            
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if result.returncode == 0:
                # 获取视频基本信息
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                
                # 读取生成的I帧图像
                i_frame_files = sorted(list(temp_i_frames_dir.glob('i_frame_*.jpg')))
                
                for i, file_path in enumerate(i_frame_files):
                    frame = cv2.imread(str(file_path))
                    if frame is not None:
                        # 估计时间戳 - 粗略估计
                        timestamp = i * (1.0 / fps) * 24  # 假设I帧间隔约为24帧
                        i_frames.append((frame, timestamp))
                
                return i_frames
        except Exception as e:
            print(f"I帧提取错误: {e}")
        
        return []
    
    def detect_scene_changes(self, frames, timestamps, threshold=None):
        """检测场景变化以识别潜在的关键帧，使用更敏感的设置"""
        if len(frames) < 2:
            return [0] if len(frames) > 0 else []
        
        if threshold is None:
            threshold = self.scene_threshold  # 使用配置的阈值，默认是30
        
        scene_change_indices = [0]  # 始终包含第一帧
        frame_diffs = []  # 保存帧差异用于调试
        
        print(f"使用场景变化阈值: {threshold}")
        
        # 计算连续帧之间的差异
        for i in range(1, len(frames)):
            prev_frame = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # 计算帧差异
            diff = cv2.absdiff(prev_frame, curr_frame)
            non_zero_count = np.count_nonzero(diff > 25)
            
            # 计算像素变化百分比
            change_percent = (non_zero_count * 100) / (prev_frame.shape[0] * prev_frame.shape[1])
            frame_diffs.append(change_percent)
            
            if change_percent > threshold:
                scene_change_indices.append(i)
                print(f"检测到场景变化: 索引={i}, 时间戳={timestamps[i]:.2f}s, 变化率={change_percent:.2f}%")
        
        # 如果场景变化太少，降低阈值再次尝试
        if len(scene_change_indices) < 3 and threshold > 15:
            lower_threshold = threshold * 0.7  # 降低30%
            print(f"场景变化太少，降低阈值到 {lower_threshold} 重新检测...")
            return self.detect_scene_changes(frames, timestamps, lower_threshold)
        
        # 如果启用调试，保存差异分析
        if self.enable_debug:
            # 创建一个简单的图表展示帧差异
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.plot(range(1, len(frames)), frame_diffs)
            plt.axhline(y=threshold, color='r', linestyle='-', label=f'阈值 ({threshold})')
            plt.xlabel('帧索引')
            plt.ylabel('变化百分比')
            plt.title('帧差异分析')
            plt.grid(True)
            
            # 标记检测到的场景变化
            for idx in scene_change_indices[1:]:  # 跳过第一帧
                plt.axvline(x=idx, color='g', linestyle='--')
            
            # 保存图表
            debug_dir = TEMP_DIR / "debug"
            debug_dir.mkdir(exist_ok=True)
            plt.savefig(str(debug_dir / f"frame_diff_analysis_{int(time.time())}.png"))
            plt.close()
        
        print(f"检测到 {len(scene_change_indices)} 个场景变化点")
        return scene_change_indices
    
    def save_frame(self, frame, output_dir, prefix, index, timestamp=None):
        """保存帧到磁盘，增加时间戳信息"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        ts_info = f"_{timestamp:.2f}s" if timestamp is not None else ""
        filename = f"{prefix}_{time_str}_{index:04d}{ts_info}.jpg"
        output_path = output_dir / filename
        
        cv2.imwrite(str(output_path), frame)
        return str(output_path)
    
    def save_all_candidate_frames(self, frames, timestamps, output_dir):
        """保存所有候选帧，用于手动筛选"""
        candidate_dir = Path(output_dir) / "candidates"
        candidate_dir.mkdir(exist_ok=True, parents=True)
        
        saved_paths = []
        for i, (frame, ts) in enumerate(zip(frames, timestamps)):
            path = self.save_frame(frame, candidate_dir, "candidate", i, ts)
            saved_paths.append(path)
            
        print(f"已保存 {len(saved_paths)} 个候选帧到 {candidate_dir}")
        return saved_paths
