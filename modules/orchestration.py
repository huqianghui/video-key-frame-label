from pathlib import Path

from config import MODEL_CONFIG, OUTPUT_DIR
from modules.event_detection import EventDetector
from modules.frame_analysis import FrameAnalyzer
from modules.video_processing import VideoProcessor


class VideoAnalysisPipeline:
    def __init__(self, initial_model="phi-4-multi", advanced_model="gpt-4o", use_yolo=True):
        self.video_processor = VideoProcessor()
        self.frame_analyzer = FrameAnalyzer(model_name=initial_model)
        
        # 设置是否使用YOLO，并获取配置的权重路径
        yolo_weights = MODEL_CONFIG["yolo_weights"] if use_yolo else None
        
        self.event_detector = EventDetector(
            model_name=advanced_model, 
            use_yolo=use_yolo,
            yolo_weights=yolo_weights
        )

    def process_video(self, video_path, output_dir):
        """处理视频并提取关键帧, 增强版本"""
        # 对视频进行分段
        segments = self.video_processor.segment_video(video_path)
        key_frames = []
        analyses = []
        timestamps = []
        
        print(f"视频分段完成，共 {len(segments)} 个片段")
        
        # 遍历每个片段
        for seg_idx, seg in enumerate(segments):
            print(f"处理片段 {seg_idx+1}/{len(segments)}: {seg}")
            # 提取帧以及对应时间戳，启用中间帧保存
            frames, frame_ts = self.video_processor.extract_frames(seg, save_intermediate=True)
            
            if not frames: 
                print(f"片段 {seg_idx+1} 未提取到帧")
                continue
                
            print(f"共提取 {len(frames)} 帧，检测场景变化...")
            
            # 保存所有候选帧，方便后续人工判断
            candidate_dir = Path(output_dir) / "candidates" / Path(seg).stem
            self.video_processor.save_all_candidate_frames(frames, frame_ts, candidate_dir)
            
            # 通过场景变化检测，选取候选帧索引，使用更低的阈值
            candidate_indices = self.video_processor.detect_scene_changes(frames, frame_ts)
            print(f"检测到 {len(candidate_indices)} 个场景变化点")
            
            # 增加最后一帧作为候选帧
            if len(frames) > 0 and (len(frames)-1) not in candidate_indices:
                candidate_indices.append(len(frames)-1)
                print(f"添加最后一帧作为候选帧，总计 {len(candidate_indices)} 个候选帧")
                
            # 如果候选帧太少，添加均匀间隔帧
            if len(candidate_indices) < 5 and len(frames) > 10:
                step = len(frames) // 5
                additional_indices = list(range(0, len(frames), step))
                for idx in additional_indices:
                    if idx not in candidate_indices:
                        candidate_indices.append(idx)
                candidate_indices.sort()  # 排序
                print(f"添加均匀间隔帧后，总计 {len(candidate_indices)} 个候选帧")
                
            # 分析所有候选帧
            for idx in candidate_indices:
                if idx >= len(frames):
                    print(f"警告：索引 {idx} 超出范围，跳过")
                    continue
                    
                frame = frames[idx]
                ts = frame_ts[idx]
                
                print(f"分析时间戳 {ts:.2f}s 的候选帧...")
                result = self.frame_analyzer.analyze_frame(frame, timestamp=ts)
                
                print(f"分析结果：置信度 {result.get('confidence', 0):.2f}, 是否为关键帧: {result.get('is_key_frame', False)}")
                
                # 降低筛选标准，保留更多帧
                if result.get("confidence", 0) >= 0.4:  # 降低置信度要求
                    key_frames.append(frame)
                    analyses.append(result)
                    timestamps.append(ts)
                    print(f"已累计 {len(key_frames)} 个关键帧")

        # 保存所有识别到的帧，无论数量如何
        if key_frames:
            saved_frames = []
            for i, (frame, ts) in enumerate(zip(key_frames, timestamps)):
                is_partial = len(key_frames) < 4
                prefix = "partial" if is_partial else "key"
                saved = self.video_processor.save_frame(
                    frame, output_dir, prefix, i, timestamp=ts
                )
                saved_frames.append(saved)
            print(f"已保存 {len(saved_frames)} 个关键帧")
            
            # 如果帧数不足4个，仍然保存现有帧并报告
            if len(key_frames) < 4:
                print(f"未检测到完整事件，关键帧数量不足(只有{len(key_frames)}个)。")
                return {
                    "event_detected": False, 
                    "reason": f"关键帧不足(只有{len(key_frames)}个，需要至少4个)", 
                    "partial_frames": len(key_frames), 
                    "saved_frames": saved_frames
                }
                
            # 否则，进行事件检测
            print(f"开始分析 {len(key_frames)} 个关键帧以检测事件...")
            event_result = self.event_detector.analyze_key_frames(key_frames, analyses, timestamps)
            event_result["saved_key_frames"] = saved_frames
            print("检测到完整事件：", event_result)
            return event_result
        else:
            print("没有找到任何关键帧")
            return {"event_detected": False, "reason": "未找到任何关键帧", "partial_frames": 0, "saved_frames": []}