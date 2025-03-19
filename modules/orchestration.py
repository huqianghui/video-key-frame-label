from pathlib import Path
from modules.video_processing import VideoProcessor
from modules.frame_analysis import FrameAnalyzer
from modules.event_detection import EventDetector

class VideoAnalysisPipeline:
    def __init__(self, initial_model="phi-4-multi", advanced_model="gpt-4o"):
        self.video_processor = VideoProcessor()
        self.frame_analyzer = FrameAnalyzer(model_name=initial_model)
        self.event_detector = EventDetector(model_name=advanced_model)

    def process_video(self, video_path, output_dir):
        # 对视频进行分段
        segments = self.video_processor.segment_video(video_path)
        key_frames = []
        analyses = []
        timestamps = []
        
        # 遍历每个片段
        for seg in segments:
            # 提取帧以及对应时间戳
            frames, frame_ts = self.video_processor.extract_frames(seg)
            if not frames: 
                continue
            # 通过场景变化检测，选取候选帧索引
            candidate_indices = self.video_processor.detect_scene_changes(frames)
            for idx in candidate_indices:
                frame = frames[idx]
                ts = frame_ts[idx]
                result = self.frame_analyzer.analyze_frame(frame, timestamp=ts)
                if result.get("is_key_frame"):
                    key_frames.append(frame)
                    analyses.append(result)
                    timestamps.append(ts)

        # 当累计关键帧数量>=4则进行事件检测
        if len(key_frames) >= 4:
            event_result = self.event_detector.analyze_key_frames(key_frames, analyses, timestamps)
            # 保存关键帧到指定输出目录
            saved_frames = []
            for i, frame in enumerate(key_frames):
                saved = self.video_processor.save_frame(frame, output_dir, "key", i)
                saved_frames.append(saved)
            event_result["saved_key_frames"] = saved_frames
            print("检测到完整事件：", event_result)
            return event_result
        else:
            print("未检测到完整事件，关键帧数量不足。")
            return {"event_detected": False, "reason": "关键帧不足"}