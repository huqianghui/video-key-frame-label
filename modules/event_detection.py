import os
import json
import time
from pathlib import Path
import openai
from PIL import Image
import base64
import io
import cv2
import numpy as np

from config import API_KEYS, EVENT_TYPES

class EventDetector:
    def __init__(self, model_name="gpt-4o"):
        self.model = model_name
        openai.api_key = API_KEYS["openai"]
    
    def _encode_image_to_base64(self, image):
        """将图像编码为base64"""
        if isinstance(image, np.ndarray):
            # OpenCV BGR 转 RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image)
        else:
            pil_img = image
            
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def analyze_key_frames(self, frames, frame_data, timestamps):
        """分析一组关键帧以检测事件"""
        if not frames or len(frames) == 0:
            return {"event_detected": False, "reason": "无关键帧提供"}
        
        # 获取图像的base64编码
        encoded_images = [self._encode_image_to_base64(frame) for frame in frames]
        
        # 创建消息列表
        messages = [
            {
                "role": "system", 
                "content": f"""你是一个安防视频分析专家。你将分析一系列来自监控视频的关键帧，并确定它们是否构成一个完整的可疑事件。
                请特别注意以下类型的事件:
                
                1. 快递相关事件：例如有人放下快递，拍照，然后拿起快递离开（这通常是快递诈骗）
                2. 车辆安全事件：例如有人绕车检查或尝试非法进入车辆
                3. 暴力事件：园区内发生的打架、袭击等暴力行为
                
                对于检测到的事件，请提供:
                1. 事件类型
                2. 事件描述
                3. 关键动作序列
                4. 可疑程度评分(1-10)
                5. 建议采取的行动
                
                如果无法确定是否存在完整可疑事件，请说明原因。"""
            },
            {"role": "user", "content": [
                {"type": "text", "text": f"我将向你展示{len(frames)}张来自安防摄像头的连续关键帧。请分析这些图像并确定它们是否构成一个完整的可疑事件。这些图像的时间戳(秒)分别是: {', '.join([str(t) for t in timestamps])}"},
            ]}
        ]
        
        # 添加图像到用户消息
        for i, img_b64 in enumerate(encoded_images):
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}",
                    "detail": "high" if i < 3 else "low"  # 前三张用高清晰度，其余用低清晰度节省token
                }
            })
            if frame_data and i < len(frame_data):
                messages[1]["content"].append({
                    "type": "text", 
                    "text": f"图像{i+1}初步分析: {frame_data[i]['analysis']}"
                })
        
        # API请求
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=1500
            )
            analysis = response.choices[0].message.content
            
            # 处理分析结果
            event_result = self._parse_event_analysis(analysis)
            return event_result
            
        except Exception as e:
            print(f"事件检测API错误: {e}")
            return {"event_detected": False, "reason": f"API错误: {str(e)}"}
    
    def _parse_event_analysis(self, analysis):
        """解析事件分析文本"""
        # 通过关键词检测是否发现事件
        event_detected = any(phrase in analysis.lower() for phrase in [
            "可疑事件", "检测到事件", "事件类型", "可疑程度", "suspicious event", 
            "event detected", "建议采取", "recommended action"
        ])
        
        # 尝试确定事件类型
        event_type = None
        for et, keywords in EVENT_TYPES.items():
            if any(kw in analysis for kw in keywords) and "事件类型" in analysis:
                event_type = et
                break
        
        # 提取可疑程度评分
        import re
        suspicion_score = None
        score_match = re.search(r'可疑程度[：:]\s*(\d+)[/\s]*10', analysis)
        if score_match:
            try:
                suspicion_score = int(score_match.group(1))
            except:
                pass
        
        result = {
            "event_detected": event_detected,
            "event_type": event_type,
            "analysis": analysis,
            "suspicion_score": suspicion_score
        }
        
        if not event_detected:
            result["reason"] = "分析未检测到完整可疑事件"
            
        return result
