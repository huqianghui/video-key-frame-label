import base64
import io
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

from config import API_KEYS, EVENT_TYPES, MODEL_CONFIG


class EventDetector:
    def __init__(self, model_name="gpt-4o", use_yolo=False, yolo_weights=None, yolo_config=None):
        self.model = model_name
        
        # Azure OpenAI配置
        self.azure_endpoint = MODEL_CONFIG["azure_openai_endpoint"]
        self.azure_key = MODEL_CONFIG["azure_openai_key"]
        self.azure_deployment = MODEL_CONFIG["azure_openai_deployment"]
        
        # YOLO配置
        self.use_yolo = use_yolo
        if yolo_weights is None:
            yolo_weights = MODEL_CONFIG["yolo_weights"]
        
        # 初始化YOLO模型（如果启用）
        if use_yolo:
            self._init_yolo_model(yolo_weights)
    
    def _init_yolo_model(self, weights_path):
        """初始化YOLO模型"""
        try:
            # 导入工具模块
            import torch  # 添加缺失的torch导入
            import torch.serialization

            from modules.detect_utils import safe_load_yolo

            # 如果权重路径不是绝对路径，假设它是相对于项目根目录的路径
            if not os.path.isabs(weights_path):
                weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), weights_path)
            
            # 确保models目录存在
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
            os.makedirs(models_dir, exist_ok=True)
            
            # 使用安全加载函数
            if hasattr(torch.serialization, 'add_safe_globals'):
                torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
            
            self.yolo_model = safe_load_yolo(weights_path, conf=MODEL_CONFIG["yolo_confidence"])
            print("YOLO模型加载成功")
        except Exception as e:
            print(f"YOLO模型加载错误: {e}")
            import traceback
            traceback.print_exc()
            self.use_yolo = False
            print("回退至基于LLM的检测")
    
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
    
    def detect_with_yolo(self, frames):
        """使用YOLO模型检测帧中的对象"""
        if not self.use_yolo or not hasattr(self, 'yolo_model'):
            return None
            
        all_detections = []
        
        try:
            # 使用更健壮的方式进行对象检测
            for frame in frames:
                try:
                    # 运行YOLO检测
                    results = self.yolo_model(frame)
                    
                    # 提取检测结果
                    detections = []
                    for result in results:
                        # 检查结果是否有效
                        if not hasattr(result, 'boxes'):
                            continue
                            
                        boxes = result.boxes
                        for box in boxes:
                            try:
                                # 获取坐标
                                if hasattr(box, 'xyxy') and len(box.xyxy) > 0:
                                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                                    # 获取置信度
                                    confidence = box.conf[0].item() if hasattr(box, 'conf') and len(box.conf) > 0 else 0.5
                                    # 获取类别
                                    class_id = int(box.cls[0].item()) if hasattr(box, 'cls') and len(box.cls) > 0 else 0
                                    class_name = result.names[class_id] if hasattr(result, 'names') and class_id in result.names else 'unknown'
                                    
                                    detections.append({
                                        "class": class_name,
                                        "confidence": confidence,
                                        "bbox": [int(x1), int(y1), int(x2), int(y2)]
                                    })
                            except (AttributeError, IndexError, TypeError) as e:
                                print(f"处理检测框时出错: {e}")
                                continue
                    
                    all_detections.append(detections)
                except Exception as e:
                    print(f"单帧检测错误: {e}")
                    all_detections.append([])  # 添加空检测结果，保持索引一致
                    
        except Exception as e:
            print(f"YOLO检测错误: {e}")
            return None
            
        return all_detections

    def analyze_yolo_detections(self, detections, timestamps):       
        """分析YOLO检测结果以识别事件"""
        if not detections:
            return {"event_detected": False, "reason": "无YOLO检测结果"}
        
        # 计算各类物体在时间序列中的变化
        person_count = [len([d for d in frame_det if d["class"] == "person"]) for frame_det in detections]
        vehicle_count = [len([d for d in frame_det if d["class"] in ["car", "truck", "bus", "motorcycle"]]) for frame_det in detections]
        
        # 检测是否有人接近车辆
        vehicle_person_proximity = []
        for frame_det in detections:
            persons = [d for d in frame_det if d["class"] == "person"]
            vehicles = [d for d in frame_det if d["class"] in ["car", "truck", "bus", "motorcycle"]]
            
            close_interactions = False
            for person in persons:
                p_box = person["bbox"]
                p_center = [(p_box[0] + p_box[2])/2, (p_box[1] + p_box[3])/2]
                
                for vehicle in vehicles:
                    v_box = vehicle["bbox"]
                    # 检查人是否在车辆附近
                    if (abs(p_center[0] - (v_box[0] + v_box[2])/2) < 100 and 
                        abs(p_center[1] - (v_box[1] + v_box[3])/2) < 100):
                        close_interactions = True
                        break
            vehicle_person_proximity.append(close_interactions)
        
        # 基于检测结果判断事件类型
        event_detected = False
        event_type = None
        reason = "未检测到可疑事件模式"
        suspicion_score = 0
        
        # 检查人员在车辆周围活动模式
        if any(vehicle_person_proximity) and max(vehicle_count) > 0:
            event_detected = True
            event_type = "车辆安全事件"
            reason = "检测到人员在车辆周围活动"
            suspicion_score = 7
        
        # 检查人员快速增减变化，可能是递送/拿走物品
        if len(person_count) > 2:
            person_changes = [abs(person_count[i] - person_count[i-1]) for i in range(1, len(person_count))]
            if sum(person_changes) > 2:
                event_detected = True
                event_type = "快递相关事件"
                reason = "检测到人员数量变化，可能涉及递送/拿走物品"
                suspicion_score = 6
        
        # 简单的分析结果
        analysis = f"""
根据YOLO对象检测分析:
- 时间段: {min(timestamps)}-{max(timestamps)}秒
- 最大人员数量: {max(person_count)}
- 最大车辆数量: {max(vehicle_count)}
- 检测到人车交互: {"是" if any(vehicle_person_proximity) else "否"}
- 事件类型: {event_type if event_type else "未确定"}
- 可疑程度: {suspicion_score}/10
- 原因: {reason}
        """
        
        return {
            "event_detected": event_detected,
            "event_type": event_type,
            "analysis": analysis,
            "suspicion_score": suspicion_score,
            "reason": reason if not event_detected else None
        }
    
    def analyze_key_frames(self, frames, frame_data, timestamps, force_llm=False):
        """分析一组关键帧以检测事件"""
        if not frames or len(frames) == 0:
            return {"event_detected": False, "reason": "无关键帧提供"}
        
        # 如果启用了YOLO且不强制使用LLM，则使用YOLO进行检测
        if self.use_yolo and not force_llm:
            detections = self.detect_with_yolo(frames)
            result = self.analyze_yolo_detections(detections, timestamps)
            
            # 如果YOLO检测到事件或检测失败，则返回结果
            if result["event_detected"] or "Error" in result.get("reason", ""):
                return result
            
            # 如果YOLO未检测到事件但可疑度达到阈值，使用LLM进行二次确认
            if result.get("suspicion_score", 0) >= 4:
                llm_result = self._analyze_with_llm(frames, frame_data, timestamps)
                
                # 综合两种检测结果
                if llm_result["event_detected"]:
                    return llm_result
                else:
                    return result
            
            return result
        
        # 使用LLM进行分析
        return self._analyze_with_llm(frames, frame_data, timestamps)
    
    def _analyze_with_llm(self, frames, frame_data, timestamps):
        """使用LLM进行图像分析"""
        # 获取图像的base64编码
        encoded_images = [self._encode_image_to_base64(frame) for frame in frames]
        
        # 系统提示词
        system_prompt = """你是一个安防视频分析专家。你将分析一系列来自监控视频的关键帧，并确定它们是否构成一个完整的可疑事件。
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
        
        # 用户提示词
        user_prompt = f"我将向你展示{len(frames)}张来自安防摄像头的连续关键帧。请分析这些图像并确定它们是否构成一个完整的可疑事件。这些图像的时间戳(秒)分别是: {', '.join([str(t) for t in timestamps])}"
        
        # 使用Azure OpenAI API进行分析
        if self.azure_endpoint and self.azure_key:
            try:
                # 构建payload
                payload = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": user_prompt}
                        ]}
                    ],
                    "max_tokens": 1500,
                    "temperature": 0.7
                }
                
                # 添加图像到用户消息
                for i, img_b64 in enumerate(encoded_images):
                    payload["messages"][1]["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                            "detail": "high" if i < 3 else "low"  # 前三张用高清晰度，其余用低清晰度节省token
                        }
                    })
                    if frame_data and i < len(frame_data):
                        payload["messages"][1]["content"].append({
                            "type": "text", 
                            "text": f"图像{i+1}初步分析: {frame_data[i]['analysis']}"
                        })
                
                # 准备请求
                headers = {
                    "Content-Type": "application/json",
                    "api-key": self.azure_key
                }
                
                # 完整的端点URL
                endpoint_url = f"{self.azure_endpoint}/openai/deployments/{self.azure_deployment}/chat/completions?api-version=2023-07-01-preview"
                
                # 发送请求
                response = requests.post(endpoint_url, headers=headers, json=payload)
                
                # 检查响应
                if response.status_code == 200:
                    result = response.json()
                    analysis = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    if not analysis:
                        raise Exception("API返回了空的分析结果")
                    
                    # 处理分析结果
                    event_result = self._parse_event_analysis(analysis)
                    return event_result
                else:
                    raise Exception(f"API请求失败: {response.status_code}, {response.text}")
            except Exception as e:
                print(f"Azure OpenAI API错误: {e}")
                return {"event_detected": False, "reason": f"API错误: {str(e)}"}
        
        # 如果Azure配置不可用，尝试使用原来的方法
        try:
            import openai
            openai.api_key = API_KEYS["openai"]
            
            # 创建消息列表
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                ]}
            ]
            
            # 添加图像到用户消息
            for i, img_b64 in enumerate(encoded_images):
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}",
                        "detail": "high" if i < 3 else "low"
                    }
                })
                if frame_data and i < len(frame_data):
                    messages[1]["content"].append({
                        "type": "text", 
                        "text": f"图像{i+1}初步分析: {frame_data[i]['analysis']}"
                    })
            
            # API请求
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
            print(f"OpenAI API错误: {e}")
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
