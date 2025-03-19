import os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import cv2
import numpy as np
from config import MODEL_CONFIG

class FrameAnalyzer:
    def __init__(self, model_name="phi-4-multi"):
        self.model_name = model_name
        self.confidence_threshold = MODEL_CONFIG["confidence_threshold"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"初始化框架分析器 使用 {model_name} 模型，设备: {self.device}")
        self._init_model()
        
    def _init_model(self):
        """初始化模型"""
        if self.model_name == "phi-4-multi":
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-multimodal-vision", trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-4-multimodal-vision", 
                torch_dtype="auto", 
                device_map=self.device,
                trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained("microsoft/Phi-4-multimodal-vision", trust_remote_code=True)
        elif self.model_name == "gemma-3":
            self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-2b-vision-preview")
            self.model = AutoModelForCausalLM.from_pretrained(
                "google/gemma-3-2b-vision-preview",
                torch_dtype="auto",
                device_map=self.device
            )
            self.processor = AutoProcessor.from_pretrained("google/gemma-3-2b-vision-preview")
        else:
            raise ValueError(f"不支持的模型: {self.model_name}")
    
    def analyze_frame(self, frame, timestamp=None):
        """分析单个帧"""
        # 转换OpenCV帧到PIL图像
        if isinstance(frame, np.ndarray):
            # BGR转RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
        else:
            image = frame
        
        # 构建分析提示
        prompt = """
        这是一个安防摄像头图像。请分析这个图像中是否存在以下可疑活动：
        1. 快递相关：放下物品、拍照、拿起物品、离开
        2. 车辆相关：绕车行走、检查车辆
        3. 可能的暴力行为
        
        如果存在这些活动，请提供详细描述并给出置信度评分（0-1）。
        如果没有可疑活动，请说明。
        """
        
        # 处理输入
        if self.model_name == "phi-4-multi":
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        else:  # gemma-3
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        # 生成分析
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512)
            analysis = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 从分析中提取置信度 (简单实现，实际需要更复杂的解析)
        confidence = self._extract_confidence(analysis)
        
        result = {
            "timestamp": timestamp,
            "analysis": analysis,
            "confidence": confidence,
            "is_key_frame": confidence >= self.confidence_threshold
        }
        
        return result
    
    def _extract_confidence(self, analysis_text):
        """从分析文本中提取置信度"""
        # 简单实现，在实际系统中应改进
        confidence_markers = ["置信度：", "置信度:", "confidence:", "confidence：", "评分：", "评分:"]
        
        for marker in confidence_markers:
            if marker in analysis_text:
                try:
                    start_idx = analysis_text.find(marker) + len(marker)
                    end_idx = start_idx + 10  # 假设数字在后面10个字符内
                    substr = analysis_text[start_idx:end_idx]
                    # 尝试找到一个浮点数
                    import re
                    match = re.search(r'(\d+\.?\d*)', substr)
                    if match:
                        return float(match.group(1))
                except:
                    pass
                
        # 如果找不到明确的置信度，基于文本内容评估
        if any(keyword in analysis_text.lower() for keyword in 
              ["可疑", "suspicious", "危险", "danger", "警告", "warning"]):
            return 0.7
        elif any(keyword in analysis_text.lower() for keyword in 
               ["注意", "attention", "检查", "check"]):
            return 0.5
            
        return 0.3  # 默认低置信度
