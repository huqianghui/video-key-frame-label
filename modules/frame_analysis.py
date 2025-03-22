import base64
import io
import json
import os

import cv2
import numpy as np
import requests
import torch
from PIL import Image

from config import MODEL_CONFIG


class FrameAnalyzer:
    def __init__(self, model_name="phi-4-multi"):
        self.model_name = model_name
        self.confidence_threshold = MODEL_CONFIG["confidence_threshold"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载Azure端点配置
        self.phi4_endpoint = MODEL_CONFIG["azure_phi4_endpoint"]
        self.phi4_key = MODEL_CONFIG["azure_phi4_key"]
        self.phi4_deployment = MODEL_CONFIG["azure_phi4_deployment"]
        
        print(f"初始化框架分析器 使用 {model_name} 模型，设备: {self.device}")
        
        # 如果是本地模型，则初始化
        if model_name == "phi-4-multi" and not self.phi4_endpoint:
            self._init_local_model()
    
    def _init_local_model(self):
        """初始化本地模型（备用方案）"""
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-multimodal-vision", trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-4-multimodal-vision", 
                torch_dtype="auto", 
                device_map=self.device,
                trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained("microsoft/Phi-4-multimodal-vision", trust_remote_code=True)
            print("成功加载本地Phi-4模型")
        except Exception as e:
            print(f"本地模型加载失败: {e}")
            print("警告: 本地模型加载失败，且未配置Azure端点。分析功能可能无法正常工作。")
    
    def _encode_image_to_base64(self, image):
        """将图像编码为base64"""
        if isinstance(image, np.ndarray):
            # OpenCV BGR转RGB
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
        else:
            pil_img = image
            
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def analyze_frame(self, frame, timestamp=None):
        """分析单个帧"""
        # 构建分析提示
        prompt = """
        这是一个安防摄像头图像。请分析这个图像中是否存在以下可疑活动：
        1. 快递相关：放下物品、拍照、拿起物品、离开
        2. 车辆相关：绕车行走、检查车辆
        3. 可能的暴力行为
        
        如果存在这些活动，请提供详细描述并给出置信度评分（0-1）。
        如果没有可疑活动，请说明。
        """
        
        # 将图像转换为base64编码
        image_base64 = self._encode_image_to_base64(frame)
        
        if self.model_name == "phi-4-multi" and self.phi4_endpoint:
            # 使用Azure Phi-4模型API进行分析
            analysis = self._analyze_with_azure_phi4(prompt, image_base64)
        else:
            # 尝试使用本地模型
            analysis = self._analyze_with_local_model(frame, prompt)
        
        # 从分析中提取置信度
        confidence = self._extract_confidence(analysis)
        
        result = {
            "timestamp": timestamp,
            "analysis": analysis,
            "confidence": confidence,
            "is_key_frame": confidence >= self.confidence_threshold
        }
        
        return result
    
    def _analyze_with_azure_phi4(self, prompt, image_base64):
        """使用Azure Phi-4模型进行分析 - 使用azure-ai-inference库"""
        try:
            # 检查配置
            if not self.phi4_endpoint or not self.phi4_key:
                return "Phi-4 API 配置缺失，请设置 AZURE_PHI4_ENDPOINT 和 AZURE_PHI4_API_KEY"
            
            # 导入 azure-ai-inference 库 - 修正导入
            from azure.ai.inference import ChatCompletionsClient
            from azure.ai.inference.models import (
                ImageContentItem,
                ImageDetailLevel,
                ImageUrl,
                SystemMessage,
                TextContentItem,
                UserMessage,
            )
            from azure.core.credentials import AzureKeyCredential
            
            print(f"使用 Phi-4 API 端点: {self.phi4_endpoint}")
            
            # 创建客户端 - 使用正确的客户端类
            client = ChatCompletionsClient(
                endpoint=self.phi4_endpoint,
                credential=AzureKeyCredential(self.phi4_key)
            )
            
            # 模型名称，默认使用 Phi-4-multimodal-instruct 或者使用部署名称
            model_name = self.phi4_deployment or "Phi-4-multimodal-instruct"
            
            # 创建临时文件保存base64图像
            temp_img_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp", "temp_image.jpg")
            os.makedirs(os.path.dirname(temp_img_path), exist_ok=True)
            
            # 将base64转为文件
            with open(temp_img_path, "wb") as image_file:
                image_file.write(base64.b64decode(image_base64))
            
            # 发送请求
            print(f"使用模型: {model_name}")
            response = client.complete(
                messages=[
                    SystemMessage(content="你是一个安防视频分析专家，擅长分析监控图像中的可疑活动。"),
                    UserMessage(
                        content=[
                            TextContentItem(text=prompt),
                            ImageContentItem(
                                image_url=ImageUrl.load(
                                    image_file=temp_img_path,
                                    image_format="jpeg",
                                    detail=ImageDetailLevel.HIGH,
                                ),
                            ),
                        ],
                    ),
                ],
                model=model_name,
                max_tokens=800,
                temperature=0.7
            )
            
            # 删除临时文件
            try:
                os.remove(temp_img_path)
            except:
                pass
                
            # 获取响应内容
            if response and response.choices and len(response.choices) > 0:
                analysis = response.choices[0].message.content
                return analysis
            else:
                return "无法获取有效的分析结果"
                
        except ImportError as e:
            print(f"导入 azure-ai-inference 错误: {e}")
            print("请安装 azure-ai-inference: pip install azure-ai-inference")
            return f"缺少必要的库: {str(e)}, 请安装 azure-ai-inference"
        except Exception as e:
            print(f"Azure Phi-4分析错误: {e}")
            return f"分析过程中发生错误: {str(e)}"
    
    def _analyze_with_local_model(self, frame, prompt):
        """使用本地模型进行分析（如果可用）"""
        try:
            if not hasattr(self, 'model') or not hasattr(self, 'processor'):
                return "本地模型未初始化或不可用"
                
            # 转换OpenCV帧到PIL图像
            if isinstance(frame, np.ndarray):
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_frame)
            else:
                image = frame
                
            # 处理输入
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            
            # 生成分析
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=512)
                analysis = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
            return analysis
            
        except Exception as e:
            print(f"本地模型分析错误: {e}")
            return f"本地模型分析错误: {str(e)}"
    
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
