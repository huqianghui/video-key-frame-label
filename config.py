import os
from pathlib import Path

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 基本路径
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"
MODELS_DIR = BASE_DIR / "models"

# 确保目录存在
for dir_path in [DATA_DIR, OUTPUT_DIR, TEMP_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True)

# 视频处理配置
VIDEO_CONFIG = {
    "segment_length": int(os.getenv("VIDEO_SEGMENT_LENGTH", "60")),  # 视频分段长度(秒)
    "frame_rate": int(os.getenv("VIDEO_FRAME_RATE", "5")),          # 提取帧的频率(每秒)
    "resize_dim": (640, 480),                                        # 重置尺寸
    "scene_threshold": float(os.getenv("SCENE_THRESHOLD", "20")),    # 场景变化阈值(百分比)
    "fixed_interval": int(os.getenv("FIXED_INTERVAL", "15")),        # 固定间隔帧提取(每N帧)
    "enable_debug": os.getenv("ENABLE_DEBUG", "True").lower() in ["true", "1", "yes", "y"],
}

# 模型配置
MODEL_CONFIG = {
    "initial_model": "phi-4-multi",  # 或 "gemma-3"
    "advanced_model": "gpt-4o",
    "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.5")),    # 降低初步筛选的置信度阈值
    
    # YOLO配置 - 修复路径问题
    "yolo_weights": os.getenv("YOLO_WEIGHTS_PATH", str(MODELS_DIR / "yolov8s.pt")),
    "yolo_confidence": float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.4")),
    
    # Azure OpenAI配置
    "azure_openai_key": os.getenv("AZURE_OPENAI_API_KEY", ""),
    "azure_openai_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    "azure_openai_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""),
    
    # Phi-4配置
    "azure_phi4_key": os.getenv("AZURE_PHI4_API_KEY", ""),
    "azure_phi4_endpoint": os.getenv("AZURE_PHI4_ENDPOINT", ""),
    "azure_phi4_deployment": os.getenv("AZURE_PHI4_DEPLOYMENT_NAME", ""),
}

# 事件类型
EVENT_TYPES = {
    "package_delivery": ["放下物品", "拍照", "拿起物品", "离开"],
    "vehicle_inspection": ["靠近车辆", "绕车", "检查部件", "离开"],
    "violence": ["争吵姿势", "突然动作", "肢体冲突", "倒地"]
}

# API密钥配置 - 兼容性保留，建议使用MODEL_CONFIG中的配置
API_KEYS = {
    "openai": os.getenv("AZURE_OPENAI_API_KEY", ""),
    "phi-4-endpoint": os.getenv("AZURE_PHI4_ENDPOINT", ""),
    "phi-4-key": os.getenv("AZURE_PHI4_API_KEY", ""),
}
