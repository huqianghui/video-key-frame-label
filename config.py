import os
from pathlib import Path

# 基本路径
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"

# 确保目录存在
for dir_path in [DATA_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

# 视频处理配置
VIDEO_CONFIG = {
    "segment_length": 60,  # 视频分段长度(秒)
    "frame_rate": 5,       # 提取帧的频率(每秒)
    "resize_dim": (640, 480),  # 重置尺寸
}

# 模型配置
MODEL_CONFIG = {
    "initial_model": "phi-4-multi",  # 或 "gemma-3"
    "advanced_model": "gpt-4o",
    "confidence_threshold": 0.65,    # 初步筛选的置信度阈值
}

# 事件类型
EVENT_TYPES = {
    "package_delivery": ["放下物品", "拍照", "拿起物品", "离开"],
    "vehicle_inspection": ["靠近车辆", "绕车", "检查部件", "离开"],
    "violence": ["争吵姿势", "突然动作", "肢体冲突", "倒地"]
}

# API密钥配置
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY", ""),
}
