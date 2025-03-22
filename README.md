# video-key-frame-label

## 概述
这是一个安防视频关键帧提取和分析工具，可以自动识别视频中的关键帧并进行分析标注。

## 特性
- 支持本地视频文件处理
- 使用场景变化检测自动提取关键帧
- 使用Phi-4多模态模型分析图像内容
- 使用YOLO进行物体检测
- 使用GPT-4o进行高级事件分析
- 支持Azure OpenAI和Azure AI Inference服务

## 依赖
- Python 3.8+
- FFmpeg
- PyTorch
- OpenCV
- YOLO v8
- Azure AI Inference SDK (用于Phi-4模型)
- Azure OpenAI SDK (用于GPT-4o模型)

## 安装

### 1. 克隆仓库
```bash
git clone https://github.com/yourusername/video-key-frame-label.git
cd video-key-frame-label
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 安装FFmpeg
FFmpeg是视频处理必需的工具，安装方法取决于您的操作系统：

- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- CentOS/RHEL: `sudo yum install ffmpeg`
- Windows: 从[FFmpeg官方网站](https://ffmpeg.org/download.html)下载

### 4. 配置
创建`.env`文件并添加必要的API密钥和端点：

```dotenv
# Azure OpenAI API配置 (用于GPT-4o)
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name

# Phi-4-Multimodel配置
AZURE_PHI4_API_KEY=your_phi4_api_key
AZURE_PHI4_ENDPOINT=https://your-resource-name.inference.ml.azure.com
AZURE_PHI4_DEPLOYMENT_NAME=your-deployment-name

# YOLO配置
YOLO_WEIGHTS_PATH=models/yolov8s.pt
YOLO_CONFIDENCE_THRESHOLD=0.4
```

### 5. 测试配置
```bash
python test_api.py --test-phi4  # 测试Phi-4 API
python test_api.py --test-openai  # 测试OpenAI API
python test_api.py --test-yolo  # 测试YOLO模型
python test_api.py --test-all  # 测试所有组件
```

## 使用方法

### 基本用法
```bash
python main.py --video_path /path/to/video.mp4 --output_dir ./output
```

### 高级用法
```bash
# 使用YOLO进行对象检测
python main.py --video_path /path/to/video.mp4 --use_yolo

# 批量处理目录下的所有视频
python main.py --video_path /path/to/videos_dir --batch

# 指定使用不同的初始分析模型
python main.py --video_path /path/to/video.mp4 --model phi-4-multi
```

## 注意事项
- 使用Phi-4模型需要Azure AI Inference服务，而非Azure OpenAI服务
- 确保YOLO模型权重文件存在于models目录
- 处理高分辨率视频时可能需要较大内存