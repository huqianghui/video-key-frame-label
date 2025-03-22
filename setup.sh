#!/bin/bash

# 颜色定义
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

echo -e "${GREEN}开始设置视频关键帧分析环境...${NC}"

# 检查Python版本
python_version=$(python3 --version 2>&1)
if [[ $python_version == *"Python 3"* ]]; then
  echo -e "${GREEN}检测到Python: $python_version${NC}"
else
  echo -e "${RED}未检测到Python 3，请安装Python 3.8或更高版本${NC}"
  exit 1
fi

# 检查ffmpeg
if command -v ffmpeg &> /dev/null && command -v ffprobe &> /dev/null; then
  echo -e "${GREEN}检测到ffmpeg已安装${NC}"
else
  echo -e "${YELLOW}未检测到ffmpeg，尝试安装...${NC}"
  if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y ffmpeg
  elif command -v yum &> /dev/null; then
    sudo yum install -y ffmpeg
  else
    echo -e "${RED}无法自动安装ffmpeg，请手动安装${NC}"
    echo -e "${YELLOW}Ubuntu/Debian: sudo apt-get install ffmpeg${NC}"
    echo -e "${YELLOW}CentOS/RHEL: sudo yum install ffmpeg${NC}"
    echo -e "${YELLOW}或访问 https://ffmpeg.org/download.html${NC}"
  fi
fi

# 创建.env文件
if [ ! -f .env ]; then
  echo -e "${YELLOW}创建.env文件...${NC}"
  cp .env.example .env
  echo -e "${YELLOW}请编辑.env文件，填入您的Azure OpenAI和Phi-4-multi-model配置${NC}"
fi

# 创建目录结构
echo -e "${GREEN}创建项目目录...${NC}"
mkdir -p data output temp models

# 安装依赖
echo -e "${GREEN}安装Python依赖...${NC}"
pip install -r requirements.txt

# 下载YOLO模型（如果不存在）
if [ ! -f models/yolov8s.pt ]; then
  echo -e "${YELLOW}下载YOLO模型...${NC}"
  python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')" || echo -e "${RED}YOLO模型下载失败，请手动下载${NC}"
  
  # 检查当前目录下是否有模型文件，并移动到models目录
  if [ -f yolov8s.pt ]; then
    mv yolov8s.pt models/
    echo -e "${GREEN}YOLO模型已移动到models目录${NC}"
  else
    echo -e "${YELLOW}请将YOLO模型放置在models/目录下${NC}"
  fi
fi

# 运行依赖检查
echo -e "${GREEN}检查系统依赖...${NC}"
python check_deps.py

echo -e "${GREEN}设置完成!${NC}"
echo -e "${YELLOW}请确保.env文件中配置了正确的API密钥和端点${NC}"
echo -e "${GREEN}使用示例:${NC}"
echo "python main.py --video_path /path/to/video.mp4 --output_dir ./output --use_yolo"
