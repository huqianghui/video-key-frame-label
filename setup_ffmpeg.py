#!/usr/bin/env python3
"""
自动安装ffmpeg的辅助脚本
支持Ubuntu/Debian/CentOS/RHEL
"""
import os
import platform
import subprocess
import sys


def is_root():
    """检查当前用户是否为root"""
    return os.geteuid() == 0 if hasattr(os, "geteuid") else False

def print_color(message, color_code="\033[0;32m"):
    """使用颜色打印消息"""
    reset_code = "\033[0m"
    print(f"{color_code}{message}{reset_code}")

def check_ffmpeg():
    """检查ffmpeg是否已安装"""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def install_ffmpeg_ubuntu():
    """在Ubuntu/Debian上安装ffmpeg"""
    cmd_prefix = "" if is_root() else "sudo "
    commands = [
        f"{cmd_prefix}apt-get update",
        f"{cmd_prefix}apt-get install -y ffmpeg"
    ]
    
    for cmd in commands:
        print_color(f"执行: {cmd}")
        result = os.system(cmd)
        if result != 0:
            print_color(f"命令执行失败: {cmd}", "\033[0;31m")
            return False
    return True

def install_ffmpeg_centos():
    """在CentOS/RHEL上安装ffmpeg"""
    cmd_prefix = "" if is_root() else "sudo "
    
    # 添加EPEL仓库
    commands = [
        f"{cmd_prefix}yum install -y epel-release",
        f"{cmd_prefix}yum install -y ffmpeg ffmpeg-devel"
    ]
    
    for cmd in commands:
        print_color(f"执行: {cmd}")
        result = os.system(cmd)
        if result != 0:
            print_color(f"命令执行失败: {cmd}", "\033[0;31m")
            return False
    return True

def main():
    """主函数"""
    if check_ffmpeg():
        print_color("ffmpeg已安装")
        return 0
    
    print_color("ffmpeg未安装，尝试自动安装...", "\033[1;33m")
    
    # 检测操作系统类型
    if platform.system() != "Linux":
        print_color("当前仅支持在Linux系统上自动安装ffmpeg", "\033[0;31m")
        print_color("请手动安装ffmpeg: https://ffmpeg.org/download.html", "\033[1;33m")
        return 1
    
    # 检测Linux发行版
    if os.path.exists("/etc/debian_version") or os.path.exists("/etc/ubuntu_version"):
        success = install_ffmpeg_ubuntu()
    elif os.path.exists("/etc/redhat-release") or os.path.exists("/etc/centos-release"):
        success = install_ffmpeg_centos()
    else:
        print_color("无法确定Linux发行版类型，请手动安装ffmpeg", "\033[0;31m")
        print_color("Ubuntu/Debian: sudo apt-get install ffmpeg", "\033[1;33m")
        print_color("CentOS/RHEL: sudo yum install ffmpeg", "\033[1;33m")
        success = False
    
    if success and check_ffmpeg():
        print_color("ffmpeg安装成功！", "\033[0;32m")
        return 0
    else:
        print_color("ffmpeg安装失败或验证失败", "\033[0;31m")
        return 1

if __name__ == "__main__":
    sys.exit(main())
