# -*- coding: utf-8 -*-
"""
RAW视频文件转换为PNG图像序列工具

该脚本遍历 Data_500 目录下的所有 .raw 文件，解析文件名中的参数，
并在 png_data 目录中生成对应的目录结构，将每帧保存为 PNG 图像。

支持的格式：
- Mono8: 灰度格式
- BayerRG8: Bayer彩色格式

使用方法:
python raw_to_images.py
"""

import os
import cv2
import numpy as np
import re
from pathlib import Path
import argparse

def parse_raw_filename(filename):
    """
    解析RAW文件名，提取视频参数
    文件名格式类似: [任意前缀]_w720_h540_pMono8_f0.raw
    返回: (width, height, pixel_format, fps)
    """
    # 使用正则表达式解析文件名
    pattern = r'(.+)_w(\d+)_h(\d+)_p([^_]+)_f(\d+)\.raw'
    match = re.match(pattern, filename)
    
    if match:
        # 前缀(group 1)不再具有明确的时间戳意义，不再返回
        width = int(match.group(2))
        height = int(match.group(3))
        pixel_format = match.group(4)
        fps = int(match.group(5))
        return width, height, pixel_format, fps
    else:
        raise ValueError(f"无法解析文件名格式: {filename}")

def calculate_total_frames(file_path, width, height, pixel_format):
    """
    计算RAW文件的总帧数
    """
    try:
        file_size = os.path.getsize(file_path)
        
        # 根据像素格式计算每帧的字节数
        if pixel_format in ['Mono8']:
            bytes_per_pixel = 1
        elif pixel_format in ['BayerRG8', 'BayerGB8', 'BayerGR8', 'BayerBG8']:
            bytes_per_pixel = 1
        elif pixel_format in ['Mono16']:
            bytes_per_pixel = 2
        else:
            print(f"警告: 未知的像素格式 {pixel_format}，假设为1字节/像素")
            bytes_per_pixel = 1
            
        frame_size = width * height * bytes_per_pixel
        total_frames = file_size // frame_size
        
        print(f"文件大小: {file_size:,} 字节")
        print(f"帧大小: {frame_size:,} 字节 ({width}x{height}, {pixel_format})")
        print(f"总帧数: {total_frames:,}")
        
        return total_frames, frame_size
    except Exception as e:
        print(f"计算帧数时出错 {file_path}: {e}")
        return 0, 0

def convert_raw_frame_to_bgr_or_gray(raw_data, width, height, pixel_format):
    """
    Convert RAW frame to BGR or single-channel Gray
    """
    # Convert buffer to numpy array
    frame = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width))
    
    if pixel_format == 'Mono8':
        # Return single channel directly for Mono8
        return frame
    elif pixel_format in ['BayerRG8', 'BayerGB8', 'BayerGR8', 'BayerBG8']:
        # Convert Bayer to RGB to intentionally swap channels for cv2.imwrite,
        # because the camera's Bayer pattern name (e.g. BayerRG8) does not map
        # straightforwardly to OpenCV's Bayer macros, and empirical tests show giving
        # imwrite RGB data correctly saves the physical channel order to disk.
        if pixel_format == 'BayerRG8':
            out_frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_RG2RGB)
        elif pixel_format == 'BayerGB8':
            out_frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_GB2RGB)
        elif pixel_format == 'BayerGR8':
            out_frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_GR2RGB)
        elif pixel_format == 'BayerBG8':
            out_frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2RGB)
        return out_frame
    else:
        print(f"Warning: Unknown format {pixel_format}, treat as gray")
        return frame

def convert_raw_to_png(raw_file_path, output_dir, verbose=True):
    """
    将单个RAW文件转换为PNG（实为JPEG）图像序列
    """
    vprint = print if verbose else lambda *a, **k: None
    try:
        filename = os.path.basename(raw_file_path)
        vprint(f"\n正在处理: {filename}")
        
        # 解析文件名获取参数
        width, height, pixel_format, fps = parse_raw_filename(filename)
        vprint(f"视频参数: {width}x{height}, {pixel_format}, {fps}fps")
        
        # 计算总帧数
        total_frames, frame_size = calculate_total_frames(raw_file_path, width, height, pixel_format)
        
        if total_frames == 0:
            vprint("跳过此文件（无法计算帧数）")
            return
        
        # 创建输出目录，并去掉空格
        video_name = os.path.splitext(filename)[0]  # 去掉.raw扩展名
        video_name = video_name.replace(' ', '')
        output_video_dir = os.path.join(output_dir, video_name)
        os.makedirs(output_video_dir, exist_ok=True)
        vprint(f"输出目录: {output_video_dir}")
        
        # 读取并转换每一帧
        with open(raw_file_path, 'rb') as f:
            for frame_idx in range(total_frames):
                # 读取一帧数据
                raw_data = f.read(frame_size)
                
                if len(raw_data) != frame_size:
                    vprint(f"警告: 第{frame_idx+1}帧数据不完整，停止转换")
                    break
                
                # 转换为BGR或单通道灰度
                out_frame = convert_raw_frame_to_bgr_or_gray(raw_data, width, height, pixel_format)
                
                # 保存为JPEG
                jpeg_filename = f"frame_{frame_idx:06d}.jpg"
                jpeg_path = os.path.join(output_video_dir, jpeg_filename)
                # 设置JPEG质量参数
                jpeg_quality = [cv2.IMWRITE_JPEG_QUALITY, 95]
                cv2.imwrite(jpeg_path, out_frame, jpeg_quality)
                
                # # 保存为PNG
                # png_filename = f"frame_{frame_idx:06d}.png"
                # png_path = os.path.join(output_video_dir, png_filename)
                # cv2.imwrite(png_path, out_frame)
                
                # 显示进度
                if (frame_idx + 1) % 50 == 0 or frame_idx == total_frames - 1:
                    progress = (frame_idx + 1) / total_frames * 100
                    vprint(f"进度: {frame_idx + 1}/{total_frames} ({progress:.1f}%)")
        
        vprint(f"转换完成: {total_frames} 帧已保存到 {output_video_dir}")
        
    except Exception as e:
        vprint(f"转换文件 {raw_file_path} 时出错: {e}")

def find_raw_files(data_dir):
    """
    遍历目录查找所有.raw文件
    """
    raw_files = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.raw'):
                file_path = os.path.join(root, file)
                # 获取相对于data_dir的相对路径，并去掉路径名中的空格
                rel_path = os.path.relpath(root, data_dir).replace(' ', '')
                raw_files.append((file_path, rel_path))
    
    return raw_files

def main():
    parser = argparse.ArgumentParser(description="RAW视频转PNG/JPEG图像序列工具")
    parser.add_argument("--data_dir", required=True, help="包含 .raw 文件的源目录")
    parser.add_argument("--output_base_dir", required=True, help="保存图像序列的基础输出目录")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_base_dir = args.output_base_dir
    
    print("RAW视频转图像序列工具")   
    print("=" * 50) 
    print(f"输入目录: {data_dir}")
    print(f"输出目录: {output_base_dir}")
    
    # 检查输入目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误: 输入目录不存在: {data_dir}")
        return
    
    # 创建输出基础目录
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 查找所有RAW文件
    print("\n正在查找.raw文件...")
    raw_files = find_raw_files(data_dir)
    
    if not raw_files:
        print("未找到任何.raw文件")
        return
    
    print(f"找到 {len(raw_files)} 个.raw文件:")
    for file_path, rel_path in raw_files:
        filename = os.path.basename(file_path)
        print(f"  {rel_path}\\{filename}")
    
    print("\n开始转换...")
    
    # 处理每个RAW文件
    for i, (file_path, rel_path) in enumerate(raw_files, 1):
        print(f"\n[{i}/{len(raw_files)}] 处理文件组: {rel_path}")
        
        # 在输出目录中创建对应的子目录结构
        output_dir = os.path.join(output_base_dir, rel_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # 转换当前文件
        convert_raw_to_png(file_path, output_dir)
    
    print("\n" + "=" * 50)
    print("所有转换完成！")
    print(f"PNG图像已保存到: {output_base_dir}")

if __name__ == '__main__':
    main()
