#!/usr/bin/env python3
"""
将PHA图（alpha通道图）转换为黑白掩码图，用于SFS（Shape from Shading）

Usage:
  python pha_to_mask.py \
    --pha_dir output/pha \
    --output_dir SFS/binary_masks \
    [--pattern "*.jpg"] [--threshold 127] [--invert]

Notes:
- 读取PHA图像（通常是灰度图表示alpha通道）
- 应用阈值将其转换为二值掩码（0或255）
- 保存为黑白掩码图，用于SFS算法
- 支持反转掩码（如果需要的话）
"""

import argparse
import os
from pathlib import Path
import glob
import cv2
import numpy as np


def load_pha_image(img_path: str) -> np.ndarray:
    """加载PHA图像"""
    # 使用GRAYSCALE模式加载，因为PHA通常是单通道alpha信息
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    return img


def pha_to_binary_mask(pha_img: np.ndarray, threshold: int = 127, invert: bool = False) -> np.ndarray:
    """
    将PHA图转换为二值掩码
    
    Args:
        pha_img: 输入的PHA图像（灰度图）
        threshold: 二值化阈值 (0-255)
        invert: 是否反转掩码
    
    Returns:
        二值掩码图像 (0或255)
    """
    # 应用阈值二值化
    if invert:
        # 反转：alpha值大于阈值的像素设为0（黑色），小于等于的设为255（白色）
        _, binary_mask = cv2.threshold(pha_img, threshold, 255, cv2.THRESH_BINARY_INV)
    else:
        # 正常：alpha值大于阈值的像素设为255（白色），小于等于的设为0（黑色）
        _, binary_mask = cv2.threshold(pha_img, threshold, 255, cv2.THRESH_BINARY)
    
    return binary_mask


def apply_morphological_operations(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    应用形态学操作清理掩码
    
    Args:
        mask: 输入二值掩码
        kernel_size: 形态学操作的核大小
        iterations: 操作迭代次数
    
    Returns:
        清理后的掩码
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # 先闭运算（去除小洞）再开运算（去除小噪点）
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel, iterations=iterations)
    
    return mask_cleaned


def analyze_pha_statistics(pha_dir: str, pattern: str = "*.jpg") -> dict:
    """分析PHA图像的统计信息，帮助确定合适的阈值"""
    pha_files = sorted(glob.glob(str(Path(pha_dir) / pattern)))
    
    if not pha_files:
        return {}
    
    print(f"分析 {len(pha_files)} 个PHA文件的统计信息...")
    
    all_values = []
    stats = {
        'file_count': len(pha_files),
        'per_file_stats': []
    }
    
    for pha_file in pha_files:
        try:
            img = load_pha_image(pha_file)
            values = img.flatten()
            all_values.extend(values)
            
            file_stats = {
                'filename': Path(pha_file).name,
                'min': int(values.min()),
                'max': int(values.max()),
                'mean': float(values.mean()),
                'median': float(np.median(values)),
                'std': float(values.std())
            }
            stats['per_file_stats'].append(file_stats)
            
        except Exception as e:
            print(f"[WARN] 无法处理文件 {pha_file}: {e}")
    
    if all_values:
        all_values = np.array(all_values)
        stats['global_stats'] = {
            'min': int(all_values.min()),
            'max': int(all_values.max()),
            'mean': float(all_values.mean()),
            'median': float(np.median(all_values)),
            'std': float(all_values.std()),
            'percentile_25': float(np.percentile(all_values, 25)),
            'percentile_75': float(np.percentile(all_values, 75))
        }
        
        print("全局统计信息:")
        print(f"  像素值范围: [{stats['global_stats']['min']}, {stats['global_stats']['max']}]")
        print(f"  平均值: {stats['global_stats']['mean']:.2f}")
        print(f"  中位数: {stats['global_stats']['median']:.2f}")
        print(f"  标准差: {stats['global_stats']['std']:.2f}")
        print(f"  25% 分位数: {stats['global_stats']['percentile_25']:.2f}")
        print(f"  75% 分位数: {stats['global_stats']['percentile_75']:.2f}")
        
        # 建议阈值
        suggested_threshold = int(stats['global_stats']['mean'])
        print(f"  建议阈值: {suggested_threshold}")
        stats['suggested_threshold'] = suggested_threshold
    
    return stats


def str2bool(v):
    """Convert string to boolean for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description='将PHA图转换为SFS用的黑白掩码图')
    parser.add_argument('--pha_dir', type=str, default='output/pha',
                       help='PHA图像输入目录')
    parser.add_argument('--output_dir', type=str, default='SFS/binary_masks',
                       help='掩码输出目录')
    parser.add_argument('--pattern', type=str, default='*.jpg',
                       help='图像文件匹配模式')
    parser.add_argument('--threshold', type=int, default=None,
                       help='二值化阈值 (0-255)，如未指定则自动计算')
    parser.add_argument('--invert', type=str2bool, default=False,
                       help='是否反转掩码（True: 透明区域为白色，False: 不透明区域为白色）')
    parser.add_argument('--morphology', type=str2bool, default=True,
                       help='是否应用形态学操作清理掩码')
    parser.add_argument('--kernel_size', type=int, default=3,
                       help='形态学操作的核大小')
    parser.add_argument('--iterations', type=int, default=1,
                       help='形态学操作的迭代次数')
    parser.add_argument('--analyze_only', type=str2bool, default=False,
                       help='仅分析PHA图像统计信息，不生成掩码')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.pha_dir):
        print(f"错误: PHA目录不存在: {args.pha_dir}")
        return
    
    # 分析PHA图像统计信息
    stats = analyze_pha_statistics(args.pha_dir, args.pattern)
    
    if not stats:
        print("错误: 未找到任何PHA文件")
        return
    
    if args.analyze_only:
        print("仅分析模式，不生成掩码文件")
        return
    
    # 确定阈值
    if args.threshold is None:
        threshold = stats.get('suggested_threshold', 127)
        print(f"使用自动计算的阈值: {threshold}")
    else:
        threshold = args.threshold
        print(f"使用指定阈值: {threshold}")
    
    # 获取PHA文件列表
    pha_files = sorted(glob.glob(str(Path(args.pha_dir) / args.pattern)))
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n开始处理 {len(pha_files)} 个PHA文件...")
    print(f"输出目录: {args.output_dir}")
    print(f"阈值: {threshold}")
    print(f"反转: {'是' if args.invert else '否'}")
    print(f"形态学清理: {'是' if args.morphology else '否'}")
    
    success_count = 0
    
    for i, pha_file in enumerate(pha_files):
        try:
            # 加载PHA图像
            pha_img = load_pha_image(pha_file)
            
            # 转换为二值掩码
            mask = pha_to_binary_mask(pha_img, threshold, args.invert)
            
            # 可选的形态学操作
            if args.morphology:
                mask = apply_morphological_operations(mask, args.kernel_size, args.iterations)
            
            # 保存掩码
            input_name = Path(pha_file).stem
            output_path = Path(args.output_dir) / f"{input_name}.png"
            
            success = cv2.imwrite(str(output_path), mask)
            if success:
                success_count += 1
                print(f"  [{i+1:2d}/{len(pha_files)}] {Path(pha_file).name} -> {output_path.name}")
            else:
                print(f"  [错误] 无法保存掩码: {output_path}")
                
        except Exception as e:
            print(f"  [错误] 处理 {pha_file} 时出错: {e}")
    
    print(f"\n处理完成! 成功生成 {success_count}/{len(pha_files)} 个掩码文件")
    print(f"掩码文件保存在: {args.output_dir}")
    
    # 输出使用说明
    print("\n使用说明:")
    print("- 生成的掩码图像为PNG格式，像素值为0（黑色）或255（白色）")
    print("- 可以在SFS算法中用作二值掩码来限制处理区域")
    print("- 如果掩码效果不理想，可以调整--threshold参数或使用--invert反转")


if __name__ == '__main__':
    main()