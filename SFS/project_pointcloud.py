#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将点云投影回各个相机视角的程序
用于验证重建质量和生成可视化结果
"""

import json
import numpy as np
import cv2
import os
import glob
from pathlib import Path
import argparse
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_calibration(json_path):
    """
    解析 calib_uniform.json 文件
    返回相机参数列表
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    cameras_data = []
    
    for cam_entry in data['Calibration']['cameras']:
        # 提取参数
        ptr_data = cam_entry['model']['ptr_wrapper']['data']
        params = ptr_data['parameters']
        
        # 图像尺寸
        img_size_data = ptr_data['CameraModelCRT']['CameraModelBase']['imageSize']
        width = img_size_data['width']
        height = img_size_data['height']
        
        # 内参矩阵
        f_val = params['f']['val']
        cx = params['cx']['val']
        cy = params['cy']['val']
        
        K = np.array([
            [f_val, 0, cx],
            [0, f_val, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # 畸变系数
        dist_coeffs = np.array([
            params['k1']['val'],
            params['k2']['val'],
            params['p1']['val'],
            params['p2']['val'],
            params['k3']['val']
        ], dtype=np.float64)

        # 外参 (W2C)
        trans_data = cam_entry['transform']
        rx = trans_data['rotation']['rx']
        ry = trans_data['rotation']['ry']
        rz = trans_data['rotation']['rz']
        tx = trans_data['translation']['x']
        ty = trans_data['translation']['y']
        tz = trans_data['translation']['z']

        r_vec = np.array([rx, ry, rz], dtype=np.float64)
        T = np.array([tx, ty, tz], dtype=np.float64).reshape(3, 1)
        
        R, _ = cv2.Rodrigues(r_vec)

        cameras_data.append({
            'K': K,
            'D': dist_coeffs,
            'R': R,
            'T': T,
            'r_vec': r_vec,
            'width': width,
            'height': height
        })

    return cameras_data

def load_pointcloud(ply_path):
    """
    加载PLY点云文件
    返回点坐标和颜色（如果有的话）
    """
    points = []
    colors = []
    has_colors = False
    
    try:
        with open(ply_path, 'r') as f:
            # 读取头部
            line = f.readline().strip()
            if line != "ply":
                raise ValueError("不是有效的PLY文件")
            
            vertex_count = 0
            in_header = True
            properties = []
            
            while in_header:
                line = f.readline().strip()
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                elif line.startswith("property"):
                    parts = line.split()
                    if len(parts) >= 3:
                        properties.append(parts[2])  # 属性名
                elif line == "end_header":
                    in_header = False
            
            # 检查是否有颜色属性
            has_colors = 'red' in properties and 'green' in properties and 'blue' in properties
            
            # 读取数据
            for i in range(vertex_count):
                line = f.readline().strip()
                if not line:
                    break
                    
                values = line.split()
                if len(values) >= 3:
                    # 坐标 (前3个值)
                    x, y, z = map(float, values[:3])
                    points.append([x, y, z])
                    
                    # 颜色 (如果有的话，通常是最后3个值)
                    if has_colors and len(values) >= 9:
                        r, g, b = map(int, values[-3:])
                        colors.append([r, g, b])
        
        points = np.array(points, dtype=np.float64)
        if has_colors and len(colors) == len(points):
            colors = np.array(colors, dtype=np.uint8)
        else:
            colors = None
            
        logger.info(f"加载点云: {len(points)} 个点, 颜色: {'是' if colors is not None else '否'}")
        return points, colors
        
    except Exception as e:
        logger.error(f"加载PLY文件失败: {e}")
        return None, None

def project_points_to_camera(points, camera, point_colors=None, point_size=2, bg_color=(0, 0, 0)):
    """
    将3D点投影到指定相机视角
    
    参数:
        points: 3D点坐标 (N, 3)
        camera: 相机参数字典
        point_colors: 点的颜色 (N, 3) RGB，可选
        point_size: 投影点的显示大小
        bg_color: 背景颜色 (B, G, R)
    
    返回:
        投影图像
    """
    width, height = camera['width'], camera['height']
    
    # 创建黑色背景图像
    image = np.full((height, width, 3), bg_color, dtype=np.uint8)
    
    if points is None or len(points) == 0:
        return image
    
    # 投影3D点到2D
    image_points, _ = cv2.projectPoints(
        points, 
        camera['r_vec'], 
        camera['T'], 
        camera['K'], 
        camera['D']
    )
    
    image_points = image_points.reshape(-1, 2)
    
    # 筛选在图像范围内的点
    valid_x = np.logical_and(image_points[:, 0] >= 0, image_points[:, 0] < width)
    valid_y = np.logical_and(image_points[:, 1] >= 0, image_points[:, 1] < height)
    valid_indices = np.logical_and(valid_x, valid_y)
    
    valid_points = image_points[valid_indices]
    
    # 绘制投影点
    for i, (x, y) in enumerate(valid_points):
        # 确定颜色
        if point_colors is not None:
            valid_idx = np.where(valid_indices)[0][i]
            color = tuple(map(int, point_colors[valid_idx][::-1]))  # RGB -> BGR
        else:
            color = (0, 255, 0)  # 默认绿色
        
        # 绘制圆点
        cv2.circle(image, (int(x), int(y)), point_size, color, -1)
    
    logger.info(f"投影了 {len(valid_points)}/{len(points)} 个点到相机视图")
    return image

def create_overlay_image(background_img, projection_img, alpha=0.7):
    """
    创建背景图像和投影点的叠加图像
    
    参数:
        background_img: 背景图像
        projection_img: 投影点图像
        alpha: 背景图像的透明度
    
    返回:
        叠加图像
    """
    if background_img.shape != projection_img.shape:
        # 调整尺寸匹配
        projection_img = cv2.resize(projection_img, (background_img.shape[1], background_img.shape[0]))
    
    # 创建掩码：投影图像中非黑色的部分
    mask = np.any(projection_img > 0, axis=2)
    
    # 叠加图像
    overlay = background_img.copy().astype(np.float32)
    overlay = overlay * alpha
    
    # 在有投影点的地方添加投影颜色
    overlay[mask] = overlay[mask] * (1 - alpha) + projection_img[mask].astype(np.float32) * alpha
    
    return overlay.astype(np.uint8)

def batch_project_pointcloud(ply_path, json_path, image_dir=None, output_dir="projections", 
                           point_size=2, create_overlay=False, save_individual=True):
    """
    批量将点云投影到所有相机视角
    
    参数:
        ply_path: PLY点云文件路径
        json_path: 相机标定文件路径
        image_dir: 原始图像目录（用于叠加），可选
        output_dir: 输出目录
        point_size: 投影点大小
        create_overlay: 是否创建与原始图像的叠加
        save_individual: 是否保存单独的投影图像
    """
    # 加载数据
    points, colors = load_pointcloud(ply_path)
    if points is None:
        logger.error("无法加载点云文件")
        return
    
    cameras = load_calibration(json_path)
    logger.info(f"加载了 {len(cameras)} 个相机参数")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if create_overlay:
        overlay_path = output_path / "overlays"
        overlay_path.mkdir(exist_ok=True)
    
    # 获取原始图像文件（如果提供）
    image_files = []
    if image_dir and os.path.exists(image_dir):
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        logger.info(f"找到 {len(image_files)} 个原始图像文件")
    
    # 处理每个相机
    for i, camera in enumerate(cameras):
        camera_name = f"camera_{i+1:03d}"
        logger.info(f"处理相机 {i+1}/{len(cameras)}: {camera_name}")
        
        # 投影点云
        projection_img = project_points_to_camera(
            points, camera, colors, point_size, bg_color=(0, 0, 0)
        )
        
        # 保存纯投影图像
        if save_individual:
            proj_filename = output_path / f"{camera_name}_projection.png"
            cv2.imwrite(str(proj_filename), projection_img)
        
        # 创建叠加图像
        if create_overlay and i < len(image_files):
            original_img = cv2.imread(image_files[i])
            if original_img is not None:
                # 调整原始图像尺寸到相机分辨率
                target_size = (camera['width'], camera['height'])
                if original_img.shape[1] != target_size[0] or original_img.shape[0] != target_size[1]:
                    original_img = cv2.resize(original_img, target_size)
                
                # 创建叠加效果
                overlay_img = create_overlay_image(original_img, projection_img, alpha=0.6)
                
                # 保存叠加图像
                overlay_filename = overlay_path / f"{camera_name}_overlay.png"
                cv2.imwrite(str(overlay_filename), overlay_img)
    
    logger.info(f"投影完成！结果保存在: {output_dir}")
    
    # 生成总结报告
    create_projection_summary(output_dir, len(cameras), len(points), colors is not None)

def create_projection_summary(output_dir, num_cameras, num_points, has_colors):
    """生成投影结果总结"""
    summary_file = Path(output_dir) / "projection_summary.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=== 点云投影总结 ===\n")
        f.write(f"处理时间: {__import__('datetime').datetime.now()}\n")
        f.write(f"相机数量: {num_cameras}\n")
        f.write(f"点云点数: {num_points}\n")
        f.write(f"包含颜色: {'是' if has_colors else '否'}\n")
        f.write(f"输出图像: {num_cameras} 个投影视图\n")
        f.write("\n文件结构:\n")
        f.write("- camera_XXX_projection.png: 纯投影图像\n")
        f.write("- overlays/camera_XXX_overlay.png: 与原始图像叠加\n")
    
    logger.info(f"总结报告保存为: {summary_file}")

def create_projection_grid(output_dir, grid_size=(4, 6), output_name="projection_grid.png"):
    """创建投影结果的网格图"""
    try:
        import matplotlib.pyplot as plt
        
        output_path = Path(output_dir)
        proj_files = sorted(list(output_path.glob("*_projection.png")))
        
        if len(proj_files) == 0:
            logger.warning("没有找到投影图像文件")
            return
        
        rows, cols = grid_size
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        axes = axes.flatten() if rows * cols > 1 else [axes]
        
        for i, proj_file in enumerate(proj_files[:rows*cols]):
            img = cv2.imread(str(proj_file))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(img_rgb)
                axes[i].set_title(f"Camera {i+1}", fontsize=10)
                axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(len(proj_files), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        grid_file = output_path / output_name
        plt.savefig(grid_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"网格图保存为: {grid_file}")
        
    except ImportError:
        logger.warning("matplotlib未安装，跳过网格图生成")
    except Exception as e:
        logger.error(f"生成网格图时出错: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将点云投影到各个相机视角')
    parser.add_argument('--ply', type=str, default='reconstruction.ply',
                       help='PLY点云文件路径')
    parser.add_argument('--calib', type=str, 
                       default='undistorted_pha_uniform/calib_uniform.json',
                       help='相机标定文件路径')
    parser.add_argument('--images', type=str, 
                       default='undistorted_pha_uniform/rectified',
                       help='原始图像目录（用于叠加显示）')
    parser.add_argument('--output', type=str, default='projections',
                       help='输出目录')
    parser.add_argument('--point-size', type=int, default=2,
                       help='投影点的显示大小')
    parser.add_argument('--overlay', action='store_true',
                       help='创建与原始图像的叠加显示')
    parser.add_argument('--grid', action='store_true',
                       help='生成投影结果的网格图')
    parser.add_argument('--no-individual', action='store_true',
                       help='不保存单独的投影图像')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.ply):
        logger.error(f"点云文件不存在: {args.ply}")
        return
    
    if not os.path.exists(args.calib):
        logger.error(f"标定文件不存在: {args.calib}")
        return
    
    # 执行投影
    batch_project_pointcloud(
        ply_path=args.ply,
        json_path=args.calib,
        image_dir=args.images if os.path.exists(args.images) else None,
        output_dir=args.output,
        point_size=args.point_size,
        create_overlay=args.overlay,
        save_individual=not args.no_individual
    )
    
    # 生成网格图
    if args.grid:
        create_projection_grid(args.output)

if __name__ == '__main__':
    main()