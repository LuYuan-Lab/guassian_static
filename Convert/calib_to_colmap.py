#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将calib_uniform.json转换为COLMAP格式的cameras.txt和images.txt
COLMAP格式参考：https://colmap.github.io/format.html
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
    
    for i, cam_entry in enumerate(data['Calibration']['cameras']):
        # 提取参数
        ptr_data = cam_entry['model']['ptr_wrapper']['data']
        params = ptr_data['parameters']
        
        # 图像尺寸
        img_size_data = ptr_data['CameraModelCRT']['CameraModelBase']['imageSize']
        width = img_size_data['width']
        height = img_size_data['height']
        
        # 内参矩阵
        fx = params['f']['val']
        ar = params['ar']['val']
        cx = params['cx']['val']
        cy = params['cy']['val']
        
        # 畸变系数 (OpenCV格式: k1, k2, p1, p2, k3)
        k1 = params['k1']['val']
        k2 = params['k2']['val']
        p1 = params['p1']['val']
        p2 = params['p2']['val']
        k3 = params['k3']['val']

        # 外参 (W2C: World-to-Camera)
        trans_data = cam_entry['transform']
        rx = trans_data['rotation']['rx']
        ry = trans_data['rotation']['ry']
        rz = trans_data['rotation']['rz']
        tx = trans_data['translation']['x']
        ty = trans_data['translation']['y']
        tz = trans_data['translation']['z']

        # Rodrigues向量转旋转矩阵
        r_vec = np.array([rx, ry, rz], dtype=np.float64)
        R_w2c, _ = cv2.Rodrigues(r_vec)
        t_w2c = np.array([tx, ty, tz], dtype=np.float64).reshape(3, 1)
        
        # 计算相机中心 (世界坐标系中的位置)
        # C = -R^T * t (从W2C转换得到)
        camera_center = -R_w2c.T @ t_w2c
        
        # COLMAP需要C2W (Camera-to-World)变换
        # R_c2w = R_w2c^T
        # t_c2w = camera_center
        R_c2w = R_w2c.T
        t_c2w = camera_center.flatten()

        cameras_data.append({
            'camera_id': i + 1,  # COLMAP相机ID从1开始
            'width': width,
            'height': height,
            'fx': fx,
            'fy': fx * ar,
            'cx': cx,
            'cy': cy,
            'k1': k1,
            'k2': k2,
            'p1': p1,
            'p2': p2,
            'k3': k3,
            'R_c2w': R_c2w,  # Camera-to-World旋转矩阵
            't_c2w': t_c2w,  # Camera-to-World平移向量
            'R_w2c': R_w2c,  # World-to-Camera旋转矩阵 (原始)
            't_w2c': t_w2c.flatten()  # World-to-Camera平移向量 (原始)
        })

    return cameras_data

def rotation_matrix_to_quaternion(R):
    """
    将旋转矩阵转换为四元数 (w, x, y, z)
    COLMAP使用四元数表示旋转
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    return np.array([qw, qx, qy, qz])

def write_cameras_txt(cameras_data, output_file):
    """
    写入COLMAP格式的cameras.txt文件
    
    格式：
    # Camera list with one line of data per camera:
    #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    # Number of cameras: N
    
    对于OPENCV相机模型：
    CAMERA_ID, OPENCV, WIDTH, HEIGHT, PARAMS[]
    """
    with open(output_file, 'w') as f:
        # 写入头部注释
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: {}\n".format(len(cameras_data)))
        
        # 写入每个相机的参数
        for cam in cameras_data:
            # 使用OPENCV相机模型
            # COLMAP的OPENCV模型参数顺序：fx, fy, cx, cy, k1, k2, p1, p2
            f.write("{} OPENCV {} {} {} {} {} {} {} {} {} {}\n".format(
                cam['camera_id'],
                cam['width'],
                cam['height'],
                cam['fx'],     # fx
                cam['fy'],     # fy
                cam['cx'],
                cam['cy'],
                cam['k1'],
                cam['k2'],
                cam['p1'],
                cam['p2']
                # 注意：COLMAP的OPENCV模型只支持k1,k2,p1,p2，不包含k3
            ))
    
    logger.info(f"cameras.txt已保存至: {output_file}")

def write_images_txt(cameras_data, image_dir, output_file, image_extension='.png'):
    """
    写入COLMAP格式的images.txt文件
    
    格式：
    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    # Number of registered images: N
    """
    
    # 获取图像文件列表
    image_files = []
    if os.path.exists(image_dir):
        pattern = os.path.join(image_dir, f"*{image_extension}")
        image_files = sorted(glob.glob(pattern))
    
    if len(image_files) == 0:
        logger.warning(f"在 {image_dir} 中没有找到图像文件 (*{image_extension})")
        # 生成默认文件名
        image_files = [f"{i+1:03d}{image_extension}" for i in range(len(cameras_data))]
    
    with open(output_file, 'w') as f:
        # 写入头部注释
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write("# Number of registered images: {}\n".format(len(cameras_data)))
        
        # 写入每个图像的信息
        for i, cam in enumerate(cameras_data):
            # 获取对应的图像文件名
            if i < len(image_files):
                image_name = os.path.basename(image_files[i])
            else:
                image_name = f"{cam['camera_id']:03d}{image_extension}"
            
            # 将旋转矩阵转换为四元数
            quat = rotation_matrix_to_quaternion(cam['R_w2c'])
            qw, qx, qy, qz = quat
            
            # 平移向量
            tx, ty, tz = cam['t_w2c']
            
            # 写入图像信息行
            f.write("{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {} {}\n".format(
                i + 1,      # IMAGE_ID (从1开始)
                qw, qx, qy, qz,  # 四元数旋转
                tx, ty, tz,      # 平移
                cam['camera_id'], # CAMERA_ID
                image_name        # 图像文件名
            ))
            
            # 写入空的2D点行（没有特征点匹配信息）
            f.write("\n")
    
    logger.info(f"images.txt已保存至: {output_file}")

def write_points3d_txt(output_file):
    """
    写入空的points3D.txt文件
    COLMAP需要这个文件，但我们没有3D点数据
    """
    with open(output_file, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")
    
    logger.info(f"points3D.txt已保存至: {output_file}")

def create_colmap_database_info(cameras_data, output_dir):
    """
    创建COLMAP数据库的信息文件
    """
    info_file = os.path.join(output_dir, "database_info.txt")
    
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("=== COLMAP数据库转换信息 ===\n")
        f.write(f"转换时间: {__import__('datetime').datetime.now()}\n")
        f.write(f"相机数量: {len(cameras_data)}\n")
        f.write(f"图像数量: {len(cameras_data)}\n")
        f.write("\n=== 相机参数统计 ===\n")
        
        # 统计相机参数
        fx_list = [cam['fx'] for cam in cameras_data]
        fy_list = [cam['fy'] for cam in cameras_data]
        widths = [cam['width'] for cam in cameras_data]
        heights = [cam['height'] for cam in cameras_data]
        
        f.write(f"图像分辨率: {widths[0]}x{heights[0]}\n")
        f.write(f"fx范围: [{min(fx_list):.2f}, {max(fx_list):.2f}]\n")
        f.write(f"fx均值: {np.mean(fx_list):.2f}\n")
        f.write(f"fy范围: [{min(fy_list):.2f}, {max(fy_list):.2f}]\n")
        f.write(f"fy均值: {np.mean(fy_list):.2f}\n")
        
        # 相机位置统计
        positions = np.array([cam['t_c2w'] for cam in cameras_data])
        f.write(f"\n=== 相机位置统计 ===\n")
        f.write(f"X范围: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]\n")
        f.write(f"Y范围: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]\n")
        f.write(f"Z范围: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]\n")
        
        f.write(f"\n=== 文件说明 ===\n")
        f.write("cameras.txt: 相机内参和畸变参数\n")
        f.write("images.txt: 图像外参（位姿）信息\n")
        f.write("points3D.txt: 3D点云数据（空文件）\n")
        f.write("\n=== 使用方法 ===\n")
        f.write("1. 将生成的文件放入COLMAP项目的sparse/0/目录\n")
        f.write("2. 使用COLMAP GUI或命令行工具加载稀疏重建\n")
        f.write("3. 可以继续进行密集重建或其他处理\n")
    
    logger.info(f"数据库信息已保存至: {info_file}")

def validate_colmap_format(output_dir):
    """
    验证生成的COLMAP文件格式
    """
    cameras_file = os.path.join(output_dir, "cameras.txt")
    images_file = os.path.join(output_dir, "images.txt")
    points_file = os.path.join(output_dir, "points3D.txt")
    
    validation_results = {}
    
    # 验证cameras.txt
    if os.path.exists(cameras_file):
        with open(cameras_file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if not line.startswith('#') and line.strip()]
            validation_results['cameras'] = {
                'exists': True,
                'num_cameras': len(lines),
                'sample_line': lines[0] if lines else None
            }
    else:
        validation_results['cameras'] = {'exists': False}
    
    # 验证images.txt
    if os.path.exists(images_file):
        with open(images_file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if not line.startswith('#') and line.strip()]
            # images.txt中每个图像占两行（第二行可能为空）
            # 计算非空行数量来确定图像数
            non_empty_lines = [line for line in lines if line]
            validation_results['images'] = {
                'exists': True,
                'num_images': len(non_empty_lines),  # 每个图像的第一行
                'total_lines': len(lines)
            }
    else:
        validation_results['images'] = {'exists': False}
    
    # 验证points3D.txt
    validation_results['points3D'] = {'exists': os.path.exists(points_file)}
    
    # 输出验证结果
    logger.info("=== COLMAP格式验证结果 ===")
    for file_type, result in validation_results.items():
        if result['exists']:
            logger.info(f"✓ {file_type}.txt: 存在")
            if 'num_cameras' in result:
                logger.info(f"  - 相机数量: {result['num_cameras']}")
            if 'num_images' in result:
                logger.info(f"  - 图像数量: {result['num_images']}")
        else:
            logger.error(f"✗ {file_type}.txt: 缺失")
    
    return validation_results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将calib_uniform.json转换为COLMAP格式')
    parser.add_argument('--calib', type=str, 
                       default='output/undistorted/calib_undistorted.json',
                       help='相机标定文件路径')
    parser.add_argument('--images', type=str, 
                       default='output/undistorted',
                       help='图像目录路径')
    parser.add_argument('--output', type=str, default='colmap_sparse',
                       help='输出目录')
    parser.add_argument('--image-ext', type=str, default='.png',
                       help='图像文件扩展名')
    parser.add_argument('--validate', action='store_true',
                       help='验证生成的文件格式')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.calib):
        logger.error(f"标定文件不存在: {args.calib}")
        return
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载相机数据
    logger.info(f"加载标定文件: {args.calib}")
    cameras_data = load_calibration(args.calib)
    logger.info(f"成功加载 {len(cameras_data)} 个相机参数")
    
    # 生成COLMAP文件
    logger.info("生成COLMAP格式文件...")
    
    cameras_file = output_dir / "cameras.txt"
    images_file = output_dir / "images.txt"
    points_file = output_dir / "points3D.txt"
    
    write_cameras_txt(cameras_data, cameras_file)
    write_images_txt(cameras_data, args.images, images_file, args.image_ext)
    write_points3d_txt(points_file)
    
    # 创建信息文件
    create_colmap_database_info(cameras_data, output_dir)
    
    # 验证格式
    if args.validate:
        validate_colmap_format(output_dir)
    
    logger.info(f"COLMAP转换完成！文件保存在: {output_dir}")
    logger.info("使用方法：")
    logger.info(f"1. 将 {output_dir} 目录重命名为 sparse/0/")
    logger.info("2. 在COLMAP中导入这个稀疏重建结果")
    logger.info("3. 继续进行密集重建或其他处理")

if __name__ == '__main__':
    main()