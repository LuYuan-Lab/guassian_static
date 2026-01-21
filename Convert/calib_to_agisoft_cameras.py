#!/usr/bin/env python3
"""
将 calib.json (libCalib 格式) 转换为 Agisoft Metashape XML 格式。
严格按照 cameras.xml 的结构生成。
"""
import argparse
import json
from pathlib import Path
import numpy as np


def extract_camera_params(camera):
    """从 camera 对象中提取内参和外参"""
    model_data = camera["model"]["ptr_wrapper"]["data"]
    params = model_data["parameters"]
    
    # 内参
    f = params["f"]["val"]
    cx = params["cx"]["val"]
    cy = params["cy"]["val"]
    k1 = params.get("k1", {}).get("val", 0.0)
    k2 = params.get("k2", {}).get("val", 0.0)
    k3 = params.get("k3", {}).get("val", 0.0)
    p1 = params.get("p1", {}).get("val", 0.0)
    p2 = params.get("p2", {}).get("val", 0.0)
    
    # 图像尺寸
    image_size = model_data["CameraModelCRT"]["CameraModelBase"]["imageSize"]
    width = image_size["width"]
    height = image_size["height"]
    
    # 外参（旋转和平移）
    transform = camera["transform"]
    rx = transform["rotation"]["rx"]
    ry = transform["rotation"]["ry"] 
    rz = transform["rotation"]["rz"]
    tx = transform["translation"]["x"]
    ty = transform["translation"]["y"]
    tz = transform["translation"]["z"]
    
    return {
        "f": f, "cx": cx, "cy": cy, "k1": k1, "k2": k2, "k3": k3, "p1": p1, "p2": p2,
        "width": width, "height": height,
        "rx": rx, "ry": ry, "rz": rz, "tx": tx, "ty": ty, "tz": tz
    }


def rodrigues_to_rotation_matrix(rvec):
    """将 Rodrigues 向量转换为旋转矩阵"""
    theta = np.linalg.norm(rvec)
    
    if theta < 1e-8:
        return np.eye(3) + skew_symmetric(rvec)
    
    k = rvec / theta
    K = skew_symmetric(k)
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    
    return R


def skew_symmetric(v):
    """构建反对称矩阵"""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def rodrigues_to_transform_matrix(rx, ry, rz, tx, ty, tz):
    """从 Rodrigues 向量和平移向量构建4x4变换矩阵 (c2w格式)"""
    rvec = np.array([rx, ry, rz])
    rotation_matrix_w2c = rodrigues_to_rotation_matrix(rvec)
    rotation_matrix_c2w = rotation_matrix_w2c.T
    tw2c = np.array([tx, ty, tz])
    tc2w = -rotation_matrix_c2w @ tw2c
    
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix_c2w
    transform_matrix[:3, 3] = tc2w
    
    return transform_matrix


def generate_xml(cameras_data, output_path):
    """手动生成与 cameras.xml 完全一致格式的 XML"""
    
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<document version="2.0.0">')
    lines.append('  <chunk label="Chunk 1" enabled="true">')
    
    num_cameras = len(cameras_data)
    
    # ===== sensors 部分 =====
    lines.append(f'    <sensors next_id="{num_cameras}">')
    
    for i, cam_data in enumerate(cameras_data):
        params = cam_data['params']
        w, h = params["width"], params["height"]
        
        # Agisoft中cx、cy是相对于图像中心的偏移
        cx_agisoft = params["cx"] - w / 2
        cy_agisoft = params["cy"] - h / 2
        
        lines.append(f'      <sensor id="{i}" label="unknown" type="frame">')
        lines.append(f'        <resolution width="{w}" height="{h}"/>')
        lines.append(f'        <property name="layer_index" value="0"/>')
        lines.append(f'        <bands>')
        lines.append(f'          <band label="Red"/>')
        lines.append(f'          <band label="Green"/>')
        lines.append(f'          <band label="Blue"/>')
        lines.append(f'        </bands>')
        lines.append(f'        <data_type>uint8</data_type>')
        lines.append(f'        <calibration type="frame" class="adjusted">')
        lines.append(f'          <resolution width="{w}" height="{h}"/>')
        lines.append(f'          <f>{params["f"]}</f>')
        lines.append(f'          <cx>{cx_agisoft}</cx>')
        lines.append(f'          <cy>{cy_agisoft}</cy>')
        lines.append(f'          <k1>{params["k1"]}</k1>')
        lines.append(f'          <k2>{params["k2"]}</k2>')
        lines.append(f'          <k3>{params["k3"]}</k3>')
        lines.append(f'          <p1>{params["p1"]}</p1>')
        lines.append(f'          <p2>{params["p2"]}</p2>')
        lines.append(f'        </calibration>')
        lines.append(f'        <black_level>0 0 0</black_level>')
        lines.append(f'        <sensitivity>1 1 1</sensitivity>')
        lines.append(f'      </sensor>')
    
    lines.append('    </sensors>')
    
    # ===== components 部分 =====
    camera_ids = ' '.join(str(i) for i in range(num_cameras))
    lines.append('    <components next_id="1" active_id="0">')
    lines.append('      <component id="0" label="Component 1">')
    lines.append('        <region>')
    lines.append('          <center>0 0 0</center>')
    lines.append('          <size>10 10 10</size>')
    lines.append('          <R>1 0 0 0 1 0 0 0 1</R>')
    lines.append('        </region>')
    lines.append('        <partition>')
    lines.append(f'          <camera_ids>{camera_ids}</camera_ids>')
    lines.append('        </partition>')
    lines.append('      </component>')
    lines.append('    </components>')
    
    # ===== cameras 部分 =====
    lines.append(f'    <cameras next_id="{num_cameras}" next_group_id="0">')
    
    for i, cam_data in enumerate(cameras_data):
        transform_str = cam_data['transform_str']
        label = f"{i+1:04d}"
        
        lines.append(f'      <camera id="{i}" sensor_id="{i}" component_id="0" label="{label}">')
        lines.append(f'        <transform>{transform_str}</transform>')
        lines.append(f'      </camera>')
    
    lines.append('    </cameras>')
    
    # ===== reference 部分 =====
    lines.append('    <reference>LOCAL_CS["Local Coordinates (m)",LOCAL_DATUM["Local Datum",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]</reference>')
    
    # ===== region 部分 =====
    lines.append('    <region>')
    lines.append('      <center>0 0 0</center>')
    lines.append('      <size>10 10 10</size>')
    lines.append('      <R>1 0 0 0 1 0 0 0 1</R>')
    lines.append('    </region>')
    
    # ===== settings 部分 =====
    lines.append('    <settings>')
    lines.append('      <property name="accuracy_tiepoints" value="1"/>')
    lines.append('      <property name="accuracy_cameras" value="10"/>')
    lines.append('      <property name="accuracy_cameras_ypr" value="10"/>')
    lines.append('      <property name="accuracy_markers" value="0.0050000000000000001"/>')
    lines.append('      <property name="accuracy_scalebars" value="0.001"/>')
    lines.append('      <property name="accuracy_projections" value="0.5"/>')
    lines.append('    </settings>')
    
    # ===== meta 部分 =====
    lines.append('    <meta>')
    lines.append('      <property name="AlignCameras/adaptive_fitting" value="false"/>')
    lines.append('      <property name="AlignCameras/cameras" value=""/>')
    lines.append('      <property name="AlignCameras/duration" value="0"/>')
    lines.append('      <property name="AlignCameras/min_image" value="2"/>')
    lines.append('      <property name="AlignCameras/point_clouds" value=""/>')
    lines.append('      <property name="AlignCameras/ram_used" value="0"/>')
    lines.append('      <property name="AlignCameras/reset_alignment" value="true"/>')
    lines.append('      <property name="AlignCameras/subdivide_task" value="true"/>')
    lines.append('    </meta>')
    
    lines.append('  </chunk>')
    lines.append('</document>')
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description="将 calib.json 转换为 Agisoft camera.xml 格式")
    parser.add_argument("--input", type=Path, default="output/calib_undistorted.json", help="输入的 calib.json 文件路径")
    parser.add_argument("--output", type=Path, default="output/cameras_generated.xml", help="输出的 XML 文件路径")
    args = parser.parse_args()

    # 加载 calib.json
    try:
        calib_data = json.loads(args.input.read_text())
        cameras = calib_data["Calibration"]["cameras"]
    except (KeyError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 无法读取或解析 {args.input}: {e}")
        return

    if not cameras:
        print("错误: 未找到相机数据")
        return

    # 准备相机数据
    cameras_data = []
    for i, camera in enumerate(cameras):
        params = extract_camera_params(camera)
        
        # 构建变换矩阵
        transform_matrix = rodrigues_to_transform_matrix(
            params["rx"], params["ry"], params["rz"],
            params["tx"], params["ty"], params["tz"]
        )
        
        # 转换为字符串格式（空格分隔的16个值）
        transform_str = ' '.join(str(v) for v in transform_matrix.flatten())
        
        cameras_data.append({
            'params': params,
            'transform_str': transform_str
        })

    # 生成XML
    generate_xml(cameras_data, args.output)
    print(f"成功将 {len(cameras)} 个相机写入到 {args.output}")


if __name__ == "__main__":
    main()
