import json
import numpy as np
import cv2
import os
import glob

def load_calibration(json_path):
    """
    解析 calib_uniform.json 文件
    注意：calib 中的 R 和 t 均为 w2c (World-to-Camera)，
    直接符合 OpenCV projectPoints 的输入要求，无需求逆。
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    cameras_data = []
    
    # 遍历 JSON 中的 cameras 列表
    for cam_entry in data['Calibration']['cameras']:
        # 1. 提取参数
        # 根据提供的 JSON 结构定位 parameters
        ptr_data = cam_entry['model']['ptr_wrapper']['data']
        params = ptr_data['parameters']
        
        # 提取图像尺寸 (width, height) 以防硬编码不匹配
        img_size_data = ptr_data['CameraModelCRT']['CameraModelBase']['imageSize']
        width = img_size_data['width']
        height = img_size_data['height']
        
        # 2. 构建内参矩阵 K
        # fx = fy = f (ar=1.0)
        f_val = params['f']['val']
        cx = params['cx']['val']
        cy = params['cy']['val']
        
        K = np.array([
            [f_val, 0, cx],
            [0, f_val, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # 3. 提取畸变系数 (OpenCV 5参数模型: k1, k2, p1, p2, k3)
        dist_coeffs = np.array([
            params['k1']['val'],
            params['k2']['val'],
            params['p1']['val'],
            params['p2']['val'],
            params['k3']['val']
        ], dtype=np.float64)

        # 4. 提取外参 (W2C: Rotation & Translation)
        # 警告：必须确认 json 中的 rx,ry,rz 是 Rodrigues 向量
        trans_data = cam_entry['transform']
        rx = trans_data['rotation']['rx']
        ry = trans_data['rotation']['ry']
        rz = trans_data['rotation']['rz']
        tx = trans_data['translation']['x']
        ty = trans_data['translation']['y']
        tz = trans_data['translation']['z']

        # 构造旋转向量 (Rodrigues) 和平移向量
        # 因为是 W2C，直接使用即可
        r_vec = np.array([rx, ry, rz], dtype=np.float64)
        T = np.array([tx, ty, tz], dtype=np.float64).reshape(3, 1)
        









        # 可选：计算旋转矩阵 R 用于调试，但 projectPoints 直接用 r_vec 更快
        R, _ = cv2.Rodrigues(r_vec)












        cameras_data.append({
            'K': K,
            'D': dist_coeffs,
            'R': R,     # 旋转矩阵 (W2C)
            'T': T,     # 平移向量 (W2C)
            'r_vec': r_vec, # 旋转向量 (W2C)
            'width': width,
            'height': height
        })

    return cameras_data

def define_voxel_grid(bounds, resolution):
    """
    创建一个 3D 坐标网格
    bounds: [min_x, max_x, min_y, max_y, min_z, max_z]
    """
    x = np.arange(bounds[0], bounds[1], resolution)
    y = np.arange(bounds[2], bounds[3], resolution)
    z = np.arange(bounds[4], bounds[5], resolution)
    
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
    voxel_points = np.vstack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T
    return voxel_points

def carve_voxels(voxel_points, cameras, mask_folder):
    """
    核心雕刻算法
    """
    # 假设所有点初始都在物体内部
    is_occupied = np.ones(voxel_points.shape[0], dtype=bool)

    # 获取所有掩码文件
    # 注意：请确保文件名排序与 JSON 中的 cameras 顺序一致 (如 000.png, 001.png)
    mask_files = sorted(glob.glob(os.path.join(mask_folder, "*.png")))
    
    if len(mask_files) == 0:
        print(f"错误: 在 {mask_folder} 未找到 .png 掩码图片。")
        return None

    print(f"开始雕刻... 初始体素数量: {len(voxel_points)}")

    # 遍历每个相机
    for i, cam in enumerate(cameras):
        if i >= len(mask_files):
            print("警告: 掩码数量少于相机数量，停止雕刻。")
            break
            
        print(f"处理视角 {i+1}/{len(cameras)} ...")
        
        # 读取掩码
        mask = cv2.imread(mask_files[i], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"无法读取图片: {mask_files[i]}")
            continue
            
        # 检查尺寸匹配，如果不匹配则缩放
        target_w, target_h = cam['width'], cam['height']
        if mask.shape[1] != target_w or mask.shape[0] != target_h:
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # 二值化 (假设白色 255 是前景)
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 优化：只计算当前剩余的体素
        current_indices = np.where(is_occupied)[0]
        if len(current_indices) == 0:
            print("所有体素已被雕刻殆尽，空间设置可能不正确。")
            break
        
        current_points = voxel_points[current_indices]

        # 投影 3D 点 -> 2D 像素坐标
        # 输入：World Points, W2C rvec, W2C tvec, K, Dist
        image_points, _ = cv2.projectPoints(
            current_points, 
            cam['r_vec'], 
            cam['T'], 
            cam['K'], 
            cam['D']
        )
        
        image_points = image_points.reshape(-1, 2)

        # 1. 剔除投影在图像外的点
        valid_x = np.logical_and(image_points[:, 0] >= 0, image_points[:, 0] < target_w)
        valid_y = np.logical_and(image_points[:, 1] >= 0, image_points[:, 1] < target_h)
        valid_indices = np.logical_and(valid_x, valid_y)

        # 2. 查询掩码值
        pixel_values = np.zeros(len(image_points), dtype=np.uint8)
        img_x = image_points[valid_indices, 0].astype(int)
        img_y = image_points[valid_indices, 1].astype(int)
        
        # OpenCV 图像索引是 [y, x]
        pixel_values[valid_indices] = binary_mask[img_y, img_x]

        # 如果点在图像内且掩码值为 0 (背景)，则该体素应被移除 (False)
        # 逻辑：保留 (在图像外) 或 (在图像内且是前景) 的点
        # 这里采用严格的 Visual Hull：如果投影在背景上，就雕刻掉。
        # 如果投影在图像外，通常保守起见认为是物体（避免 FOV 裁剪掉物体）。
        
        # 掩码测试：如果在图像内 (valid_indices 为 True) 且 像素为黑 (pixel_values == 0)，则剔除
        to_remove = np.logical_and(valid_indices, pixel_values == 0)
        
        # 更新状态
        is_occupied[current_indices] = np.logical_and(is_occupied[current_indices], ~to_remove)

    return voxel_points[is_occupied]

def compute_normals(points, k=10):
    """使用邻域PCA估计法向量。
    对每个点，取最近k个邻居，计算协方差矩阵的最小特征向量作为法向。
    当点数较少或稀疏时，法向可能不稳定。
    """
    if points is None or len(points) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    try:
        from scipy.spatial import cKDTree
    except Exception:
        # 无scipy时退化为零法向
        return np.zeros((len(points), 3), dtype=np.float64)

    tree = cKDTree(points)
    normals = np.zeros((len(points), 3), dtype=np.float64)

    for i, p in enumerate(points):
        # 查询最近邻（包含自身）
        _, idxs = tree.query(p, k=min(k, len(points)))
        neigh = points[idxs]
        # 去中心化
        c = neigh.mean(axis=0)
        X = neigh - c
        # 协方差与特征分解
        C = X.T @ X
        w, v = np.linalg.eigh(C)
        # 最小特征值对应的特征向量
        n = v[:, np.argmin(w)]
        # 归一化
        n = n / (np.linalg.norm(n) + 1e-12)
        normals[i] = n

    return normals

def sample_colors_from_view(points, cam, image_path):
    """将点投影到指定视图，采样颜色(BGR->RGB)。投影到图外的点使用默认灰色。"""
    # 默认颜色
    colors = np.full((len(points), 3), 128, dtype=np.uint8)

    img = cv2.imread(image_path)
    if img is None:
        return colors

    h, w = img.shape[:2]
    image_points, _ = cv2.projectPoints(points, cam['r_vec'], cam['T'], cam['K'], cam['D'])
    uv = image_points.reshape(-1, 2)

    # 有效索引
    valid_x = np.logical_and(uv[:, 0] >= 0, uv[:, 0] < w)
    valid_y = np.logical_and(uv[:, 1] >= 0, uv[:, 1] < h)
    valid = np.logical_and(valid_x, valid_y)

    xi = uv[valid, 0].astype(int)
    yi = uv[valid, 1].astype(int)
    # BGR -> RGB
    sampled = img[yi, xi][:, ::-1]
    colors[valid] = sampled
    return colors

def save_point_cloud_with_attributes(points, normals=None, colors=None, filename="output.ply"):
    """保存包含坐标、法向与颜色的PLY，满足用户指定格式。
    - 坐标与法向使用double
    - 颜色使用uchar
    """
    n_pts = len(points)
    if normals is None or len(normals) != n_pts:
        normals = np.zeros((n_pts, 3), dtype=np.float64)
    if colors is None or len(colors) != n_pts:
        colors = np.full((n_pts, 3), 128, dtype=np.uint8)

    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n_pts}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "property double nx\n"
        "property double ny\n"
        "property double nz\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    with open(filename, 'w') as f:
        f.write(header)
        for (x, y, z), (nx, ny, nz), (r, g, b) in zip(points, normals, colors):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {nx:.6f} {ny:.6f} {nz:.6f} {int(r)} {int(g)} {int(b)}\n")
    print(f"点云已保存至 {filename} (点数: {n_pts})")

# ================= 配置与运行 =================

# 1. 设置路径
json_file_path = "undistorted_pha_uniform/calib_uniform.json"  # 修改为你的实际 JSON 文件名
mask_folder_path = "binary_masks"      # 存放 .png 掩码的文件夹

# 2. 设置体素空间 (Bounding Box)
# 重要：请确认你的物体位于此范围内。
# 如果 json 中的 T 是 W2C (t = -R * C)，且值很大，说明相机很远。
# 这里的 bounds 应该包含物体在世界坐标系中的位置。
# 如果物体在世界原点 (0,0,0) 附近，请使用 [-0.5, 0.5] 之类的范围。
bounds = [-0.4, 0.3,  # min_x, max_x
          -0.2, 0.38,  # min_y, max_y
          0.8, 1.1]  # min_z, max_z

# 3. 设置精度
step_size = 0.005

if __name__ == "__main__":
    if not os.path.exists(json_file_path):
        print(f"找不到文件: {json_file_path}")
    else:
        # 加载数据
        cameras = load_calibration(json_file_path)
        print(f"成功加载 {len(cameras)} 个相机参数")

        # 初始化体素
        voxels = define_voxel_grid(bounds, step_size)
        
        # 检查掩码文件夹
        if not os.path.exists(mask_folder_path):
            print(f"请创建文件夹 '{mask_folder_path}' 并放入对应的掩码图片")
        else:
            # 执行体素雕刻
            final_points = carve_voxels(voxels, cameras, mask_folder_path)

            # 保存（带法向与颜色）
            if final_points is not None and len(final_points) > 0:
                # 1) 估计法向
                normals = compute_normals(final_points, k=12)
                # 2) 使用第一视角采样颜色（你也可以改为更合适的视角）
                rectified_dir = "undistorted_pha_uniform/rectified"
                first_img = None
                if os.path.isdir(rectified_dir):
                    jpgs = sorted(glob.glob(os.path.join(rectified_dir, "*.jpg")))
                    if jpgs:
                        first_img = jpgs[0]
                colors = None
                if first_img is not None:
                    colors = sample_colors_from_view(final_points, cameras[0], first_img)
                
                save_point_cloud_with_attributes(final_points, normals, colors, "reconstruction.ply")
            else:
                print("重建结果为空，请检查 bounds 设置是否覆盖了物体位置。")