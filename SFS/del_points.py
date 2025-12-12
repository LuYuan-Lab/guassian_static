import numpy as np
import os

def load_ply_as_numpy(filename):
    """
    PLY 读取器，读取顶点坐标 (x, y, z), 法向量 (nx, ny, nz) 和颜色 (r, g, b)
    返回: (points, normals, colors) 或者 (points, None, None) 如果没有法向量和颜色
    """
    points = []
    normals = []
    colors = []
    reading_points = False
    has_normals = False
    has_colors = False
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 解析头部，确定是否包含法向量和颜色
    for line in lines:
        line = line.strip()
        if "property" in line and "nx" in line:
            has_normals = True
        elif "property" in line and "red" in line:
            has_colors = True
        elif "end_header" in line:
            reading_points = True
            continue
        
        if reading_points:
            # 读取数据行
            parts = line.split()
            if len(parts) >= 3:
                try:
                    # 坐标 (必须)
                    pt = [float(parts[0]), float(parts[1]), float(parts[2])]
                    points.append(pt)
                    
                    # 法向量 (如果有)
                    if has_normals and len(parts) >= 6:
                        normal = [float(parts[3]), float(parts[4]), float(parts[5])]
                        normals.append(normal)
                    
                    # 颜色 (如果有)
                    if has_colors:
                        if has_normals and len(parts) >= 9:
                            # 有法向量: x y z nx ny nz r g b
                            color = [int(parts[6]), int(parts[7]), int(parts[8])]
                        elif not has_normals and len(parts) >= 6:
                            # 无法向量: x y z r g b
                            color = [int(parts[3]), int(parts[4]), int(parts[5])]
                        else:
                            color = [128, 128, 128]  # 默认灰色
                        colors.append(color)
                    
                except (ValueError, IndexError):
                    continue
    
    points = np.array(points)
    normals = np.array(normals) if normals else None
    colors = np.array(colors) if colors else None
    
    return points, normals, colors

def save_ply(points, filename, normals=None, colors=None):
    """
    保存点云为 .ply 格式，包含法向量和颜色
    """
    n_pts = len(points)
    
    # 如果没有法向量，生成零法向量
    if normals is None:
        normals = np.zeros((n_pts, 3), dtype=np.float64)
    
    # 如果没有颜色，生成默认灰色
    if colors is None:
        colors = np.full((n_pts, 3), 128, dtype=np.uint8)
    
    # 确保数组维度正确
    if len(normals) != n_pts:
        normals = np.zeros((n_pts, 3), dtype=np.float64)
    if len(colors) != n_pts:
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
        for i in range(n_pts):
            x, y, z = points[i]
            nx, ny, nz = normals[i]
            r, g, b = colors[i]
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {nx:.6f} {ny:.6f} {nz:.6f} {int(r)} {int(g)} {int(b)}\n")
    
    print(f"文件已保存至: {filename}")
    print(f"包含 {n_pts} 个点，带法向量和颜色信息")

def compute_normals_for_surface(points, k=12):
    """
    为表面点计算法向量
    使用邻域PCA方法估计法向量
    """
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("警告: 未安装scipy，使用零法向量")
        return np.zeros((len(points), 3), dtype=np.float64)
    
    print("正在计算表面法向量...")
    tree = cKDTree(points)
    normals = np.zeros((len(points), 3), dtype=np.float64)
    
    for i, p in enumerate(points):
        # 查询最近邻（包含自身）
        _, idxs = tree.query(p, k=min(k, len(points)))
        if len(idxs.shape) == 0:  # 单个点
            idxs = [idxs]
        
        neigh = points[idxs]
        
        # 去中心化
        center = neigh.mean(axis=0)
        centered = neigh - center
        
        # 计算协方差矩阵
        if len(centered) > 1:
            cov_matrix = centered.T @ centered
            # 特征分解
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            # 最小特征值对应的特征向量作为法向量
            normal = eigenvecs[:, np.argmin(eigenvals)]
            # 归一化
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normals[i] = normal / norm
            else:
                normals[i] = np.array([0, 0, 1])  # 默认向上
        else:
            normals[i] = np.array([0, 0, 1])  # 默认向上
        
        if (i + 1) % 1000 == 0:
            print(f"  处理进度: {i + 1}/{len(points)}")
    
    print("法向量计算完成")
    return normals

def generate_surface_colors(points, method='height'):
    """
    为表面点生成颜色
    method: 'height' - 基于高度渐变, 'random' - 随机颜色, 'uniform' - 统一颜色
    """
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    
    if method == 'height':
        # 基于Z坐标的高度颜色映射
        z_coords = points[:, 2]
        z_min, z_max = z_coords.min(), z_coords.max()
        
        if z_max > z_min:
            # 归一化到 [0, 1]
            normalized_z = (z_coords - z_min) / (z_max - z_min)
            
            # 使用彩虹颜色映射 (蓝->绿->红)
            for i, nz in enumerate(normalized_z):
                if nz < 0.5:
                    # 蓝色到绿色
                    r = 0
                    g = int(255 * (nz * 2))
                    b = int(255 * (1 - nz * 2))
                else:
                    # 绿色到红色
                    r = int(255 * ((nz - 0.5) * 2))
                    g = int(255 * (1 - (nz - 0.5) * 2))
                    b = 0
                
                colors[i] = [r, g, b]
        else:
            colors[:] = [128, 128, 128]  # 灰色
            
    elif method == 'random':
        # 随机颜色
        np.random.seed(42)  # 固定种子保证重现性
        colors = np.random.randint(0, 256, (len(points), 3), dtype=np.uint8)
        
    else:  # uniform
        # 统一颜色 (浅蓝色)
        colors[:] = [100, 150, 200]
    
    return colors
    """
    自动估算点云的分辨率（步长）
    方法：计算第一个点到其他所有点的距离，取最小的非零距离
    """
    # 取前100个点进行采样，避免计算量过大
    sample_size = min(len(points), 100)
    sample_points = points[:sample_size]
    
    min_dists = []
    for i in range(sample_size):
        # 计算当前点到所有采样点的距离
        diff = np.abs(sample_points - sample_points[i])
        dist = np.sum(diff, axis=1) # 曼哈顿距离近似，或者直接看轴差距
        # 排除自己 (0距离)
        dist = dist[dist > 1e-6]
        if len(dist) > 0:
            min_dists.append(np.min(dist))
            
    if not min_dists:
        return 0.01 # 默认回退值
    
    # 取中位数作为估算的步长
    estimated_res = np.median(min_dists)
    print(f"自动检测到的体素步长(Resolution): {estimated_res:.6f}")
    return estimated_res

def estimate_resolution(points):
    """
    自动估算点云的分辨率（步长）
    方法：计算第一个点到其他所有点的距离，取最小的非零距离
    """
    # 取前100个点进行采样，避免计算量过大
    sample_size = min(len(points), 100)
    sample_points = points[:sample_size]
    
    min_dists = []
    for i in range(sample_size):
        # 计算当前点到所有采样点的距离
        diff = np.abs(sample_points - sample_points[i])
        dist = np.sum(diff, axis=1) # 曼哈顿距离近似，或者直接看轴差距
        # 排除自己 (0距离)
        dist = dist[dist > 1e-6]
        if len(dist) > 0:
            min_dists.append(np.min(dist))
            
    if not min_dists:
        return 0.01 # 默认回退值
    
    # 取中位数作为估算的步长
    estimated_res = np.median(min_dists)
    print(f"自动检测到的体素步长(Resolution): {estimated_res:.6f}")
    return estimated_res

def extract_surface_from_solid(points, normals=None, colors=None):
    """
    核心剥皮算法，同时处理法向量和颜色
    """
    print(f"正在处理 {len(points)} 个点...")
    
    # 1. 估算步长
    res = estimate_resolution(points)
    
    # 2. 坐标整数化 (Grid Coordinate)
    grid_indices = np.round(points / res).astype(np.int32)
    
    # 3. 建立哈希表用于快速查找
    point_set = set(map(tuple, grid_indices))
    
    surface_mask = []
    
    # 4. 6邻域偏移量
    neighbors_offset = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    
    for idx_tuple in grid_indices:
        x, y, z = idx_tuple
        is_inner = True
        
        # 检查 6 个方向
        for dx, dy, dz in neighbors_offset:
            neighbor = (x + dx, y + dy, z + dz)
            if neighbor not in point_set:
                # 只要发现有一个方向没有邻居，说明这个面暴露在空气中
                is_inner = False
                break
        
        # 如果不是内部点，就是表面点
        surface_mask.append(not is_inner)
    
    # 5. 应用掩码提取表面点
    surface_mask = np.array(surface_mask)
    surface_points = points[surface_mask]
    
    # 提取对应的法向量和颜色
    surface_normals = None
    surface_colors = None
    
    if normals is not None and len(normals) == len(points):
        surface_normals = normals[surface_mask]
    
    if colors is not None and len(colors) == len(points):
        surface_colors = colors[surface_mask]
    
    print(f"处理完成！")
    print(f"原始点数: {len(points)}")
    print(f"剩余表面点数: {len(surface_points)}")
    print(f"移除了 {len(points) - len(surface_points)} 个内部点")
    
    return surface_points, surface_normals, surface_colors

# ================= 使用方法 =================

# 将此处替换为你生成的实心点云文件路径
input_file = "reconstruction.ply" 
output_file = "surface_only.ply"

if __name__ == "__main__":
    if os.path.exists(input_file):
        # 1. 读取PLY文件（包含法向量和颜色）
        pts, input_normals, input_colors = load_ply_as_numpy(input_file)
        
        if len(pts) > 0:
            print(f"输入点云: {len(pts)} 个点")
            if input_normals is not None:
                print(f"包含法向量: {len(input_normals)} 个")
            if input_colors is not None:
                print(f"包含颜色: {len(input_colors)} 个")
            
            # 2. 执行表面提取
            surface_pts, surface_normals, surface_colors = extract_surface_from_solid(
                pts, input_normals, input_colors
            )
            
            # 3. 如果没有输入法向量，为表面点计算法向量
            if surface_normals is None and len(surface_pts) > 0:
                print("输入文件无法向量，正在为表面点计算法向量...")
                surface_normals = compute_normals_for_surface(surface_pts)
            
            # 4. 如果没有输入颜色，生成颜色
            if surface_colors is None and len(surface_pts) > 0:
                print("输入文件无颜色，正在生成基于高度的颜色...")
                surface_colors = generate_surface_colors(surface_pts, method='height')
            
            # 5. 保存结果
            if len(surface_pts) > 0:
                save_ply(surface_pts, output_file, surface_normals, surface_colors)
            else:
                print("警告: 表面提取后没有剩余点")
        else:
            print("点云文件是空的。")
    else:
        print(f"找不到文件: {input_file}")