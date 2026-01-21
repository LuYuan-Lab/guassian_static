import os
import shutil
from pathlib import Path

def merge_images_from_subfolders(source_dir, output_dir):
    """
    将源目录下所有子文件夹中的图片文件合并到一个输出文件夹中
    
    Args:
        source_dir: 包含子文件夹的源目录
        output_dir: 输出目录，用于存放所有图片
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的图片格式
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
    
    # 统计信息
    total_copied = 0
    total_skipped = 0
    
    # 遍历源目录下的所有子文件夹
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"错误: 源目录不存在: {source_dir}")
        return
    
    print(f"正在扫描目录: {source_dir}")
    print(f"输出目录: {output_dir}\n")
    
    # 获取所有子文件夹
    subfolders = [f for f in source_path.iterdir() if f.is_dir()]
    
    if not subfolders:
        print("未找到任何子文件夹")
        return
    
    print(f"找到 {len(subfolders)} 个子文件夹\n")
    
    # 遍历每个子文件夹
    for subfolder in subfolders:
        print(f"正在处理: {subfolder.name}")
        
        # 遍历子文件夹中的所有文件
        for file in subfolder.iterdir():
            if file.is_file() and file.suffix.lower() in image_extensions:
                # 构建目标文件路径
                dest_file = Path(output_dir) / file.name
                
                # 如果目标文件已存在，添加子文件夹名称前缀
                if dest_file.exists():
                    new_name = f"{subfolder.name}_{file.name}"
                    dest_file = Path(output_dir) / new_name
                    print(f"  - 文件重名，重命名为: {new_name}")
                
                # 复制文件
                try:
                    shutil.copy2(file, dest_file)
                    total_copied += 1
                    print(f"  ✓ 复制: {file.name}")
                except Exception as e:
                    print(f"  ✗ 复制失败: {file.name} - {e}")
                    total_skipped += 1
    
    # 输出统计信息
    print(f"\n{'='*50}")
    print(f"处理完成!")
    print(f"成功复制: {total_copied} 个文件")
    print(f"跳过/失败: {total_skipped} 个文件")
    print(f"输出目录: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    # 设置源目录和输出目录
    source_directory = r"2026-1-18\Plate"
    output_directory = r"Plate_Merged"
    
    # 执行合并
    merge_images_from_subfolders(source_directory, output_directory)
