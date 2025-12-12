#!/usr/bin/env python3
"""
Converts a custom camera format file (相机.txt) to COLMAP's
cameras.txt and images.txt format.

Input Format (相机.txt - boujou export):
- Header lines include: '#Image Size W H'
- Then 22 lines follow; each line has 13 numbers:
    R(0,0) ... R(2,2)  Tx Ty Tz  Fpix
    - Rotation R is w2c (applied before translation)
    - Translation T is camera center in world (c2w)
    - F(pix) is focal length in pixels
We parse the first 22 numeric lines and stop.

COLMAP Output Format:
- cameras.txt: Intrinsics in OPENCV format.
- images.txt: Extrinsics in c2w format (Quaternion + Translation).
"""

import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import re

def parse_custom_format(filepath, max_cameras=22):
    """Parse boujou export: read '#Image Size W H' and 13-number lines (R 9 + T 3 + Fpix)."""
    width = None
    height = None
    cameras = []

    def parse_image_size(line: str):
        m = re.match(r"#\s*Image\s*Size\s*(\d+)\s*(\d+)", line)
        return m

    def parse_numeric(line: str):
        parts = line.replace('\t', ' ').split()
        try:
            vals = list(map(float, parts))
            return vals
        except Exception:
            return None

    with open(filepath, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = parse_image_size(line)
            if m:
                width = int(m.group(1))
                height = int(m.group(2))
                continue
            vals = parse_numeric(line)
            if vals is None:
                continue
            if len(vals) == 13:
                R_vals = vals[0:9]
                T_vals = vals[9:12]
                Fpix = vals[12]
                R_w2c = np.array(R_vals).reshape(3, 3)
                T_c2w = np.array(T_vals)
                T_w2c =  -R_w2c @ T_c2w
                if width is None or height is None:
                    cx = 0.0
                    cy = 0.0
                    w = 0
                    h = 0
                else:
                    cx = width / 2.0
                    cy = height / 2.0
                    w = width
                    h = height
                cameras.append({
                    'id': len(cameras) + 1,
                    'intrinsics': {'fx': Fpix, 'fy': Fpix, 'cx': cx, 'cy': cy},
                    'R_w2c': R_w2c,
                    'T_w2c': T_w2c,
                    'size': {'width': w, 'height': h},
                    'name': f"{len(cameras)+1:03d}.png"
                })
                if len(cameras) >= max_cameras:
                    break
    return cameras

def main():
    parser = argparse.ArgumentParser(description="Convert custom camera file to COLMAP format.")
    parser.add_argument('--input', type=str, default='data/相机.txt',
                        help='Path to the input custom camera file (相机.txt).')
    parser.add_argument('--output_dir', type=str, default='colmap_from_custom',
                        help='Directory to save cameras.txt and images.txt.')
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)

    try:
        camera_data = parse_custom_format(args.input, max_cameras=22)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error reading or parsing file: {e}")
        return


    # --- Write images.txt ---
    images_txt_path = output_path / 'images.txt'
    with open(images_txt_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[]\n")
        f.write(f"# Number of images: {len(camera_data)}\n")
        
        for cam in camera_data:
            # Convert R_w2c to R_c2w for COLMAP
            R_w2c = cam['R_w2c']
            
            # T_c2w is already in the correct format
            T_w2c = cam['T_w2c']
            
            # Convert rotation matrix to quaternion (qw, qx, qy, qz)
            quat = R.from_matrix(R_w2c).as_quat() # Returns (x, y, z, w)
            # Reorder to COLMAP's (w, x, y, z)
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            
            tx, ty, tz = T_w2c
            
            f.write(f"{cam['id']} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {cam['id']} {cam['name']}\n")
            # Add a second line with empty points2D for each image
            f.write("\n")

    print(f"Successfully wrote {len(camera_data)} images to {images_txt_path}")
    print("\nConversion complete.")

if __name__ == '__main__':
    main()