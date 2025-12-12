from scipy.spatial.transform import Rotation as R
from plyfile import PlyData
import numpy as np
import cv2

'''qw, qx, qy, qz, tx, ty, tz= 0.99894581808757521, -0.035958291913691119, -0.027786545560654685, -0.0064932005822049303, 1.9000352301859911, 0.55986166786044844, 1.8994342765064418 
rot1 = R.from_quat([qx, qy, qz, qw])


t1w2c = np.array([tx, ty, tz])
R1w2c = rot1.as_matrix()

qw, qx, qy, qz, tx, ty, tz= 0.91137699069591982, -0.40264762515576347, -0.0052151084425394777, -0.085086270516492729, 1.6739137800531352, -3.5961183517784847, 2.538859933421211
rot2 = R.from_quat([qx, qy, qz, qw])
t2w2c = np.array([tx, ty, tz])
R2w2c = rot2.as_matrix()

points_world = np.array([-3.13809484483791, -1.2018260864490522, 4.5156040419232344 ]).reshape(1,3)


rx,ry,rz = 1.9213556704511046,0.41406012744771853,0.096482349573763
rot = R.from_euler('xyz', [rx, ry, rz])
Rc2w = rot.as_matrix()
tx, ty, tz = -0.2509364195412356,0.8462314181470894,1.3122698313034937
tc2w = np.array([tx, ty, tz])

Rw2c = Rc2w
tw2c =  tc2w
'''
r1, r2, r3, r4, r5, r6, r7, r8, r9, tx, ty, tz = 8.8576023638361467e-01, -1.7523452900635933e-01, 4.2979258193407949e-01, 3.3443676327284239e-01, -4.0112786018618862e-01, -8.5278865561992179e-01, 3.2183979712463845e-01, 8.9910472116787910e-01, -2.9669823956404412e-01, -2.1842048827883875e-01, -8.0024718406773365e-01, 1.2381809885415518e+00
Rw2c = np.array([[r1, r2, r3],
                 [r4, r5, r6],
                 [r7, r8, r9]])
tc2w = np.array([tx, ty, tz])
tw2c = -Rw2c @ tc2w


fx,fy, cx, cy, w, h =  7055.150067559398, 7117.406429106748, 2866.1322037458817, 1672.681005738936, 5760, 3240


plydata = PlyData.read("data/liang.ply")
v= plydata['vertex']
points = np.vstack([v['x'], v['y'], v['z']]).T #points is Nx3
print(points)
points_cameras = (Rw2c@points.T).T + tw2c  # Nx3

Z = points_cameras[:, 2]
mask_front = Z > 0
pts_cam = points_cameras[mask_front]

x_norm = pts_cam[:, 0] / pts_cam[:, 2]
y_norm = pts_cam[:, 1] / pts_cam[:, 2]


u = fx * x_norm + cx
v = fy * y_norm + cy

uv = np.stack([u, v], axis=1) # Mx2
mask_in = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
uv_in = uv[mask_in]
print(f"投影到图像内的点数量: {uv_in.shape[0]} / {points.shape[0]}")
uv_int = np.round(uv_in).astype(np.int32)

img = np.zeros((h, w, 3), dtype=np.uint8)
for (u, v) in uv_int:
    cv2.circle(img, (int(u), int(v)), 2, (255, 255, 0), -1)
cv2.imwrite("agisoft/output/projection_overlay2.png", img)