import numpy as np
import imageio

import open3d as o3d
from open3d.visualization import gui, rendering


def estimate_extrinsic_from_ray_depth(ray_path: str,
                                      depth_path: str,
                                      agent_settings: dict,
                                      forward_uv=None,
                                      up_uv=None,
                                      right_uv=None):
    # 1. 读图 & 解码
    ray = imageio.imread(ray_path).astype(np.float32) / 255.0
    # 假设映射方式 (d+1)/2 -> [0,1]，因此反推
    ray = ray * 2.0 - 1.0
    # 归一化每个方向向量
    norms = np.linalg.norm(ray, axis=2, keepdims=True) + 1e-8
    ray = ray / norms

    depth = imageio.imread(depth_path).astype(np.float32)
    # 如果你的深度图是可视化过的 heatmap，需要先还原为真实深度。
    # 假设这里 depth 已经是米为单位的真实深度。

    H, W, _ = ray.shape

    # 2. 默认像素
    if forward_uv is None:
        forward_uv = (W//2, H//2)
    if up_uv is None:
        up_uv = (W//2, H//4)
    if right_uv is None:
        right_uv = (3*W//4, H//2)

    # 3. 提取方向向量
    f = ray[forward_uv[1], forward_uv[0]]
    u = ray[up_uv[1],      up_uv[0]]
    r = ray[right_uv[1],   right_uv[0]]
    # 再归一化
    f /= np.linalg.norm(f)
    u /= np.linalg.norm(u)
    r /= np.linalg.norm(r)

    # 4. 构造旋转矩阵 R：world→camera
    #    使 R @ world_vec = cam_vec, 而相机轴 X,Y,Z 分别对应 r,u,f
    R = np.vstack([r, u, f])  # shape (3,3)

    # 5. 计算平移 t = -R @ C_world, C_world = (0,0,height)
    h = agent_settings['height']
    C = np.array([0.0, 0.0, h], dtype=np.float32)
    t = - R @ C

    # 6. 拼 extrinsic
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3,:3] = R
    extrinsic[:3, 3] = t

    print("Estimated extrinsic (world→camera):")
    print(extrinsic)
    return extrinsic


def visualize_with_extrinsic(ply_path, agent_settings, extrinsic):
    """
    用给定的 extrinsic（world→camera）和内参在 Open3D 中可视化点云。
    """
    # 读取点云
    pcd = o3d.io.read_point_cloud(ply_path)

    # 构造 PinholeCameraIntrinsic
    intr = agent_settings['intrinsic']
    pinhole = o3d.camera.PinholeCameraIntrinsic(
        intr['width'], intr['height'],
        intr['fx'], intr['fy'],
        intr['cx'], intr['cy']
    )

    # 1) GUI 可视化
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        width=intr['width'], height=intr['height'],
        visible=False  # 如果你在本地有显示器可以设为 True
    )
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()

    cam_param = o3d.camera.PinholeCameraParameters()
    cam_param.intrinsic = pinhole
    cam_param.extrinsic = extrinsic

    print(">> 传入的 extrinsic：\n", extrinsic)
    ctr.convert_from_pinhole_camera_parameters(cam_param)
    # 立即回读，看 Open3D 真正给存的是哪个
    cp2 = ctr.convert_to_pinhole_camera_parameters()
    print(">> Vis 中当前 extrinsic：\n", np.array(cp2.extrinsic))

    vis.poll_events()
    vis.update_renderer()
    vis.run()



agent_settings = {
        'height': 1.5,
        'intrinsic': {
            'width':  800, 'height': 600,
            'fx': 400.0, 'fy': 400.0,
            'cx': 400.0, 'cy': 300.0
        }
    }
extr = estimate_extrinsic_from_ray_depth(
        "data/output/view_001/view_001_rays.png",
        "data/output/view_001/view_001_depth.png",
        agent_settings
    )
visualize_with_extrinsic("data/output/view_001/view_001.ply", agent_settings, extr)