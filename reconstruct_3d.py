import os

import numpy as np
import imageio

import open3d as o3d
from matplotlib import pyplot as plt



def estimate_extrinsic_from_ray_depth(ray_path: str,
                                      depth_path: str,
                                      agent_settings: dict,
                                      forward_uv=None,
                                      up_uv=None,
                                      right_uv=None):
    # load the ray image
    ray = imageio.imread(ray_path).astype(np.float32) / 255.0
    ray = ray * 2.0 - 1.0
    # normolize
    norms = np.linalg.norm(ray, axis=2, keepdims=True) + 1e-8
    ray = ray / norms

    depth = imageio.imread(depth_path).astype(np.float32)

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
    y_new = np.array([0, 1, 0], dtype=np.float32)  # 世界竖直方向
    f_new = f / np.linalg.norm(f)                 # 保持前向
    r_new = np.cross(y_new, f_new)                 # r = y × f
    r_new /= np.linalg.norm(r_new)                 # 归一化
    u_new = np.cross(f_new, r_new)                 # u = f × r，确保右手系

    R = np.vstack([r_new, u_new, f_new])           # 更新 R
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

def rotate_extrinsic(extrinsic, yaw_deg=0, pitch_deg=0, roll_deg=0):
    """
    Rotate the camera extrinsic by given yaw (left-right), pitch (up-down), and roll (tilt) angles in degrees.
    Rotation is applied relative to the camera's current orientation.

    Args:
        extrinsic (np.ndarray): (4,4) world→camera extrinsic matrix.
        yaw_deg (float): Rotation around Y-axis (left/right head turn).
        pitch_deg (float): Rotation around X-axis (up/down head nod).
        roll_deg (float): Rotation around Z-axis (head tilt).

    Returns:
        np.ndarray: New (4,4) extrinsic matrix after rotation.
    """

    # 1. Invert to get camera → world
    cam2world = np.linalg.inv(extrinsic)

    # 2. Build individual rotation matrices
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)

    R_yaw = np.array([
        [ np.cos(yaw), 0, np.sin(yaw)],
        [ 0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)]
    ])

    R_roll = np.array([
        [ np.cos(roll), -np.sin(roll), 0],
        [ np.sin(roll),  np.cos(roll), 0],
        [0, 0, 1]
    ])

    # 3. Combine rotations (roll -> pitch -> yaw order)
    R_combined = R_yaw @ R_pitch @ R_roll

    # 4. Apply to cam2world
    R_old = cam2world[:3, :3]
    t_old = cam2world[:3, 3]
    R_new = R_old @ R_combined
    cam2world_new = np.eye(4)
    cam2world_new[:3, :3] = R_new
    cam2world_new[:3, 3] = t_old
    # 5. Invert back to get new world → camera
    extrinsic_new = np.linalg.inv(cam2world_new)
    return extrinsic_new

def visualize_with_extrinsic(ply_path, agent_settings, extrinsic,save_path):
    """
    Visualize a point cloud using the given extrinsic (world→camera) and intrinsic in Open3D.
    """
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_path)

    # Build camera intrinsics
    intr = agent_settings['intrinsic']
    pinhole = o3d.camera.PinholeCameraIntrinsic(
        intr['width'], intr['height'],
        intr['fx'], intr['fy'],
        intr['cx'], intr['cy']
    )

    # Setup visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=intr['width'], height=intr['height'], visible=True)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    # extr_left = rotate_extrinsic(extrinsic, yaw_deg=-15)
    # extr_right = rotate_extrinsic(extrinsic, yaw_deg=15)

    # Apply camera parameters
    cam_param = o3d.camera.PinholeCameraParameters()
    cam_param.intrinsic = pinhole
    cam_param.extrinsic =  extrinsic
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(cam_param, allow_arbitrary=True)
    ctr.set_zoom(0.33)


    ctr.rotate(60.0, 0.0)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(f"{save_path}/right.png", do_render=True)

    ctr.convert_from_pinhole_camera_parameters(cam_param, allow_arbitrary=True)
    vis.poll_events()
    vis.update_renderer()

    ctr.rotate(-60.0, 0.0)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(f"{save_path}/left.png", do_render=True)

    # Check
    # print(">> Applied extrinsic:\n", extrinsic)
    # cp2 = ctr.convert_to_pinhole_camera_parameters()
    # print(">> Current viewer extrinsic:\n", np.array(cp2.extrinsic))
    return

def visualize_with_offrendering(ply_path, agent_settings, extrinsic, output_path=None, show=False):
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_path)

    # Build camera intrinsics
    intr = agent_settings['intrinsic']
    width, height = intr['width'], intr['height']
    pinhole = o3d.camera.PinholeCameraIntrinsic(
        width, height,
        intr['fx'], intr['fy'],
        intr['cx'], intr['cy']
    )
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"  # no lighting, faster
    mat.point_size = 2.0  # tweak for visibility
    renderer.scene.add_geometry("pcd", pcd, mat)

    renderer.setup_camera(pinhole, extrinsic)

    o3d_img = renderer.render_to_image()
    img = np.asarray(o3d_img)  # H×W×3 uint8

    # 7) optional: save to disk
    if output_path:
        o3d.io.write_image(output_path, o3d_img)

    # 8) optional: show inline with matplotlib
    if show:
        plt.figure(figsize=(width / 100, height / 100), dpi=100)
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return img

def get_different_view(view_path,save_path):
    ray_path = None
    depth_path = None
    ply_path = None

    agent_settings = {
        'height': 1.5,
        'intrinsic': {
            'width': 800, 'height': 600,
            'fx': 400.0, 'fy': 400.0,
            'cx': 400.0, 'cy': 300.0
        }
    }

    for file in os.listdir(view_path):
        lower_file = file.lower()
        if "ray" in lower_file and lower_file.endswith(".png"):
            ray_path = os.path.join(view_path, file)
        elif "depth" in lower_file and lower_file.endswith(".png"):
            depth_path = os.path.join(view_path, file)
        elif lower_file.endswith(".ply"):
            ply_path = os.path.join(view_path, file)

    if ray_path is None or depth_path is None:
        raise FileNotFoundError("Ray or depth PNG not found in the given folder!")

    # 2. Estimate extrinsic
    extr = estimate_extrinsic_from_ray_depth(
        ray_path,
        depth_path,
        agent_settings
    )
    visualize_with_extrinsic(
        ply_path=ply_path,
        agent_settings=agent_settings,
        extrinsic=extr,
        save_path=save_path
    )
    return

get_different_view("data/output/view_002","data/output/view_002")
# img = visualize_with_offrendering(
#     "data/output/view_001/view_001.ply",
#     agent_settings,
#     extr,
#     output_path="view.png",
#     show=True,
# )