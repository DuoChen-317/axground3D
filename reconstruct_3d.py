import os

import imageio
import numpy as np
import open3d as o3d


class ViewRenderer:
    def __init__(self, ply, depth, rays):
        """
        Initialize the viewer with point cloud and camera parameters.
        """
        agent_settings = {
            'height': 1.5,
            'intrinsic': {
                'width': 512, 'height': 512,
                'fx': 256.0, 'fy': 256.0,
                'cx': 256.0, 'cy': 256.0
            }
        }
        self.agent_settings = agent_settings
        self.ply = ply

        # estimate camera extrinsic
        extrinsic = estimate_extrinsic_from_ray_depth(depth=depth,ray=rays,agent_settings=agent_settings)
        # Build camera intrinsics
        intrinsic = agent_settings['intrinsic']
        pinhole = o3d.camera.PinholeCameraIntrinsic(
            intrinsic['width'], intrinsic['height'],
            intrinsic['fx'], intrinsic['fy'],
            intrinsic['cx'], intrinsic['cy']
        )
        self.cam_param = o3d.camera.PinholeCameraParameters()
        self.cam_param.intrinsic = pinhole
        self.cam_param.extrinsic = extrinsic

        self.extrinsic = extrinsic

        # Create visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            width=agent_settings['intrinsic']['width'],
            height=agent_settings['intrinsic']['height'],
            visible=False
        )
        # Load point cloud
        self.pcd = ply
        self.vis.add_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
        self.ctr = self.vis.get_view_control()

        self.ctr.convert_from_pinhole_camera_parameters(self.cam_param, allow_arbitrary=True)

    def move_forward(self, distance):
        """
        Move the camera forward along its current front direction by a given distance (in meters).
        """
        # 1. Get camera rotation and translation
        R = self.extrinsic[:3, :3]  # world -> camera rotation
        t = self.extrinsic[:3, 3]  # world -> camera translation

        # 2. Compute the camera front direction
        # Because extrinsic maps world → camera, camera Z axis points to [0,0,1]
        cam_front = R.T @ np.array([0, 0, 1], dtype=np.float32)  # Get camera front in world coords

        # 3. Move camera center along front
        cam_center = -R.T @ t  # recover camera center in world coordinates
        cam_center_new = cam_center + distance * cam_front  # move forward

        # 4. Recompute translation
        t_new = -R @ cam_center_new

        # 5. Update extrinsic
        self.extrinsic[:3, 3] = t_new

    def rotate_extrinsic(self, yaw_deg=0, pitch_deg=0, roll_deg=0):
        """
        Rotate the camera extrinsic by given yaw (left-right), pitch (up-down), and roll (tilt) angles in degrees.
        Rotation is applied relative to the camera's current orientation.
        Returns:
            np.ndarray: New (4,4) extrinsic matrix after rotation.
        """
        # 1. Invert to get camera → world
        cam2world = np.linalg.inv(self.extrinsic)

        # 2. Build individual rotation matrices
        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        roll = np.deg2rad(roll_deg)

        R_yaw = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])

        R_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])

        R_roll = np.array([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
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
        self.extrinsic = extrinsic_new

    def render_view(self):
        self.cam_param.extrinsic = self.extrinsic
        self.ctr.convert_from_pinhole_camera_parameters(self.cam_param, allow_arbitrary=True)
        self.vis.poll_events()
        self.vis.update_renderer()
        img = np.asarray(self.vis.capture_screen_float_buffer(do_render=True))  # float in [0,1]
        img = (img * 255).clip(0, 255).astype(np.uint8)
        return img


    def save_image(self, save_path):
        self.vis.poll_events()
        self.vis.update_renderer()
        self.vis.capture_screen_image(save_path, do_render=True)

    def close(self):
        self.vis.destroy_window()
        return

def estimate_extrinsic_from_ray_depth(ray: np.ndarray,
                                      depth: np.ndarray,
                                      agent_settings: dict,
                                      forward_uv=None,
                                      up_uv=None,
                                      right_uv=None):
    """
    Estimate extrinsic given ray and depth images (already loaded as numpy arrays).

    Args:
        ray (np.ndarray): Ray direction image, shape (H,W,3), value in [-1,1] or [0,255].
        depth (np.ndarray): Depth map, shape (H,W).
        agent_settings (dict): Contains height, intrinsic info.
        forward_uv (tuple): Pixel location (u,v) for front vector.
        up_uv (tuple): Pixel location (u,v) for up vector.
        right_uv (tuple): Pixel location (u,v) for right vector.

    Returns:
        np.ndarray: 4x4 Extrinsic matrix (world → camera)
    """

    # If input ray is [0,255], rescale to [-1,1]
    if ray.max() > 1.5:
        ray = ray.astype(np.float32) / 255.0
        ray = ray * 2.0 - 1.0

    # Normalize ray directions
    norms = np.linalg.norm(ray, axis=2, keepdims=True) + 1e-8
    ray = ray / norms

    H, W, _ = ray.shape

    # 2. Default pixel selection
    if forward_uv is None:
        forward_uv = (W // 2, H // 2)
    if up_uv is None:
        up_uv = (W // 2, H // 4)
    if right_uv is None:
        right_uv = (3 * W // 4, H // 2)

    # 3. Extract direction vectors
    f = ray[forward_uv[1], forward_uv[0]]
    u = ray[up_uv[1], up_uv[0]]
    r = ray[right_uv[1], right_uv[0]]

    # Normalize again
    f /= np.linalg.norm(f)
    u /= np.linalg.norm(u)
    r /= np.linalg.norm(r)

    # 4. Construct rotation matrix R (world → camera)
    y_new = np.array([0, 1, 0], dtype=np.float32)  # Global vertical
    f_new = f / np.linalg.norm(f)
    r_new = np.cross(y_new, f_new)
    r_new /= np.linalg.norm(r_new)
    u_new = np.cross(f_new, r_new)

    R = np.vstack([r_new, u_new, f_new])

    # 5. Translation
    h = agent_settings['height']
    C = np.array([0.0, 0.0, h], dtype=np.float32)
    C = C - f_new * 1.0
    t = - R @ C

    # 6. Assemble extrinsic
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t

    print("Estimated extrinsic (world→camera):")
    print(extrinsic)
    return extrinsic



def test():
    # 1. Load test files
    # Load a test PLY (you should have a ply file already)
    pcd = o3d.io.read_point_cloud("data/scenes/1LXtFkjw3qL/1LXtFkjw3qL.ply")

    # Load test depth and ray images (they can come from UniK3D output or saved .pngs)
    depth = imageio.imread("data/scenes/1LXtFkjw3qL/1LXtFkjw3qL_depth.png").astype(np.float32)
    ray = imageio.imread("data/scenes/1LXtFkjw3qL/1LXtFkjw3qL_rays.png").astype(np.float32)

    # If ray is uint8 0-255, you can preprocess inside the class (already handled)

    # 2. Create renderer
    renderer = ViewRenderer(ply=pcd, depth=depth, rays=ray)

    # 3. Test rendering front view
    renderer.render_view()
    renderer.save_image("original.png")

    # move forward for 1 meter
    renderer.move_forward(1)
    renderer.render_view()
    renderer.save_image("forward.png")

    # # 4. Rotate right 30 degrees and save
    # renderer.rotate_extrinsic(yaw_deg=15)
    # renderer.render_view()
    # renderer.save_image("right30.png")

    # 5. Rotate left 60 degrees and save
    renderer.rotate_extrinsic(yaw_deg=-30)
    renderer.render_view()
    renderer.save_image("left30.png")



test()
