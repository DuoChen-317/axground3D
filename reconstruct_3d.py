import argparse
import json
import os.path
import time
import numpy as np
import open3d as o3d

def estimate_intrinsics(width, height, fov_deg=60):
    import math
    fov = math.radians(fov_deg)
    fx = fy = 0.5 * width / math.tan(fov / 2)
    cx = width / 2
    cy = height / 2
    return fx, fy, cx, cy

def load_and_view(ply_path, out_path, info_path):
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(ply_path)
    # get the basename of the scene
    scene_id = os.path.splitext(os.path.basename(ply_path))[0]
    # get the image info(width and height)
    with open(info_path) as f:
        info = json.load(f)
    img_width = info["width"]
    img_height = info["height"]
    print(f"Height: {img_height}, Width: {img_width}")

    fx, fy, cx, cy = estimate_intrinsics(img_width, img_height, fov_deg=60)
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],  # Flip Y-axis
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    pcd.transform([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, -1, 0],  # Flip Z-axis
                   [0, 0, 0, 1]])
    # Visualize with a custom key‐callback for camera motion
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=img_width, height=img_height, visible=True)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    # set the camera intrinsics
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(img_width, img_height, fx, fy, cx, cy)
    ctr = vis.get_view_control()
    # Wait a moment for everything to load
    time.sleep(0.5)

    def capture(name):
        vis.poll_events()
        vis.update_renderer()
        save_path = os.path.join(out_path, f"{scene_id}_{name}.png")
        vis.capture_screen_image(save_path, do_render=True)
        print(f"Saved: {save_path}")

    # Center view
    capture("center")

    # Turn right 75 degree
    ctr.rotate(-75, 0)
    capture("right")

    # Turn left 75 degree from right -75
    ctr.rotate(150, 0)
    capture("left")

    vis.run()
    vis.destroy_window()


load_and_view("data/out/bedroom/bedroom.ply","data/out/bedroom","data/out/bedroom/bedroom_info.json")