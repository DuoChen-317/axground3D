import os

import imageio
import numpy as np
os.environ["OMP_NUM_THREADS"]       = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]      = "1"

from reconstruct_3d import ViewRenderer
import open3d as o3d
from diffusion_process import fill_in
from config_util import (
    MP3D_DATASET_PATH,
    MP3D_DATASET_SCENE_IDS_LIST,
    NUM_OF_NODES_PRE_SCENE,
    ACTION_CHUNK,
)

save_intermediate = True

for scene_id in MP3D_DATASET_SCENE_IDS_LIST[0:1]:
    for ids in range(NUM_OF_NODES_PRE_SCENE):
        
        ply_path = os.path.join("./data/scenes",f"{scene_id}_s{ids}", f"{scene_id}_s{ids}.ply")
        depth_path = os.path.join("./data/scenes",f"{scene_id}_s{ids}", f"{scene_id}_s{ids}_depth.png")
        ray_path = os.path.join("./data/scenes",f"{scene_id}_s{ids}", f"{scene_id}_s{ids}_rays.png")
        
        pcd = o3d.io.read_point_cloud(ply_path)
        depth = imageio.imread(depth_path).astype(np.float32)
        ray = imageio.imread(ray_path).astype(np.float32)
        myVRD = ViewRenderer(pcd,depth,ray)

        for _id, action in enumerate(ACTION_CHUNK):
            myVRD.render_view()
            if _id != 2 and _id != 3:

                if action["name"] == "move_forward":
                    myVRD.move_forward(action["repeat"])
                elif action["name"] == "turn_left":
                    myVRD.rotate_extrinsic(-30*action["repeat"])
                elif action["name"] == "turn_right":
                    myVRD.rotate_extrinsic(30*action["repeat"])
                rgb_image, mask_image =  myVRD.render_view()
                if save_intermediate:
                    rgb_image.save(f"./frames/interm/{scene_id}_s{ids}_step{_id}.png")
                    mask_image.save(f"./frames/interm/{scene_id}_s{ids}_step{_id}_mask.png")
                
                result_img = fill_in(rgb_image, mask_image)
                result_img.save(f"./frames/fake/{scene_id}_s{ids}_step{_id}.png")
            else:
                if action["name"] == "move_forward":
                    myVRD.move_forward(action["repeat"])
                elif action["name"] == "turn_left":
                    myVRD.rotate_extrinsic(-30*action["repeat"])
                elif action["name"] == "turn_right":
                    myVRD.rotate_extrinsic(30*action["repeat"])