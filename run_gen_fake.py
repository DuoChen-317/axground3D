import os
os.environ["OMP_NUM_THREADS"]       = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]      = "1"

from reconstruct_3d import ViewRenderer
# from diffusion_process import fill_in
from config_util import (
    MP3D_DATASET_PATH,
    MP3D_DATASET_SCENE_IDS_LIST,
    NUM_OF_NODES_PRE_SCENE,
    ACTION_CHUNK,
)


for scene_id in MP3D_DATASET_SCENE_IDS_LIST:
    ply_path = os.path.join("./data/scenes",scene_id, f"{scene_id}_s0_base.ply")
    depth_path = os.path.join("./data/scenes",scene_id, f"{scene_id}_s0_base_depth.png")
    ray_path = os.path.join("./data/scenes",scene_id, f"{scene_id}_s0_base_rays.png")

    print(ply_path, depth_path, ray_path)