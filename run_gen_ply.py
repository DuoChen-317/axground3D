from config_util import (
    MP3D_DATASET_PATH,
    MP3D_DATASET_SCENE_IDS_LIST,
    NUM_OF_NODES_PRE_SCENE,
    ACTION_CHUNK,
)

from joblib import Parallel, delayed
from PIL import Image
from img_to_3d import infer
import os

def run_gen_ply(frame_path:str,save_path):
    # load the image
    img = Image.open(frame_path)
    filename = os.path.basename(frame_path)
    # get the save name
    save_name = "_".join(filename.split('_')[0:2])
   
    # infer
    output = infer(img,save_path,save_name,True,True)




if __name__ == "__main__":
    save_path = "./data/scenes/"
    print(MP3D_DATASET_SCENE_IDS_LIST)
    results = Parallel(n_jobs=2)(
        # stack the scene into path
        delayed(run_gen_ply)(
            f"./frames/state_base/{scene_id}_s0_step_base.png",  # frame_path
            save_path
        )
        for scene_id in MP3D_DATASET_SCENE_IDS_LIST
    )