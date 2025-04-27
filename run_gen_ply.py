from config_util import MP3D_DATASET_PATH,MP3D_DATASET_SCENE_IDS_LIST,NUM_OF_NODES_PRE_SCENE

# from img_to_3d import infer
from sim_connect.hb import init_simulator,create_viewer
import os
import magnum as mn
import numpy as np

  
def random_sample(pathfinder, num_nodes: int):
    samples = []
    attempts = 0
    max_attempts = num_nodes * 10  # Avoid infinite loops
    while len(samples) < num_nodes and attempts < max_attempts:
        pt = pathfinder.get_random_navigable_point()
        if not any(np.allclose(pt, s['point'], atol=1e-3) for s in samples):  # avoid near-duplicates
            sample = {'point': pt, 'radius': None}
            samples.append(sample)
        attempts += 1
    return samples

def run_gen_ply():


    # action design: turn_left, turn_right x 2, turn_left x 2, move_forward, 
    # turn_left, turn_right x 2,

    action_chunck = [{'name': 'turn_left', 'repeat': 1},
                     {'name': 'turn_right', 'repeat': 2},
                     {'name': 'turn_left', 'repeat': 2},
                     {'name': 'move_forward', 'repeat': 1},
                     {'name': 'turn_left', 'repeat': 1},
                     {'name': 'turn_right', 'repeat': 2}]



    for scene_id in MP3D_DATASET_SCENE_IDS_LIST[0:2]:
        scene_path = os.path.join(MP3D_DATASET_PATH, scene_id,f"{scene_id}.glb")
        sim = init_simulator(scene_path,is_physics=True)
        viewer = create_viewer(scene_path)
        pathfinder = sim.pathfinder

        samples = random_sample(pathfinder, NUM_OF_NODES_PRE_SCENE)
        for s in samples:
            pos = s['point']
            if isinstance(pos, str):
                pos = list(map(float, pos.split(',')))
            vec = mn.Vector3(*pos)
            viewer.transit_to_goal(vec)
            file_name = os.path.join("./frames/state_base", f"{scene_id}_step_base.png")
            viewer.save_viewpoint_image(file_name)
            for _id, action in enumerate(action_chunck):
                if _id != 2 and _id != 3:
                    
                    viewer.move_and_look(action['name'], action['repeat'])
                    file_name = os.path.join("./frames/real", f"{scene_id}_step{_id}.png")
                    viewer.save_viewpoint_image(file_name)
                    
                else:
                    viewer.move_and_look(action['name'], action['repeat'])
        
        viewer.close()
        sim.close()

if __name__ == "__main__":
    run_gen_ply()