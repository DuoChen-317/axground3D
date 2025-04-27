import yaml
import os
CONFIG_FILE_PATH = './local.yaml'


def load_config(file_path):
    """
    Load the configuration file.
    :param file_path: Path to the configuration file.
    :return: Configuration dictionary.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config(CONFIG_FILE_PATH)

MP3D_DATASET_PATH = config['mp3d_habitat_scene_dataset_path']

VLM_MODEL_PATH = config['vlm_model_path']

NUM_OF_NODES_PRE_SCENE = config['number_of_node_per_scene']

def get_scene_list(MP3D_DATASET_PATH:str):
    """
    Read sub level scene list from folder.
    :param MP3D_DATASET_PATH: Path to the MP3D dataset.
    :return: List of scene names.
    """
    
    scene_ids_list = [d for d in os.listdir(MP3D_DATASET_PATH)
                 if os.path.isdir(os.path.join(MP3D_DATASET_PATH, d))
                 and os.path.exists(os.path.join(MP3D_DATASET_PATH, d, f"{d}.glb"))]
    

    return scene_ids_list

MP3D_DATASET_SCENE_IDS_LIST = get_scene_list(MP3D_DATASET_PATH)

# check if ./frames folder exists
if not os.path.exists('./frames'):
    os.makedirs('./frames')
# check if ./frames/fake folder exists
if not os.path.exists('./frames/fake'):
    os.makedirs('./frames/fake')
# check if ./frames/real folder exists
if not os.path.exists('./frames/real'):
    os.makedirs('./frames/real')
# check if ./frames/state_base folder exists
if not os.path.exists('./frames/state_base'):
    os.makedirs('./frames/state_base')