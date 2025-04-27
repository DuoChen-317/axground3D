import yaml

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