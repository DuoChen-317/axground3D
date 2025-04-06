import argparse
import json
import os
import numpy as np
import torch
from PIL import Image
from unik3d.models import UniK3D
from unik3d.utils.visualization import colorize, save_file_ply




def save(rgb, outputs, name, base_path, save_map, save_pointcloud):
    """
    Save the depth map, rays, and point cloud to disk after inference.
    Args:
        rgb (np.ndarray): original RGB image
        outputs (dict): Dictionary containing the depth, rays, and points
        name (str): Name of the output files
        base_path (str): Base directory to save the output files
        save_map (bool): Whether to save the depth map and rays as PNG images
        save_pointcloud (bool): Whether to save the point cloud as a PLY file
    """
    os.makedirs(base_path, exist_ok=True)

    depth = outputs["depth"]
    rays = outputs["rays"]
    points = outputs["points"]
    info = outputs["info"]
    # save the output info as JSON file
    info_json = os.path.join(base_path, f"{name}_info.json")
    with open(info_json, "w") as f:
        json.dump(info, f, indent=4)
    print(f"Info saved to {info_json}")
    # save the depth and ray as image
    depth = depth.cpu().numpy()
    rays = ((rays + 1) * 127.5).clip(0, 255)
    if save_map:
        Image.fromarray(colorize(depth.squeeze())).save(
            os.path.join(base_path, f"{name}_depth.png")
        )
        Image.fromarray(rays.squeeze().permute(1, 2, 0).byte().cpu().numpy()).save(
            os.path.join(base_path, f"{name}_rays.png")
        )
        print(f"Depth map and rays saved to {base_path}")
    # save the point cloud as a PLY file
    if save_pointcloud:
        predictions_3d = points.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
        rgb = rgb.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
        save_file_ply(predictions_3d, rgb, os.path.join(base_path, f"{name}.ply"))
        print(f"Point cloud saved to {base_path}")


def infer(input_path, output_path,portrait = False,save_ply= True, save_map=True, resolution_level=9 ,interpolation_mode="bilinear", camera_config = None):
    """
    Inference function for UniK3D model.
    Args:
        input_path (str): Path to the input RGB image
        save_ply (bool): Whether to save the point cloud as a PLY file
        save_map (bool): Whether to save the depth map and rays as PNG images
        output_path (str): Base directory to save the output files
        portrait: if the image is a portrait
        resolution_level (int): Resolution level for the model (default: 9)
        interpolation_mode (str): Interpolation mode for the model (default: "bilinear")
        camera_config (str): Path to the camera configuration file (default: None)
    Returns:
        dict: Dictionary containing the depth, rays, and points
    """
    print("Starting to load the model")
    # Load the model
    model = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl")
    model.resolution_level = resolution_level
    model.interpolation_mode = interpolation_mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'using device:{device}')
    model = model.to(device).eval()
    # open the input image
    img = Image.open(input_path)
    width, height = img.size # get the size of image
    # preprocess the image
    rgb = np.array(img)
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    # camera setting
    camera = None
    camera_path = camera_config # get the camera path config
    if camera_path is not None:
        with open(camera_path, "r") as f:
            camera_dict = json.load(f)
        params = torch.tensor(camera_dict["params"])
        name = camera_dict["name"]
        assert name in ["Fisheye624", "Spherical", "OPENCV", "Pinhole", "MEI"]
        camera = eval(name)(params=params)

    # inference
    print("start inference...")
    outputs = model.infer(rgb=rgb_torch, camera=camera, normalize=True, rays=None)

    # add the image size and portrait information to the output
    outputs["info"] = {
        "width": width,
        "height": height,
        "portrait": portrait
    }

    print("inference finished")
    # get the output path and create a dict to store all the output
    name = os.path.splitext(os.path.basename(input_path))[0]
    save_dir = os.path.join(output_path, name)
    os.makedirs(save_dir, exist_ok=True)

    # save ply and other images
    if save_map or save_ply:
        save(rgb_torch, outputs, name=name, base_path=save_dir, save_map=save_map, save_pointcloud=save_ply)
        print("output saved")
    return outputs


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Inference script', conflict_handler='resolve')
    parser.add_argument("--input", type=str, required=True, help="Path to input image.")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory.")
    parser.add_argument("--portrait", default=False, action="store_true", help="Is the image a portrait?")
    parser.add_argument("--camera-path", type=str, default=None, help="Path to camera parameters json file.")
    parser.add_argument("--save", default=True,action="store_true", help="Save outputs as (colorized) png.")
    parser.add_argument("--save-ply",default=True, action="store_true", help="Save pointcloud as ply.")
    parser.add_argument("--resolution-level", type=int, default=9, choices=list(range(10)), help="Resolution level in [0,10).")
    parser.add_argument("--interpolation-mode", type=str, default="bilinear", choices=["nearest", "nearest-exact", "bilinear"], help="Interpolation method.")
    args = parser.parse_args()

    # ✅ Correct function call:
    infer(
        input_path=args.input,
        portrait=args.portrait,
        save_ply=args.save_ply,
        save_map=args.save,
        output_path=args.output,
        resolution_level=args.resolution_level,
        interpolation_mode=args.interpolation_mode,
        camera_config=args.camera_path
    )
