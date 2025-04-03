import argparse
import json
import os
import numpy as np
import torch
from PIL import Image
from unik3d.models import UniK3D
from unik3d.utils.visualization import colorize, save_file_ply




def save(rgb, outputs, name, base_path, save_map=False, save_pointcloud=False):
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


def infer(input_path, save_ply, save_map, output_path, resolution_level=9 ,interpolation_mode="bilinear", camera_config = None):
    """
    Inference function for UniK3D model.
    Args:
        input_path (str): Path to the input RGB image
        save_ply (bool): Whether to save the point cloud as a PLY file
        save_map (bool): Whether to save the depth map and rays as PNG images
        output_path (str): Base directory to save the output files
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
    rgb = np.array(Image.open(input_path))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)

    camera = None
    camera_path = camera_config # get the camera path config
    if camera_path is not None:
        # use default setting for now
        with open(camera_path, "r") as f:
            camera_dict = json.load(f)
        params = torch.tensor(camera_dict["params"])
        name = camera_dict["name"]
        assert name in ["Fisheye624", "Spherical", "OPENCV", "Pinhole", "MEI"]
        camera = eval(name)(params=params)
    print("start inference...")
    # inference
    outputs = model.infer(rgb=rgb_torch, camera=camera, normalize=True, rays=None)
    name = input_path.split("/")[-1].split(".")[0]
    # save
    if save_map or save_ply:
        save(rgb_torch, outputs, name=name, base_path=output_path, save_map=save_map, save_pointcloud=save_ply)
    print("inference finished")
    return outputs


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Inference script', conflict_handler='resolve')
    parser.add_argument("--input", type=str, required=True, help="Path to input image.")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory.")
    parser.add_argument("--camera-path", type=str, default=None, help="Path to camera parameters json file.")
    parser.add_argument("--save", action="store_true", help="Save outputs as (colorized) png.")
    parser.add_argument("--save-ply", action="store_true", help="Save pointcloud as ply.")
    parser.add_argument("--resolution-level", type=int, default=9, choices=list(range(10)), help="Resolution level in [0,10).")
    parser.add_argument("--interpolation-mode", type=str, default="bilinear", choices=["nearest", "nearest-exact", "bilinear"], help="Interpolation method.")
    args = parser.parse_args()

    # ✅ Correct function call:
    infer(
        input_path=args.input,
        save_ply=args.save_ply,
        save_map=args.save,
        output_path=args.output,
        resolution_level=args.resolution_level,
        interpolation_mode=args.interpolation_mode,
        camera_config=args.camera_path
    )
