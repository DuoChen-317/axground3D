import os
os.environ["OMP_NUM_THREADS"]       = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]      = "1"


from diffusers import StableDiffusionXLInpaintPipeline,utils,DPMSolverMultistepScheduler
import torch
from PIL import Image
import sys
import numpy as np
import time
import pybullet as p
import pybullet_data
# from reconstruct_3d import load_and_view,estimate_intrinsics


model = "./model/RealVisXL_V4.0_inpainting"
device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

def fill_in(init_image:Image.Image, mask_image:Image.Image) -> Image.Image:
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        model, revision="fp16",variant="fp16"
    )
    # 2) swap in DPM++ 2M Karras scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()  # reduce VRAM usage
    # Inpaint
    result = pipe(
        prompt= "a indoor room",  # empty prompt to rely solely on surrounding context
        image=init_image,
        mask_image=mask_image,
        guidance_scale=7.5,  # adjust as needed
        num_inference_steps=25
    ).images[0]
    result.show()  # Display the result
    return result

if __name__ == "__main__":
    
    # load_and_view(ply_path='data/bedroom.ply',out_path="data/out/bedroom",info_path="data/out/bedroom/bedroom_info.json")


    import open3d as o3d
    import numpy as np
    from PIL import Image

    # # 加载点云
    # pcd = o3d.io.read_point_cloud("data/bedroom.ply")

    # # 创建可视化器
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(width=640, height=480, visible=True)  # visible=False 不弹窗
    # vis.add_geometry(pcd)

    # # 配置相机（参考 PyBullet 里那些 cam_eye/cam_target）
    # ctr = vis.get_view_control()
    # param = ctr.convert_to_pinhole_camera_parameters()
    # param.extrinsic = np.array([
    #     [1,0,0, 0],     # 这里填你的相机姿态矩阵
    #     [0,1,0, 0],
    #     [0,0,1, 1.5],   # 提升到 1.5m 高度
    #     [0,0,0, 1],
    # ])
    # ctr.convert_from_pinhole_camera_parameters(param)

    # # 渲染并截图
    # vis.poll_events()
    # vis.update_renderer()
    # img = vis.capture_screen_float_buffer(do_render=True)
    # vis.destroy_window()

    # # 储存
    # img = (255 * np.asarray(img)).astype(np.uint8)
    # Image.fromarray(img).save("frame.png")
    # image = utils.load_image(image_path)
    # filled_image = fill_in(image)

    # filled_image.show()  # Display the result


    # 1) 读入纯点云 PLY
    pcd = o3d.io.read_point_cloud("data/view_001.ply")

    # 2) 创建可视化器（不弹窗，直接渲染到缓冲区）
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=512, height=512, visible=True)
    vis.add_geometry(pcd)

    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],  # Flip Y-axis
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    pcd.transform([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, -1, 0],  # Flip Z-axis
                   [0, 0, 0, 1]])


    # 3) 设置固定投影参数（FOV, 近平面, 远平面）
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    param.intrinsic.set_intrinsics(
        width=480, height=480,
        fx=525, fy=525,  # 根据需要调整
        cx=320, cy=240
    )




    # 4) 定义一条轨迹：从 (0,0,1) 沿 X 轴走到 (2,0,1)
    num_steps = 1
    eyes = np.linspace([0,0,1], [2,0,1], num_steps)

    os.makedirs("frames", exist_ok=True)
    last_frame = 0
    for i, eye in enumerate(eyes):
        # 朝向：看向 +X 方向
        center = eye + np.array([1,0,0])
        up     = np.array([0,0,1])
        # 构造外参
        extrinsic = np.eye(4)
        # R: 用 look_at 方法计算，或者直接填旋转矩阵
        extrinsic[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, 0])
        extrinsic[:3, 3]  = eye
        param.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(param)

        # 渲染 & 截图
        vis.poll_events()
        vis.update_renderer()
        # 捕获色彩缓冲与深度缓冲
        color_buf = np.asarray(vis.capture_screen_float_buffer(do_render=False))  # float [0,1]
        depth_buf = np.asarray(vis.capture_depth_float_buffer(do_render=False))   # float depth

        # 转为 8-bit RGB
        rgb8 = (np.clip(color_buf, 0.0, 1.0) * 255).astype(np.uint8)
        rgb_image = utils.load_image(Image.fromarray(rgb8))
        rgb_image.show()
        alpha = ((depth_buf != 0.0).astype(np.uint8) * 255)
        # reverse alpha
        alpha = 255 - alpha
        mask_image = utils.load_image(Image.fromarray(alpha))
        mask_image.show()
        # 合成 RGBA 并保存
        rgba = np.dstack((rgb8, alpha))
        Image.fromarray(rgba, mode="RGBA")
        


    vis.destroy_window()
   
    fill_in(rgb_image,mask_image)