import os
import cv2
import torch
import numpy as np
import pandas as pd
import pyvista
from tqdm import tqdm
from PIL import Image
from diffdrr.drr import DRR
from diffdrr.data import read
from diffdrr.pose import convert, RigidTransform
from diffdrr.visualization import drr_to_mesh, img_to_mesh

# 假设这两个工具函数已经存在或被正确导入
from utils import crop_ct_vert, extract_traditional_edge


def animate_combined_process():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 1. 基础参数与路径配置
    caseName = 'case1'
    vertName = 'L2'
    sampleName = f'{caseName}_{vertName}'
    SDD = 1100
    imgsize = 256
    delx = 0.8
    fps = 15.0  # GIF 的帧率

    print(f"正在处理: {sampleName}")

    # 修改为全新的目录结构
    ct_path = f'Data/{caseName}/ct.nii.gz'
    vert_seg_path = f'Data/{caseName}/ct_seg.nii.gz'
    save_dir = f'Data/{caseName}/{caseName}_{vertName}.nii.gz'

    # 结果路径
    results_path = f'results/{sampleName}/cmaes_pose.csv'
    gif_name = f'results/{sampleName}/combined_animation.gif'
    bg_gt_path = f'results/{sampleName}/bg_gt.png'
    gt_line_path = f'results/{sampleName}/gt_line.png'

    if not os.path.exists(bg_gt_path) or not os.path.exists(gt_line_path):
        print("错误：找不到保存的 PNG 背景图或边缘图！请确保先运行并保存了配准结果。")
        return

    # --- 2. 准备 2D 背景资源 ---
    bg_gt_img = cv2.imread(bg_gt_path, cv2.IMREAD_COLOR)
    gt_line_img = cv2.imread(gt_line_path, cv2.IMREAD_GRAYSCALE)

    # --- 3. 初始化 2D DRR 生成器 ---
    # 更新了 crop_ct_vert 的应用方式，加入 vert_name 映射
    offsetx, offsety, offsetz = crop_ct_vert(ct_path, vert_seg_path, crop_vert_path=save_dir, vert_name=vertName)
    offset_trans = np.array([offsetx, offsety, offsetz])

    vert_mat = np.array([
        [1.0, 0.0, 0.0, -offset_trans[0]],
        [0.0, 1.0, 0.0, -offset_trans[1]],
        [0.0, 0.0, 1.0, offset_trans[2]],
        [0.0, 0.0, 0.0, 1.0],
    ])
    vert_mat = RigidTransform(torch.FloatTensor(vert_mat)).to(device)

    subject = read(save_dir, bone_attenuation_multiplier=10.5)
    drr_gene = DRR(subject, sdd=SDD, height=imgsize, delx=delx, reverse_x_axis=True).to(device, dtype=torch.float32)

    # --- 4. 准备 3D PyVista 资源 ---
    print("正在生成 3D 椎骨 Mesh (Surface Nets)...")
    vert_mesh = drr_to_mesh(drr_gene.subject, "surface_nets", threshold=300, verbose=False)

    true_params = {
        "sdr": SDD, "alpha": 0.0, "beta": 0.0, "gamma": 0.0,
        "bx": 0.0, "by": 900.0, "bz": 0.0,
    }
    gt_rot = torch.tensor([[true_params["alpha"], true_params["beta"], true_params["gamma"]]])
    gt_trans = torch.tensor([[true_params["bx"], true_params["by"], true_params["bz"]]])
    gt_pose = convert(gt_rot, gt_trans, parameterization="euler_angles", convention="ZXY").to(device)

    gt_camera, gt_ap_detector, gt_texture, gt_principal_ray = img_to_mesh(drr_gene, gt_pose)

    # 初始化 3D 绘图窗口 (设定为 512x512 大小以配合 2D 图像拼接)
    plotter = pyvista.Plotter(off_screen=True, window_size=[512, 512])
    plotter.add_mesh(vert_mesh, color="ivory", name="static_vert_mesh")
    plotter.add_mesh(gt_camera, show_edges=True, line_width=1.5, color="green", name="gt_camera")
    plotter.add_mesh(gt_principal_ray, color="lime", line_width=3, name="gt_ray")
    plotter.add_mesh(gt_ap_detector, texture=gt_texture, name="gt_detector")

    # --- 5. 遍历 Pose 开始生成合并动画 ---
    poses_data = pd.read_csv(results_path)
    frames_list = []

    print(f"开始生成并合并配准 GIF，共 {len(poses_data)} 帧...")

    for idx, row in tqdm(poses_data.iterrows(), total=len(poses_data), ncols=100):
        rot_tensor = torch.tensor([[row["alpha"], row["beta"], row["gamma"]]], dtype=torch.float32, device=device)
        trans_tensor = torch.tensor([[row["bx"], row["by"], row["bz"]]], dtype=torch.float32, device=device)
        pose = convert(rot_tensor, trans_tensor, parameterization="euler_angles", convention="ZXY").to(device)

        # ==================================
        # 步骤 A：生成并处理 2D 图像
        # ==================================
        with torch.no_grad():
            estimate = drr_gene(pose.compose(vert_mat))

        mov_drr_np = estimate.squeeze().detach().cpu().numpy()
        mov_drr_np = (mov_drr_np - mov_drr_np.min()) / (mov_drr_np.max() - mov_drr_np.min() + 1e-8) * 255.0
        mov_drr_np = mov_drr_np.astype(np.uint8)

        # [A-1] 左侧面板：投影重叠 (背景=灰, Moving=黄)
        mov_color = np.zeros_like(bg_gt_img)
        mov_color[:, :, 1] = mov_drr_np
        mov_color[:, :, 2] = mov_drr_np
        left_panel = cv2.addWeighted(bg_gt_img, 0.8, mov_color, 0.6, 0)

        # [A-2] 中间面板：边缘线重叠 (背景黑, GT=红, Moving=绿)
        mov_line_np = extract_traditional_edge(estimate, threshold_ratio=0.08, margin_ratio=0.15)
        mov_line_np = (mov_line_np * 255).astype(np.uint8)

        mid_panel = np.zeros_like(bg_gt_img)
        mid_panel[:, :, 2] = gt_line_img
        mid_panel[:, :, 1] = np.maximum(mid_panel[:, :, 1], mov_line_np)

        # 拼接 2D 面板得到 (256, 512, 3) 图像，并转为 RGB
        frame_2d = np.hstack((left_panel, mid_panel))
        frame_2d_rgb = cv2.cvtColor(frame_2d, cv2.COLOR_BGR2RGB)

        # 将 2D 图像高度放大到 512，与 3D 渲染窗口高度保持一致 (变成 512x1024)
        frame_2d_resized = cv2.resize(frame_2d_rgb, (1024, 512), interpolation=cv2.INTER_LINEAR)

        # ==================================
        # 步骤 B：生成并提取 3D 图像
        # ==================================
        camera, detector, texture, principal_ray = img_to_mesh(drr_gene, pose)
        plotter.add_mesh(camera, show_edges=True, line_width=1.5, color="red", name="dynamic_camera")
        plotter.add_mesh(principal_ray, color="lime", line_width=3, name="dynamic_ray")
        plotter.add_mesh(detector, texture=texture, name="dynamic_detector")

        # 主动渲染并提取图像数组为 RGB (尺寸为 512x512x3)
        plotter.render()
        frame_3d = plotter.screenshot(transparent_background=False, return_img=True)
        # ==================================
        # 步骤 C：拼接 2D 与 3D 图像并保存
        # ==================================
        # 最终合并图像尺寸为 (512, 1536, 3)
        combined_frame = np.hstack((frame_2d_resized, frame_3d))
        frames_list.append(Image.fromarray(combined_frame))

    # 关闭 3D 渲染器
    plotter.close()

    # --- 6. 统一保存为 GIF ---
    duration_ms = int(1000 / fps)
    frames_list[0].save(
        gif_name,
        save_all=True,
        append_images=frames_list[1:],
        duration=duration_ms,
        loop=0
    )

    del drr_gene
    print(f"合并 GIF 生成完毕！保存在: {gif_name}")


if __name__ == '__main__':
    animate_combined_process()
