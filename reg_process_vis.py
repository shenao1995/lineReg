import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from diffdrr.drr import DRR
from diffdrr.data import read
from diffdrr.pose import convert, RigidTransform
from utils import crop_ct_vert, extract_traditional_edge
from PIL import Image
from diffdrr.visualization import drr_to_mesh, img_to_mesh
import pyvista
import SimpleITK as sitk


def animate_reg_process():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 基础参数配置
    caseName = 'bimeihua'  # 替换为你要生成视频的样本名称
    vertName = 'L5'
    sampleName = f'{caseName}_{vertName}'
    imgsize = 256
    SDD = 1100
    delx = 0.8
    fps = 15.0  # 视频帧率（GIF 每帧的持续时间将根据它计算）

    # 路径配置：将后缀改为 .gif
    gif_name = f'results/tuodao/{sampleName}_pose_animation.gif'
    results_path = f'results/tuodao/{sampleName}_cmaes_pose.csv'
    origin_ct_path = f'Data/total/{caseName}/{caseName}.nii.gz'
    seg_path = f'Data/total/{caseName}/{vertName}_seg.nii.gz'
    save_dir = f'Data/total/{caseName}/{caseName}_{vertName}.nii.gz'

    # --- 1. 读取之前的背景 Ground Truth 和 GT Line (PNG 格式) ---
    bg_gt_path = f'results/tuodao/{sampleName}_bg_gt.png'
    gt_line_path = f'results/tuodao/{sampleName}_gt_line.png'

    if not os.path.exists(bg_gt_path) or not os.path.exists(gt_line_path):
        print("错误：找不到保存的 PNG 背景图或边缘图！")
        print("请确保先运行了配准程序 reg_method() 并保存了结果。")
        return

    # bg_gt 读为彩色图 (方便之后叠加带颜色的 moving 图像)
    bg_gt_img = cv2.imread(bg_gt_path, cv2.IMREAD_COLOR)
    # gt_line 读为单通道灰度图
    gt_line_img = cv2.imread(gt_line_path, cv2.IMREAD_GRAYSCALE)

    h, w, _ = bg_gt_img.shape

    # --- 2. 初始化 DRR 生成器所需参数 ---
    offsetx, offsety, offsetz = crop_ct_vert(origin_ct_path, seg_path, save_dir)
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

    # 读取包含位姿演进过程的 CSV 文件
    poses_data = pd.read_csv(results_path)

    # --- 3. 准备收集 GIF 帧 ---
    frames_list = []
    print(f"开始生成可视化配准 GIF: {gif_name}")

    for idx, row in tqdm(poses_data.iterrows(), total=len(poses_data), ncols=100):
        # 读取当前代（帧）的位姿并转换为转换矩阵
        rot_tensor = torch.tensor([[row["alpha"], row["beta"], row["gamma"]]], dtype=torch.float32, device=device)
        trans_tensor = torch.tensor([[row["bx"], row["by"], row["bz"]]], dtype=torch.float32, device=device)
        est_pose = convert(rot_tensor, trans_tensor, parameterization="euler_angles", convention="ZXY").to(device)

        # 投影出当前的 moving 图像
        with torch.no_grad():
            estimate = drr_gene(est_pose.compose(vert_mat))

        # 将生成的 Tensor 转为 Numpy 0-255 数组
        mov_drr_np = estimate.squeeze().detach().cpu().numpy()
        mov_drr_np = (mov_drr_np - mov_drr_np.min()) / (mov_drr_np.max() - mov_drr_np.min() + 1e-8) * 255.0
        mov_drr_np = mov_drr_np.astype(np.uint8)

        # =======================================================
        # 左侧面板：背景图 + 动态投影图 (Yellow)
        # =======================================================
        # 创建空白的三通道图像，将 moving 图像放在绿 (G=1) 和红 (R=2) 通道，混合成黄色 (在 OpenCV 的 BGR 空间中)
        mov_color = np.zeros_like(bg_gt_img)
        mov_color[:, :, 1] = mov_drr_np
        mov_color[:, :, 2] = mov_drr_np

        # 将黄色 moving 图像透明地叠加在背景图上
        left_panel = cv2.addWeighted(bg_gt_img, 0.8, mov_color, 0.6, 0)

        # =======================================================
        # 右侧面板：黑色背景 + GT Line (Red) + Moving Line (Green)
        # =======================================================
        # 提取当前投影图的边缘
        mov_line_np = extract_traditional_edge(estimate, threshold_ratio=0.08, margin_ratio=0.15)
        mov_line_np = (mov_line_np * 255).astype(np.uint8)

        right_panel = np.zeros_like(bg_gt_img)
        # 红色通道 (OpenCV 索引 2) 写入固定的 GT 边缘线
        right_panel[:, :, 2] = gt_line_img
        # 绿色通道 (OpenCV 索引 1) 写入动态的 Moving 边缘线
        right_panel[:, :, 1] = np.maximum(right_panel[:, :, 1], mov_line_np)

        # =======================================================
        # 图像拼接与写入
        # =======================================================
        # 水平拼接左右两张图像 (h, w*2, 3)
        frame = np.hstack((left_panel, right_panel))

        # 将 OpenCV 的 BGR 格式转换为 PIL 用的 RGB 格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 转换为 PIL Image 对象并存入列表
        frames_list.append(Image.fromarray(frame_rgb))

    # 保存为 GIF，配置持续时间（每帧毫秒数）和无限循环
    duration_ms = int(1000 / fps)
    frames_list[0].save(
        gif_name,
        save_all=True,
        append_images=frames_list[1:],
        duration=duration_ms,
        loop=0
    )

    del drr_gene
    print(f"GIF 生成完毕！保存在: {gif_name}")


def reg_process_visual_3d():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 1. 基础参数与路径配置
    caseName = 'bimeihua'
    vertName = 'L5'
    SDD = 1100
    imgsize = 256
    delx = 0.8
    sampleName = f'{caseName}_{vertName}'
    print(f"正在处理: {sampleName}")

    results_path = f'results/tuodao/{sampleName}_cmaes_pose.csv'
    origin_ct_path = f'Data/total/{caseName}/{caseName}.nii.gz'
    seg_path = f'Data/total/{caseName}/{vertName}_seg.nii.gz'
    save_dir = f'Data/total/{caseName}/{caseName}_{vertName}.nii.gz'

    # 读取配准过程的 Pose 数据
    poses_data = pd.read_csv(results_path)

    # 2. 直接使用 crop_ct_vert 计算偏移量 (替代外部 CSV 读取)
    offsetx, offsety, offsetz = crop_ct_vert(origin_ct_path, seg_path, save_dir)
    # 依照你的逻辑，将 x 轴的偏移量取反
    offset_trans = np.array([-offsetx, offsety, offsetz])
    true_params = {
        "sdr": SDD, "alpha": 0.0, "beta": 0.0, "gamma": 0.0,
        "bx": 0.0, "by": 900.0, "bz": 0.0,
    }
    # 3. 生成 DRR 投影仪及 3D 椎骨 Mesh
    # 注意：这里已经适配了新版 get_drr 的 3 个返回值
    # vert_drr, gt_rotations, gt_translations = get_drr(img_path=save_dir, offset_trans=offset_trans, device=device)
    # ct_drr, ct_gt_rotations, ct_gt_translations = get_drr(img_path=origin_ct_path, device=device)
    subject = read(save_dir)

    # 3. 初始化生成器 drr_gene
    vert_drr = DRR(subject, sdd=SDD, height=imgsize, delx=delx, reverse_x_axis=True).to(device, dtype=torch.float32)

    print("正在生成 3D 椎骨 Mesh (Surface Nets)...")
    vert_mesh = drr_to_mesh(vert_drr.subject, "surface_nets", threshold=300, verbose=False)
    gt_rot = torch.tensor([[true_params["alpha"], true_params["beta"], true_params["gamma"]]])
    gt_trans = torch.tensor([[true_params["bx"], true_params["by"], true_params["bz"]]])
    gt_pose = convert(gt_rot, gt_trans, parameterization="euler_angles", convention="ZXY").to(device)

    # _, gt_ap_detector, _, _ = img_to_mesh(vert_drr, gt_pose)
    gt_camera, gt_ap_detector, gt_texture, gt_principal_ray = img_to_mesh(vert_drr, gt_pose)

    # 4. 初始化 PyVista Plotter (off_screen=True 避免弹窗干扰)
    gif_path = f'results/singleView_lines/{sampleName}_reg_process.gif'
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    plotter = pyvista.Plotter(off_screen=True)
    plotter.open_gif(gif_path, fps=15)

    # 添加静态的椎骨 Mesh (仅需添加一次)
    plotter.add_mesh(vert_mesh, color="ivory", name="static_vert_mesh")
    plotter.add_mesh(gt_camera, show_edges=True, line_width=1.5, color="green", name="gt_camera")
    plotter.add_mesh(gt_principal_ray, color="lime", line_width=3, name="gt_ray")
    plotter.add_mesh(gt_ap_detector, texture=gt_texture, name="gt_detector")
    # 若需添加背景 CT 参考，可解开以下注释
    # add_other_mesh(plotter, ct_gt_rotations, ct_gt_translations, ct_drr, gt_ap_detector, isCT=True)

    print(f"正在渲染 3D 动画，共 {len(poses_data)} 帧...")

    # 5. 遍历 Pose 演化过程，生成动画
    for idx, row in tqdm(poses_data.iterrows(), total=len(poses_data), ncols=100):
        rotate = torch.tensor(row[["alpha", "beta", "gamma"]].values, dtype=torch.float32, device=device).unsqueeze(0)
        transl = torch.tensor(row[["bx", "by", "bz"]].values, dtype=torch.float32, device=device).unsqueeze(0)

        pose = convert(rotate, transl, parameterization='euler_angles', convention="ZYX")

        # 根据当前位姿生成相机、探测器面、纹理和主射线
        camera, detector, texture, principal_ray = img_to_mesh(vert_drr, pose)

        # 添加动态组件 (利用 name 属性自动覆盖上一帧的组件)
        plotter.add_mesh(camera, show_edges=True, line_width=1.5, color="red", name="dynamic_camera")
        plotter.add_mesh(principal_ray, color="lime", line_width=3, name="dynamic_ray")
        plotter.add_mesh(detector, texture=texture, name="dynamic_detector")

        # 写入帧
        plotter.write_frame()

    # 6. 完成绘制并保存
    plotter.close()
    print(f"3D 配准过程可视化已保存至: {gif_path}")



def add_other_mesh(pl, rotate, transl, drr, gt_det=None, isCT=False, mesh_name_prefix="other"):
    pose = convert(rotate.float(), transl.float(), parameterization='euler_angles', convention="ZYX")
    camera, detector, texture, principal_ray = img_to_mesh(drr, pose)

    if isCT and gt_det is not None:
        det_offset = np.array(detector.center) - np.array(gt_det.center)
        offset_arr = np.array([det_offset[0], det_offset[1], det_offset[2]])

        detector.translate(-offset_arr, inplace=True)
        camera.translate(-offset_arr, inplace=True)
        principal_ray.translate(-offset_arr, inplace=True)

    pl.add_mesh(detector, texture=texture, name=f"{mesh_name_prefix}_detector")
    pl.add_mesh(camera, show_edges=True, line_width=1.5, name=f"{mesh_name_prefix}_camera")
    pl.add_mesh(principal_ray, color="lime", line_width=3, name=f"{mesh_name_prefix}_ray")

    return detector


if __name__ == '__main__':
    # animate_reg_process()
    reg_process_visual_3d()
