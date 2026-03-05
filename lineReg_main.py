import os
import torch
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
from tqdm import tqdm
from cmaes import CMA

from diffdrr.drr import DRR
from diffdrr.data import read
from diffdrr.pose import convert, RigidTransform
from diffdrr.metrics import NormalizedCrossCorrelation2d
from utils import crop_ct_vert, extract_traditional_edge


def reg_method(origin_ct_path, seg_path, save_dir, sampleName, vertName):
    # 1. 读取椎骨CT及中心偏移
    offsetx, offsety, offsetz = crop_ct_vert(origin_ct_path, seg_path, crop_vert_path=save_dir, vert_name=vertName)
    offset_trans = np.array([offsetx, offsety, offsetz])

    # 2. 投影仪及位姿参数设置
    SDD = 1100
    imgsize = 256
    delx = 0.8
    true_params = {
        "sdr": SDD, "alpha": 0.0, "beta": 0.0, "gamma": 0.0,
        "bx": 0.0, "by": 900.0, "bz": 0.0,
    }
    bg_subject = read(origin_ct_path, bone_attenuation_multiplier=10.5)
    bg_drr_gene = DRR(bg_subject, sdd=SDD, height=imgsize, delx=delx, reverse_x_axis=True).to(device,
                                                                                              dtype=torch.float32)

    subject = read(save_dir)

    # 3. 初始化生成器 drr_gene
    drr_gene = DRR(subject, sdd=SDD, height=imgsize, delx=delx, reverse_x_axis=True).to(device, dtype=torch.float32)

    # 构建 Ground Truth (正位 DRR)
    gt_rot = torch.tensor([[true_params["alpha"], true_params["beta"], true_params["gamma"]]])
    gt_trans = torch.tensor([[true_params["bx"], true_params["by"], true_params["bz"]]])
    gt_pose = convert(gt_rot, gt_trans, parameterization="euler_angles", convention="ZXY").to(device)

    vert_mat = np.array([
        [1.0, 0.0, 0.0, -offset_trans[0]],
        [0.0, 1.0, 0.0, -offset_trans[1]],
        [0.0, 0.0, 1.0, offset_trans[2]],
        [0.0, 0.0, 0.0, 1.0],
    ])
    vert_mat = RigidTransform(torch.FloatTensor(vert_mat)).to(device)

    ground_truth = drr_gene(gt_pose.compose(vert_mat))
    bg_ground_truth = bg_drr_gene(gt_pose)

    print(f"GT Shape: {ground_truth.shape}")

    # --- 提取 GT 的侧边线 ---
    gt_line_np = extract_traditional_edge(ground_truth, threshold_ratio=0.08, margin_ratio=0.15)
    gt_img_np = ground_truth.squeeze().detach().cpu().numpy()

    # --- 新增：保存 bg_ground_truth 和 gt_line_np 为 png 图像 ---
    os.makedirs(f'results/{sampleName}/', exist_ok=True)

    # 提取 bg_ground_truth 的 numpy 数组并归一化到 0-255
    bg_img_np = bg_ground_truth.squeeze().detach().cpu().numpy()
    bg_norm = (bg_img_np - bg_img_np.min()) / (bg_img_np.max() - bg_img_np.min() + 1e-8) * 255.0
    cv2.imwrite(f'results/{sampleName}/bg_gt.png', bg_norm.astype(np.uint8))

    # 将 mask 边缘线 (0 或 1) 转为 0 和 255 保存
    cv2.imwrite(f'results/{sampleName}/gt_line.png', (gt_line_np * 255).astype(np.uint8))
    print(f"已保存参考图像至 results/{sampleName}/_bg_gt.png 和 {sampleName}_gt_line.png")

    # 可视化 GT 与其连续的侧边缘线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth (AP)")
    plt.imshow(gt_img_np, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("GT Edge Line (Sides Only)")
    plt.imshow(gt_img_np, cmap='gray')
    plt.imshow(gt_line_np, cmap='Reds', alpha=0.5 * (gt_line_np > 0))
    plt.show()

    # 4. 生成初始扰动位姿
    ini_rot, ini_trans, pose_init = get_initial_parameters(true_params)

    # 5. 调用 CMA-ES 优化 (传入 bg_ground_truth 用于最后的重叠可视化)
    optimize(drr_gene, vert_mat, ground_truth, bg_ground_truth, gt_line_np, sampleName, ini_rot, ini_trans)

    del drr_gene


def optimize(reg: DRR, vert_mat, gt_img, bg_img, gt_line_np, samplename, initial_rot, initial_trans, n_itrs=150):
    T1 = time.time()

    gncc_metric = NormalizedCrossCorrelation2d().to(device)

    rot = initial_rot.cpu().numpy().squeeze()
    trans = initial_trans.cpu().numpy().squeeze()
    rtvec = np.concatenate([rot, trans])

    kDEG2RAD = np.pi / 180
    offset_angle = 30 * kDEG2RAD
    bound = [
        [rot[0] - offset_angle, rot[0] + offset_angle],
        [rot[1] - offset_angle, rot[1] + offset_angle],
        [rot[2] - offset_angle, rot[2] + offset_angle],
        [trans[0] - 100, trans[0] + 100],
        [trans[1] - 300, trans[1] + 300],
        [trans[2] - 100, trans[2] + 100]
    ]
    bound = np.array(bound)

    optimizer = CMA(mean=rtvec, sigma=1.5, bounds=bound, population_size=50)

    params_history = []
    loss_history = []

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # 定义 Dice Loss 的权重
    dice_weight = 0.05

    for itr in tqdm(range(n_itrs), ncols=100):
        solutions = []
        op_loss_sum = 0
        gncc_loss_sum = 0
        dice_loss_sum = 0

        for _ in range(optimizer.population_size):
            x_eval = optimizer.ask()

            rot_tensor = torch.from_numpy(x_eval[:3]).float().unsqueeze(0).to(device)
            trans_tensor = torch.from_numpy(x_eval[3:]).float().unsqueeze(0).to(device)

            est_pose = convert(rot_tensor, trans_tensor, parameterization="euler_angles", convention="ZXY").to(device)

            estimate = reg(est_pose.compose(vert_mat))

            # 1. 计算 NCC Loss
            gncc_val = gncc_metric(estimate, gt_img)
            gncc_loss = 1.0 - gncc_val.item()

            # 2. 提取 moving 图像边缘，计算 Dice Loss
            est_line_np = extract_traditional_edge(estimate, threshold_ratio=0.08, margin_ratio=0.15)

            intersection = np.sum(est_line_np * gt_line_np)
            union = np.sum(est_line_np) + np.sum(gt_line_np)
            dice_loss = 1.0 - (2.0 * intersection + 1e-8) / (union + 1e-8)

            # 3. 组合总 Loss
            total_loss = gncc_loss + dice_weight * dice_loss

            solutions.append((x_eval, total_loss))
            op_loss_sum += total_loss
            gncc_loss_sum += gncc_loss
            dice_loss_sum += dice_loss

        optimizer.tell(solutions)

        cur_loss = op_loss_sum / optimizer.population_size
        avg_ncc = gncc_loss_sum / optimizer.population_size
        avg_dice = dice_loss_sum / optimizer.population_size

        loss_history.append(cur_loss)
        best_params = optimizer._mean
        params_history.append(best_params.tolist())

        tqdm.write(f"Itr {itr + 1:03d} | Total: {cur_loss:.4f} (NCC: {avg_ncc:.4f}, Dice: {avg_dice:.4f})")

        if itr > 20 and abs(loss_history[itr - 1] - cur_loss) < 1e-5:
            tqdm.write(f"Converged early in {itr + 1} iterations.")
            break

    plt.ioff()
    plt.close(fig)
    T2 = time.time()
    print(f'配准耗时: {T2 - T1:.4f} 秒')

    # --- 配准完成后的重叠可视化 ---
    with torch.no_grad():
        final_rot = torch.from_numpy(np.array(params_history[-1][:3])).float().unsqueeze(0).to(device)
        final_trans = torch.from_numpy(np.array(params_history[-1][3:])).float().unsqueeze(0).to(device)
        final_pose = convert(final_rot, final_trans, parameterization="euler_angles", convention="ZXY").to(device)
        final_drr = reg(final_pose.compose(vert_mat))

    # --- 新增：提取 bg_ground_truth 作为背景 ---
    bg_np = bg_img.squeeze().detach().cpu().numpy()
    final_drr_np = final_drr.squeeze().detach().cpu().numpy()
    final_drr_np = (final_drr_np - final_drr_np.min()) / (final_drr_np.max() - final_drr_np.min() + 1e-8)

    # 1. 图像像素重叠对比
    colors = plt.cm.Wistia(np.linspace(0, 1, 256))
    colors[:, 3] = np.linspace(0.0, 0.7, 256)
    yellow_cmp = mcolors.ListedColormap(colors)

    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.title("Final Registration Overlay (BG=Gray, Moving=Yellow)")
    # 此处使用 bg_np 作为底层背景
    plt.imshow(bg_np, cmap='gray')
    plt.imshow(final_drr_np, cmap=yellow_cmp)
    plt.axis('off')

    # 2. 边缘线条重叠对比
    final_line_np = extract_traditional_edge(final_drr, threshold_ratio=0.08, margin_ratio=0.15)

    plt.subplot(1, 2, 2)
    plt.title("Edge Lines Overlay (GT=Red, Moving=Green)")
    # 此处也换成 bg_np 作为背景，取代全黑背景
    plt.imshow(bg_np, cmap='gray')

    # GT 为红色，Moving 为绿色。重叠部分由于色彩混合会偏黄。
    plt.imshow(gt_line_np, cmap='Reds', alpha=0.8 * (gt_line_np > 0))
    plt.imshow(final_line_np, cmap='Greens', alpha=0.8 * (final_line_np > 0))
    plt.axis('off')

    plt.show()

    df = pd.DataFrame(params_history, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
    df["loss"] = loss_history
    csv_path = f'results/{samplename}/cmaes_pose.csv'
    df.to_csv(csv_path, index=False)
    print(f"优化结果已保存至: {csv_path}")


def pose_from_carm(sid, tx, ty, alpha, beta, gamma):
    rot = torch.tensor([[alpha, beta, gamma]])
    xyz = torch.tensor([[tx, sid, ty]])
    return convert(rot, xyz, parameterization="euler_angles", convention="ZXY")


def get_initial_parameters(true_params):
    alpha = true_params["alpha"] + np.random.uniform(-np.pi / 12, np.pi / 12)
    beta = true_params["beta"] + np.random.uniform(-np.pi / 18, np.pi / 18)
    gamma = true_params["gamma"] + np.random.uniform(-np.pi / 18, np.pi / 18)
    bx = true_params["bx"] + np.random.uniform(-30.0, 30.0)
    by = true_params["by"] + np.random.uniform(-20.0, 20.0)
    bz = true_params["bz"] + np.random.uniform(-30.0, 30.0)

    pose = pose_from_carm(by, bx, bz, alpha, beta, gamma).to(device)
    rotations, translations = pose.convert("euler_angles", "ZXY")
    return rotations, translations, pose


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 修改这里的参数配置
    caseName = 'case1'
    vert = 'L2'  # 只要修改这里为 L1 到 L5，程序就会自动去分割文件里找 21 到 25 的标签

    # 适配新的目录结构
    ct_path = f'Data/{caseName}/ct.nii.gz'
    vert_seg_path = f'Data/{caseName}/ct_seg.nii.gz'

    # 裁剪后的椎骨体积保存路径
    vert_save_path = f'Data/{caseName}/{caseName}_{vert}.nii.gz'

    # 将 vertName (即 vert) 作为参数传给 reg_method
    reg_method(ct_path, vert_seg_path, vert_save_path, f'{caseName}_{vert}', vertName=vert)
