import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import SimpleITK as sitk
from diffdrr.drr import DRR, Registration
from diffdrr.metrics import NormalizedCrossCorrelation2d, GradientNormalizedCrossCorrelation2d, MultiscaleNormalizedCrossCorrelation2d
from monai.losses import DiceLoss, DiceCELoss, GeneralizedDiceLoss, SSIMLoss
from diffdrr.visualization import plot_drr
from diffdrr.visualization import animate
from base64 import b64encode
from monai.transforms import LoadImage, ScaleIntensity, Resize
from diffdrr.detector import diffdrr_to_deepdrr
# import cv2
from CNCC import CannyCrossCorrelation2d
import time
import torch.nn as nn
import nibabel as nib
from line_infer import infer_method
from tools import get_initial_parameters
from monai.networks.utils import one_hot
from losses.MaskDistLoss import MaskDistanceLoss
from losses.MaskedGNCCLoss import MaskGradientNormalizedCrossCorrelation2d, MaskNormalizedCrossCorrelation2d
from tools import get_drr, get_lineCenter_offset
from sklearn.metrics import mean_absolute_error
import math
import os


def reg_method():
    # 读取椎骨CT
    reader = LoadImage(ensure_channel_first=True, image_only=False)
    np.random.seed(442)
    # ct_Dir = 'Data/gncc_data/tuodao/dingjunmei/dingjunmei.nii.gz'
    # vert_tissue_Dir = 'Data/gncc_data/tuodao/dingjunmei/dingjunmei_L2.nii.gz'
    vert_Dir = 'Data/gncc_data/tuodao/dingjunmei/dingjunmei_L2.nii.gz'
    if len(os.path.split(vert_Dir)[-1].split('_')) > 2:
        case_name = '_'.join(os.path.split(vert_Dir)[-1].split('_')[:-1])
    else:
        case_name = os.path.split(vert_Dir)[-1].split('.')[0]
    print(case_name)
    excel_path = 'Data/gncc_data/total_data_info.csv'
    info_data = pd.read_csv(excel_path)
    offsetx = info_data.loc[info_data['name'] == case_name]['offsetx'].item()
    offsety = info_data.loc[info_data['name'] == case_name]['offsety'].item()
    offsetz = info_data.loc[info_data['name'] == case_name]['offsetz'].item()
    offset_trans = np.array([- offsetx, offsety, offsetz])
    # ap_vert_gt, _, _, _, _ = get_drr(vert_tissue_Dir, offset_trans, 'ap', tissue=True)
    # la_vert_gt, _, _, _, _ = get_drr(vert_tissue_Dir, offset_trans, 'lar', tissue=True)
    _, drr_gene, ap_true_params, gt_rotations, gt_translations = get_drr(vert_Dir, offset_trans, 'ap', tissue=True)
    # ap_x_line_path = 'Data/gncc_data/tuodao/bimeihua/X/pzp_ap_L3_line.nii.gz'
    # la_x_line_path = 'Data/gncc_data/tuodao/bimeihua/X/pzp_la_L3_line.nii.gz'
    ap_x_ray_path = 'Data/gncc_data/tuodao/dingjunmei/X/djm_resized_ap.nii.gz'
    la_x_ray_path = 'Data/gncc_data/tuodao/dingjunmei/X/djm_resized_la.nii.gz'
    ap_mask_path = 'Data/gncc_data/tuodao/dingjunmei/X/djm_resized_ap_L2_bbx.nii.gz'
    la_mask_path = 'Data/gncc_data/tuodao/dingjunmei/X/djm_resized_la_L2_bbx.nii.gz'
    ap_mask = reader(ap_mask_path)
    la_mask = reader(la_mask_path)
    ap_xray = reader(ap_x_ray_path)
    la_xray = reader(la_x_ray_path)
    # gt_ap, _, _, _, _ = get_drr(ct_Dir)
    # gt_la, _, _, _, _ = get_drr(ct_Dir, poseture='lar', tissue=True)
    # out_ap_gt = torch.permute(gt_ap, (0, 1, 3, 2))
    # out_la_gt = torch.permute(gt_la, (0, 1, 3, 2))
    # out_ap = sitk.GetImageFromArray(out_ap_gt.squeeze().detach().cpu().numpy())
    # sitk.WriteImage(out_ap, 'Data/diff_lines/pzp_L3_ap_gt.nii.gz')
    # out_bg = sitk.GetImageFromArray(out_la_gt.squeeze().detach().cpu().numpy())
    # sitk.WriteImage(out_bg, 'Data/gncc_data/exp2024.3.27/dualView/{name}_la_gt.nii.gz'.format(name=case_name))
    # sitk.WriteImage(out_la, 'Data/diff_lines/pzp_L3_la_gt.nii.gz')
    # ap_gt_line = infer_method(ap_vert_gt)
    # ap_gt_line = torch.argmax(ap_gt_line, dim=1)
    # print(ap_gt_line.shape)
    # la_gt_line = infer_method(la_vert_gt)
    # la_gt_line = torch.argmax(la_gt_line, dim=1)
    # print(la_gt_line.shape)
    scaleInen = ScaleIntensity()
    ap_vert_gt = scaleInen(ap_xray[0])
    la_vert_gt = scaleInen(la_xray[0])
    ap_vert_gt = torch.unsqueeze(ap_vert_gt, dim=0).to(device)
    la_vert_gt = torch.unsqueeze(la_vert_gt, dim=0).to(device)
    ap_mask_gt = torch.unsqueeze(ap_mask[0][:, :, :, 0], dim=0).to(device)
    la_mask_gt = torch.unsqueeze(la_mask[0][:, :, :, 0], dim=0).to(device)
    # ap_gt_line = ap_x_line[0][:, :, :, 0]
    # la_gt_line = la_x_line[0][:, :, :, 0]
    # 随机变换待配准椎骨的初始位姿
    # ap_ini_rotations, ap_ini_translations = get_initial_parameters(ap_true_params, device)
    mov_drr = drr_gene(
        gt_rotations,
        gt_translations,
        parameterization="euler_angles",
        convention="ZYX",
    )
    ini_drr = torch.permute(mov_drr, (0, 1, 3, 2))
    print(ap_mask_gt.shape)
    bbx_gt = torch.permute(ap_mask_gt, (0, 1, 3, 2))
    # bbx_gt = torch.permute(ap_vert_gt, (0, 1, 3, 2))
    # out_img = sitk.GetImageFromArray(ini_drr.squeeze().detach().cpu().numpy())
    # sitk.WriteImage(out_img, 'Data/gncc_data/tuodao/peizongping/pzp_L3_ap_ini.nii.gz')
    mov_line = infer_method(mov_drr)
    pred_mask = torch.argmax(mov_line, dim=1)
    # line_tensor = torch.permute(ap_gt_line.unsqueeze(0), (0, 1, 3, 2))
    pred_mask = torch.permute(pred_mask, (0, 2, 1))
    coarse_x, coarse_y = get_lineCenter_offset(ini_drr, bbx_gt)
    coarse_trans = torch.tensor([[coarse_x, 0, coarse_y-50]]).to(device)
    ap_ini_translations = gt_translations + coarse_trans

    # 可以生成带梯度的drr图像
    coarse_reg = Registration(
        drr_gene,
        gt_rotations.clone(),
        ap_ini_translations.clone(),
        parameterization="euler_angles",
        convention="ZYX",
        dual_view=True
    )
    coarse_drr = coarse_reg()
    plt.subplot(2, 2, 1)
    plt.imshow(ap_vert_gt.squeeze().detach().cpu().numpy())
    plt.subplot(2, 2, 2)
    plt.imshow(mov_drr.squeeze().detach().cpu().numpy())
    plt.subplot(2, 2, 3)
    plt.imshow(coarse_drr[0, :].squeeze().detach().cpu().numpy())
    plt.subplot(2, 2, 4)
    plt.imshow(coarse_drr[1, :].squeeze().detach().cpu().numpy())
    plt.show()
    # 双视图的位姿更新
    gt_list = [ap_vert_gt, la_vert_gt]
    # line_list = [ap_gt_line, la_gt_line]
    mask_list = [ap_mask_gt, la_mask_gt]
    optimize(coarse_reg, gt_list, case_name, scaleInen, gt_rotations, gt_translations, 0.01, 1e1, optimizer="adam", gt_lines=mask_list)
    # params = optimize(reg, x_img_tensor, scaleInen, tensor_mask, 1e-2, 1e1, momentum=0.9, dampening=0.1)
    # bg_img_tensor = torch.permute(ground_truth, (0, 1, 3, 2))
    # animate_in_browser(params, len(params), drr, ground_truth)
    del drr_gene


def optimize(
        reg: Registration,
        gt_imgs,
        samplename,
        scaler,
        ini_rot,
        ini_trans,
        lr_rotations=5.3e-2,
        lr_translations=7.5e1,
        momentum=0,
        dampening=0,
        n_itrs=300,
        optimizer="sgd",  # 'sgd' or `adam`
        gt_lines=None
):
    T1 = time.time()
    # 损失函数，先尝试归一化互相关loss
    GNCC_loss = MaskGradientNormalizedCrossCorrelation2d()
    # NCC_loss = MultiscaleNormalizedCrossCorrelation2d(patch_sizes=[5, None], patch_weights=[0.5, 1.0])
    NCC_loss = MaskNormalizedCrossCorrelation2d()
    # SSIM_loss = SSIMLoss(spatial_dims=2)
    # 选择优化器
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(
            [
                {"params": [reg.rotation], "lr": lr_rotations},
                {"params": [reg.translation], "lr": lr_translations},
            ],
            momentum=momentum,
            dampening=dampening,
            maximize=True,
        )
    else:
        optimizer = torch.optim.Adam(
            [
                {"params": [reg.rotation], "lr": lr_rotations},
                {"params": [reg.translation], "lr": lr_translations},
            ],
            maximize=True,
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5, last_epoch=-1)
    params = []
    losses = []
    # ground_truth = one_hot(ground_truth, num_classes=2)
    # 迭代循环
    for itr in tqdm(range(n_itrs), ncols=100):
        # Save the current set of parameters
        alpha, beta, gamma = reg.get_rotation().squeeze().tolist()
        bx, by, bz = reg.get_translation().squeeze().tolist()
        # _, _, bz = reg.get_translation()[1, :].squeeze().tolist()
        # print(reg.translation)
        params.append([i for i in [alpha, beta, gamma, bx, by, bz]])
        # Run the optimization loop
        optimizer.zero_grad()
        # 待配准的drr图像
        estimate = reg()
        # ap_pred_line = infer_method(estimate[0, :].unsqueeze(0))
        # la_pred_line = infer_method(estimate[1, :].unsqueeze(0))
        # print(estimate.shape)

        # plt.subplot(1, 2, 1)
        # plt.imshow(torch.argmax(ap_pred_line, dim=1).squeeze().detach().cpu().numpy())
        # plt.subplot(1, 2, 2)
        # plt.imshow(torch.argmax(la_pred_line, dim=1).squeeze().detach().cpu().numpy())
        # plt.show()
        # 将图像进行z-score归一化
        estimate = scaler(estimate)
        # print(estimate[0, :].unsqueeze(0).shape)
        # print(gt_imgs[0].shape)
        ap_ncc_loss = NCC_loss(estimate[0, :].unsqueeze(0).float(), gt_imgs[0].float(), gt_lines[0].float())
        la_ncc_loss = NCC_loss(estimate[1, :].unsqueeze(0).float(), gt_imgs[1].float(), gt_lines[1].float())
        # ap_ssim_loss = SSIM_loss(estimate[0, :].unsqueeze(0).float(), gt_imgs[0].float())
        # la_ssim_loss = SSIM_loss(estimate[1, :].unsqueeze(0).float(), gt_imgs[1].float())
        # ap_gncc_loss = GNCC_loss(estimate[0, :].unsqueeze(0).float(), gt_imgs[0].float(), gt_lines[0].float())
        # la_gncc_loss = GNCC_loss(estimate[1, :].unsqueeze(0).float(), gt_imgs[1].float(), gt_lines[1].float())
        # ap_line_loss = NCC_loss(ap_pred_line[:, 1, :].unsqueeze(0).float(), gt_lines[0].unsqueeze(0).float())
        # la_line_loss = NCC_loss(la_pred_line[:, 1, :].unsqueeze(0).float(), gt_lines[1].unsqueeze(0).float())
        # loss = (ap_ncc_loss + la_ncc_loss) / 2 * 0.9 + (ap_gncc_loss + la_gncc_loss) / 2 * 0.1
        loss = (ap_ncc_loss + la_ncc_loss) / 2
        # loss = la_ncc_loss
        # print("line loss: " + str((ap_line_loss.item() + la_line_loss.item()) / 2))
        print(loss.item())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        scheduler.step()
        if abs(losses[itr - 1] - loss.item()) < 0.000001 and itr > 5:
            tqdm.write(f"Converged in {itr} iterations")
            # print(reg.get_rotation() - ini_rot, reg.get_translation() - ini_trans)
            # print(ini_rot.cpu())
            # print(reg.get_rotation()[0, :].unsqueeze(0))
            rot_mae1 = mean_absolute_error(ini_rot.cpu(), reg.get_rotation(), multioutput='raw_values')
            trans_mae1 = mean_absolute_error(ini_trans.cpu(), reg.get_translation(), multioutput='raw_values')
            # rot_mae2 = mean_absolute_error(ini_rot.cpu(), reg.get_rotation()[1, :].unsqueeze(0), multioutput='raw_values')
            # trans_mae2 = mean_absolute_error(ini_trans.cpu(), reg.get_translation()[1, :].unsqueeze(0), multioutput='raw_values')
            print('x轴旋转方向上误差:{:.4f}°, y轴旋转方向上误差:{:.4f}°, z轴旋转方向上误差:{:.4f}°'.
                  format(rot_mae1[1] / math.pi * 180, rot_mae1[0] / math.pi * 180, rot_mae1[2] / math.pi * 180))
            print('x平移方向上误差:{:.4f} mm, y平移方向上误差:{:.4f} mm, z平移方向上误差:{:.4f} mm'.
                  format(trans_mae1[0], trans_mae1[2], trans_mae1[1]))
            T2 = time.time()
            print('配准耗时为:%.4s秒' % (T2 - T1))
            plt.subplot(2, 2, 1)
            plt.imshow(estimate[0, :].squeeze().detach().cpu().numpy(), cmap='gray')
            plt.subplot(2, 2, 2)
            plt.imshow(gt_imgs[0].squeeze().detach().cpu().numpy(), cmap='gray')
            plt.subplot(2, 2, 3)
            plt.imshow(estimate[1, :].squeeze().detach().cpu().numpy(), cmap='gray')
            plt.subplot(2, 2, 4)
            plt.imshow(gt_imgs[1].squeeze().detach().cpu().numpy(), cmap='gray')
            plt.show()
            break
    # 记录了每次迭代生成的位姿，优化结束后，可以运行results_metric.py中的函数可视化配准过程
    df = pd.DataFrame(params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
    df["loss"] = losses
    print(df)
    # 用于存储pose，换视图训练，最好改名字
    df.to_csv('results/tuodao/{}_dualview_pose.csv'.format(samplename), index=False)
    # return df


def animate_in_browser(df, max_length, drr_mov, gt):
    n = max_length - len(df)
    df = pd.concat([df, df.iloc[[-1] * n]])
    out = animate(
        "<bytes>",
        df,
        drr_mov,
        ground_truth=gt,
        verbose=True,
        device=device,
        extension=".webp",
        duration=5,
        parameterization="euler_angles",
        convention="ZYX",
    )
    html_filename = "drr.html"

    # 打开HTML文件并写入内容
    with open(html_filename, "w") as f:
        f.write("<html><body>")
        f.write(f"""<img src='{"data:img/gif;base64," + b64encode(out).decode()}'>""")
        f.write("</body></html>")

    print("HTML文件创建成功")


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    reg_method()
