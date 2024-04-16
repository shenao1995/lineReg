import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import SimpleITK as sitk
from diffdrr.drr import DRR, Registration
from diffdrr.metrics import NormalizedCrossCorrelation2d, GradientNormalizedCrossCorrelation2d, \
    MultiscaleNormalizedCrossCorrelation2d
from monai.losses import DiceLoss, DiceCELoss, GeneralizedDiceLoss, HausdorffDTLoss
from diffdrr.visualization import plot_drr
from diffdrr.visualization import animate
from base64 import b64encode
from monai.transforms import LoadImage, ScaleIntensity, Resize
import cv2
import time
import torch.nn as nn
import nibabel as nib
from line_infer import infer_method
from tools import get_initial_parameters
from monai.networks import one_hot
from monai.networks.utils import one_hot
from losses.MaskDistLoss import MaskDistanceLoss
import math
from sklearn.metrics import mean_absolute_error
from tools import get_drr, get_lineCenter_offset
import os
from losses.ZNCC import ZNCC


def reg_method():
    # 读取椎骨CT
    reader = LoadImage(ensure_channel_first=True, image_only=False)
    # np.random.seed(33)
    # Make the ground truth X-ray
    ct_Dir = 'Data/gncc_data/tuodao/peizongping/bimeihua.nii.gz'
    vert_tissue_Dir = 'Data/gncc_data/tuodao/bimeihua/bimeihua_L5_tissue.nii.gz'
    vert_Dir = 'Data/gncc_data/tuodao/dingjunmei/dingjunmei_L2.nii.gz'
    # sample_name = os.path.split(vert_Dir)[-1].split('_')
    if len(os.path.split(vert_Dir)[-1].split('_')) > 2:
        case_name = '_'.join(os.path.split(vert_Dir)[-1].split('_')[:-1])
    else:
        case_name = os.path.split(vert_Dir)[-1].split('.')[0]
    # case_name = '_'.join(os.path.split(vert_Dir)[-1].split('_')[:-1])
    print(case_name)
    excel_path = 'Data/gncc_data/total_data_info.csv'
    info_data = pd.read_csv(excel_path)
    offsetx = info_data.loc[info_data['name'] == case_name]['offsetx'].item()
    offsety = info_data.loc[info_data['name'] == case_name]['offsety'].item()
    offsetz = info_data.loc[info_data['name'] == case_name]['offsetz'].item()
    # ctDir = 'Data/gncc_data/CT_L4.nii.gz'
    offset_trans = np.array([- offsetx, offsety, offsetz])
    used_gt, drr_gene, true_params, gt_rotations, gt_translations = get_drr(vert_Dir, offset_trans, tissue=True)
    # ground_truth, _, _, _, _ = get_drr(vert_tissue_Dir, offset_trans)
    # gt_ct, _, _, _, _ = get_drr(ct_Dir)
    # out_gt = torch.permute(ground_truth, (0, 1, 3, 2))
    # out_img = sitk.GetImageFromArray(out_gt.squeeze().detach().cpu().numpy())
    # sitk.WriteImage(out_img, 'Data/gncc_data/exp2024.3.27/{name}_ap_gt.nii.gz'.format(name=case_name))
    # gt_line = infer_method(used_gt)
    # gt_line = torch.argmax(gt_line, dim=1)
    # print(gt_line.shape)
    # 读取目标椎骨的边缘线
    x_line_path = 'Data/gncc_data/tuodao/dingjunmei/X/djm_ap_L2_line.nii.gz'
    x_ray_path = 'Data/gncc_data/tuodao/dingjunmei/X/djm_ap_L2.nii.gz'
    # mask_path = 'Data/gncc_data/tuodao/peizongping/peizongping_180_x_L3_seg.nii.gz'
    # bg_path = 'Data/gncc_data/91B_resized.nii.gz'
    line_img = reader(x_line_path)
    xray = reader(x_ray_path)
    # mask = reader(mask_path)
    scaleInen = ScaleIntensity()
    # ground_truth = scaleInen(ground_truth)
    # print(xray[0].shape)
    ground_truth = scaleInen(xray[0])
    # print(ground_truth.shape)
    # print(line_img[0].shape)
    gt_line = line_img[0][:, :, :, 0]
    # gt_line = line_img[0]
    # gt_line = scaleInen(gt_line)
    # gt_mask = mask[0][:, :, :, 0]
    # gt_img = xray[0]
    # print(gt_img.shape)
    # resize = Resize(spatial_size=(256, 256), mode='bilinear', align_corners=True)
    # ground_truth = resize(ground_truth)
    # gt_line = resize(gt_line)
    ground_truth = torch.unsqueeze(ground_truth, dim=0).to(device)
    gt_line = torch.unsqueeze(gt_line, dim=0).to(device)
    # print(gt_line.shape)
    # print(ground_truth.shape)
    # mask_tensor = torch.unsqueeze(gt_mask, dim=0).to(device)
    # bg_img_tensor = torch.unsqueeze(bg_img[0], dim=0).to(device)
    # line_tensor = torch.permute(line_tensor, (0, 1, 3, 2))
    # ground_truth = torch.permute(ground_truth, (0, 1, 3, 2))
    # out_img = sitk.GetImageFromArray(ground_truth.squeeze().detach().cpu().numpy())
    # sitk.WriteImage(out_img, 'Data/diff_lines/peizongping_L3_drr.nii.gz')
    # x_img_tensor = torch.permute(x_img_tensor, (0, 1, 3, 2))
    # plt.subplot(1, 2, 1)
    # plt.imshow(x_img_tensor.squeeze().detach().cpu().numpy())
    # plt.subplot(1, 2, 2)
    # plt.imshow(ground_truth.squeeze().detach().cpu().numpy())
    # plt.show()
    # 随机变换待配准椎骨的初始位姿
    # rotations, translations = get_initial_parameters(true_params, device)
    # print(rotations)
    # print(translations)
    # print(rotations-ini_rotations, translations-ini_translations)
    mov_drr = drr_gene(
        gt_rotations,
        gt_translations,
        parameterization="euler_angles",
        convention="ZYX",
    )
    # ini_drr = torch.permute(ini_drr, (0, 1, 3, 2))
    # out_img = sitk.GetImageFromArray(ini_drr.squeeze().detach().cpu().numpy())
    # sitk.WriteImage(out_img, 'Data/diff_lines/peizongping_L3_pred.nii.gz')
    mov_line = infer_method(mov_drr)
    pred_mask = torch.argmax(mov_line, dim=1)
    pred_mask = torch.permute(pred_mask.unsqueeze(0), (0, 1, 3, 2))
    line_tensor = torch.permute(gt_line, (0, 1, 3, 2))

    # pred_mask = torch.permute(pred_mask, (0, 2, 1))
    print(pred_mask.shape)
    coarse_x, coarse_y = get_lineCenter_offset(pred_mask, line_tensor)
    coarse_trans = torch.tensor([[coarse_x, 0, coarse_y - 50]]).to(device)
    translations = gt_translations + coarse_trans
    # print(coarse_trans)
    # 61.3930, -10.5655, 78.0537
    # 可以生成带梯度的drr图像
    coarse_reg = Registration(
        drr_gene,
        gt_rotations.clone(),
        translations.clone(),
        parameterization="euler_angles",
        convention="ZYX",
    )
    coarse_drr = coarse_reg()
    plt.subplot(1, 3, 1)
    plt.imshow(line_tensor.squeeze().detach().cpu().numpy())
    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask.squeeze().detach().cpu().numpy())
    plt.subplot(1, 3, 3)
    plt.imshow(coarse_drr.squeeze().detach().cpu().numpy())
    plt.show()
    plt.close()
    # 优化算法
    optimize(coarse_reg, ground_truth, gt_line, case_name, scaleInen, gt_rotations, gt_translations, 0.01, 1e1, optimizer="adam")
    # params = optimize(reg, x_img_tensor, scaleInen, tensor_mask, 1e-2, 1e1, momentum=0.9, dampening=0.1)
    # bg_img_tensor = torch.permute(ground_truth, (0, 1, 3, 2))
    # animate_in_browser(params, len(params), drr, ground_truth)
    del drr_gene


def optimize(
        reg: Registration,
        ground_truth,
        gt_line,
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
):
    T1 = time.time()
    # 损失函数，先尝试的归一化互相关loss
    GNCC_loss = GradientNormalizedCrossCorrelation2d()
    NCC_loss = MultiscaleNormalizedCrossCorrelation2d()
    # ZNCC_loss = ZNCC()
    # HD_loss = HausdorffDTLoss()
    # DCE_loss = DiceCELoss(to_onehot_y=True, softmax=True)
    # criterion = GeneralizedDiceLoss(include_background=False, to_onehot_y=True)
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
        # print(reg.translation)
        params.append([i for i in [alpha, beta, gamma, bx, by, bz]])
        # Run the optimization loop
        optimizer.zero_grad()
        # 待配准的drr图像
        estimate = reg()
        # 调用神经网络提取椎骨的边缘线
        # pred_line = infer_method(estimate)
        plt.subplot(1, 2, 1)
        # plt.imshow(torch.argmax(pred_line, dim=1).squeeze().detach().cpu().numpy())
        # # plt.imshow(pred_line.squeeze()[1, ].detach().cpu().numpy())
        plt.imshow(ground_truth.squeeze().detach().cpu().numpy())
        plt.subplot(1, 2, 2)
        plt.imshow(estimate.squeeze().detach().cpu().numpy())
        plt.show()
        # plt.close()
        # pred_line = torch.argmax(pred_line, dim=1)
        # print(pred_line.shape)
        # print(ground_truth.shape)
        # pred_line = torch.tensor(pred_line, requires_grad=True, device=device)
        # print(torch.argmax(pred_line, dim=1).shape)
        # loss1 = dice_loss(pred_line, ground_truth)
        # loss2 = gncc_loss(estimate, xray, v_mask)
        # loss = loss1 + loss2
        # arg_line = torch.argmax(pred_line, dim=1)
        # bbx1 = masks_to_boxes(ground_truth.squeeze(0))
        # bbx2 = masks_to_boxes(arg_line)
        # print(bbx1)
        # print(bbx2)
        # sig_pred = sig_pred.unsqueeze(0)
        # dce_loss = DCE_loss(pred_line, gt_line.unsqueeze(0))
        # print(pred_line[:, 1, :].shape)
        # 将图像进行z-score归一化
        estimate = scaler(estimate)
        # line_loss = NCC_loss(pred_line[:, 1, :].unsqueeze(0).float(), gt_line.unsqueeze(0).float())
        # print(gt_line.shape)
        # print(pred_line[:, 1, :].unsqueeze(0).shape)
        # line_loss = NCC_loss(pred_line[:, 1, :].unsqueeze(0).float(), gt_line.float())
        # hd_loss = HD_loss(pred_line[:, 1, :].unsqueeze(0).float(), gt_line.unsqueeze(0).float())
        # print(hd_loss.item())
        ncc_loss = NCC_loss(estimate.float(), ground_truth.float())
        # gncc_loss = GNCC_loss(estimate.float(), ground_truth.float())
        loss = ncc_loss
        # loss = ncc_loss
        # print(gncc_loss.item())
        # print(dce_loss)
        # print("gncc=" + str(gncc_loss.item()))
        # loss.requires_grad_(True)
        # print(loss.item())
        loss.backward()
        optimizer.step()
        # print(loss.item())
        losses.append(loss.item())
        # print(abs(losses[itr - 1] - loss.item()))
        scheduler.step()
        if abs(losses[itr - 1] - loss.item()) < 0.000001 and itr > 5:
            # if loss > 0.95:
            # if loss > 0.8:
            tqdm.write(f"Converged in {itr} iterations")
            rot_mae = mean_absolute_error(ini_rot.cpu(), reg.get_rotation(), multioutput='raw_values')
            trans_mae = mean_absolute_error(ini_trans.cpu(), reg.get_translation(), multioutput='raw_values')
            # print(reg.get_rotation() - ini_rot.cpu(), reg.get_translation() - ini_trans.cpu())
            print('x轴旋转方向上误差:{:.4f}°, y轴旋转方向上误差:{:.4f}°, z轴旋转方向上误差:{:.4f}°'.
                  format(rot_mae[1] / math.pi * 180, rot_mae[0] / math.pi * 180, rot_mae[2] / math.pi * 180))
            print('x平移方向上误差:{:.4f} mm, y平移方向上误差:{:.4f} mm, z平移方向上误差:{:.4f} mm'.
                  format(trans_mae[0], trans_mae[2], trans_mae[1]))
            T2 = time.time()
            print('配准耗时为:%.4s秒' % (T2 - T1))
            break
    # 记录了每次迭代生成的位姿，优化结束后，可以运行results_metric.py中的函数可视化配准过程
    df = pd.DataFrame(params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
    df["loss"] = losses
    print(df)
    df.to_csv('results/tuodao/{}_pose.csv'.format(samplename), index=False)
    return df


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
