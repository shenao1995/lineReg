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
from tools import get_drr, get_lineCenter_offset, crop_ct_vert, gaussian_preprocess, resample_img, bbx_crop_gt_vert
from DualView_CMAES_Reg import preliminary_process, line_preprocess, get_cam_wld_mat


def reg_method(origin_ct_path, seg_path, xray_paths, boundingbox_paths, case_name, save_dir, x_dirs,
               cam_wld, line_paths=None):
    # 读取椎骨CT
    offsetx, offsety, offsetz, vert_img = crop_ct_vert(origin_ct_path, seg_path, save_dir)
    offset_trans = np.array([-offsetx, offsety, offsetz])
    ap_gt, cropped_spacing = preliminary_process(xray_paths[0], boundingbox_paths[0], x_dirs[0])
    la_gt, _ = preliminary_process(xray_paths[1], boundingbox_paths[1], x_dirs[1])
    # gt_mask = torch.tensor(x_mask_arr)
    # gt_mask = torch.unsqueeze(gt_mask, dim=0).to(device)
    # gt_mask = torch.unsqueeze(gt_mask, dim=0).to(device)
    # gt_mask = torch.permute(gt_mask, (0, 1, 3, 2))
    ap_line = line_preprocess(line_paths[0])
    img_center = np.array([-3.9975469970703125e+02, 1.5983113098144531e+02, 8.2500099182128906e+01])
    xray_center = np.array([7.0934796142578125e+02, -1.8016872406005859e+01, 8.2500099182128906e+01])
    SDR = np.sqrt(np.sum((img_center - xray_center) ** 2))
    print(SDR / 2)
    used_gt, drr_gene, true_params, ini_rot, ini_trans = get_drr(cam_wld, img_meta=vert_img,
                                                                 SDR=SDR / 2,
                                                                 DELX=cropped_spacing,
                                                                 offset_trans=offset_trans,
                                                                 tissue=True)
    scaleInen = ScaleIntensity()
    mov_drr = drr_gene(
        ini_rot,
        ini_trans,
        parameterization="euler_angles",
        convention="ZYX"
    )
    print(mov_drr.shape)
    ini_drr = torch.permute(mov_drr, (0, 1, 3, 2))
    bbx_gt = torch.permute(ap_gt, (0, 1, 3, 2))
    # print(bbx_gt.shape)
    # out_img = sitk.GetImageFromArray(ini_drr.squeeze().detach().cpu().numpy())
    # sitk.WriteImage(out_img, 'Data/natong/case3/L5_mov.nii.gz')
    mov_line = infer_method(mov_drr)
    pred_mask = torch.argmax(mov_line, dim=1)
    # pred_mask = torch.permute(pred_mask.unsqueeze(0), (0, 1, 3, 2))
    # line_tensor = torch.permute(gt_line, (0, 1, 3, 2))
    # pred_mask = torch.permute(pred_mask, (0, 2, 1))
    # print(ini_drr.shape)
    # 90., 92., 255., 193.
    # 128., 19., 255., 127.
    # coarse_x, coarse_y = get_lineCenter_offset(ini_drr, bbx_gt)
    # coarse_x, coarse_y = get_lineCenter_offset(ini_drr, bbx_gt)
    # coarse_trans = torch.tensor([[coarse_x, 0, coarse_y]]).to(device)
    # print(coarse_x, coarse_y)
    # translations = gt_translations + coarse_trans
    # translations = torch.tensor([[0.0, 0.0, 0.0]]).to(device)
    # print(coarse_trans)
    # print(gt_rotations)
    # print(gt_translations)
    # 可以生成带梯度的drr图像
    coarse_reg = Registration(
        drr_gene,
        ini_rot.clone(),
        ini_trans.clone(),
        parameterization="euler_angles",
        convention="ZYX",
        dual_view=True
    )
    # coarse_drr = coarse_reg()
    # coarse_drr = torch.permute(coarse_drr, (0, 1, 3, 2))
    plt.subplot(2, 2, 1)
    plt.imshow(ap_gt.squeeze().detach().cpu().numpy())
    plt.subplot(2, 2, 2)
    plt.imshow(la_gt.squeeze().detach().cpu().numpy())
    plt.subplot(2, 2, 3)
    plt.imshow(pred_mask.squeeze().detach().cpu().numpy())
    plt.subplot(2, 2, 4)
    plt.imshow(ap_line.squeeze().detach().cpu().numpy())
    plt.show()
    plt.close()
    # 优化算法
    # optimize(drr_gene, [ap_gt, la_gt], ap_line, case_name, scaleInen, ini_rot, ini_trans)
    optimize(coarse_reg, [ap_gt, la_gt], case_name, scaleInen, ini_rot, ini_trans, 0.01, 1e1, optimizer="adam")
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
        ap_ncc_loss = NCC_loss(estimate[0, :].unsqueeze(0).float(), gt_imgs[0].float())
        la_ncc_loss = NCC_loss(estimate[1, :].unsqueeze(0).float(), gt_imgs[1].float())
        # ap_ssim_loss = SSIM_loss(estimate[0, :].unsqueeze(0).float(), gt_imgs[0].float())
        # la_ssim_loss = SSIM_loss(estimate[1, :].unsqueeze(0).float(), gt_imgs[1].float())
        # ap_gncc_loss = GNCC_loss(estimate[0, :].unsqueeze(0).float(), gt_imgs[0].float())
        # la_gncc_loss = GNCC_loss(estimate[1, :].unsqueeze(0).float(), gt_imgs[1].float())
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
    df.to_csv('results/tuodao/{}_dualview_adam_pose.csv'.format(samplename), index=False)
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
    caseName = 'dukemei'
    ct_path = 'Data/tuodao/{}/{}.nii.gz'.format(caseName, caseName)
    vert_seg_path = 'Data/tuodao/{}/L3_seg.nii.gz'.format(caseName)
    ap_gt_path = 'Data/tuodao/{}/X/dukemei_ap.nii.gz'.format(caseName)
    ap_bbx_path = 'Data/tuodao/{}/X/L3_bbx_ap.nii.gz'.format(caseName)
    la_gt_path = 'Data/tuodao/{}/X/dukemei_la.nii.gz'.format(caseName)
    la_bbx_path = 'Data/tuodao/{}/X/L3_bbx_la.nii.gz'.format(caseName)
    vert_save_path = 'Data/tuodao/{}/{}_L3.nii.gz'.format(caseName, caseName)
    resized_x_ap_save_path = 'Data/tuodao/{}/X/{}_resized_x_ap.nii.gz'.format(caseName, caseName)
    resized_x_la_save_path = 'Data/tuodao/{}/X/{}_resized_x_la.nii.gz'.format(caseName, caseName)
    line_ap_path = 'Data/tuodao/{}/X/L3_line_ap.nii.gz'.format(caseName)
    line_la_path = 'Data/tuodao/{}/X/L3_line_la.nii.gz'.format(caseName)
    x_vect = np.array([1.5833039581775665e-01, 9.8738616704940796e-01, 0.0])
    y_vect = np.array([0.0, 0.0, 1.0])
    cam_mat = get_cam_wld_mat(x_vect, y_vect)
    reg_method(ct_path, vert_seg_path, [ap_gt_path, la_gt_path], [ap_bbx_path, la_bbx_path], '{}_L3'.format(caseName),
               vert_save_path, [resized_x_ap_save_path, resized_x_la_save_path], cam_mat, [line_ap_path, line_la_path])
