import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import SimpleITK as sitk
from diffdrr.drr import DRR
from diffdrr.registration import Registration
from diffdrr.metrics import NormalizedCrossCorrelation2d, GradientNormalizedCrossCorrelation2d, \
    MultiscaleNormalizedCrossCorrelation2d
from monai.losses import DiceLoss, DiceCELoss, GeneralizedDiceLoss, SSIMLoss
from diffdrr.visualization import plot_drr
from diffdrr.visualization import animate
from base64 import b64encode
from monai.transforms import LoadImage, ScaleIntensity, Resize
# import cv2
from CNCC import CannyCrossCorrelation2d
import time
import torch.nn as nn
import nibabel as nib
from tools import get_initial_parameters
from monai.networks.utils import one_hot
from losses.MaskedGNCCLoss import MaskGradientNormalizedCrossCorrelation2d, MaskNormalizedCrossCorrelation2d
from tools import resample_img, HE_optimize, read_xml, get_ext_pose, update_pose, get_wld_pose
from sklearn.metrics import mean_absolute_error
import math
import os
from losses.ZNCC import ncc, gradncc
from cmaes import CMA, CMAwM
from diffdrr.pose import convert, RigidTransform
from diffdrr.data import read


def reg_method(origin_ct_path, seg_path, xray_paths, case_name, x_saves, xml_paths):
    # 读取椎骨CT
    ap_Xdir, ap_Ydir, ap_spacing, SDD, Xray_H, ap_Wld_Offset = read_xml(xml_paths[0])
    la_Xdir, la_Ydir, la_spacing, _, _, la_Wld_Offset = read_xml(xml_paths[1])
    ap_gt = xray_process(xray_paths[0], ap_spacing, x_saves[0])
    la_gt = xray_process(xray_paths[1], la_spacing, x_saves[1])
    ini_pose = torch.zeros(1, 6).to(device)
    # print(ini_pose.shape)
    DELX = Xray_H / 122 * ap_spacing
    print(DELX)
    ap_extrinsic_update = get_ext_pose(ap_Xdir, ap_Ydir, ap_Wld_Offset, ini_pose, view='ap')
    la_extrinsic_update = get_ext_pose(la_Xdir, la_Ydir, la_Wld_Offset, ini_pose, view='la')
    # ap_init = update_pose(ap_extrinsic_update, ini_pose)
    # la_init = update_pose(la_extrinsic_update, ini_pose)
    ap_init = get_wld_pose(ap_extrinsic_update)
    la_init = get_wld_pose(la_extrinsic_update)
    subject = read(origin_ct_path, bone_attenuation_multiplier=10.5)
    drr_gene = DRR(subject, sdd=SDD, height=122, delx=DELX, reverse_x_axis=False).to(device)
    scaleInen = ScaleIntensity()
    ap_ini_drr = drr_gene(ap_init)
    la_ini_drr = drr_gene(la_init)
    # print(ap_ini_drr.shape)
    # infer_ap_drr = torch.permute(ap_ini_drr, (0, 1, 3, 2))
    # infer_la_drr = torch.permute(la_ini_drr, (0, 1, 3, 2))
    plt.subplot(2, 2, 1)
    plt.imshow(ap_gt.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(la_gt.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.subplot(2, 2, 3)
    plt.imshow(ap_ini_drr.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(la_ini_drr.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.show()
    plt.close()
    # 优化算法
    # optimize(drr_gene, [ap_gt, la_gt], case_name, scaleInen, ini_pose,
    #          [ap_Xdir, ap_Ydir, la_Wld_Offset], [la_Xdir, la_Ydir, ap_Wld_Offset], ap_extrinsic_update, la_extrinsic_update)
    # bg_img_tensor = torch.permute(ground_truth, (0, 1, 3, 2))
    # animate_in_browser(params, len(params), drr, ground_truth)
    del drr_gene


def optimize(
        reg: DRR,
        gt_imgs,
        samplename,
        scaler,
        initial_pose,
        ap_cam_param,
        la_cam_param,
        ap_wld_extrinsic_update,
        la_wld_extrinsic_update,
        n_itrs=150,
):
    T1 = time.time()
    # 损失函数，先尝试的归一化互相关loss
    GNCC_loss = gradncc
    # SSIM_loss = SSIMLoss(spatial_dims=2)
    # NCC_loss = NormalizedCrossCorrelation2d()
    # HD_loss = HausdorffDTLoss()
    # criterion = GeneralizedDiceLoss(include_background=False, to_onehot_y=True)
    min_generation = 0
    params = []
    losses = []
    # ground_truth = one_hot(ground_truth, num_classes=2)
    offset_range = 18
    rot = initial_pose[:, :3].cpu().numpy().squeeze()
    trans = initial_pose[:, 3:].cpu().numpy().squeeze()
    bound = [[rot[0] - np.pi / offset_range, rot[0] + np.pi / offset_range],
             [rot[1] - np.pi / offset_range, rot[1] + np.pi / offset_range],
             [rot[2] - np.pi / offset_range, rot[2] + np.pi / offset_range],
             [trans[0] - 65, trans[0] + 65], [trans[1] - 110, trans[1] + 110], [trans[2] - 60, trans[2] + 60]]
    bound = np.array(bound)
    # 迭代循环
    early_stop = False
    rtvec = np.concatenate([rot, trans])
    steps = np.concatenate([np.zeros(3), np.zeros(3)])
    # optimizer = CMA(mean=rtvec, sigma=2.0, bounds=bound, population_size=50, lr_adapt=True)
    kDEG2RAD = np.pi / 180
    covs = np.diag([15 * kDEG2RAD, 30 * kDEG2RAD, 15 * kDEG2RAD, 65, 110, 60])
    # cov0s = [15 * kDEG2RAD, 15 * kDEG2RAD, 30 * kDEG2RAD, 25, 25, 50]
    # optimizer = CMAwM(mean=rtvec, sigma=2.0, bounds=bound, population_size=100, steps=steps, cov=covs)
    optimizer = CMA(mean=rtvec, sigma=2.0, bounds=bound, cov=covs, population_size=50)
    for itr in tqdm(range(n_itrs), ncols=100):
        solutions = []
        op_loss = 0
        ap_gncc_loss = 0
        la_gncc_loss = 0
        for _ in range(optimizer.population_size):
            # x_eval, x_tell = optimizer.ask()
            x_eval = optimizer.ask()
            # print(x_tell)
            x_eval = torch.unsqueeze(torch.tensor(x_eval, dtype=torch.float, requires_grad=False,
                                                  device=device), 0)
            # print(x_eval)
            # ap_extrinsic_update = get_ext_pose(ap_cam_param[0], ap_cam_param[1], ap_cam_param[2], x_eval, view='ap')
            # la_extrinsic_update = get_ext_pose(la_cam_param[0], la_cam_param[1], la_cam_param[2], x_eval, view='la')
            # print(ap_extrinsic_update.matrix.shape)
            ap_extrinsic_update = update_pose(ap_wld_extrinsic_update, x_eval)
            la_extrinsic_update = update_pose(la_wld_extrinsic_update, x_eval)
            dual_pose = torch.concat((ap_extrinsic_update.matrix, la_extrinsic_update.matrix), dim=0)
            estimate = reg(dual_pose, parameterization="matrix")
            # ncc_loss = GNCC_loss(estimate.float(), ground_truth.float())
            ap_ncc_loss = GNCC_loss(estimate[0, :].unsqueeze(0), gt_imgs[0].float())
            la_ncc_loss = GNCC_loss(estimate[1, :].unsqueeze(0), gt_imgs[1].float())
            # ap_ss_loss = SSIM_loss(estimate[0, :].unsqueeze(0).float(), gt_imgs[0].float())
            # la_ss_loss = SSIM_loss(estimate[1, :].unsqueeze(0).float(), gt_imgs[1].float())
            # gncc_loss = (ap_ncc_loss + la_ncc_loss) / 2
            gncc_loss = ap_ncc_loss * 0.7 + la_ncc_loss * 0.3
            # gncc_loss = ap_ncc_loss
            total_loss = gncc_loss
            solutions.append((x_eval.detach().squeeze().cpu().numpy(), total_loss.detach().squeeze().cpu().numpy()))
            # print(line_loss.item())
            # solutions.append((x_tell, ncc_loss.detach().squeeze().cpu().numpy()))
            # print(solutions)
            # gncc_loss = GNCC_loss(estimate.float(), ground_truth.float())
            op_loss += gncc_loss.item()
            ap_gncc_loss += ap_ncc_loss.item()
            la_gncc_loss += la_ncc_loss.item()
            # plt.subplot(1, 2, 1)
            # # plt.imshow(estimate[0, :].squeeze().detach().cpu().numpy())
            # plt.imshow(torch.argmax(pred_lines[0, :], dim=0).squeeze().detach().cpu().numpy())
            # plt.subplot(1, 2, 2)
            # plt.imshow(torch.argmax(pred_lines[1, :], dim=0).squeeze().detach().cpu().numpy())
            # plt.show()
        optimizer.tell(solutions)
        result = torch.unsqueeze(torch.tensor(optimizer._mean, dtype=torch.float, requires_grad=False,
                                              device=device), 0)
        cur_loss = op_loss / optimizer.population_size
        ap_cur_loss = ap_gncc_loss / optimizer.population_size
        la_cur_loss = la_gncc_loss / optimizer.population_size
        losses.append(cur_loss)
        print(f"itr {itr + 1} ap_gncc loss: {ap_cur_loss:.4f} la_gncc loss: {la_cur_loss:.4f}")
        alpha, beta, gamma = result[:, :3, ].squeeze().tolist()
        bx, by, bz = result[:, 3:, ].squeeze().tolist()
        params.append([i for i in [alpha, beta, gamma, bx, by, bz]])
        # if optimizer.should_stop():
        #     print('early stop at generation:', itr)
        #     break
        # if min_generation + 5 < itr and early_stop:
        #     print('early stop at generation:', itr)
        #     break
        if abs(losses[itr - 1] - cur_loss) < 0.00001 and itr > 5:
            # if losses[itr-1] < cur_loss:
            tqdm.write(f"Converged in {itr} iterations")
            T2 = time.time()
            print('配准耗时为:%.4s秒' % (T2 - T1))
            break
    # out_estimate = torch.permute(estimate[0, :], (0, 2, 1))
    out_img = sitk.GetImageFromArray(estimate[0, :].squeeze().detach().cpu().numpy())
    out_arr = sitk.GetArrayFromImage(out_img)
    out_arr = np.where(out_arr != 0, 1, 0)
    out_img = sitk.GetImageFromArray(out_arr)
    sitk.WriteImage(out_img, 'Data/tuodao/{}/X/{}_spine_proj_seg.nii.gz'.format(caseName, samplename))
    # 记录了每次迭代生成的位姿，优化结束后，可以运行results_metric.py中的函数可视化配准过程
    df = pd.DataFrame(params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
    df["loss"] = losses
    print(df)
    df.to_csv('results/tuodao/{}_spine_dual_pose.csv'.format(samplename), index=False)


def xray_process(xray_path, spacing=3.0360001325607300e-01, save_dir=None):
    xray = sitk.ReadImage(xray_path)
    if len(xray.GetSize()) > 2:
        xray = xray[:, :, 0]
    xray.SetSpacing((spacing, spacing))
    # x_arr = sitk.GetArrayFromImage(xray)
    # if len(x_arr.shape) > 2:
    #     img_arr = np.max(x_arr) - x_arr[0, :, :]
    # else:
    #     img_arr = np.max(x_arr) - x_arr
    # inverse_xray = sitk.GetImageFromArray(img_arr)
    # inverse_xray.CopyInformation(xray)
    xray_arr = sitk.GetArrayFromImage(xray)
    processed_arr = HE_optimize(xray_arr)
    processed_img = sitk.GetImageFromArray(processed_arr)
    processed_img.SetSpacing(xray.GetSpacing())
    resized_x = resample_img(processed_img, new_width=122, save_path=save_dir)
    x_arr = sitk.GetArrayFromImage(resized_x)
    ground_truth = torch.tensor(x_arr)
    ground_truth = torch.unsqueeze(ground_truth, dim=0).to('cuda')
    ground_truth = torch.unsqueeze(ground_truth, dim=0).to('cuda')
    # ground_truth = torch.permute(ground_truth, (0, 1, 3, 2))
    return ground_truth


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    caseName = 'dukemei'
    vert_num = 'L3'
    ct_path = 'Data/tuodao/{}/{}.nii.gz'.format(caseName, caseName)
    vert_seg_path = 'Data/tuodao/{}/{}_seg.nii.gz'.format(caseName, vert_num)
    ap_gt_path = 'Data/tuodao/{}/X/{}_ap.nii.gz'.format(caseName, caseName)
    la_gt_path = 'Data/tuodao/{}/X/{}_la.nii.gz'.format(caseName, caseName)
    ap_xml_path = 'Data/tuodao/{}/X/View/180/calib_view.xml'.format(caseName)
    la_xml_path = 'Data/tuodao/{}/X/View/1/calib_view.xml'.format(caseName)
    resized_x_ap_save_path = 'Data/tuodao/{}/X/{}_122_x_ap.nii.gz'.format(caseName, caseName)
    resized_x_la_save_path = 'Data/tuodao/{}/X/{}_122_x_la.nii.gz'.format(caseName, caseName)
    reg_method(ct_path, vert_seg_path, [ap_gt_path, la_gt_path], '{}_{}'.format(caseName, vert_num),
               [resized_x_ap_save_path, resized_x_la_save_path], [ap_xml_path, la_xml_path])
