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
from line_infer import infer_method
from tools import get_initial_parameters
from monai.networks.utils import one_hot
from losses.MaskedGNCCLoss import MaskGradientNormalizedCrossCorrelation2d, MaskNormalizedCrossCorrelation2d
from tools import get_drr, get_lineCenter_offset, crop_ct_vert, gaussian_preprocess, resample_img, \
    bbx_crop_gt_vert, HE_optimize, read_xml, get_ext_pose
from sklearn.metrics import mean_absolute_error
import math
import os
from losses.ZNCC import ncc, gradncc
from cmaes import CMA, CMAwM
from diffdrr.pose import convert, RigidTransform


def reg_method(origin_ct_path, seg_path, xray_paths, boundingbox_paths, case_name, save_dir, x_saves,
               xml_paths, line_paths=None):
    # 读取椎骨CT
    offsetx, offsety, offsetz, vert_img, vert_path = crop_ct_vert(origin_ct_path, seg_path, save_dir)
    offset_trans = np.array([offsetx, offsety, offsetz])
    ap_Xdir, ap_Ydir, ap_spacing, SDD, Xray_H, ap_Wld_Offset = read_xml(xml_paths[0])
    la_Xdir, la_Ydir, la_spacing, _, _, la_Wld_Offset = read_xml(xml_paths[1])
    # offset_trans[0] = float(ap_Wld_Offset[1]) / SDD * offset_trans[0]
    # offset_trans[1] = float(ap_Wld_Offset[1]) / SDD * offset_trans[1]
    print(offset_trans)
    ap_gt, cropped_spacing = preliminary_process(xray_paths[0], ap_spacing, boundingbox_paths[0], x_saves[0])
    la_gt, _ = preliminary_process(xray_paths[1], la_spacing, boundingbox_paths[1], x_saves[1])
    # gt_mask = torch.tensor(x_mask_arr)
    # gt_mask = torch.unsqueeze(gt_mask, dim=0).to(device)
    # gt_mask = torch.unsqueeze(gt_mask, dim=0).to(device)
    # gt_mask = torch.permute(gt_mask, (0, 1, 3, 2))
    ap_line = line_preprocess(line_paths[0])
    ini_pose = torch.zeros(1, 6).to(device)
    # print(ini_pose.shape)
    ap_extrinsic_update = get_ext_pose(ap_Xdir, ap_Ydir, float(ap_Wld_Offset[1]), ini_pose, offset_trans)
    la_extrinsic_update = get_ext_pose(la_Xdir, la_Ydir, float(la_Wld_Offset[0]), ini_pose, offset_trans)
    print(SDD)
    print(cropped_spacing)
    drr_gene = get_drr(img_path=vert_path,
                       SDD=SDD,
                       DELX=cropped_spacing)
    scaleInen = ScaleIntensity()
    ap_ini_drr = drr_gene(ap_extrinsic_update)
    la_ini_drr = drr_gene(la_extrinsic_update)
    # print(ap_ini_drr.shape)
    ini_drr = torch.permute(ap_ini_drr, (0, 1, 3, 2))
    ap_gt = torch.permute(ap_gt, (0, 1, 3, 2))
    out_img = sitk.GetImageFromArray(ap_ini_drr.squeeze().detach().cpu().numpy())
    sitk.WriteImage(out_img, 'Data/tuodao/{}/L3_drr.nii.gz'.format(caseName))
    la_gt = torch.permute(la_gt, (0, 1, 3, 2))
    # print(bbx_gt.shape)
    mov_line = infer_method(ini_drr)
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
    # coarse_reg = Registration(
    #     drr_gene,
    #     gt_rotations.clone(),
    #     gt_translations.clone(),
    #     parameterization="euler_angles",
    #     convention="ZYX",
    #     dual_view=True
    # )
    # coarse_drr = coarse_reg()
    # coarse_drr = torch.permute(coarse_drr, (0, 1, 3, 2))
    plt.subplot(2, 2, 1)
    plt.imshow(ap_gt.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(la_gt.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.subplot(2, 2, 3)
    plt.imshow(ap_line.squeeze().detach().cpu().numpy())
    plt.subplot(2, 2, 4)
    plt.imshow(pred_mask.squeeze().detach().cpu().numpy())
    plt.show()
    plt.close()
    # 优化算法
    optimize(drr_gene, [ap_gt, la_gt], ap_line, case_name, scaleInen, ini_pose,
             [ap_Xdir, ap_Ydir, float(ap_Wld_Offset[1])], [la_Xdir, la_Ydir, float(la_Wld_Offset[0])])
    # bg_img_tensor = torch.permute(ground_truth, (0, 1, 3, 2))
    # animate_in_browser(params, len(params), drr, ground_truth)
    del drr_gene


def optimize(
        reg: DRR,
        gt_imgs,
        gt_lines,
        samplename,
        scaler,
        initial_pose,
        ap_cam_param,
        la_cam_param,
        n_itrs=100,
):
    T1 = time.time()
    # 损失函数，先尝试的归一化互相关loss
    GNCC_loss = gradncc
    # SSIM_loss = SSIMLoss(spatial_dims=2)
    # NCC_loss = NormalizedCrossCorrelation2d()
    # HD_loss = HausdorffDTLoss()
    DCE_loss = DiceCELoss(to_onehot_y=True, softmax=True)
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
             [trans[0] - 100, trans[0] + 100], [trans[1] - 100, trans[1] + 100], [trans[2] - 100, trans[2] + 100]]
    bound = np.array(bound)
    # 迭代循环
    early_stop = False
    rtvec = np.concatenate([rot, trans])
    steps = np.concatenate([np.zeros(3), np.zeros(3)])
    # optimizer = CMA(mean=rtvec, sigma=2.0, bounds=bound, population_size=50, lr_adapt=True)
    kDEG2RAD = np.pi / 180
    covs = np.diag([15 * kDEG2RAD, 30 * kDEG2RAD, 15 * kDEG2RAD, 50, 100, 50])
    # cov0s = [15 * kDEG2RAD, 15 * kDEG2RAD, 30 * kDEG2RAD, 25, 25, 50]
    # optimizer = CMAwM(mean=rtvec, sigma=2.0, bounds=bound, population_size=100, steps=steps, cov=covs)
    optimizer = CMA(mean=rtvec, sigma=2.0, bounds=bound, cov=covs, population_size=50)
    for itr in tqdm(range(n_itrs), ncols=100):
        solutions = []
        op_loss = 0
        dce_loss = 0
        for _ in range(optimizer.population_size):
            # x_eval, x_tell = optimizer.ask()
            x_eval = optimizer.ask()
            # print(x_tell)
            x_eval = torch.unsqueeze(torch.tensor(x_eval, dtype=torch.float, requires_grad=False,
                                                  device=device), 0)
            # print(x_eval)
            ap_extrinsic_update = get_ext_pose(ap_cam_param[0], ap_cam_param[1], ap_cam_param[2], x_eval)
            # la_extrinsic_update = get_ext_pose(la_cam_param[0], la_cam_param[1], la_cam_param[2], x_eval)
            # print(ap_extrinsic_update.matrix.shape)
            # dual_pose = torch.concat((ap_extrinsic_update.matrix, la_extrinsic_update.matrix), dim=0)
            # estimate = reg(dual_pose, parameterization="matrix")
            ap_estimate = reg(ap_extrinsic_update)
            # la_estimate = reg(la_extrinsic_update)
            # ncc_loss = GNCC_loss(estimate.float(), ground_truth.float())
            ap_input = torch.permute(ap_estimate, (0, 1, 3, 2))
            ap_pred_line = infer_method(ap_input)
            # print(gt_lines.shape)
            # print(ap_pred_line.shape)
            line_loss = DCE_loss(ap_pred_line, gt_lines)
            ap_ncc_loss = GNCC_loss(ap_estimate, gt_imgs[0].float())
            # la_ncc_loss = GNCC_loss(estimate[1, :].unsqueeze(0).float(), gt_imgs[1].float())
            # ap_ss_loss = SSIM_loss(estimate[0, :].unsqueeze(0).float(), gt_imgs[0].float())
            # la_ss_loss = SSIM_loss(estimate[1, :].unsqueeze(0).float(), gt_imgs[1].float())
            # ncc_loss = (ap_ncc_loss + la_ncc_loss) / 2 * 0.9 + 0.1 * line_loss
            # ncc_loss = (ap_ncc_loss + la_ncc_loss) / 2
            ncc_loss = ap_ncc_loss * 0.9 + 0.1 * line_loss
            # ncc_loss = ap_ncc_loss
            solutions.append((x_eval.detach().squeeze().cpu().numpy(), ncc_loss.detach().squeeze().cpu().numpy()))
            # print(line_loss.item())
            # solutions.append((x_tell, ncc_loss.detach().squeeze().cpu().numpy()))
            # print(solutions)
            # gncc_loss = GNCC_loss(estimate.float(), ground_truth.float())
            loss = ncc_loss
            dce_loss += line_loss.item()
            op_loss += loss.item()
            # plt.subplot(1, 2, 1)
            # plt.imshow(estimate[0, :].squeeze().detach().cpu().numpy())
            # plt.subplot(1, 2, 2)
            # plt.imshow(estimate[1, :].squeeze().detach().cpu().numpy())
            # plt.show()
        optimizer.tell(solutions)
        result = torch.unsqueeze(torch.tensor(optimizer._mean, dtype=torch.float, requires_grad=False,
                                              device=device), 0)
        cur_loss = op_loss / optimizer.population_size
        cur_line_loss = dce_loss / optimizer.population_size
        losses.append(cur_loss)
        print(f"itr {itr + 1} average loss: {cur_loss:.4f} dce loss: {cur_line_loss:.4f}")
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
    out_img = sitk.GetImageFromArray(ap_estimate.squeeze().detach().cpu().numpy())
    out_img = resample_img(out_img, new_width=976)
    out_arr = sitk.GetArrayFromImage(out_img)
    out_arr = np.where(out_arr != 0, 1, 0)
    out_img = sitk.GetImageFromArray(out_arr)
    sitk.WriteImage(out_img, 'Data/tuodao/{}/X/{}_proj_seg.nii.gz'.format(caseName, samplename))
    # 记录了每次迭代生成的位姿，优化结束后，可以运行results_metric.py中的函数可视化配准过程
    df = pd.DataFrame(params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
    df["loss"] = losses
    print(df)
    df.to_csv('results/tuodao/{}_dual_pose.csv'.format(samplename), index=False)
    # return df


def preliminary_process(xray_path, spacing=3.0360001325607300e-01, boundingbox_path=None, x_save=None):
    xray = sitk.ReadImage(xray_path)
    # print(xray.GetSize())
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
    # xray_arr = xray_arr.astype(float)
    # img_tensor = torch.tensor(xray_arr)
    # img_tensor = gaussian_preprocess(img_tensor.unsqueeze(0), cropped=True)
    processed_img = sitk.GetImageFromArray(processed_arr)
    processed_img.SetSpacing(xray.GetSpacing())
    xray_mask = sitk.ReadImage(boundingbox_path)
    xray_mask.SetSpacing((spacing, spacing, 1))
    resized_x = resample_img(processed_img, new_width=256, save_path=x_save)
    resized_mask = resample_img(xray_mask, new_width=256)
    cropped_x = bbx_crop_gt_vert(resized_x, resized_mask, inverse=False)
    # print(resized_x.GetSpacing()[0])
    x_arr = sitk.GetArrayFromImage(cropped_x)
    ground_truth = torch.tensor(x_arr)
    ground_truth = torch.unsqueeze(ground_truth, dim=0).to('cuda')
    ground_truth = torch.unsqueeze(ground_truth, dim=0).to('cuda')
    ground_truth = torch.permute(ground_truth, (0, 1, 3, 2))
    return ground_truth, resized_x.GetSpacing()[0]


def line_preprocess(line_path):
    line_img = sitk.ReadImage(line_path)
    ap_line = resample_img(line_img, new_width=256, interpolator_method=sitk.sitkNearestNeighbor)
    line_arr = sitk.GetArrayFromImage(ap_line)
    line_arr = line_arr.astype(float)
    line_tensor = torch.tensor(line_arr)
    line_tensor = torch.unsqueeze(line_tensor, dim=0).to('cuda')
    line_tensor = torch.unsqueeze(line_tensor, dim=0).to('cuda')
    line_tensor = torch.permute(line_tensor, (0, 1, 3, 2))
    return line_tensor


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    caseName = 'dukemei'
    vert_num = 'L3'
    ct_path = 'Data/tuodao/{}/{}.nii.gz'.format(caseName, caseName)
    vert_seg_path = 'Data/tuodao/{}/{}_seg.nii.gz'.format(caseName, vert_num)
    ap_gt_path = 'Data/tuodao/{}/X/{}_ap.nii.gz'.format(caseName, caseName)
    ap_bbx_path = 'Data/tuodao/{}/X/{}_bbx_ap.nii.gz'.format(caseName, vert_num)
    la_gt_path = 'Data/tuodao/{}/X/{}_la.nii.gz'.format(caseName, caseName)
    la_bbx_path = 'Data/tuodao/{}/X/{}_bbx_la.nii.gz'.format(caseName, vert_num)
    vert_save_path = 'Data/tuodao/{}/{}_{}.nii.gz'.format(caseName, caseName, vert_num)
    resized_x_ap_save_path = 'Data/tuodao/{}/X/{}_resized_x_ap.nii.gz'.format(caseName, caseName)
    resized_x_la_save_path = 'Data/tuodao/{}/X/{}_resized_x_la.nii.gz'.format(caseName, caseName)
    line_ap_path = 'Data/tuodao/{}/X/{}_line_ap.nii.gz'.format(caseName, vert_num)
    line_la_path = 'Data/tuodao/{}/X/{}_line_la.nii.gz'.format(caseName, vert_num)
    ap_xml_path = 'Data/tuodao/{}/X/View/180/calib_view.xml'.format(caseName)
    la_xml_path = 'Data/tuodao/{}/X/View/1/calib_view.xml'.format(caseName)
    reg_method(ct_path, vert_seg_path, [ap_gt_path, la_gt_path], [ap_bbx_path, la_bbx_path],
               '{}_{}'.format(caseName, vert_num), vert_save_path, [resized_x_ap_save_path, resized_x_la_save_path],
               [ap_xml_path, la_xml_path], [line_ap_path, line_la_path])
