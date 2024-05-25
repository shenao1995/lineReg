import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import SimpleITK as sitk
from diffdrr.drr import DRR
from monai.losses import DiceLoss, DiceCELoss, GeneralizedDiceLoss, SSIMLoss, HausdorffDTLoss
import time
from line_infer import infer_method
from tools import get_initial_parameters
from monai.networks.utils import one_hot
from losses.MaskedGNCCLoss import MaskGradientNormalizedCrossCorrelation2d, MaskNormalizedCrossCorrelation2d
from tools import get_drr, get_lineCenter_offset, crop_ct_vert, gaussian_preprocess, resample_img, \
    bbx_crop_gt_vert, HE_optimize, read_xml, get_ext_pose, update_pose, masks_to_boxes
from sklearn.metrics import mean_absolute_error
from losses.ZNCC import ncc, gradncc
from cmaes import CMA
from diffdrr.data import read
from monai.networks.nets import AttentionUnet


def reg_method(origin_ct_path, seg_path, xray_paths, boundingbox_paths, case_name, save_dir, x_saves,
               xml_paths, line_paths=None):
    # 读取椎骨CT
    offsetx, offsety, offsetz, vert_img, vert_path = crop_ct_vert(origin_ct_path, seg_path, save_dir)
    offset_trans = np.array([offsetx, offsety, offsetz])
    ap_Xdir, ap_Ydir, ap_spacing, SDD, Xray_H, ap_Wld_Offset = read_xml(xml_paths[0])
    la_Xdir, la_Ydir, la_spacing, _, _, la_Wld_Offset = read_xml(xml_paths[1])
    ap_refine_gt, ap_mask = preliminary_process(xray_paths[0], 256, ap_spacing, boundingbox_paths[0], x_saves[0])
    la_refine_gt, la_mask = preliminary_process(xray_paths[1], 256, la_spacing, boundingbox_paths[1], x_saves[1])
    ap_coarse_gt, _ = preliminary_process(xray_paths[0], 122, ap_spacing, boundingbox_paths[0], x_saves[0])
    la_coarse_gt, _ = preliminary_process(xray_paths[1], 122, la_spacing, boundingbox_paths[1], x_saves[1])
    ap_line = line_preprocess(line_paths[0])
    la_line = line_preprocess(line_paths[1])
    ini_pose = torch.zeros(1, 6).to(device)
    ap_extrinsic_update = get_ext_pose(ap_Xdir, ap_Ydir, ap_Wld_Offset, ini_pose, view='ap')
    la_extrinsic_update = get_ext_pose(la_Xdir, la_Ydir, la_Wld_Offset, ini_pose, view='la')
    ap_init = update_pose(ap_extrinsic_update, ini_pose)
    la_init = update_pose(la_extrinsic_update, ini_pose)
    print(SDD)
    COARSE_DELX = Xray_H / 122 * ap_spacing
    REFINE_DELX = Xray_H / 256 * ap_spacing
    print(COARSE_DELX)
    # drr_gene = get_drr(img_path=vert_path,
    #                    SDD=SDD,
    #                    DELX=cropped_spacing)
    ct_subject = read(origin_ct_path, bone_attenuation_multiplier=2.5)
    coarse_drr = DRR(ct_subject, sdd=SDD, height=122, delx=COARSE_DELX, reverse_x_axis=False).to(device)
    vert_subject = read(vert_path, bone_attenuation_multiplier=2.5)
    refine_drr = DRR(vert_subject, sdd=SDD, height=256, delx=REFINE_DELX, reverse_x_axis=False).to(device)
    ap_ini_drr = coarse_drr(ap_init)
    la_ini_drr = coarse_drr(la_init)
    plt.subplot(2, 2, 1)
    plt.imshow(ap_ini_drr.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(la_ini_drr.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.subplot(2, 2, 3)
    plt.imshow(ap_coarse_gt.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(la_coarse_gt.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.show()
    plt.close()
    # 优化算法
    T1 = time.time()
    coare_pose = coarse_reg_opt(coarse_drr, [ap_coarse_gt, la_coarse_gt], case_name, ini_pose,
                                ap_extrinsic_update, la_extrinsic_update)
    coare_pose = torch.from_numpy(coare_pose).unsqueeze(0).to(device)
    coare_pose[:, 3], coare_pose[:, 4], coare_pose[:, 5] = coare_pose[:, 3] - offset_trans[0], \
                                                           coare_pose[:, 4] - offset_trans[1], \
                                                           coare_pose[:, 5] + offset_trans[2]
    print(coare_pose)
    refine_reg_optimize(refine_drr, [ap_refine_gt, la_refine_gt], [ap_line, la_line], [ap_mask, la_mask], case_name,
                        coare_pose, ap_extrinsic_update, la_extrinsic_update)
    T2 = time.time()
    print('配准耗时为:%.4s秒' % (T2 - T1))
    del coarse_drr


def coarse_reg_opt(
        reg: DRR,
        gt_imgs,
        samplename,
        initial_pose,
        ap_wld_extrinsic_update,
        la_wld_extrinsic_update,
        n_itrs=150):
    T1 = time.time()
    # 损失函数，先尝试的归一化互相关loss
    GNCC_loss = gradncc
    params = []
    losses = []
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
            x_eval = optimizer.ask()
            x_eval = torch.unsqueeze(torch.tensor(x_eval, dtype=torch.float, requires_grad=False,
                                                  device=device), 0)
            ap_extrinsic_update = update_pose(ap_wld_extrinsic_update, x_eval)
            la_extrinsic_update = update_pose(la_wld_extrinsic_update, x_eval)
            dual_pose = torch.concat((ap_extrinsic_update.matrix, la_extrinsic_update.matrix), dim=0)
            estimate = reg(dual_pose, parameterization="matrix")
            # ncc_loss = GNCC_loss(estimate.float(), ground_truth.float())
            ap_ncc_loss = GNCC_loss(estimate[0, :].unsqueeze(0), gt_imgs[0].float())
            la_ncc_loss = GNCC_loss(estimate[1, :].unsqueeze(0), gt_imgs[1].float())
            # gncc_loss = (ap_ncc_loss + la_ncc_loss) / 2
            gncc_loss = ap_ncc_loss * 0.7 + la_ncc_loss * 0.3
            # gncc_loss = ap_ncc_loss
            total_loss = gncc_loss
            solutions.append((x_eval.detach().squeeze().cpu().numpy(), total_loss.detach().squeeze().cpu().numpy()))
            op_loss += gncc_loss.item()
            ap_gncc_loss += ap_ncc_loss.item()
            la_gncc_loss += la_ncc_loss.item()
        optimizer.tell(solutions)
        pose_op = optimizer._mean
        cur_loss = op_loss / optimizer.population_size
        ap_cur_loss = ap_gncc_loss / optimizer.population_size
        la_cur_loss = la_gncc_loss / optimizer.population_size
        losses.append(cur_loss)
        print(f"itr {itr + 1} ap_gncc loss: {ap_cur_loss:.4f} la_gncc loss: {la_cur_loss:.4f}")
        alpha, beta, gamma, bx, by, bz = pose_op.tolist()
        params.append([i for i in [alpha, beta, gamma, bx, by, bz]])
        if abs(losses[itr - 1] - cur_loss) < 0.00001 and itr > 5:
            # if losses[itr-1] < cur_loss:
            tqdm.write(f"Converged in {itr} iterations")
            T2 = time.time()
            print('配准耗时为:%.4s秒' % (T2 - T1))
            break
    result = optimizer._mean
    return result


def refine_reg_optimize(
        reg: DRR,
        gt_imgs,
        gt_lines,
        gt_masks,
        samplename,
        initial_pose,
        ap_wld_extrinsic_update,
        la_wld_extrinsic_update,
        n_itrs=150,
):
    T1 = time.time()
    # 损失函数，先尝试的归一化互相关loss
    GNCC_loss = gradncc
    # SSIM_loss = SSIMLoss(spatial_dims=2)
    # NCC_loss = NormalizedCrossCorrelation2d()
    # DCE_loss = HausdorffDTLoss(to_onehot_y=True, softmax=True)
    DCE_loss = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
    MSE_loss = torch.nn.MSELoss()
    params = []
    gncc_losses = []
    line_losses = []
    # ground_truth = one_hot(ground_truth, num_classes=2)
    offset_range = 36
    rot = initial_pose[:, :3].cpu().numpy().squeeze()
    trans = initial_pose[:, 3:].cpu().numpy().squeeze()
    rtvec = initial_pose.cpu().numpy().squeeze()
    # rtvec = np.array([-0.011898027, -0.053354662, -0.168720275, -61.61990738, -99.07395172, -34.39813614])
    bound = [[rot[0] - np.pi / offset_range, rot[0] + np.pi / offset_range],
             [rot[1] - np.pi / offset_range, rot[1] + np.pi / offset_range],
             [rot[2] - np.pi / offset_range, rot[2] + np.pi / offset_range],
             [trans[0] - 15, trans[0] + 15], [trans[1] - 30, trans[1] + 30], [trans[2] - 15, trans[2] + 15]]
    bound = np.array(bound)
    # 迭代循环
    kDEG2RAD = np.pi / 180
    covs = np.diag([0.5 * kDEG2RAD, 2 * kDEG2RAD, 0.5 * kDEG2RAD, 15, 30, 15])
    # cov0s = [15 * kDEG2RAD, 15 * kDEG2RAD, 30 * kDEG2RAD, 25, 25, 50]
    optimizer = CMA(mean=rtvec, sigma=2.0, bounds=bound, cov=covs, population_size=50)
    log_dir = 'line_model/AttUNet_model1.pth'
    net = AttentionUnet(spatial_dims=2,
                        in_channels=1,
                        out_channels=2,
                        channels=(16, 32, 64, 128, 256),
                        strides=(2, 2, 2, 2)).to(device)
    net.load_state_dict(torch.load(log_dir))
    for itr in tqdm(range(n_itrs), ncols=100):
        solutions = []
        op_loss = 0
        dce_loss = 0
        center_loss = 0
        ap_gncc_sum_loss = 0
        la_gncc_sum_loss = 0
        for _ in range(optimizer.population_size):
            # x_eval, x_tell = optimizer.ask()
            x_eval = optimizer.ask()
            # print(x_tell)
            x_eval = torch.unsqueeze(torch.tensor(x_eval, dtype=torch.float, requires_grad=False,
                                                  device=device), 0)
            ap_extrinsic_update = update_pose(ap_wld_extrinsic_update, x_eval)
            la_extrinsic_update = update_pose(la_wld_extrinsic_update, x_eval)
            dual_pose = torch.concat((ap_extrinsic_update.matrix, la_extrinsic_update.matrix), dim=0)
            estimate = reg(dual_pose, parameterization="matrix")
            ap_input = torch.permute(estimate[0, :].unsqueeze(0), (0, 1, 3, 2))
            la_input = torch.permute(estimate[1, :].unsqueeze(0), (0, 1, 3, 2))
            pred_lines = infer_method(net, input_list=[torch.squeeze(ap_input), torch.squeeze(la_input)])
            ap_line_loss = DCE_loss(pred_lines[0, :].unsqueeze(0), gt_lines[0])
            la_line_loss = DCE_loss(pred_lines[1, :].unsqueeze(0), gt_lines[1])
            line_loss = (ap_line_loss + la_line_loss) / 2
            # line_loss = ap_line_loss * 0.7 + la_line_loss * 0.3
            ap_ncc_loss = GNCC_loss(estimate[0, :].unsqueeze(0), gt_imgs[0].float(), mask_used=True, mask=gt_masks[0])
            la_ncc_loss = GNCC_loss(estimate[1, :].unsqueeze(0), gt_imgs[1].float(), mask_used=True, mask=gt_masks[1])
            # gncc_loss = ap_ncc_loss * 0.3 + la_ncc_loss * 0.7
            gncc_loss = (ap_ncc_loss + la_ncc_loss) / 2
            # gncc_loss = la_ncc_loss
            total_loss = gncc_loss * 0.8 + line_loss * 0.2
            # total_loss = gncc_loss
            solutions.append((x_eval.detach().squeeze().cpu().numpy(), total_loss.detach().squeeze().cpu().numpy()))
            dce_loss += line_loss.item()
            op_loss += gncc_loss.item()
            ap_gncc_sum_loss += ap_ncc_loss.item()
            la_gncc_sum_loss += la_ncc_loss.item()
        optimizer.tell(solutions)
        result = torch.unsqueeze(torch.tensor(optimizer._mean, dtype=torch.float, requires_grad=False,
                                              device=device), 0)
        cur_loss = op_loss / optimizer.population_size
        cur_line_loss = dce_loss / optimizer.population_size
        ap_cur_loss = ap_gncc_sum_loss / optimizer.population_size
        la_cur_loss = la_gncc_sum_loss / optimizer.population_size
        gncc_losses.append(cur_loss)
        line_losses.append(cur_line_loss)
        print(f"itr {itr + 1} ap_gncc loss: {ap_cur_loss:.4f} la_gncc loss: {la_cur_loss:.4f} "
              f"dce loss: {cur_line_loss:.4f}")
        alpha, beta, gamma = result[:, :3, ].squeeze().tolist()
        bx, by, bz = result[:, 3:, ].squeeze().tolist()
        params.append([i for i in [alpha, beta, gamma, bx, by, bz]])
        if abs(gncc_losses[itr - 1] - cur_loss) < 0.00001 and itr > 5:
            # if losses[itr-1] < cur_loss:
            tqdm.write(f"Converged in {itr} iterations")
            T2 = time.time()
            print('配准耗时为:%.4s秒' % (T2 - T1))
            break
    # out_estimate = torch.permute(estimate[0, :], (0, 2, 1))
    # out_img = sitk.GetImageFromArray(estimate[0, :].squeeze().detach().cpu().numpy())
    # out_img = resample_img(out_img, new_width=976)
    # out_arr = sitk.GetArrayFromImage(out_img)
    # out_arr = np.where(out_arr != 0, 1, 0)
    # out_img = sitk.GetImageFromArray(out_arr)
    # sitk.WriteImage(out_img, 'Data/tuodao/{}/X/{}_proj_seg.nii.gz'.format(caseName, samplename))
    # 记录了每次迭代生成的位姿，优化结束后，可以运行results_metric.py中的函数可视化配准过程
    df = pd.DataFrame(params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
    df["gncc_loss"] = gncc_losses
    df["line_loss"] = line_losses
    print(df)
    df.to_csv('results/tuodao/{}_dual_pose.csv'.format(samplename), index=False)


def preliminary_process(xray_path, resize=256, spacing=3.0360001325607300e-01, boundingbox_path=None, x_save=None):
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
    # xray_arr = xray_arr.astype(float)
    # img_tensor = torch.tensor(xray_arr)
    # img_tensor = gaussian_preprocess(img_tensor.unsqueeze(0), cropped=True)
    processed_img = sitk.GetImageFromArray(processed_arr)
    processed_img.SetSpacing(xray.GetSpacing())
    xray_mask = sitk.ReadImage(boundingbox_path)
    xray_mask.SetSpacing((spacing, spacing, 1))
    resized_x = resample_img(processed_img, new_width=resize, save_path=x_save)
    resized_mask = resample_img(xray_mask, new_width=resize)
    mask_arr = sitk.GetArrayFromImage(resized_mask)
    gt_bbx = torch.from_numpy(mask_arr / 1.0)
    gt_bbx = torch.unsqueeze(gt_bbx, dim=0).to('cuda')
    # cropped_x = bbx_crop_gt_vert(resized_x, resized_mask, inverse=False)
    x_arr = sitk.GetArrayFromImage(resized_x)
    ground_truth = torch.tensor(x_arr)
    ground_truth = torch.unsqueeze(ground_truth, dim=0).to('cuda')
    # ground_truth = torch.permute(ground_truth, (0, 1, 3, 2))
    return ground_truth.unsqueeze(0), gt_bbx.unsqueeze(0)


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
