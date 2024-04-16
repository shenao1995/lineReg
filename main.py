import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import SimpleITK as sitk
from diffdrr.drr import DRR, Registration
from diffdrr.metrics import NormalizedCrossCorrelation2d, GradientNormalizedCrossCorrelation2d, MultiscaleNormalizedCrossCorrelation2d
from diffdrr.visualization import plot_drr
from diffdrr.visualization import animate
from base64 import b64encode
from monai.transforms import LoadImage, ScaleIntensity, Resize
from diffdrr.detector import diffdrr_to_deepdrr
import cv2
from CNCC import CannyCrossCorrelation2d
import time
import torch.nn as nn


def reg_method():
    T1 = time.time()
    reader = LoadImage(ensure_channel_first=True, image_only=False)
    np.random.seed(88)
    # Make the ground truth X-ray
    SDR = 500.0
    HEIGHT = 256
    DELX = 2.0
    ctDir = 'Data/gncc_data/tuodao/peizongping/peizongping_L3.nii.gz'
    # ctDir = 'Data/gncc_data/CT_L4.nii.gz'
    used_ct_arr = reader(ctDir)
    spacing = used_ct_arr[1]['pixdim']
    spacing = np.array((spacing[1], spacing[2], spacing[3]), dtype=np.float64)
    # print(spacing)
    print(used_ct_arr[0][0].shape)
    # normalized_arr = (used_ct_arr[0][0] - used_ct_arr[0][0].min()) / (used_ct_arr[0][0].max() - used_ct_arr[0][0].min())
    bx, by, bz = torch.tensor(used_ct_arr[0][0].shape) * torch.tensor(spacing) / 2
    true_params = {
        "sdr": SDR,
        "alpha": torch.pi / 2,  # 沿y轴旋转,逆时针旋转
        "beta": 0,
        "gamma": torch.pi,  # 沿x轴旋转
        "bx": bx,
        "by": by,  # 沿z轴平移
        "bz": bz,  # 沿y轴平移
    }
    # print(used_ct_arr[0].clone().detach().shape)
    drr = DRR(used_ct_arr[0][0], spacing, sdr=SDR, height=HEIGHT, delx=DELX).to(device)
    rotations = torch.tensor(
        [
            [
                true_params["alpha"],
                true_params["beta"],
                true_params["gamma"],
            ]
        ]
    ).to(device)
    translations = torch.tensor(
        [
            [
                true_params["bx"],
                true_params["by"],
                true_params["bz"],
            ]
        ]
    ).to(device)
    print(rotations)
    print(translations)
    ground_truth = drr(
        rotation=rotations,
        translation=translations,
        parameterization="euler_angles",
        convention="ZYX",
    )
    x_ray_path = 'Data/gncc_data/tuodao/peizongping/peizongping_180_x.nii.gz'
    mask_path = 'Data/gncc_data/tuodao/peizongping/peizongping_180_x_L3_seg.nii.gz'
    # bg_path = 'Data/gncc_data/91B_resized.nii.gz'
    x_img = reader(x_ray_path)
    mask = reader(mask_path)
    scaleInen = ScaleIntensity()
    out_img = scaleInen(x_img[0][:, :, :, 0])
    # bg_img = reader(bg_path)
    # print(x_img[0].shape)
    # print(x_img[1]['pixdim'])
    # x_img = sitk.ReadImage(x_ray_path)
    # x_img_arr = sitk.GetArrayFromImage(x_img)
    # x_img_tensor = torch.from_numpy(x_img_arr)
    print(out_img.shape)
    print(mask[0].shape)
    # x_img_tensor = torch.unsqueeze(out_img, dim=0).to(device)
    resize = Resize(spatial_size=(256, 256), mode='bilinear', align_corners=True)
    x_img_tensor = resize(out_img)
    resized_mask = resize(mask[0][:, :, :, 0])
    tensor_mask = torch.unsqueeze(resized_mask, dim=0).to(device)
    x_img_tensor = torch.unsqueeze(x_img_tensor, dim=0).to(device)
    # bg_img_tensor = torch.unsqueeze(bg_img[0], dim=0).to(device)
    # print(ground_truth.shape)
    # rot_tensor = torch.zeros(1, 3, dtype=torch.float, device=device)
    # trans_tensor = torch.zeros(1, 3, dtype=torch.float, device=device)
    # gt_drrs = get_drr(trans_tensor, rot_tensor)
    # ground_truth = scaleInen(ground_truth)
    # canny = CannyFilter()
    # thresh = canny(ground_truth)
    # thresh = CannyFilter(ground_truth)
    # img_normalized = cv2.normalize(ground_truth.cpu().numpy(), None, 0, 255.0,
    #                                cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # thresh = cv2.Canny(np.uint8(img_normalized.squeeze()), 1, 40)
    # thresh = torch.from_numpy(thresh).to(device)
    # thresh = torch.unsqueeze(thresh, dim=0)
    # thresh = torch.unsqueeze(thresh, dim=0)
    # print(thresh.shape)
    # ground_truth = torch.permute(ground_truth, (0, 1, 3, 2))
    # plot_drr(ground_truth)
    # plt.show()
    # return
    # criterion = NormalizedCrossCorrelation2d()
    # criterion(ground_truth, est).item()
    # drr = DRR(used_ct_arr[0][0], spacing, sdr=SDR, height=HEIGHT, delx=DELX).to(device)
    # # # print(rotations)
    # # # print(translations)
    # # # tensor([[1.3901, -0.0063, 3.1101]], device='cuda:0')
    # # # tensor([[66.6104, 139.4143, 90.0197]], device='cuda:0', dtype=torch.float64)
    # rotations, translations = get_initial_parameters(true_params, device)
    reg = Registration(
        drr,
        rotations.clone(),
        translations.clone(),
        parameterization="euler_angles",
        convention="ZYX",
    )
    params = optimize(reg, x_img_tensor, scaleInen, tensor_mask, 1e-2, 0.5, optimizer="adam")
    # params = optimize(reg, x_img_tensor, scaleInen, tensor_mask, 1e-2, 1e1, momentum=0.9, dampening=0.1)
    print(params)
    params.to_csv('results/tuodao/pzp_pose1.csv', index=False)
    T2 = time.time()
    print('程序运行时间:%s秒' % (T2 - T1))
    # params_adam = optimize(reg, ground_truth)
    # bg_img_tensor = torch.permute(ground_truth, (0, 1, 3, 2))
    # animate_in_browser(params, len(params), drr, ground_truth)
    del drr


def optimize(
        reg: Registration,
        ground_truth,
        scaler=None,
        masked=None,
        lr_rotations=5.3e-2,
        lr_translations=7.5e1,
        momentum=0,
        dampening=0,
        n_itrs=300,
        optimizer="sgd",  # 'sgd' or `adam`
):
    criterion = NormalizedCrossCorrelation2d()
    # criterion = gradncc
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5, last_epoch=-1)
    params = []
    losses = []
    for itr in tqdm(range(n_itrs), ncols=100):
        # Save the current set of parameters
        alpha, beta, gamma = reg.get_rotation().squeeze().tolist()
        bx, by, bz = reg.get_translation().squeeze().tolist()
        params.append([i for i in [alpha, beta, gamma, bx, by, bz]])
        # Run the optimization loop
        optimizer.zero_grad()
        estimate = reg()
        # print(estimate.shape)
        # estimate_thresh = CannyFilter(estimate)
        estimate = scaler(estimate)
        # print(estimate.shape)
        # print(ground_truth.shape)
        # loss = criterion(ground_truth, estimate)
        loss = criterion(estimate, ground_truth, masked)
        loss.backward()
        optimizer.step()
        # print(loss.item())
        losses.append(loss.item())
        # print(abs(losses[itr - 1] - loss.item()))
        # scheduler.step()
        if abs(losses[itr - 1] - loss.item()) < 0.000001 and itr > 0:
        # if loss > 0.999:
            tqdm.write(f"Converged in {itr} iterations")
            break
    df = pd.DataFrame(params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
    df["loss"] = losses
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


def CannyFilter(input_img):
    kernel = np.array([[-1, -1, -1],  # 边缘检测
                       [-1, 8, -1],
                       [-1, -1, -1]])
    conv2d = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel.shape, padding=1, stride=(1, 1)).to(
        device)
    kernel = torch.from_numpy(kernel.astype(np.float32)).reshape((1, 1, kernel.shape[0], kernel.shape[1])).to(device)
    conv2d.weight.data = kernel
    canny_img = conv2d(input_img)
    return canny_img


def cal_ncc(I, J, eps):
    # compute local sums via convolution
    cross = (I - torch.mean(I)) * (J - torch.mean(J))
    I_var = (I - torch.mean(I)) * (I - torch.mean(I))
    J_var = (J - torch.mean(J)) * (J - torch.mean(J))

    cc = torch.sum(cross) / torch.sum(torch.sqrt(I_var * J_var + eps))

    test = torch.mean(cc)
    return test


# Gradient-NCC Loss
def gradncc(I, J,  target_mask, device='cuda', win=None, eps=1e-10):
    # compute filters
    with torch.no_grad():
        kernel_X = torch.Tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]])
        kernel_X = torch.nn.Parameter(kernel_X, requires_grad=False)
        kernel_Y = torch.Tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]])
        kernel_Y = torch.nn.Parameter(kernel_Y, requires_grad=False)
        SobelX = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        SobelX.weight = kernel_X
        SobelY = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        SobelY.weight = kernel_Y
        SobelX = SobelX.to(device)
        SobelY = SobelY.to(device)

    Ix = SobelX(I)
    Iy = SobelY(I)
    Jx = SobelX(J)
    Jy = SobelY(J)
    Jx = Jx * target_mask
    Jy = Jy * target_mask
    Ix = torch.permute(Ix, (0, 1, 3, 2))
    Jx = torch.permute(Jx, (0, 1, 3, 2))
    Iy = torch.permute(Iy, (0, 1, 3, 2))
    Jy = torch.permute(Jy, (0, 1, 3, 2))
    plt.subplot(1, 4, 1)
    plt.imshow(Ix.squeeze().detach().cpu().numpy())
    plt.subplot(1, 4, 2)
    plt.imshow(Jx.squeeze().detach().cpu().numpy())
    plt.subplot(1, 4, 3)
    plt.imshow(Iy.squeeze().detach().cpu().numpy())
    plt.subplot(1, 4, 4)
    plt.imshow(Jy.squeeze().detach().cpu().numpy())
    plt.show()

    return 1 - 0.5 * cal_ncc(Ix, Jx, eps) - 0.5 * cal_ncc(Iy, Jy, eps)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    reg_method()
