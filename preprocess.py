import SimpleITK as sitk
import numpy as np
from monai.transforms import LoadImage, ScaleIntensity
import torch
from diffdrr.drr import DRR
from diffdrr.visualization import plot_drr
import matplotlib.pyplot as plt
from tools import get_initial_parameters
import os
from pycocotools.coco import COCO
import cv2
from monai.transforms import Resize
import nibabel as nib
import csv
from diffdrr.metrics import NormalizedCrossCorrelation2d, GradientNormalizedCrossCorrelation2d
from monai.networks.utils import one_hot
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from torchvision.ops import masks_to_boxes
from tools import gaussian_preprocess, resample_img


def crop_image():
    img_path = 'Data/natong/case2/X/AP.nii.gz'
    img = sitk.ReadImage(img_path)
    img_arr = sitk.GetArrayFromImage(img)
    img_arr = img_arr.max() - img_arr
    print(img_arr.shape)
    crop_with = (img_arr.shape[-1] - 2048) // 2
    cropped_arr = img_arr[0:, 3028 - 2048:, crop_with:2048 + crop_with]

    # print(resized_img.shape)
    out_img = sitk.GetImageFromArray(cropped_arr)
    print(cropped_arr.shape)
    out_img.SetSpacing(img.GetSpacing())
    # out_img.CopyInformation(img)
    sitk.WriteImage(out_img, 'Data/natong/case2/X/cropped_AP.nii.gz')


def inverse_img():
    img_path = 'Data/gncc_data/91B_resized.nii.gz'
    img = sitk.ReadImage(img_path)
    img_arr = sitk.GetArrayFromImage(img)
    out_img = sitk.GetImageFromArray(img_arr.max() - img_arr)
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, 'inverse_img.nii')


def crop_region():
    img_path = 'Data/gncc_data/tuodao/peizongping/X/peizongping_180_x.nii.gz'
    seg_path = 'Data/gncc_data/tuodao/peizongping/X/peizongping_180_x_L3_bbx.nii.gz'
    img = sitk.ReadImage(img_path)
    seg = sitk.ReadImage(seg_path)
    # img_arr = sitk.GetArrayFromImage(img)
    seg_arr = sitk.GetArrayFromImage(seg)
    seg_arr = np.where(seg_arr != 0, 1, 0)
    used_mask = sitk.GetImageFromArray(seg_arr)
    used_mask.CopyInformation(seg)
    lesion_filter = sitk.LabelShapeStatisticsImageFilter()
    lesion_filter.Execute(used_mask)
    lesion_boxing = lesion_filter.GetBoundingBox(1)
    boxing_size = (
        lesion_boxing[int(len(lesion_boxing) / 2):][0], lesion_boxing[int(len(lesion_boxing) / 2):][1],
        int(lesion_boxing[5]))
    start_boxing = (
        lesion_boxing[0:int(len(lesion_boxing) / 2)][0], lesion_boxing[0:int(len(lesion_boxing) / 2)][1],
        int(lesion_boxing[2]))
    # if len(seg_arr.shape) > 2:
    #     seg_arr = seg_arr[0]
    cropped_img = sitk.RegionOfInterest(img, boxing_size, start_boxing)
    # cropped_mask = sitk.RegionOfInterest(mask, boxing_size, start_boxing)
    # print(img_save_path)
    # sitk.WriteImage(cropped_img, img_save_path)
    # sitk.WriteImage(cropped_mask, mask_save_path)
    sitk.WriteImage(cropped_img, 'Data/gncc_data/tuodao/peizongping/X/pzp_L3.nii.gz')


def generate_gt_drr():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    reader = LoadImage(ensure_channel_first=True, image_only=False)
    np.random.seed(88)
    # Make the ground truth X-ray
    SDR = 1200.0
    HEIGHT = 256
    DELX = 1.2
    ctDir = 'Data/gncc_data/tuodao/peizongping/peizongping_L3.nii.gz'
    used_ct_arr = reader(ctDir)
    spacing = used_ct_arr[1]['pixdim']
    spacing = np.array((spacing[1], spacing[2], spacing[3]), dtype=np.float64)
    # print(spacing)
    bx, by, bz = torch.tensor(used_ct_arr[0][0].shape) * torch.tensor(spacing) / 2
    true_params = {
        "sdr": SDR,
        "alpha": torch.pi / 2,  # 沿y轴旋转
        "beta": 0,
        "gamma": torch.pi,  # 沿z轴旋转
        "bx": bx,
        "by": by,  # 沿z轴平移
        "bz": bz,  # 沿y轴平移
    }
    # print(used_ct_arr[0].clone().detach().shape)
    drr = DRR(used_ct_arr[0][0], spacing, sdr=SDR, height=HEIGHT, delx=DELX, bone_attenuation_multiplier=5.5).to(device)
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
    # print(rotations)
    # print(translations)
    # rotations, translations = get_initial_parameters(true_params, device)
    print(rotations, translations)
    ground_truth = drr(
        rotation=rotations,
        translation=translations,
        parameterization="euler_angles",
        convention="ZYX",
    )
    ground_truth = torch.permute(ground_truth, (0, 1, 3, 2))
    plot_drr(ground_truth)
    plt.show()
    ground_truth = ground_truth.squeeze().cpu().numpy()
    ground_truth = sitk.GetImageFromArray(ground_truth)
    sitk.WriteImage(ground_truth, 'Data/pzp_gt.nii.gz')


def get_single_vert():
    # img_path = 'Data/gncc_data/weng_fang_qi.nii.gz'
    seg_path = 'Data/gncc_data/weng_fang_qi_L1_L5_seg.nii.gz'
    # img = sitk.ReadImage(img_path)
    seg = sitk.ReadImage(seg_path)
    # img_arr = sitk.GetArrayFromImage(img)
    seg_arr = sitk.GetArrayFromImage(seg)
    out_arr = np.where(seg_arr == 2, seg_arr, 0)
    print(out_arr.shape)
    out_img = sitk.GetImageFromArray(out_arr)
    out_img.CopyInformation(seg)
    sitk.WriteImage(out_img, 'Data/gncc_data/weng_L2_CT_seg.nii')


def seg_preprocess():
    data_fold = 'Data/natong/case3'
    ct_path = 'Data/natong/case3/case3_ct.nii.gz'
    img = sitk.ReadImage(ct_path)
    # for i in range(1, 5):
    for file in os.listdir(data_fold):
        if 'seg' in file:
            # if 'ct' in file:
            # seg_path = 'Data/natong/case1/case1_16_prediction_resampled.nii.gz'
            # img_path = 'Data/natong/case1/case1_ct.nii'
            seg_path = os.path.join(data_fold, file)
            print(seg_path)
            target_orient = 'RAS'
            seg = sitk.ReadImage(seg_path)
            seg_arr = sitk.GetArrayFromImage(seg)
            out_arr = np.where(seg_arr != 0, 1, 0)
            out_mask = sitk.GetImageFromArray(out_arr)
            # img = sitk.ReadImage(img_path)
            # if '_ct' not in file:
            #     seg_arr = sitk.GetArrayFromImage(seg)
            #     out_arr = np.where(seg_arr != 0, 1, 0)
            #     out_mask = sitk.GetImageFromArray(out_arr)
            # else:
            #     out_mask = seg
            Direction = out_mask.GetDirection()
            Orient_Matrix = np.asmatrix(Direction).reshape(3, 3)
            Orient_dict = {1: 'L', 2: 'P', 3: 'S',
                           -1: 'R', -2: 'A', -3: 'I'}
            orientKey = np.squeeze(np.asarray(Orient_Matrix.T.dot(np.array([1, 2, 3])), dtype=np.int8))
            origin_orient = Orient_dict[orientKey[0]] + Orient_dict[orientKey[1]] + Orient_dict[orientKey[2]]
            print(origin_orient)
            if origin_orient != target_orient:
                print('Change the orientation to ', target_orient)
                sitk_vol = sitk.DICOMOrient(out_mask, target_orient)
                # sitk_vol.CopyInformation(seg)
                sitk_vol.SetOrigin(img.GetOrigin())
                sitk_vol.SetSpacing(seg.GetSpacing())
                sitk.WriteImage(sitk_vol, seg_path)
                # sitk.WriteImage(sitk_mask, seg_path)


def read_xray_and_crop():
    largerX = 50
    largerY = 20
    file_name = 'Data/gncc_data/exp2024.1.4/test.png'
    # print(file_names)
    # img = io.imread(file_name)
    # img_new = rgb2gray(img)
    img = cv2.imread(file_name)
    img_new = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    count = 0
    # for anno in annos:
    #     count += 1
    # if count == len(annos):
    #     pass
    # else:
    y = int(71.90629577636719)
    x = int(200.06675720214844)
    height = int(118.33663940429688)
    width = int(178.9586944580078)
    mask = np.zeros(img_new.shape, dtype="uint8")
    # print(y)
    cv2.rectangle(mask, (x - largerX, y - largerY), (x + width + largerX * 2, y + height + largerY * 2),
                  1, -1)
    # cv2.imshow("Rectangular Mask", mask)
    masked = cv2.bitwise_and(img_new, img_new, mask=mask)
    norm_masked = (masked - np.min(masked)) / (np.max(masked) - np.min(masked))
    # img_arr = torch.from_numpy(norm_masked)
    # img_arr = torch.unsqueeze(img_arr, dim=0)
    # crop = CenterSpatialCrop(roi_size=[1024, 1024])
    # cropped_img = crop(img_arr)
    # resize = Resize(spatial_size=(256, 256), mode='bilinear', align_corners=True)
    # resized_img = resize(cropped_img)
    # print(resized_img.shape)
    # out_img = sitk.GetImageFromArray(resized_img.squeeze().cpu().numpy())
    out_img = sitk.GetImageFromArray(norm_masked)
    save_path = os.path.join('Data/gncc_data/exp2024.1.4/test_L2_cropped.nii')
    sitk.WriteImage(out_img, save_path)
    # plt.subplot(1, 2, 1)
    # plt.imshow(img_new, cmap='gray')
    # plt.subplot(1, 2, 2)
    # plt.imshow(masked, cmap='gray')
    # plt.show()


def png2nii():
    file_path = 'Data/tuodao/dingjunmei/X/x_la.tif'
    img = cv2.imread(file_path)
    img_new = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_tensor = torch.tensor(img_new)
    # img_tensor = gaussian_preprocess(img_tensor.unsqueeze(0))
    # processed_img = sitk.GetImageFromArray(img_tensor.squeeze().cpu().numpy())
    out_img = sitk.GetImageFromArray(img_new)
    # out_img.SetSpacing([3.0360001325607300e-01, 3.0360001325607300e-01])
    # resized_img = sitk.Shrink(out_img, [4, 4])
    # resized_img = resample_img(out_img, new_width=256)
    print(out_img.GetSize())
    print(out_img.GetSpacing())
    sitk.WriteImage(out_img, 'Data/tuodao/dingjunmei/X/dingjunmei_la.nii.gz')


def resize_img():
    mask_path = 'Data/gncc_data/exp2024.1.15/weng_L2_seg_pred.nii'
    mask = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(mask)
    print(mask_arr.shape)
    # x_img_arr = np.where(mask_arr, x_img_arr, 0)
    mask_arr = torch.from_numpy(mask_arr / 1.0)
    mask_arr = torch.unsqueeze(mask_arr, dim=0)
    resize = Resize(spatial_size=(256, 256), mode='bilinear', align_corners=True)
    resized_img = resize(mask_arr)
    print(resized_img.shape)
    # plt.figure('show', (12, 5), dpi=300)
    # plt.subplot(1, 2, 1)
    # plt.title("Origin")
    # plt.imshow(img_arr[0], cmap='gray')
    # plt.subplot(1, 2, 2)
    # plt.title("Resized")
    # plt.imshow(resized_img[0], cmap='gray')
    # plt.show()
    out_img = sitk.GetImageFromArray(resized_img.squeeze().cpu().numpy())
    sitk.WriteImage(out_img, 'Data/gncc_data/exp2024.1.15/weng_L2_seg_resized_pred.nii')


def generate_mov_drr():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_drrs = 5000
    ct_path = 'Data/gncc_data/tuodao/peizongping/peizongping_L3.nii.gz'
    save_fold = 'Data/dl_drr'
    csv_path = 'Data/pose1.csv'
    volume, spacing, true_params = get_true_drr(ct_path)
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(
            [
                "name",
                "rx",
                "ry",
                "rz",
                "tx",
                "ty",
                "tz",
            ]
        )
        drr_moving = DRR(
            volume,
            spacing,
            sdr=1200,
            height=256,
            delx=1.2,
        ).to(device)
        for i in range(n_drrs):
            rx, ry, rz, tx, ty, tz = get_transform_parameters()
            writer.writerow([str(i), rx, ry, rz, tx, ty, tz])
            mov_rot = torch.tensor([[true_params["rx"] + rx, true_params["ry"] + ry, true_params["rz"] + rz]],
                                   device=device)
            mov_trans = torch.tensor([[true_params["tx"] + tx, true_params["ty"] + ty, true_params["tz"] + tz]],
                                     device=device)
            moving_drr = drr_moving(mov_rot, mov_trans, parameterization="euler_angles", convention="ZYX")
            moving_drr = torch.permute(moving_drr, (0, 1, 3, 2))
            moving_drr = moving_drr.squeeze().cpu().numpy()
            # plt.imsave('Data/moving_drr/chazhilin_moving_{num}.png'.format(num=str(i)), moving_drr, cmap='gray')
            moving_img = sitk.GetImageFromArray(moving_drr)
            save_path = os.path.join(save_fold, 'ap_' + str(i) + '.nii.gz')
            sitk.WriteImage(moving_img, save_path)


def get_true_drr(img_path):
    """Get parameters for the fixed DRR."""
    np.random.seed(88)
    SDR = 1200.0
    origin_img = nib.load(img_path)
    spacing = origin_img.header.get_zooms()
    spacing = np.array((spacing[0], spacing[1], spacing[2]), dtype=np.float64)
    bx, by, bz = torch.tensor(origin_img.shape) * torch.tensor(spacing) / 2
    true_params = {
        "sdr": SDR,
        "rx": torch.pi / 2,  # 沿y轴旋转，正数是逆时针旋转，0是右，torch.pi是左
        "ry": 0,
        "rz": torch.pi,  # 沿z轴旋转
        "tx": bx,
        "ty": by,  # 沿z轴平移
        "tz": bz,  # 沿y轴平移
    }
    return origin_img.get_fdata(), spacing, true_params


def get_transform_parameters():
    rot_range = 18
    trans_range = 20.0
    """Get starting parameters for the moving DRR by perturbing the true params."""
    rx = np.random.uniform(-np.pi / rot_range, np.pi / rot_range)
    ry = np.random.uniform(-np.pi / rot_range, np.pi / rot_range)
    rz = np.random.uniform(-np.pi / rot_range, np.pi / rot_range)
    tx = np.random.uniform(-trans_range, trans_range)
    ty = np.random.uniform(-trans_range, trans_range)
    tz = np.random.uniform(-10, 40)
    return rx, ry, rz, tx, ty, tz


def test_ncc():
    ncc = NormalizedCrossCorrelation2d()
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    target_path = 'Data/diff_lines/pzp_lines/pzp_ap_0_seg.nii.gz'
    mov_path = 'Data/diff_lines/pzp_lines/pzp_ap_2_seg.nii.gz'
    target_img = sitk.ReadImage(target_path, sitk.sitkFloat32)
    mov_img = sitk.ReadImage(mov_path, sitk.sitkFloat32)
    target_arr = sitk.GetArrayFromImage(target_img)
    mov_arr = sitk.GetArrayFromImage(mov_img)
    target_tensor = torch.from_numpy(target_arr).unsqueeze(0)
    mov_tensor = torch.from_numpy(mov_arr).unsqueeze(0)
    # target_tensor = one_hot(target_tensor, num_classes=2)
    # mov_tensor = one_hot(mov_tensor, num_classes=2)
    print(target_tensor.shape)
    bbx1 = masks_to_boxes(target_tensor.squeeze(0))
    bbx2 = masks_to_boxes(mov_tensor.squeeze(0))
    print(bbx1)
    print(bbx2)
    # print(mov_tensor[:, 0, ].shape)
    # metric = ncc(mov_tensor, target_tensor)
    dice = dice_metric(mov_tensor, target_tensor)
    # plot_drr(mov_tensor[:, 1, ].unsqueeze(0))
    # plot_drr(target_tensor)
    # plt.show()
    print(dice)


def bbx_crop_gt_vert():
    img_path = 'Data/gncc_data/tuodao/dingjunmei/X/djm_resized_la.nii.gz'
    mask_path = 'Data/gncc_data/tuodao/dingjunmei/X/djm_resized_la_L2_bbx.nii.gz'
    img = sitk.ReadImage(img_path)
    mask = sitk.ReadImage(mask_path)
    img_arr = sitk.GetArrayFromImage(img)
    print(img_arr.shape)
    # if len(img_arr.shape) > 2:
    #     img_arr = np.max(img_arr) - img_arr[0, :, :]
    # else:
    #     img_arr = np.max(img_arr) - img_arr
    mask_arr = sitk.GetArrayFromImage(mask)
    mask = sitk.GetImageFromArray(mask_arr[0, :, :])
    print(mask_arr.shape)
    # if mask_arr.sum() > 500:
    # img_arr = np.where(img_arr == mask_arr)
    # print(img_arr[0])
    # mask_arr = sitk.GetArrayFromImage(mask)
    # print(img_arr[0])
    lesion_filter = sitk.LabelShapeStatisticsImageFilter()
    lesion_filter.Execute(mask)
    lesion_boxing = lesion_filter.GetBoundingBox(1)
    print(lesion_boxing)
    img_arr[:, :lesion_boxing[0]] = 0
    img_arr[:, lesion_boxing[0] + lesion_boxing[2]:] = 0
    img_arr[:lesion_boxing[1], :] = 0
    img_arr[lesion_boxing[1] + lesion_boxing[3]:, :] = 0
    out_img = sitk.GetImageFromArray(img_arr)
    out_img.CopyInformation(img)
    # save_path = os.path.join(gt_vert_save_fold, seg)
    # print(save_path)
    # print(seg_save_path)
    sitk.WriteImage(out_img, 'Data/gncc_data/tuodao/dingjunmei/X/djm_la_L2.nii.gz')


def dcm2nii():
    # data_path = 'data/T2_driven'
    # save_path = 'data/T2_nii'
    data_path = 'F:/natong/SpineData/case1/Spine3DImage(1)/Spine3DImage/0403103718'
    save_fold = 'Data/natong/case1'
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(data_path)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_path, series_IDs[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image = series_reader.Execute()
    # print(os.path.join(save_path, fold + '.nii'))
    # print(os.path.split(data_path))
    sitk.WriteImage(image, os.path.join(save_fold, 'case1_ct.nii'))


def correct_seg():
    seg_path = 'Data/tuodao/peizongping/L3_seg.nii.gz'
    seg = sitk.ReadImage(seg_path)
    seg_arr = sitk.GetArrayFromImage(seg)
    seg_arr = np.where(seg_arr != 0, 1, 0)
    out_seg = sitk.GetImageFromArray(seg_arr)
    out_seg.CopyInformation(seg)
    sitk.WriteImage(out_seg, 'Data/tuodao/peizongping/L3_seg_temp.nii.gz')


if __name__ == '__main__':
    # test_ncc()
    png2nii()
    # dcm2nii()
    # bbx_crop_gt_vert()
    # read_xray_and_crop()
    # generate_gt_drr()
    # generate_mov_drr()
    # crop_image()
    # inverse_img()
    # crop_region()
    # get_single_vert()
    # seg_preprocess()
    # resize_img()
    # correct_seg()
