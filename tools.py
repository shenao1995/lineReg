import os.path
from skimage import exposure
import skimage
from PIL import Image
import torch
import numpy as np
import nibabel as nib
from diffdrr.drr import DRR
from torchvision.ops import masks_to_boxes
import cv2
from skimage.feature import canny
import matplotlib.pyplot as plt
from monai.transforms import Resize
from torchvision.transforms.functional import center_crop, gaussian_blur
import SimpleITK as sitk
from diffdrr.pose import convert, RigidTransform, matrix_to_euler_angles
import xml.etree.ElementTree as ET
from diffdrr.data import read


def get_drr(img_path=None, img_meta=None, mask_path=None, SDD=562.0, DELX=1.2, offset_trans=None, poseture='ap',
            device='cuda', V2D_distance=0.0):
    if poseture == 'ap':
        alpha = 0
    elif poseture == 'lar':
        alpha = torch.pi
    else:
        alpha = torch.pi / 2
    HEIGHT = 256
    if img_path:
        subject = read(img_path, bone_attenuation_multiplier=10.5)
    else:
        ct_img = img_meta
        volume = sitk.GetArrayFromImage(ct_img)
        volume = volume.astype(float)
    # 获得初始位姿，其中bxbybz后面是偏移量，裁剪完椎骨后需要的
    if offset_trans is None:
        ini_rotations = torch.tensor([[alpha, 0, 0]]).to(device)
        ini_translations = torch.tensor([[0, V2D_distance, 0]]).to(device)
    else:
        ini_rotations = torch.tensor([[alpha, 0, 0]]).to(device)
        ini_translations = torch.tensor([[offset_trans[0], V2D_distance+offset_trans[1], offset_trans[2]]]).to(device)
    # volume = vert_img.get_fdata()
    # if not tissue:
    #     process_volume = (volume - volume.min()) / (volume.max() - volume.min())
    # else:
    #     process_volume = volume
    # print(volume.shape)
    # drr生成器
    drr = DRR(subject, sdd=SDD, height=HEIGHT, delx=DELX).to(device)
    # cam_pose = convert(ini_rotations, ini_translations, parameterization="euler_angles", convention="ZYX")
    # extrinsic_update = (cam_pose.compose(wld_extrinsic))
    # drr_img = drr(
    #     extrinsic_update
    # )
    # # print(extrinsic_update.matrix[0, :3, 3:].T)
    # update_translation = extrinsic_update.matrix[0, :3, 3:].T
    # update_rotation = matrix_to_euler_angles(extrinsic_update.matrix[0, :3, :3], convention='ZYX')
    # update_rotation = update_rotation.unsqueeze(0)

    return drr


def get_initial_parameters(true_params, used_device):
    # np.random.seed(114)
    rot_range = 6
    trans_range = 50
    alpha = true_params["alpha"] + np.random.uniform(-np.pi / rot_range, np.pi / rot_range)
    beta = true_params["beta"] + np.random.uniform(-np.pi / rot_range, np.pi / rot_range)
    gamma = true_params["gamma"] + np.random.uniform(-np.pi / rot_range, np.pi / rot_range)
    bx = true_params["bx"] + np.random.uniform(-trans_range, trans_range)
    by = true_params["by"] + np.random.uniform(-trans_range, trans_range)
    bz = true_params["bz"] + np.random.uniform(-trans_range, trans_range)
    rotation = torch.tensor([[alpha, beta, gamma]]).to(used_device)
    translation = torch.tensor([[bx, by, bz]]).to(used_device)
    return rotation, translation


def get_lineCenter_offset(pred, target):
    pred_bbx = masks_to_boxes(pred.squeeze(0))
    target_bbx = masks_to_boxes(target.squeeze(0))
    print(pred_bbx)
    # print(target_bbx)
    pred_center_x = pred_bbx[0, 0] + (pred_bbx[0, 2] - pred_bbx[0, 0]) / 2
    pred_center_y = pred_bbx[0, 1] + (pred_bbx[0, 3] - pred_bbx[0, 1]) / 2
    target_center_x = target_bbx[0, 0] + (target_bbx[0, 2] - target_bbx[0, 0]) / 2
    target_center_y = target_bbx[0, 1] + (target_bbx[0, 3] - target_bbx[0, 1]) / 2
    # 初始x相差-5 初始y相差-2，中心点的差距x方向还原取负数，y方向直接加
    offset_x = -(target_center_x - pred_center_x)
    offset_y = target_center_y - pred_center_y
    # print(offset_x)
    # print(offset_y)
    return offset_x, offset_y


def read_bg_img(img_path, reader, isResize=False):
    ground_truth = reader(img_path)
    if isResize:
        print(ground_truth[0].shape)
        resize = Resize(spatial_size=(256, 256), mode='bilinear', align_corners=True)
        ground_truth = resize(ground_truth[0])
        ground_truth = torch.permute(ground_truth, (0, 2, 1))
    else:
        ground_truth = torch.permute(ground_truth[0], (0, 2, 1))
    ground_truth = cv2.cvtColor(ground_truth.squeeze().cpu().numpy(), cv2.COLOR_GRAY2RGB)
    norm_image = cv2.normalize(ground_truth, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    rgb_gt = norm_image * 255
    rgb_gt = rgb_gt.astype(np.uint8)
    return rgb_gt


def extract_img_overlay(img, sigma=1.0, eps=1e-5):
    img_normalized = cv2.normalize(img, None, 0, 255.0,
                                   cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # thresh = cv2.Canny(np.uint8(img_normalized), 1, 150)
    pred = (img_normalized - img_normalized.min()) / (img_normalized.max() - img_normalized.min() + eps)
    edges = canny(pred, sigma=sigma)
    rgb_mov = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
    # edges = np.ma.masked_where(~edges, edges)
    rgb_mov[:, :, 0] = edges
    rgb_mov[:, :, 1] = edges
    rgb_mov[:, :, 2] = edges
    # plt.imshow(edges)
    # plt.colorbar()
    # plt.show()
    # print(itr.squeeze().cpu().numpy().shape)
    # result = cv2.addWeighted(frame, 0.7, x_line_img, 0.7, 0)
    rgb_mov[:, :, 1] = rgb_mov[:, :, 1] * 255
    return rgb_mov


def dual_view_joint(img1, img2):
    height, width, channels = img1.shape
    new_width = width + img2.shape[1]
    # 创建一张新的图片
    new_img = np.zeros((height, new_width, channels), dtype=np.uint8)
    # 将第一张图片复制到新图片的左侧
    new_img[:, :width, :] = img1
    # 将第二张图片复制到新图片的右侧
    new_img[:, width:, :] = img2
    return new_img


def gaussian_preprocess(img, cropped=False, size=None, initial_energy=torch.tensor(65487.0)):
    """
    Recover the line integral: $L[i,j] = \log I_0 - \log I_f[i,j]$

    (1) Remove edge due to collimator
    (2) Smooth the image to make less noisy
    (3) Subtract the log initial energy for each ray
    (4) Recover the line integral image
    (5) Rescale image to [0, 1]
    """
    if cropped:
        img = center_crop(img, [950, 950])
    img = gaussian_blur(img, (5, 5), sigma=1.0)
    img = initial_energy.log() - img.log()
    img = (img - img.min()) / (img.max() - img.min())
    return img


def crop_ct_vert(img_path, mask_path, save_path=None):
    img = sitk.ReadImage(img_path)
    ct_arr = sitk.GetArrayFromImage(img)
    # normalized_arr = (ct_arr - ct_arr.min()) / (ct_arr.max() - ct_arr.min())
    mask = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(mask)
    # seg_arr = np.where(seg_arr != 0, 1, 0)
    normalized_arr = np.where(mask_arr == 1, ct_arr, ct_arr.min())
    # seg_arr = np.where(seg_arr == vert_num, seg_arr, 0)
    processed_img = sitk.GetImageFromArray(normalized_arr)
    # out_seg = sitk.GetImageFromArray(seg_arr)
    # out_seg.CopyInformation(mask)
    processed_img.CopyInformation(img)
    lesion_filter = sitk.LabelShapeStatisticsImageFilter()
    lesion_filter.Execute(mask)
    lesion_boxing = lesion_filter.GetBoundingBox(1)
    boxing_size = (
        lesion_boxing[int(len(lesion_boxing) / 2):][0], lesion_boxing[int(len(lesion_boxing) / 2):][1],
        int(lesion_boxing[5]))
    start_boxing = (
        lesion_boxing[0:int(len(lesion_boxing) / 2)][0], lesion_boxing[0:int(len(lesion_boxing) / 2)][1],
        int(lesion_boxing[2]))
    # print(boxing_size)
    # print(start_boxing)
    spacing = img.GetSpacing()
    # print(spacing)
    # print(img.GetSize())
    ver_center_x, ver_center_y, ver_center_z = start_boxing[0] + boxing_size[0] / 2, \
                                               start_boxing[1] + boxing_size[1] / 2, \
                                               start_boxing[2] + boxing_size[2] / 2
    ct_center_x, ct_center_y, ct_center_z = img.GetSize()[0] / 2, img.GetSize()[1] / 2, img.GetSize()[2] / 2
    ver_center = np.array((ver_center_x, ver_center_y, ver_center_z), dtype=np.float64)
    ct_center = np.array((ct_center_x, ct_center_y, ct_center_z), dtype=np.float64)
    bx, by, bz = (ct_center - ver_center) * spacing
    # print(bx, by, bz)
    # origin_img = nib.load(img_path)
    # x, y, z = nib.aff2axcodes(origin_img.affine)
    # orient = x + y + z
    cropped_img = sitk.RegionOfInterest(processed_img, boxing_size, start_boxing)
    # cropped_mask = sitk.RegionOfInterest(mask, boxing_size, start_boxing)
    cropped_img.SetSpacing(img.GetSpacing())
    cropped_img.SetOrigin(img.GetOrigin())
    cropped_img.SetDirection(img.GetDirection())
    # cropped_arr = sitk.GetArrayFromImage(cropped_img)
    # save_path = os.path.join(save_fold, os.path.split(img_path)[-1])
    # seg_save_path = os.path.join(seg_save_fold, os.path.split(mask_path)[-1])
    # # print(save_path)
    # # print(seg_save_path)
    if save_path:
        if not os.path.exists(save_path):
            sitk.WriteImage(cropped_img, save_path)
    else:
        pass
    # sitk.WriteImage(cropped_mask, seg_save_path)
    return bx, by, bz, cropped_img, save_path


def resample_img(input_img, new_width=None, save_path=None, interpolator_method=sitk.sitkLinear):
    # image_file_reader = sitk.ImageFileReader()
    # # only read DICOM images
    # image_file_reader.SetImageIO("GDCMImageIO")
    # image_file_reader.SetFileName(input_file_name)
    # image_file_reader.ReadImageInformation()
    image_size = list(input_img.GetSize())
    if len(image_size) == 3 and image_size[2] == 1:
        input_img = input_img[:, :, 0]
    # input_img.Set(image_size)
    image = input_img
    if new_width:
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()
        new_spacing = [(original_size[0]) * original_spacing[0] / new_width] * 2
        new_size = [
            new_width,
            int((original_size[1]) * original_spacing[1] / new_spacing[1]),
        ]
        image = sitk.Resample(
            image1=image,
            size=new_size,
            transform=sitk.Transform(),
            interpolator=interpolator_method,
            outputOrigin=image.GetOrigin(),
            outputSpacing=new_spacing,
            outputDirection=image.GetDirection(),
            defaultPixelValue=0,
            outputPixelType=image.GetPixelID(),
        )
    # If a single channel image, rescale to [0,255]. Also modify the
    # intensity values based on the photometric interpretation. If
    # MONOCHROME2 (minimum should be displayed as black) we don't need to
    # do anything, if image has MONOCRHOME1 (minimum should be displayed as
    # white) we flip # the intensities. This is a constraint imposed by ITK
    # which always assumes MONOCHROME2.
    # if image.GetNumberOfComponentsPerPixel() == 1:
    #     image = sitk.RescaleIntensity(image, 0, 255)
    #     if image_file_reader.GetMetaData("0028|0004").strip() == "MONOCHROME1":
    #         image = sitk.InvertIntensity(image, maximum=255)
    #     image = sitk.Cast(image, sitk.sitkUInt8)
    # sitk.WriteImage(image, output_file_name)
    if save_path:
        if not os.path.exists(save_path):
            sitk.WriteImage(image, save_path)
    return image


def bbx_crop_gt_vert(img, mask, inverse=True):
    # img_path = 'Data/gncc_data/tuodao/dingjunmei/X/djm_resized_la.nii.gz'
    # mask_path = 'Data/gncc_data/tuodao/dingjunmei/X/djm_resized_la_L2_bbx.nii.gz'
    # img = sitk.ReadImage(img_path)
    # mask = sitk.ReadImage(mask_path)
    img_arr = sitk.GetArrayFromImage(img)
    if inverse:
        if len(img_arr.shape) > 2:
            img_arr = np.max(img_arr) - img_arr[0, :, :]
        else:
            img_arr = np.max(img_arr) - img_arr
    lesion_filter = sitk.LabelShapeStatisticsImageFilter()
    lesion_filter.Execute(mask)
    lesion_boxing = lesion_filter.GetBoundingBox(1)
    # print(lesion_boxing)
    img_arr[:, :lesion_boxing[0]] = 0
    img_arr[:, lesion_boxing[0] + lesion_boxing[2]:] = 0
    img_arr[:lesion_boxing[1], :] = 0
    img_arr[lesion_boxing[1] + lesion_boxing[3]:, :] = 0
    out_img = sitk.GetImageFromArray(img_arr)
    out_img.CopyInformation(img)
    return out_img
    # save_path = os.path.join(gt_vert_save_fold, seg)
    # print(save_path)
    # print(seg_save_path)


def HE_optimize(img):
    # img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = np.max(img) - img
    # 添加噪声
    # noisy = skimage.util.random_noise(img, mode='gaussian', var=0.01)
    # img_HE = img + noisy

    # 基础的HE方式效果并不好
    # img_HE = cv2.equalizeHist(img)

    # AHE方法
    # img1 = exposure.equalize_adapthist(img)
    # img_AHE = Image.fromarray(np.uint8(img1 * 255))
    # img_AHE = np.array(img_AHE)

    # CLAHE方法
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(10, 10))
    img_CLAHE = clahe.apply(img)
    # 添加噪声，高斯
    noisyCLAHE = skimage.util.random_noise(img_CLAHE, mode='gaussian', var=0.05)
    img_CLAHE_g = img_CLAHE + noisyCLAHE
    # 泊松
    noisyCLAHE2 = skimage.util.random_noise(img_CLAHE, mode='poisson', clip=True)
    img_CLAHE_p = img_CLAHE + noisyCLAHE2
    # 尝试其他参数进行调参
    # clahe2 = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
    # img_CLAHE2 = clahe2.apply(img)
    # 演示效果
    # plot_show_n(img, img_CLAHE, img_CLAHE_g, img_CLAHE_p)
    # 返回修改后的图像
    # print("ready to write new picture")
    # dir_name = "D:\\xregAPI\\spatial_data2024\\tuodao_ceshi\\xray\\pzp_la60_HE.png"
    # cv2.imwrite(dir_name, img_CLAHE_g)
    return img_CLAHE_p


def read_xml(xml_path):
    # 读取XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # 打印根元素的名称
    ImgCenterWld = root[2].text.lstrip().split(' ')
    XrayCenterWld = root[4].text.lstrip().split(' ')
    ImgXdir = root[5].text.lstrip().split(' ')
    ImgYdir = root[6].text.lstrip().split(' ')
    X_Spacing = float(root[0].text.lstrip())
    img_center = np.array([float(ImgCenterWld[0]), float(ImgCenterWld[1]), float(ImgCenterWld[2])])
    xray_center = np.array([float(XrayCenterWld[0]), float(XrayCenterWld[1]), float(XrayCenterWld[2])])
    SDD = np.sqrt(np.sum((img_center - xray_center) ** 2))
    Xray_H = root[3].text.lstrip().split(' ')[0]
    return ImgXdir, ImgYdir, X_Spacing, SDD, Xray_H, XrayCenterWld


def get_ext_pose(Xdir, Ydir, V2D_distance, transform_vect, offset_trans=None, device='cuda'):
    # 获得初始位姿，其中bxbybz后面是偏移量，裁剪完椎骨后需要的
    if offset_trans is None:
        ini_rotations = transform_vect[:, :3]
        # ini_rotations = torch.tensor([[0.0, 0.0, 0.0]]).to(device)
        # print(ini_rotations)
        ini_translations = transform_vect[:, 3:] + torch.tensor([[0, V2D_distance, 0]]).to(device)
    else:
        offset_tensor = torch.from_numpy(offset_trans).unsqueeze(0).to(device)
        # print(offset_tensor)
        ini_rotations = transform_vect[:, :3]
        ini_translations = transform_vect[:, 3:] + offset_tensor + torch.tensor([[0, V2D_distance, 0]]).to(device)
    # pose_unfixed = convert(ini_rotations, ini_translations, parameterization="euler_angles", convention="ZXY")
    x_dir = np.array([float(Xdir[0]), float(Xdir[1]), float(Xdir[2])])
    y_dir = np.array([float(Ydir[0]), float(Ydir[1]), float(Ydir[2])])
    z_dir = np.cross(x_dir, y_dir)
    wld_extrinsic_R = np.array([x_dir, z_dir, y_dir])
    wld_extrinsic_T = np.array([0, 0, 0]).reshape(3, 1)
    wld_extrinsic = np.concatenate((wld_extrinsic_R, wld_extrinsic_T.reshape(3, 1)), axis=1)
    wld_extrinsic = np.vstack((wld_extrinsic, np.array([0.0, 0.0, 0.0, 1.0])))
    wld_extrinsic = torch.FloatTensor(wld_extrinsic)
    wld_extrinsic = RigidTransform(wld_extrinsic).to(device)
    update_rotation = matrix_to_euler_angles(wld_extrinsic.matrix[0, :3, :3], convention='ZXY').unsqueeze(0)
    # print(update_rotation)
    ini_rotations = ini_rotations - update_rotation.unsqueeze(0)
    zero = torch.tensor([[0.0, 0.0, 0.0]]).to(ini_rotations)
    R = convert(
        ini_rotations,
        zero,
        parameterization="euler_angles",
        convention="ZXY",
    )
    t = convert(
        zero,
        ini_translations,
        parameterization="euler_angles",
        convention="ZXY",
    )
    pose_update = t.compose(R)
    return pose_update
