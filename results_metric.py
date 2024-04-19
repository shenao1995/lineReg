import SimpleITK as sitk
import numpy as np
import os
from skimage import transform
import torch
import pandas as pd
from PIL import Image
import nibabel as nib
import cv2
import imageio
from monai.transforms import Resize, CenterSpatialCrop
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from pycocotools.coco import COCO
import skimage.io as io
from skimage.color import rgb2gray
from mpl_toolkits.mplot3d import Axes3D
import h5py
from monai.transforms import LoadImage, ScaleIntensity
from diffdrr.drr import DRR
import pyvista
from diffdrr.visualization import drr_to_mesh, img_to_mesh
from IPython.display import IFrame
# from diffpose.visualization import fiducials_to_mesh, lines_to_mesh
from sklearn.metrics import mean_absolute_error
from diffdrr.pose import convert
from tools import get_drr, read_bg_img, extract_img_overlay, dual_view_joint, resample_img
from diffpose.visualization import overlay_edges
import seaborn as sns
from matplotlib import font_manager, rcParams


def animate_reg_process():
    caseName = 'dukemei'
    isDualView = False
    video_name = 'results/tuodao/{}_L3_la.mp4'.format(caseName)
    results_path = 'results/tuodao/{}_L3_pose.csv'.format(caseName)
    poses_data = pd.read_csv(results_path)
    reader = LoadImage(ensure_channel_first=True, image_only=False)
    SDR = 570
    HEIGHT = 256
    DELX = 1.1266406741924584
    ctDir = 'Data/tuodao/{}/{}_L3.nii.gz'.format(caseName, caseName)
    used_ct_arr = reader(ctDir)
    spacing = used_ct_arr[1]['pixdim']
    # print(spacing1)
    spacing = np.array((spacing[1], spacing[2], spacing[3]), dtype=np.float64)
    bg_path = 'Data/tuodao/{}/X/{}_resized_x_la.nii.gz'.format(caseName, caseName)
    la_bg_path = 'Data/tuodao/{}/X/{}_resized_x_la.nii.gz'.format(caseName, caseName)
    rgb_ap_gt = read_bg_img(bg_path, reader)
    if isDualView:
        rgb_la_gt = read_bg_img(la_bg_path, reader)
    plt.imshow(rgb_ap_gt)
    plt.colorbar()
    plt.show()
    # plt.imshow(rgb_la_gt)
    # plt.colorbar()
    # plt.show()
    drr_generator = DRR(used_ct_arr[0][0].cpu().numpy(), spacing, sdr=SDR * 2, height=HEIGHT, delx=DELX,
                        bone_attenuation_multiplier=10.5).to(device)
    if isDualView:
        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, (rgb_ap_gt.shape[0] * 2, rgb_ap_gt.shape[1]))
    else:
        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, (rgb_ap_gt.shape[0], rgb_ap_gt.shape[1]))
    for idx, row in poses_data.iterrows():
        # fig, ax_opt = make_fig() if ground_truth is None else make_fig(ground_truth)
        # params = row[["alpha", "beta", "gamma", "bx", "by", "bz"]].values
        rotations = (
            torch.tensor(row[["alpha", "beta", "gamma"]].values).unsqueeze(0)
            .to(device)
        )
        # la_rotations = rotations + torch.tensor([[-torch.pi / 2, 0, 0]]).to(device)
        # print(rotations)
        # rotations = rotations + to
        translations = (
            torch.tensor(row[["bx", "by", "bz"]].values).unsqueeze(0).to(device)
        )
        # print(row[["bx", "by", "bz"]].values)
        itr = drr_generator(rotations.float(), translations.float(), parameterization="euler_angles",
                            convention="ZYX", dual_view=isDualView)
        # print(itr.shape)
        itr = torch.permute(itr, (0, 1, 3, 2))
        itr = itr.squeeze().cpu().numpy()
        # print(itr.shape)
        if isDualView:
            ap_contour = extract_img_overlay(itr[0, :])
            la_contour = extract_img_overlay(itr[1, :])
            rgb_mov = dual_view_joint(ap_contour, la_contour)
            rgb_gt = dual_view_joint(rgb_ap_gt, rgb_la_gt)
            result = cv2.addWeighted(rgb_gt, 1, rgb_mov, 0.8, 0)
        else:
            ap_contour = extract_img_overlay(itr, sigma=0.5)
            result = cv2.addWeighted(rgb_ap_gt, 1, ap_contour, 0.8, 0)
        video.write(result)
    cv2.destroyAllWindows()
    video.release()
    del drr_generator


def reg_process_visual_3d():
    # Read in the volume and get the isocenter
    # reader = LoadImage(ensure_channel_first=True)
    pose_path = 'results/singleView_lines/dingjunmei_L2_pose.csv'
    poses_data = pd.read_csv(pose_path)
    vert_Dir = 'Data/gncc_data/tuodao/dingjunmei/dingjunmei_L2.nii.gz'
    ct_Dir = 'Data/gncc_data/tuodao/dingjunmei/dingjunmei.nii.gz'
    # Make a mesh from the CT volume
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
    offset_trans = np.array([- offsetx, offsety, offsetz])
    # print(offset_trans)
    _, vert_drr, _, gt_rotations, gt_translations = get_drr(vert_Dir, offset_trans=offset_trans, tissue=True)
    _, ct_drr, _, ct_gt_rotations, ct_gt_translations = get_drr(ct_Dir, tissue=True)
    # ct_mesh = drr_to_mesh(ct_drr, "surface_nets", threshold=300, verbose=False)
    vert_mesh = drr_to_mesh(vert_drr, "surface_nets", threshold=300, verbose=False)
    # ct_offset = np.array(ct_mesh.center) - np.array(vert_mesh.center)
    # print(vert_mesh.points.shape)
    # print(ct_mesh.points.shape)
    # vert_mesh.points = vert_mesh.points + ct_offset
    # ct.plot()
    # return
    # Make a mesh from the camera and detector plane
    fiducials_2d = np.array([
        (115, -125.0, 260),  # MOF-r
        (50, -125.0, 260),  # MOF-l
        (48, -125.0, 280),  # IOF-r
        (118, -125.0, 280),  # IOF-l
        (50, -125.0, 230),  # SPS-r
        (120, -125.0, 230)  # SPS-l
    ], dtype="double")
    lines = []
    # print(camera.center)
    # for pt in fiducials_2d:
    #     line = pyvista.Line(pt, camera.center)
    #     lines.append(line)
    # lines = lines_to_mesh(camera, fiducials_2d)
    # print(principal_ray.shape)
    # Make the plot
    plotter = pyvista.Plotter()
    plotter.add_mesh(vert_mesh)
    plotter.open_gif('results/singleView_lines/{}_reg_process.gif'.format(case_name), fps=20)
    vert_gt_ap_pose = convert(gt_rotations.float(), gt_translations.float(), parameterization='euler_angles',
                              convention="ZYX")
    _, gt_ap_detector, _, _ = img_to_mesh(
        vert_drr, vert_gt_ap_pose
    )
    gt_la_rotations = gt_rotations + torch.tensor([[-torch.pi / 2, 0, 0]]).to(device)
    vert_gt_la_pose = convert(gt_la_rotations.float(), gt_translations.float(), parameterization='euler_angles',
                              convention="ZYX")
    _, gt_la_detector, _, _ = img_to_mesh(
        vert_drr, vert_gt_la_pose
    )
    ct_gt_la_rotations = ct_gt_rotations + torch.tensor([[-torch.pi / 2, 0, 0]]).to(device)
    # translation_list = [translations1, translations1]
    # rotation_list = [rotations1, rotations2]
    # for transl, rotate in zip(translation_list, rotation_list):
    for idx, row in poses_data.iterrows():
        rotate = (
            torch.tensor(row[["alpha", "beta", "gamma"]].values).unsqueeze(0)
            .to(device)
        )
        # la_rotations = rotate + torch.tensor([[-torch.pi / 2, 0, 0]]).to(device)
        # print(rotations)
        transl = (
            torch.tensor(row[["bx", "by", "bz"]].values).unsqueeze(0).to(device)
        )
        # print(rotate)
        # print(transl)
        pose = convert(rotate.float(), transl.float(), parameterization='euler_angles', convention="ZYX")
        camera, detector, texture, principal_ray = img_to_mesh(
            vert_drr, pose
        )
        add_other_mesh(plotter, ct_gt_rotations, ct_gt_translations, ct_drr, gt_ap_detector, isCT=True)
        # add_other_mesh(plotter, ct_gt_la_rotations, ct_gt_translations, ct_drr, gt_la_detector, isCT=True)

        # plotter.clear()
        plotter.add_mesh(vert_mesh)
        # add_other_mesh(plotter, la_rotations, transl, vert_drr)
        plotter.add_mesh(camera, show_edges=True, line_width=1.5)
        plotter.add_mesh(principal_ray, color="lime", line_width=3)
        plotter.add_mesh(detector, texture=texture)
        plotter.write_frame()
        plotter.clear()
        # print(detector.center)
        # for line in lines:
        #     plotter.add_mesh(line, color="lime")
        # Render the plot
        # Make a mesh from the camera and detector plane
    plotter.close()
    # plotter.add_bounding_box()
    # plotter.add_axes()
    # plotter.export_html("render.html")
    # # plotter.plot()
    # IFrame("render.html", height=500, width=749)


def add_other_mesh(pl, rotate, transl, drr, gt_det=None, isCT=False):
    pose = convert(rotate.float(), transl.float(), parameterization='euler_angles', convention="ZYX")
    camera, detector, texture, principal_ray = img_to_mesh(
        drr, pose
    )
    if isCT:

        det_offset = np.array(detector.center) - np.array(gt_det.center)
        if rotate[0][0] == 0:
            offset_arr = np.array([det_offset[0], det_offset[1], det_offset[2]])
        else:
            offset_arr = np.array([det_offset[0], det_offset[1], det_offset[2]])
        detector.points = detector.points - offset_arr
        camera.points = camera.points - offset_arr
        principal_ray.points = principal_ray.points - offset_arr
        pl.add_mesh(detector, texture=texture)
        pl.add_mesh(camera, show_edges=True, line_width=1.5)
        pl.add_mesh(principal_ray, color="lime", line_width=3)
    else:
        pl.add_mesh(camera, show_edges=True, line_width=1.5)
        pl.add_mesh(principal_ray, color="lime", line_width=3)
        pl.add_mesh(detector, texture=texture)
    return detector


def cal_error():
    error_data = pd.read_csv('results/exp2024.1.15/dai_pose3.csv')
    rot_pred = np.array(error_data.iloc[-1, :3])
    trans_pred = np.array(error_data.iloc[-1, 3:-1])
    rot_pred = np.expand_dims(rot_pred, axis=0)
    trans_pred = np.expand_dims(trans_pred, axis=0)
    rot_true = [1.5708, 0.0000, 3.1416]
    trans_true = [94.5000, 94.5000, 136.7500]
    # 75.5000, 75.5000, 127.2000
    # 125.0000, 125.0000, 132.2500
    rot_true = np.expand_dims(rot_true, axis=0)
    trans_true = np.expand_dims(trans_true, axis=0)
    # print(rot_pred)
    rot_mse = mean_absolute_error(rot_pred, rot_true, multioutput='raw_values')
    trans_mse = mean_absolute_error(trans_pred, trans_true, multioutput='raw_values')
    print((np.round((rot_mse * np.pi), 3)).tolist())
    print((np.round(trans_mse, 3)).tolist())


def proj_visual_3d():
    pose_path = 'results/dualView_line/dukemei_L3_pose.csv'
    poses_data = pd.read_csv(pose_path)
    vert_Dir = 'Data/gncc_data/tuodao/dukemei/dukemei_L3.nii.gz'
    ct_Dir = 'Data/gncc_data/tuodao/dukemei/dukemei.nii.gz'
    # Make a mesh from the CT volume
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
    offset_trans = np.array([-offsetx, offsety, offsetz])
    print(offset_trans)
    _, vert_drr, _, gt_rotations, gt_translations = get_drr(vert_Dir, offset_trans=offset_trans, tissue=True)
    _, ct_drr, _, ct_gt_rotations, ct_gt_translations = get_drr(ct_Dir, tissue=True)
    vert_mesh = drr_to_mesh(vert_drr, "surface_nets", threshold=300, verbose=False)
    ct_mesh = drr_to_mesh(ct_drr, "surface_nets", threshold=300, verbose=False)
    print(vert_mesh.center)
    print(ct_mesh.center)
    ct_offset = np.array(ct_mesh.center) - np.array(vert_mesh.center)
    # print(vert_mesh.points.shape)
    # print(ct_mesh.points.shape)
    ct_mesh.points = ct_mesh.points - ct_offset
    # gt_rotations = gt_rotations + torch.tensor([[0, torch.pi / 9, 0]]).to(device)
    # gt_translations = gt_translations + torch.tensor([[108, 113, 86]]).to(device)
    # ct.plot()
    # return
    # Make a mesh from the camera and detector plane
    rotations1 = torch.tensor([[torch.pi / 2, 0, torch.pi]], device=device)
    rotations2 = torch.tensor([[0, 0, torch.pi]], device=device)
    fiducials_2d = np.array([
        (115, -125.0, 260),  # MOF-r
        (50, -125.0, 260),  # MOF-l
        (48, -125.0, 280),  # IOF-r
        (118, -125.0, 280),  # IOF-l
        (50, -125.0, 230),  # SPS-r
        (120, -125.0, 230)  # SPS-l
    ], dtype="double")
    lines = []
    # print(camera.center)
    # for pt in fiducials_2d:
    #     line = pyvista.Line(pt, camera.center)
    #     lines.append(line)
    # lines = lines_to_mesh(camera, fiducials_2d)
    # print(principal_ray.shape)
    # Make the plot
    pose = convert(gt_rotations.float(), gt_translations.float(), parameterization='euler_angles', convention="ZYX")
    camera, detector, texture, principal_ray = img_to_mesh(
        vert_drr, pose
    )
    plotter = pyvista.Plotter()
    plotter.add_mesh(vert_mesh)
    # ct_det = add_gt_xray_mesh(plotter, ct_gt_rotations, ct_gt_translations, ct_drr, isCT=True)
    ct_pose = convert(ct_gt_rotations.float(), ct_gt_translations.float(), parameterization='euler_angles',
                      convention="ZYX")
    _, ct_det, ct_texture, _ = img_to_mesh(
        ct_drr, ct_pose
    )
    print(detector.center)
    det_offset = np.array(ct_det.center) - np.array(detector.center)
    ct_det.points = ct_det.points - np.array([det_offset[0], det_offset[1], det_offset[2]])
    plotter.add_mesh(ct_det, texture=ct_texture)
    print(detector.center)
    plotter.add_mesh(camera, show_edges=True, line_width=1.5)
    plotter.add_mesh(principal_ray, color="lime", line_width=3)
    plotter.add_mesh(detector, texture=texture)
    plotter.add_bounding_box()
    plotter.add_axes()
    plotter.export_html("render.html")
    # plotter.plot()
    IFrame("render.html", height=500, width=749)


def draw_box_plot():
    data = pd.read_excel('results/exp.xlsx', sheet_name='Sheet5')
    # data = sns.load_dataset()
    # view the dataset
    print(data.head(5))
    # 字体加载
    font_path = "F:/迅雷下载/timessimsun.ttf"
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    # print(prop.get_name())  # 显示当前使用字体的名称
    # 字体设置
    rcParams['font.family'] = 'sans-serif'  # 使用字体中的无衬线体
    rcParams['font.sans-serif'] = prop.get_name()  # 根据名称设置字体
    rcParams['font.size'] = 20  # 设置字体大小
    # rcParams['axes.unicode_minus'] = False  # 使坐标轴刻度标签正常显示正负号
    label_list = ["x轴方向", "y轴方向", "z轴方向"]
    # plt.figure(figsize=(5, 6))
    b = sns.boxplot(x='方向', y='误差', order=label_list, hue='方法', data=data, palette='Set3')
    font_options = {"size": 20}
    # plt.rcParams['font.sans-serif'] = 'SimSun'
    # b.axes.set_title("Title", fontsize=15)
    b.set_xlabel("")
    b.set_ylabel("旋转误差", fontsize=20)
    # b.tick_params(labelsize=15)
    # b.legend().set_visible(False)
    plt.legend(prop=font_options)
    plt.savefig('results/dual_rot_box.jpg', dpi=300)
    plt.show()


def cal_ssim():
    mov_path = 'E:/pythonWorkplace/xreg/data/tuodao/dual_view/new_tuodao_1122/L3_2VIEW_DRR/drr_remap_000_000.nii'
    fixed_path = 'E:/pythonWorkplace/xreg/data/tuodao/L3/AP/L3_resized.nii.gz'
    mask_path = ''
    mov_img = sitk.ReadImage(mov_path)
    mov_arr = sitk.GetArrayFromImage(mov_img)
    fixed_img = sitk.ReadImage(fixed_path)
    resized_fix = resample_img(fixed_img, new_width=256)
    fixed_arr = sitk.GetArrayFromImage(resized_fix)
    mask = sitk.ReadImage(mask_path)
    resized_mask = resample_img(mask, new_width=256)
    mask_arr = sitk.GetArrayFromImage(resized_mask)
    fixed_arr = (fixed_arr - np.min(fixed_arr)) / (np.max(fixed_arr) - np.min(fixed_arr))
    # plt.imshow(mov_arr)
    # plt.show()
    print(ssim(mov_arr, fixed_arr * mask_arr))


def ssim(imageA, imageB):
    """
    考虑到人眼对于图像细节的敏感程度，
    比MSE更能反映图像的相似度。
    SSIM计算公式较为复杂，包含对于亮度、对比度、结构等因素的综合考虑。
    @param imageA: 图片1
    @param imageB: 图片2
    @return: SSIM计算公式较为复杂包含对于亮度、对比度、结构等因素的综合考虑。
    """
    # ssim_val = cv2.SSIM(imageA, imageB)
    # ssim_val = structural_similarity(imageA, imageB, data_range=255, multichannel=True)
    ssim_val = structural_similarity(imageA, imageB, data_range=1)
    return ssim_val


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    animate_reg_process()
    # reg_process_visual_3d()
    # cal_error()
    # draw_box_plot()
    # proj_visual_3d()
    # cal_ssim()
