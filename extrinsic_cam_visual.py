from diffdrr.drr import DRR
from diffdrr.pose import convert, RigidTransform
from diffdrr.data import load_example_ct, read
import torch
import numpy as np
import nibabel as nib
from diffdrr.visualization import plot_drr
import matplotlib.pyplot as plt
import pyvista
from diffdrr.visualization import drr_to_mesh, img_to_mesh
from IPython.display import IFrame
from tools import read_xml, get_ext_pose, update_pose, tuodao_to_diffdrr, extract_img_overlay, dual_view_joint, \
    read_bg_img
import SimpleITK as sitk
import cv2
from monai.transforms import LoadImage


def show_cam_Ext_mat(img_path, ap_xml_path, la_xml_path):
    itk_img = sitk.ReadImage(img_path)
    spacing = itk_img.GetSpacing()
    size = itk_img.GetSize()
    origin = np.array(size) * np.array(spacing)
    print(origin / 2)
    ctOffset = origin / 2
    caseName = 'peizongping'
    vert_num = 'L3'
    vert_seg_path = 'Data/tuodao/{}/{}_seg.nii.gz'.format(caseName, vert_num)
    ap_Xdir, ap_Ydir, X_Spacing, SDD, Xray_H, ap_wld_T = read_xml(ap_xml_path)
    la_Xdir, la_Ydir, _, _, _, la_wld_T = read_xml(la_xml_path)
    # print(ap_wld_T)
    # print(la_wld_T)
    # print(SDD)
    HEIGHT = 256
    DELX = float(Xray_H) / HEIGHT * X_Spacing
    # print(DELX)
    subject = read(img_path, vert_seg_path, labels=[1], bone_attenuation_multiplier=10.5)
    ini_pose = torch.zeros(1, 6).to(device)
    # ini_pose[:, 3], ini_pose[:, 4], ini_pose[:, 5] = 156.49996948, 156.49996948, 115.312
    reader = LoadImage(ensure_channel_first=True, image_only=False)
    bg_path = 'Data/tuodao/{}/X/{}_resized_x_ap.nii.gz'.format(caseName, caseName)
    la_bg_path = 'Data/tuodao/{}/X/{}_resized_x_la.nii.gz'.format(caseName, caseName)
    rgb_ap_gt = read_bg_img(bg_path, reader)
    rgb_la_gt = read_bg_img(la_bg_path, reader)
    ap_extrinsic_update = get_ext_pose(ap_Xdir, ap_Ydir, ap_wld_T, ini_pose, view='ap')
    la_extrinsic_update = get_ext_pose(la_Xdir, la_Ydir, la_wld_T, ini_pose, view='la')
    gt_mat = torch.tensor(
        [
                    [1.0,0.0,0.0,-7.606590270996094],
                    [0.0,1.0,0.0,-11.101516723632812],
                    [0.0,0.0,1.0,-102.17507934570312],
                    [0.0,0.0,0.0,1.0],
        ]
    ).to(device, dtype=torch.float32)


    ap_pose = update_pose(ap_extrinsic_update, gt_mat, ctOffset)
    la_pose = update_pose(la_extrinsic_update, gt_mat, ctOffset)
    # print(subject.shape)
    drr = DRR(
        subject,  # An object storing the CT volume, origin, and voxel spacing
        sdd=SDD,  # Source-to-detector distance (i.e., focal length)
        height=HEIGHT,  # Image height (if width is not provided, the generated DRR is square)
        delx=DELX,  # Pixel spacing (in mm)
        reverse_x_axis=True
    ).to(device, dtype=torch.float32)
    # print(extrinsic_update.matrix)
    # ap_pose = tuodao_to_diffdrr(drr, ap_pose)
    # la_pose = tuodao_to_diffdrr(drr, la_pose)
    ap_gt = drr(ap_pose)
    la_gt = drr(la_pose)
    # out_img = sitk.GetImageFromArray(ground_truth.squeeze().detach().cpu().numpy())
    # sitk.WriteImage(out_img, 'Data/tuodao/dukemei/drr.nii.gz')
    camera1, detector1, texture1, principal_ray1 = img_to_mesh(drr, ap_pose)
    camera2, detector2, texture2, principal_ray2 = img_to_mesh(drr, la_pose)
    volume_mesh = drr_to_mesh(subject, "surface_nets", threshold=225, verbose=True)
    # Make the plot
    plotter = pyvista.Plotter()
    plotter.add_mesh(volume_mesh)
    plotter.add_mesh(camera1, show_edges=True, line_width=1.5)
    plotter.add_mesh(principal_ray1, color="lime", line_width=3)
    plotter.add_mesh(detector1, texture=texture1)
    plotter.add_mesh(camera2, show_edges=True, line_width=1.5)
    plotter.add_mesh(principal_ray2, color="lime", line_width=3)
    plotter.add_mesh(detector2, texture=texture2)
    # Render the plot
    plotter.add_axes()
    plotter.add_bounding_box()
    plotter.show_grid()
    plotter.export_html("render_wld1.html")
    IFrame("render_wld1.html", height=500, width=749)
    ap_gt = ap_gt.squeeze().cpu().numpy()
    la_gt = la_gt.squeeze().cpu().numpy()
    ap_contour = extract_img_overlay(ap_gt)
    la_contour = extract_img_overlay(la_gt)
    rgb_mov = dual_view_joint(ap_contour, la_contour)
    rgb_gt = dual_view_joint(rgb_ap_gt, rgb_la_gt)
    result = cv2.addWeighted(rgb_gt, 1, rgb_mov, 0.5, 0)
    plt.imshow(result)
    plt.show()
    # plt.subplot(1, 2, 1)
    # plt.imshow(ap_gt.squeeze().detach().cpu().numpy(), cmap='gray')
    # plt.subplot(1, 2, 2)
    # plt.imshow(la_gt.squeeze().detach().cpu().numpy(), cmap='gray')
    # plt.show()
    # plt.close()


if __name__ == '__main__':
    name = 'peizongping'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    show_cam_Ext_mat("Data/tuodao/{}/{}.nii.gz".format(name, name), "Data/tuodao/{}/X/View/180/calib_view.xml".format(name),
                     "Data/tuodao/{}/X/View/1/calib_view.xml".format(name))
