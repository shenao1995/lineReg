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
from tools import read_xml, get_ext_pose
import SimpleITK as sitk


def show_cam_Ext_mat(img_path, ap_xml_path, la_xml_path):
    ap_Xdir, ap_Ydir, X_Spacing, SDD, Xray_H, ap_Wld_Offset = read_xml(ap_xml_path)
    la_Xdir, la_Ydir, _, _, _, la_Wld_Offset = read_xml(la_xml_path)
    # print(Xray_H)
    print(SDD)
    HEIGHT = 256
    DELX = float(Xray_H) / HEIGHT * X_Spacing
    print(DELX)
    subject = read(img_path, bone_attenuation_multiplier=10.5)
    print(subject.shape)
    drr = DRR(
        subject,  # An object storing the CT volume, origin, and voxel spacing
        sdd=SDD,  # Source-to-detector distance (i.e., focal length)
        height=HEIGHT,  # Image height (if width is not provided, the generated DRR is square)
        delx=DELX,  # Pixel spacing (in mm)
    ).to(device)
    ini_pose = torch.zeros(1, 6).to(device)
    ap_extrinsic_update = get_ext_pose(ap_Xdir, ap_Ydir, float(ap_Wld_Offset[1]), ini_pose)
    la_extrinsic_update = get_ext_pose(la_Xdir, la_Ydir, float(la_Wld_Offset[0]), ini_pose)
    # print(extrinsic_update.matrix)
    ground_truth = drr(ap_extrinsic_update)
    out_img = sitk.GetImageFromArray(ground_truth.squeeze().detach().cpu().numpy())
    sitk.WriteImage(out_img, 'Data/tuodao/dukemei/drr.nii.gz')
    # camera1, detector1, texture1, principal_ray1 = img_to_mesh(drr, ap_extrinsic_update)
    # camera2, detector2, texture2, principal_ray2 = img_to_mesh(drr, la_extrinsic_update)
    # volume_mesh = drr_to_mesh(subject, "surface_nets", threshold=225, verbose=True)
    # Make the plot
    # plotter = pyvista.Plotter()
    # plotter.add_mesh(volume_mesh)
    # plotter.add_mesh(camera1, show_edges=True, line_width=1.5)
    # plotter.add_mesh(principal_ray1, color="lime", line_width=3)
    # plotter.add_mesh(detector1, texture=texture1)
    # plotter.add_mesh(camera2, show_edges=True, line_width=1.5)
    # plotter.add_mesh(principal_ray2, color="lime", line_width=3)
    # plotter.add_mesh(detector2, texture=texture2)
    # Render the plot
    # plotter.add_axes()
    # plotter.add_bounding_box()
    # plotter.export_html("render_wld1.html")
    # IFrame("render_wld1.html", height=500, width=749)
    # ground_truth = torch.permute(ground_truth, (0, 1, 3, 2))
    plot_drr(ground_truth)
    # plt.imshow(ground_truth.squeeze().detach().cpu().numpy(), cmap='gray')
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    show_cam_Ext_mat("Data/tuodao/dukemei/dukemei.nii.gz", "Data/tuodao/dukemei/X/View/180/calib_view.xml",
                     "Data/tuodao/dukemei/X/View/1/calib_view.xml")
