"""LineReg camera convention on top of nanodrr.

LineReg's original DiffDRR setup used centered RAS volumes, ZXY Euler angles
in radians, AP orientation, and reverse_x_axis=True.  Keep those choices
explicit here so saved pose parameters and fiducial coordinates remain valid.
"""

import torch
from nanodrr.data import Subject
from nanodrr.drr import DRR as NanoDRR
from nanodrr.geometry import convert
from torchio import ScalarImage


# DiffDRR's AP reorientation and horizontal detector reversal together give
# exactly this camera-to-centered-RAS basis for nanodrr's pixel convention.
AP_CAMERA_TO_RAS = torch.tensor(
    [[-1., 0., 0., 0.],
     [0., 0., -1., 0.],
     [0., -1., 0., 0.],
     [0., 0., 0., 1.]],
    dtype=torch.float32,
)


def _diffdrr_density(hu, bone_attenuation_multiplier):
    """Preserve DiffDRR's historical HU contrast while changing renderer."""
    hu = hu.float()
    soft = (hu > -800) & (hu <= 350)
    if not torch.any(soft):
        raise ValueError("CT contains no soft-tissue voxels in (-800, 350] HU")
    floor = hu[soft].min()
    density = torch.where(hu <= -800, floor, hu)
    density = torch.where(hu > 350, hu * bone_attenuation_multiplier, density)
    density = density - density.min()
    peak = density.max()
    if peak > 0:
        density = density / peak
    return density


def load_subject(path, bone_attenuation_multiplier=1.0, fiducials=None, preserve_hu=False):
    """Read a CT in the centered RAS frame formerly returned by diffdrr.read."""
    image = ScalarImage(path)
    center = torch.as_tensor(image.get_center(), dtype=torch.float32)
    affine = image.affine.copy()
    affine[:3, 3] -= center.cpu().numpy()
    density = image.data.float() if preserve_hu else _diffdrr_density(image.data, bone_attenuation_multiplier)
    centered_image = ScalarImage(tensor=density, affine=affine)
    subject = Subject.from_images(centered_image, convert_to_mu=False)
    if fiducials is not None:
        # extract_ap_body_side_fiducials_from_target_volume returns RAS mm.
        subject.register_buffer("fiducials", fiducials.float() - center)
    return subject


def pose_matrix(rotation, translation):
    """DiffDRR-compatible ZXY camera pose; angles are radians, distances mm."""
    return convert(rotation, translation, "euler", convention="ZXY", degrees=False)


def compose_vertebra_offset(pose, offset):
    """Match DiffDRR's pose.compose(vert_mat) == vert_mat @ pose."""
    return offset @ pose


class LineRegDRR(torch.nn.Module):
    def __init__(self, subject, sdd=1100.0, height=256, delx=0.8, n_samples=500):
        super().__init__()
        self.subject = subject
        self.projector = NanoDRR.from_carm_intrinsics(
            sdd=sdd, delx=delx, dely=delx, x0=0.0, y0=0.0,
            height=height, width=height, dtype=torch.float32,
        )
        self.register_buffer("ap_camera_to_ras", AP_CAMERA_TO_RAS.clone())
        self.n_samples = n_samples

    def camera_to_world(self, pose):
        return pose @ self.ap_camera_to_ras

    def forward(self, pose):
        return self.projector(
            self.subject, self.camera_to_world(pose),
            n_samples=self.n_samples, backend="triton",
        )

    def perspective_projection(self, pose, points):
        """Project centered RAS points to continuous (column, row) pixels."""
        rt = self.camera_to_world(pose)
        points_h = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
        camera = torch.einsum("bij,bnj->bni", torch.linalg.inv(rt), points_h)[..., :3]
        uvw = torch.einsum("bij,bnj->bni", torch.linalg.inv(self.projector.k_inv), camera)
        return uvw[..., :2] / uvw[..., 2:3]
