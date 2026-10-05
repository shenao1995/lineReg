"""Verify LineReg geometry and render a real L2 DRR without changing Data/."""

from pathlib import Path

import cv2
import numpy as np
import torch

from nanodrr_adapter import LineRegDRR, load_subject, pose_matrix, compose_vertebra_offset


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    subject = load_subject("Data/case1/case1_L2.nii.gz").to(device)
    np.testing.assert_allclose(subject.isocenter.cpu().numpy(), [0.0, 0.0, 0.0], atol=1e-3)
    drr = LineRegDRR(subject).to(device)
    rotation = torch.zeros((1, 3), dtype=torch.float32, device=device)
    translation = torch.tensor([[0.0, 900.0, 0.0]], device=device)
    pose = pose_matrix(rotation, translation)

    # DiffDRR AP + reverse_x_axis=True: +RAS x and +RAS z both move toward
    # smaller image coordinates.  A centered point lands at (W/2, H/2).
    points = torch.tensor(
        [[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 10.0]]],
        device=device,
    )
    pixels = drr.perspective_projection(pose, points)[0].cpu().numpy()
    expected_shift = 1100.0 * 10.0 / (900.0 * 0.8)
    np.testing.assert_allclose(pixels[0], [128.0, 128.0], atol=1e-3)
    np.testing.assert_allclose(pixels[1], [128.0 - expected_shift, 128.0], atol=1e-3)
    np.testing.assert_allclose(pixels[2], [128.0, 128.0 - expected_shift], atol=1e-3)

    # The first detector ray must be the old DiffDRR first ray in centered RAS.
    first_target = drr.projector.tgt[:, :1]
    first_world = torch.einsum(
        "bij,bnj->bni",
        drr.camera_to_world(pose)[:, :3, :3],
        first_target,
    ) + drr.camera_to_world(pose)[:, None, :3, 3]
    np.testing.assert_allclose(
        first_world[0, 0].cpu().numpy(),
        [127.5 * 0.8, -200.0, 127.5 * 0.8],
        atol=1e-3,
    )

    # DiffDRR's ZXY convention is Rz @ Rx @ Ry, and its compose method
    # multiplies the vertebra offset on the left.
    angles = torch.tensor([[0.12, -0.07, 0.05]], device=device)
    actual_pose = pose_matrix(angles, translation)
    a, b, c = angles[0].unbind()
    zero, one = torch.zeros_like(a), torch.ones_like(a)
    rz = torch.stack((torch.cos(a), -torch.sin(a), zero,
                      torch.sin(a), torch.cos(a), zero,
                      zero, zero, one)).reshape(3, 3)
    rx = torch.stack((one, zero, zero,
                      zero, torch.cos(b), -torch.sin(b),
                      zero, torch.sin(b), torch.cos(b))).reshape(3, 3)
    ry = torch.stack((torch.cos(c), zero, torch.sin(c),
                      zero, one, zero,
                      -torch.sin(c), zero, torch.cos(c))).reshape(3, 3)
    expected_rotation = rz @ rx @ ry
    torch.testing.assert_close(actual_pose[0, :3, :3], expected_rotation, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_pose[0, :3, 3], expected_rotation @ translation[0])

    old_ap = torch.tensor(
        [[1., 0., 0., 0.], [0., 0., -1., 0.],
         [0., 1., 0., 0.], [0., 0., 0., 1.]],
        device=device,
    ).unsqueeze(0)
    off_axis = torch.tensor(
        [[[12., -8., 5.], [-15., 2., 11.], [3., 20., -7.]]],
        device=device,
    )
    homogeneous = torch.cat((off_axis, torch.ones_like(off_axis[..., :1])), dim=-1)
    old_camera = torch.einsum(
        "bij,bnj->bni", torch.linalg.inv(actual_pose @ old_ap), homogeneous,
    )[..., :3]
    old_pixels = torch.stack((
        128.0 - 1100.0 * old_camera[..., 0] / (0.8 * old_camera[..., 2]),
        128.0 - 1100.0 * old_camera[..., 1] / (0.8 * old_camera[..., 2]),
    ), dim=-1)
    torch.testing.assert_close(
        drr.perspective_projection(actual_pose, off_axis), old_pixels,
        atol=1e-4, rtol=1e-5,
    )

    offset = torch.eye(4, device=device).unsqueeze(0)
    offset[0, :3, 3] = torch.tensor([11.0, -4.0, 7.0], device=device)
    torch.testing.assert_close(compose_vertebra_offset(actual_pose, offset), offset @ actual_pose)

    with torch.no_grad():
        image = drr(pose).squeeze().cpu().numpy()
    assert image.shape == (256, 256)
    assert np.isfinite(image).all()
    assert image.max() > image.min(), "Rendered DRR is blank"
    active = image > image.max() * 0.05
    assert active.sum() > 100, "Rendered DRR has too few active pixels"

    output = Path("results/case1_L2/nanodrr_gt.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized = (image - image.min()) / (image.max() - image.min())
    assert cv2.imwrite(str(output), (normalized * 255).astype(np.uint8))
    print(f"CUDA: {torch.cuda.is_available()}")
    print(f"Projection: min={image.min():.6f}, max={image.max():.6f}, active={active.sum()}")
    print(f"Geometry pixels: {pixels.tolist()}")
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
