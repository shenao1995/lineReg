"""Real radiograph 2D/3D registration with one shared CT pose and true GNCC.

All transforms act on homogeneous COLUMN vectors. Distances are mm. The saved
ct_to_world matrix maps the declared (legacy local RAS or physical RAS) CT frame
to the XML acquisition world. No dependency on lineReg_temp is required.
"""

import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from cmaes import CMA
from nanodrr.data import Subject
from nanodrr.drr import DRR
from torchio import ScalarImage

from gncc import GradientNormalizedCrossCorrelation2d
from nanodrr_adapter import AP_CAMERA_TO_RAS, _diffdrr_density


def translation(vector):
    mat = np.eye(4)
    mat[:3, 3] = vector
    return mat


def validate_rigid(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-4)):
        raise ValueError("ct_to_world must be a finite rigid 4x4 matrix (det R = +1)")
    return matrix


def pose_from_parameters(params, anchor):
    """XYZ angles in DEGREES; independent world translation, rotate about anchor."""
    x, y, z = np.deg2rad(params[:3])
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    out = np.eye(4)
    out[:3, :3] = rx @ ry @ rz
    out[:3, 3] = np.asarray(anchor) + params[3:] - out[:3, :3] @ anchor
    return out


def read_calibration(path, convention):
    """Read named tags, not fragile XML element positions from tools2.read_xml."""
    root = ET.parse(path).getroot()

    def values(tag):
        element = root.find(tag)
        if element is None or not element.text:
            raise ValueError(f"Missing XML tag {tag}")
        return np.fromstring(element.text, sep=" ")

    source, center = values("XrayCenterWld"), values("ImgCenterWld")
    x, y = values("ImgXdir"), values("ImgYdir")
    width, height = values("ImgResolution").astype(int)
    spacing = values("PixelSpacing")
    spacing = np.repeat(spacing, 2) if spacing.size == 1 else spacing
    if source.size != 3 or center.size != 3 or x.size != 3 or y.size != 3:
        raise ValueError("Calibration vectors must have three components")
    x, y = x / np.linalg.norm(x), y / np.linalg.norm(y)
    if abs(x @ y) > 1e-4 or spacing.size != 2 or (spacing <= 0).any():
        raise ValueError("Invalid detector axes or spacing")
    camera = np.eye(4)
    camera[:3, 3] = source
    if convention == "legacy_temp":
        # Exact get_ext_pose row order, then the old DiffDRR AP detector basis.
        camera[:3, :3] = np.stack([x, np.cross(x, y), y])
        camera = camera @ AP_CAMERA_TO_RAS.numpy()
        sdd = np.linalg.norm(center - source)
        principal = np.array([width / 2, height / 2])
    else:
        normal = np.cross(x, y)
        if normal @ (center - source) < 0:
            normal = -normal
        sdd = normal @ (center - source)
        camera[:3, :3] = np.column_stack([x, y, normal])
        principal = values("ImgCenter")
    if not np.isfinite(sdd) or sdd <= 0 or min(width, height) < 3:
        raise ValueError("Invalid detector dimensions or source-to-detector distance")
    k = np.array([[sdd / spacing[0], 0, principal[0]],
                  [0, sdd / spacing[1], principal[1]], [0, 0, 1.]])
    return camera, k, float(sdd), (int(width), int(height))


def prepare_volume(config, base, device):
    """Crop without discarding affine origin/directions; explicitly track center."""
    image = ScalarImage(str(resolve(base, config["ct"])))
    hu, affine = image.data.float(), image.affine.copy()
    full_shape = np.array(hu.shape[1:])
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    start, crop_shape = np.zeros(3, dtype=int), full_shape.copy()
    density = _diffdrr_density(hu, config.get("bone_attenuation_multiplier", 10.5))
    if config.get("segmentation"):
        seg = ScalarImage(str(resolve(base, config["segmentation"])))
        seg_space = config.get("segmentation_space", "physical")
        if seg_space not in ("physical", "voxel"):
            raise ValueError("segmentation_space must be physical or voxel")
        if seg.shape != image.shape:
            raise ValueError("CT and segmentation must share voxel dimensions")
        if seg_space == "physical" and not np.allclose(seg.affine, affine, atol=1e-4):
            raise ValueError("CT/seg affines differ; explicitly use segmentation_space=voxel ONLY if index-aligned")
        if seg_space == "voxel" and not np.allclose(seg.affine, affine, atol=1e-4):
            print("Using explicitly declared voxel-aligned segmentation; ignoring its affine.")
        mask = seg.data[0] == int(config.get("label", 1))
        indices = torch.nonzero(mask).cpu().numpy()
        if not len(indices):
            raise ValueError("Requested segmentation label is empty")
        start, stop = indices.min(0), indices.max(0) + 1
        crop_shape = stop - start
        density = density * mask.unsqueeze(0)
        density = density[(slice(None), *(slice(a, b) for a, b in zip(start, stop)))]
        affine[:3, 3] += affine[:3, :3] @ start
    physical_center = affine[:3, :3] @ ((crop_shape - 1) / 2) + affine[:3, 3]
    affine[:3, 3] -= physical_center
    convention = config.get("coordinate_convention", "legacy_temp")
    if convention == "legacy_temp":
        # tools2.crop_ct_vert assumes CT direction=identity in LPS. Refuse
        # silently incorrect legacy offsets for oblique/flipped CT acquisitions.
        if not np.allclose(image.affine[:3, :3] / spacing,
                           np.diag([-1., -1., 1.]), atol=1e-4):
            raise ValueError("legacy_temp needs identity LPS CT direction; use ras for other affines")
        offset = (full_shape / 2 - (start + crop_shape / 2)) * spacing
        delta = offset * np.array([-1., -1., 1.])
        # Matches vert_mat @ inv(temp) in tools2.update_pose_v2. Historical
        # size/2 (not (size-1)/2) is deliberate only in compatibility mode.
        anchor = full_shape * spacing / 2 - delta
    else:
        anchor = physical_center
    centered_to_ct = translation(anchor)
    # Optional anatomical anchor, independent of the rendering crop's center.
    point = config.get("landmark")
    if config.get("landmarks_json"):
        data = json.loads(resolve(base, config["landmarks_json"]).read_text(encoding="utf-8-sig"))
        point = data["prediction"][0][config["vertebra"]]
    if point is not None:
        point = np.asarray(point, dtype=float).reshape(-1)
        space = config.get("landmark_space")
        if point.size != 3 or not np.isfinite(point).all():
            raise ValueError("Landmark must be one finite 3D point")
        if space == "legacy_lps_mm" and convention == "legacy_temp":
            # Exactly tools2.readJsonGet3dPoints, NOT physical LPS coordinates.
            anchor = point * [-1, -1, 1] + full_shape * spacing * [1, 1, 0]
        elif space == "physical_ras" and convention == "ras":
            anchor = point
        elif space == "voxel" and convention == "ras":
            anchor = image.affine[:3, :3] @ point + image.affine[:3, 3]
        else:
            raise ValueError("Declare landmark_space: legacy_lps_mm for legacy_temp; physical_ras or voxel for ras")
    subject = Subject.from_images(ScalarImage(tensor=density, affine=affine),
                                  convert_to_mu=False).to(device)
    return subject, centered_to_ct, anchor


def resolve(base, value):
    path = Path(value)
    return path if path.is_absolute() else base / path


def read_bbox(spec, config, base, native_size):
    if "bbox" in spec:
        box = np.array(spec["bbox"], dtype=float)
    elif spec.get("bbox_json"):
        data = json.loads(resolve(base, spec["bbox_json"]).read_text(encoding="utf-8-sig"))
        matches = [item["bbox"] for item in data
                   if item["category_name"] == config["vertebra"]]
        if len(matches) != 1:
            raise ValueError("bbox_json must contain exactly one matching vertebra")
        box = np.array(matches[0], dtype=float)
    else:
        return None
    if box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= 0).any():
        raise ValueError("bbox must be finite [x, y, width, height]")
    reference = np.array(spec.get("bbox_size", native_size), dtype=float)
    scale = np.array(native_size) / reference
    return box * np.tile(scale, 2)


def prepare_view(spec, config, base, device, metric):
    convention = config.get("coordinate_convention", "legacy_temp")
    camera, k, sdd, native_size = read_calibration(resolve(base, spec["calibration"]), convention)
    image = cv2.imread(str(resolve(base, spec["image"])), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read radiograph: {spec['image']}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[::-1] != native_size:
        raise ValueError("Radiograph size does not match XML ImgResolution")
    image = image.astype(np.float32)
    if not np.isfinite(image).all() or np.ptp(image) <= 0:
        raise ValueError("Radiograph must contain finite nonconstant intensities")
    box = read_bbox(spec, config, base, native_size)
    # Homogeneous pixel coordinates use boundary origin: first pixel center=.5.
    h = np.eye(3)
    if spec.get("rotate_180", True):
        image = np.rot90(image, 2).copy()
        h = np.array([[-1., 0, native_size[0]], [0, -1., native_size[1]], [0, 0, 1.]])
        if box is not None and spec.get("bbox_frame", "processed") == "raw":
            box[:2] = np.array(native_size) - box[:2] - box[2:]
    if spec.get("bbox_frame", "processed") not in ("raw", "processed"):
        raise ValueError("bbox_frame must be raw or processed")
    if convention == "ras":
        k = h @ k  # Reindex rays WITH the image, including principal point.
    # Legacy intentionally rotates only measured image, matching the reference.
    if spec.get("invert", True):
        image = image.max() - image
    image = (image - image.min()) / np.ptp(image)
    if spec.get("clahe", True):
        image = cv2.createCLAHE(clipLimit=2, tileGridSize=(10, 10)).apply(
            np.round(image * 65535).astype(np.uint16)).astype(np.float32)
        image = (image - image.min()) / max(float(np.ptp(image)), 1e-8)
    width, height = config.get("image_size", [256, 256])
    width, height = int(width), int(height)
    if min(width, height) < 3:
        raise ValueError("image_size must be [width, height], both >= 3")
    resize = np.diag([width / native_size[0], height / native_size[1], 1.])
    k = resize @ k
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    mask = np.ones((height, width), dtype=np.float32)
    center = np.array([width / 2, height / 2])
    if box is not None:
        box *= np.tile(np.diag(resize)[:2], 2)
        center = box[:2] + box[2:] / 2
        margin = np.array(spec.get("bbox_margin", [5, 5]), dtype=float)
        start = np.maximum(np.floor(box[:2] - margin).astype(int), 1)
        stop = np.minimum(np.ceil(box[:2] + box[2:] + margin).astype(int), [width - 1, height - 1])
        if (stop - start < 3).any():
            raise ValueError("BBox has insufficient overlap with image")
        mask[:] = 0
        mask[start[1]:stop[1], start[0]:stop[0]] = 1
    else:
        mask[[0, -1], :] = 0
        mask[:, [0, -1]] = 0
    target = torch.tensor(image, device=device)[None, None]
    projector = DRR(torch.tensor(np.linalg.inv(k), dtype=torch.float32, device=device)[None],
                    torch.tensor([sdd], dtype=torch.float32, device=device), height, width).to(device)
    direction = camera[:3, :3] @ np.linalg.inv(k) @ np.r_[center, 1.]
    direction /= np.linalg.norm(direction)
    return dict(name=spec["name"], camera=camera, k=k, projector=projector,
                target=target, target_gradients=metric.gradients(target),
                mask=torch.tensor(mask, device=device)[None, None],
                ray=(camera[:3, 3], direction), weight=float(spec.get("weight", 1)))


def initialize(config, views, anchor):
    if config.get("initial_ct_to_world") is not None:
        return validate_rigid(config["initial_ct_to_world"])
    if len(views) == 1:
        distance = config.get("source_to_target_mm")
        if distance is None or not np.isfinite(distance) or distance <= 0:
            raise ValueError("Single view needs initial_ct_to_world or positive source_to_target_mm")
        source, direction = views[0]["ray"]
        target = source + distance * direction
    else:
        (a, da), (b, db) = (v["ray"] for v in views)
        if np.linalg.norm(np.cross(da, db)) < 0.05:
            raise ValueError("Two initialization rays are nearly parallel; supply initial_ct_to_world")
        t, u = np.linalg.lstsq(np.column_stack([da, -db]), b - a, rcond=None)[0]
        if min(t, u) <= 0:
            raise ValueError("Triangulated target lies behind source; check coordinates/bboxes")
        target = (a + t * da + b + u * db) / 2
        print(f"Initialization ray separation: {np.linalg.norm(a+t*da-b-u*db):.3f} mm")
    return translation(target - anchor)


def save_image(path, tensor):
    image = tensor.detach().cpu().numpy().squeeze()
    image = (image - image.min()) / max(float(np.ptp(image)), 1e-8)
    if not cv2.imwrite(str(path), np.round(image * 255).astype(np.uint8)):
        raise OSError(f"Cannot save {path}")


def run_registration(config, base, mode=None, iterations=None, output=None):
    mode = mode or config.get("mode", "dual")
    convention = config.get("coordinate_convention", "legacy_temp")
    if mode not in ("single", "dual") or convention not in ("legacy_temp", "ras"):
        raise ValueError("mode=single|dual; coordinate_convention=legacy_temp|ras")
    specs = config["views"]
    if mode == "single":
        requested = config.get("single_view", specs[0]["name"])
        specs = [v for v in specs if v["name"] == requested]
    if len(specs) != (1 if mode == "single" else 2):
        raise ValueError("Single mode needs one selected view; dual mode needs exactly two")
    names = [v["name"] for v in specs]
    if len(set(names)) != len(names) or any(Path(n).name != n or n in ("", ".", "..") for n in names):
        raise ValueError("View names must be unique simple filenames")
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    backend = config.get("backend", "triton" if device.type == "cuda" else "torch")
    samples = int(config.get("n_samples", 500))
    if backend not in ("torch", "triton", "auto") or samples < 2:
        raise ValueError("Invalid backend or n_samples")
    metric = GradientNormalizedCrossCorrelation2d().to(device)
    subject, centered_to_ct, anchor = prepare_volume(config, base, device)
    views = [prepare_view(v, config, base, device, metric) for v in specs]
    if any(not np.isfinite(v["weight"]) or v["weight"] <= 0 for v in views):
        raise ValueError("View weights must be positive")
    initial = initialize(config, views, anchor)
    # Delta rotations around the initially placed target, not a remote world origin.
    world_anchor = (initial @ np.r_[anchor, 1.])[:3]
    folder = resolve(base, output or config.get("output", "results/real_xray"))
    folder.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        def evaluate(params, keep=False):
            ct_to_world = pose_from_parameters(params, world_anchor) @ initial
            world_to_centered = np.linalg.inv(ct_to_world @ centered_to_ct)
            scores, images = [], []
            for view in views:
                camera_to_centered = torch.tensor(world_to_centered @ view["camera"],
                                                 dtype=torch.float32, device=device)[None]
                drr = view["projector"](subject, camera_to_centered,
                                        n_samples=samples, backend=backend)
                if not torch.isfinite(drr).all():
                    raise RuntimeError("Renderer returned non-finite image")
                gradients = metric.gradients(drr)
                # A volume moved out of the ROI must not beat a negative NCC
                # simply by producing a constant/blank projection (NCC=0).
                if float((gradients.square() * view["mask"]).sum()) <= 1e-12:
                    scores.append(-1.)
                else:
                    score = metric.from_gradients(gradients, view["target_gradients"], view["mask"])
                    scores.append(float(score.item()))
                if keep:
                    images.append(drr)
            loss = 1 - np.average(scores, weights=[v["weight"] for v in views])
            return float(loss), ct_to_world, scores, images

        best_params = np.zeros(6)
        best_loss, _, _, initial_images = evaluate(best_params, keep=True)
        if any(float(image.max()) <= 0 for image in initial_images):
            raise ValueError("Initial DRR is blank in at least one view; check calibration/initial pose")
        initial_loss = best_loss
        for view, image in zip(views, initial_images):
            save_image(folder / f"{view['name']}_target.png", view["target"])
            save_image(folder / f"{view['name']}_initial.png", image)
            save_image(folder / f"{view['name']}_mask.png", view["mask"])
            if float((view["target_gradients"].square() * view["mask"]).sum()) <= 1e-12:
                raise ValueError(f"Target ROI has no gradients: {view['name']}")
        steps = int(iterations if iterations is not None else config.get("iterations", 100))
        scales = np.array(config.get("parameter_scales", [3, 3, 3, 10, 10, 10]), dtype=float)
        if steps < 0 or scales.shape != (6,) or not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("iterations >= 0 and six positive parameter_scales are required")
        population = int(config.get("population_size", 12))
        if population < 4:
            raise ValueError("population_size must be >= 4")
        optimizer = CMA(mean=np.zeros(6), sigma=1, seed=int(config.get("seed", 0)),
                        population_size=population)
        with (folder / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["generation", "loss", "rx_deg", "ry_deg", "rz_deg", "tx_mm", "ty_mm", "tz_mm"])
            writer.writerow([-1, best_loss, *best_params])
            for generation in range(steps):
                solutions = []
                for _ in range(optimizer.population_size):
                    candidate = optimizer.ask()
                    params = candidate * scales
                    loss, _, _, _ = evaluate(params)
                    solutions.append((candidate, loss))
                    if loss < best_loss:
                        best_loss, best_params = loss, params.copy()
                optimizer.tell(solutions)
                writer.writerow([generation, best_loss, *best_params])
                handle.flush()
                print(f"{generation+1}/{steps}: best GNCC={1-best_loss:.6f}", flush=True)
                if optimizer.should_stop():
                    break
        best_loss, final, scores, images = evaluate(best_params, keep=True)
        for view, image in zip(views, images):
            save_image(folder / f"{view['name']}_registered.png", image)
            a = image / image.max().clamp_min(1e-8)
            overlay = torch.cat([view["target"], a, view["target"]], dim=1)[0].permute(1, 2, 0)
            cv2.imwrite(str(folder / f"{view['name']}_overlay.png"),
                        (overlay.clamp(0, 1).cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8))
    result = dict(mode=mode, coordinate_convention=convention, metric="Sobel GNCC",
                  initial_loss=initial_loss, loss=best_loss, per_view_gncc=dict(zip(names, scores)),
                  ct_to_world=final.tolist(), initial_ct_to_world=initial.tolist(),
                  centered_volume_to_ct=centered_to_ct.tolist(), ct_anchor=anchor.tolist(),
                  delta_parameters=best_params.tolist(),
                  parameter_units=["deg", "deg", "deg", "mm", "mm", "mm"],
                  backend=backend, n_samples=samples)
    (folder / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Results: {folder.resolve()}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=["single", "dual"])
    parser.add_argument("--view", help="Selected view name in single mode")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--output", help="Output directory; relative to config file")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if args.view:
        config["single_view"] = args.view
    run_registration(config, args.config.resolve().parent, args.mode, args.iterations, args.output)


if __name__ == "__main__":
    main()
