import os.path
import numpy as np
import SimpleITK as sitk
import cv2
from scipy import ndimage as ndi
import matplotlib.pyplot as plt


def crop_ct_vert(img_path, mask_path, crop_vert_path=None, crop_vert_seg_path=None, vert_name='L1'):
    # 建立椎骨名称到标签值的映射字典
    label_dict = {'L1': 21, 'L2': 22, 'L3': 23, 'L4': 24, 'L5': 25}
    if vert_name not in label_dict:
        raise ValueError(f"不支持的椎骨名称: {vert_name}。必须是 {list(label_dict.keys())} 之一。")

    target_label = label_dict[vert_name]

    # 1. 读取原始锥体CT图像及体积像素
    img = sitk.ReadImage(img_path)
    ct_arr = sitk.GetArrayFromImage(img)

    # 2. 读取包含所有椎骨的全局 mask 图像及体积像素
    mask = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(mask)

    # 3. 核心修改：生成特定椎骨的二值化掩膜 (目标椎骨设为1，其余全为0)
    binary_mask_arr = np.where(mask_arr == target_label, 1, 0).astype(np.uint8)
    binary_mask = sitk.GetImageFromArray(binary_mask_arr)
    binary_mask.CopyInformation(mask)

    # 对于待配准椎骨区域保留原像素值，背景区域使用最小像素值代替
    normalized_arr = np.where(binary_mask_arr == 1, ct_arr, ct_arr.min())
    processed_img = sitk.GetImageFromArray(normalized_arr)
    processed_img.CopyInformation(img)

    # 4. 实例化滤波器，并在二值化掩膜上执行统计
    lesion_filter = sitk.LabelShapeStatisticsImageFilter()
    lesion_filter.Execute(binary_mask)

    # 检查是否成功找到了该标签
    if not lesion_filter.HasLabel(1):
        raise RuntimeError(f"在分割文件中未找到标签值为 {target_label} ({vert_name}) 的区域！")

    # 读取目标区域（标签1）的检测框信息
    lesion_boxing = lesion_filter.GetBoundingBox(1)

    # 检测框尺寸和起始位置
    boxing_size = (lesion_boxing[3], lesion_boxing[4], lesion_boxing[5])
    start_boxing = (lesion_boxing[0], lesion_boxing[1], lesion_boxing[2])

    # 计算中心点偏移量
    spacing = img.GetSpacing()
    ver_center_x = start_boxing[0] + boxing_size[0] / 2
    ver_center_y = start_boxing[1] + boxing_size[1] / 2
    ver_center_z = start_boxing[2] + boxing_size[2] / 2
    ver_center = np.array((ver_center_x, ver_center_y, ver_center_z), dtype=np.float64)

    ct_center_x, ct_center_y, ct_center_z = img.GetSize()[0] / 2, img.GetSize()[1] / 2, img.GetSize()[2] / 2
    ct_center = np.array((ct_center_x, ct_center_y, ct_center_z), dtype=np.float64)

    bx, by, bz = (ct_center - ver_center) * spacing

    # 5. 裁剪图像和对应的二值掩膜
    cropped_img = sitk.RegionOfInterest(processed_img, boxing_size, start_boxing)
    cropped_mask = sitk.RegionOfInterest(binary_mask, boxing_size, start_boxing)

    # 检查保存路径并写入文件
    if crop_vert_path:
        # 创建父目录（防止目录不存在报错）
        os.makedirs(os.path.dirname(crop_vert_path), exist_ok=True)
        sitk.WriteImage(cropped_img, crop_vert_path)

    if crop_vert_seg_path:
        os.makedirs(os.path.dirname(crop_vert_seg_path), exist_ok=True)
        sitk.WriteImage(cropped_mask, crop_vert_seg_path)

    return bx, by, bz


def extract_traditional_edge(img_tensor, threshold_ratio=0.08, margin_ratio=0.15):
    """
    改进的传统边缘提取算法：
    1. 找到椎骨的上下边界，舍弃顶部和底部的 margin_ratio（如15%），避免提取到底部边缘。
    2. 扫描中间行，提取左右极值点。
    3. 利用 OpenCV 将提取的离散点连接成两条连续的线（左边缘线和右边缘线）。
    """
    img_np = img_tensor.squeeze().detach().cpu().numpy()
    img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    mask = img_norm > threshold_ratio

    edge_map = np.zeros_like(img_np, dtype=np.uint8)

    # 1. 找到图像中有像素的所有行 (Y轴范围)
    active_rows = np.any(mask, axis=1)
    y_indices = np.where(active_rows)[0]

    if len(y_indices) == 0:
        return edge_map.astype(np.float32)

    y_min, y_max = y_indices[0], y_indices[-1]
    v_height = y_max - y_min

    # 2. 裁剪掉顶部和底部的区域，专门针对“两侧”
    trim = int(v_height * margin_ratio)
    valid_y_min = y_min + trim
    valid_y_max = y_max - trim

    left_points = []
    right_points = []

    # 3. 收集有效行内的左右边缘点
    for y in range(valid_y_min, valid_y_max + 1):
        x_indices = np.where(mask[y, :])[0]
        if len(x_indices) > 0:
            left_points.append([x_indices[0], y])
            right_points.append([x_indices[-1], y])

    # 4. 将点按顺序连接成线
    if len(left_points) > 1 and len(right_points) > 1:
        # cv2.polylines 需要 int32 类型且 shape 为 (-1, 1, 2) 的坐标数组
        left_pts_arr = np.array(left_points, dtype=np.int32).reshape((-1, 1, 2))
        right_pts_arr = np.array(right_points, dtype=np.int32).reshape((-1, 1, 2))

        # 画线，color=1 表示 mask 的值为1，thickness=2 代表线宽（自动加粗）
        cv2.polylines(edge_map, [left_pts_arr], isClosed=False, color=1, thickness=2)
        cv2.polylines(edge_map, [right_pts_arr], isClosed=False, color=1, thickness=2)

    return edge_map.astype(np.float32)


def _largest_component_3d(mask):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask.astype(bool)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == np.argmax(counts)


def _largest_component_2d(mask_u8):
    mask_u8 = (mask_u8 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    if num <= 1:
        return mask_u8
    keep = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == keep).astype(np.uint8)


def _odd_kernel_size(mm, spacing, min_size=3):
    k = int(round(mm / max(float(spacing), 1e-6)))
    k = max(min_size, k)
    if k % 2 == 0:
        k += 1
    return k


def geodesic_reconstruct(seed, mask):
    """
    2D 形态学重建：从 seed 出发，在 mask 约束内膨胀直到稳定
    """
    seed = (seed > 0).astype(np.uint8)
    mask = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    prev = np.zeros_like(seed)
    cur = seed.copy()

    while True:
        dil = cv2.dilate(cur, kernel, iterations=1)
        cur = ((dil > 0) & (mask > 0)).astype(np.uint8)
        if np.array_equal(cur, prev):
            break
        prev = cur.copy()

    return cur


def _largest_contiguous_run(indices):
    indices = np.asarray(indices, dtype=np.int32)
    if len(indices) == 0:
        return None

    split_ids = np.where(np.diff(indices) > 1)[0] + 1
    runs = np.split(indices, split_ids)

    # 优先选最长的连续区间，通常就是椎体主体
    return max(runs, key=len)


def extract_ap_body_side_fiducials_from_target_volume(
        vol_path,
        source="cropped_ct",
        label_value=None,
        bg_eps=1e-3,
        z_keep_ratio=(0.25, 0.75),
        z_step=1,
        x_profile_ratio=0.22,
        morph_open_mm=1.5,
        side_band_mm=1.5,
        y_samples_per_slice=1,
        min_slice_area=30,
        min_run_width_vox=5,
        lps_to_ras=True,
):
    """
    从目标椎节 volume 中提取 AP 正位下的椎体左右侧 3D 点。

    source:
        "cropped_ct" : 输入是 crop_ct_vert 输出的 save_dir，背景为 ct_arr.min()
        "binary_mask": 输入是 cropped binary mask
        "seg_label"  : 输入是全脊柱 label mask，需要指定 label_value，例如 L2=22

    返回:
        points_xyz: [N, 3] RAS 世界坐标（nanodrr_adapter 会减去 CT 中心）
    """

    img = sitk.ReadImage(vol_path)
    arr = sitk.GetArrayFromImage(img)  # [z, y, x]
    spacing = np.array(img.GetSpacing(), dtype=np.float32)  # [sx, sy, sz]

    if source == "cropped_ct":
        bg = float(arr.min())
        mask = arr > bg + bg_eps

    elif source == "binary_mask":
        mask = arr > 0

    elif source == "seg_label":
        if label_value is None:
            raise ValueError("source='seg_label' 时必须指定 label_value，例如 L2=22")
        mask = arr == label_value

    else:
        raise ValueError(f"Unsupported source: {source}")

    mask = mask.astype(bool)

    # 基础清理
    mask = ndi.binary_fill_holes(mask)
    mask = _largest_component_3d(mask)

    z_ids = np.where(mask.any(axis=(1, 2)))[0]
    if len(z_ids) == 0:
        raise RuntimeError(f"No target vertebra voxels found in {vol_path}")

    zmin, zmax = int(z_ids[0]), int(z_ids[-1])
    zlen = zmax - zmin + 1

    z0 = zmin + int(round(z_keep_ratio[0] * zlen))
    z1 = zmin + int(round(z_keep_ratio[1] * zlen))

    sx, sy, sz = spacing

    if morph_open_mm is not None and morph_open_mm > 0:
        kx = _odd_kernel_size(morph_open_mm, sx)
        ky = _odd_kernel_size(morph_open_mm, sy)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky))
    else:
        open_kernel = None

    side_band_vox = max(1, int(round(side_band_mm / sx)))

    points_zyx = []

    for z in range(z0, z1 + 1, z_step):
        sl_raw = mask[z].astype(np.uint8)

        if int(sl_raw.sum()) < min_slice_area:
            continue

        # 轻微 opening：去掉很细的横突/椎板连接
        if open_kernel is not None:
            sl = cv2.morphologyEx(sl_raw, cv2.MORPH_OPEN, open_kernel)
            if int(sl.sum()) < min_slice_area:
                sl = sl_raw
        else:
            sl = sl_raw

        # 只取最大 2D 连通域
        sl = _largest_component_2d(sl)

        if int(sl.sum()) < min_slice_area:
            continue

        # 沿 y 方向统计每个 x 的厚度
        x_profile = sl.sum(axis=0)

        if x_profile.max() <= 0:
            continue

        # 高厚度区域，通常对应椎体主体，而不是横突
        valid_x = np.where(x_profile >= x_profile_ratio * x_profile.max())[0]

        if len(valid_x) == 0:
            continue

        body_run = _largest_contiguous_run(valid_x)

        if body_run is None or len(body_run) < min_run_width_vox:
            continue

        x_left = int(body_run[0])
        x_right = int(body_run[-1])

        # 在左右边缘附近取真实 mask 中的点
        def append_side(x_edge, side_name):
            x0 = max(0, x_edge - side_band_vox)
            x1 = min(sl_raw.shape[1] - 1, x_edge + side_band_vox)

            band = sl_raw[:, x0:x1 + 1]
            ys, xs_local = np.where(band > 0)

            if len(ys) == 0:
                return

            xs = xs_local + x0

            if y_samples_per_slice <= 1:
                qs = [0.5]
            else:
                qs = np.linspace(0.2, 0.8, y_samples_per_slice)

            for q in qs:
                y = int(round(np.quantile(ys, q)))

                near = np.where(np.abs(ys - y) <= 1)[0]
                if len(near) == 0:
                    near = np.arange(len(ys))

                if side_name == "left":
                    idx = near[np.argmin(xs[near])]
                else:
                    idx = near[np.argmax(xs[near])]

                points_zyx.append((int(z), int(ys[idx]), int(xs[idx])))

        append_side(x_left, "left")
        append_side(x_right, "right")

    if len(points_zyx) == 0:
        raise RuntimeError(
            "No AP body side fiducials extracted. "
            "Try z_keep_ratio=(0.2,0.8), lower x_profile_ratio, or lower morph_open_mm."
        )

    # z,y,x index -> physical xyz
    points_xyz = []
    for z, y, x in points_zyx:
        p = img.TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z)))
        points_xyz.append(p)

    points_xyz = np.asarray(points_xyz, dtype=np.float32)

    # 保持和你原 create_ap_la_lines 里的坐标习惯一致
    if lps_to_ras:
        points_xyz[:, 0] = -points_xyz[:, 0]
        points_xyz[:, 1] = -points_xyz[:, 1]

    return points_xyz


def build_line_distance_map(line_np, dilate_radius=1):
    line_np = (line_np > 0).astype(np.uint8)

    if dilate_radius > 0:
        k = 2 * dilate_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        line_np = cv2.dilate(line_np, kernel, iterations=1)

    inv = (1 - line_np).astype(np.uint8)
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    return dt.astype(np.float32), line_np


def edge_dt_dice_loss(moving_line_np, gt_line_np, gt_dt_np):
    moving_line_np = (moving_line_np > 0).astype(np.uint8)
    gt_line_np = (gt_line_np > 0).astype(np.uint8)

    ys, xs = np.where(moving_line_np > 0)

    if len(xs) == 0:
        return 1.0

    h, w = gt_line_np.shape

    # projected points 到 GT edge 的平均距离
    dt_loss = float(np.mean(gt_dt_np[ys, xs]) / max(h, w))

    # 辅助 Dice，避免只靠距离图
    inter = np.sum(moving_line_np * gt_line_np)
    union = np.sum(moving_line_np) + np.sum(gt_line_np)
    dice_loss = 1.0 - (2.0 * inter + 1e-8) / (union + 1e-8)

    return dt_loss + 0.2 * dice_loss


def debug_show_body_extraction(seg_path, z_list, anterior_is_low_y=True):
    img = sitk.ReadImage(seg_path)
    arr = sitk.GetArrayFromImage(img)
    mask = (arr > 0).astype(np.uint8)

    for z in z_list:
        sl = mask[z].copy()
        ys, xs = np.where(sl > 0)
        if len(xs) == 0:
            continue

        y_min, y_max = ys.min(), ys.max()
        y_len = y_max - y_min + 1
        anterior_keep_ratio = 0.60

        if anterior_is_low_y:
            y_cut = y_min + int(round(anterior_keep_ratio * y_len))
            sl_ap = sl.copy()
            sl_ap[y_cut + 1:, :] = 0
        else:
            y_cut = y_max - int(round(anterior_keep_ratio * y_len))
            sl_ap = sl.copy()
            sl_ap[:y_cut, :] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        seed = cv2.erode(sl_ap, kernel, iterations=1)
        seed = _largest_component_2d(seed)
        body = geodesic_reconstruct(seed, sl_ap)
        body = _largest_component_2d(body)

        plt.figure(figsize=(12, 3))
        plt.subplot(1, 4, 1); plt.title(f"slice {z}"); plt.imshow(sl, cmap='gray')
        plt.subplot(1, 4, 2); plt.title("anterior"); plt.imshow(sl_ap, cmap='gray')
        plt.subplot(1, 4, 3); plt.title("seed"); plt.imshow(seed, cmap='gray')
        plt.subplot(1, 4, 4); plt.title("body"); plt.imshow(body, cmap='gray')
        plt.show()
