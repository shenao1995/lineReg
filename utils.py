import os.path
import numpy as np
import SimpleITK as sitk
import cv2


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