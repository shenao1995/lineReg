import os.path
import numpy as np
import SimpleITK as sitk
import cv2


def crop_ct_vert(img_path, mask_path, crop_vert_path=None, crop_vert_seg_path=None):
    # 读取原始锥体CT图像
    img = sitk.ReadImage(img_path)

    # 读取原始锥体CT图像中的体积像素（结果为numpy数组）
    ct_arr = sitk.GetArrayFromImage(img)

    # 读取待配准椎骨CT的mask图像
    mask = sitk.ReadImage(mask_path)

    # 读取待配准椎骨CT的mask图像的体积像素（结果为numpy数组）
    mask_arr = sitk.GetArrayFromImage(mask)

    # 注意，这里img.GetSize()返回的是形状为(width, height, depth)的元组，ct_arr.shape返回的是形状为(depth, height, width)的numpy数组
    # ct_arr与mask_arr的形状相同
    # print(img.GetSize())
    # print(ct_arr.shape)
    # print(mask_arr.shape)

    # 对于待配准椎骨CT的mask图像中像素值为1的区域（即前景区域），保留原始锥体CT图像中的像素值；对于像素值不为1的区域（即背景区域），使用原始锥体CT图像中的最小像素值来代替
    # 这种操作的目的是将mask图像中的目标区域（即待配准椎骨）提取出来，并将背景区域设置为一个较小的像素值，从而使得后续的处理操作（如裁剪、分割等）更容易处理
    normalized_arr = np.where(mask_arr == 1, ct_arr, ct_arr.min())

    # 用经过处理的原始锥体CT图像的体积像素生成处理后锥体CT图像processed_img
    processed_img = sitk.GetImageFromArray(normalized_arr)

    # 将原始锥体CT图像的信息复制到处理后锥体CT图像processed_img中
    processed_img.CopyInformation(img)

    # 实例化simpleitk中的LabelShapeStatisticsImageFilter对象lesion_filter，用于执行形状统计分析（如面积、体积、最大直径、最小直径等）
    lesion_filter = sitk.LabelShapeStatisticsImageFilter()

    # 计算待配准椎骨CT的mask图像中每个标签的形状特征并将结果保存在lesion_filter对象中供后续访问
    # 详解：
    # 在图像处理中，特别是在分割任务中，"标签"通常用来表示图像中不同区域或对象的标识符。在mask图像中，每个标签通常代表一个连通区域，或者说一个对象或目标
    # 具体来说，在mask图像中，每个像素都被分配了一个标签值，用来标识它所属的对象或区域
    # 这些标签值通常是正整数，从1开始递增，每个不同的连通区域分配一个不同的标签值。通常情况下，像素值为0的区域被视为背景，而其他标签值表示不同的目标或区域
    lesion_filter.Execute(mask)

    # 读取待配准椎骨CT的mask图像中标签值为1的连通区域的检测框信息（检测框是一个轴对齐的矩形框，用来包围目标区域）
    lesion_boxing = lesion_filter.GetBoundingBox(1)

    # (检测框x轴方向上的起始位置, 检测框y轴方向上的起始位置, 检测框z轴方向上的起始位置, 检测框x轴方向上的尺寸, 检测框y轴方向上的尺寸, 检测框z轴方向上的尺寸)
    # print(lesion_boxing)

    # 检测框x、y、z轴方向上的尺寸
    boxing_size = (
        lesion_boxing[int(len(lesion_boxing) / 2):][0], lesion_boxing[int(len(lesion_boxing) / 2):][1],
        int(lesion_boxing[5]))

    # 检测框x、y、z轴方向上的起始位置
    start_boxing = (
        lesion_boxing[0:int(len(lesion_boxing) / 2)][0], lesion_boxing[0:int(len(lesion_boxing) / 2)][1],
        int(lesion_boxing[2]))

    # 读取原始锥体CT图像的体积像素间距，即分别在x、y、z轴上每个像素之间的距离
    spacing = img.GetSpacing()

    # 待配准椎骨CT图像的中心坐标（x、y、z轴的起始位置加上边框长度的一半）
    ver_center_x, ver_center_y, ver_center_z = start_boxing[0] + boxing_size[0] / 2, \
                                               start_boxing[1] + boxing_size[1] / 2, \
                                               start_boxing[2] + boxing_size[2] / 2

    # 原始锥体CT图像的中心坐标（x、y、z轴长度的一半）
    ct_center_x, ct_center_y, ct_center_z = img.GetSize()[0] / 2, img.GetSize()[1] / 2, img.GetSize()[2] / 2

    # 将待配准椎骨CT图像的中心坐标转换成numpy数组
    ver_center = np.array((ver_center_x, ver_center_y, ver_center_z), dtype=np.float64)

    # 将原始锥体CT图像的中心坐标转换成numpy数组
    ct_center = np.array((ct_center_x, ct_center_y, ct_center_z), dtype=np.float64)

    # 计算原始锥体CT图像的中心坐标与待配准椎骨CT图像的中心坐标在物理空间上的偏移量
    bx, by, bz = (ct_center - ver_center) * spacing

    # 根据检测框尺寸和起始位置对处理后锥体CT图像进行裁剪，得到待配准椎骨CT图像cropped_img
    cropped_img = sitk.RegionOfInterest(processed_img, boxing_size, start_boxing)

    cropped_mask = sitk.RegionOfInterest(mask, boxing_size, start_boxing)

    # 将待配准椎骨CT图像的体积像素间距、原点和方向设为与原始锥体CT图像一致
    cropped_img.SetSpacing(img.GetSpacing())
    cropped_img.SetOrigin(img.GetOrigin())
    cropped_img.SetDirection(img.GetDirection())

    cropped_mask.SetSpacing(mask.GetSpacing())
    cropped_mask.SetOrigin(mask.GetOrigin())
    cropped_mask.SetDirection(mask.GetDirection())

    # 检查保存路径crop_vert_path，将裁剪出来的待配准椎骨CT图像cropped_img写入到指定路径文件中
    if crop_vert_path:
        # if not os.path.exists(crop_vert_path):
            sitk.WriteImage(cropped_img, crop_vert_path)

    if crop_vert_seg_path:
        if not os.path.exists(crop_vert_seg_path):
            sitk.WriteImage(cropped_mask, crop_vert_seg_path)

    # 返回原始锥体CT图像的中心坐标与待配准椎骨CT图像的中心坐标在物理空间上的偏移量、裁剪出来的待配准椎骨CT图像及其保存路径
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