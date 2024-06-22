import logging
import os
import sys
import tempfile
from glob import glob
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt
from monai.config import print_config
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from monai.data import Dataset, DataLoader, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from sklearn.metrics import accuracy_score, recall_score
from monai.networks.nets import UNet, DynUNet, SwinUNETR, AttentionUnet, UNETR
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    CropForegroundd,
    Orientationd,
    Resized,
    SaveImaged,
    ScaleIntensityd,
    NormalizeIntensityd,
    ConcatItemsd,
    SaveImage,
    Activations,
    AsDiscrete,
    EnsureType,
    ToTensord,
    KeepLargestConnectedComponent,
    RemoveSmallObjects,
    MapTransform,
    Resize,
)
from collections import Counter


def inference_method(inference_files, log_dir, modelName):
    # print(inference_files)
    img_size = 256
    keys = ["img", "seg"]
    # keys = ["img"]
    infer_transforms = Compose(
        [
            LoadImaged(image_only=True, keys=keys, ensure_channel_first=True, reader="PILReader", reverse_indexing=False),
            # EnsureChannelFirstd(keys=keys),
            # ConvertToMultiVertebraClassesd(keys="seg"),
            # CropForegroundd(keys=keys, source_key="img"),
            Resized(keys=keys, spatial_size=(img_size, img_size), mode=('bilinear', 'nearest'),
                    align_corners=(True, None)),
            ScaleIntensityd(keys=keys[:-1]),
            # ScaleIntensityd(keys=keys),
            # ScaleIntensityd(keys="dis"),
            # NormalizeIntensityd(keys=keys[:-1]),
            # ConcatItemsd(keys=keys[:-1], name="inputs"),
        ]
    )
    # define dataset and dataloader
    val_ds = Dataset(data=inference_files, transform=infer_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, collate_fn=list_data_collate)
    # define post transforms
    # post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5), RemoveSmallObjects(min_size=50)])
    # post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
    num_classes = 1
    #  计算指标时用这个trans
    # post_trans = Compose([AsDiscrete(argmax=True, to_onehot=num_classes), RemoveSmallObjects(min_size=40)])
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    # post_label = Compose([AsDiscrete(to_onehot=num_classes)])

    #  输出结果时用这个trans
    # post_trans = Compose([Activations(softmax=True), AsDiscrete(argmax=True), RemoveSmallObjects(min_size=40, independent_channels=False)])
    # post_label = Compose([AsDiscrete(to_onehot=num_classes)])
    #  设置保存路径
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    # dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    hd_metric = HausdorffDistanceMetric(include_background=True, percentile=95)
    saver = SaveImage(output_dir="./output/dual_view", output_ext=".nii", output_postfix="seg",
                      separate_folder=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net = UNet(
    #     spatial_dims=2,
    #     in_channels=1,
    #     out_channels=num_classes,
    #     channels=(16, 32, 64, 128, 256),
    #     strides=(2, 2, 2, 2),
    #     num_res_units=2,
    # ).to(device)
    net = AttentionUnet(spatial_dims=2,
                        in_channels=1,
                        out_channels=num_classes,
                        channels=(16, 32, 64, 128, 256),
                        strides=(2, 2, 2, 2)).to(device)
    # net = UNETR(
    #     in_channels=1,
    #     out_channels=num_classes,
    #     img_size=(img_size, img_size),
    #     feature_size=48,
    #     spatial_dims=2,
    # ).to(device)
    # net = SwinUNETR(
    #     img_size=(img_size, img_size),
    #     in_channels=1,
    #     out_channels=num_classes,
    #     feature_size=48,
    #     spatial_dims=2
    # ).to(device)
    # net = EAUNETR(img_size=(img_size, img_size),
    #               in_channels=1,
    #               out_channels=num_classes,
    #               feature_size=16,
    #               spatial_dims=2,
    #               hidden_size=768).to(device)
    net.load_state_dict(torch.load(log_dir))
    net.eval()
    dice_results = []
    hd_results = []
    iou_results = []
    with torch.no_grad():
        count = 0
        for val_data in val_loader:
            count += 1
            val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
            # val_images = val_data["img"].to(device)
            # src_image_dir_list = val_data['Img_meta_dict']['filename_or_obj']
            # print(src_image_dir_list)
            # val_labels = torch.squeeze(val_labels, 4)
            # val_labels = torch.squeeze(val_labels, 2)
            # define sliding window size and batch size for windows inference
            roi_size = (img_size, img_size)
            sw_batch_size = 4
            # val_score_feats5, val_fused_feats = sliding_window_inference(val_images, roi_size, sw_batch_size, net)
            val_outputs = sliding_window_inference(
                val_images, roi_size, sw_batch_size, net)
            # val_labels = val_labels.to(dtype=torch.long)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            # val_output = post_trans(val_outputs[0])
            # val_labels = [post_label(i) for i in decollate_batch(val_labels)]
            # val_bn_outputs = torch.stack(val_outputs, dim=0)
            # val_bn_outputs = torch.argmax(val_bn_outputs, dim=1)
            # val_bn_outputs = torch.where(val_bn_outputs > 1, 1, val_bn_outputs)
            # val_bn_labels = torch.where(val_labels > 1, 1, val_labels)
            # val_bn_outputs = torch.unsqueeze(val_bn_outputs, dim=1)
            # print(val_outputs[0].shape)
            # print(val_labels.shape)

            # show_val_pred(val_images, val_labels, val_outputs, count)

            # visualize the 3 channels label corresponding to this image
            # plt.subplot(1, 3, 1)
            # plt.title(f"输入图像", fontproperties='SimHei', fontsize=10)
            # plt.axis('off')  # 去坐标轴
            # plt.xticks([])  # 去 x 轴刻度
            # plt.yticks([])  # 去 y 轴刻度
            # plt.imshow(torch.rot90(val_images[0, 0, :, :], -1).cpu(), cmap="gray")
            # plt.subplot(1, 3, 2)
            # plt.title(f"标签", fontproperties='SimHei', fontsize=10)
            # plt.axis('off')  # 去坐标轴
            # plt.xticks([])  # 去 x 轴刻度
            # plt.yticks([])  # 去 y 轴刻度
            # plt.imshow(torch.rot90(val_labels[0, 2, :, :].detach(), -1).cpu())
            # plt.subplot(1, 3, 3)
            # plt.title(f"预测", fontproperties='SimHei', fontsize=10)
            # plt.axis('off')  # 去坐标轴
            # plt.xticks([])  # 去 x 轴刻度
            # plt.yticks([])  # 去 y 轴刻度
            # plt.imshow(torch.rot90(val_output[2, :, :].detach(), -1).cpu())
            # plt.savefig('output/multi-verb/result{num}.jpg'.format(num=count), dpi=300, bbox_inches='tight')
            # plt.show()
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)
            iou_metric(y_pred=val_outputs, y=val_labels)
            hd_metric(y_pred=val_outputs, y=val_labels)
            # print(dice_metric(y_pred=val_outputs, y=val_labels))
            # dice_results.append((dice_metric(y_pred=val_outputs, y=val_labels)[0][0].item() +
            #                      dice_metric(y_pred=val_outputs, y=val_labels)[0][1].item()) / 2)
            # hd_results.append((hd_metric(y_pred=val_outputs, y=val_labels)[0][0].item() +
            #                    hd_metric(y_pred=val_outputs, y=val_labels)[0][1].item()) / 2)
            # iou_results.append((iou_metric(y_pred=val_outputs, y=val_labels)[0][0].item() +
            #                     iou_metric(y_pred=val_outputs, y=val_labels)[0][1].item()) / 2)
            # dice_metric_batch(y_pred=val_outputs, y=val_labels)
            # val_outputs = torch.argmax(val_outputs, dim=1)
            # val_outputs = torch.unsqueeze(val_outputs, 1).to(device)
            # for val_output in val_outputs:
            #     saver(val_output)
        print("dice metric:%.4f, iou:%.4f, hd:%.4f" % (dice_metric.aggregate().item(),
                                                       iou_metric.aggregate().item(),
                                                       hd_metric.aggregate().item()))
        # # metric_batch_org = dice_metric_batch.aggregate()
        # dice_metric.reset()
        # iou_metric.reset()
        # hd_metric.reset()
        # print(f'dice为:{np.mean(dice_results):.4f}±{np.std(dice_results, ddof=1):.4f}')
        # print(f'hd为:{np.mean(hd_results):.4f}±{np.std(hd_results, ddof=1):.4f}')
        # print(f'iou为:{np.mean(iou_results):.4f}±{np.std(iou_results, ddof=1):.4f}')
        # torch.set_printoptions(precision=4)
    # metric_v1, metric_v2, metric_v3 = metric_batch_org[0].item(), metric_batch_org[1].item(), metric_batch_org[
    #     2].item()
    # print(f"metric_v1: {metric_v1:.4f}")
    # print(f"metric_v2: {metric_v2:.4f}")
    # print(f"metric_v3: {metric_v3:.4f}")
    # dice_metric_batch.reset()


def show_val_pred(images, labels, outputs, i):
    # val_images = torch.permute(images, (0, 1, 3, 2))
    # val_labels = torch.permute(labels, (0, 1, 3, 2))
    # val_outputs = torch.permute(outputs, (0, 1, 3, 2))
    plt.figure("check", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title(f"输入图像", fontproperties='SimHei', fontsize=20)
    plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去 x 轴刻度
    plt.yticks([])  # 去 y 轴刻度
    plt.imshow(images[0, 0, :, :].cpu(), cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title(f"标签", fontproperties='SimHei', fontsize=20)
    plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去 x 轴刻度
    plt.yticks([])  # 去 y 轴刻度
    plt.imshow(labels[0, 0, :, :].cpu())
    plt.subplot(1, 3, 3)
    plt.title(f"预测", fontproperties='SimHei', fontsize=20)
    plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去 x 轴刻度
    plt.yticks([])  # 去 y 轴刻度
    # plt.imshow(torch.argmax(val_outputs[0], dim=0).cpu())
    plt.imshow(outputs[0][0, :, :].cpu())
    # plt.savefig('output/xjt_4/result{num}.jpg'.format(num=i), dpi=300)
    plt.show()


class ConvertToMultiVertebraClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(d[key] == 1)

            result.append(torch.logical_or(d[key] == 1, d[key] == 2))

            result.append(torch.logical_or(torch.logical_or(d[key] == 1, d[key] == 2), d[key] == 3))

            d[key] = torch.stack(result, dim=0).float()
        return d
