import os
import numpy as np
import SimpleITK as sitk
from sklearn.model_selection import KFold, train_test_split
import csv
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from inference import inference_method
from monai.losses import DiceLoss, DiceCELoss, GeneralizedDiceLoss, HausdorffDTLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import monai
from monai.data import create_test_image_3d, list_data_collate, decollate_batch, pad_list_data_collate
from monai.inferers import sliding_window_inference
from monai.networks.nets import DynUNet, UNet, dynunet, SwinUNETR, AttentionUnet, UNETR
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    SqueezeDimd,
    AsDiscrete,
    Compose,
    LoadImaged,
    Resized,
    ScaleIntensityd,
    ConcatItemsd,
    RandCropByPosNegLabeld,
    RandRotate90d,
    RandRotated,
    RandFlipd,
    RandZoomd,
    CropForegroundd,
    Orientationd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    Spacingd,
    EnsureType,
    RandAffined,
    Rotated,
    MapTransform,
    ToTensord
)
from monai.visualize import plot_2d_or_3d_image
import matplotlib.pyplot as plt
from line_test import inference_method


def model_engine():
    fold_path = 'Data/lines_data/png_lines'
    img_arr, seg_arr = get_img_only(fold_path)
    img_train, img_test, seg_train, seg_test = train_test_split(img_arr, seg_arr, test_size=0.2, random_state=2)
    train_files = [{"img": Img, "seg": seg}
                   for Img, seg in zip(img_train, seg_train)]
    val_files = [{"img": Img, "seg": seg}
                 for Img, seg in zip(img_test, seg_test)]
    # infer_files = [{"img": Img} for Img in img_arr]
    # print(val_files)
    # print(len(Image_list))
    # print(len(seg_list))
    log_dir = 'sed_model/20240620/AttUNet_model2.pth'
    # log_dir = 'sed_model/UNETR_multi-ver.pth'
    training(train_files, val_files, log_dir, 'AttUNet')
    # inference_method(val_files, log_dir, 'AttUNet')


def get_img_only(data_path):
    Image_list = []
    seg_list = []
    for file in os.listdir(data_path):
        if '.png' in file and 'seg' not in file:
            Image_list.append(os.path.join(data_path, file))
            # print(file)
        for mask in os.listdir(data_path):
            # if 'seg' in mask and file[:-4] == mask.split('_seg')[0]:
            if 'seg' in mask and file.split('.')[0] == mask.split('_seg')[0]:
                # print(file)
                # print(mask.split('_seg')[0])
                # print(mask)
                seg_list.append(os.path.join(data_path, mask))
    # print(Image_list)
    # print(seg_list)
    return np.array(Image_list), np.array(seg_list)



def training(train_files, val_files, model_dir, model_name):
    keys = ["img", "seg"]
    img_size = 256
    # 训练集预处理
    train_transforms = Compose(
        [
            LoadImaged(image_only=True, keys=keys, ensure_channel_first=True, reader="PILReader", reverse_indexing=False),
            # AddChanneld(keys=keys),,
            # CropForegroundd(keys=keys, source_key="img"),
            # 尺寸归一化
            Resized(keys=keys, spatial_size=(img_size, img_size), mode=('bilinear', 'nearest'),
                    align_corners=(True, None)),
            # 像素归一化
            # ConvertToMultiVertebraClassesd(keys="seg"),
            ScaleIntensityd(keys=keys[:-1]),
            # CropForegroundd(keys=keys, source_key="img"),
            # ScaleIntensityd(keys="dis"),
            # NormalizeIntensityd(keys=keys[:-1]),
            # RandScaleIntensityd(keys=keys[:-1], factors=0.1, prob=0.5),
            # RandRotated(keys=keys, range_x=np.pi / 12, prob=0.5, keep_size=True),
            # RandFlipd(keys=keys, spatial_axis=0, prob=0.5),
            # RandZoomd(keys=keys, min_zoom=0.6, max_zoom=1.4, prob=0.5),
        ]
    )
    # 验证集预处理
    val_transforms = Compose(
        [
            LoadImaged(image_only=True, keys=keys, ensure_channel_first=True, reader="PILReader", reverse_indexing=False),
            # EnsureChannelFirstd(keys=keys),
            # ConvertToMultiVertebraClassesd(keys="seg"),
            # CropForegroundd(keys=keys, source_key="T1C"),
            # Orientationd(keys=keys, axcodes="RAS"),
            # Spacingd(
            #     keys=keys,
            #     pixdim=(1.0, 1.0, 2.0),
            #     mode=("bilinear", "nearest"), align_corners=True
            # ),
            # CropForegroundd(keys=keys, source_key="img"),
            Resized(keys=keys, spatial_size=(img_size, img_size), mode=('bilinear', 'nearest'),
                    align_corners=(True, None)),
            ScaleIntensityd(keys=keys[:-1]),
            # NormalizeIntensityd(keys=keys[:-1]),
            # ConcatItemsd(keys=keys[:-1], name="inputs"),
        ]
    )
    # create a training data loader
    # 加载训练集
    batch_size = 16
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=list_data_collate
    )
    # create a validation data loader
    # 加载测试集
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, collate_fn=list_data_collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 单卡训练
    num_classes = 1
    model = AttentionUnet(spatial_dims=2,
                          in_channels=1,
                          out_channels=num_classes,
                          channels=(16, 32, 64, 128, 256),
                          strides=(2, 2, 2, 2)).to(device)
    # model = UNet(
    #     spatial_dims=2,
    #     in_channels=1,
    #     out_channels=num_classes,
    #     channels=(16, 32, 64, 128, 256),
    #     strides=(2, 2, 2, 2),
    #     num_res_units=2,
    # ).to(device)

    # model = UNETR(
    #     in_channels=1,
    #     out_channels=num_classes,
    #     img_size=(img_size, img_size),
    #     feature_size=48,
    #     spatial_dims=2,
    # ).to(device)
    # model = SwinUNETR(
    #     img_size=(img_size, img_size),
    #     in_channels=1,
    #     out_channels=num_classes,
    #     feature_size=48,
    #     spatial_dims=2
    # ).to(device)
    # model = EAUNETR(img_size=(img_size, img_size),
    #                 in_channels=1,
    #                 out_channels=num_classes,
    #                 feature_size=16,
    #                 spatial_dims=2,
    #                 hidden_size=768).to(device)
    # model = EGEUNet(num_classes=num_classes,
    #                 input_channels=1,
    #                 c_list=[16, 24, 32, 64, 128, 256],
    #                 bridge=True,
    #                 gt_ds=True,
    #                 ).to(device)
    # reg_model = pidinet().to(device)
    # reg_model = CASENet_resnet101(num_classes=7).to(device)
    # loss_function = torch.nn.CrossEntropyLoss()
    DCE_loss = DiceCELoss(sigmoid=True)
    # DC_loss = DiceLoss(sigmoid=True)
    # HD_loss = HausdorffDTLoss(to_onehot_y=True, softmax=True)
    # MSE_loss = torch.nn.MSELoss()
    # DCE_loss = DiceCELoss(squared_pred=True, to_onehot_y=False, sigmoid=True)
    # GDice_loss = GeneralizedDiceLoss(to_onehot_y=True, softmax=True)
    # Boundary_loss = BoundaryLoss(idc=[0, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.5, last_epoch=-1)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    hd_metric = HausdorffDistanceMetric(include_background=True, percentile=95)
    # dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    # start a typical PyTorch training
    epoch_num = 100
    # optimizer = torch.optim.Adam(model.parameters(), 0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)
    # 每隔2次验证一次
    val_interval = 2
    # 保存最佳的AUC
    best_metric = 0
    # 测试集dice最高时的epoch次数
    best_metric_epoch = -1
    # 用于保存训练集的loss
    epoch_loss_values = []
    # 用于保存测试集的loss
    val_loss_list = []
    # 用于保存每次验证时的Dice
    metric_values = []
    metric_values_v1 = []
    metric_values_v2 = []
    metric_values_v3 = []
    # post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True)])
    # post_label = Compose([AsDiscrete(to_onehot=num_classes)])
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    # 训练循环
    for epoch in range(epoch_num):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        # a -= 0.01
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["img"].to(device),
                batch_data["seg"].to(device),
                # batch_data["dis"].to(device)
            )
            optimizer.zero_grad()
            # score_feats5, fused_feats = reg_model(inputs)
            outputs = model(inputs)
            # 计算损失函数
            # feats5_loss = loss_function(score_feats5, labels.long())
            # plt.subplot(1, 2, 1)
            # # plt.imshow(torch.argmax(outputs[0, :], dim=0).squeeze().detach().cpu().numpy())
            # plt.imshow(inputs[0].squeeze().detach().cpu().numpy())
            # plt.subplot(1, 2, 2)
            # plt.imshow(labels[0].squeeze().detach().cpu().numpy())
            # plt.show()
            dce_loss = DCE_loss(outputs, labels)
            loss = dce_loss
            # 损失回传
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size + 1}, "
                f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        scheduler.step()
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        # 开始验证测试集
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_epoch_loss = 0
                val_step = 0
                for val_data in val_loader:
                    val_step += 1
                    val_inputs, val_labels = (
                        val_data["img"].to(device),
                        val_data["seg"].to(device),
                    )
                    roi_size = (img_size, img_size)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model)
                    # val_dcel = DCE_loss(val_outputs, val_labels)
                    # val_bl = Boundary_loss(val_outputs, val_labels)
                    val_loss = DCE_loss(val_outputs, val_labels)
                    val_epoch_loss += val_loss.item()
                    # val_bn_labels = torch.where(val_labels > 1, 1, val_labels)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    # val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    # val_bn_outputs = torch.stack(val_outputs, dim=0)
                    # print(val_outputs.shape)
                    # val_bn_outputs = torch.argmax(val_bn_outputs, dim=1)
                    # val_bn_outputs = torch.where(val_bn_outputs > 1, 1, val_bn_outputs)
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    hd_metric(y_pred=val_outputs, y=val_labels)
                    # dice_metric_batch(y_pred=val_outputs, y=val_labels)
                val_epoch_loss /= val_step
                val_loss_list.append(val_epoch_loss)
                # aggregate the final mean dice result
                # 计算Dice
                metric = dice_metric.aggregate().item()
                hd_result = hd_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric >= best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), model_dir)
                    print("saved new best metric sed_model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f} current mean hd: {hd_result:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")
    # writer.close()
    # 绘制损失值曲线
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    val_x = [val_interval * (i + 1) for i in range(len(val_loss_list))]
    val_y = val_loss_list
    plt.xlabel('Epoch')
    plt.plot(x, y)
    plt.plot(val_x, val_y)
    plt.legend(['Training Loss', 'Validation Loss'])
    # 保存曲线图像
    # plt.savefig('sed_image_results/20240430/{name}_Loss.jpg'.format(name=model_name), dpi=300)
    plt.close()
    plt.show()
    # val_acc = pd.DataFrame(data=val_accuracy_list)  # 数据有三列，列名分别为one,two,three
    # val_acc.to_csv(model_dir.split('.p')[0] + '_acc_per_epoch.csv', encoding='gbk')
    # val_miu = pd.DataFrame(data=val_miu)  # 数据有三列，列名分别为one,two,three
    # val_miu.to_csv(model_dir.split('.p')[0] + '_miu_per_epoch.csv', encoding='gbk')
    # 绘制测试集的Dice训练次数的关系曲线
    plt.title("Val Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    # plt.legend()
    # 保存曲线图像
    # plt.savefig('sed_image_results/240327/{name}_total_Dice.jpg'.format(name=model_name), dpi=300)
    plt.show()


if __name__ == '__main__':
    model_engine()
