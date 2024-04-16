import os

import torch
import matplotlib.pyplot as plt
from monai.data import Dataset, DataLoader, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet, SwinUNETR, UNETR, AttentionUnet
from monai.transforms import (
    Activationsd,
    Compose,
    EnsureChannelFirstd,
    ScaleIntensityd,
    NormalizeIntensityd,
    Resized,
    SaveImage,
    Activations,
    AsDiscrete,
    ToTensord,
    RemoveSmallObjects
)
import numpy as np
import SimpleITK as sitk


# import os
# import monai


def infer_method(input_tensor=None, img_list=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 256
    img_size = 256
    img_list = [torch.squeeze(input_tensor).to(device)]
    log_dir = 'line_model/AttUNet_model_total.pth'
    # img_list = [img]
    test_files = [{"img": Img}
                  for Img in img_list]
    keys = ["img"]
    infer_transforms = Compose(
        [
            # LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys, channel_dim='no_channel'),
            Resized(keys=keys, spatial_size=(256, 256), mode='bilinear',
                    align_corners=True),
            ScaleIntensityd(keys=keys),
            # NormalizeIntensityd(keys=keys[:-1]),
        ]
    )
    # print('data load')
    # save_path = 'output/xjt_4'
    # define dataset and dataloader
    test_ds = Dataset(data=test_files, transform=infer_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0, collate_fn=list_data_collate)
    # post_trans = Compose([Activations(softmax=True), AsDiscrete(argmax=True), RemoveSmallObjects(min_size=10)])
    # post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5), RemoveSmallObjects(min_size=60), Flip(spatial_axis=0)])
    # saver = SaveImage(output_dir="./output", output_ext=".nii", output_postfix="seg",
    #                   separate_folder=False)
    # net = SwinUNETR(
    #     img_size=(img_size, img_size),
    #     in_channels=1,
    #     out_channels=2,
    #     feature_size=48,
    #     spatial_dims=2
    # ).to(device)
    net = AttentionUnet(spatial_dims=2,
                        in_channels=1,
                        out_channels=2,
                        channels=(16, 32, 64, 128, 256),
                        strides=(2, 2, 2, 2)).to(device)
    # print('pred')
    # net = UNet(
    #     spatial_dims=2,
    #     in_channels=1,
    #     out_channels=2,
    #     channels=(16, 32, 64, 128, 256),
    #     strides=(2, 2, 2, 2),
    #     num_res_units=2,
    # ).to(device)
    # net = UNETR(
    #     in_channels=1,
    #     out_channels=2,
    #     img_size=(img_size, img_size),
    #     feature_size=16,
    #     spatial_dims=2,
    # ).to(device)
    net.load_state_dict(torch.load(log_dir))
    # net.eval()
    # print("save contour image")
    # count = 0
    with torch.enable_grad():
        for d in test_loader:
            # test_list = [d for d in test_loader]
            # d = next(iter(test_loader))
            image = d["img"].to(device)
            # test_inputs = torch.unsqueeze(image, 1).to(device)
            # define sliding window size and batch size for windows inference
            # image = torch.unsqueeze(image, 1).to(device)
            # print(image.shape)
            # roi_size = (img_size, img_size)
            # sw_batch_size = 4
            # outputs = sliding_window_inference(
            #     image, roi_size, sw_batch_size, net)
            # plt.imshow(image.squeeze().cpu().numpy())
            # plt.show()
            outputs = net(image)
            # print(torch.argmax(outputs, dim=1).shape)
            # labels = labels.to(dtype=torch.long)
            # print(outputs)
            # outputs = torch.argmax(outputs, dim=1)
            # outputs = torch.permute(outputs, (0, 2, 1))
            # plt.imshow(outputs.squeeze().cpu().numpy())
            # plt.show()
            # print(outputs)
            # outputs = torch.unsqueeze(outputs, 1).to(device)
            # outputs = torch.permute(outputs, (0, 1, 3, 2))
            # outputs = [post_trans(i) for i in decollate_batch(outputs)]
            # for output in outputs:
            #     saver(output)
    # test_output = outputs[0]
    # test_output = torch.unsqueeze(test_output, dim=0)
    # print(test_output.shape)
    # plt.imshow(test_output.squeeze().cpu().numpy())
    # plt.show()
    return outputs


# def custom_formatter(metadict, saver):
#     """Returns a kwargs dict for :py:meth:`FolderLayout.filename`,
#     according to the input metadata and SaveImage transform."""
#     subject = getattr(saver, "_data_index", 0)
#     patch_index = metadict.get(monai.utils.ImageMetaKey.PATCH_INDEX, None) if metadict else None
#     return {"subject": f"{subject}", "idx": patch_index}


if __name__ == '__main__':
    # test_method("E:/spinal_navigation/weng_fang_qi/single_vertebra/single_view/DRR/000/contour_img")
    # test_img_path = 'E:/pythonWorkplace/xreg/data/4/remap_drr/drr_remap_013_000.nii'
    test_img_path = 'Data/natong'
    input_list = []
    for root, subdir, files in os.walk(test_img_path):
        for file in files:
            if 'mov' in file:
                img = sitk.ReadImage(os.path.join(root, file))
                img_arr = sitk.GetArrayFromImage(img)
                img_tensor = torch.tensor(img_arr)
                # img_tensor = torch.unsqueeze(img_tensor, dim=0).to('cuda')
                # img_tensor = torch.unsqueeze(img_tensor, dim=0).to('cuda')
                print(img_tensor.shape)
                img_tensor = torch.permute(img_tensor, (1, 0))
                input_list.append(img_tensor)
                # img_tensor = torch.permute(img_tensor, (0, 1, 3, 2))
    infer_method(img_list=input_list)
    # img_path = "multi-DRR2/"
    # img_list = []
    # test_img = sitk.ReadImage(test_img_path)
    # test_arr = sitk.GetArrayFromImage(test_img)
    # img_list.append(test_img)
    # img_arr = np.array(img_list)
    # img_arr = img_arr.flatten()

