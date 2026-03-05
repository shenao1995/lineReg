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
    RemoveSmallObjects,
    Resize
)
import numpy as np
import SimpleITK as sitk


def infer_method(model, input_tensor=None, input_list=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if input_tensor is not None:
        img_list = [torch.squeeze(input_tensor).to(device)]
    else:
        img_list = input_list
    test_files = [{"img": Img}
                  for Img in img_list]
    keys = ["img"]
    infer_transforms = Compose(
        [
            EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
            Resized(keys=keys, spatial_size=(256, 256), mode='bilinear',
                    align_corners=True),
            ScaleIntensityd(keys=keys),
            # NormalizeIntensityd(keys=keys[:-1]),
        ]
    )
    test_ds = Dataset(data=test_files, transform=infer_transforms)
    test_loader = DataLoader(test_ds, batch_size=2, num_workers=0, collate_fn=list_data_collate)
    post_trans = Compose([Activations(sigmoid=True),
                          AsDiscrete(threshold=0.5),
                          RemoveSmallObjects(min_size=1)])
    # saver = SaveImage(output_dir="./output", output_ext=".nii", output_postfix="seg",
    #                   separate_folder=False)
    with torch.no_grad():
        for d in test_loader:
            image = d["img"].to(device)
            # print(image.shape)
            outputs = model(image)
            outputs = [post_trans(i) for i in decollate_batch(outputs)]
    resize = Resize(spatial_size=img_list[0].shape, mode='bilinear')
    ap_line = resize(outputs[0])
    # resize = Resize(spatial_size=img_list[1].shape, mode='bilinear')
    # la_line = resize(outputs[1])
    # return ap_line, la_line
    return ap_line
