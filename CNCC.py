from diffdrr.metrics import NormalizedCrossCorrelation2d
import numpy as np
import torch
import matplotlib.pyplot as plt


class CannyCrossCorrelation2d(NormalizedCrossCorrelation2d):
    """Compute Normalized Cross Correlation between the image edges of two batches of images."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.canny = Canny()
        # canny_img = conv2d(input_img)

    def forward(self, x1, x2):
        # plt.subplot(1, 2, 1)
        # plt.imshow(self.canny(x1).squeeze().detach().cpu().numpy())
        # plt.colorbar()
        # plt.subplot(1, 2, 2)
        # plt.imshow(self.canny(x2).squeeze().detach().cpu().numpy())
        # plt.colorbar()
        # plt.show()
        return super().forward(self.canny(x1), self.canny(x2))


class Canny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        kernel = np.array([[-1, -1, -1],  # 边缘检测
                           [-1, 8, -1],
                           [-1, -1, -1]])
        self.filter = torch.nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=kernel.shape, padding=1, stride=(1, 1), device='cuda:0')
        kernel = torch.from_numpy(kernel.astype(np.float32)).reshape((1, 1, kernel.shape[0], kernel.shape[1])).to('cuda:0')
        self.filter.weight.data = kernel

    def forward(self, img):
        x = self.filter(img)
        return x