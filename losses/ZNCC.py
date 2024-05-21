import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Calculate normalized cross-correlation
def cal_ncc(I, J, eps=1e-10):
    # Compute local sums via convolution

    B, C, _, _ = I.shape
    I = I.reshape(B, C, -1)
    J = J.reshape(B, C, -1)
    I = I - I.mean(dim=-1, keepdim=True)
    J = J - J.mean(dim=-1, keepdim=True)
    # cross = (I - I.mean(dim=-1,keepdim=True)) * (J -  J.mean(dim=-1,keepdim=True))

    # cc = torch.sum(cross) / torch.sum(torch.sqrt(I_var * J_var + eps))
    cc = torch.sum(I * J, dim=-1) / (
                eps + torch.sqrt(torch.sum(I ** 2, dim=-1)) * torch.sqrt(torch.sum(J ** 2, dim=-1)))
    # cc = torch.clamp(cc, -1., 1.)
    # test = torch.mean(cc)
    return torch.mean(cc)


def ncc(I, J, device='cuda', win=None, eps=1e-10):
    return 1 - cal_ncc(I, J, eps)


# Gradient-NCC loss
def gradncc(I, J, mask=None, device='cuda', win=None, eps=1e-10):
    # Compute filters
    with torch.no_grad():
        kernel_X = torch.Tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]])
        kernel_X = torch.nn.Parameter(kernel_X, requires_grad=False)
        kernel_Y = torch.Tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]])
        kernel_Y = torch.nn.Parameter(kernel_Y, requires_grad=False)
        SobelX = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        SobelX.weight = kernel_X
        SobelY = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        SobelY.weight = kernel_Y

        SobelX = SobelX.to(device)
        SobelY = SobelY.to(device)

    Ix = SobelX(I)
    Iy = SobelY(I)
    Jx = SobelX(J)
    Jy = SobelY(J)
    Jx = Jx * mask
    Jy = Jy * mask
    # plt.subplot(1, 4, 1)
    # plt.imshow(Ix[0, :].squeeze().detach().cpu().numpy())
    # plt.subplot(1, 4, 2)
    # plt.imshow(Jx[0, :].squeeze().detach().cpu().numpy())
    # plt.subplot(1, 4, 3)
    # plt.imshow(Iy[0, :].squeeze().detach().cpu().numpy())
    # plt.subplot(1, 4, 4)
    # plt.imshow(Jy[0, :].squeeze().detach().cpu().numpy())
    # plt.show()
    return 1 - 0.5 * cal_ncc(Ix, Jx, eps) - 0.5 * cal_ncc(Iy, Jy, eps)
