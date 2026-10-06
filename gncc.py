"""Sobel gradient NCC, with statistics evaluated only inside an optional ROI."""

import torch
import torch.nn.functional as F


class GradientNormalizedCrossCorrelation2d(torch.nn.Module):
    """Return mean signed NCC of horizontal/vertical gradients (not raw NCC).

    Masking is applied AFTER differentiation and to both images' statistics,
    avoiding artificial edges at an ROI boundary. Constant gradients score 0.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.register_buffer("kernel", torch.tensor([
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
        ]).unsqueeze(1) / 8)

    def gradients(self, image):
        if image.ndim != 4 or image.shape[1] != 1 or min(image.shape[-2:]) < 3:
            raise ValueError("GNCC expects B x 1 x H x W grayscale images, H/W >= 3")
        return F.conv2d(F.pad(image, (1, 1, 1, 1), mode="reflect"),
                        self.kernel.to(image))

    def from_gradients(self, a, b, mask=None):
        if a.shape != b.shape:
            raise ValueError("GNCC images must have matching shapes")
        weights = torch.ones_like(a[:, :1]) if mask is None else mask.to(a)
        if weights.ndim != 4 or weights.shape[1:] != (1, *a.shape[-2:]):
            raise ValueError("Mask must be B x 1 x H x W")
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Mask weights must be finite and nonnegative")
        count = weights.sum((-2, -1), keepdim=True)
        if (count < 2).any():
            raise ValueError("GNCC ROI must contain at least two pixels")
        a = a - (a * weights).sum((-2, -1), keepdim=True) / count
        b = b - (b * weights).sum((-2, -1), keepdim=True) / count
        cov = (weights * a * b).sum((-2, -1)) / count.squeeze(-1).squeeze(-1)
        va = (weights * a.square()).sum((-2, -1)) / count.squeeze(-1).squeeze(-1)
        vb = (weights * b.square()).sum((-2, -1)) / count.squeeze(-1).squeeze(-1)
        score = cov / (va * vb).clamp_min(self.eps ** 2).sqrt()
        return score.clamp(-1, 1).mean(1)

    def forward(self, a, b, mask=None):
        return self.from_gradients(self.gradients(a), self.gradients(b), mask)
