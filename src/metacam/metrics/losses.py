import math
import numpy as np
import torch
import torch.nn.functional as F


def ssim(img1, img2, window_size=11, window_sigma=1.5, data_range=1.0, k1=0.01, k2=0.03):
    # Compute the mean and variance of the input images
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)
    sigma1 = F.avg_pool2d(img1**2, window_size, stride=1, padding=window_size // 2) - mu1**2
    sigma2 = F.avg_pool2d(img2**2, window_size, stride=1, padding=window_size // 2) - mu2**2
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1 * mu2

    # Compute the SSIM
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2)
    ssim_map = numerator / denominator

    # Average the SSIM map
    ssim_val = ssim_map.mean()
    return ssim_val 


def PSNR(img1, img2):
    mse = torch.mean((img1 - img2) ** 2).detach().cpu()
    if mse == 0:
        return float("inf")

    return 20 * math.log10(2 * math.pi) / torch.sqrt(mse)


def Tv_loss(input_phase, norm: bool = True):
    Tv_loss = torch.sum(
        torch.sqrt(
            (torch.pow(input_phase[:, :, 1:, :-1] - input_phase[:, :, 1:, 1:], 2))
            + (torch.pow(input_phase[:, :, :-1, 1:] - input_phase[:, :, 1:, 1:], 2))
        )
    )

    if norm:
        Tv_loss = Tv_loss / np.prod(np.array(input_phase.shape))

    return Tv_loss


# def tv_loss(img, norm=False):
#     # tensordim = torch.tensor(img.shape, requires_grad=False).to(device)
#     # norm = torch.prod(tensordim)
#     Npx = np.prod(np.array(img.shape))

#     if norm:
#         img = img / torch.max(img)

#     tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
#     tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()

#     return (tv_h + tv_w) / Npx


# def tv_loss(img, norm=False, order=1):
#     tv_h = (torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]) ** order).sum()
#     tv_w = (torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]) ** order).sum()

#     if norm:
#         tv = (tv_h + tv_w) / img.size(2) / img.size(3) / torch.max(img.abs().detach())

#     tv = (tv_h + tv_w) / img.size(2) / img.size(3)  # / torch.max(img.detach())
#     return tv


# def tv_loss(img, norm=False, order=1):
#     dh = img[:, :, 1:, 1:] - img[:, :, :-1, 1:]
#     dw = img[:, :, 1:, 1:] - img[:, :, 1:, :-1]

#     tv = (torch.norm(dh.abs() + dw.abs(), p=order)) / img.size(2) / img.size(3)

#     if norm:
#         tv = tv / (img.abs().detach().mean())

#     return tv


def tv_loss(img, norm=False, order=1):
    dh = img[:, :, 1:, :] - img[:, :, :-1, :]
    dw = img[:, :, :, 1:] - img[:, :, :, :-1]

    tv = (torch.norm(dh.abs(), p=order) + torch.norm(dw.abs(), p=order)) / img.size(2) / img.size(3)

    if norm:
        tv = tv / (img.abs().detach().mean())

    return tv


def huber_penalty(img, tune):
    g_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]) ** 2
    g_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]) ** 2

    # g_h와 g_w의 작은 쪽에 맞추기
    min_dim = min(g_h.size(3), g_w.size(3))
    g_h = g_h[:, :, :, :min_dim]
    g_w = g_w[:, :, :, :min_dim]

    min_dim = min(g_h.size(2), g_w.size(2))
    g_h = g_h[:, :, :min_dim, :]
    g_w = g_w[:, :, :min_dim, :]

    hg = torch.sum(torch.sqrt(0j + 1 + (g_h + g_w) / tune**2).abs() - 1)

    return hg / img.size(2) / img.size(3)


def tv_spk_loss(img, order=1):
    dh = torch.abs(img[:, :, 1:, 1:] - img[:, :, :-1, 1:])
    dw = torch.abs(img[:, :, 1:, 1:] - img[:, :, 1:, :-1])

    # dh = TF.gaussian_blur(dh.abs(), kernel_size=(3, 3))
    # dw = TF.gaussian_blur(dw.abs(), kernel_size=(3, 3))
    # dh = dh.abs()
    # dw = dw.abs()

    threshold = dh.mean() + 3 * dh.std()
    dh[dh > threshold] = dh[dh > threshold] * 0.1

    threshold = dw.mean() + 3 * dw.std()
    dw[dw > threshold] = dw[dw > threshold] * 0.1

    tv = (torch.norm(dh + dw, p=order)) / img.size(2) / img.size(3)
    # tv = torch.norm(dh+dw, p=order) / img.size(2) / img.size(3)

    return tv
