import torch
from torch.nn.functional import pad, conv2d, interpolate
import numpy as np
import math
import torch.fft as fft
import torchvision.transforms.functional as TF


# def torch_crop_center(H, dim, shift=(0, 0)):
#     batch, channel, Nh, Nw = H.size()

#     return H[:, :, (Nh - dim) // 2 + shift[0] : (Nh + dim) // 2 + shift[0], (Nw - dim) // 2 + shift[1] : (Nw + dim) // 2 + shift[1]]


# def torch_crop_center_rect(H, dim, shift=(0, 0)):
#     batch, channel, Nh, Nw = H.size()

#     return H[:, :, (Nh - dim[0]) // 2 + shift[0] : (Nh + dim[0]) // 2 + shift[0], (Nw - dim[1]) // 2 + shift[1] : (Nw + dim[1]) // 2 + shift[1]]


def torch_crop_center(H, dim, shift=(0, 0)):
    """
    Crops the center region of a tensor. Supports both square and rectangular crops.

    Args:
        H (torch.Tensor): Input tensor of shape (batch, channel, height, width).
        dim (int or tuple): Size of the crop. If int, a square crop is performed. If tuple, it specifies (height, width).
        shift (tuple): Optional shift (y, x) to apply to the center position.

    Returns:
        torch.Tensor: Cropped tensor.
    """

    batch, channel, Nh, Nw = H.size()

    if isinstance(dim, tuple):
        dim_h, dim_w = dim  # Rectangular crop
    else:
        dim_h, dim_w = dim, dim  # Square crop

    return H[:, :, (Nh - dim_h) // 2 + shift[0] : (Nh + dim_h) // 2 + shift[0], (Nw - dim_w) // 2 + shift[1] : (Nw + dim_w) // 2 + shift[1]]


def torch_pad_center(H, pdim, xy: bool = False, padval=0):
    if xy:
        pxdim = pdim[0]
        pydim = pdim[1]
    else:
        pxdim = pdim[1]
        pydim = pdim[0]

    oxdim = H.shape[-1]
    oydim = H.shape[-2]

    oxcen = np.fix(oxdim / 2) + 1
    pxcen = np.fix(pxdim / 2) + 1
    lpx = pxcen - oxcen
    rpx = (pxdim - oxdim) - lpx

    oycen = np.fix(oydim / 2) + 1
    pycen = np.fix(pydim / 2) + 1
    lpy = pycen - oycen
    rpy = (pydim - oydim) - lpy

    # padH = pad(H, pad=(int(lpx), int(rpx), int(lpy), int(rpy)), value=padval)
    return pad(H, pad=(int(lpx), int(rpx), int(lpy), int(rpy)), value=padval)

    # h = torch.ones(2, 2).to(device)
    #
    # hpad = torch_pad_center(h, (3, 3))
    # hpad = hpad.cpu().detach().numpy()
    # plt.figure(figsize=(15, 5))
    # plt.subplot(1,1,1)
    # plt.imshow(hpad, cmap='gray')
    # plt.show()


def NA_to_Circ(wavelength, pixel_size, N, NA):
    kmax = np.pi / pixel_size

    # kx_list = kmax * np.arange(-1, 1, 1/((N - 1) / 2))
    kx_list = kmax * np.linspace(-1, 1, N)
    # print(kx_list.shape)
    # ky_list = kmax * np.arange(-1, 1, 1/((N - 1) / 2))

    ky_list = kmax * np.linspace(-1, 1, N)

    [kx_grid, ky_grid] = np.meshgrid(kx_list, ky_list, indexing="xy")

    cutoff = 2 * np.pi * NA / wavelength

    NA_circle = np.float32((kx_grid**2 + ky_grid**2) < cutoff**2)
    NA_circle = torch.from_numpy(NA_circle)

    return NA_circle


def Circ(pixel_size, r, N):
    rmax = N * pixel_size / 2

    kx_list = rmax * np.linspace(-1, 1, N)
    print(kx_list.shape)

    ky_list = rmax * np.linspace(-1, 1, N)

    [kx_grid, ky_grid] = np.meshgrid(kx_list, ky_list, indexing="xy")

    NA_circle = np.float32((kx_grid**2 + ky_grid**2) < r**2)
    NA_circle = torch.from_numpy(NA_circle)

    return NA_circle


def return2DGaussian(resolution, sigma, offset=0, device="cpu"):
    kernel_size = resolution

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size).to(device)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = ((kernel_size - 1) + offset) / 2.0
    variance = sigma**2.0

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(-torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance))
    # # Make sure sum of values in gaussian kernel equals 1.
    # gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.max(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    # Reshape to 2d depthwise convolutional weight
    return gaussian_kernel.view(1, 1, kernel_size, kernel_size)


# 241129_memory opt
def normxcorr2_fft(tensor1, tensor2, norm=False):
    # 입력 텐서의 크기 가져오기
    batch_size, channels, height1, width1 = tensor1.size()
    _, _, height2, width2 = tensor2.size()

    # 두 텐서의 높이와 너비의 최대값 사용
    max_height = max(height1, height2)
    max_width = max(width1, width2)

    # Compute means
    mean_t1 = torch.mean(tensor1, dim=(2, 3), keepdim=True)
    mean_t2 = torch.mean(tensor2, dim=(2, 3), keepdim=True)

    # Compute standard deviations
    sigma_t1 = torch.std(tensor1, dim=(2, 3), keepdim=True)
    sigma_t2 = torch.std(tensor2, dim=(2, 3), keepdim=True)

    # tensor2를 tensor1과 같은 크기로 패딩
    tensor1 = torch_pad_center(tensor1, (max_width, max_height), padval=torch.mean(mean_t1.abs()).item())
    tensor2 = torch_pad_center(tensor2, (max_width, max_height), padval=torch.mean(mean_t2.abs()).item())

    # 푸리에 변환
    fft1 = fft.fft2((tensor1 - mean_t1) / sigma_t1, dim=(-2, -1))
    fft2 = fft.fft2((tensor2 - mean_t2) / sigma_t2, dim=(-2, -1))

    # 복소 곱
    temp = fft1 * torch.conj(fft2)

    del fft1, fft2

    # 역 푸리에 변환
    temp = fft.ifft2(temp, dim=(-2, -1))

    # fftshift를 사용하여 결과를 이동시킴
    temp = fft.fftshift(temp, dim=(-2, -1))

    # 실수 부분만 추출하여 크로스 상관을 얻음
    temp = temp.abs()

    # 정규화하여 결과를 [0, 1] 사이로 조정
    temp = temp / (min(height1, height2) * min(width1, width2))

    return temp


# 240709
# def normxcorr2_fft(tensor1, tensor2, norm=False):
#     # 입력 텐서의 크기 가져오기
#     batch_size, channels, height1, width1 = tensor1.size()
#     _, _, height2, width2 = tensor2.size()

#     # 두 텐서의 높이와 너비의 최대값 사용
#     max_height = max(height1, height2)
#     max_width = max(width1, width2)

#     if norm:
#         # power normlaization
#         tensor1 = tensor1 / torch.sum(tensor1, dim=(2, 3), keepdim=True)
#         tensor2 = tensor2 / torch.sum(tensor2, dim=(2, 3), keepdim=True)

#     # Compute means
#     mean_t1 = torch.mean(tensor1, dim=(2, 3), keepdim=True)
#     mean_t2 = torch.mean(tensor2, dim=(2, 3), keepdim=True)

#     # Compute standard deviations
#     sigma_t1 = torch.std(tensor1, dim=(2, 3), keepdim=True)
#     sigma_t2 = torch.std(tensor2, dim=(2, 3), keepdim=True)

#     # tensor2를 tensor1과 같은 크기로 패딩
#     padded_tensor1 = torch_pad_center(tensor1, (max_width, max_height), padval=torch.mean(mean_t1.abs()).item())
#     padded_tensor2 = torch_pad_center(tensor2, (max_width, max_height), padval=torch.mean(mean_t2.abs()).item())

#     # 푸리에 변환
#     fft1 = fft.fft2((padded_tensor1 - mean_t1) / sigma_t1, dim=(-2, -1))
#     fft2 = fft.fft2((padded_tensor2 - mean_t2) / sigma_t2, dim=(-2, -1))

#     # 복소 곱
#     fft_product = fft1 * torch.conj(fft2)

#     # 역 푸리에 변환
#     correlation = fft.ifft2(fft_product, dim=(-2, -1))

#     # fftshift를 사용하여 결과를 이동시킴
#     correlation = fft.fftshift(correlation, dim=(-2, -1))

#     # 실수 부분만 추출하여 크로스 상관을 얻음
#     R = correlation.abs()

#     # 정규화하여 결과를 [0, 1] 사이로 조정
#     NCC = R / (min(height1, height2) * min(width1, width2))

#     return NCC


# 240905
def xcorr2_fft(tensor1, tensor2, norm=True):
    # 입력 텐서의 크기 가져오기
    batch_size, channels, height1, width1 = tensor1.size()
    _, _, height2, width2 = tensor2.size()

    # 두 텐서의 높이와 너비의 최대값 사용
    max_height = max(height1, height2)
    max_width = max(width1, width2)

    # tensor2를 tensor1과 같은 크기로 패딩
    padded_tensor1 = torch_pad_center(tensor1, (max_width, max_height))
    padded_tensor2 = torch_pad_center(tensor2, (max_width, max_height))

    # 푸리에 변환
    fft1 = fft.fft2(padded_tensor1, dim=(2, 3))
    fft2 = fft.fft2(padded_tensor2, dim=(2, 3))

    # 복소 곱
    fft_product = fft1 * torch.conj(fft2)

    # 역 푸리에 변환
    correlation = fft.ifft2(fft_product, dim=(2, 3))

    # fftshift를 사용하여 결과를 이동시킴
    correlation = fft.fftshift(correlation)

    # 실수 부분만 추출하여 크로스 상관을 얻음
    R = correlation.real

    # 정규화하여 결과를 [0, 1] 사이로 조정
    NCC = R / (min(height1, height2) * min(width1, width2))

    if norm:
        # NCC = (NCC - torch.mean(NCC.detach(), dim=(2, 3), keepdim=True)) / torch.std(NCC.detach(), dim=(2, 3), keepdim=True)
        NCC = NCC / torch.mean(NCC, dim=(2, 3), keepdim=True)

    return NCC


# # 240625
# def normxcorr2_fft(tensor1, tensor2):
#     # 입력 텐서의 크기 가져오기
#     batch_size, channels, height1, width1 = tensor1.size()
#     _, _, height2, width2 = tensor2.size()

#     # 두 텐서의 높이와 너비의 최대값 사용
#     max_height = max(height1, height2)
#     max_width = max(width1, width2)

#     # power normlaization
#     tensor1 = tensor1 / torch.sum(tensor1, dim=(2, 3), keepdim=True)
#     tensor2 = tensor2 / torch.sum(tensor2, dim=(2, 3), keepdim=True)

#     # Compute means
#     mean_t1 = torch.mean(tensor1, dim=(2, 3), keepdim=True)
#     mean_t2 = torch.mean(tensor2, dim=(2, 3), keepdim=True)

#     # Compute standard deviations
#     sigma_t1 = torch.std(tensor1, dim=(2, 3), keepdim=True)
#     sigma_t2 = torch.std(tensor2, dim=(2, 3), keepdim=True)

#     # tensor2를 tensor1과 같은 크기로 패딩
#     padded_tensor1 = torch_pad_center(
#         tensor1, (max_width, max_height), padval=torch.mean(mean_t1).item()
#     )
#     padded_tensor2 = torch_pad_center(
#         tensor2, (max_width, max_height), padval=torch.mean(mean_t2).item()
#     )

#     # 푸리에 변환
#     fft1 = fft.fft2((padded_tensor1 - mean_t1) / sigma_t1, dim=(-2, -1))
#     fft2 = fft.fft2((padded_tensor2 - mean_t2) / sigma_t2, dim=(-2, -1))

#     # 복소 곱
#     fft_product = fft1 * torch.conj(fft2)

#     # 역 푸리에 변환
#     correlation = fft.ifft2(fft_product, dim=(-2, -1))

#     # fftshift를 사용하여 결과를 이동시킴
#     correlation = fft.fftshift(correlation)

#     # 실수 부분만 추출하여 크로스 상관을 얻음
#     R = correlation.real

#     # 정규화하여 결과를 [0, 1] 사이로 조정
#     NCC = R / (min(height1, height2) * min(width1, width2))

#     return NCC


# # No spatial padding, upsample at fourier domain
# def normxcorr2_fft(tensor1, tensor2):

#     # 입력 텐서의 크기 가져오기
#     batch_size, channels, height1, width1 = tensor1.size()
#     _, _, height2, width2 = tensor2.size()

#     # 두 텐서의 높이와 너비의 최대값 사용
#     max_height = max(height1, height2)
#     max_width = max(width1, width2)

#     # power normalization
#     tensor1 = tensor1 / torch.sum(tensor1, dim=(2, 3), keepdim=True)
#     tensor2 = tensor2 / torch.sum(tensor2, dim=(2, 3), keepdim=True)

#     # Compute means
#     mean_t1 = torch.mean(tensor1, dim=(2, 3), keepdim=True)
#     mean_t2 = torch.mean(tensor2, dim=(2, 3), keepdim=True)

#     # Compute standard deviations
#     sigma_t1 = torch.std(tensor1, dim=(2, 3), keepdim=True)
#     sigma_t2 = torch.std(tensor2, dim=(2, 3), keepdim=True)

#     # 푸리에 변환
#     fft1 = fft.fft2((tensor1 - mean_t1) / sigma_t1, dim=(-2, -1))
#     fft2 = fft.fft2((tensor2 - mean_t2) / sigma_t2, dim=(-2, -1))

#     fft1 = interpolate(
#         fft1,
#         size=[max_height, max_width],
#         # mode="bicubic",
#         # antialias=False,
#     )

#     fft2 = interpolate(
#         fft2,
#         size=[max_height, max_width],
#         # mode="bicubic",
#         # antialias=False,
#     )

#     # 복소 곱
#     fft_product = fft1 * torch.conj(fft2)

#     # 역 푸리에 변환
#     correlation = fft.ifft2(fft_product, dim=(-2, -1))

#     # fftshift를 사용하여 결과를 이동시킴
#     correlation = fft.fftshift(correlation)

#     # 실수 부분만 추출하여 크로스 상관을 얻음
#     R = correlation.real

#     # 정규화하여 결과를 [0, 1] 사이로 조정
#     NCC = R / (min(height1, height2) * min(width1, width2))

#     return NCC


def xcorr2(t_large, t_small):
    # Get the size of the template

    batch_size, channels, Hl, Wl = t_large.size()
    batch_size, channels, Hs, Ws = t_small.size()

    t_large = torch_pad_center(
        t_large,
        (Wl + round(Ws / 2), Hl + round(Hs / 2)),
        padval=torch.finfo(torch.float32).eps,
    )

    # Calculate the mean and standard deviation of the template
    t_small_mean = torch.mean(t_small)
    t_small_std = torch.std(t_small)

    t_large_mean = torch.mean(t_large)
    t_large_std = torch.std(t_large)

    # Normalize the template
    t_small = (t_small - t_small_mean) / t_small_std
    t_large = (t_large - t_large_mean) / t_large_std

    # Perform the correlation
    correlation = conv2d(t_large, t_small)

    return correlation


def fftconv(a1, a2):
    r = fft.fftshift(fft.ifft2(fft.fft2(a1) * fft.fft2(a2))).real
    return r



# 추후 채널 차원들을 배치에 이어붙여서 병렬처리가 될 수 없는지 확인
def normxcorr2(template, image, mode="full"):
    """
    Compute the normalized cross-correlation between a template and an image using PyTorch.

    Args:
    - template: N-D tensor, the template or filter used for cross-correlation.
    Must be of equal or lesser dimensions than the image.
    Length of each dimension must be less than or equal to the corresponding dimension of the image.
    - image: N-D tensor
    - mode: str, "full", "valid", or "same"
        "full" (default): The output is the full discrete linear convolution of the inputs.
        "valid": The output consists only of those elements that do not rely on the zero-padding.
        "same": The output is the same size as image, centered with respect to the 'full' output.

    Returns:
    - out: N-D tensor of the same dimensions as the image. Size depends on the mode parameter.
    """

    # Check dimensions
    if any(template.shape[i] > image.shape[i] for i in range(template.ndim)):
        raise ValueError("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    batch_size, channels, Hl, Wl = image.size()
    batch_size, channels, Hs, Ws = template.size()

    # Ensure inputs are float tensors
    template = template.to(torch.float32)
    image = image.to(torch.float32)

    mean_template = torch.mean(template, dim=(2, 3), keepdim=True)
    mean_image = torch.mean(image, dim=(2, 3), keepdim=True)

    image = torch_pad_center(
        image,
        (Wl + round(Ws / 2), Hl + round(Hs / 2)),
        padval=torch.finfo(torch.float32).eps,
    )

    # Subtract mean
    template = template - mean_template
    image = image - mean_image

    # Create an array of ones with the same shape as the template
    a1 = torch.ones_like(template)

    # Compute the cross-correlation using convolution
    out = conv2d(image, torch.conj(template), padding=0)

    # Compute the local sum of squares
    image_square = conv2d(image**2, a1, padding=0)
    local_sum_sq = image_square - conv2d(image, a1, padding=0) ** 2 / (Hs * Ws)

    # Remove small machine precision errors
    local_sum_sq = torch.max(local_sum_sq, torch.tensor(0.0))

    # Sum of squares of the template
    # template_sum_sq = torch.sum(template**2)
    template_sum_sq = torch.sum(template**2, dim=(2, 3), keepdim=True)

    # Compute normalized cross-correlation
    out = out / torch.sqrt(local_sum_sq * template_sum_sq)

    # Handle divisions by zero or near-zero
    out[~torch.isfinite(out)] = 0

    # Adjust output size according to the mode
    if mode == "same":
        # Calculate padding to maintain the same dimensions
        h_pad = template.shape[-2] // 2
        w_pad = template.shape[-1] // 2
        # Pad the output tensor to maintain the same dimensions
        out = pad(out, (w_pad, w_pad, h_pad, h_pad))

    elif mode == "valid":
        h_pad = template.shape[-2] - 1
        w_pad = template.shape[-1] - 1
        # Crop the output tensor to maintain the valid dimensions
        out = out[:, :, h_pad:-h_pad, w_pad:-w_pad]

    return out


# def normxcorr2(t_large, t_small):
#     # Get the size of the template

#     batch_size, channels, Hl, Wl = t_large.size()
#     batch_size, channels, Hs, Ws = t_small.size()

#     t_large = torch_pad_center(
#         t_large,
#         (Wl + round(Ws / 2), Hl + round(Hs / 2)),
#         padval=torch.finfo(torch.float32).eps,
#     )

#     # Calculate the mean and standard deviation of the template
#     t_small_mean = torch.mean(t_small)
#     t_small_std = torch.std(t_small)

#     # Normalize the template
#     t_small = (t_small - t_small_mean) / t_small_std

#     # Perform the correlation
#     correlation = conv2d(t_large, t_small)

#     # Calculate the mean and standard deviation of the image patches
#     t_large_mean = conv2d(t_large, torch.ones_like(t_small) / t_small.numel())
#     t_large_sqr_mean = conv2d(t_large**2, torch.ones_like(t_small) / t_small.numel())
#     t_large_std = torch.sqrt(
#         (t_large_sqr_mean - t_large_mean**2).to(torch.complex64)
#     ).to(torch.float32)

#     # Normalize the correlation result
#     normalized_correlation = correlation / (t_large_std * t_small.numel())

#     # print(torch.min((t_large_std * t_small.numel())))
#     return normalized_correlation


def lee_filter_torch(img, size):
    img_mean = TF.gaussian_blur(img, kernel_size=(size, size), sigma=size / 4)
    img_sqr_mean = TF.gaussian_blur(img**2, kernel_size=(size, size), sigma=size / 4)
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = torch.var(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output
