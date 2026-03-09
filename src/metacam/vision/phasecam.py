import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from torchvision.io import read_image
import torchvision.transforms.functional as TF

from metacam.data.io import load_mat_file
from metacam.ops.torch_ops import torch_pad_center


def _maybe_empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_and_process_speckle_image(root_path, filename, varname, device=torch.device("cpu")):
    data = load_mat_file(root_path / filename)  # Loading .mat file
    speckle_img_mean = data[varname]
    speckle_img_mean = torch.tensor(speckle_img_mean).to(device=device).to(torch.float32).unsqueeze(0).unsqueeze(0)
    spk_real_amp = speckle_img_mean**0.5  # Apply square root
    return spk_real_amp


def plot_tensor_image(img_tensor, subplot_params=None, title="Speckle Image", cmap="gray", down=1):
    """Plot a tensor image. Create a subplot if subplot_params is provided."""

    # Check if subplot parameters are given
    if subplot_params:
        rows, cols, subplot_index = subplot_params
        plt.subplot(rows, cols, subplot_index)  # Create a subplot
    else:
        plt.figure(figsize=(3, 3))  # Create a new figure if no subplot parameters

    if down != 1:
        img_tensor = F.interpolate(img_tensor, scale_factor=1 / down, mode="area", antialias=False)
    
    # Plot the image
    plt.imshow(img_tensor[0, 0, :, :].cpu().detach().numpy(), cmap=cmap)
    plt.colorbar(), plt.title(title), plt.tight_layout()

def plot_tensor_image_dl(img_tensor, subplot_params=None, title=None, cmap="gray", down=1):
    """Plot a tensor image. Create a subplot if subplot_params is provided."""

    # Check if subplot parameters are given
    if subplot_params:
        rows, cols, subplot_index = subplot_params
        plt.subplot(rows, cols, subplot_index)  # Create a subplot
    else:
        plt.figure(figsize=(3, 3))  # Create a new figure if no subplot parameters

    img_tensor = F.interpolate(img_tensor, scale_factor=1 / down)
    # Plot the image
    plt.imshow(img_tensor[0, 0, :, :].cpu().detach().numpy(), cmap=cmap,vmax=0.2,vmin=0)
    # plt.colorbar()
    plt.title(title), plt.tight_layout()


def plot_images(image_tensor, titles=None, cmap="gray", figsize=(10, 5)):
    # Get the number of images from the first dimension
    num_images = image_tensor.shape[0]  # N

    # Determine the number of rows and columns based on the number of images
    cols = int(np.ceil(np.sqrt(num_images)))  # Number of columns
    rows = int(np.ceil(num_images / cols))  # Number of rows

    # Create the figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)  # Adjust figsize as needed
    if num_images == 1:
        axes = np.array([axes])    
    axes = axes.flatten()  # Flatten in case it's a 2D grid

    for i in range(num_images):
        image = image_tensor[i, 0, :, :].cpu().detach().numpy()  # Convert to numpy array for plotting
        axes[i].imshow(image, cmap=cmap)
        axes[i].axis("off")  # Hide axes for cleaner visualization

        if titles is not None:
            axes[i].set_title(titles[i])

    # Hide any remaining axes if num_images is less than rows * cols
    for j in range(num_images, len(axes)):
        axes[j].axis("off")  # Hide unused axes

    plt.tight_layout()
    plt.show()



class ASM:
    def __init__(self, kernel, device):
        self.ASM_kernel = kernel.to(device)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = torch.fft.ifft2(torch.fft.fft2(x) * self.ASM_kernel)
        _maybe_empty_cuda_cache()
        return x


class SASM:
    def __init__(self, dH, Q1, device):
        self.dH = dH.to(device)
        self.Q1 = Q1.to(device)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        _maybe_empty_cuda_cache()

        x = torch.fft.fft2(x)
        x = x * self.dH
        x = torch.fft.ifft2(x)
        # _maybe_empty_cuda_cache()

        x = x * self.Q1
        x = torch.fft.ifftshift(x)
        x = torch.fft.fft2(x)
        x = torch.fft.fftshift(x)
        _maybe_empty_cuda_cache()

        return x

def img_to_complexobj(img, sim_N):
    img -= img.min()
    img = img / img.max()

    ampimg = img * 0.5 + 0.5
    ampimg = torch_pad_center(ampimg, (sim_N, sim_N), padval=1)

    phimg = img
    phimg = torch_pad_center(phimg, (sim_N, sim_N), padval=1)

    U_obj = ampimg * torch.exp(1j * phimg)
    U_obj[0, 0, sim_N // 2 : sim_N // 2 + 1, sim_N // 2 : sim_N // 2 + 5] = 1 + 0.5j
    U_obj[0, 0, sim_N // 2 + 2 : sim_N // 2 + 3, sim_N // 2 : sim_N // 2 + 5] = 1 + 0.5j
    U_obj[0, 0, sim_N // 2 + 4 : sim_N // 2 + 5, sim_N // 2 : sim_N // 2 + 5] = 1 + 0.5j
    U_obj[0, 0, sim_N // 2 : sim_N // 2 + 5, sim_N // 2 - 3 : sim_N // 2 - 2] = 1 + 0.5j
    U_obj[0, 0, sim_N // 2 : sim_N // 2 + 5, sim_N // 2 - 5 : sim_N // 2 - 4] = 1 + 0.5j
    U_obj[0, 0, sim_N // 2 : sim_N // 2 + 5, sim_N // 2 - 7 : sim_N // 2 - 6] = 1 + 0.5j

    return U_obj
