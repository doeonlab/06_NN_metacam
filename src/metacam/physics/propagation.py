from math import pi

import numpy as np
import torch
from torch.nn.functional import pad


def circ_mask(xdim, ydim, rad):
    [columnsInImage, rowsInImage] = np.meshgrid(np.arange(1, xdim + 1), np.arange(1, ydim + 1))

    # % Next create the circle in the image.
    centerX = np.fix(xdim / 2) + 1
    centerY = np.fix(ydim / 2) + 1
    filter = (rowsInImage - centerY) ** 2 + (columnsInImage - centerX) ** 2 <= rad**2

    return filter


def elip_mask(xdim, ydim, rad):
    device = rad.device

    hX = xdim // 2
    hY = ydim // 2
    xx, yy = torch.meshgrid(
        torch.arange(-hX, hX - (1 - (xdim % 2)) + 1, device=device),
        torch.arange(-hY, hY - (1 - (ydim % 2)) + 1, device=device),
        indexing="xy",
    )
    filter1 = (yy.float()) ** 2 / rad**2 + (xx.float()) ** 2 / hX**2 <= 1
    filter2 = (yy.float()) ** 2 / hY**2 + (xx.float()) ** 2 / rad**2 <= 1
    filter = filter1 * filter2
    return filter


def torch_fft(H):
    H = torch.fft.fftshift(torch.fft.fft2(H), dim=(-2, -1))
    return H


def torch_ifft(H):
    H = torch.fft.ifft2(torch.fft.ifftshift(H, dim=(-2, -1)))
    return H


# 240705
class asm_master_alltorch:
    def __init__(self, sim_fov_target, sim_px, device):
        # Non wavelength dependent ###########################################################
        sim_xdim = sim_fov_target / sim_px

        cenx = np.fix(sim_xdim / 2) + 1
        ceny = np.fix(sim_xdim / 2) + 1

        fov_xend_n = -(cenx - 1) * sim_px
        fov_xend_p = (sim_xdim - cenx) * sim_px

        x_grid = torch.arange(fov_xend_n, fov_xend_p, sim_px)
        sim_xdim = x_grid.shape[0]
        sim_ydim = sim_xdim
        tot_fov_x = sim_xdim * sim_px
        tot_fov_y = sim_ydim * sim_px

        dkx = 2 * pi * 1 / tot_fov_x
        dky = 2 * pi * 1 / tot_fov_y

        # same logic with matlab
        zero_freq_index_x = np.fix(sim_xdim / 2) + 1  # +1 is correct for SASM
        zero_freq_index_y = np.fix(sim_ydim / 2) + 1  # +1 is correct for SASM

        # discrete kx, ky grid matched with fft
        kx_list = dkx * torch.arange(-(zero_freq_index_x - 1), (sim_xdim - zero_freq_index_x) + 1, device=device)
        ky_list = dky * torch.arange(-(zero_freq_index_y - 1), (sim_ydim - zero_freq_index_y) + 1, device=device)

        [kx_grid, ky_grid] = torch.meshgrid(kx_list, ky_list, indexing="xy")
        ########################################################################################

        self.fov = tot_fov_x
        self.px = sim_px
        self.dkx = dkx
        self.dky = dky
        self.xdim = sim_xdim
        self.ydim = sim_ydim
        self.fov_xend_p = fov_xend_p
        self.fov_xend_n = fov_xend_n
        self.zero_freq_idx = zero_freq_index_x
        self.device = device
        self.kx_grid = kx_grid
        self.ky_grid = ky_grid
        self.kx_list = kx_list.reshape(1, 1, 1, self.ydim)
        self.ky_list = ky_list.reshape(1, 1, self.xdim, 1)
        self.scale = 1.0 / (self.xdim * self.ydim)      # 1 / N

    def get_kz(self, wavelength_sim):
        k0 = 2 * pi / wavelength_sim

        # kz = k0**2 - (self.kx_grid**2 + self.ky_grid**2)
        kz = k0**2 - (self.kx_list**2 + self.ky_list**2)  # Broadcasting
        kz = torch.sqrt(0j + kz * (kz > 0) * 1)

        return kz

    def get_kernel(self, wavelength_sim, kz, z, BLfactor=1):
        k_max_aperture = self.BL_ASM_mask(wavelength_sim, z, BLfactor)
        ASM_kernel = (kz.abs() > 0) * torch.exp(1j * kz * z) * k_max_aperture
        ASM_kernel = torch.fft.ifftshift(ASM_kernel)
        return ASM_kernel

    def prop_w_kernel(self, U, ASM_kernel, Prefft=False):
        # U_prop = torch.fft.ifft2(torch.fft.fft2(U) * ASM_kernel) #torch_ifft(torch_fft(U) * ASM_kernel)
        # return U_prop
        if Prefft:
            return torch.fft.ifft2(U * ASM_kernel)
        else:
            return torch.fft.ifft2(torch.fft.fft2(U) * ASM_kernel)
        # return Uout

    def sasm_get_kernel(self, wl, z, BLfactor=0.5):
        k = 2 * pi / wl
        Sfov = wl * z / (self.px)
        fov = self.fov
        kx = self.kx_list
        ky = self.ky_list
        x = kx / self.dkx * self.px
        y = ky / self.dky * self.px

        ax = wl * kx / (2 * pi)
        ay = wl * ky / (2 * pi)
        f1 = 2 * torch.abs(z * (ax / torch.sqrt(0j + (1 - ax**2 - ay**2)) - ax)) <= (BLfactor * fov)
        f2 = 2 * torch.abs(z * (ay / torch.sqrt(0j + (1 - ax**2 - ay**2)) - ay)) <= (BLfactor * fov)
        W = f1 * f2

        pf = 1j * k * z
        delta_H = W * torch.exp(pf * ((torch.sqrt(0j + 1 - (wl**2 * (kx**2 + ky**2)) / (4 * pi**2))) - (1 - wl**2 * (kx**2 + ky**2) / (8 * pi**2))))
        delta_H = torch.fft.ifftshift(delta_H)

        Q_1 = torch.exp(1j * k / (2 * z) * (x**2 + y**2))

        return delta_H, Q_1, Sfov

    def sasm_get_kernel_az(self, wl, z, zf_cvfr, BLfactor=0.5):
        k = 2 * pi / wl
        Sfov = wl * zf_cvfr / (self.px)
        fov = self.fov
        kx = self.kx_list
        ky = self.ky_list

        ax = wl * kx / (2 * pi)
        ay = wl * ky / (2 * pi)
        f1 = 2 * torch.abs(z * (ax / torch.sqrt(0j + (1 - ax**2 - ay**2)) - ax)) <= (BLfactor * fov)
        f2 = 2 * torch.abs(z * (ay / torch.sqrt(0j + (1 - ax**2 - ay**2)) - ay)) <= (BLfactor * fov)
        W = f1 * f2
        del f1, f2

        pf_ASM = 1j * k * z
        pf_cvfr = 1j * k * zf_cvfr
        delta_H = W * torch.exp(
            pf_ASM * (torch.sqrt(0j + 1 - (wl**2 * (kx**2 + ky**2)) / (4 * pi**2))) - pf_cvfr * (1 - wl**2 * (kx**2 + ky**2) / (8 * pi**2))
        )
        delta_H = torch.fft.ifftshift(delta_H)

        x = kx / self.dkx * self.px
        y = ky / self.dky * self.px

        Q_1 = torch.exp(1j * k / (2 * zf_cvfr) * (x**2 + y**2))

        # q_px = Sfov / self.xdim
        # q_y = ky / self.dky * q_px
        # q_x = kx / self.dkx * q_px
        # Q_2 = torch.exp(1j * k * zf_cvfr) / (1j * wl * zf_cvfr) * torch.exp(1j * k / (2 * zf_cvfr) * (q_x**2 + q_y**2))

        return delta_H, Q_1, Sfov

    def sasm_get_2kernel(self, wl, z, BLfactor=0.5):
        k = 2 * pi / wl
        Sfov = wl * z / (self.px)
        fov = self.fov
        kx = self.kx_list
        ky = self.ky_list
        x = kx / self.dkx * self.px
        y = ky / self.dky * self.px

        ax = wl * kx / (2 * pi)
        ay = wl * ky / (2 * pi)
        f1 = 2 * torch.abs(z * (ax / torch.sqrt(0j + (1 - ax**2 - ay**2)) - ax)) <= (BLfactor * fov)
        f2 = 2 * torch.abs(z * (ay / torch.sqrt(0j + (1 - ax**2 - ay**2)) - ay)) <= (BLfactor * fov)
        W = (f1 * f2).to(torch.bool)

        pf = 1j * k * z
        delta_H = W * torch.exp(pf * ((torch.sqrt(0j + 1 - (wl**2 * (kx**2 + ky**2)) / (4 * pi**2))) - (1 - wl**2 * (kx**2 + ky**2) / (8 * pi**2))))

        Q_1 = torch.exp(1j * k / (2 * z) * (x**2 + y**2))

        q_px = Sfov / self.xdim
        q_y = ky / self.dky * q_px
        q_x = kx / self.dkx * q_px
        Q_2 = torch.exp(1j * k * z) / (1j * wl * z) * torch.exp(1j * k / (2 * z) * (q_x**2 + q_y**2))

        delta_H = torch.fft.ifftshift(delta_H)

        return delta_H, Q_1, Q_2, Sfov

    def sasm_get_bandlim(self, wl, z, BLfactor=0.5):
        fov = self.fov
        kx = self.kx_list
        ky = self.ky_list

        ax = wl * kx / (2 * pi)
        ay = wl * ky / (2 * pi)
        # f1 = 2*torch.abs(z * (ax / torch.sqrt(0j + (1 - ax**2 - ay**2)) - ax)) <= (BLfactor * fov)
        # f2 = 2*torch.abs(z * (ay / torch.sqrt(0j + (1 - ax**2 - ay**2)) - ay)) <= (BLfactor * fov)
        # W = f1 * f2
        W = (2 * torch.abs(z * (ax / torch.sqrt(0j + (1 - ax**2 - ay**2)) - ax)) <= (BLfactor * fov)) * (
            2 * torch.abs(z * (ay / torch.sqrt(0j + (1 - ax**2 - ay**2)) - ay)) <= (BLfactor * fov)
        )

        return W

    # def sasm_prop_w_kernel(self, U, delta_H, Q_1, Prefft=False):
    #     if Prefft:
    #         return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(torch.fft.ifft2(U * delta_H) * Q_1)))
    #     else:
    #         return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fft2(U) * delta_H) * Q_1)))
    #     # return Uout

    def sasm_prop_w_kernel(self, U, delta_H, Q_1, Prefft=False):
        # U = torch.fft.fft2(U) 
        U = torch.fft.fft2(U,               norm="ortho")
        U = U * delta_H
        # U = torch.fft.ifft2(U) 
        U = torch.fft.ifft2(U,              norm="ortho")
        U = U * Q_1
        U = torch.fft.ifftshift(U) 
        # U = torch.fft.fft2(U)
        U = torch.fft.fft2(U,               norm="ortho")
        U = torch.fft.fftshift(U) 

        return U

    def sasm_prop_w_2kernel(self, U, delta_H, Q_1, Q_2, Prefft=False):
        if Prefft:
            return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(torch.fft.ifft2(U * delta_H) * Q_1))) * Q_2
        else:
            return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fft2(U) * delta_H) * Q_1))) * Q_2
        # return Uout

    def BL_ASM_mask(self, wavelength_sim, z, BLfactor=1):
        kx = self.kx_list
        ky = self.ky_list
        dkx = self.dkx
        dky = self.dky
        xdim = self.xdim
        ydim = self.ydim

        k0 = 2 * pi / wavelength_sim
        klim = BLfactor * k0 / (torch.sqrt(1 / pi**2 * z**2 * dkx**2 + 1))

        filter = ((ky) ** 2 / klim**2 + (kx) ** 2 / (dkx * xdim / 2) ** 2 <= 1) * ((ky) ** 2 / (dky * ydim / 2) ** 2 + (kx) ** 2 / klim**2 <= 1)
        filter = filter + torch.logical_not(filter) * 1e-6
        return filter

    def BL_NA_mask(self, wavelength_sim, NA):
        kx = self.kx_grid
        ky = self.ky_grid
        dkx = self.dkx
        dky = self.dky
        xdim = self.xdim
        ydim = self.ydim

        k0 = 2 * pi / wavelength_sim
        klim = k0 * NA

        filter = ((ky) ** 2 / klim**2 + (kx) ** 2 / (dkx * xdim / 2) ** 2 <= 1) * ((ky) ** 2 / (dky * ydim / 2) ** 2 + (kx) ** 2 / klim**2 <= 1)
        return filter

    def Circ_NA_mask(self, wavelength_sim, NA):
        kx = self.kx_grid
        ky = self.ky_grid

        k0 = 2 * pi / wavelength_sim
        klim = k0 * NA

        filter = (ky) ** 2 + (kx) ** 2 <= klim**2
        return filter

    def fresnel_get_kernel(self, wl, z, BLfactor=0.5):
        k = 2 * pi / wl
        Sfov = wl * z / (self.px)
        fov = self.fov
        kx = self.kx_list
        ky = self.ky_list
        x = kx / self.dkx * self.px
        y = ky / self.dky * self.px

        Q_1 = torch.exp(1j * k / (2 * z) * (x**2 + y**2))

        q_px = Sfov / self.xdim
        q_y = ky / self.dky * q_px
        q_x = kx / self.dkx * q_px
        Q_2 = torch.exp(1j * k * z) / (1j * wl * z) * torch.exp(1j * k / (2 * z) * (q_x**2 + q_y**2))

        return Q_1, Q_2

    def fresnel_prop_w_kernel(self, U, Q_1, Q_2):
        psi_p_final = torch_fft(U * Q_1) * Q_2
        return psi_p_final


# only for sqaure
def sasm(U, px, wl, z, BLfactor=0.5, UsePhase=True):
    batch, channel, sim_ydim, sim_xdim = U.size()
    device = U.device

    cen = np.fix(sim_xdim / 2) + 1  # this is correct for SASM

    tot_fov_x = sim_xdim * px

    dkx = 2 * pi * 1 / tot_fov_x

    # discrete kx, ky grid matched with fft
    kx_list = dkx * torch.arange(-(cen - 1), (sim_xdim - cen) + 1, device=device)

    kx = kx_list.reshape(1, 1, 1, sim_xdim)
    ky = kx_list.reshape(1, 1, sim_xdim, 1)

    k = 2 * pi / wl
    Sfov = wl * z / (px)
    fov = tot_fov_x
    x = kx / dkx * px
    y = ky / dkx * px

    ax = wl * kx / (2 * pi)
    ay = wl * ky / (2 * pi)
    az2 = ax**2 - ay**2

    f1 = 2 * torch.abs(z * (ax / torch.sqrt(0j + (1 - az2)) - ax)) <= (BLfactor * fov)
    f2 = 2 * torch.abs(z * (ay / torch.sqrt(0j + (1 - az2)) - ay)) <= (BLfactor * fov)
    W = (f1 * f2).to(torch.bool)

    pf = 1j * k * z
    kz2 = wl**2 * (kx**2 + ky**2)

    delta_H = W * torch.exp(pf * ((torch.sqrt(0j + 1 - kz2 / (4 * pi**2))) - (1 - kz2 / (8 * pi**2))))

    Q_1 = torch.exp(1j * k / (2 * z) * (x**2 + y**2))

    px_d = wl * z / (px * sim_xdim)

    if UsePhase:
        q_px = Sfov / sim_xdim
        q_y = ky / dkx * q_px
        q_x = kx / dkx * q_px
        Q_2 = torch.exp(1j * k * z) / (1j * wl * z) * torch.exp(1j * k / (2 * z) * (q_x**2 + q_y**2))

        U_final = (
            torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fft2(U) * torch.fft.ifftshift(delta_H)) * Q_1))) * Q_2
        )
    else:
        U_final = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fft2(U) * torch.fft.ifftshift(delta_H)) * Q_1)))

    return U_final, px_d


# only for sqaure
# 240705
def asm(U, px, wl, z, BLfactor=0.5):
    batch, channel, sim_ydim, sim_xdim = U.size()
    device = U.device
    cen = np.fix(sim_xdim / 2) + 1  # this is correct for SASM
    tot_fov_x = sim_xdim * px
    dkx = 2 * pi * 1 / tot_fov_x

    # discrete kx, ky grid matched with fft
    kx = dkx * torch.arange(-(cen - 1), (sim_xdim - cen) + 1, device=device)
    kx = kx.reshape(1, 1, 1, sim_xdim)
    ky = kx.reshape(1, 1, sim_xdim, 1)

    k = 2 * pi / wl

    # kz = k0**2 - (self.kx_grid**2 + self.ky_grid**2)
    kz = k**2 - (kx**2 + ky**2)  # Broadcasting
    kz = torch.sqrt(0j + kz * (kz > 0) * 1)

    klim = BLfactor * k / (torch.sqrt(1 / pi**2 * z**2 * dkx**2 + 1))

    filter = ((ky) ** 2 / klim**2 + (kx) ** 2 / (dkx * sim_xdim / 2) ** 2 <= 1) * ((ky) ** 2 / (dkx * sim_ydim / 2) ** 2 + (kx) ** 2 / klim**2 <= 1)

    U_prop = torch.fft.ifft2(
        torch.fft.fft2(U) * torch.fft.ifftshift(torch.exp(1j * kz * z) * filter)
    )  # torch_ifft(torch_fft(U) * torch.exp(1j * kz * z) * filter)

    return U_prop


class FieldPropagator:
    def __init__(self, wavelength_sim, sim_fov_target, sim_px, device):
        sim_xdim = sim_fov_target / sim_px

        cenx = np.fix(sim_xdim / 2) + 1
        ceny = np.fix(sim_xdim / 2) + 1

        fov_xend_n = -(cenx - 1) * sim_px
        fov_xend_p = (sim_xdim - cenx) * sim_px

        x_grid = np.arange(fov_xend_n, fov_xend_p, sim_px)
        sim_xdim = x_grid.shape[0]
        sim_ydim = sim_xdim

        # # Fourier parameters
        k0 = 2 * pi / wavelength_sim

        tot_fov_x = sim_xdim * sim_px
        tot_fov_y = sim_ydim * sim_px

        dkx = 2 * pi * 1 / tot_fov_x
        dky = 2 * pi * 1 / tot_fov_y

        # same logic with matlab
        zero_freq_index_x = np.fix(sim_xdim / 2) + 1
        zero_freq_index_y = np.fix(sim_ydim / 2) + 1

        # kx = 2 * pi * 1 / (tot_fov_x) * fftpxindex

        # discrete kx, ky grid matched with fft
        kx_list = dkx * np.arange(-(zero_freq_index_x - 1), (sim_xdim - zero_freq_index_x) + 1)
        ky_list = dky * np.arange(-(zero_freq_index_y - 1), (sim_ydim - zero_freq_index_y) + 1)

        [kx_grid, ky_grid] = np.meshgrid(kx_list, ky_list, indexing="xy")

        kz = k0**2 - (kx_grid**2 + ky_grid**2)
        kz = np.sqrt(kz * (kz > 0) * 1)
        kz = torch.from_numpy(kz).to(device=device)

        self.fov = tot_fov_x
        self.px = sim_px
        self.lamb = wavelength_sim
        self.kz = kz
        self.dkx = dkx
        self.dky = dky
        self.k0 = k0
        self.xdim = sim_xdim
        self.ydim = sim_ydim
        self.kx_grid = kx_grid
        self.ky_grid = ky_grid
        self.fov_xend_p = fov_xend_p
        self.device = device
        # self.mask = torch.ones_like(kz).to(device)

    def set_wavelength(self, wavelength_sim):
        self.lamb = wavelength_sim
        k0 = 2 * pi / wavelength_sim

        kz = k0**2 - (self.kx_grid**2 + self.ky_grid**2)
        kz = np.sqrt(kz * (kz > 0) * 1)
        kz = torch.from_numpy(kz).to(device=self.device)

        self.kz = kz
        self.k0 = k0

    def prop(self, U, z):
        # tan_theta = 1.1 * self.fov_xend_p / abs(z)
        tan_theta = 1 * self.fov_xend_p / abs(z.detach().cpu().numpy())
        max_theta_rad = np.arctan(tan_theta)  # [rad]
        k_maxrad_prop = self.k0 * np.sin(max_theta_rad)  # ; % radius
        k_max_aperture = circ_mask(self.xdim, self.ydim, k_maxrad_prop / self.dkx)
        k_max_aperture = torch.from_numpy(k_max_aperture).to(device=self.device)

        U_prop = torch_ifft(torch_fft(U) * torch.exp(1j * self.kz * z) * k_max_aperture)
        # U_prop = torch_ifft(torch_fft(U) * torch.exp(1j * self.kz * z))
        # U_prop = torch_ifft(torch_fft(U) * self.kernel)

        return U_prop


def lensphase(f, sim_px, sim_N, wavelength ):

    # 좌표 그리드 생성 (중심을 0으로 설정)
    # SLM의 인덱스에 대응하는 물리적 좌표 (x, y)를 계산합니다.
    # - ((N_slm-1)/2)*sim_px 를 기준으로 대칭적인 좌표를 만듭니다.
    x = torch.linspace(-((sim_N-1)/2)*sim_px, ((sim_N-1)/2)*sim_px, steps=int(sim_N))
    y = torch.linspace(-((sim_N-1)/2)*sim_px, ((sim_N-1)/2)*sim_px, steps=int(sim_N))
    xx, yy = torch.meshgrid(x, y, indexing='ij')  # (N_slm x N_slm)

    # 각 좌표에서의 반지름 제곱 계산 (x^2 + y^2)
    r_squared = xx**2 + yy**2
    # 얇은 렌즈의 phase delay 공식: φ(x,y) = -π (x^2 + y^2) / (wavelength * f)
    phase = -np.pi * r_squared / (wavelength * f)
    # 복소수 형태의 렌즈 phase map 생성 (SLM에 적용할 수 있는 형식)
    lens_phase_map = torch.exp(1j * phase)
    # lens_phase_map의 크기는 (N_slm, N_slm)이며, 필요시 shape을 (1, 1, N_slm, N_slm)으로 맞출 수 있습니다.
    lens_phase_map = lens_phase_map.unsqueeze(0).unsqueeze(0)

    return lens_phase_map

import matplotlib.pyplot as plt

def fft_conv2d(U_temp, PSF):
    """
    FFT 기반 2D 컨볼루션 (출력 크기 500x500 유지)
    - FFT 기반 컨볼루션 결과를 원래 중심에 배치하도록 수정 (fftshift 사용)

    Args:
        U_temp (torch.Tensor): 입력 이미지 (1, 1, 500, 500)
        PSF (torch.Tensor): 컨볼루션 필터 (1, 1, 128, 128)

    Returns:
        torch.Tensor: 500x500 크기의 FFT 기반 컨볼루션 결과 (중앙 정렬)
    """
    batch, channels, H, W = U_temp.shape  # H = 500, W = 500
    _, _, H_k, W_k = PSF.shape  # H_k = 128, W_k = 128

    # 1️⃣ PSF를 중앙에 배치하기 위해 제로패딩 추가
    pad_h = H - H_k  # 500 - 128 = 372
    pad_w = W - W_k  # 500 - 128 = 372

    # PSF를 중앙에 배치
    pad_PSF = torch.nn.functional.pad(PSF, (pad_w // 2, pad_w - pad_w // 2, 
                                            pad_h // 2, pad_h - pad_h // 2))

    # 2️⃣ FFT 변환 (2D 푸리에 변환)
    U_temp_fft = torch.fft.fft2(U_temp)  # 입력 이미지 FFT 변환
    PSF_fft = torch.fft.fft2(pad_PSF)  # PSF FFT 변환 (이미 중앙 배치됨)

    # 3️⃣ 주파수 도메인에서 점곱 (Convolution in Fourier Space)
    conv_result_fft = U_temp_fft * PSF_fft  # Hadamard Product

    # 4️⃣ 역 FFT 변환 후 중심 정렬
    conv_result = torch.fft.ifft2(conv_result_fft).real  # 실수값만 가져옴
    conv_result = torch.fft.fftshift(conv_result)  # 중앙으로 이동

    # 5️⃣ 출력 크기가 500x500으로 유지됨
    return conv_result








