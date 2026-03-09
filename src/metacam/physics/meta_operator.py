from torch.nn import functional as F
import torch
from pathlib import Path

import numpy as np
import torchvision.transforms.functional as TF

from metacam.data.io import load_mat_file
from metacam.ops.torch_ops import torch_crop_center, torch_pad_center
from metacam.physics.propagation import asm_master_alltorch


class MetaOperator:
    def __init__(self, wavelength_meter, sim_px, sim_N, cam_pp, device):
        self.wavelength_meter = wavelength_meter
        self.sim_px = sim_px
        self.sim_N = sim_N
        self.sim_fov = self.sim_px * self.sim_N
        self.cam_pp = cam_pp
        self.resize_ratio = cam_pp / sim_px
        self.device = device
        self.asmt = asm_master_alltorch(self.sim_fov, self.sim_px, device)

    def to(self, device):
        self.device = device
        self.asmt = asm_master_alltorch(self.sim_fov, self.sim_px, device)

    def forward(self, x, deg1=0, obj_prop=False, measurement_wind=0, shift=(0, 0), **kwargs):
        x = torch_pad_center(x, (self.asmt.ydim, self.asmt.xdim), padval=0)

        if obj_prop:
            x = self.asmt.prop_w_kernel(
                U=torch_pad_center(x, (self.asmt.ydim, self.asmt.xdim), padval=0) * self.aperture_obj, ASM_kernel=self.kernel_obj
            )

        x = self.asmt.sasm_prop_w_kernel(U=self.U_s1 * x, delta_H=self.delta_H, Q_1=self.Q_1)  # SASM
 
        x = torch.abs(x)  # ** 2
        x = TF.rotate(x, deg1, interpolation=TF.InterpolationMode.BILINEAR, expand=False) 
        x = F.interpolate(x, scale_factor=1 / (self.resize_ratio / self.MASM), mode="bilinear", antialias=True)  # SASM 
        print(1 / (self.resize_ratio / self.MASM))
        # x = F.interpolate(x, scale_factor=1 / (self.resize_ratio / self.MASM), mode="area" )  # SASM 

        if measurement_wind != 0:
            x = torch_crop_center(x, measurement_wind, shift=shift)

        return x

    def forward_error(self, x, U_s1_error, deg1=0, obj_prop=False, measurement_wind=0, shift=(0, 0), **kwargs):
        x = torch_pad_center(x, (self.asmt.ydim, self.asmt.xdim), padval=0)

        if obj_prop:
            x = self.asmt.prop_w_kernel(
                U=torch_pad_center(x, (self.asmt.ydim, self.asmt.xdim), padval=0) * self.aperture_obj, ASM_kernel=self.kernel_obj
            )

        x = self.asmt.sasm_prop_w_kernel(U=U_s1_error * self.U_s1 * x, delta_H=self.delta_H, Q_1=self.Q_1)  # SASM

        x = torch.abs(x)  # ** 2
        x = TF.rotate(x, deg1, interpolation=TF.InterpolationMode.BILINEAR, expand=False)
        x = F.interpolate(x, scale_factor=1 / (self.resize_ratio / self.MASM), mode="bilinear", antialias=True)  # SASM
        # x = F.interpolate(x, scale_factor=1 / (self.resize_ratio / self.MASM), mode="area" )  # SASM

        if measurement_wind != 0:
            x = torch_crop_center(x, measurement_wind, shift=shift)

        return x

    def get_asm_kernel(self, z, BLfactor=1):
        wl = torch.tensor([[self.wavelength_meter]]).reshape(1, 1, 1, 1).to(self.device)
        z = torch.tensor([z]).reshape(1, 1, 1, 1).to(self.device)

        kz = self.asmt.get_kz(wl)
        kernel = self.asmt.get_kernel(wl, kz=kz, z=z, BLfactor=BLfactor)
        return kernel

    def register_asm_sensor(self, z, BLfactor=1):
        self.kernel_sensor = self.get_asm_kernel(z=z, BLfactor=BLfactor)
        return

    def register_asm_obj(self, z, BLfactor=1):
        self.kernel_obj = self.get_asm_kernel(z=z, BLfactor=BLfactor)
        return

    def register_sasm_sensor(self, z1, BLfactor=0.5):
        wl = torch.tensor([[self.wavelength_meter]]).reshape(1, 1, 1, 1).to(self.device)
        z1 = torch.tensor([z1]).reshape(1, 1, 1, 1).to(self.device)

        self.delta_H, self.Q_1, self.Sfov = self.asmt.sasm_get_kernel(wl=wl, z=z1, BLfactor=BLfactor)
        self.MASM = self.Sfov.item() / self.sim_fov
        return

    def get_aperture(self, aperture_obj_width_px, trans=0):
        aperture_obj = torch.ones(1, 1, aperture_obj_width_px, aperture_obj_width_px)
        aperture_obj = torch_pad_center(aperture_obj, (self.asmt.ydim, self.asmt.xdim), padval=trans).to(self.device)
        return aperture_obj

    def register_aperture_obj(self, aperture_obj_width_px):
        self.aperture_obj = self.get_aperture(aperture_obj_width_px)

    def register_aperture_meta(self, aperture_meta_width_px, trans=0):
        self.aperture_meta = self.get_aperture(aperture_meta_width_px, trans)

    def register_meta_phasemap(self, p_widthmap, p_lookup, wmap_k=2, RCWA=False):
        # Read meta widthmap
        wmap_file_dir = Path(p_widthmap)
        wmapdata = load_mat_file(wmap_file_dir)  # mapped_width
        meta_wmap = wmapdata["mapped_width"]

        meta_wmap_idx_1 = torch.tensor(meta_wmap - 60, dtype=torch.int, device=self.device).unsqueeze(0).unsqueeze(0)  # in matlab, 60 --> 59
        meta_wmap_idx_1 = torch.rot90(meta_wmap_idx_1, k=wmap_k, dims=(2, 3))
        # meta_wmap_idx_1 = F.interpolate(meta_wmap_idx_1.to(torch.float32), scale_factor=2).to(torch.int32)
        # meta_wmap_idx_1 = F.interpolate(meta_wmap_idx_1.to(torch.float32), scale_factor=350e-9 / self.sim_px, mode="bilinear", antialias=True).to(
        #     torch.int32
        # )
        if round(350e-9 / self.sim_px, 3) != 1:
            print("interpolate meta")
            print(350e-9 / self.sim_px)
            meta_wmap_idx_1 = F.interpolate(meta_wmap_idx_1.to(torch.float32), scale_factor=350e-9 / self.sim_px).to(torch.int32)
 
        # Read meta width-phase look-up table
        LUT_file_dir = Path(p_lookup)
        LUTdata = load_mat_file(LUT_file_dir)
        
        if RCWA: 
            lut_opt_interp = LUTdata["lut_phase_unwrap"] 
        else: 
            lut_opt_interp = LUTdata["lut_opt_interp"]
            
        ldim = lut_opt_interp.shape[0]
        # wl_interp = LUTdata["wl_interp"]

        def wl_to_idx(wl):
            return np.uint16((wl - 440) * 10)

        # Specify wavelength
        lut_est = torch.tensor(lut_opt_interp[:, wl_to_idx(self.wavelength_meter * 1e9)]).reshape(1, 1, 1, ldim).to(self.device).to(torch.float32)
        # lut_est = torch.tensor(lut_opt_interp[:, wl_to_idx(self.wavelength_meter * 1e9)]).reshape(1, 1, 1, ldim).to(self.device).to(torch.float64)
         
        # Calculate meta phasemap
        meta_phmap = lut_est.squeeze()[meta_wmap_idx_1.long()]
        meta_phmap_simgrid = torch_pad_center(meta_phmap, (self.asmt.ydim, self.asmt.xdim))

        self.lut_init = lut_est
        self.meta_wmap = meta_wmap_idx_1
        self.U_s1 = self.aperture_meta * torch.exp(1j * meta_phmap_simgrid)
        return

    def psf(self, gx, gy, gz, w0=100e-9, wl=0, method="spherical"):
        if wl == 0:
            # wl = self.wavelength_meter
            wl = torch.tensor([[self.wavelength_meter]]).reshape(1, 1, 1, 1).to(self.device)

        x = self.asmt.kx_list / self.asmt.dkx * self.asmt.px
        y = self.asmt.ky_list / self.asmt.dky * self.asmt.px
        k = 2 * np.pi / wl

        if method == "spherical":
            wf = 1 / torch.sqrt((x - gx) ** 2 + (y - gy) ** 2 + gz**2) * torch.exp(1j * k * torch.sqrt((x - gx) ** 2 + (y - gy) ** 2 + gz**2))

        elif method == "spherical_phaseonly":
            wf = torch.exp(1j * k * torch.sqrt((x - gx) ** 2 + (y - gy) ** 2 + gz**2))
        
        elif method == "gaussian":
            z_r = np.pi * w0**2 / wl
            W = w0 * torch.sqrt(1 + (gz / z_r) ** 2)
            R = gz * (1 + (z_r / gz) ** 2)
            # g1d_x = w0 / W * torch.exp(-((x - gx) ** 2) / W**2) * torch.exp(1j * (k * gz - torch.atan(gz / z_r) + k * (x - gx) ** 2 / (2 * R)))
            # g1d_y = w0 / W * torch.exp(-((y - gy) ** 2) / W**2) * torch.exp(1j * (k * gz - torch.atan(gz / z_r) + k * (y - gy) ** 2 / (2 * R)))
            # wf = g1d_y * g1d_x
            # wf = wf  # / wf.abs().max()
            
            wf = w0 / W * torch.exp(-((x - gx) ** 2 + (y - gy) ** 2) / W**2) * torch.exp(1j * (k * gz - torch.atan(gz / z_r) + k * ((x - gx) ** 2 + (y - gy) ** 2)  / (2 * R)))
            # wf = w0 / W * torch.exp(-( (x**2+y**2) / W**2))* torch.exp(1j * (k * gz - torch.atan(gz / z_r) + k * (x ** 2 + y ** 2) / (2 * R)))


        elif method == "quadratic":
            # wf = torch.exp(1j * (x * self.asmt.dkx * gx + y * self.asmt.dky * gy)) * torch.exp(1j * (2 * np.pi / wl) / (2 * (gz)) * (x**2 + y**2))
            wf = torch.exp(1j * -(x*k*np.sin(np.arctan(gx/gz)) + y*k*np.sin(np.arctan(gy/gz)))) * torch.exp(1j * (2 * np.pi / wl) / (2 * (gz)) * (x**2 + y**2))


        elif method == "fresnel":
            k = 2 * np.pi / wl
            swf = 1 / gz * torch.exp(1j * k * (gz + ((x - gx) ** 2 + (y - gy) ** 2) / (2 * gz)))
            return swf
        
        elif method == "tilt":
            # gx, gy는 각각 x, y축에 대한 tilt 각도(degree)
            theta_x = torch.deg2rad(torch.tensor(gx, device=self.device))  # degree → radian
            theta_y = torch.deg2rad(torch.tensor(gy, device=self.device))
            wf = torch.exp(1j * k * (x * torch.sin(theta_x) + y * torch.sin(theta_y)))
 
        return wf
