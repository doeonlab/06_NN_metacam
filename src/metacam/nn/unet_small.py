"""Compact U-Net blocks for reduced PhaseCam reconstruction models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(1, out_channels // 8), num_channels=out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(1, out_channels // 8), num_channels=out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        return skip, self.pool(skip)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetSmall(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        depth: int = 4,
    ) -> None:
        super().__init__()
        channels = [base_channels * (2**idx) for idx in range(depth)]
        self.down_blocks = nn.ModuleList()
        prev_channels = in_channels
        for channel in channels:
            self.down_blocks.append(DownBlock(prev_channels, channel))
            prev_channels = channel
        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2)
        self.up_blocks = nn.ModuleList()
        current = channels[-1] * 2
        for skip_channels in reversed(channels):
            self.up_blocks.append(UpBlock(current, skip_channels, skip_channels))
            current = skip_channels
        self.head = nn.Conv2d(current, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for down in self.down_blocks:
            skip, x = down(x)
            skips.append(skip)
        x = self.bottleneck(x)
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip)
        return self.head(x)

