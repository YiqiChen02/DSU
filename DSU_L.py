# -*- coding: utf-8 -*-
import torch
from torch import nn
import einops
from typing import Union
import torch.nn.functional as F
from Dynamic_L import OptimizedOddAdjustment_init

class EncoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x

class DSU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size : int = 9,
        scale: int = 2,
        extend_scope=1.0,
        if_offset: bool = True,
        device: Union[str, torch.device]  = "cuda",
        vis : bool = False
    ):
        super(DSU, self).__init__()  
        self.device = device
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.scale = scale
        self.extend_scope = extend_scope
        self.relu = nn.ReLU(inplace=True)
        self.if_offset = if_offset
        self.out_channels = out_channels
        self.vis = vis

        self.up00 = nn.Upsample(scale_factor=self.scale, mode='bilinear', align_corners=True)

        self.conv0x = DSUper_L(
            in_channels,
            self.out_channels,
            self.scale,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
            self.vis
        )
        self.conv0y = DSUper_L(
            in_channels,
            self.out_channels,
            self.scale,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
            self.vis
        )
        self.conv1 = EncoderConv(3 * self.out_channels, self.out_channels)
        self.length_predictor = OptimizedOddAdjustment_init(input_channels=in_channels)

    def forward(self, x):
        # Step 1: Predict L
        L_odd = self.length_predictor(x)
        
        # Step 2: Convert L_odd to integer (retain gradient flow via STE)
        L_odd_int = torch.round(L_odd).int().item()  # Keep STE gradient
        
        # print(f'{shape}_L_odd_int', L_odd_int)

        x_00_0 = self.up00(x)   
        if self.vis:
            feature_map = x
            x_0x_0 = self.conv0x(x, L_odd_int, feature_map)
            x_0y_0 = self.conv0y(x, L_odd_int, feature_map)
        else:
            x_0x_0 = self.conv0x(x, L_odd_int)
            x_0y_0 = self.conv0y(x, L_odd_int)

        x_0_1 = self.conv1(torch.cat([x_00_0, x_0x_0, x_0y_0], dim=1))

        return x_0_1

class  DSUper_L(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        scale: int = 2,
        kernel_size: int = 9,
        extend_scope: float = 1.0,
        morph: int = 0,
        if_offset: bool = True,   
        device: Union[str, torch.device]  = "cuda",
        vis : bool = False
    ):

        super().__init__()

        if morph not in (0, 1):
            raise ValueError("morph should be 0 or 1.")

        self.kernel_size = kernel_size
        self.scale = scale
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = torch.device(device)
        self.vis = vis
        self.to(device)

        self.gn = nn.GroupNorm(out_channels // 4, out_channels) 
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        # * self.scale**2 is to expand the number of channels by 4 times for the subsequent pixel shuffle operation
        self.offset_conv = DynamicChannelConv(in_channels, 2 * kernel_size * self.scale**2,self.scale)
        self.dydsu_conv_x = DynamicKernelConv2d(  
            in_channels*self.scale**2,
            out_channels*self.scale**2,
            morph= self.morph
        )
        self.dydsu_conv_y = DynamicKernelConv2d(
            in_channels*self.scale**2,
            out_channels*self.scale**2,
            morph= self.morph
        )

    def forward(self, input: torch.Tensor,  L = 9, feature_map=None):
        offset_scaled = self.offset_conv(input,L)
        offset_scaled = apply_dynamic_groupnorm(offset_scaled, L, self.scale)
        offset_scaled = self.tanh(offset_scaled)
        offset_scaled_groups = torch.split(offset_scaled, L*2,dim=1)
        deformed_feature_groups = []   
        x_coordinate_map_groups = []
        y_coordinate_map_groups = []
        for offset, init_pos in zip(offset_scaled_groups, [(-0.25,-0.25),(0.25,-0.25),(-0.25,0.25),(0.25,0.25)]):
        # Run deformative up
            y_coordinate_map, x_coordinate_map = get_coordinate_map_UP(
                offset=offset,
                morph=self.morph,
                extend_scope=self.extend_scope,
                device=self.device,
                initial_position=init_pos
            )
            x_coordinate_map_groups.append(x_coordinate_map)
            y_coordinate_map_groups.append(y_coordinate_map)
            deformed_feature = get_interpolated_feature(
                input,
                y_coordinate_map,
                x_coordinate_map,
                interpolate_mode='bilinear'
            )
            deformed_feature_groups.append(deformed_feature)

        deformed_feature = torch.cat(deformed_feature_groups,dim=1)

        if self.morph == 0:
            output = self.dydsu_conv_x(deformed_feature,L)
        elif self.morph == 1:  
            output = self.dydsu_conv_y(deformed_feature,L)
        output = nn.functional.pixel_shuffle(output, upscale_factor=self.scale)
        # Groupnorm & ReLU
        output = self.gn(output)
        output = self.relu(output)

        return output
    
def apply_dynamic_groupnorm(x, kernel_size, scale=2):
    """
    x: input tensor (B, C, H, W)
    kernel_size: current kernel size (int)
    scale: scale factor used for upsampling or offset, default is 2

    Dynamically create and apply GroupNorm based on kernel_size without explicit initialization
    """
    num_channels = 2 * kernel_size * (scale**2)
    gn = nn.GroupNorm(
        num_groups=kernel_size,  # Number of groups = kernel size
        num_channels=num_channels,
        affine=True  # Whether to use learnable scale and bias; can be set as needed
    ).to(x.device)
    return gn(x)

class DynamicChannelConv(nn.Module):
    def __init__(self, in_channels, max_out_channels, scale=2):
        super().__init__()
        self.in_channels = in_channels
        self.max_out_channels = max_out_channels
        self.scale = scale

        self.offset_conv = nn.Conv2d(
            in_channels, 
            self.max_out_channels, 
            kernel_size=3, 
            padding=1
        )

    def forward(self, x, L):
        # Calculate the target number of output channels
        target_channels = 2 * L * (self.scale**2)
        
        # Safety check for channel count
        if target_channels > self.max_out_channels:
            raise ValueError(
                f"Target number of channels {target_channels} exceeds the preset maximum {self.max_out_channels}. "
                f"Please increase the max_out_channels parameter."
            )

        # Perform standard convolution
        out = self.offset_conv(x)
        
        # Compute symmetric cropping range
        start = (self.max_out_channels - target_channels) // 2
        end = start + target_channels
        
        # Channel slicing
        return out[:, start:end, :, :]

class DynamicKernelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 max_kernel_size=9, 
                 stride=(9,1), 
                 padding_mode='auto',
                 morph=0):
        super().__init__()
        self.max_kernel_size = max_kernel_size  # Maximum kernel size (H or W)
        self.morph = morph

        # Initialize the weights of the maximum convolution kernel
        if morph == 0:
            self.weight = nn.Parameter(
                torch.randn(out_channels, in_channels, max_kernel_size, 1)
            )
        else:
            self.weight = nn.Parameter(
                torch.randn(out_channels, in_channels, 1, max_kernel_size)
            )

        # Handle padding automatically based on kernel orientation
        if padding_mode == 'auto':
            if morph == 0:
                self.padding = (max_kernel_size // 2, 0)
            else:
                self.padding = (0, max_kernel_size // 2)
        else:
            self.padding = padding_mode
        self.stride = stride

    def forward(self, x, L):
        # Generate the dynamic kernel mask based on the given kernel length L
        kernel_mask = self._generate_kernel_mask(L, self.morph)

        # Apply the mask to the kernel weights
        masked_weight = self.weight * kernel_mask

        # Update stride dynamically according to kernel orientation
        self.stride = (L, 1) if self.morph == 0 else (1, L)

        # Perform convolution with masked kernel
        out = F.conv2d(x, masked_weight, padding=self.padding, stride=self.stride)
        return out

    def _generate_kernel_mask(self, L, morph):
        """
        Generate a binary mask to activate a central L-sized region of the kernel.
        """
        device = self.weight.device
        if morph == 0:
            mask = torch.zeros((self.max_kernel_size, 1), device=device)
        else:
            mask = torch.zeros((1, self.max_kernel_size), device=device)

        center = self.max_kernel_size // 2
        start = center - L // 2
        end = start + L

        if morph == 0:
            mask[start:end, :] = 1.0
        else:
            mask[:, start:end] = 1.0

        return mask.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W] for broadcasting


def get_coordinate_map_UP(
    offset: torch.Tensor,
    morph: int,
    extend_scope: float = 1.0,
    device: Union[str, torch.device] = "cuda", 
    initial_position: tuple = (0,0)
):
    """This version is used to adjust the initial position of upsampling, ensuring that the upsampled pixel initially aligns with the corresponding position in the original image. 

    Args:
        offset: offset predict by network with shape [B, 2*K, W, H]. Here K refers to kernel size.
        morph: the morphology of the convolution kernel is mainly divided into two types along the x-axis (0) and the y-axis (1) (see the paper for details).
        extend_scope: the range to expand. Defaults to 1 for this method.
        device: location of data. Defaults to 'cuda'.
        initial_position: the initial position of the upsampling operation. Defaults to (0,0).

    Return:
        y_coordinate_map: coordinate map along y-axis with shape [B, K_H * H, K_W * W]
        x_coordinate_map: coordinate map along x-axis with shape [B, K_H * H, K_W * W]
    """

    if morph not in (0, 1):
        raise ValueError("morph should be 0 or 1.")

    batch_size, _, width, height = offset.shape
    kernel_size = offset.shape[1] // 2  # N
    center = kernel_size // 2
    device = torch.device(device)

    y_offset_, x_offset_ = torch.split(offset, kernel_size, dim=1)  
    y_center_ = torch.arange(0, width, dtype=torch.float32, device=device)
    y_center_ = y_center_ + initial_position[1]
    y_center_ = einops.repeat(y_center_, "w -> k w h", k=kernel_size, h=height)

    x_center_ = torch.arange(0, height, dtype=torch.float32, device=device)
    x_center_ = x_center_ + initial_position[0]
    x_center_ = einops.repeat(x_center_, "h -> k w h", k=kernel_size, w=width)

    if morph == 0:
        """
        Initialize the kernel and flatten the kernel
            y: only need 0
            x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
        """
        y_spread_ = torch.zeros([kernel_size], device=device)
        x_spread_ = torch.linspace(-center, center, kernel_size, device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        y_offset_ = einops.rearrange(y_offset_, "b k w h -> k b w h")
        y_offset_new_ = y_offset_.detach().clone()

        # The center position remains unchanged and the rest of the positions begin to swing
        # This part is quite simple. The main idea is that "offset is an iterative process"

        y_offset_new_[center] = 0  

        for index in range(1, center + 1):
            y_offset_new_[center + index] = (
                y_offset_new_[center + index - 1] + y_offset_[center + index]
            )
            y_offset_new_[center - index] = (
                y_offset_new_[center - index + 1] + y_offset_[center - index]
            )

        y_offset_new_ = einops.rearrange(y_offset_new_, "k b w h -> b k w h")

        y_new_ = y_new_.add(y_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b (w k) h")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b (w k) h")

    elif morph == 1:
        """
        Initialize the kernel and flatten the kernel
            y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
            x: only need 0
        """
        y_spread_ = torch.linspace(-center, center, kernel_size, device=device)
        x_spread_ = torch.zeros([kernel_size], device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        x_offset_ = einops.rearrange(x_offset_, "b k w h -> k b w h")
        x_offset_new_ = x_offset_.detach().clone()

        # The center position remains unchanged and the rest of the positions begin to swing
        # This part is quite simple. The main idea is that "offset is an iterative process"

        x_offset_new_[center] = 0

        for index in range(1, center + 1):
            x_offset_new_[center + index] = (
                x_offset_new_[center + index - 1] + x_offset_[center + index]
            )
            x_offset_new_[center - index] = (
                x_offset_new_[center - index + 1] + x_offset_[center - index]
            )

        x_offset_new_ = einops.rearrange(x_offset_new_, "k b w h -> b k w h")

        x_new_ = x_new_.add(x_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b w (h k)")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b w (h k)")

    return y_coordinate_map, x_coordinate_map

def get_interpolated_feature(
    input_feature: torch.Tensor,
    y_coordinate_map: torch.Tensor,
    x_coordinate_map: torch.Tensor,
    interpolate_mode: str = "bilinear",
):
    """From coordinate map interpolate feature of DSCNet based on: TODO

    Args:
        input_feature: feature that to be interpolated with shape [B, C, H, W]
        y_coordinate_map: coordinate map along y-axis with shape [B, K_H * H, K_W * W]
        x_coordinate_map: coordinate map along x-axis with shape [B, K_H * H, K_W * W]
        interpolate_mode: the arg 'mode' of nn.functional.grid_sample, can be 'bilinear' or 'bicubic' . Defaults to 'bilinear'.
                            Note: when it comes to use to upsample feature from thin things, 'nearest' mode may be more suitable.
    Return:
        interpolated_feature: interpolated feature with shape [B, C, K_H * H, K_W * W]
    """

    if interpolate_mode not in ("bilinear", "bicubic", "nearest"):
        raise ValueError("interpolate_mode should be 'bilinear' or 'bicubic' or 'nearest'.")

    y_max = input_feature.shape[-2] - 1
    x_max = input_feature.shape[-1] - 1


    y_coordinate_map_ = _coordinate_map_scaling(y_coordinate_map, origin=[0, y_max])
    x_coordinate_map_ = _coordinate_map_scaling(x_coordinate_map, origin=[0, x_max])

    y_coordinate_map_ = torch.unsqueeze(y_coordinate_map_, dim=-1)
    x_coordinate_map_ = torch.unsqueeze(x_coordinate_map_, dim=-1)

    # Note here grid with shape [B, H, W, 2]
    # Where [:, :, :, 2] refers to [x ,y]
    grid = torch.cat([x_coordinate_map_, y_coordinate_map_], dim=-1)

    interpolated_feature = nn.functional.grid_sample(
        input=input_feature,
        grid=grid,
        mode=interpolate_mode,
        padding_mode="zeros",
        align_corners=True,
    )

    return interpolated_feature

def _coordinate_map_scaling(
    coordinate_map: torch.Tensor,
    origin: list,
    target: list = [-1, 1],
):
    """Map the value of coordinate_map from origin=[min, max] to target=[a,b] for DSCNet based on: TODO

    Args:
        coordinate_map: the coordinate map to be scaled
        origin: original value range of coordinate map, e.g. [coordinate_map.min(), coordinate_map.max()]
        target: target value range of coordinate map,Defaults to [-1, 1]

    Return:
        coordinate_map_scaled: the coordinate map after scaling
    """
    min, max = origin
    a, b = target

    coordinate_map_scaled = torch.clamp(coordinate_map, min, max)

    scale_factor = (b - a) / (max - min)
    coordinate_map_scaled = a + scale_factor * (coordinate_map_scaled - min)

    return coordinate_map_scaled

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Simulate the input tensor (batch_size, channels, height, width)
    x = torch.rand(1, 4, 64, 64).to(device)  
    print("input.shape:", x.shape)

    uper = DSU(4, 4).to(device)
    output = uper(x)

    print("output.shape:", output.shape)