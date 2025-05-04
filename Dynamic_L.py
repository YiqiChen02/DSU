import torch
import torch.nn as nn

class OddSamplingSTE(torch.autograd.Function):
    """
    Custom Straight-Through Estimator: Round to the nearest odd number during forward pass,
    preserve gradient continuity during backward pass.
    """
    @staticmethod
    def forward(ctx, input):
        # Compute the nearest odd number (discretization during forward pass)
        adjusted = 2 * torch.round((input - 1) / 2) + 1
        return adjusted

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: directly pass the gradient, ignoring discretization
        return grad_output.clone()

class OptimizedOddAdjustment_init(nn.Module):
    def __init__(self, input_channels, lambda_range=0.5):
        super().__init__()
        self.lambda_range = lambda_range

        # Channel and spatial compression module (output a scalar)
        self.channel_compressor = nn.Sequential(
            nn.Conv2d(input_channels, 1, kernel_size=1),   # [B,1,H,W]
            nn.AdaptiveAvgPool2d(1),                        # [B,1,1,1]
            nn.Flatten(),                                   # [B,1]
            nn.Linear(1, 1)                                 # [B,1] -> [B,1]
        )

        # Dynamic adjustment factor predictor (output a scalar)
        self.lambda_predictor = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Tanh()
        )

    def forward(self, feature_map):
        # Step 0: Get input feature map size (assume input is [B, C, H, W])
        _, _, h, w = feature_map.size()
        size = max(h, w)

        # Step 0.1: Set base_length dynamically based on input size
        if size <= 60:
            base_length = 4
        elif size <= 120:
            base_length = 6
        elif size <= 250:
            base_length = 8
        else:
            base_length = 8  # or a custom maximum value

        # Step 1: Compress to scalar feature [B,1,1,1] -> [B,1] -> scalar
        x = self.channel_compressor(feature_map)            # [B,1]
        x = x.mean(dim=0, keepdim=True)                     # average across batch [1,1]

        # Step 2: Predict adjustment factor (scalar)
        lambda_ = self.lambda_predictor(x) * self.lambda_range  # [1,1]

        # Step 3: Compute dynamic length and map to nearest odd number
        adjusted_continuous = base_length * (1 + lambda_)
        adjusted_odd = 2 * torch.round((adjusted_continuous - 1) / 2) + 1
        adjusted_odd = torch.clamp(adjusted_odd, min=3, max=9)  # constrain range

        return adjusted_odd.squeeze()  # return scalar (non-tensor)