import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            # First convolution
            nn.Conv2d(
                in_channels, 
                mid_channels,
                kernel_size=3,
                padding=1,
                bias=False  # No bias when using batch norm
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            # Second convolution
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False  # No bias when using batch norm
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Optional: Add residual connection if input and output channels match
        self.use_residual = in_channels == out_channels
        
    def forward(self, x):
        if self.use_residual:
            return self.double_conv(x) + x
        return self.double_conv(x)
    
class TemporalBlock(nn.Module):
    def __init__(self, channels, temporal_kernel_size=3):
        super().__init__()
        
        padding = temporal_kernel_size // 2
        
        self.temporal_conv = nn.Sequential(
            # Depthwise temporal conv
            nn.Conv1d(channels, channels,
                     kernel_size=temporal_kernel_size,
                     padding=padding,
                     groups=channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            
            # Point-wise conv
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
    
class UNet2DPlusTemporal(nn.Module):
    def __init__(self, input_frames=None, base_filters=32):
        super(UNet2DPlusTemporal, self).__init__()
        
        self.input_frames = input_frames
        
        # Purely spatial encoder (first layer)
        self.enc1 = DoubleConv2D(1, base_filters)
        self.pool1 = nn.MaxPool2d(2)
        
        # Spatio-temporal encoders
        self.enc2 = DoubleConv2D(base_filters, base_filters*2)
        self.temporal_enc2 = nn.ModuleList([
            TemporalBlock(channels=base_filters*2, temporal_kernel_size=3)
            for _ in range(2)
        ])
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv2D(base_filters*2, base_filters*4)
        self.temporal_enc3 = nn.ModuleList([
            TemporalBlock(channels=base_filters*4, temporal_kernel_size=3)
            for _ in range(3)
        ])
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv2D(base_filters*4, base_filters*8)
        self.temporal_enc4 = nn.ModuleList([
            TemporalBlock(channels=base_filters*8, temporal_kernel_size=3)
            for _ in range(4)
        ])
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck with 5 temporal blocks
        self.bottleneck_spatial = DoubleConv2D(base_filters*8, base_filters*16)
        self.temporal_bottleneck = nn.ModuleList([
            TemporalBlock(channels=base_filters*16, temporal_kernel_size=3)
            for _ in range(5)
        ])
        
        # Decoder Path
        self.upconv4 = nn.ConvTranspose2d(base_filters*16, base_filters*8, 
                                         kernel_size=2, stride=2)
        self.dec4 = DoubleConv2D(base_filters*16, base_filters*8)
        
        self.upconv3 = nn.ConvTranspose2d(base_filters*8, base_filters*4, 
                                         kernel_size=2, stride=2)
        self.dec3 = DoubleConv2D(base_filters*8, base_filters*4)
        
        self.upconv2 = nn.ConvTranspose2d(base_filters*4, base_filters*2, 
                                         kernel_size=2, stride=2)
        self.dec2 = DoubleConv2D(base_filters*4, base_filters*2)
        
        self.upconv1 = nn.ConvTranspose2d(base_filters*2, base_filters, 
                                         kernel_size=2, stride=2)
        self.dec1 = DoubleConv2D(base_filters*2, base_filters)
        
        self.final_conv = nn.Conv2d(base_filters, 1, kernel_size=1)
        
    def temporal_processing(self, x, temporal_blocks):
        # Apply multiple temporal blocks sequentially
        for block in temporal_blocks:
            x = block(x)
        return x
    
    def forward(self, x):
        # x shape: [batch, time, channel, height, width]
        b, t, c, h, w = x.shape
        assert t == self.input_frames, f"Expected {self.input_frames} frames, got {t}"
        
        # Process first layer spatially only
        spatial_features = []
        for i in range(t):
            curr_frame = x[:, i]  # [B, C, H, W]
            e1 = self.enc1(curr_frame)
            spatial_features.append(e1)
        
        # Store last frame's features for skip connection
        skip1 = spatial_features[-1]
        
        # Pool and prepare for next layer
        spatial_features = torch.stack([self.pool1(feat) for feat in spatial_features], dim=2)
        
        # Second encoder layer (spatio-temporal)
        e2 = self.enc2(spatial_features.view(b*t, -1, h//2, w//2))
        e2 = e2.view(b, -1, t, h//2, w//2)
        e2 = self.temporal_processing(e2, self.temporal_enc2)
        skip2 = torch.mean(e2, dim=2)  # Aggregate temporal information
        e2_pooled = self.pool2(e2.view(b*t, -1, h//2, w//2)).view(b, -1, t, h//4, w//4)
        
        # Third encoder layer
        e3 = self.enc3(e2_pooled.view(b*t, -1, h//4, w//4))
        e3 = e3.view(b, -1, t, h//4, w//4)
        e3 = self.temporal_processing(e3, self.temporal_enc3)
        skip3 = torch.mean(e3, dim=2)
        e3_pooled = self.pool3(e3.view(b*t, -1, h//4, w//4)).view(b, -1, t, h//8, w//8)
        
        # Fourth encoder layer
        e4 = self.enc4(e3_pooled.view(b*t, -1, h//8, w//8))
        e4 = e4.view(b, -1, t, h//8, w//8)
        e4 = self.temporal_processing(e4, self.temporal_enc4)
        skip4 = torch.mean(e4, dim=2)
        e4_pooled = self.pool4(e4.view(b*t, -1, h//8, w//8)).view(b, -1, t, h//16, w//16)
        
        # Bottleneck
        bottle = self.bottleneck_spatial(e4_pooled.view(b*t, -1, h//16, w//16))
        bottle = bottle.view(b, -1, t, h//16, w//16)
        bottle = self.temporal_processing(bottle, self.temporal_bottleneck)
        bottle = bottle[:, :, -1]  # Take last temporal state
        
        # Decoder path with skip connections
        d4 = self.upconv4(bottle)
        d4 = torch.cat([d4, skip4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, skip3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, skip2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, skip1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final_conv(d1)
    