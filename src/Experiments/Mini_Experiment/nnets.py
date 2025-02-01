import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet2DPlusTemporal(nn.Module):
    def __init__(self, input_frames=None, base_filters=32):
        super().__init__()
        
        self.input_frames = input_frames
        
        # Spatial Encoder Path (2D)
        self.enc1 = DoubleConv2D(1, base_filters)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv2D(base_filters, base_filters*2)
        self.pool2 = nn.MaxPool2d(2)
        
        # Last encoder with temporal processing
        self.enc3 = DoubleConv2D(base_filters*2, base_filters*4)
        self.temporal_enc = TemporalBlock(
            channels=base_filters*4,
            temporal_kernel_size=3
        )
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck with temporal processing
        self.bottleneck_spatial = DoubleConv2D(base_filters*4, base_filters*8)
        self.temporal_bottleneck = TemporalBlock(
            channels=base_filters*8,
            temporal_kernel_size=3
        )
        
        # Decoder Path (2D)
        self.upconv3 = nn.ConvTranspose2d(base_filters*8, base_filters*4, 
                                         kernel_size=2, stride=2)
        self.dec3 = DoubleConv2D(base_filters*8, base_filters*4)
        
        self.upconv2 = nn.ConvTranspose2d(base_filters*4, base_filters*2, 
                                         kernel_size=2, stride=2)
        self.dec2 = DoubleConv2D(base_filters*4, base_filters*2)
        
        self.upconv1 = nn.ConvTranspose2d(base_filters*2, base_filters, 
                                         kernel_size=2, stride=2)
        self.dec1 = DoubleConv2D(base_filters*2, base_filters)
        
        # Final layers
        self.final_spatial = nn.Conv2d(base_filters, base_filters, kernel_size=1)
        self.final_temporal_pool = nn.AdaptiveAvgPool1d(1)  # Temporal reduction
        self.final_conv = nn.Conv2d(base_filters, 1, kernel_size=1)
    
    def forward(self, x):
        # x shape: [batch, time, height, width]
        b, t, h, w = x.shape
        assert t == self.input_frames, f"Expected {self.input_frames} frames, got {t}"
        
        # Process each time step through initial spatial encoders
        encoder_features = []
        enc3_features = []
        
        for i in range(t):
            curr_frame = x[:, i].unsqueeze(1)  # [B, 1, H, W]
            
            # Initial encoder path
            e1 = self.enc1(curr_frame)
            p1 = self.pool1(e1)
            
            e2 = self.enc2(p1)
            p2 = self.pool2(e2)
            
            # Store for skip connections
            encoder_features.append((e1, e2))
            
            # Last encoder
            e3 = self.enc3(p2)
            enc3_features.append(e3)
        
        # Process enc3 features temporally
        enc3_features = torch.stack(enc3_features, dim=2)  # [B, C, T, H, W]
        enc3_processed = self.temporal_enc(enc3_features)
        
        # Pool spatially after temporal processing
        b, c, t, h, w = enc3_processed.shape
        enc3_pooled = enc3_processed.view(b*t, c, h, w)
        enc3_pooled = self.pool3(enc3_pooled)
        enc3_pooled = enc3_pooled.view(b, c, t, h//2, w//2)
        
        # Bottleneck processing
        # First, apply spatial convolutions to each temporal slice
        bottle_features = []
        for i in range(t):
            curr_feat = enc3_pooled[:, :, i]  # [B, C, H, W]
            bottle_feat = self.bottleneck_spatial(curr_feat)
            bottle_features.append(bottle_feat)
        
        # Stack and apply temporal processing in bottleneck
        bottle_features = torch.stack(bottle_features, dim=2)  # [B, C, T, H, W]
        bottle_processed = self.temporal_bottleneck(bottle_features)
        
        # Take last temporal state for decoder
        bottle_final = bottle_processed[:, :, -1]  # [B, C, H, W]
        
        # Decoder path (using last frame's encoder features)
        e1, e2 = encoder_features[-1]
        e3 = enc3_processed[:, :, -1]
        
        d3 = self.upconv3(bottle_final)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Final processing
        out = self.final_spatial(d1)
        out = self.final_conv(out)
        
        return out


class TemporalBlock(nn.Module):
    def __init__(self, channels, temporal_kernel_size=3):
        super().__init__()
        
        padding = temporal_kernel_size // 2
        
        self.temporal_conv = nn.Sequential(
            # Reshape-friendly temporal convolution
            nn.Conv1d(channels, channels,
                     kernel_size=temporal_kernel_size,
                     padding=padding,
                     groups=channels),  # Depthwise temporal conv
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            
            # Point-wise conv for channel mixing
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # x shape: [B, C, T, H, W]
        b, c, t, h, w = x.shape
        
        # Reshape for temporal convolution
        x_temp = x.permute(0, 3, 4, 1, 2)  # [B, H, W, C, T]
        x_temp = x_temp.reshape(b*h*w, c, t)
        
        # Apply temporal convolution
        x_temp = self.temporal_conv(x_temp)
        
        # Reshape back
        x_temp = x_temp.reshape(b, h, w, c, t)
        x_temp = x_temp.permute(0, 3, 4, 1, 2)  # [B, C, T, H, W]
        
        return x_temp


def test_model():
    # Test with different temporal dimensions
    temporal_sizes = [8, 16, 32]
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    for T in temporal_sizes:
        print(f"\nTesting with T={T}")
        model = UNet2DPlusTemporal(input_frames=T).to(device)
        x = torch.randn(2, T, 256, 256).to(device)
        
        try:
            out = model(x)
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {out.shape}")
            print("Test passed!")
        except Exception as e:
            print(f"Error with T={T}: {str(e)}")

if __name__ == "__main__":
    test_model()


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.double_conv = nn.Sequential(
            # First 3D convolution
            nn.Conv3d(in_channels, out_channels, 
                     kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            
            # Second 3D convolution
            nn.Conv3d(out_channels, out_channels, 
                     kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)

class UNet3DTemporal(nn.Module):
    def __init__(self, in_channels=1, base_filters=32, input_frames=None):
        super().__init__()
        
        self.input_frames = input_frames
        
        # Encoder Path
        self.enc1 = DoubleConv3D(1, base_filters)
        # [B, 1, T, 256, 256] -> [B, 32, T, 256, 256]
        
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # [B, 32, T, 256, 256] -> [B, 32, T/2, 128, 128]
        
        self.enc2 = DoubleConv3D(base_filters, base_filters*2)
        # [B, 32, T/2, 128, 128] -> [B, 64, T/2, 128, 128]
        
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # [B, 64, T/2, 128, 128] -> [B, 64, T/4, 64, 64]
        
        self.enc3 = DoubleConv3D(base_filters*2, base_filters*4)
        # [B, 64, T/4, 64, 64] -> [B, 128, T/4, 64, 64]
        
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # [B, 128, T/4, 64, 64] -> [B, 128, T/8, 32, 32]
        
        # Bottleneck
        self.bottleneck = DoubleConv3D(base_filters*4, base_filters*8)
        # [B, 128, T/8, 32, 32] -> [B, 256, T/8, 32, 32]
        
        # Decoder Path
        self.upconv3 = nn.ConvTranspose3d(
            base_filters*8, base_filters*4,
            kernel_size=2, stride=2
        )
        # [B, 256, T/8, 32, 32] -> [B, 128, T/4, 64, 64]
        
        self.dec3 = DoubleConv3D(base_filters*8, base_filters*4)
        
        self.upconv2 = nn.ConvTranspose3d(
            base_filters*4, base_filters*2,
            kernel_size=2, stride=2
        )
        # [B, 128, T/4, 64, 64] -> [B, 64, T/2, 128, 128]
        
        self.dec2 = DoubleConv3D(base_filters*4, base_filters*2)
        
        self.upconv1 = nn.ConvTranspose3d(
            base_filters*2, base_filters,
            kernel_size=2, stride=2
        )
        # [B, 64, T/2, 128, 128] -> [B, 32, T, 256, 256]
        
        self.dec1 = DoubleConv3D(base_filters*2, base_filters)
        
        # Final temporal reduction
        self.final_temporal_conv = nn.Conv3d(
            base_filters, base_filters,
            kernel_size=(input_frames, 1, 1),
            stride=(1, 1, 1)
        )
        
        self.final_conv = nn.Conv3d(base_filters, 1, kernel_size=1)
        
        self.instance_norm = nn.InstanceNorm3d(1)
        
    def forward(self, x):
        # Input: [B, T, H, W]
        b, t, h, w = x.shape
        assert t == self.input_frames, f"Expected {self.input_frames} frames, got {t}"
        
        # Add channel dimension and normalize
        x = x.unsqueeze(1)  # [B, 1, T, 256, 256]
        x = self.instance_norm(x)
        
        # Encoder Path with skip connections
        e1 = self.enc1(x)         # [B, 32, T, 256, 256]
        p1 = self.pool1(e1)       # [B, 32, T/2, 128, 128]
        
        e2 = self.enc2(p1)        # [B, 64, T/2, 128, 128]
        p2 = self.pool2(e2)       # [B, 64, T/4, 64, 64]
        
        e3 = self.enc3(p2)        # [B, 128, T/4, 64, 64]
        p3 = self.pool3(e3)       # [B, 128, T/8, 32, 32]
        
        # Bottleneck
        bottle = self.bottleneck(p3)  # [B, 256, T/8, 32, 32]
        
        # Decoder Path
        d3 = self.upconv3(bottle)     # [B, 128, T/4, 64, 64]
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)         # [B, 64, T/2, 128, 128]
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)         # [B, 32, T, 256, 256]
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Final convolutions
        out = self.final_temporal_conv(d1)  # [B, 32, 1, 256, 256]
        out = self.final_conv(out)          # [B, 1, 1, 256, 256]
        
        return out.squeeze(2)  # [B, 1, 256, 256]


def test_model():
    # Test with different temporal dimensions
    temporal_sizes = [8, 16, 32]  # Must be multiples of 8 due to 3 pooling layers
    
    for T in temporal_sizes:
        print(f"\nTesting with T={T}")
        model = UNet3DTemporal(input_frames=T)
        x = torch.randn(2, T, 256, 256)  # batch_size=2
        
        try:
            out = model(x)
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {out.shape}")
            print("Test passed!")
        except Exception as e:
            print(f"Error with T={T}: {str(e)}")

# ---------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Down part
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


class AttentionUNet4D(nn.Module):
    """
    4D Attention U-Net architecture that processes sequences of 3D volumes.
    Incorporates attention gates to focus on relevant features during decoding.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        features (list): List of feature dimensions for each level of the U-Net
    """
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()  # Downsampling path modules
        self.ups = nn.ModuleList()    # Upsampling path modules
        self.attention_gates = nn.ModuleList()  # Attention gates
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder pathway
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder pathway with attention gates
        for feature in reversed(features):
            # Add attention gate for each decoder level
            self.attention_gates.append(
                AttentionGate(
                    g_channel=feature*2,    # Gatjjjing signal channels
                    l_channel=feature,      # Input feature channels
                    int_channel=feature//2  # Intermediate channels
                )
            )
            # Upsampling convolution
            self.ups.append(
                nn.ConvTranspose3d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            # Feature processing after concatenation
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x_seq):
        """
        Forward pass of the network.
        
        Args:
            x_seq (torch.Tensor): Input sequence of shape (batch, sequence_len, channels, d, h, w)
        
        Returns:
            torch.Tensor: Output sequence of same temporal length as input
        """
        batch_size, seq_len = x_seq.shape[0], x_seq.shape[1]
        outputs = []

        # Process each timestep independently
        for t in range(seq_len):
            x = x_seq[:,t]
            skip_connections = []

            # Encoder: extract features at multiple scales
            for down in self.downs:
                x = down(x)
                skip_connections.append(x)
                x = self.pool(x)

            x = self.bottleneck(x)
            skip_connections = skip_connections[::-1]  # Reverse for decoder

            # Decoder: reconstruct with attention-guided skip connections
            for idx in range(0, len(self.ups), 2):
                x = self.ups[idx](x)  # Upsample
                skip_connection = skip_connections[idx//2]

                # Apply attention mechanism to skip connection
                attended_skip = self.attention_gates[idx//2](g=x, l=skip_connection)

                # Ensure compatible dimensions for concatenation
                if x.shape != attended_skip.shape:
                    x = F.interpolate(x, size=attended_skip.shape[2:])

                # Concatenate and process features
                concat_skip = torch.cat((attended_skip, x), dim=1)
                x = self.ups[idx+1](concat_skip)

            outputs.append(self.final_conv(x))

        # Combine temporal outputs
        return torch.stack(outputs, dim=1)

class AttentionGate(nn.Module):
    """
    Attention Gate module for focusing on relevant features.
    
    Args:
        g_channel (int): Number of channels in gating signal
        l_channel (int): Number of channels in input features
        int_channel (int): Number of intermediate channels
    """
    def __init__(self, g_channel, l_channel, int_channel):
        super().__init__()
        # Transform gating signal
        self.Wg = nn.Conv3d(g_channel, int_channel, kernel_size=1)
        # Transform input features
        self.Wl = nn.Conv3d(l_channel, int_channel, kernel_size=1)
        # Generate attention weights
        self.psi = nn.Conv3d(int_channel, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, l):
        """
        Forward pass of attention gate.
        
        Args:
            g (torch.Tensor): Gating signal from coarser scale
            l (torch.Tensor): Input features from skip connection
            
        Returns:
            torch.Tensor: Attended features
        """
        g1 = self.Wg(g)
        l1 = self.Wl(l)
        
        # Ensure compatible dimensionsx
        if g1.shape[2:] != l1.shape[2:]:
            g1 = F.interpolate(g1, size=l1.shape[2:])
        
        # Generate attention weights    
        psi = self.relu(g1 + l1)
        psi = self.psi(psi)
        attention = self.sigmoid(psi)
        
        # Apply attention weights to input features
        return l * attention



