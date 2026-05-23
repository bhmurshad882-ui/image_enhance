import torch
import torch.nn as nn

# ==========================================
# 1. ANN Model: CNN Autoencoder
# ==========================================

class CNNAutoencoder(nn.Module):
    def __init__(self):
        super(CNNAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 128 -> 64
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 64 -> 32
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 32 -> 16
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2) # 16 -> 8
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), # 8 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), # 16 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), # 32 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2), # 64 -> 128
            nn.Sigmoid() # Output pixels in range [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# ==========================================
# 2. TransGAN Model: U-Net with Bottleneck Self-Attention (Transformer attention)
# ==========================================

class SelfAttention(nn.Module):
    """
    Self-Attention block for capturing global spatial dependencies (Transformer attention mechanism).
    """
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1) # B x (W*H) x C'
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height) # B x C' x (W*H)
        
        energy = torch.bmm(proj_query, proj_key) # B x (W*H) x (W*H)
        attention = self.softmax(energy) # B x (W*H) x (W*H)
        
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height) # B x C x (W*H)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # B x C x (W*H)
        out = out.view(m_batchsize, C, width, height)
        
        out = self.gamma * out + x
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionUNetGenerator(nn.Module):
    """
    U-Net Generator with Self-Attention in the deepest latent bottleneck.
    This acts as the 'Transformer-based Encoder-Decoder' for translation.
    """
    def __init__(self):
        super(AttentionUNetGenerator, self).__init__()
        
        # Encoder (Down)
        self.inc = DoubleConv(3, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # Bottleneck with Self-Attention (Transformer Attention layer)
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2), # 16 -> 8
            DoubleConv(256, 512),
            SelfAttention(512) # Global spatial attention mapping
        )
        
        # Decoder (Up)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256) # 256 (skip) + 256 = 512 input channels
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(64, 32)
        
        self.outc = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Latent Space
        bottleneck = self.bottleneck(x4)
        
        # Decoder with skip connections
        u1 = self.up1(bottleneck)
        u1 = torch.cat([u1, x4], dim=1)
        u1 = self.conv_up1(u1)
        
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x3], dim=1)
        u2 = self.conv_up2(u2)
        
        u3 = self.up3(u2)
        u3 = torch.cat([u3, x2], dim=1)
        u3 = self.conv_up3(u3)
        
        u4 = self.up4(u3)
        u4 = torch.cat([u4, x1], dim=1)
        u4 = self.conv_up4(u4)
        
        logits = self.outc(u4)
        return logits


class CNNPatchDiscriminator(nn.Module):
    """
    PatchGAN CNN-based classifier for realism (Discriminator).
    Takes either real or enhanced images and outputs a classification patch.
    """
    def __init__(self):
        super(CNNPatchDiscriminator, self).__init__()
        
        # Simple PatchGAN structure
        self.model = nn.Sequential(
            # Input: (3, 128, 128)
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), # -> (64, 64, 64)
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> (128, 32, 32)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # -> (256, 16, 16)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1), # -> (512, 15, 15)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1) # -> (1, 14, 14) classification patch
        )

    def forward(self, x):
        return self.model(x)

# Quick verification test
if __name__ == '__main__':
    x = torch.randn(2, 3, 128, 128)
    
    # Test Autoencoder
    ann = CNNAutoencoder()
    print("ANN Out shape:", ann(x).shape)
    
    # Test TransGAN Generator
    gen = AttentionUNetGenerator()
    print("Generator Out shape:", gen(x).shape)
    
    # Test Discriminator
    disc = CNNPatchDiscriminator()
    print("Discriminator Out shape:", disc(gen(x)).shape)
    
    print("Model definitions successfully verified!")
