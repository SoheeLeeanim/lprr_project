# rm_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + ReLU"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Encoder(nn.Module):
    """
    Encoder module.
    Input:  (B, 3, 128, 128)
    Output: (B, latent_ch, 8, 8)  <- Spatial latent feature map
    """
    def __init__(self, in_ch=3, base_ch=64, latent_ch=256):
        super().__init__()
        # Downsamples 128x128 -> 64x64
        self.b1 = nn.Sequential(
            ConvBlock(in_ch, base_ch),
            ConvBlock(base_ch, base_ch),
            nn.MaxPool2d(2)
        )
        # Downsamples 64x64 -> 32x32
        self.b2 = nn.Sequential(
            ConvBlock(base_ch, base_ch * 2),
            ConvBlock(base_ch * 2, base_ch * 2),
            nn.MaxPool2d(2)
        )
        # Downsamples 32x32 -> 16x16
        self.b3 = nn.Sequential(
            ConvBlock(base_ch * 2, base_ch * 4),
            ConvBlock(base_ch * 4, base_ch * 4),
            nn.MaxPool2d(2)
        )
        # Downsamples 16x16 -> 8x8
        self.b4 = nn.Sequential(
            ConvBlock(base_ch * 4, latent_ch),
            ConvBlock(latent_ch, latent_ch),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        z = self.b4(x)
        return z


class Decoder(nn.Module):
    """
    Decoder module.
    Input:  (B, latent_ch, 8, 8)
    Output: (B, 3, 128, 128)
    """
    def __init__(self, out_ch=3, base_ch=64, latent_ch=256):
        super().__init__()
        # Upsamples 8x8 -> 16x16
        self.u1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(latent_ch, base_ch * 4),
            ConvBlock(base_ch * 4, base_ch * 4)
        )
        # Upsamples 16x16 -> 32x32
        self.u2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(base_ch * 4, base_ch * 2),
            ConvBlock(base_ch * 2, base_ch * 2)
        )
        # Upsamples 32x32 -> 64x64
        self.u3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(base_ch * 2, base_ch),
            ConvBlock(base_ch, base_ch)
        )
        # Upsamples 64x64 -> 128x128
        self.u4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(base_ch, base_ch),
            ConvBlock(base_ch, base_ch)
        )

        self.out = nn.Conv2d(base_ch, out_ch, kernel_size=1)
        # Use Sigmoid to ensure output is in [0,1] range, matching ToTensor()
        self.out_act = nn.Sigmoid()

    def forward(self, z):
        x = self.u1(z)
        x = self.u2(x)
        x = self.u3(x)
        x = self.u4(x)
        x = self.out_act(self.out(x))
        return x


class RM_AutoEncoder(nn.Module):
    """
    Reenactment AutoEncoder with one shared Encoder and two Decoders.
    - dec_s: for source reconstruction
    - dec_t: for target reenactment
    """
    def __init__(self, in_ch=3, base_ch=64, latent_ch=256):
        super().__init__()
        self.enc = Encoder(in_ch=in_ch, base_ch=base_ch, latent_ch=latent_ch)
        self.dec_s = Decoder(out_ch=in_ch, base_ch=base_ch, latent_ch=latent_ch)
        self.dec_t = Decoder(out_ch=in_ch, base_ch=base_ch, latent_ch=latent_ch)

    def forward(self, src_patch):
        z = self.enc(src_patch)
        src_rec = self.dec_s(z)
        tgt_pred = self.dec_t(z)
        return src_rec, tgt_pred
