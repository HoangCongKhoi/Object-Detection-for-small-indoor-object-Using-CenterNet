import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CenterNet(nn.Module):
    def __init__(self, num_classes=5):
        super(CenterNet, self).__init__()
        # Backbone ResNet18 with pretrained weights
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remove avgpool and fc
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # Neck(Deconvolution)
        # x2 (1/32 -> 1/16)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # x2 (1/16 -> 1/8)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # x2 (1/8 -> 1/4)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Heatmap Head
        self.hm_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # WH Head (Predict W,H of Box)
        self.wh_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Offset (Reg) Head
        self.reg_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.hm_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        feat = self.backbone(x)
        # Upsampling
        feat = self.deconv1(feat)
        feat = self.deconv2(feat)
        feat = self.deconv3(feat)
        hm = self.hm_head(feat)
        hm = torch.clamp(torch.sigmoid(hm), min=1e-4, max=1 - 1e-4)

        wh = self.wh_head(feat)
        reg = self.reg_head(feat)

        return hm, wh, reg