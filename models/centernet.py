import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class CenterNet(nn.Module):
    def __init__(self, num_classes=5):
        super(CenterNet, self).__init__()
        # Dùng ResNet50 mạnh hơn, bắt đặc trưng tốt hơn
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # BỎ ĐÓNG BĂNG (Unfreeze) toàn bộ trọng số để mạng học lại từ đầu trên miền dữ liệu mới
        for param in self.backbone.parameters():
            param.requires_grad = True

        # Neck (Deconvolution layers)
        # ResNet50 layer cuối xuất ra 2048 channels (khác với 512 của ResNet18)
        self.deconv1 = self._make_deconv_layer(2048, 256)
        self.deconv2 = self._make_deconv_layer(256, 128)
        self.deconv3 = self._make_deconv_layer(128, 64)

        # Heads
        self.hm_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.wh_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.reg_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Gán bias khởi tạo cho heatmap
        self.hm_head[-1].bias.data.fill_(-2.19)
        self._init_weights()

    def _make_deconv_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                if m != self.hm_head[-1]:
                    nn.init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.deconv1(feat)
        feat = self.deconv2(feat)
        feat = self.deconv3(feat)

        hm = self.hm_head(feat)
        hm = torch.clamp(torch.sigmoid(hm), min=1e-4, max=1 - 1e-4)

        # Dùng clamp để tránh width/height bị bằng 0 gây lỗi chia 0 trong GIoU
        wh = torch.clamp(self.wh_head(feat), min=1e-4)
        reg = self.reg_head(feat)

        return hm, wh, reg