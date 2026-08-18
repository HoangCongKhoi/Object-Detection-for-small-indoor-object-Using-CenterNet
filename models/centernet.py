import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CenterNet(nn.Module):
    def __init__(self, num_classes=5):
        super(CenterNet, self).__init__()
        # Backbone ResNet18 with pretrained weights
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remove avgpool and fc (Giữ lại các tầng trích xuất đặc trưng)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        for param in list(self.backbone.children())[:5]:
            for p in param.parameters():
                p.requires_grad = False

        # Neck (Deconvolution layers)
        self.deconv1 = self._make_deconv_layer(512, 256)
        self.deconv2 = self._make_deconv_layer(256, 128)
        self.deconv3 = self._make_deconv_layer(128, 64)

        # Heatmap Head (Phân loại tâm vật thể)
        self.hm_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # WH Head (Dự đoán Chiều rộng & Chiều cao của Box)
        self.wh_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Offset (Reg) Head (Dự đoán độ dịch chuyển sub-pixel)
        self.reg_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Gán bias khởi tạo cho heatmap theo chuẩn CenterNet (p = 0.1)
        self.hm_head[-1].bias.data.fill_(-2.19)

        # Khởi tạo trọng số tối ưu chống bùng nổ gradient ở epoch đầu
        self._init_weights()

    def _make_deconv_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _init_weights(self):
        """Khởi tạo trọng số các tầng deconv và head để mô hình hội tụ mượt mà hơn"""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                if m != self.hm_head[-1]:  # Giữ nguyên bias -2.19 đã set riêng cho hm_head
                    nn.init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat = self.backbone(x)

        # Upsampling qua 3 lớp Deconvolution (1/32 -> 1/16 -> 1/8 -> 1/4)
        feat = self.deconv1(feat)
        feat = self.deconv2(feat)
        feat = self.deconv3(feat)

        # Heatmap dự đoán tâm
        hm = self.hm_head(feat)
        hm = torch.clamp(torch.sigmoid(hm), min=1e-4, max=1 - 1e-4)

        # WH: Dùng torch.relu() để đảm bảo kích thước width/height luôn dương (> 0)
        # Khắc phục triệt để lỗi sinh box âm hoặc suy biến kích thước
        wh = torch.relu(self.wh_head(feat))

        # Offset (Reg) dự đoán dịch chuyển tâm
        reg = self.reg_head(feat)

        return hm, wh, reg