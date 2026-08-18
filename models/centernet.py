import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CenterNet(nn.Module):
    def __init__(self, num_classes=5):
        super(CenterNet, self).__init__()

        # 1. Khởi tạo Backbone ResNet18 với pretrained weights
        # ResNet18 sẽ giảm kích thước ảnh đi 32 lần (stride 32)
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Loại bỏ 2 lớp cuối cùng (avgpool và fc)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # 2. Xây dựng Neck (Deconvolution) để tăng kích thước đặc trưng
        # Phóng to x2 (từ 1/32 lên 1/16)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # Phóng to x2 (từ 1/16 lên 1/8)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # Phóng to x2 (từ 1/8 lên 1/4)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 3. Xây dựng Heads (3 đầu ra)
        # Heatmap Head (Dự đoán xác suất tâm vật thể cho 5 classes)
        self.hm_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # WH Head (Dự đoán chiều rộng, chiều cao của Box)
        self.wh_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Offset (Reg) Head (Dự đoán độ lệch pixel để tinh chỉnh tâm)
        self.reg_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Khởi tạo trọng số đặc biệt cho lớp cuối của Heatmap
        # Giúp tránh Loss quá lớn ở các Epoch đầu tiên (theo bài báo RetinaNet / CenterNet)
        self.hm_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        # Đặc trưng từ backbone
        feat = self.backbone(x)

        # Upsampling
        feat = self.deconv1(feat)
        feat = self.deconv2(feat)
        feat = self.deconv3(feat)

        # Phân rã ra 3 tensor đầu ra
        hm = self.hm_head(feat)
        # Dùng Sigmoid cho Heatmap để ép giá trị về khoảng [0, 1]
        hm = torch.clamp(torch.sigmoid(hm), min=1e-4, max=1 - 1e-4)

        wh = self.wh_head(feat)
        reg = self.reg_head(feat)

        return hm, wh, reg