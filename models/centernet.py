import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CenterNet(nn.Module):
    def __init__(self, num_classes=5):
        super(CenterNet, self).__init__()

        # 1. Khởi tạo Backbone ResNet18
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Tách các block của ResNet18 để trích xuất đặc trưng tại nhiều tỉ lệ (FPN)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # Đầu ra: Stride 4, Channels 64
        self.layer2 = resnet.layer2  # Đầu ra: Stride 8, Channels 128
        self.layer3 = resnet.layer3  # Đầu ra: Stride 16, Channels 256
        self.layer4 = resnet.layer4  # Đầu ra: Stride 32, Channels 512

        # 2. Xây dựng Neck (Deconvolution kết hợp Skip-connections)
        # Phóng to từ Stride 32 lên 16
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(inplace=True)

        # Phóng to từ Stride 16 lên 8
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        # Phóng to từ Stride 8 lên 4
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        # 3. Xây dựng Heads (Đầu dự đoán)
        # Heatmap Head: Dự đoán tâm vật thể
        self.hm_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # WH Head: Dự đoán chiều rộng và chiều cao
        self.wh_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Reg Head: Dự đoán phần dư (offset) của tâm điểm ảnh
        self.reg_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Khởi tạo trọng số đặc biệt cho lớp cuối của Heatmap theo bài báo CenterNet
        self.hm_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        # --- ENCODER (Backbone) ---
        x = self.stem(x)
        f1 = self.layer1(x)  # Feature map 1: 64 channels, stride 4
        f2 = self.layer2(f1)  # Feature map 2: 128 channels, stride 8
        f3 = self.layer3(f2)  # Feature map 3: 256 channels, stride 16
        f4 = self.layer4(f3)  # Feature map 4: 512 channels, stride 32

        # --- DECODER (Neck với Skip-connections) ---
        u1 = self.relu1(self.bn1(self.up1(f4)))  # Từ 512 -> 256
        u1 = u1 + f3  # Dung hợp (Fusion) với Feature map 3

        u2 = self.relu2(self.bn2(self.up2(u1)))  # Từ 256 -> 128
        u2 = u2 + f2  # Dung hợp (Fusion) với Feature map 2

        u3 = self.relu3(self.bn3(self.up3(u2)))  # Từ 128 -> 64
        u3 = u3 + f1  # Dung hợp (Fusion) với Feature map 1

        # --- HEADS ---
        hm = self.hm_head(u3)
        hm = torch.clamp(torch.sigmoid(hm), min=1e-4, max=1 - 1e-4)

        wh = self.wh_head(u3)
        reg = self.reg_head(u3)

        return hm, wh, reg