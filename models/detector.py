import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ConvBlock(nn.Module):
    """
    Block tích chập cơ bản: Conv2d -> BatchNorm -> ReLU
    Giúp chuẩn hóa dòng Gradient, tránh hiện tượng Vanishing Gradient.
    """

    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class IndoorDetector(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # 1. BACKBONE: ResNet50 (Pre-trained)
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Stride 8: Giữ lại chi tiết tốt để bắt các vật nhỏ như "bottle", "cup"
        self.layer2 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2
        )
        # Stride 16: Tầm nhìn trung bình ("laptop", "backpack")
        self.layer3 = backbone.layer3
        # Stride 32: Tầm nhìn bao quát toàn ảnh ("chair" khổng lồ)
        self.layer4 = backbone.layer4

        # 2. NECK: Multi-scale Feature Fusion
        # Giảm số lượng channel về chung 256 để tính toán nhẹ hơn
        self.lat_c4 = nn.Conv2d(2048, 256, 1)
        self.lat_c3 = nn.Conv2d(1024, 256, 1)
        self.lat_c2 = nn.Conv2d(512, 256, 1)

        # Công cụ ép các Feature Map về chung kích thước Stride 16
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.down = nn.Conv2d(256, 256, 3, stride=2, padding=1)

        # Làm mượt sau khi cộng dồn
        self.smooth = ConvBlock(256, 256)

        # 3. DECOUPLED HEAD (Phép màu tăng mAP cho Chair và Backpack)
        # Nhánh 1: Dự đoán Class và Objectness (Có vật thể hay không)
        self.cls_convs = nn.Sequential(
            ConvBlock(256, 256),
            ConvBlock(256, 256)
        )
        self.cls_pred = nn.Conv2d(256, num_classes + 1, 1)  # 5 class + 1 conf

        # Nhánh 2: Dự đoán Tọa độ hộp bao (Bounding Box Regression)
        self.reg_convs = nn.Sequential(
            ConvBlock(256, 256),
            ConvBlock(256, 256)
        )
        self.reg_pred = nn.Conv2d(256, 4, 1)  # tx, ty, tw, th

    def forward(self, x):
        # Trích xuất 3 tầng đặc trưng
        f2 = self.layer2(x)  # (B, 512, S*2, S*2) - Chi tiết nhỏ
        f3 = self.layer3(f2)  # (B, 1024, S, S)    - Trung bình
        f4 = self.layer4(f3)  # (B, 2048, S/2, S/2)- Ngữ cảnh rộng

        # Chuẩn hóa số channel về 256
        p4 = self.lat_c4(f4)
        p3 = self.lat_c3(f3)
        p2 = self.lat_c2(f2)

        # FUSION: Ép tất cả về chung lưới Stride 16
        p4_up = self.up(p4)  # Phóng to Stride 32 lên 16
        p2_down = self.down(p2)  # Thu nhỏ Stride 8 xuống 16

        # Cộng dồn: Lưới Stride 16 lúc này chứa thông tin của cả vật nhỏ bé và khổng lồ
        fused = p3 + p4_up + p2_down
        fused = self.smooth(fused)

        # Đi qua Decoupled Head
        cls_feat = self.cls_convs(fused)
        reg_feat = self.reg_convs(fused)

        cls_out = self.cls_pred(cls_feat)
        reg_out = self.reg_pred(reg_feat)

        # Ghép lại thành Tensor duy nhất (B, 10, S, S) để truyền vào Loss Function
        # Channels: 0-4 (Class), 5 (Objectness), 6-9 (Bbox coords)
        out = torch.cat([cls_out, reg_out], dim=1)

        return out