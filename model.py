import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class CenterNet(nn.Module):
    def __init__(self, num_classes=5):
        super(CenterNet, self).__init__()

        # 1. BACKBONE: Sử dụng ResNet50 đã được pre-train
        # Lấy từ lớp đầu tiên đến hết layer 4, loại bỏ phần fully connected
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        # Đầu ra của backbone sẽ có kích thước (Batch, 2048, H/32, W/32)

        # 2. NECK: Mạng giải chập (Deconvolution Layers)
        # Kéo độ phân giải từ stride 32 lên stride 4.
        # Độ phân giải lớn (stride 4) là "vũ khí bí mật" để bắt được "bottle" và "cup"
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Đầu ra lúc này là (Batch, 64, H/4, W/4)

        # 3. HEADS: 3 Nhánh dự đoán độc lập (Anchor-Free)

        # Nhánh 1: Heatmap (Dự đoán vị trí tâm và phân loại 5 classes)
        self.hm_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

        # Nhánh 2: Size (Dự đoán chiều rộng w và chiều cao h của hộp bao)
        self.wh_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)
        )

        # Nhánh 3: Offset (Bù đắp sai số tọa độ khi ảnh bị downsample 4 lần)
        self.offset_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)
        )

        # Khởi tạo trọng số đặc biệt cho nhánh Heatmap để lúc mới train không bị văng Loss
        self.hm_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        # Trích xuất đặc trưng
        feat = self.backbone(x)
        out = self.upsample(feat)

        # Đẩy qua 3 nhánh
        hm = self.hm_head(out)
        hm = torch.sigmoid(hm)  # Đưa về [0, 1] để biểu diễn độ tin cậy (confidence / objectness score)

        wh = self.wh_head(out)
        offset = self.offset_head(out)

        # Trả về một dictionary chứa các tensor kết quả
        return {'hm': hm, 'wh': wh, 'offset': offset}


# Test nhanh kích thước tensor để đảm bảo model chạy đúng
if __name__ == "__main__":
    model = CenterNet(num_classes=5)
    dummy_input = torch.randn(2, 3, 512, 512)
    outputs = model(dummy_input)
    print("Heatmap shape:", outputs['hm'].shape)  # Output chuẩn: [2, 5, 128, 128]
    print("Size (wh) shape:", outputs['wh'].shape)  # Output chuẩn: [2, 2, 128, 128]
    print("Offset shape:", outputs['offset'].shape)  # Output chuẩn: [2, 2, 128, 128]