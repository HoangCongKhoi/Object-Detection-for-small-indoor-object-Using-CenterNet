import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import ObjectDetectionDataset
from models.detector import IndoorDetector
from utils.loss import IndoorDetectorLoss


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Đang huấn luyện trên thiết bị: {device} ---")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_dataset = ObjectDetectionDataset(args.train_data, args.image_dir, is_train=True, img_size=512, stride=16)
    val_dataset = ObjectDetectionDataset(args.val_data, args.val_image_dir, is_train=False, img_size=512, stride=16)

    train_loader = DataLoader(train_dataset, args.epochs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, args.epochs, shuffle=False, num_workers=4)

    model = IndoorDetector(num_classes=5).to(device)
    criterion = IndoorDetectorLoss(stride=16).to(device)

    # Optimizer & Scheduler xịn giúp tăng mAP
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)

    best_loss = float('inf')
    num_epochs = 30

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss, l_cls, l_obj, l_box = criterion(preds, targets)

            loss.backward()
            # Chống hiện tượng bùng nổ gradient thường gặp ở object detection
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}", cls=f"{l_cls.item():.3f}",
                             obj=f"{l_obj.item():.3f}", box=f"{l_box.item():.3f}")

        scheduler.step()

        # Đánh giá trên tập Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                preds = model(images)
                loss, _, _, _ = criterion(preds, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"--> [Epoch {epoch + 1}] Validation Loss: {val_loss:.4f}")

        # Lưu mô hình tốt nhất
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best.pth'))
            print(">>> Đã cập nhật mô hình tốt nhất vào models/best.pth <<<")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--val_image_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    train(args)