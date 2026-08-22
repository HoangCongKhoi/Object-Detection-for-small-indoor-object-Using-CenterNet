import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

# Import các module chúng ta đã viết
from dataset import create_dataloaders
from model import CenterNet
from loss import CenterNetLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện CenterNet từ đầu")
    parser.add_argument('--train_data', type=str, required=True, help='Đường dẫn file train.json')
    parser.add_argument('--val_data', type=str, required=True, help='Đường dẫn file val.json')
    parser.add_argument('--image_dir', type=str, required=True, help='Đường dẫn thư mục ảnh train')
    parser.add_argument('--val_image_dir', type=str, required=True, help='Đường dẫn thư mục ảnh val')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Thư mục lưu model best.pth')

    # Các tham số siêu tham số (Hyperparameters)
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (giảm xuống 4 nếu hết VRAM)')
    parser.add_argument('--epochs', type=int, default=50, help='Số lượng epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate ban đầu')
    parser.add_argument('--img_size', type=int, default=512, help='Kích thước ảnh đầu vào')
    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    hm_loss_total, wh_loss_total, off_loss_total = 0.0, 0.0, 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, targets in pbar:
        images = images.to(device)

        # Optimizer zero grad
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Tính Loss
        loss, hm_loss, wh_loss, off_loss = criterion(outputs, targets)

        # Backward pass & Tối ưu trọng số
        loss.backward()
        # Clip gradient để tránh nổ gradient (Exploding Gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        # Cập nhật thông số
        running_loss += loss.item()
        hm_loss_total += hm_loss.item()
        wh_loss_total += wh_loss.item()
        off_loss_total += off_loss.item()

        pbar.set_postfix({'loss': loss.item(), 'hm': hm_loss.item()})

    num_batches = len(dataloader)
    return running_loss / num_batches, hm_loss_total / num_batches


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False)
    for images, targets in pbar:
        images = images.to(device)
        outputs = model(images)
        loss, _, _, _ = criterion(outputs, targets)

        running_loss += loss.item()
        pbar.set_postfix({'val_loss': loss.item()})

    return running_loss / len(dataloader)


def main():
    args = parse_args()

    # Tạo thư mục checkpoint nếu chưa có
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Bắt đầu huấn luyện trên thiết bị: {device}")

    # 1. Khởi tạo DataLoaders
    print("[*] Đang tải dữ liệu...")
    train_loader, val_loader = create_dataloaders(
        train_json=args.train_data,
        val_json=args.val_data,
        train_img_dir=args.image_dir,
        val_img_dir=args.val_image_dir,
        batch_size=args.batch_size,
        img_size=args.img_size
    )

    # 2. Khởi tạo Mô hình và Hàm mất mát
    model = CenterNet(num_classes=5).to(device)
    criterion = CenterNetLoss().to(device)

    # 3. Optimizer và Scheduler
    # Dùng AdamW với weight_decay giúp regularization tốt hơn Adam thường
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float('inf')

    # 4. Vòng lặp huấn luyện (Training Loop)
    for epoch in range(1, args.epochs + 1):
        train_loss, hm_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, val_loader, criterion, device, epoch)

        # Giảm learning rate
        scheduler.step()

        print(f"Epoch {epoch:03d}/{args.epochs:03d} | "
              f"Train Loss: {train_loss:.4f} (HM: {hm_loss:.4f}) | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Lưu best model dựa trên val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.checkpoint_dir, 'best.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f" -> Đã lưu mô hình tốt nhất tại Epoch {epoch} (Val Loss giảm xuống {val_loss:.4f})")


if __name__ == '__main__':
    main()