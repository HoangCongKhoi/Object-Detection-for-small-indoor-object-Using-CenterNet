import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.centernet import CenterNet
from utils.dataset import CenterNetDataset


def focal_loss(pred, target):
    """Penalty-reduced Focal Loss cho CenterNet Heatmap"""
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    neg_weights = torch.pow(1 - target, 4)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    if num_pos == 0:
        return -neg_loss.sum()
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


def gather_feat(feat, ind):
    """Hàm 'nhặt' đặc trưng (WH, Reg) tại đúng các vị trí tâm vật thể do mảng `ind` chỉ định"""
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    return feat


def transpose_and_gather_feat(feat, ind):
    """Đảo chiều Tensor từ (B, C, H, W) sang (B, H*W, C) rồi nhặt đặc trưng"""
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat


def reg_l1_loss(pred, target, mask):
    """L1 Loss sử dụng Mask 1D cho các vật thể thực tế có trong ảnh"""
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
    return loss / (mask.sum() + 1e-4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True, type=str)
    parser.add_argument("--val_data", required=True, type=str)
    parser.add_argument("--image_dir", required=True, type=str)
    parser.add_argument("--val_image_dir", required=True, type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--epochs", type=int, default=30)  # Tăng epochs lên một chút
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[INFO] Bắt đầu huấn luyện trên thiết bị: {device}")

    # Khởi tạo Dataloader
    train_dataset = CenterNetDataset(args.train_data, args.image_dir, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Khởi tạo Model & Optimizer
    model = CenterNet(num_classes=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # THÊM MỚI: Learning Rate Scheduler (Tự động giảm LR khi Loss đi ngang)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in pbar:
            images = batch['image'].to(device)
            target_hm = batch['hm'].to(device)
            target_wh = batch['wh'].to(device)
            target_reg = batch['reg'].to(device)
            ind = batch['ind'].to(device)
            reg_mask = batch['reg_mask'].to(device)

            # 1. Feed Forward
            pred_hm, pred_wh, pred_reg = model(images)

            # 2. Rút trích WH và Reg tại các đỉnh (peaks) thay vì tính toán toàn bộ ảnh
            pred_wh = transpose_and_gather_feat(pred_wh, ind)
            pred_reg = transpose_and_gather_feat(pred_reg, ind)

            # 3. Calculate Losses
            hm_loss = focal_loss(pred_hm, target_hm)
            wh_loss = 0.1 * reg_l1_loss(pred_wh, target_wh, reg_mask)
            reg_loss = 1.0 * reg_l1_loss(pred_reg, target_reg, reg_mask)

            loss = hm_loss + wh_loss + reg_loss

            # 4. Backprop
            optimizer.zero_grad()
            loss.backward()

            # THÊM MỚI: Cắt xén gradient (Gradient Clipping) để tránh bùng nổ gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'HM': f"{hm_loss.item():.4f}",
                'WH': f"{wh_loss.item():.4f}"
            })

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")

        # Cập nhật Scheduler
        scheduler.step(avg_loss)

        # Lưu model tốt nhất
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(args.checkpoint_dir, 'best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] Cập nhật mô hình tốt nhất tại: {save_path}")


if __name__ == '__main__':
    main()