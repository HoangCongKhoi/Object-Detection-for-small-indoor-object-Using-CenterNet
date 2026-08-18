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


def reg_l1_loss(pred, target, mask):
    """L1 Loss có Mask (chỉ tính sai số tại pixel là tâm vật thể, bỏ qua pixel nền)"""
    mask = mask.expand_as(pred)
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
    return loss / (mask.sum() + 1e-4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True, type=str)
    parser.add_argument("--val_data", required=True, type=str)
    parser.add_argument("--image_dir", required=True, type=str)
    parser.add_argument("--val_image_dir", required=True, type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Chọn Device (Hỗ trợ CUDA cho NVIDIA, MPS cho Mac, Fallback về CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Training on: {device}")

    # Khởi tạo Dataloader
    train_dataset = CenterNetDataset(args.train_data, args.image_dir, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Khởi tạo Model & Optimizer
    model = CenterNet(num_classes=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
            reg_mask = batch['reg_mask'].to(device)

            # Feed Forward
            pred_hm, pred_wh, pred_reg = model(images)

            # Calculate Losses
            hm_loss = focal_loss(pred_hm, target_hm)
            wh_loss = 0.1 * reg_l1_loss(pred_wh, target_wh, reg_mask)  # Trọng số WH nhỏ lại
            reg_loss = 1.0 * reg_l1_loss(pred_reg, target_reg, reg_mask)

            loss = hm_loss + wh_loss + reg_loss

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'hm': hm_loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")

        # Lưu model tốt nhất
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(args.checkpoint_dir, 'best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")


if __name__ == '__main__':
    main()