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
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    neg_weights = torch.pow(1 - target, 4)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    num_pos = pos_inds.float().sum()
    return -neg_loss.sum() if num_pos == 0 else -(pos_loss.sum() + neg_loss.sum()) / num_pos


def reg_l1_loss(pred, target, mask):
    mask = mask.expand_as(pred)
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
    return loss / (mask.sum() + 1e-4)


def giou_loss(pred_wh, pred_reg, target_wh, target_reg, reg_mask):
    """Tính toán Generalized IoU (GIoU) Loss để tối ưu mAP"""
    mask = reg_mask.squeeze(1).bool()
    if mask.sum() == 0: return torch.tensor(0.0, device=pred_wh.device)

    # Lấy thông số tại các điểm có vật thể
    p_wh = pred_wh.permute(0, 2, 3, 1)[mask]
    p_reg = pred_reg.permute(0, 2, 3, 1)[mask]
    t_wh = target_wh.permute(0, 2, 3, 1)[mask]
    t_reg = target_reg.permute(0, 2, 3, 1)[mask]

    # Tính tọa độ box [x1, y1, x2, y2] tương đối
    p_x1, p_y1 = p_reg[:, 0] - p_wh[:, 0] / 2, p_reg[:, 1] - p_wh[:, 1] / 2
    p_x2, p_y2 = p_reg[:, 0] + p_wh[:, 0] / 2, p_reg[:, 1] + p_wh[:, 1] / 2
    t_x1, t_y1 = t_reg[:, 0] - t_wh[:, 0] / 2, t_reg[:, 1] - t_wh[:, 1] / 2
    t_x2, t_y2 = t_reg[:, 0] + t_wh[:, 0] / 2, t_reg[:, 1] + t_wh[:, 1] / 2

    p_area = p_wh[:, 0] * p_wh[:, 1]
    t_area = t_wh[:, 0] * t_wh[:, 1]

    # Diện tích giao (Intersection)
    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    union_area = p_area + t_area - inter_area + 1e-6
    iou = inter_area / union_area

    # Diện tích bao nhỏ nhất (Convex hull)
    enc_x1 = torch.min(p_x1, t_x1)
    enc_y1 = torch.min(p_y1, t_y1)
    enc_x2 = torch.max(p_x2, t_x2)
    enc_y2 = torch.max(p_y2, t_y2)
    enc_area = torch.clamp(enc_x2 - enc_x1, min=0) * torch.clamp(enc_y2 - enc_y1, min=0) + 1e-6

    giou = iou - (enc_area - union_area) / enc_area
    loss = 1 - giou
    return loss.mean()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--val_image_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)  # TĂNG EPOCH LÊN 100
    parser.add_argument("--batch_size", type=int, default=8)  # Nếu OOM Kaggle, hãy giảm xuống 4
    parser.add_argument("--lr", type=float, default=2e-4)  # Tăng LR khởi điểm một chút
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(CenterNetDataset(args.train_data, args.image_dir, is_train=True),
                              batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(CenterNetDataset(args.val_data, args.val_image_dir, is_train=False),
                            batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = CenterNet(num_classes=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Thêm LR Scheduler giảm dần learning rate hình sin
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in pbar:
            images = batch['image'].float().to(device)
            target_hm, target_wh, target_reg, reg_mask = batch['hm'].to(device), batch['wh'].to(device), batch[
                'reg'].to(device), batch['reg_mask'].to(device)

            pred_hm, pred_wh, pred_reg = model(images)

            # Tổ hợp Loss MỚI
            hm_loss = focal_loss(pred_hm, target_hm)
            wh_loss = 0.1 * reg_l1_loss(pred_wh, target_wh, reg_mask)
            reg_loss = 1.0 * reg_l1_loss(pred_reg, target_reg, reg_mask)
            giou = 1.5 * giou_loss(pred_wh, pred_reg, target_wh, target_reg, reg_mask)  # TRỌNG SỐ LỚN CHO GIOU

            loss = hm_loss + wh_loss + reg_loss + giou

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            pbar.set_postfix(
                {'loss': f"{loss.item():.3f}", 'giou': f"{giou.item():.3f}", 'lr': optimizer.param_groups[0]['lr']})

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation (Tương tự vòng lặp train)
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].float().to(device)
                target_hm, target_wh, target_reg, reg_mask = batch['hm'].to(device), batch['wh'].to(device), batch[
                    'reg'].to(device), batch['reg_mask'].to(device)
                pred_hm, pred_wh, pred_reg = model(images)

                loss = focal_loss(pred_hm, target_hm) + 0.1 * reg_l1_loss(pred_wh, target_wh,
                                                                          reg_mask) + 1.0 * reg_l1_loss(pred_reg,
                                                                                                        target_reg,
                                                                                                        reg_mask) + 1.5 * giou_loss(
                    pred_wh, pred_reg, target_wh, target_reg, reg_mask)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best.pth'))
            print(f"--> Saved best model (Val Loss: {best_val_loss:.4f})")


if __name__ == '__main__':
    main()