import torch
import torch.nn as nn
import torch.nn.functional as F


def generalized_iou(box1, box2):
    """
    Tính GIoU (Generalized Intersection over Union).
    Giúp ép hộp bao bám sát góc cạnh vật thể, cực kỳ hiệu quả cho chair và backpack.
    Format input: [x1, y1, x2, y2]
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # Tính phần giao nhau (Intersection)
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Tính diện tích từng box
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area + 1e-6

    iou = inter_area / union_area

    # Tính hộp bao nhỏ nhất chứa cả 2 box (Enclosing box C)
    c_x1 = torch.min(b1_x1, b2_x1)
    c_y1 = torch.min(b1_y1, b2_y1)
    c_x2 = torch.max(b1_x2, b2_x2)
    c_y2 = torch.max(b1_y2, b2_y2)
    c_area = torch.clamp(c_x2 - c_x1, min=0) * torch.clamp(c_y2 - c_y1, min=0) + 1e-6

    # Công thức GIoU = IoU - (Area(C) - Union) / Area(C)
    giou = iou - (c_area - union_area) / c_area
    return giou


class IndoorDetectorLoss(nn.Module):
    def __init__(self, stride=16):
        super().__init__()
        self.stride = stride
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, targets):
        """
        preds, targets shape: (B, 10, S, S)
        Channels: 0-4 (Class), 5 (Objectness), 6-9 (Bbox: tx, ty, tw, th)
        """
        B, C, S, _ = preds.shape
        device = preds.device

        # Tách các thành phần dự đoán
        pred_cls = preds[:, 0:5, :, :]
        pred_obj = preds[:, 5, :, :]
        pred_txty = preds[:, 6:8, :, :]
        pred_twth = preds[:, 8:10, :, :]

        # Tách các thành phần nhãn thực tế (Ground Truth)
        target_cls = targets[:, 0:5, :, :]
        target_obj = targets[:, 5, :, :]
        target_box = targets[:, 6:10, :, :]

        # -------------------------------------------------------------
        # 1. OBJECTNESS LOSS (Dùng Focal Loss để trị triệt để Background)
        # -------------------------------------------------------------
        bce_obj = self.bce(pred_obj, target_obj)
        prob_obj = torch.sigmoid(pred_obj)

        # Alpha=0.25 (cân bằng class), Gamma=2.0 (giảm trọng số cho background dễ đoán)
        alpha, gamma = 0.25, 2.0
        pt = prob_obj * target_obj + (1 - prob_obj) * (1 - target_obj)
        focal_weight = (alpha * target_obj + (1 - alpha) * (1 - target_obj)) * (1 - pt).pow(gamma)

        loss_obj = (bce_obj * focal_weight).mean()

        # Tạo mặt nạ lọc ra các ô lưới có chứa vật thể thực sự
        obj_mask = target_obj == 1

        loss_cls = torch.tensor(0.0, device=device)
        loss_box = torch.tensor(0.0, device=device)

        if obj_mask.sum() > 0:
            # -------------------------------------------------------------
            # 2. CLASSIFICATION LOSS (Cross Entropy cho các ô có vật thể)
            # -------------------------------------------------------------
            active_preds = pred_cls.permute(0, 2, 3, 1)[obj_mask]
            active_targets = torch.argmax(target_cls.permute(0, 2, 3, 1)[obj_mask], dim=1)
            loss_cls = F.cross_entropy(active_preds, active_targets)

            # -------------------------------------------------------------
            # 3. BOUNDING BOX LOSS (GIoU Loss - Siêu vũ khí cho vật góc cạnh)
            # -------------------------------------------------------------
            # Tạo lưới tọa độ cell
            grid_y, grid_x = torch.meshgrid(torch.arange(S, device=device), torch.arange(S, device=device),
                                            indexing='ij')
            grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
            grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

            # Decode tọa độ dự đoán (Predicted Boxes)
            pred_cx = (torch.sigmoid(pred_txty[:, 0, :, :]) + grid_x)[obj_mask] * self.stride
            pred_cy = (torch.sigmoid(pred_txty[:, 1, :, :]) + grid_y)[obj_mask] * self.stride
            pred_w = torch.exp(pred_twth[:, 0, :, :])[obj_mask] * self.stride
            pred_h = torch.exp(pred_twth[:, 1, :, :])[obj_mask] * self.stride

            # Decode tọa độ nhãn thật (Target Boxes)
            target_cx = (target_box[:, 0, :, :] + grid_x)[obj_mask] * self.stride
            target_cy = (target_box[:, 1, :, :] + grid_y)[obj_mask] * self.stride
            target_w = torch.exp(target_box[:, 2, :, :])[obj_mask] * self.stride
            target_h = torch.exp(target_box[:, 3, :, :])[obj_mask] * self.stride

            # Đổi về dạng [x1, y1, x2, y2]
            pred_boxes = torch.stack(
                [pred_cx - pred_w / 2, pred_cy - pred_h / 2, pred_cx + pred_w / 2, pred_cy + pred_h / 2], dim=-1)
            target_boxes = torch.stack([target_cx - target_w / 2, target_cy - target_h / 2, target_cx + target_w / 2,
                                        target_cy + target_h / 2], dim=-1)

            # Tính GIoU Loss
            giou = generalized_iou(pred_boxes, target_boxes)
            loss_box = (1.0 - giou).mean()

        # Tổng hợp Loss với hệ số tối ưu hóa chuyên biệt
        # Tăng trọng số loss_box và loss_obj lên để mô hình tập trung bám sát khung hình và tìm vật thể
        total_loss = 1.0 * loss_cls + 8.0 * loss_obj + 6.0 * loss_box
        return total_loss, loss_cls, loss_obj, loss_box