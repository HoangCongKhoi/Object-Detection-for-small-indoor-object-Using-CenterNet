import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==========================================
# PHẦN 1: TẠO NHÃN (GROUND TRUTH GENERATOR)
# ==========================================
def draw_umich_gaussian(heatmap, center, radius, k=1):
    """
    Vẽ phân phối Gauss 2D lên heatmap tại vị trí tâm vật thể.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = torch.meshgrid(torch.arange(-m, m + 1), torch.arange(-n, n + 1), indexing='ij')
    h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def build_targets(targets, batch_size, num_classes, output_w, output_h, device):
    """
    Chuyển đổi Bbox thô thành Heatmap, Size map và Offset map.
    """
    hm = torch.zeros((batch_size, num_classes, output_h, output_w), device=device)
    wh = torch.zeros((batch_size, 2, output_h, output_w), device=device)
    offset = torch.zeros((batch_size, 2, output_h, output_w), device=device)
    reg_mask = torch.zeros((batch_size, output_h, output_w), device=device)

    for b in range(batch_size):
        boxes = targets[b]['boxes']
        labels = targets[b]['labels']

        for box, label in zip(boxes, labels):
            # Scale tọa độ xuống stride 4 (vì output của mô hình đã bị downsample 4 lần)
            xmin, ymin, xmax, ymax = box / 4.0

            h, w = ymax - ymin, xmax - xmin
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))

                # Tính tâm vật thể
                ct_x, ct_y = (xmin + xmax) / 2, (ymin + ymax) / 2
                ct_int_x, ct_int_y = int(ct_x), int(ct_y)

                # Vẽ Gauss lên Heatmap
                draw_umich_gaussian(hm[b, label], (ct_int_x, ct_int_y), radius)

                # Lưu thông tin Width, Height và Offset tại vị trí tâm
                wh[b, 0, ct_int_y, ct_int_x] = w
                wh[b, 1, ct_int_y, ct_int_x] = h

                offset[b, 0, ct_int_y, ct_int_x] = ct_x - ct_int_x
                offset[b, 1, ct_int_y, ct_int_x] = ct_y - ct_int_y

                # Mask để chỉ tính loss kích thước/offset tại các điểm có vật thể
                reg_mask[b, ct_int_y, ct_int_x] = 1

    return hm, wh, offset, reg_mask


# ==========================================
# PHẦN 2: HÀM MẤT MÁT (LOSS FUNCTIONS)
# ==========================================
class CenterNetLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(CenterNetLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def focal_loss(self, pred, gt):
        """
        Modified Focal Loss chuyên dụng cho Heatmap của CenterNet.
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, self.beta)
        pred = torch.clamp(pred, 1e-4, 1 - 1e-4)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            return -neg_loss
        return -(pos_loss + neg_loss) / num_pos

    def reg_l1_loss(self, pred, mask, target):
        """
        L1 Loss có mặt nạ (Masked L1 Loss) - chỉ tính tại điểm có tâm vật thể.
        """
        pred = pred.permute(0, 2, 3, 1)  # Đưa channel về cuối
        target = target.permute(0, 2, 3, 1)

        # Mở rộng mask cho khớp với số channel (2 channel: w,h hoặc offset_x, offset_y)
        mask = mask.unsqueeze(-1).expand_as(pred)

        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss

    def forward(self, outputs, targets_dict):
        # Trích xuất predictions
        hm_pred = outputs['hm']
        wh_pred = outputs['wh']
        offset_pred = outputs['offset']

        # Lấy kích thước batch và output size
        batch_size, _, output_h, output_w = hm_pred.shape
        device = hm_pred.device

        # Sinh Ground Truth
        hm_gt, wh_gt, offset_gt, reg_mask = build_targets(
            targets_dict, batch_size, num_classes=5,
            output_w=output_w, output_h=output_h, device=device
        )

        # 1. Tính Heatmap Loss (Focal Loss)
        hm_loss = self.focal_loss(hm_pred, hm_gt)

        # 2. Tính Size (Width/Height) Loss (L1) với trọng số 0.1 để tránh lấn át Heatmap
        wh_loss = self.reg_l1_loss(wh_pred, reg_mask, wh_gt)

        # 3. Tính Offset Loss (L1)
        off_loss = self.reg_l1_loss(offset_pred, reg_mask, offset_gt)

        # Tổng hợp Loss
        total_loss = hm_loss + 0.1 * wh_loss + off_loss

        return total_loss, hm_loss, wh_loss, off_loss