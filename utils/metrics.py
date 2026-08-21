import torch


def iou_pytorch(boxes1, boxes2):
    """
    Tính IoU giữa các hộp bao theo chuẩn PyTorch.
    boxes format: [x1, y1, x2, y2]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = torch.clamp(rb - lt, min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter + 1e-6
    return inter / union


def custom_nms(boxes, scores, iou_threshold=0.45):
    """
    Non-Maximum Suppression (NMS) tự code.
    Giúp loại bỏ các hộp bao chồng chéo (ví dụ: nhiều box cùng bao quanh một chiếc ghế).
    """
    if len(boxes) == 0:
        return []

    keep = []
    idxs = scores.argsort(descending=True)

    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current.item())
        if len(idxs) == 1:
            break

        ious = iou_pytorch(boxes[current].unsqueeze(0), boxes[idxs[1:]]).squeeze(0)
        # Giữ lại các box có IoU thấp hơn ngưỡng (không bị trùng lặp)
        idxs = idxs[1:][ious <= iou_threshold]

    return keep