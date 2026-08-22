import os
import json
import argparse
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gdown  # Thư viện tải file từ Google Drive

from model import CenterNet

# Mapping từ ID về tên class để xuất JSON
ID_TO_CLASS_NAME = {0: "bottle", 1: "cup", 2: "chair", 3: "laptop", 4: "backpack"}


# ==========================================
# 1. HÀM KHỬ TRÙNG HỘP BAO (NMS TỰ CÀI ĐẶT)
# ==========================================
def nms_per_class(boxes, scores, labels, iou_threshold=0.4):
    """
    Thực hiện Non-Maximum Suppression độc lập cho từng lớp.
    Không dùng torchvision.ops.nms để đảm bảo tiêu chí "tự cài đặt".
    """
    keep_indices = []

    unique_labels = torch.unique(labels)
    for label in unique_labels:
        # Lấy các hộp bao thuộc lớp hiện tại
        class_indices = (labels == label).nonzero(as_tuple=False).squeeze(1)
        class_boxes = boxes[class_indices]
        class_scores = scores[class_indices]

        # Sắp xếp theo độ tin cậy giảm dần
        sorted_scores, order = class_scores.sort(descending=True)
        sorted_boxes = class_boxes[order]
        sorted_indices = class_indices[order]

        keep_class = []
        while sorted_boxes.size(0) > 0:
            # Luôn giữ lại hộp bao có độ tin cậy cao nhất
            keep_class.append(sorted_indices[0].item())
            if sorted_boxes.size(0) == 1:
                break

            # Tính IoU giữa hộp bao cao nhất và các hộp bao còn lại
            box1 = sorted_boxes[0]
            boxes_rest = sorted_boxes[1:]

            x1 = torch.max(box1[0], boxes_rest[:, 0])
            y1 = torch.max(box1[1], boxes_rest[:, 1])
            x2 = torch.min(box1[2], boxes_rest[:, 2])
            y2 = torch.min(box1[3], boxes_rest[:, 3])

            inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            boxes_rest_area = (boxes_rest[:, 2] - boxes_rest[:, 0]) * (boxes_rest[:, 3] - boxes_rest[:, 1])
            union_area = box1_area + boxes_rest_area - inter_area

            iou = inter_area / union_area

            # Loại bỏ các hộp bao có IoU > iou_threshold
            mask = iou <= iou_threshold
            sorted_boxes = boxes_rest[mask]
            sorted_indices = sorted_indices[1:][mask]

        keep_indices.extend(keep_class)

    return keep_indices


# ==========================================
# 2. GIẢI MÃ ĐẦU RA (DECODE PREDICTIONS)
# ==========================================
def decode_predictions(hm, wh, offset, conf_threshold=0.3):
    """Giải mã Heatmap thành tọa độ Bounding Box"""
    batch, cat, height, width = hm.size()

    # Kỹ thuật NMS cực nhanh của riêng CenterNet: Dùng MaxPool 3x3
    hm_pool = F.max_pool2d(hm, kernel_size=3, padding=1, stride=1)
    keep_mask = (hm == hm_pool).float()
    hm = hm * keep_mask

    # Trải phẳng tensor để tìm top K điểm sáng nhất
    hm = hm.view(batch, -1)
    scores, indices = torch.topk(hm, k=100)  # Lấy tối đa 100 vật thể mỗi ảnh

    classes = (indices // (height * width)).int()
    indices = indices % (height * width)

    ys = (indices // width).float()
    xs = (indices % width).float()

    # Trích xuất width, height, offset
    wh = wh.view(batch, 2, -1)
    offset = offset.view(batch, 2, -1)

    boxes = []
    final_scores = []
    final_classes = []

    for b in range(batch):
        b_scores = scores[b]
        mask = b_scores > conf_threshold

        valid_scores = b_scores[mask]
        valid_classes = classes[b][mask]
        valid_idx = indices[b][mask]
        valid_ys = ys[b][mask]
        valid_xs = xs[b][mask]

        if len(valid_scores) == 0:
            boxes.append(torch.zeros((0, 4)))
            final_scores.append(torch.zeros((0,)))
            final_classes.append(torch.zeros((0,)))
            continue

        # Lấy offset và w, h (lưu ý: gom theo index đã tìm được)
        b_offset = offset[b, :, valid_idx]
        b_wh = wh[b, :, valid_idx]

        valid_xs = valid_xs + b_offset[0, :]
        valid_ys = valid_ys + b_offset[1, :]

        half_w = b_wh[0, :] / 2
        half_h = b_wh[1, :] / 2

        # Tọa độ tại feature map (stride 4) -> Nhân 4 để về kích thước 512x512
        xmin = (valid_xs - half_w) * 4
        ymin = (valid_ys - half_h) * 4
        xmax = (valid_xs + half_w) * 4
        ymax = (valid_ys + half_h) * 4

        b_boxes = torch.stack([xmin, ymin, xmax, ymax], dim=1)

        boxes.append(b_boxes)
        final_scores.append(valid_scores)
        final_classes.append(valid_classes)

    return boxes, final_scores, final_classes


# ==========================================
# 3. HÀM SUY LUẬN CHÍNH
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = './models/best.pth'

    # TỰ ĐỘNG TẢI TRỌNG SỐ NẾU CHƯA CÓ
    if not os.path.exists(weight_path):
        print("[!] Không tìm thấy trọng số tại ./models/best.pth")
        print("[*] Đang tự động tải trọng số từ Google Drive...")
        os.makedirs('./models', exist_ok=True)
        # BẠN THAY file_id BẰNG ID FILE TRÊN GOOGLE DRIVE CỦA BẠN
        file_id = '1a2b3c4d5e6f7g8h9i0j'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, weight_path, quiet=False)

    model = CenterNet(num_classes=5).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # Tiền xử lý giống hệt file utils/transforms.py lúc test
    transform = A.Compose([
        A.LongestMaxSize(max_size=512),
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    results = []

    print("[*] Đang tiến hành suy luận...")
    for img_name in os.listdir(args.image_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(args.image_dir, img_name)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None: continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]

        transformed = transform(image=image_rgb)
        image_tensor = transformed['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)

        # Giải mã và lấy hộp bao
        boxes, scores, labels = decode_predictions(outputs['hm'], outputs['wh'], outputs['offset'], conf_threshold=0.35)

        boxes = boxes[0].cpu()
        scores = scores[0].cpu()
        labels = labels[0].cpu()

        # Áp dụng Custom NMS per class
        if len(boxes) > 0:
            keep = nms_per_class(boxes, scores, labels, iou_threshold=0.45)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

        # ÁNH XẠ TỌA ĐỘ VỀ ẢNH GỐC (Undo Letterbox Padding)
        scale = 512 / max(orig_h, orig_w)
        resized_h, resized_w = int(orig_h * scale), int(orig_w * scale)
        pad_left = (512 - resized_w) // 2
        pad_top = (512 - resized_h) // 2

        pred_boxes = []
        for i in range(len(boxes)):
            xmin = (boxes[i][0].item() - pad_left) / scale
            ymin = (boxes[i][1].item() - pad_top) / scale
            xmax = (boxes[i][2].item() - pad_left) / scale
            ymax = (boxes[i][3].item() - pad_top) / scale

            # Cắt gọn hộp bao không vượt quá viền ảnh
            xmin, ymin = max(0, int(xmin)), max(0, int(ymin))
            xmax, ymax = min(orig_w, int(xmax)), min(orig_h, int(ymax))

            # Chỉ nhận box có kích thước hợp lý
            if xmax > xmin and ymax > ymin:
                pred_boxes.append({
                    "class": ID_TO_CLASS_NAME[labels[i].item()],
                    "confidence": round(scores[i].item(), 3),
                    "bbox": [xmin, ymin, xmax, ymax]
                })

        results.append({
            "image_id": img_name,
            "boxes": pred_boxes
        })

    # Ghi ra JSON
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[*] Hoàn tất! Đã lưu kết quả tại {args.output}")


if __name__ == '__main__':
    main()