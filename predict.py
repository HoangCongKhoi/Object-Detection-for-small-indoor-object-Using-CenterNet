import os
import json
import argparse
import urllib.request
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.ops as ops
from tqdm import tqdm

from models.centernet import CenterNet
from utils.dataset import CLASSES


def download_model_if_missing(checkpoint_path):
    """
    Tự động tải file weight nếu chưa tồn tại.
    SINH VIÊN CẦN THAY THẾ URL DƯỚI ĐÂY bằng link direct tới file best.pth của mình.
    """
    if not os.path.exists(checkpoint_path):
        print(f"[INFO] Weight file không tồn tại ở {checkpoint_path}. Bắt đầu tải xuống...")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        # TODO: Đổi link GitHub Releases hoặc Google Drive của bạn sau khi train xong
        url = "https://github.com/your-username/your-repo/releases/download/v1.0/best.pth"
        try:
            # urllib.request.urlretrieve(url, checkpoint_path) # Bỏ comment khi đã cấu hình URL thật
            print("[WARNING] Bạn chưa cấu hình URL tải file trong predict.py!")
        except Exception as e:
            print(f"Lỗi khi tải weight: {e}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--conf_thresh", type=float, default=0.3)  # Ngưỡng tin cậy
    parser.add_argument("--iou_thresh", type=float, default=0.4)  # Ngưỡng NMS
    return parser.parse_args()


def letterbox_inference(img, expected_size=(512, 512)):
    """Resize ảnh giữ nguyên tỷ lệ để feed vào model, trả về các tham số để scale ngược lại"""
    ih, iw, _ = img.shape
    ew, eh = expected_size
    scale = min(ew / iw, eh / ih)
    nw, nh = int(iw * scale), int(ih * scale)

    img_resized = cv2.resize(img, (nw, nh))
    new_img = np.full((eh, ew, 3), 128, dtype=np.uint8)

    dx, dy = (ew - nw) // 2, (eh - nh) // 2
    new_img[dy:dy + nh, dx:dx + nw, :] = img_resized

    return new_img, scale, dx, dy


def decode_centernet(hm, wh, reg, K=100):
    """Giải mã Heatmap thành Bboxes, sử dụng Max Pooling làm NMS cục bộ"""
    batch, cat, height, width = hm.size()

    # Tìm đỉnh (Peak) bằng Max Pooling 3x3
    hmax = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
    keep = (hmax == hm).float()
    hm = hm * keep

    hm = hm.view(batch, -1)
    scores, indices = torch.topk(hm, K)

    classes = (indices // (height * width)).int()
    indices = indices % (height * width)
    ys = (indices // width).int()
    xs = (indices % width).int()

    reg = reg.view(batch, 2, -1)
    wh = wh.view(batch, 2, -1)

    xs_reg = xs.view(batch, 1, K).expand(batch, 2, K)
    ys_reg = ys.view(batch, 1, K).expand(batch, 2, K)

    out_reg = reg.gather(2, ys_reg * width + xs_reg)
    out_wh = wh.gather(2, ys_reg * width + xs_reg)

    xs = xs.float() + out_reg[:, 0, :]
    ys = ys.float() + out_reg[:, 1, :]
    w = out_wh[:, 0, :]
    h = out_wh[:, 1, :]

    # Nhân 4 (down_ratio) để trả về kích thước 512x512
    bboxes = torch.stack([
        (xs - w / 2) * 4,
        (ys - h / 2) * 4,
        (xs + w / 2) * 4,
        (ys + h / 2) * 4
    ], dim=2)

    return bboxes, scores, classes


def main():
    args = parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    checkpoint_path = os.path.join("models", "best.pth")
    download_model_if_missing(checkpoint_path)

    model = CenterNet(num_classes=5).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("[INFO] Đã tải trọng số mô hình thành công.")
    model.eval()

    results = []
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    image_files = [f for f in os.listdir(args.image_dir) if f.endswith(('.jpg', '.png'))]

    for img_name in tqdm(image_files, desc="Predicting"):
        img_path = os.path.join(args.image_dir, img_name)
        img_orig = cv2.imread(img_path)
        if img_orig is None:
            continue

        h_orig, w_orig = img_orig.shape[:2]
        img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

        # Tiền xử lý Letterbox
        img_pad, scale, dx, dy = letterbox_inference(img_rgb, (512, 512))

        img_norm = img_pad.astype(np.float32) / 255.0
        img_norm = (img_norm - mean) / std
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            hm, wh, reg = model(img_tensor)
            bboxes, scores, classes = decode_centernet(hm, wh, reg, K=100)

        bboxes = bboxes[0]
        scores = scores[0]
        classes = classes[0]

        # Khôi phục tọa độ: Trừ đi phần viền (dx, dy) và chia cho tỷ lệ scale
        bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - dx) / scale
        bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - dy) / scale

        # Kẹp tọa độ chặt chẽ trong phạm vi ảnh thật
        bboxes[:, 0] = torch.clamp(bboxes[:, 0], min=0, max=w_orig)
        bboxes[:, 1] = torch.clamp(bboxes[:, 1], min=0, max=h_orig)
        bboxes[:, 2] = torch.clamp(bboxes[:, 2], min=0, max=w_orig)
        bboxes[:, 3] = torch.clamp(bboxes[:, 3], min=0, max=h_orig)

        img_result = {"image_id": img_name, "boxes": []}

        # NMS độc lập cho từng lớp (Yêu cầu của Rubric)
        for cls_id in range(len(CLASSES)):
            mask = (classes == cls_id) & (scores >= args.conf_thresh)
            cls_boxes = bboxes[mask]
            cls_scores = scores[mask]

            if len(cls_boxes) == 0:
                continue

            # NMS của Torchvision triệt tiêu các hộp bao trùng lặp
            keep_indices = ops.nms(cls_boxes, cls_scores, args.iou_thresh)

            for idx in keep_indices:
                box = cls_boxes[idx].cpu().numpy().tolist()
                score = cls_scores[idx].item()
                img_result["boxes"].append({
                    "class": CLASSES[cls_id],
                    "confidence": round(float(score), 4),
                    "bbox": [round(b, 2) for b in box]
                })

        results.append(img_result)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Đã xuất dự đoán ra {args.output}")


if __name__ == '__main__':
    main()