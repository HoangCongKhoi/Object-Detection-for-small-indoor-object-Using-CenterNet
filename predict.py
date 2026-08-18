import os
import json
import argparse
import urllib.request
import cv2
import torch
import torch.nn.functional as F
import torchvision.ops as ops
from tqdm import tqdm

from models.centernet import CenterNet
from utils.dataset import CLASSES, CLASS_TO_ID


def download_model_if_missing(checkpoint_path):
    """
    Tự động tải file weight nếu chưa tồn tại.
    SINH VIÊN CẦN THAY THẾ URL DƯỚI ĐÂY bằng link direct tới file best.pth của mình
    (Ví dụ: Github Releases, hoặc Google Drive direct link).
    """
    if not os.path.exists(checkpoint_path):
        print(f"[INFO] Weight file không tồn tại ở {checkpoint_path}. Bắt đầu tải xuống...")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        # THAY THẾ LINK NÀY KHI BẠN ĐÃ HUẤN LUYỆN XONG VÀ UPLOAD LÊN MẠNG
        url = "https://github.com/your-username/your-repo/releases/download/v1.0/best.pth"
        try:
            # urllib.request.urlretrieve(url, checkpoint_path) # Bỏ comment khi có URL thật
            print("[WARNING] Bạn chưa cấu hình URL tải file trong predict.py!")
            print("[WARNING] Quá trình dự đoán sẽ bị lỗi do không tìm thấy weight.")
        except Exception as e:
            print(f"Lỗi khi tải weight: {e}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--conf_thresh", type=float, default=0.35)  # Ngưỡng độ tin cậy
    parser.add_argument("--iou_thresh", type=float, default=0.5)  # Ngưỡng NMS
    return parser.parse_args()


def decode_centernet(hm, wh, reg, K=100):
    """Giải mã Heatmap thành Bboxes (Max-pool đóng vai trò như NMS sơ bộ)"""
    batch, cat, height, width = hm.size()

    # 3x3 Max pooling trên heatmap để tìm đỉnh (Peak)
    hmax = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
    keep = (hmax == hm).float()
    hm = hm * keep  # Xóa các pixel không phải đỉnh sáng nhất

    hm = hm.view(batch, -1)
    scores, indices = torch.topk(hm, K)

    classes = (indices // (height * width)).int()
    indices = indices % (height * width)
    ys = (indices // width).int()
    xs = (indices % width).int()

    # Lấy thông số W, H, Offset tại các đỉnh tương ứng
    reg = reg.view(batch, 2, -1)
    wh = wh.view(batch, 2, -1)

    xs_reg = xs.view(batch, 1, K).expand(batch, 2, K)
    ys_reg = ys.view(batch, 1, K).expand(batch, 2, K)

    # Trích xuất
    out_reg = reg.gather(2, ys_reg * width + xs_reg)
    out_wh = wh.gather(2, ys_reg * width + xs_reg)

    xs = xs.float() + out_reg[:, 0, :]
    ys = ys.float() + out_reg[:, 1, :]
    w = out_wh[:, 0, :]
    h = out_wh[:, 1, :]

    # Tính tọa độ Bbox (scale lại kích thước đầu vào của mạng là * 4)
    bboxes = torch.stack([
        (xs - w / 2) * 4,
        (ys - h / 2) * 4,
        (xs + w / 2) * 4,
        (ys + h / 2) * 4
    ], dim=2)

    return bboxes, scores, classes


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    checkpoint_path = os.path.join("models", "best.pth")
    download_model_if_missing(checkpoint_path)
    print(f"\n[DEBUG] Đang load weight từ: {os.path.abspath(checkpoint_path)}\n")

    model = CenterNet(num_classes=5).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    results = []

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    image_files = [f for f in os.listdir(args.image_dir) if f.endswith(('.jpg', '.png'))]

    for img_name in tqdm(image_files, desc="Predicting"):
        img_path = os.path.join(args.image_dir, img_name)
        img_orig = cv2.imread(img_path)
        h_orig, w_orig = img_orig.shape[:2]

        img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)

        img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

        with torch.no_grad():
            hm, wh, reg = model(img_tensor)
            bboxes, scores, classes = decode_centernet(hm, wh, reg, K=100)

        bboxes = bboxes[0]
        scores = scores[0]
        classes = classes[0]

        # Scale về kích thước ảnh gốc
        bboxes[:, [0, 2]] *= (w_orig / 512)
        bboxes[:, [1, 3]] *= (h_orig / 512)

        # Kẹp tọa độ không vượt quá ảnh
        bboxes[:, 0] = torch.clamp(bboxes[:, 0], min=0, max=w_orig)
        bboxes[:, 1] = torch.clamp(bboxes[:, 1], min=0, max=h_orig)
        bboxes[:, 2] = torch.clamp(bboxes[:, 2], min=0, max=w_orig)
        bboxes[:, 3] = torch.clamp(bboxes[:, 3], min=0, max=h_orig)

        img_result = {"image_id": img_name, "boxes": []}

        # NMS theo TỪNG LỚP độc lập (Yêu cầu khắt khe của Rubric)
        for cls_id in range(len(CLASSES)):
            mask = (classes == cls_id) & (scores >= args.conf_thresh)
            cls_boxes = bboxes[mask]
            cls_scores = scores[mask]

            if len(cls_boxes) == 0:
                continue

            # Áp dụng NMS của Torchvision
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
    print(f"\n[INFO] Đã lưu kết quả suy luận vào {args.output}")


if __name__ == '__main__':
    import numpy as np

    main()