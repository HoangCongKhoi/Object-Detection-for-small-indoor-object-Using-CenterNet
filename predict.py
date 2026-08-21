import argparse
import os
import json
import cv2
import torch
import requests
from tqdm import tqdm
from torchvision.transforms import functional as F

from models.detector import IndoorDetector
from utils.metrics import custom_nms
from utils.dataset import CLASSES


def download_weights(save_path):
    """
    Cơ chế tự động tải trọng số khi file best.pth chưa tồn tại.
    LƯU Ý: Bạn cần thay thế URL bên dưới bằng link tải trực tiếp từ Google Drive hoặc GitHub Releases của bạn sau khi train xong.
    """
    url = "https://github.com/your-username/your-repo/releases/download/v1.0/best.pth"
    print(f"Không tìm thấy trọng số cục bộ. Đang tự động tải từ: {url}")
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Tải trọng số thành công!")
        else:
            print("Cảnh báo: Không thể tải từ URL trên. Vui lòng kiểm tra lại đường dẫn.")
    except Exception as e:
        print(f"Lỗi kết nối khi tải trọng số: {e}")


def decode_predictions(preds, conf_thresh=0.25, stride=16, img_size=512):
    pred_cls = torch.softmax(preds[0:5, :, :], dim=0)
    pred_obj = torch.sigmoid(preds[5, :, :])
    pred_txty = torch.sigmoid(preds[6:8, :, :])
    pred_twth = torch.exp(preds[8:10, :, :])

    S = preds.shape[1]
    grid_y, grid_x = torch.meshgrid(torch.arange(S), torch.arange(S), indexing='ij')

    scores_obj = pred_obj
    mask = scores_obj > conf_thresh

    if mask.sum() == 0:
        return []

    grid_x = grid_x[mask]
    grid_y = grid_y[mask]

    scores = scores_obj[mask]
    classes = torch.argmax(pred_cls[:, mask], dim=0)

    cx = (pred_txty[0][mask] + grid_x) * stride
    cy = (pred_txty[1][mask] + grid_y) * stride
    w = pred_twth[0][mask] * stride
    h = pred_twth[1][mask] * stride

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    boxes = torch.clamp(boxes, min=0, max=img_size)

    results = []
    for c in range(5):
        c_mask = classes == c
        if c_mask.sum() == 0:
            continue

        c_boxes = boxes[c_mask]
        c_scores = scores[c_mask]

        # Áp dụng NMS riêng biệt cho từng class
        keep = custom_nms(c_boxes, c_scores, iou_threshold=0.45)
        for k in keep:
            results.append({
                "class": CLASSES[c],
                "confidence": round(float(c_scores[k]), 4),
                "bbox": c_boxes[k].tolist()
            })
    return results


def predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "./models/best.pth"

    # Tự động tải file tạ nếu chưa có
    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        download_weights(model_path)

    model = IndoorDetector(num_classes=5).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    results = []
    image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for img_name in tqdm(image_files, desc="Đang suy luận (Inference)"):
        img_path = os.path.join(args.image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        h_orig, w_orig = image.shape[:2]

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(image_rgb, (512, 512))

        tensor = F.to_tensor(img_resized)
        tensor = F.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(tensor)[0].cpu()

        boxes_pred = decode_predictions(preds, conf_thresh=0.25, stride=16, img_size=512)

        # Khôi phục tọa độ về kích thước ảnh gốc
        scale_x = w_orig / 512.0
        scale_y = h_orig / 512.0

        final_boxes = []
        for det in boxes_pred:
            x1, y1, x2, y2 = det['bbox']
            final_boxes.append({
                "class": det['class'],
                "confidence": det['confidence'],
                "bbox": [
                    round(x1 * scale_x, 2),
                    round(y1 * scale_y, 2),
                    round(x2 * scale_x, 2),
                    round(y2 * scale_y, 2)
                ]
            })

        results.append({
            "image_id": img_name,
            "boxes": final_boxes
        })

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Đã xuất kết quả dự đoán thành công vào: {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    predict(args)