import os
import json
import cv2
import math
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

CLASSES = ["bottle", "cup", "chair", "laptop", "backpack"]
CLASS_TO_ID = {cls: i for i, cls in enumerate(CLASSES)}


def gaussian_radius(det_size, min_overlap=0.7):
    """Tính toán bán kính Gaussian dựa trên kích thước vật thể và ngưỡng IoU tối thiểu"""
    height, width = det_size
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """Vẽ vùng Gaussian xung quanh tâm vật thể trên Heatmap"""
    diameter = 2 * radius + 1
    gaussian = np.zeros((diameter, diameter), dtype=np.float32)

    # THÊM MỚI: Ngăn lỗi chia cho 0 khi radius = 0 (vật thể quá nhỏ)
    sigma = radius / 3 if radius > 0 else 1e-4

    for i in range(diameter):
        for j in range(diameter):
            gaussian[i, j] = np.exp(-((i - radius) ** 2 + (j - radius) ** 2) / (2 * sigma ** 2))

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def letterbox(img, bboxes, expected_size=(512, 512)):
    """Resize ảnh giữ nguyên tỷ lệ, bù viền màu xám, cập nhật lại bboxes"""
    ih, iw, _ = img.shape
    ew, eh = expected_size
    scale = min(ew / iw, eh / ih)
    nw, nh = int(iw * scale), int(ih * scale)

    img_resized = cv2.resize(img, (nw, nh))
    new_img = np.full((eh, ew, 3), 128, dtype=np.uint8)  # Nền xám

    dx, dy = (ew - nw) // 2, (eh - nh) // 2
    new_img[dy:dy + nh, dx:dx + nw, :] = img_resized

    new_bboxes = []
    for bbox in bboxes:
        xmin = bbox[0] * scale + dx
        ymin = bbox[1] * scale + dy
        xmax = bbox[2] * scale + dx
        ymax = bbox[3] * scale + dy
        new_bboxes.append([xmin, ymin, xmax, ymax, bbox[4]])

    return new_img, np.array(new_bboxes), scale, dx, dy


class CenterNetDataset(Dataset):
    def __init__(self, json_file, img_dir, expected_size=(512, 512), is_train=True):
        self.img_dir = img_dir
        self.expected_size = expected_size
        self.is_train = is_train
        self.down_ratio = 4
        self.output_size = (expected_size[0] // self.down_ratio, expected_size[1] // self.down_ratio)

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.images = {img['id']: img for img in data['images']}
        self.annotations = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            bbox = ann['bbox']  # [xmin, ymin, xmax, ymax]
            cls_id = CLASS_TO_ID[ann['class']]
            self.annotations[img_id].append(bbox + [cls_id])

        self.image_ids = list(self.images.keys())

        # Chuẩn hóa ảnh
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'].split('/')[-1])

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bboxes = self.annotations.get(img_id, [])
        bboxes = np.array(bboxes, dtype=np.float32)

        # 1. Letterbox resizing
        img, bboxes, _, _, _ = letterbox(img, bboxes, self.expected_size)

        # 2. Data Augmentation cho tập Train
        if self.is_train:
            # Color Jitter
            img_pil = T.ToPILImage()(img)
            img = np.array(self.color_jitter(img_pil))

            # Lật ngang (Horizontal Flip) xác suất 50%
            if np.random.rand() < 0.5:
                img = cv2.flip(img, 1)
                if len(bboxes) > 0:
                    ew = self.expected_size[0]
                    bboxes[:, [0, 2]] = ew - bboxes[:, [2, 0]]

        # Chuẩn hóa ảnh
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # HWC -> CHW

        # 3. Khởi tạo nhãn đầu ra (Heatmap, WH, Reg)
        num_classes = len(CLASSES)
        out_w, out_h = self.output_size

        hm = np.zeros((num_classes, out_h, out_w), dtype=np.float32)
        wh = np.zeros((100, 2), dtype=np.float32)  # Giới hạn tối đa 100 vật thể
        reg = np.zeros((100, 2), dtype=np.float32)
        ind = np.zeros((100), dtype=np.int64)
        reg_mask = np.zeros((100), dtype=np.uint8)

        # Sinh Ground Truth
        for i, bbox in enumerate(bboxes):
            if i >= 100: break

            # Thu nhỏ bbox theo down_ratio
            bbox_down = bbox[:4] / self.down_ratio
            cls_id = int(bbox[4])

            h, w = bbox_down[3] - bbox_down[1], bbox_down[2] - bbox_down[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))

                ct = np.array([(bbox_down[0] + bbox_down[2]) / 2, (bbox_down[1] + bbox_down[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                # Vẽ Heatmap
                draw_umich_gaussian(hm[cls_id], ct_int, radius)

                # Lưu thông tin W, H và Offset
                wh[i] = 1. * w, 1. * h
                ind[i] = ct_int[1] * out_w + ct_int[0]
                reg[i] = ct - ct_int
                reg_mask[i] = 1

        return {
            'image': torch.from_numpy(img),
            'hm': torch.from_numpy(hm),
            'wh': torch.from_numpy(wh),
            'reg': torch.from_numpy(reg),
            'ind': torch.from_numpy(ind),
            'reg_mask': torch.from_numpy(reg_mask)
        }