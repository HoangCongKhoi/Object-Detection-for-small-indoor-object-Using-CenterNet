import json
import os
import math
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

CLASSES = ["bottle", "cup", "chair", "laptop", "backpack"]
CLASS_TO_ID = {cls: idx for idx, cls in enumerate(CLASSES)}


def gaussian_radius(det_size, min_overlap=0.7):
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


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


class CenterNetDataset(Dataset):
    def __init__(self, annotation_path, img_dir, input_size=512, down_ratio=4, is_train=True):
        super().__init__()
        self.img_dir = img_dir
        self.input_size = input_size
        self.down_ratio = down_ratio
        self.output_size = input_size // down_ratio  #128x128
        self.is_train = is_train

        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.images_info = {img['id']: img for img in data['images']}

        #group by image_id
        self.annotations = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        self.image_ids = list(self.images_info.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images_info[img_id]
        img_path = os.path.join(self.img_dir,
                                img_info['file_name'].split('/')[-1] if '/' in img_info['file_name'] else img_info[
                                    'file_name'])

        image = cv2.imread(img_path)
        if image is None: raise FileNotFoundError(f"Cannot read image {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image.shape[:2]

        bboxes = []
        classes = []
        if img_id in self.annotations:
            for ann in self.annotations[img_id]:
                bboxes.append(ann['bbox'])
                classes.append(CLASS_TO_ID[ann['class']])
        bboxes = np.array(bboxes, dtype=np.float32) if len(bboxes) > 0 else np.zeros((0, 4))
        classes = np.array(classes, dtype=np.int32)

        # ================= AUGMENTATION =================
        if self.is_train:
            # 1. Random Crop (Cắt ngẫu nhiên 80% - 100% ảnh gốc)
            if np.random.rand() > 0.5:
                crop_scale = np.random.uniform(0.8, 1.0)
                crop_h, crop_w = int(h_orig * crop_scale), int(w_orig * crop_scale)
                top = np.random.randint(0, h_orig - crop_h + 1)
                left = np.random.randint(0, w_orig - crop_w + 1)

                image = image[top:top + crop_h, left:left + crop_w]

                if len(bboxes) > 0:
                    bboxes[:, [0, 2]] -= left
                    bboxes[:, [1, 3]] -= top
                    bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, crop_w)
                    bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, crop_h)

                    # Giữ lại các box còn tồn tại sau khi cắt
                    keep = (bboxes[:, 2] - bboxes[:, 0] > 5) & (bboxes[:, 3] - bboxes[:, 1] > 5)
                    bboxes = bboxes[keep]
                    classes = classes[keep]
                h_orig, w_orig = crop_h, crop_w

            # 2. Lật ngang
            if np.random.rand() > 0.5:
                image = image[:, ::-1, :]
                if len(bboxes) > 0:
                    bboxes[:, [0, 2]] = w_orig - bboxes[:, [2, 0]]

            # 3. Color Jitter
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.7, 1.3)
                beta = np.random.randint(-30, 30)
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        # ================================================

        # Resize & Normalize
        image = cv2.resize(image, (self.input_size, self.input_size))
        if len(bboxes) > 0:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * (self.input_size / w_orig)
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * (self.input_size / h_orig)

        image = image.astype(np.float32) / 255.0
        mean, std = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3), np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        image = (image - mean) / std
        image = image.transpose(2, 0, 1)

        # TARGETS
        num_classes = len(CLASSES)
        hm = np.zeros((num_classes, self.output_size, self.output_size), dtype=np.float32)
        wh = np.zeros((2, self.output_size, self.output_size), dtype=np.float32)
        reg = np.zeros((2, self.output_size, self.output_size), dtype=np.float32)
        reg_mask = np.zeros((1, self.output_size, self.output_size), dtype=np.float32)

        for i in range(len(bboxes)):
            bbox = bboxes[i] / self.down_ratio
            cls_id = classes[i]
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)))))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                draw_umich_gaussian(hm[cls_id], ct_int, radius)
                wh[0, ct_int[1], ct_int[0]] = w
                wh[1, ct_int[1], ct_int[0]] = h
                reg[0, ct_int[1], ct_int[0]] = ct[0] - ct_int[0]
                reg[1, ct_int[1], ct_int[0]] = ct[1] - ct_int[1]
                reg_mask[0, ct_int[1], ct_int[0]] = 1

        return {
            'image': torch.from_numpy(image),
            'hm': torch.from_numpy(hm),
            'wh': torch.from_numpy(wh),
            'reg': torch.from_numpy(reg),
            'reg_mask': torch.from_numpy(reg_mask)
        }