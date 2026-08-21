import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASSES = ["bottle", "cup", "chair", "laptop", "backpack"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


class ObjectDetectionDataset(Dataset):
    def __init__(self, annotation_file, img_dir, is_train=True, img_size=512, stride=16):
        self.img_dir = img_dir
        self.img_size = img_size
        self.stride = stride
        self.grid_size = img_size // stride

        with open(annotation_file, 'r') as f:
            data = json.load(f)

        self.images = {img['id']: img for img in data['images']}
        self.annotations = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        self.image_ids = list(self.images.keys())

        if is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.4),
                # Dùng Affine thay thế ShiftScaleRotate để tránh warning phiên bản mới
                A.Affine(scale=(0.85, 1.15), translate_percent=(-0.06, 0.06), rotate=(-10, 10), p=0.5,
                         mode=cv2.BORDER_CONSTANT),
                # CoarseDropout tương thích chuẩn với albumentations mới
                A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.15), hole_width_range=(0.05, 0.15),
                                fill=0, p=0.4),

                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.3))
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'].split('/')[-1])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes, labels = [], []
        if img_id in self.annotations:
            for ann in self.annotations[img_id]:
                x1, y1, x2, y2 = ann['bbox']
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_info['width'], x2), min(img_info['height'], y2)

                if x2 > x1 + 2 and y2 > y1 + 2:
                    bboxes.append([x1, y1, x2, y2])
                    labels.append(CLASS_TO_IDX[ann['class']])

        transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
        image = transformed['image']
        trans_bboxes = transformed['bboxes']
        trans_labels = transformed['labels']

        target = torch.zeros((10, self.grid_size, self.grid_size))

        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in trans_bboxes]
        sorted_indices = np.argsort(areas)[::-1]

        for i in sorted_indices:
            bbox = trans_bboxes[i]
            label = trans_labels[i]

            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1

            cell_x = int(cx / self.stride)
            cell_y = int(cy / self.stride)

            if 0 <= cell_x < self.grid_size and 0 <= cell_y < self.grid_size:
                # Ép kiểu int() cho label, cell_y, cell_x để tránh lỗi IndexError float
                l_idx = int(label)
                cy_idx = int(cell_y)
                cx_idx = int(cell_x)

                target[l_idx, cy_idx, cx_idx] = 1.0
                target[5, cy_idx, cx_idx] = 1.0

                target[6, cy_idx, cx_idx] = (cx / self.stride) - cx_idx
                target[7, cy_idx, cx_idx] = (cy / self.stride) - cy_idx
                target[8, cy_idx, cx_idx] = np.log((w / self.stride) + 1e-8)
                target[9, cy_idx, cx_idx] = np.log((h / self.stride) + 1e-8)

        return image, target