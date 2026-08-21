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
        self.grid_size = img_size // stride  # Với img_size=512, stride=16 -> Grid 32x32

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

        # TĂNG CƯỜNG DỮ LIỆU ĐẶC TRỊ CHO CHAIR & BACKPACK
        if is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                # Thay đổi ánh sáng mạnh để cứu balo bị chìm vào nền tối
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.4),
                # Scale & Crop giúp mô hình học các nửa cái ghế / ghế bị cắt viền
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=10, p=0.5,
                                   border_mode=cv2.BORDER_CONSTANT),
                # VŨ KHÍ BÍ MẬT: Giả lập che khuất. Xóa các vùng ngẫu nhiên
                # Ép mô hình phải học toàn bộ đặc trưng của ghế/balo thay vì học vẹt 1 góc
                A.CoarseDropout(max_holes=4, max_height=32, max_width=32, fill_value=0, p=0.4),

                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.3))
            # min_visibility=0.3: Tránh lỗi khi augmentation cắt mất >70% bbox thì bỏ bbox đó đi
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
                # Xử lý nhiễu data: Đôi khi nhãn bị lố ra ngoài ảnh
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_info['width'], x2), min(img_info['height'], y2)

                if x2 > x1 + 2 and y2 > y1 + 2:  # Bỏ qua rác (vật thể quá hẹp)
                    bboxes.append([x1, y1, x2, y2])
                    labels.append(CLASS_TO_IDX[ann['class']])

        # Áp dụng Augmentation
        transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
        image = transformed['image']
        trans_bboxes = transformed['bboxes']
        trans_labels = transformed['labels']

        # Khởi tạo target tensor: 5 class + 1 conf + 4 bbox = 10 channels
        target = torch.zeros((10, self.grid_size, self.grid_size))

        # CHIẾN THUẬT: Sắp xếp bboxes theo diện tích giảm dần.
        # Lý do: Nếu cái cốc (nhỏ) nằm trên cái bàn/ghế (to) và trùng tâm cell,
        # vòng lặp sẽ gán cái to trước, cái nhỏ sau đè lên. Không bị mất object nhỏ.
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in trans_bboxes]
        sorted_indices = np.argsort(areas)[::-1]  # Từ to đến nhỏ

        for idx in sorted_indices:
            bbox = trans_bboxes[idx]
            label = trans_labels[idx]

            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1

            # Tính chỉ số cell trên ô lưới
            cell_x = int(cx / self.stride)
            cell_y = int(cy / self.stride)

            if 0 <= cell_x < self.grid_size and 0 <= cell_y < self.grid_size:
                target[label, cell_y, cell_x] = 1.0  # One-hot Class
                target[5, cell_y, cell_x] = 1.0  # Objectness (Có vật thể)

                # Bounding box regression targets (Scale độc lập với kích thước ảnh)
                target[6, cell_y, cell_x] = (cx / self.stride) - cell_x  # tx (0 -> 1)
                target[7, cell_y, cell_x] = (cy / self.stride) - cell_y  # ty (0 -> 1)
                target[8, cell_y, cell_x] = np.log((w / self.stride) + 1e-8)  # tw
                target[9, cell_y, cell_x] = np.log((h / self.stride) + 1e-8)  # th

        return image, target