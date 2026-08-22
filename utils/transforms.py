import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_transforms(train=True, img_size=512):
    if train:
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            # SỬA: Đổi 'value=0' thành 'fill_value=0' (hoặc 'fill=0' tùy version)
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, fill_value=0),

            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),

            # SỬA: Dùng Affine thay cho ShiftScaleRotate để tương thích version mới
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.4,
                     mode=cv2.BORDER_CONSTANT, cval=0),

            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.2))
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, fill_value=0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))