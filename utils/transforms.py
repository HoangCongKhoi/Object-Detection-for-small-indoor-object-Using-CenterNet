import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_transforms(train=True, img_size=512):
    """
    Sử dụng Albumentations để Augment data.
    - img_size=512 là đủ để detect vật thể nhỏ mà không quá nặng cho RAM.
    """
    if train:
        return A.Compose([
            # 1. Letterbox Resize: Giữ tỷ lệ khung hình, đệm viền đen
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=0),

            # 2. Tăng cường dữ liệu (Chỉ áp dụng khi train)
            A.HorizontalFlip(p=0.5),  # Lật ngang ảnh
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),  # Thay đổi màu sắc
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4,
                               border_mode=cv2.BORDER_CONSTANT, value=0),  # Dịch, phóng to, xoay nhẹ

            # 3. Chuẩn hóa về Tensor
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.2))
        # format 'pascal_voc' tương đương với [xmin, ymin, xmax, ymax] của đề bài
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))