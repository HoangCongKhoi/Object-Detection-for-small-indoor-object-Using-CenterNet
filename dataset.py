import os
import json
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from utils.transforms import get_transforms

# Mapping nhãn theo đúng yêu cầu
CLASS_NAME_TO_ID = {"bottle": 0, "cup": 1, "chair": 2, "laptop": 3, "backpack": 4}
ID_TO_CLASS_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}


class CustomObjDetDataset(Dataset):
    def __init__(self, json_file, img_dir, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms

        # Đọc file JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.images_info = {img['id']: img for img in data['images']}
        self.image_ids = list(self.images_info.keys())

        # Gom nhóm annotations theo từng image_id
        self.annotations = {img_id: [] for img_id in self.image_ids}
        for ann in data.get('annotations', []):
            self.annotations[ann['image_id']].append(ann)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images_info[img_id]

        # Xử lý đường dẫn cẩn thận (vì file_name trong json có thể chứa sẵn 'train/images/')
        file_name = os.path.basename(img_info['file_name'])
        img_path = os.path.join(self.img_dir, file_name)

        # Đọc ảnh bằng OpenCV và chuyển BGR sang RGB
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Lấy Bboxes và Labels
        anns = self.annotations[img_id]
        bboxes = []
        labels = []

        for ann in anns:
            xmin, ymin, xmax, ymax = ann['bbox']
            # Chặn lỗi tọa độ âm hoặc bbox sai logic từ dữ liệu thô
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = max(xmin + 1, xmax), max(ymin + 1, ymax)

            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_NAME_TO_ID[ann['class']])

        # Áp dụng Augmentation
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=bboxes, category_ids=labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['category_ids']
        else:
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        # Đưa về Tensor
        if len(bboxes) > 0:
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": bboxes,
            "labels": labels,
            "image_id": img_id
        }

        # Resize Info: Lưu lại tỷ lệ đã scale để lúc Inference (predict.py) suy ngược lại tọa độ gốc
        target["orig_size"] = torch.tensor([img_info['width'], img_info['height']])

        return image, target


# Custom collate_fn rất quan trọng vì số lượng bbox mỗi ảnh là khác nhau
# PyTorch DataLoader mặc định sẽ báo lỗi nếu không có hàm này
def collate_fn(batch):
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    images = torch.stack(images, dim=0)
    return images, targets


# Hàm khởi tạo DataLoader nhanh gọn
def create_dataloaders(train_json, val_json, train_img_dir, val_img_dir, batch_size=4, img_size=512):
    train_dataset = CustomObjDetDataset(train_json, train_img_dir,
                                        transforms=get_transforms(train=True, img_size=img_size))
    val_dataset = CustomObjDetDataset(val_json, val_img_dir, transforms=get_transforms(train=False, img_size=img_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn,
                            pin_memory=True)

    return train_loader, val_loader