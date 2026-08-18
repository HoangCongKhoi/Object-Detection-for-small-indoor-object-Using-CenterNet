# Cài đặt mô hình phát hiện đối tượng từ đầu

Cài đặt một mô hình **phát hiện đối tượng** (object detection) từ đầu và đánh giá trên bộ dữ liệu ảnh được cung cấp.

Sinh viên cần tự xây dựng quy trình huấn luyện và suy luận cho bài toán phát hiện đối tượng, bao gồm:

- Đọc dữ liệu và tiền xử lý ảnh/nhãn.
- Tăng cường dữ liệu.
- Mạng trích xuất đặc trưng, ví dụ mạng tích chập CNN.
- Đầu dự đoán phát hiện đối tượng.
- Hàm mất mát.
- Suy luận, ngưỡng độ tin cậy và khử trùng hộp bao bằng NMS (Non-Maximum Suppression).

Không được sử dụng các bộ phát hiện đối tượng hoàn chỉnh như YOLOv5/v8, Detectron2, MMDetection, hoặc Faster R-CNN/SSD có sẵn trong torchvision. Được phép dùng PyTorch, các lớp mạng cơ bản, và mạng trích xuất đặc trưng đã huấn luyện trước nếu giảng viên cho phép.

## Bộ Dữ Liệu

Link download dữ liệu

Bộ dữ liệu gồm ảnh tự nhiên có đối tượng thuộc 5 lớp:

- `bottle`
- `cup`
- `chair`
- `laptop`
- `backpack`

Cấu trúc thư mục:

```text
public/
├── classes.json
├── train/
│   └── images/
├── val/
│   └── images/
├── annotations/
│   ├── train.json
│   └── val.json
└── tools/
    └── evaluate_predictions.py
```

Trong thư mục public/ chỉ có tập huấn luyện và tập kiểm định. Khi chấm tự động, hệ thống mới cung cấp thư mục ảnh kiểm tra ẩn cho `predict.py`; nhãn của tập kiểm tra ẩn được giữ riêng và không công bố.

## Định Dạng Nhãn

Tệp `train.json` và `val.json` có dạng:

```json
{
  "classes": ["bottle", "cup", "chair", "laptop", "backpack"],
  "images": [
    {
      "id": "img_a13f42c9d8b0.jpg",
      "file_name": "train/images/img_a13f42c9d8b0.jpg",
      "width": 500,
      "height": 375
    }
  ],
  "annotations": [
    {
      "image_id": "img_a13f42c9d8b0.jpg",
      "class": "chair",
      "bbox": [48, 72, 210, 356]
    }
  ]
}
```

Quy ước hộp bao:

- `bbox = [xmin, ymin, xmax, ymax]`
- Tọa độ tính theo điểm ảnh trên ảnh gốc.

## Yêu Cầu Kĩ Thuật

### 1. Quy Trình Dữ Liệu

Sinh viên cần cài đặt:

- Bộ đọc dữ liệu.
- Thay đổi kích thước ảnh và chuẩn hóa giá trị điểm ảnh.
- Xử lý nhiều đối tượng trong cùng một ảnh.
- Tăng cường dữ liệu, tối thiểu gồm lật ngang ảnh. Khuyến khích cắt ngẫu nhiên, thay đổi màu sắc, và huấn luyện với nhiều kích thước ảnh.

### 2. Mô Hình Phát Hiện Đối Tượng

Mô hình cần dự đoán:

- Hộp bao.
- Nhãn lớp.
- Điểm độ tin cậy hoặc điểm có đối tượng.

Sinh viên có thể chọn một trong các hướng:

- Mô hình dùng hộp neo (anchor-based).
- Mô hình không dùng hộp neo (anchor-free).
- Mô hình dựa trên lưới kiểu YOLO nhỏ.
- Mô hình kiểu SSD tự cài đặt.

### 3. Hàm Mất Mát

Hàm mất mát cần có các thành phần phù hợp với thiết kế mô hình:

- Mất mát phân lớp.
- Mất mát định vị hộp bao.
- Mất mát độ tin cậy hoặc điểm có đối tượng nếu mô hình có thành phần này.

Khuyến khích dùng Cross Entropy, BCE, Smooth L1, IoU/GIoU/DIoU.

### 4. Suy Luận

Sinh viên cần cài đặt:

- Ngưỡng độ tin cậy.
- NMS theo từng lớp.
- Chuyển hộp bao về tọa độ ảnh gốc.

## Yêu Cầu Nộp Bài

Sinh viên nộp tệp nén của thư mục `<my_submission>/` gồm:

```text
<my_submission>/
├── public/ <vị trí của public/, không cần nộp lại thư mục public>
├── models/
├── utils/
├── train.py
├── predict.py
├── README.md
└── requirements.txt
```

Lệnh suy luận bắt buộc:

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json
```

Lệnh huấn luyện bắt buộc:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```

Lệnh huấn luyện trên phải chạy được và lưu mô hình tốt nhất vào `./models/best.pth`. Sinh viên có thể thêm tham số khác nếu cần.

Khi suy luận, `predict.py` phải tự động tải trọng số về `./models/best.pth` nếu tệp này chưa tồn tại. Sinh viên phải mô tả nguồn tải và cơ chế tải trọng số trong `README.md`.

Không được đưa `./models/best.pth` hoặc bất kỳ tệp trọng số mô hình nào vào tệp nén nộp bài. Bài nộp không tuân thủ hướng dẫn này sẽ bị trừ **1,0 điểm**.

Tệp `README.md` cần nêu rõ:

- Cách cài đặt môi trường.
- Cách huấn luyện.
- Cách chạy suy luận.
- Vị trí đặt mô hình hoặc trọng số mô hình.
- URL và cơ chế tự động tải trọng số khi chạy `predict.py`.

## Định Dạng Kết Quả Dự Đoán

Tệp `predictions.json` phải là một mảng JSON:

```json
[
  {
    "image_id": "img_7fd91a4c2e30.jpg",
    "boxes": [
      {
        "class": "chair",
        "confidence": 0.91,
        "bbox": [48, 72, 210, 356]
      }
    ]
  }
]
```

Quy định:

- `image_id` là tên file ảnh trong thư mục ảnh cần suy luận.
- `class` thuộc 5 lớp quy định.
- `confidence` là độ tin cậy, nằm trong đoạn `[0, 1]`.
- `bbox` là `[xmin, ymin, xmax, ymax]`.
- Tọa độ `bbox` tính theo điểm ảnh trên ảnh gốc.
- Ảnh không phát hiện đối tượng nào vẫn cần xuất `"boxes": []`.
- Trong `public/annotations` có các file predictions.json (chính xác 100%) để làm mẫu.

## Chấm Tự Động

Sinh viên có thể tự kiểm tra định dạng và điểm trên tập huấn luyện/kiểm định bằng tệp chấm đi kèm trong thư mục `public/`. Ví dụ, sau khi xuất dự đoán cho tập kiểm định:

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions val_predictions.json \
  --output val_score.json
```

Hệ thống chấm sẽ:

- Kiểm tra JSON có đúng định dạng không.
- Kiểm tra hộp bao hợp lệ và nằm trong ảnh.
- Kiểm tra nhãn lớp hợp lệ.
- Tính IoU, độ chính xác (precision), độ bao phủ (recall) và mAP@0.5 trên tập kiểm tra ẩn.

Lệnh chấm mẫu:

```bash
python tools/evaluate_predictions.py \
  --ground_truth ./private/hidden_test_annotations.json \
  --predictions predictions.json \
  --output score.json
```

## Thang Điểm

| Nội dung | Điểm |
|---|---:|
| Quy trình dữ liệu | 20 |
| Kiến trúc mô hình | 20 |
| Hàm mất mát và quy trình huấn luyện | 20 |
| Suy luận và NMS | 20 |
| Kết quả trên tập kiểm tra ẩn | 20 |

Thang điểm kết quả:

| mAP@0.5 | Điểm |
|---|---:|
| `< 0.30` | 0 |
| `0.30 - < 0.45` | 5 |
| `0.45 - < 0.60` | 10 |
| `0.60 - < 0.75` | 15 |
| `>= 0.75` | 20 |

Sau đó được chuẩn hóa: người có kết quả cao nhất được 20 điểm, người có kết quả thấp nhất được 0 điểm.

## Quy Định Đạo Văn

Không được sao chép toàn bộ mã nguồn của mô hình phát hiện đối tượng có sẵn.

Có thể tham khảo tài liệu nhưng phải tự triển khai phần chính.

Giảng viên có thể yêu cầu phỏng vấn mã nguồn trực tiếp.

Bài có dấu hiệu sao chép sẽ bị xử lý theo quy chế của nhà trường.
