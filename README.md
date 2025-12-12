# Phân loại Ảnh và Văn bản sử dụng Transformer Encoder

Dự án này bao gồm các mã nguồn Python (Jupyter Notebooks) để minh họa việc áp dụng kiến trúc **Transformer Encoder** (cốt lõi của các mô hình như BERT, ViT) cho hai bài toán học sâu cơ bản: Phân loại ảnh (Image Classification) và Phân loại văn bản (Text Classification).

Dự án làm việc trên 2 bộ dữ liệu chính:
1.  **Dữ liệu CIFAR-10** (Bộ dữ liệu ảnh chuẩn thường dùng trong học máy, được tải tự động qua `torchvision`).
2.  **Dữ liệu TextData** (Bộ dữ liệu văn bản tiếng Việt được phân loại theo nhiều chủ đề khác nhau như Âm nhạc, Ẩm thực, Bất động sản...).

## Cấu trúc Thư mục

```
├── TransformerEncoder_ImageClassification.ipynb  # Notebook phân loại ảnh với Vision Transformer (ViT)
├── TransformerEncoder_TextClassification.ipynb   # Notebook phân loại văn bản tiếng Việt với Transformer
├── TransformerEncoder_MultiTask.ipynb            # Notebook đa nhiệm (Ảnh & Văn bản) với kiến trúc thống nhất
├── README.md                                     # File mô tả dự án
├── data/                                         # Thư mục chứa dữ liệu CIFAR-10 (được tạo tự động sau khi chạy)
└── TextData/                                     # Thư mục chứa dữ liệu văn bản tiếng Việt
    ├── Am nhac/
    ├── Am thuc/
    ├── Bat dong san/
    └── ...
```

### 1. `TransformerEncoder_ImageClassification.ipynb` (Phân loại Ảnh)
Notebook này xây dựng một mô hình lấy cảm hứng từ **Vision Transformer (ViT)** để phân loại ảnh trên bộ dữ liệu CIFAR-10.

*   **Mục tiêu:** Hiểu cách Transformer xử lý dữ liệu hình ảnh thông qua việc chia nhỏ ảnh thành các "patch".
*   **Các bước chính:**
    *   **Tiền xử lý (Preprocessing):**
        *   Tăng cường dữ liệu (Data Augmentation): RandomCrop, RandomHorizontalFlip.
        *   Chuẩn hóa (Normalization) dữ liệu ảnh.
    *   **Patch Embedding:**
        *   Chia ảnh đầu vào (64x64) thành các patch nhỏ (ví dụ 4x4).
        *   Chiếu các patch này vào không gian vector (Embedding) và cộng thêm thông tin vị trí (Positional Embedding).
    *   **Mô hình hóa (Modeling):**
        *   Sử dụng `nn.TransformerEncoder` của PyTorch.
        *   Sử dụng token đặc biệt `[CLS]` để đại diện cho toàn bộ bức ảnh dùng cho việc phân loại.
    *   **Huấn luyện & Đánh giá:**
        *   Sử dụng hàm mất mát `CrossEntropyLoss` và tối ưu hóa `AdamW`.
        *   Đánh giá độ chính xác (Accuracy) trên tập kiểm tra.
    *   **Dự đoán:** Hiển thị ảnh và nhãn dự đoán trực quan.

### 2. `TransformerEncoder_TextClassification.ipynb` (Phân loại Văn bản)
Notebook này áp dụng kiến trúc Transformer Encoder để phân loại chủ đề của các đoạn văn bản tiếng Việt.

*   **Mục tiêu:** Xây dựng mô hình phân loại văn bản mạnh mẽ có khả năng hiểu ngữ cảnh tốt hơn so với RNN/LSTM truyền thống.
*   **Các bước chính:**
    *   **Load dữ liệu (Data Loading):**
        *   Đọc các file `.txt` (encoding utf-16) từ thư mục `TextData`.
        *   Tự động gán nhãn dựa trên tên thư mục chứa file.
    *   **Xử lý văn bản (Text Processing):**
        *   **Tokenization:** Tách văn bản thành các từ/token.
        *   **Vocabulary:** Xây dựng bộ từ điển và ánh xạ từ sang chỉ số (index).
        *   **Padding/Truncating:** Đưa các câu về cùng một độ dài cố định (`MAX_LEN`).
    *   **Mô hình hóa (Modeling):**
        *   **Embedding Layer:** Chuyển đổi index từ thành vector.
        *   **Transformer Encoder:** Xử lý chuỗi vector để trích xuất đặc trưng ngữ cảnh.
        *   **Attention Pooling:** Sử dụng cơ chế Attention để tổng hợp thông tin từ các từ quan trọng thay vì chỉ lấy trung bình (Mean Pooling).
    *   **Huấn luyện (Training):**
        *   Sử dụng cơ chế **Early Stopping** để chống Overfitting.
        *   Lưu lại mô hình tốt nhất (`transformer_text_classification.pth`).
    *   **Dự đoán (Inference):**
        *   Thử nghiệm dự đoán trên các câu tiếng Việt mẫu thuộc nhiều chủ đề khác nhau.

### 3. `TransformerEncoder_MultiTask.ipynb` (Đa Nhiệm: Ảnh & Văn bản)
Notebook này hợp nhất hai tác vụ trên vào một file duy nhất, sử dụng một kiến trúc mô hình linh hoạt có thể chuyển đổi giữa xử lý ảnh và văn bản.

*   **Mục tiêu:** Minh họa tính đa năng của kiến trúc Transformer Encoder (có thể xử lý nhiều loại dữ liệu khác nhau miễn là chúng được chuyển về dạng chuỗi vector).
*   **Điểm nổi bật:**
    *   **Kiến trúc Thống nhất (Unified Architecture):** Sử dụng cùng một `TransformerEncoder` làm nòng cốt (backbone) cho cả hai tác vụ.
    *   **Embedding dạng Mô-đun:**
        *   `PatchEmbedding`: Dùng cho ảnh (Vision Transformer).
        *   `TextEmbedding`: Dùng cho văn bản (NLP).
    *   **Cấu hình Linh hoạt:** Dễ dàng chuyển đổi bài toán bằng cách thay đổi biến `TASK_TYPE` ('image' hoặc 'text').
    *   **Quy trình chuẩn hóa:** Tích hợp toàn bộ quy trình từ Tải dữ liệu -> Huấn luyện -> Đánh giá -> Dự đoán trong một luồng xử lý gọn gàng.

## Yêu cầu cài đặt

Để chạy các notebook, cần cài đặt các thư viện Python sau:

```bash
pip install torch torchvision torchtext scikit-learn tqdm matplotlib numpy datasets
```
*Lưu ý: Cần cài đặt phiên bản PyTorch phù hợp với CUDA nếu muốn sử dụng GPU.*
