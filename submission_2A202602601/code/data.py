"""data.py — Chuẩn bị dữ liệu cho bài toán Forest CoverType.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import torch

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def _resolve_processed_dir(processed_dir: str) -> Path:
    """Tự động tìm kiếm thư mục processed nếu đường dẫn tương đối thay đổi."""
    p = Path(processed_dir)
    if p.exists() and (p / "train.npz").exists():
        return p
    candidates = [
        Path(processed_dir),
        Path("..") / processed_dir,
        Path("../..") / processed_dir,
        Path("data/processed"),
        Path("../data/processed"),
        Path("../../data/processed"),
    ]
    for c in candidates:
        if c.exists() and (c / "train.npz").exists():
            return c
    return p


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    p = _resolve_processed_dir(processed_dir)
    train_file = p / "train.npz"
    eval_file = p / "eval.npz"

    if not train_file.exists() or not eval_file.exists():
        raise FileNotFoundError(
            f"Không tìm thấy train.npz hoặc eval.npz trong '{p}'. "
            "Vui lòng chạy 'python scripts/split_data.py' từ thư mục gốc trước!"
        )

    train_data = np.load(train_file)
    eval_data = np.load(eval_file)

    X_train_full = train_data["X"].astype(np.float32)
    y_train_full = train_data["y"].astype(np.int64)

    X_eval = eval_data["X"].astype(np.float32)
    y_eval = eval_data["y"].astype(np.int64)
    eval_row_id = eval_data["row_id"].astype(np.int64)

    assert X_train_full.shape == (464809, 54) and X_train_full.dtype == np.float32, (
        f"Lỗi shape/dtype X_train: {X_train_full.shape}, {X_train_full.dtype}"
    )
    assert y_train_full.shape == (464809,) and y_train_full.dtype == np.int64, (
        f"Lỗi shape/dtype y_train: {y_train_full.shape}, {y_train_full.dtype}"
    )
    assert X_eval.shape == (116203, 54) and X_eval.dtype == np.float32, (
        f"Lỗi shape/dtype X_eval: {X_eval.shape}, {X_eval.dtype}"
    )
    assert y_eval.shape == (116203,) and y_eval.dtype == np.int64, (
        f"Lỗi shape/dtype y_eval: {y_eval.shape}, {y_eval.dtype}"
    )
    assert eval_row_id.shape == (116203,), (
        f"Lỗi shape eval_row_id: {eval_row_id.shape}"
    )

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=val_fraction,
        stratify=y,
        random_state=seed
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    Trả lời: Để tránh rò rỉ dữ liệu (data leakage) từ tập validation và eval vào mô hình.
    """
    mean = np.mean(X_tr[:, :N_NUMERIC], axis=0).astype(np.float32)
    std = np.std(X_tr[:, :N_NUMERIC], axis=0).astype(np.float32)
    # Tránh chia cho 0 nếu có cột std = 0
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    X_scaled = X.copy()
    X_scaled[:, :N_NUMERIC] = (X_scaled[:, :N_NUMERIC] - mean) / std
    return X_scaled


def prepare_data(device: str | torch.device, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    device = torch.device(device)
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)

    # 1. Tách val từ train
    X_tr, y_tr, X_val, y_val = make_val_split(
        X_train_full, y_train_full, val_fraction=val_fraction, seed=seed
    )

    # 2. Chuẩn hoá CHỈ dùng mean, std của train con
    mean, std = fit_standardizer(X_tr)
    X_tr_std = apply_standardizer(X_tr, mean, std)
    X_val_std = apply_standardizer(X_val, mean, std)
    X_eval_std = apply_standardizer(X_eval, mean, std)

    # 3. Đưa lên device
    data = {
        "X_tr": torch.tensor(X_tr_std, dtype=torch.float32, device=device),
        "y_tr": torch.tensor(y_tr, dtype=torch.int64, device=device),
        "X_val": torch.tensor(X_val_std, dtype=torch.float32, device=device),
        "y_val": torch.tensor(y_val, dtype=torch.int64, device=device),
        "X_eval": torch.tensor(X_eval_std, dtype=torch.float32, device=device),
        "y_eval": torch.tensor(y_eval, dtype=torch.int64, device=device),
        "eval_row_id": eval_row_id,
    }

    # 4. In thông tin kiểm tra
    val_counts = np.bincount(y_val, minlength=7)
    maj_class = val_counts.argmax()
    maj_acc = val_counts[maj_class] / len(y_val)

    print("=" * 60)
    print("CHUẨN BỊ DỮ LIỆU THÀNH CÔNG:")
    print(f"  Device           : {device}")
    print(f"  Train con (X_tr) : {data['X_tr'].shape}")
    print(f"  Validation (X_val): {data['X_val'].shape}")
    print(f"  Eval cuối (X_eval): {data['X_eval'].shape}")
    print(f"  Mean 10 cột đầu (train): {np.round(mean, 2)}")
    print(f"  Std 10 cột đầu (train) : {np.round(std, 2)}")
    print(f"  Lớp đa số trên val: Lớp {maj_class} ({val_counts[maj_class]}/{len(y_val)} mẫu)")
    print(f"  Accuracy đoán lớp đa số trên val: {maj_acc:.4f} (mốc thấp nhất cần vượt)")
    print("=" * 60)

    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size và được giữ nguyên để không bỏ sót mẫu.
    """
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
