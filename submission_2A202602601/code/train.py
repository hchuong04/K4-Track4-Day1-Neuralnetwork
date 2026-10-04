"""train.py — Quản lý vòng lặp huấn luyện, đánh giá và xuất file nộp.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base).
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # Đã kiểm tra hợp lý trên val
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    Đúng chuẩn công thức trong scripts/evaluate.py.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô; gom argmax(dim=1); torch.cat.
    """
    model.eval()
    preds = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        preds.append(torch.argmax(logits, dim=1))
    return torch.cat(preds, dim=0)


@torch.no_grad()
def evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
             loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    n = len(X)

    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)

        if loss_name == "ce":
            batch_loss = F.cross_entropy(logits, yb, reduction="sum")
        elif loss_name == "mse":
            y_onehot = F.one_hot(yb, num_classes=logits.shape[1]).float()
            batch_loss = F.mse_loss(logits, y_onehot, reduction="sum")
        else:
            raise ValueError(f"Hàm loss '{loss_name}' không hợp lệ")

        total_loss += float(batch_loss.item())
        all_preds.append(torch.argmax(logits, dim=1))

    pred_all = torch.cat(all_preds, dim=0)
    y_cpu = y.cpu().numpy()
    pred_cpu = pred_all.cpu().numpy()

    acc = float((pred_cpu == y_cpu).mean())
    avg_loss = float(total_loss / n)

    # Ma trận nhầm lẫn 7x7
    cm = np.zeros((7, 7), dtype=np.int64)
    np.add.at(cm, (y_cpu, pred_cpu), 1)
    macro_f1 = macro_f1_from_confusion(cm)

    return {"loss": avg_loss, "acc": acc, "macro_f1": macro_f1}


def compute_loss(logits: torch.Tensor, y: torch.Tensor, loss_name: str) -> torch.Tensor:
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y.
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name == "mse":
        y_onehot = F.one_hot(y, num_classes=logits.shape[1]).float()
        return F.mse_loss(logits, y_onehot)
    else:
        raise ValueError(f"Hàm mất mát '{loss_name}' không được hỗ trợ (chỉ 'ce' hoặc 'mse')")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)
    """
    # 0. Set seed & tạo model
    seed = cfg.get("seed", 1)
    set_seed(seed)

    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init = cfg.get("init", "he")
    model = MLP(hidden=hidden, dropout=dropout, init=init)

    expected = EXPECTED_PARAMS.get(hidden)
    if expected is not None:
        assert count_params(model) == expected, (
            f"Số tham số không khớp: tính được {count_params(model)}, kỳ vọng {expected}"
        )

    device = data["X_tr"].device
    model.to(device)

    # Tạo optimizer
    lr = float(cfg["lr"])
    weight_decay = float(cfg.get("weight_decay", 0.0))
    momentum = float(cfg.get("momentum", 0.9))
    optimizer = build_optimizer(
        cfg.get("optimizer", "sgd_momentum"),
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum
    )

    precision = cfg.get("precision", "fp32").lower()
    is_cuda = (device.type == "cuda")
    scaler = None
    if precision == "fp16" and is_cuda:
        scaler = torch.amp.GradScaler("cuda")

    # 1. Đo loss bước 0 trên tập val trước khi cập nhật
    loss_name = cfg.get("loss", "ce")
    step0_res = evaluate(model, data["X_val"], data["y_val"], loss_name=loss_name)
    step0_loss = step0_res["loss"]

    # Tập train cố định (50.000 mẫu) để đánh giá train_loss sau mỗi epoch
    n_train_sub = min(50000, len(data["X_tr"]))
    X_tr_sub = data["X_tr"][:n_train_sub]
    y_tr_sub = data["y_tr"][:n_train_sub]

    epochs = int(cfg.get("epochs", 20))
    batch_size = int(cfg.get("batch", 512))
    clip_norm = cfg.get("clip_norm")

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": []
    }

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    diverged = False

    print(f"\nBắt đầu thí nghiệm: {cfg.get('exp_id', 'unnamed')} | Opt: {cfg.get('optimizer')} | lr: {lr} | Epochs: {epochs}")
    print(f"Loss bước 0 trên Val: {step0_loss:.4f} (Kỳ vọng ~ln 7 = 1.946 nếu CE)")

    # 2. Vòng lặp huấn luyện
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        grad_norms_epoch = []

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size=batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            if precision == "fp16" and is_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, loss_name)

                if torch.isnan(loss) or torch.isinf(loss):
                    diverged = True
                    break

                scaler.scale(loss).backward()
                # Phải unscale trước khi đo / cắt gradient
                scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), clip_norm)
                grad_norms_epoch.append(gn)
                scaler.step(optimizer)
                scaler.update()

            elif precision == "bf16" and is_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, loss_name)

                if torch.isnan(loss) or torch.isinf(loss):
                    diverged = True
                    break

                loss.backward()
                gn = clip_gradients(model.parameters(), clip_norm)
                grad_norms_epoch.append(gn)
                optimizer.step()

            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, loss_name)

                if torch.isnan(loss) or torch.isinf(loss):
                    diverged = True
                    break

                loss.backward()
                gn = clip_gradients(model.parameters(), clip_norm)
                grad_norms_epoch.append(gn)
                optimizer.step()

        if is_cuda:
            torch.cuda.synchronize()
        epoch_time = time.time() - t0

        if diverged:
            print(f"  [CẢNH BÁO] Epoch {epoch}: Mô hình phân kỳ (Loss là NaN/Inf)! Dừng sớm.")
            break

        # Cuối epoch: đo đạc ở chế độ eval
        tr_eval = evaluate(model, X_tr_sub, y_tr_sub, loss_name=loss_name)
        val_eval = evaluate(model, data["X_val"], data["y_val"], loss_name=loss_name)
        avg_gn = float(np.mean(grad_norms_epoch)) if grad_norms_epoch else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_eval["loss"])
        history["val_loss"].append(val_eval["loss"])
        history["val_acc"].append(val_eval["acc"])
        history["val_macro_f1"].append(val_eval["macro_f1"])
        history["grad_norm"].append(avg_gn)
        history["epoch_time_s"].append(epoch_time)

        # Cập nhật checkpoint best epoch (dựa trên val_loss thấp nhất)
        if val_eval["loss"] < best_val_loss:
            best_val_loss = val_eval["loss"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {tr_eval['loss']:.4f} | "
                  f"Val Loss: {val_eval['loss']:.4f} | Val Acc: {val_eval['acc']:.4f} | "
                  f"Val F1: {val_eval['macro_f1']:.4f} | GN: {avg_gn:.3f} | {epoch_time:.2f}s")

    # 3. Tổng hợp summary
    peak_mem_MB = 0.0
    if is_cuda:
        peak_mem_MB = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

    best_idx = (best_epoch - 1) if (best_epoch > 0 and len(history["val_loss"]) >= best_epoch) else -1
    summary = {
        "step0_loss": float(step0_loss),
        "best_val_loss": float(best_val_loss) if best_val_loss != float("inf") else None,
        "best_epoch": int(best_epoch) if best_epoch > 0 else None,
        "final_train_loss": float(history["train_loss"][-1]) if history["train_loss"] else None,
        "final_val_loss": float(history["val_loss"][-1]) if history["val_loss"] else None,
        "val_acc": float(history["val_acc"][best_idx]) if best_idx >= 0 else None,
        "val_macro_f1": float(history["val_macro_f1"][best_idx]) if best_idx >= 0 else None,
        "time_per_epoch_s": float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else None,
        "peak_mem_MB": peak_mem_MB,
        "diverged": diverged,
    }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state
    }


def write_predictions(row_id: np.ndarray, preds: np.ndarray, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"row_id": row_id.astype(int), "pred": preds.astype(int)})
    df.to_csv(p, index=False)
    print(f"Đã ghi file dự đoán thành công: {path} ({len(df)} dòng)")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions."""
    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    model = MLP(hidden=hidden, dropout=dropout)

    assert result["best_state"] is not None, "Không tìm thấy best_state trong kết quả huấn luyện!"
    model.load_state_dict(result["best_state"])
    device = data["X_eval"].device
    model.to(device)

    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
