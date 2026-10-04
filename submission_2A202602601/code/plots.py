"""plots.py — Vẽ đồ thị huấn luyện và biểu đồ so sánh.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính, có nhãn trục và chú thích.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    cfg = result["cfg"]
    hist = result["history"]
    epochs = hist["epoch"]
    best_epoch = result["summary"].get("best_epoch")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Ô 1: Train Loss & Val Loss
    axes[0].plot(epochs, hist["train_loss"], label="Train Loss (eval mode)", color="#1f77b4", marker="o", markersize=3)
    axes[0].plot(epochs, hist["val_loss"], label="Val Loss", color="#ff7f0e", marker="s", markersize=3)
    if best_epoch:
        axes[0].axvline(best_epoch, color="red", linestyle="--", alpha=0.7, label=f"Best Ep ({best_epoch})")
    axes[0].set_title("Train & Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Ô 2: Val Accuracy & Macro-F1
    axes[1].plot(epochs, hist["val_acc"], label="Val Accuracy", color="#2ca02c", marker="^", markersize=3)
    axes[1].plot(epochs, hist["val_macro_f1"], label="Val Macro-F1", color="#9467bd", marker="d", markersize=3)
    if best_epoch:
        axes[1].axvline(best_epoch, color="red", linestyle="--", alpha=0.7)
    axes[1].set_title("Val Accuracy & Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Ô 3: Gradient Norm (trước khi clip)
    axes[2].plot(epochs, hist["grad_norm"], label="Grad Norm (pre-clip)", color="#d62728", marker="x", markersize=3)
    axes[2].set_title("Average Gradient Norm")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("L2 Norm")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # Tiêu đề chung
    title_str = (
        f"[{cfg.get('exp_id')}] Opt: {cfg.get('optimizer')} | lr: {cfg.get('lr')} | "
        f"batch: {cfg.get('batch')} | init: {cfg.get('init')} | dropout: {cfg.get('dropout')}"
    )
    fig.suptitle(title_str, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu biểu đồ thành công: {path}")


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số của nhiều thí nghiệm trên cùng một trục, mỗi thí nghiệm một đường."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        exp_id = r["cfg"].get("exp_id", "unnamed")
        epochs = r["history"]["epoch"]
        vals = r["history"].get(metric, [])
        ax.plot(epochs, vals, marker="o", markersize=3, label=exp_id)

    ax.set_title(title or f"So sánh {metric} giữa các thí nghiệm", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu ảnh so sánh thành công: {path}")


def plot_all_comparisons(all_results: dict[str, dict], out_dir: str = "../figures") -> None:
    """Tự động gom nhóm các thí nghiệm và vẽ 7 ảnh so sánh chuẩn RUBRIC."""
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)

    # 1. So sánh Loss: base-s1 vs loss-mse
    loss_exps = [all_results[k] for k in ["base-s1", "loss-mse"] if k in all_results]
    if len(loss_exps) >= 2:
        plot_compare(loss_exps, "val_macro_f1", str(p / "compare_loss.png"), "Chủ đề 1 — So sánh Loss: CE vs MSE (Val Macro-F1)")

    # 2. So sánh Optimizer: base-s1, opt-sgd-lr0.05, opt-adam-lr1e-3, opt-adam-lr3e-4, opt-adamw-lr1e-3
    opt_keys = ["base-s1", "opt-sgd-lr0.05", "opt-adam-lr1e-3", "opt-adam-lr3e-4", "opt-adamw-lr1e-3"]
    opt_exps = [all_results[k] for k in opt_keys if k in all_results]
    if len(opt_exps) >= 2:
        plot_compare(opt_exps, "val_loss", str(p / "compare_optimizer.png"), "Chủ đề 2 — So sánh các Bộ tối ưu (Val Loss)")

    # 3. So sánh Hparam: Batch size
    batch_keys = ["hparam-batch128", "base-s1", "hparam-batch2048"]
    batch_exps = [all_results[k] for k in batch_keys if k in all_results]
    if len(batch_exps) >= 2:
        plot_compare(batch_exps, "val_macro_f1", str(p / "compare_hparam_batch.png"), "Chủ đề 3 — So sánh Batch Size (Val Macro-F1)")

    # 3b. So sánh Kiến trúc: M-base, M-wide, M-deep
    arch_keys = ["base-s1", "hparam-wide", "hparam-deep"]
    arch_exps = [all_results[k] for k in arch_keys if k in all_results]
    if len(arch_exps) >= 2:
        plot_compare(arch_exps, "val_macro_f1", str(p / "compare_architecture.png"), "Chủ đề 3 — So sánh Kiến trúc M-base, M-wide, M-deep")

    # 4. So sánh Dropout: base-s1 (q=0), drop-0.1, drop-0.3
    drop_keys = ["base-s1", "drop-0.1", "drop-0.3"]
    drop_exps = [all_results[k] for k in drop_keys if k in all_results]
    if len(drop_exps) >= 2:
        plot_compare(drop_exps, "val_loss", str(p / "compare_dropout.png"), "Chủ đề 4 — So sánh Dropout (Val Loss)")

    # 5. So sánh Gradient Clipping (lr=1.0: không clip vs có clip 1.0)
    clip_keys = ["clip-highlr-noclip", "clip-highlr-clip1.0"]
    clip_exps = [all_results[k] for k in clip_keys if k in all_results]
    if len(clip_exps) >= 2:
        plot_compare(clip_exps, "val_loss", str(p / "compare_clipping.png"), "Chủ đề 5 — Cứu vãn huấn luyện ở lr cao bằng Clipping (Val Loss)")

    # 6. So sánh Mixed Precision: base-s1 (fp32) vs amp-fp16
    amp_keys = ["base-s1", "amp-fp16"]
    amp_exps = [all_results[k] for k in amp_keys if k in all_results]
    if len(amp_exps) >= 2:
        plot_compare(amp_exps, "val_macro_f1", str(p / "compare_amp.png"), "Chủ đề 6 — So sánh Mixed Precision: FP32 vs FP16")

    # 7. So sánh Khởi tạo tham số: base-s1 (He), init-xavier, init-normal, init-zeros
    init_keys = ["base-s1", "init-xavier", "init-normal", "init-zeros"]
    init_exps = [all_results[k] for k in init_keys if k in all_results]
    if len(init_exps) >= 2:
        plot_compare(init_exps, "val_loss", str(p / "compare_init.png"), "Chủ đề 7 — So sánh Khởi tạo tham số (Val Loss)")
