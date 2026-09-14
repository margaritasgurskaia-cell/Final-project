import numpy as np
import matplotlib.pyplot as plt

def plot_training_curves(history, save_path):
    epochs = range(1, len(history.history["loss"]) + 1)
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    
    plt.figure(figsize=(14, 5))
    
    # 1. Графік Loss (Функція втрат)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history["loss"], "b-o", label="Train Loss", markersize=4)
    plt.plot(epochs, history.history["val_loss"], "r-s", label="Validation Loss", markersize=4)
    plt.axvline(best_epoch, color="g", linestyle="--", label=f"Найкраща епоха ({best_epoch})")
    plt.title("Динаміка Loss (Похибка)", fontsize=13, fontweight="bold")
    plt.xlabel("Епоха")
    plt.ylabel("Loss")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    
    # 2. Графік Accuracy (Точність)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history["accuracy"], "b-o", label="Train Accuracy", markersize=4)
    plt.plot(epochs, history.history["val_accuracy"], "r-s", label="Validation Accuracy", markersize=4)
    plt.axvline(best_epoch, color="g", linestyle="--", label=f"Найкраща епоха ({best_epoch})")
    plt.title("Динаміка Accuracy (Точність)", fontsize=13, fontweight="bold")
    plt.xlabel("Епоха")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Графік успішно збережено в: {save_path}")


plot_training_curves(history, EXPERIMENT_DIR / "learning_curves.png")