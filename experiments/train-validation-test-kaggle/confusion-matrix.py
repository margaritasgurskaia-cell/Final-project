import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Отримуємо справжні мітки та передбачення для тестового набору
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = best_model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

cm = confusion_matrix(y_true, y_pred)
print("\nДетальний звіт класифікації (Classification Report):")
print(classification_report(y_true, y_pred, target_names=class_names))

# Побудова матриці
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Матриця помилок на Test Dataset", fontsize=14, fontweight="bold")
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)

# Підпис кожного квадрата числами
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=11
        )

plt.ylabel("Справжній клас (Ground Truth)")
plt.xlabel("Передбачений клас (Predicted)")
plt.tight_layout()
plt.savefig(EXPERIMENT_DIR / "confusion_matrix.png", dpi=300)
plt.close()