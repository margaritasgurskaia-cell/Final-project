import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# ==========================================
# 1. КОНФІГУРАЦІЯ
# ==========================================
DATA_DIR = Path(__file__).resolve().parents[2]/"data"
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 35
EXPERIMENT_DIR = Path("experiments/plant_evaluation_demo")
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

print("Путь к данным:", DATA_DIR)
print("Папка существует:", DATA_DIR.exists())
print("Train существует:", (DATA_DIR / "train").exists())
# ==========================================
# 2. ЗАВАНТАЖЕННЯ ДАНИХ (3 НАБОРИ)
# ==========================================
# Якщо у вас одна папка з усіма фото, можна розділити її за допомогою
# tf.keras.utils.image_dataset_from_directory(..., validation_split=0.2, subset="...")
# Але найкраща практика — мати окремі папки train, validation, test:

print("Завантаження наборів даних...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "validation",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Знайдено класів: {num_classes} -> {class_names}")

# Зберігаємо класи у файл
with open(EXPERIMENT_DIR / "classes.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

# Оптимізація конвеєра даних у пам'яті
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 3. АРХІТЕКТУРА МОДЕЛІ
# ==========================================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(*IMAGE_SIZE, 3)),
    tf.keras.layers.Rescaling(1.0 / 255),  # нормалізація пікселів 0..1
    
    # Блок згорток 1
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    
    # Блок згорток 2
    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    
    # Блок згорток 3
    tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    
    # Класифікаційна голова з Dropout для захисту від перенавчання
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ==========================================
# 4. НАЛАШТУВАННЯ ЗБЕРЕЖЕННЯ (CALLBACKS)
# ==========================================
best_model_path = EXPERIMENT_DIR / "best_plant_model.keras"

callbacks = [
    # 1. Зберігаємо ЛИШЕ ту епоху, де валідаційний loss був мінімальним
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    ),
    # 2. Зупиняємося завчасно, якщо за 6 епох val_loss не покращився
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=6,
        restore_best_weights=True,
        verbose=1,
    ),
]

# ==========================================
# 5. ФАЗА 1 ТА 2: НАВЧАННЯ ТА ВАЛІДАЦІЯ
# ==========================================
print("\nПочаток навчання (Train + Validation)...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# Зберігаємо історію навчання в CSV
history_df = pd.DataFrame(history.history)
history_df.to_csv(EXPERIMENT_DIR / "history.csv", index=False)

# ==========================================
# 6. ФАЗА 3: ФІНАЛЬНИЙ ТЕСТ НАЙКРАЩОЇ МОДЕЛІ
# ==========================================
print("\nЗавантаження найкращої збереженої моделі для фінального тесту...")
best_model = tf.keras.models.load_model(best_model_path)

test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)
print(f"\n==========================================")
print(f"ФІНАЛЬНИЙ РЕЗУЛЬТАТ НА TEST DATASET:")
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"==========================================")

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