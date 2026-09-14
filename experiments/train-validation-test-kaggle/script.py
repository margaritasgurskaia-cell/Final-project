import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


DATA_DIR = Path("data")  
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 35
EXPERIMENT_DIR = Path("experiments/plant_evaluation_demo")
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)


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


with open(EXPERIMENT_DIR / "classes.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


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


print("\nЗавантаження найкращої збереженої моделі для фінального тесту...")
best_model = tf.keras.models.load_model(best_model_path)

test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)
print(f"\n==========================================")
print(f"ФІНАЛЬНИЙ РЕЗУЛЬТАТ НА TEST DATASET:")
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"==========================================")