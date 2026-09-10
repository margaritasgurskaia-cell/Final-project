from pathlib import Path
import json
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D

train_dataset = tf.keras.utils.image_dataset_from_directory( 
    "../../data/train",
    image_size=(150, 150),
    batch_size=32,
)

validation_dataset=tf.keras.utils.image_dataset_from_directory(
    "../../data/validation",
    image_size=(150, 150),
    batch_size=32,
)

print(train_dataset.class_names)
print(validation_dataset.class_names)

class_count=len(validation_dataset.class_names)

model = Sequential([
    Input(shape=(150, 150, 1)),
    Conv2D(32, kernel_size=3, activation="relu"),
    MaxPooling2D(pool_size=2),
    Conv2D(64, kernel_size=3, activation="relu"),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(class_count, activation="softmax"),
])


model = tf.keras.Sequential([
    tf.keras.Input(shape=(150, 150, 3)),
    tf.keras.layers.Rescaling(1.0 / 255),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(class_count, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

experiment_dir = Path("experiments/00-baseline")
experiment_dir.mkdir(parents=True, exist_ok=True)

config = {
    "image_size": [150, 150],
    "batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 0.001,
    "augmentation": "none",
    "epochs": 50,
    "seed": 42,
}

(experiment_dir / "config.json").write_text(
    json.dumps(config, indent=2), encoding="utf-8"
)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=50,
)

model.save(experiment_dir / "model.keras")

metrics = {
    "best_validation_accuracy": max(history.history["val_accuracy"]),
    "best_validation_loss": min(history.history["val_loss"]),
}
(experiment_dir / "metrics.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)

history_frame = pd.DataFrame(history.history)
history_frame.to_csv(experiment_dir / "history.csv", index=False)

plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.legend()
plt.savefig(experiment_dir / "loss-curves.png")
plt.close()