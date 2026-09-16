import tensorflow as tf
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

DATA_DIR = Path(__file__).resolve().parents[2]/"data"
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 35
EXPERIMENT_DIR = Path("experiments/plant_evaluation_demo")
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = tf.keras.models.load_model(
    "../train-validation-test-kaggle/experiments/plant_evaluation_demo/best_plant_model.keras"
)

y_true = np.concatenate([
    labels.numpy()
    for images, labels in test_dataset
])

predictions = model.predict(test_dataset)

y_pred = np.argmax(predictions, axis=1)

print("Количество настоящих меток:", len(y_true))
print("Количество предсказаний:", len(y_pred))


cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=test_dataset.class_names,
    yticklabels=test_dataset.class_names
)

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix")

plt.tight_layout()
plt.savefig(EXPERIMENT_DIR / "confusion-matrix.png")
plt.show()