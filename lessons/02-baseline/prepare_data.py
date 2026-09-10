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

print(train_dataset.class_names)

model = Sequential([
    Input(shape=(150, 150, 1)),
    Conv2D(32, kernel_size=3, activation="relu"),
    MaxPooling2D(pool_size=2),
    Conv2D(64, kernel_size=3, activation="relu"),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax"),
])

num_classes = len(train_dataset.class_names)

images, labels = next(iter(train_dataset))

print(labels)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(150, 150, 3)),
    tf.keras.layers.Rescaling(1.0 / 255),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train_dataset,
    epochs=50,
)

image = images[0:1]
probabilities = model.predict(image, verbose=0)[0]
predicted_digit = int(np.argmax(probabilities))
predicted_plant = train_dataset.class_names[np.argmax(probabilities)]

plt.imshow(images[0].numpy().astype("uint8"))
plt.title(f"Прогноз: {predicted_plant}, правильна відповідь: {train_dataset.class_names[labels[0]]}")
plt.axis("off")
plt.show()


