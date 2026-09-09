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

normalization = tf.keras.layers.Rescaling(1.0 / 255)

num_classes = len(train_dataset.class_names)

images, labels = next(iter(train_dataset))

plt.imshow(images[0].numpy().astype("uint8"))
plt.title(f"Правильна відповідь: {train_dataset.class_names[labels[0]]}")
plt.axis("off")
plt.show()

plt.imshow(images[1].numpy().astype("uint8"))
plt.title(f"Правильна відповідь: {train_dataset.class_names[labels[1]]}")
plt.axis("off")
plt.show()

model = tf.keras.Sequential([
    tf.keras.Input(shape=(150, 150, 3)),
    tf.keras.layers.Rescaling(1.0 / 255),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
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

plt.plot(history.history["accuracy"], label="training accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

image = images[0:1]
probabilities = model.predict(image, verbose=0)[0]
predicted_digit = int(np.argmax(probabilities))
predicted_plant = train_dataset.class_names[np.argmax(probabilities)]

plt.imshow(images[0].numpy().astype("uint8"))
plt.title(f"Прогноз: {predicted_plant}, правильна відповідь: {train_dataset.class_names[labels[0]]}")
plt.axis("off")
plt.show()

probabilities = model.predict(images, verbose=0)
predictions = np.argmax(probabilities, axis=1)
wrong_indices = np.where(predictions != labels.numpy())[0]


print(f"Кількість помилок: {len(wrong_indices)}")

for index in wrong_indices[:9]:
    plt.figure(figsize=(2, 2))
    plt.imshow(images[index].numpy().astype("uint8"))
    plt.title(
        f"Прогноз: {predictions[index]}, правильна: {images[index]}"
    )
    plt.axis("off")
    plt.show()