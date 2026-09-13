import tensorflow as tf

train_dataset = tf.keras.utils.image_dataset_from_directory( 
    "../../../../data/train",
    image_size=(150, 150),
    batch_size=32,
)


augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.9),
])

print(train_dataset.class_names)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(150, 150, 3)),
    tf.keras.layers.Rescaling(1.0 / 255),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(20, activation="softmax"),
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
